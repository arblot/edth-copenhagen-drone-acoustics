#!/usr/bin/env python3
"""
Train a fusion classifier for 3-class audio using:
- Spectrogram CNN branch (log-mel dBFS, absolute scale preserved)
- Tabular MLP branch (engineered features + MFCC stats)
- Split-aware data loading (pre-existing split or auto 70/15/15)
- (Optional) SpecAugment on mel (TRAIN ONLY, CNN branch only)
- Label smoothing, seeded loaders, and early stopping

Example:
  python -u -m hs_hackathon_drone_acoustics.train_fusion ^
    --data_root "C:\\Users\\strik\\Downloads\\drone_acoustics_train_val_data" ^
    --epochs 20 --batch_size 8 --specaugment ^
    --sa_time_width 20 --sa_time_masks 1 --sa_freq_width 8 --sa_freq_masks 1 ^
    --sa_p 0.7 --label_smoothing 0.05
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import librosa

# ---- Project package (import from your repo) ----
from hs_hackathon_drone_acoustics import CLASSES
from hs_hackathon_drone_acoustics.base import AudioWaveform


# ----------------- Utils -----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------- Feature Extraction -----------------

@dataclass
class SpecConfig:
    sr: int = 16000           # resample target for consistency
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    fmin: int = 20
    fmax: int | None = None   # None -> sr/2


def logmel_dbfs(y: np.ndarray, cfg: SpecConfig) -> np.ndarray:
    """
    Log-mel in dBFS (ref=1.0) to preserve absolute level across files.
    Returns (n_mels, T) float32
    """
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop, window="hann")) ** 2
    M = librosa.feature.melspectrogram(
        S=S, sr=cfg.sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
    )
    M_db = librosa.power_to_db(M, ref=1.0)  # absolute dBFS
    return M_db.astype(np.float32)


def mfcc_stats_from_logmel_db(M_db: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """
    MFCC (incl. c0) + deltas, then per-coeff stats (mean/std/p10/p90).
    Shape: 20*4*3 = 240
    """
    mfcc = librosa.feature.mfcc(S=M_db, n_mfcc=n_mfcc)            # (n_mfcc, T)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(X: np.ndarray) -> np.ndarray:
        return np.concatenate([
            X.mean(axis=1), X.std(axis=1),
            np.percentile(X, 10, axis=1), np.percentile(X, 90, axis=1)
        ])

    vec = np.concatenate([stats(mfcc), stats(d1), stats(d2)], axis=0)
    return vec.astype(np.float32)


def engineered_features(y: np.ndarray, sr: int, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    """
    Absolute-scale engineered features (no per-file normalization).
    Returns ~19 dims (RMS dBFS stats, peak/crest/dynamic range, spectral aggregates, ZCR).
    """
    eps = 1e-12
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0] + eps
    peak = np.max(np.abs(y)) + eps
    crest = 20 * np.log10(peak / (np.mean(rms) + eps))
    dyn_range = 20 * np.log10(np.percentile(rms, 95) / (np.percentile(rms, 5) + eps))

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) + eps
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bw   = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)

    def agg(v: np.ndarray) -> np.ndarray:
        return np.array([v.mean(), v.std()], dtype=np.float32)

    rms_db = 20 * np.log10(rms)
    feats = [
        # RMS stats (dBFS)
        rms_db.mean(), rms_db.std(),
        np.percentile(rms_db, 10), np.percentile(rms_db, 50), np.percentile(rms_db, 90),
        # Peaks & dynamics
        20 * np.log10(peak), crest, dyn_range,
        # Spectral aggregates
        *agg(cent), *agg(bw), *agg(roll), *agg(flat), *agg(flux),
        # Temporal
        librosa.feature.zero_crossing_rate(y).mean().astype(np.float32),
    ]
    return np.asarray(feats, dtype=np.float32)


# ----------------- SpecAugment -----------------

def specaugment(mel_db: np.ndarray,
                time_w: int = 30, time_m: int = 2,
                freq_w: int = 12, freq_m: int = 2) -> np.ndarray:
    """
    Simple SpecAugment on log-mel **in dBFS** (train only, CNN branch only).
    Masked regions filled with the 1st percentile dB value (gentle) to preserve absolute loudness semantics.
    """
    M = mel_db.copy()
    F, T = M.shape
    fill = float(np.percentile(M, 1))  # softer than absolute min

    # Frequency masks
    for _ in range(max(0, freq_m)):
        w = np.random.randint(0, min(freq_w, F) + 1)
        if w == 0: 
            continue
        f0 = np.random.randint(0, F - w + 1)
        M[f0:f0 + w, :] = fill

    # Time masks
    for _ in range(max(0, time_m)):
        w = np.random.randint(0, min(time_w, T) + 1)
        if w == 0:
            continue
        t0 = np.random.randint(0, T - w + 1)
        M[:, t0:t0 + w] = fill

    return M


# ----------------- Dataset -----------------

class FusionDataset(Dataset):
    def __init__(self, filepaths: List[Path], labels: List[int], cfg: SpecConfig,
                 time_crop: float | None = 5.0,
                 specaugment: bool = False,
                 sa_time_w: int = 30, sa_time_m: int = 2,
                 sa_freq_w: int = 12, sa_freq_m: int = 2,
                 sa_p: float = 1.0):
        self.filepaths = filepaths
        self.labels = labels
        self.cfg = cfg
        self.time_crop = time_crop  # seconds

        # SpecAugment controls
        self.use_sa = specaugment
        self.sa_time_w, self.sa_time_m = sa_time_w, sa_time_m
        self.sa_freq_w, self.sa_freq_m = sa_freq_w, sa_freq_m
        self.sa_p = float(sa_p)

    def __len__(self) -> int:
        return len(self.filepaths)

    def _load_and_preprocess(self, path: Path):
        wf = AudioWaveform.load(path)
        y = wf.data.numpy().astype(np.float32)
        sr_in = int(wf.sample_rate)

        # Mono
        if y.ndim > 1:
            # (T, C) or (C, T) -> average channels; be robust:
            if y.shape[0] < y.shape[-1]:
                y = np.mean(y, axis=1).astype(np.float32)
            else:
                y = np.mean(y, axis=0).astype(np.float32)

        # Resample (preserve scale)
        if sr_in != self.cfg.sr:
            y = librosa.resample(y, orig_sr=sr_in, target_sr=self.cfg.sr, res_type="kaiser_best")
            sr = self.cfg.sr
        else:
            sr = sr_in

        # Center crop/pad to fixed seconds
        if self.time_crop is not None and self.time_crop > 0:
            L = int(self.time_crop * sr)
            if y.shape[0] >= L:
                start = (y.shape[0] - L) // 2
                y = y[start:start+L]
            else:
                pad = L - y.shape[0]
                y = np.pad(y, (pad//2, pad - pad//2), mode="constant")

        return y, sr

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        label = self.labels[idx]
        y, sr = self._load_and_preprocess(path)

        # Base mel (clean, absolute dBFS)
        M_base = logmel_dbfs(y, SpecConfig(sr=self.cfg.sr, n_fft=self.cfg.n_fft, hop=self.cfg.hop,
                                           n_mels=self.cfg.n_mels, fmin=self.cfg.fmin, fmax=self.cfg.fmax))

        # CNN mel (maybe augmented)
        M_cnn = M_base
        if self.use_sa and (np.random.rand() < self.sa_p):
            M_cnn = specaugment(M_base,
                                time_w=self.sa_time_w, time_m=self.sa_time_m,
                                freq_w=self.sa_freq_w, freq_m=self.sa_freq_m)

        mel_tensor = torch.from_numpy(M_cnn).unsqueeze(0)  # (1, n_mels, T)

        # Tabular from clean mel + waveform
        mfcc_vec = mfcc_stats_from_logmel_db(M_base, n_mfcc=20)   # 240
        eng_vec  = engineered_features(y, sr)                      # ~19
        tab_vec  = np.concatenate([eng_vec, mfcc_vec], axis=0).astype(np.float32)
        tab_tensor = torch.from_numpy(tab_vec)

        return mel_tensor, tab_tensor, torch.tensor(label, dtype=torch.long)


# ----------------- Model -----------------

class SpecCNN(nn.Module):
    """
    ResNet-18 backbone adapted to 1-channel input; produces a 256-D embedding.
    """
    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            backbone = models.resnet18(weights=None)
        w = backbone.conv1.weight.data
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_feats, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.proj(z)
        return z


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, tab_dim: int, num_classes: int = 3, img_dim: int = 256, tab_emb: int = 64, p: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.img = SpecCNN(out_dim=img_dim, pretrained=pretrained)
        self.tab = TabularMLP(in_dim=tab_dim, out_dim=tab_emb, hidden=256, p=p)
        self.head = nn.Sequential(
            nn.Linear(img_dim + tab_emb, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, num_classes),
        )

    def forward(self, mel, tab):
        zi = self.img(mel)
        zt = self.tab(tab)
        z = torch.cat([zi, zt], dim=1)
        return self.head(z)


# ----------------- Train / Eval Loops -----------------

def train_one_epoch(model, loader, opt, device, label_smoothing=0.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for mel, tab, y in loader:
        mel, tab, y = mel.to(device), tab.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(mel, tab)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    for mel, tab, y in loader:
        mel, tab, y = mel.to(device), tab.to(device), y.to(device)
        logits = model(mel, tab)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])
    return loss_sum / max(1, total), correct / max(1, total), y_true, y_pred


# ----------------- Data discovery helpers -----------------

def discover_wavs(root: Path) -> Tuple[List[Path], List[int]]:
    """FLAT layout: root/<class>/*.wav"""
    filepaths, labels = [], []
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    for c in CLASSES:
        for p in sorted((root / c).glob("*.wav")):
            filepaths.append(p)
            labels.append(class_to_idx[c])
    return filepaths, labels


def has_split_dirs(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "val").is_dir()


def discover_split(root: Path) -> Dict[str, Tuple[List[Path], List[int]]]:
    """SPLIT layout under root/train, root/val, (optional) root/test"""
    out: Dict[str, Tuple[List[Path], List[int]]] = {}
    for subset in ["train", "val", "test"]:
        subdir = root / subset
        if subdir.is_dir():
            paths, labels = discover_wavs(subdir)
            out[subset] = (paths, labels)
    return out


def preflight_print(root: Path, layout_note: str, sets: Dict[str, Tuple[List[Path], List[int]]]) -> None:
    print(f"\n[Preflight] data_root = {root.resolve()}")
    print(f"[Preflight] layout    = {layout_note}")
    for subset in ["train", "val", "test"]:
        if subset in sets:
            paths, labels = sets[subset]
            counts = {c: 0 for c in CLASSES}
            for p, y in zip(paths, labels):
                counts[CLASSES[y]] += 1
            counts_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
            print(f"[Preflight] {subset:>5}: total={len(paths)} | {counts_str}")
    print("")


# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Either FLAT layout or folder with train/val[/test].")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spec_sr", type=int, default=16000)
    parser.add_argument("--time_crop", type=float, default=5.0)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet pretrained weights for ResNet (faster start, slightly worse).")

    # Output path
    parser.add_argument("--out", type=Path, default=Path("fusion_model.pt"),
                        help="Where to save final best model.")

    # Regularization / tricks
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Cross-entropy label smoothing (train only). Use 0.0 to disable.")
    parser.add_argument("--specaugment", action="store_true",
                        help="Apply SpecAugment to mel spectrograms (TRAIN only, CNN branch only).")
    parser.add_argument("--sa_time_masks", type=int, default=2)
    parser.add_argument("--sa_time_width", type=int, default=30,
                        help="Max time-mask width in frames (e.g., 20–40).")
    parser.add_argument("--sa_freq_masks", type=int, default=2)
    parser.add_argument("--sa_freq_width", type=int, default=12,
                        help="Max freq-mask width in mel bins (e.g., 8–16).")
    parser.add_argument("--sa_p", type=float, default=0.7,
                        help="Probability of applying SpecAugment to a training sample.")
    parser.add_argument("--ft_disable_sa", action="store_true",
                        help="Turn off SpecAugment during fine-tuning.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.data_root
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    # ----- Figure out layout -----
    if has_split_dirs(root):
        splits = discover_split(root)
        if "train" not in splits or "val" not in splits:
            raise FileNotFoundError("SPLIT layout requires train/ and val/ under data_root.")
        X_tr, y_tr = np.array(splits["train"][0]), np.array(splits["train"][1])
        X_val, y_val = np.array(splits["val"][0]),  np.array(splits["val"][1])
        if "test" in splits:
            X_test, y_test = np.array(splits["test"][0]), np.array(splits["test"][1])
        else:
            # Create a test split from TRAIN only, keep provided VAL intact
            sss_t = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            tr_keep_idx, test_idx = next(sss_t.split(X_tr, y_tr))
            X_test, y_test = X_tr[test_idx], y_tr[test_idx]
            X_tr,  y_tr  = X_tr[tr_keep_idx], y_tr[tr_keep_idx]
        preflight_print(root, "split", {"train": (list(X_tr), list(y_tr)),
                                        "val": (list(X_val), list(y_val)),
                                        "test": (list(X_test), list(y_test))})
    else:
        # FLAT layout: auto split 70/15/15
        filepaths, labels = discover_wavs(root)
        if not filepaths:
            raise FileNotFoundError(f"No wav files found under {root}/<class>/*.wav "
                                    f"(or provide split layout at {root}/train & {root}/val)")
        filepaths = np.array(filepaths)
        labels = np.array(labels)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(sss1.split(filepaths, labels))
        X_train, X_test, y_train, y_test = filepaths[train_idx], filepaths[test_idx], labels[train_idx], labels[test_idx]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size/(1-args.test_size), random_state=args.seed)
        tr_idx, val_idx = next(sss2.split(X_train, y_train))
        X_tr, X_val, y_tr, y_val = X_train[tr_idx], X_train[val_idx], y_train[tr_idx], y_train[val_idx]

        preflight_print(root, "flat", {"train": (list(X_tr), list(y_tr)),
                                       "val": (list(X_val), list(y_val)),
                                       "test": (list(X_test), list(y_test))})

    cfg = SpecConfig(sr=args.spec_sr)

    # Probe tab dim
    ds_probe = FusionDataset([X_tr[0]], [int(y_tr[0])], cfg=cfg, time_crop=args.time_crop)
    _, tab_probe, _ = ds_probe[0]
    tab_dim = int(tab_probe.numel())

    # Seeded generator for workers
    g = torch.Generator()
    g.manual_seed(args.seed)
    pin = torch.cuda.is_available()

    # ---- Datasets / Loaders
    # Warm-up (SA according to --specaugment)
    train_ds_warm = FusionDataset(list(X_tr), list(map(int, y_tr)), cfg=cfg, time_crop=args.time_crop,
                                  specaugment=args.specaugment,
                                  sa_time_w=args.sa_time_width, sa_time_m=args.sa_time_masks,
                                  sa_freq_w=args.sa_freq_width, sa_freq_m=args.sa_freq_masks,
                                  sa_p=args.sa_p)

    # Fine-tune (optionally disable SA)
    train_ds_ft = FusionDataset(list(X_tr), list(map(int, y_tr)), cfg=cfg, time_crop=args.time_crop,
                                specaugment=(args.specaugment and not args.ft_disable_sa),
                                sa_time_w=args.sa_time_width, sa_time_m=args.sa_time_masks,
                                sa_freq_w=args.sa_freq_width, sa_freq_m=args.sa_freq_masks,
                                sa_p=args.sa_p)

    val_ds   = FusionDataset(list(X_val), list(map(int, y_val)), cfg=cfg, time_crop=args.time_crop, specaugment=False)
    test_ds  = FusionDataset(list(X_test), list(map(int, y_test)), cfg=cfg, time_crop=args.time_crop, specaugment=False)

    train_dl_warm = DataLoader(train_ds_warm, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)
    train_dl_ft   = DataLoader(train_ds_ft,   batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)
    val_dl        = DataLoader(val_ds,        batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)
    test_dl       = DataLoader(test_ds,       batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)

    model = FusionModel(tab_dim=tab_dim, num_classes=len(CLASSES),
                        pretrained=not args.no_pretrained).to(device)

    # Warmup (freeze CNN)
    for p in model.img.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_acc, best_state = -1.0, None
    patience, bad = 7, 0

    print(f"Classes: {CLASSES}")
    print(f"Train/Val/Test sizes: {len(train_ds_warm)} / {len(val_ds)} / {len(test_ds)}")

    # ---- Warm-up
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl_warm, opt, device,
                                          label_smoothing=args.label_smoothing)
        val_loss, val_acc, _, _ = evaluate(model, val_dl, device)
        dt = time.time() - t0
        scheduler.step()
        spb = dt / max(1, len(train_dl_warm))
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | epoch_time {dt:.1f}s | sec/batch {spb:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("Early stopping (warmup).")
            break

    # ---- Fine-tune (unfreeze all)
    if best_state is not None:
        model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val_acc, best_state, bad = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl_ft, opt, device,
                                          label_smoothing=args.label_smoothing)
        val_loss, val_acc, _, _ = evaluate(model, val_dl, device)
        dt = time.time() - t0
        scheduler.step()
        spb = dt / max(1, len(train_dl_ft))
        print(f"[FT] Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | epoch_time {dt:.1f}s | sec/batch {spb:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("Early stopping (finetune).")
            break

    # ---- Evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model.to(device), test_dl, device)
    print(f"Test loss {test_loss:.4f} | Test acc {test_acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    torch.save(model.state_dict(), args.out)
    print(f"Saved weights to {args.out.resolve()}")


if __name__ == "__main__":
    main()
