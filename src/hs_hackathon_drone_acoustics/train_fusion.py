#!/usr/bin/env python3

"""
(example):
  python -m hs_hackathon_drone_acoustics.train_fusion ^
    --data_root "C:\\Users\\you\\drone_acoustics_train_val_data" ^
    --epochs 20 --batch_size 8 ^
    --lr 5e-4 --ft_lr_mult 0.1 ^
    --label_smoothing 0.0 ^
    --specaugment --sa_time_width 20 --sa_time_masks 1 --sa_freq_width 8 --sa_freq_masks 1 ^
    --sa_p 0.4 --ft_disable_sa ^
    --shift_prob 0.3 --shift_max_sec 0.2 ^
    --n_mels 128 --dropout 0.3 ^
    --tta_val 1 --tta_test 5 ^
    --out .\\fusion_seed0.pt --seed 0
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
import librosa

# ---- Project package ----
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
    sr: int = 16000
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    fmin: int = 20
    fmax: int | None = None

def logmel_dbfs(y: np.ndarray, cfg: SpecConfig) -> np.ndarray:
    """
    Log-mel in dBFS (ref=1.0) to preserve absolute level across files.
    Returns (n_mels, T) float32
    """
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop, window="hann")) ** 2
    M = librosa.feature.melspectrogram(
        S=S, sr=cfg.sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
    )
    M_db = librosa.power_to_db(M, ref=1.0)
    return M_db.astype(np.float32)

def mfcc_stats_from_logmel_db(M_db: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """
    MFCC (incl. c0) + deltas, then per-coeff stats (mean/std/p10/p90).
    Shape: 20*4*3 = 240
    """
    mfcc = librosa.feature.mfcc(S=M_db, n_mfcc=n_mfcc)
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
        rms_db.mean(), rms_db.std(),
        np.percentile(rms_db, 10), np.percentile(rms_db, 50), np.percentile(rms_db, 90),
        20 * np.log10(peak), crest, dyn_range,
        *agg(cent), *agg(bw), *agg(roll), *agg(flat), *agg(flux),
        librosa.feature.zero_crossing_rate(y).mean().astype(np.float32),
    ]
    return np.asarray(feats, dtype=np.float32)


# ----------------- SpecAugment (mel) -----------------

def specaugment(mel_db: np.ndarray,
                time_w: int = 30, time_m: int = 2,
                freq_w: int = 12, freq_m: int = 2) -> np.ndarray:
    """
    Simple SpecAugment on log-mel **in dBFS** (train only, CNN branch only).
    Masked regions filled with the 1st percentile dB value (gentle).
    """
    M = mel_db.copy()
    F, T = M.shape
    fill = float(np.percentile(M, 1))

    for _ in range(max(0, freq_m)):
        w = np.random.randint(0, min(freq_w, F) + 1)
        if w:
            f0 = np.random.randint(0, F - w + 1)
            M[f0:f0 + w, :] = fill

    for _ in range(max(0, time_m)):
        w = np.random.randint(0, min(time_w, T) + 1)
        if w:
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
                 sa_p: float = 1.0,
                 enable_shift: bool = False,
                 shift_prob: float = 0.5,
                 shift_max_sec: float = 0.25):
        self.filepaths = filepaths
        self.labels = labels
        self.cfg = cfg
        self.time_crop = time_crop

        self.use_sa = specaugment
        self.sa_time_w, self.sa_time_m = sa_time_w, sa_time_m
        self.sa_freq_w, self.sa_freq_m = sa_freq_w, sa_freq_m
        self.sa_p = float(sa_p)

        self.enable_shift = bool(enable_shift)
        self.shift_prob = float(shift_prob)
        self.shift_max_sec = float(shift_max_sec)

    def __len__(self) -> int:
        return len(self.filepaths)

    def _load_and_preprocess(self, path: Path):
        wf = AudioWaveform.load(path)
        y = wf.data.numpy().astype(np.float32)
        sr_in = int(wf.sample_rate)

        if y.ndim > 1:
            if y.shape[0] < y.shape[-1]:
                y = np.mean(y, axis=1).astype(np.float32)
            else:
                y = np.mean(y, axis=0).astype(np.float32)

        if sr_in != self.cfg.sr:
            y = librosa.resample(y, orig_sr=sr_in, target_sr=self.cfg.sr, res_type="kaiser_best")
            sr = self.cfg.sr
        else:
            sr = sr_in

        if self.enable_shift and (np.random.rand() < self.shift_prob):
            max_shift = int(self.shift_max_sec * sr)
            if max_shift > 0:
                k = np.random.randint(-max_shift, max_shift + 1)
                if k != 0:
                    y = np.roll(y, k)
                    if k > 0:
                        y[:k] = 0.0
                    else:
                        y[k:] = 0.0

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

        M_base = logmel_dbfs(y, self.cfg)

        M_cnn = M_base
        if self.use_sa and (np.random.rand() < self.sa_p):
            M_cnn = specaugment(M_base,
                                time_w=self.sa_time_w, time_m=self.sa_time_m,
                                freq_w=self.sa_freq_w, freq_m=self.sa_freq_m)

        mel_tensor = torch.from_numpy(M_cnn).unsqueeze(0)

        mfcc_vec = mfcc_stats_from_logmel_db(M_base, n_mfcc=20)
        eng_vec  = engineered_features(y, sr)                      # like 19
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
    def __init__(self, tab_dim: int, num_classes: int = 3,
                 img_dim: int = 256, tab_emb: int = 64, p: float = 0.2, pretrained: bool = True):
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


# ----------------- EMA wrapper -----------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd, esd = model.state_dict(), self.ema.state_dict()
        for k in esd.keys():
            m = msd[k]
            e = esd[k]
            if torch.is_floating_point(e):
                e.mul_(d).add_(m, alpha=1.0 - d)
            else:
                e.copy_(m)


# ----------------- Helpers -----------------

def _set_bn_eval(m):
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        m.eval()


# ----------------- Train / Eval -----------------

def train_one_epoch(model, loader, opt, device, label_smoothing=0.0,
                    ema: ModelEMA | None = None,
                    scaler: torch.amp.GradScaler | None = None,
                    grad_clip: float | None = 5.0,
                    freeze_backbone_bn: bool = False):
    model.train()
    if freeze_backbone_bn:
        model.img.backbone.apply(_set_bn_eval)

    total, correct, loss_sum = 0, 0, 0.0
    for mel, tab, y in loader:
        mel, tab, y = mel.to(device, non_blocking=True), tab.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits = model(mel, tab)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)

@torch.no_grad()
def evaluate(model, loader, device, tta: int = 0):
    """
    If tta > 0, apply time-roll TTA: [-2%, 0, +2%] for 3; add ±4% when >=5.
    """
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []

    def shifts_for_T(T: int, n: int):
        if n <= 1:
            return [0]
        fracs = [0.0, -0.02, 0.02] if n <= 3 else [0.0, -0.02, 0.02, -0.04, 0.04]
        return [int(f * T) for f in fracs]

    for mel, tab, y in loader:
        mel, tab, y = mel.to(device), tab.to(device), y.to(device)
        if tta <= 1:
            logits = model(mel, tab)
        else:
            T = mel.shape[-1]
            outs = []
            for s in shifts_for_T(T, tta):
                m = mel if s == 0 else torch.roll(mel, shifts=s, dims=-1)
                outs.append(model(m, tab))
            logits = torch.stack(outs, dim=0).mean(0)

        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])
    return loss_sum / max(1, total), correct / max(1, total), y_true, y_pred


# ----------------- Data discovery helpers -----------------

def discover_wavs(root: Path) -> Tuple[List[Path], List[int]]:
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
    parser.add_argument("--ft_lr_mult", type=float, default=0.1,
                        help="Fine-tune LR = lr * ft_lr_mult")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spec_sr", type=int, default=16000)
    parser.add_argument("--time_crop", type=float, default=5.0)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet pretrained weights for ResNet.")
    parser.add_argument("--n_mels", type=int, default=128,
                        help="Mel bins (e.g., 160).")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--out", type=Path, default=Path("fusion_model.pt"),
                        help="Where to save final best model (EMA state_dict).")

    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--specaugment", action="store_true",
                        help="Apply SpecAugment to mel spectrograms (TRAIN only, CNN branch).")
    parser.add_argument("--sa_time_masks", type=int, default=2)
    parser.add_argument("--sa_time_width", type=int, default=30)
    parser.add_argument("--sa_freq_masks", type=int, default=2)
    parser.add_argument("--sa_freq_width", type=int, default=12)
    parser.add_argument("--sa_p", type=float, default=0.7)
    parser.add_argument("--ft_disable_sa", action="store_true",
                        help="Turn off SpecAugment during fine-tuning.")
    parser.add_argument("--shift_prob", type=float, default=0.5,
                        help="Waveform time-shift prob (train datasets).")
    parser.add_argument("--shift_max_sec", type=float, default=0.25,
                        help="Max absolute shift seconds.")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--tta", type=int, default=None,
                        help="If set, overrides both --tta_val and --tta_test.")
    parser.add_argument("--tta_val", type=int, default=1,
                        help="TTA during validation.")
    parser.add_argument("--tta_test", type=int, default=3,
                        help="TTA during final test.")

    parser.add_argument("--balanced_sampler", action="store_true",
                        help="Use class-balanced WeightedRandomSampler for training loaders.")

    args = parser.parse_args()
    if args.tta is not None:
        args.tta_val = args.tta_test = args.tta

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
            sss_t = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            tr_keep_idx, test_idx = next(sss_t.split(X_tr, y_tr))
            X_test, y_test = X_tr[test_idx], y_tr[test_idx]
            X_tr,  y_tr  = X_tr[tr_keep_idx], y_tr[tr_keep_idx]
        preflight_print(root, "split", {"train": (list(X_tr), list(y_tr)),
                                        "val": (list(X_val), list(y_val)),
                                        "test": (list(X_test), list(y_test))})
    else:
        filepaths, labels = discover_wavs(root)
        if not filepaths:
            raise FileNotFoundError(f"No wav files found under {root}/<class>/*.wav")
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

    cfg = SpecConfig(sr=args.spec_sr, n_mels=args.n_mels)

    ds_probe = FusionDataset([X_tr[0]], [int(y_tr[0])], cfg=cfg, time_crop=args.time_crop)
    _, tab_probe, _ = ds_probe[0]
    tab_dim = int(tab_probe.numel())

    g = torch.Generator()
    g.manual_seed(args.seed)
    pin = torch.cuda.is_available()

    # ---- Datasets
    train_ds_warm = FusionDataset(list(X_tr), list(map(int, y_tr)), cfg=cfg, time_crop=args.time_crop,
                                  specaugment=args.specaugment,
                                  sa_time_w=args.sa_time_width, sa_time_m=args.sa_time_masks,
                                  sa_freq_w=args.sa_freq_width, sa_freq_m=args.sa_freq_masks,
                                  sa_p=args.sa_p,
                                  enable_shift=(args.shift_prob > 0.0), shift_prob=args.shift_prob, shift_max_sec=args.shift_max_sec)

    train_ds_ft = FusionDataset(list(X_tr), list(map(int, y_tr)), cfg=cfg, time_crop=args.time_crop,
                                specaugment=(args.specaugment and not args.ft_disable_sa),
                                sa_time_w=args.sa_time_width, sa_time_m=args.sa_time_masks,
                                sa_freq_w=args.sa_freq_width, sa_freq_m=args.sa_freq_masks,
                                sa_p=args.sa_p,
                                enable_shift=(args.shift_prob > 0.0), shift_prob=args.shift_prob, shift_max_sec=args.shift_max_sec)

    val_ds   = FusionDataset(list(X_val), list(map(int, y_val)), cfg=cfg, time_crop=args.time_crop,
                             specaugment=False, enable_shift=False)
    test_ds  = FusionDataset(list(X_test), list(map(int, y_test)), cfg=cfg, time_crop=args.time_crop,
                             specaugment=False, enable_shift=False)

    sampler = None
    if args.balanced_sampler:
        counts = np.bincount(y_tr, minlength=len(CLASSES)).astype(np.float32)
        class_w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1.0))
        sample_w = class_w[y_tr]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True
        )

    # ---- Loaders
    train_dl_warm = DataLoader(train_ds_warm, batch_size=args.batch_size,
                               shuffle=(sampler is None), sampler=sampler,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)
    train_dl_ft   = DataLoader(train_ds_ft,   batch_size=args.batch_size,
                               shuffle=(sampler is None), sampler=sampler,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)
    val_dl        = DataLoader(val_ds,        batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)
    test_dl       = DataLoader(test_ds,       batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=pin,
                               worker_init_fn=seed_worker, generator=g)

    # ---- Model
    model = FusionModel(tab_dim=tab_dim, num_classes=len(CLASSES),
                        pretrained=not args.no_pretrained, p=args.dropout).to(device)

    ema = ModelEMA(model, decay=args.ema_decay)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- Warmup (freeze CNN) + keep BN in eval to stop catastrophic thingy
    for p in model.img.backbone.parameters():
        p.requires_grad = False
    model.img.backbone.eval()

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_acc, best_state = -1.0, None
    patience, bad = 10, 0

    print(f"Classes: {CLASSES}")
    print(f"Train/Val/Test sizes: {len(train_ds_warm)} / {len(val_ds)} / {len(test_ds)}")

    # ---- Warm-up
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_dl_warm, opt, device,
            label_smoothing=args.label_smoothing,
            ema=ema, scaler=scaler, grad_clip=args.grad_clip,
            freeze_backbone_bn=True
        )

        val_loss, val_acc, _, _ = evaluate(model, val_dl, device, tta=args.tta_val)
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

    # ---- Fine-tune (unfreeze all) + re-enable BN updates
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        ema.ema.load_state_dict(best_state, strict=True)

    for p in model.parameters():
        p.requires_grad = True
    model.img.backbone.train()

    ft_lr = max(1e-6, args.lr * args.ft_lr_mult)
    opt = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val_acc, best_state, bad = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_dl_ft, opt, device,
            label_smoothing=args.label_smoothing,
            ema=ema, scaler=scaler, grad_clip=args.grad_clip,
            freeze_backbone_bn=False
        )
        val_loss, val_acc, _, _ = evaluate(ema.ema, val_dl, device, tta=args.tta_val)
        dt = time.time() - t0
        scheduler.step()
        spb = dt / max(1, len(train_dl_ft))
        print(f"[FT] Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | epoch_time {dt:.1f}s | sec/batch {spb:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in ema.ema.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("Early stopping (finetune).")
            break

    # ---- Evaluate on test using EMA + TTA
    if best_state is not None:
        ema.ema.load_state_dict(best_state, strict=True)

    test_loss, test_acc, y_true, y_pred = evaluate(ema.ema.to(device), test_dl, device, tta=args.tta_test)
    print(f"Test loss {test_loss:.4f} | Test acc {test_acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    torch.save(ema.ema.state_dict(), args.out)
    print(f"Saved EMA weights to {args.out.resolve()}")


if __name__ == "__main__":
    main()
