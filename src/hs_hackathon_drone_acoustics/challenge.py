#!/usr/bin/env python3
# Set CPU threading env BEFORE importing numpy/torch for lower latency on small workloads
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import io
import time
import argparse
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa  # still used for MFCC stats (matches training exactly)
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # if compile backend unsupported, silently fall back to eager

# Optional fast front-end (CPU or CUDA)
try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except Exception:
    HAS_TORCHAUDIO = False

# Import your model + feature functions (must match training)
from hs_hackathon_drone_acoustics.train_fusion import (
    FusionModel, SpecConfig, engineered_features, mfcc_stats_from_logmel_db
)
from hs_hackathon_drone_acoustics import CLASSES

# ---------------- Defaults (override via CLI) ----------------
BASE_URL_DEFAULT = "http://172.104.137.51:8080"
TOKEN_DEFAULT = "5d1c9048-9349-47cf-8783-ea19054657bb"
TAB_DIM = 259  # ~19 engineered + 240 MFCC stats

# Match your training config
cfg = SpecConfig(sr=16000, n_fft=1024, hop=256, n_mels=128, fmin=20, fmax=None)

# ---------------- AMP (new API) ----------------
def amp_ctx(device: torch.device):
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16) \
           if device.type == "cuda" else nullcontext()

# ---------------- Cached mel basis (librosa, CPU path) ----------------
_MEL_BASIS = None
def mel_basis(sr, n_fft, n_mels, fmin, fmax):
    global _MEL_BASIS
    if _MEL_BASIS is None:
        _MEL_BASIS = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=(sr//2 if fmax is None else fmax),
            htk=False, norm="slaney"  # matches librosa default used in training
        ).astype(np.float32)
    return _MEL_BASIS

def fast_logmel_dbfs(y: np.ndarray, cfg: SpecConfig) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop, window="hann")) ** 2
    M = mel_basis(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax) @ S
    M_db = librosa.power_to_db(M, ref=1.0, top_db=80.0)  # absolute dBFS, clamped
    return M_db.astype(np.float32)  # (n_mels, T)

# ---------------- Light TTA utilities (optional) ----------------
def specaugment_mild(mel_db: np.ndarray, time_w: int = 12, freq_w: int = 6) -> np.ndarray:
    M = mel_db.copy()
    Fm, Tm = M.shape
    fill = float(np.percentile(M, 1))
    w = np.random.randint(0, min(freq_w, Fm) + 1)
    if w > 0:
        f0 = np.random.randint(0, Fm - w + 1)
        M[f0:f0+w, :] = fill
    w = np.random.randint(0, min(time_w, Tm) + 1)
    if w > 0:
        t0 = np.random.randint(0, Tm - w + 1)
        M[:, t0:t0+w] = fill
    return M

def mel_time_shift(mel_db: np.ndarray, frac: float) -> np.ndarray:
    if frac == 0.0:
        return mel_db
    Tm = mel_db.shape[1]
    shift = int(np.sign(frac) * max(1, int(abs(frac) * Tm)))
    return np.roll(mel_db, shift=shift, axis=1)

# ---------------- Model loading ----------------
def maybe_channels_last(model: nn.Module, use: bool) -> nn.Module:
    if use:
        model = model.to(memory_format=torch.channels_last)
    return model

def maybe_compile(model: nn.Module, use: bool) -> nn.Module:
    if use and hasattr(torch, "compile"):
        backend = "inductor"
        try:
            import triton  # noqa: F401
        except Exception:
            backend = "aot_eager"  # Windows-friendly fallback
        try:
            model = torch.compile(model, backend=backend, mode="reduce-overhead")
            print(f"[compile] backend={backend}")
        except Exception as e:
            print(f"[compile] disabled (fallback to eager): {e}")
    return model

def load_model(weights_path: str, device: torch.device,
               channels_last: bool = False,
               quantize_dynamic: bool = False,
               compile_model: bool = False) -> FusionModel:
    model = FusionModel(tab_dim=TAB_DIM, num_classes=len(CLASSES), pretrained=False).to(device)
    # Future-safe torch.load
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if device.type == "cpu" and quantize_dynamic:
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = maybe_channels_last(model, channels_last)

    model = maybe_compile(model, compile_model)
    return model

# ---------------- Energy crop (torch) ----------------
def energy_crop_5s_torch(y_t: torch.Tensor, sr: int) -> torch.Tensor:
    L = int(5.0 * sr)
    if y_t.numel() <= L:
        pad = L - y_t.numel()
        return F.pad(y_t, (pad // 2, pad - pad // 2))
    p = y_t * y_t
    cs = torch.cat([torch.zeros(1, device=y_t.device, dtype=torch.float64),
                    p.to(torch.float64).cumsum(0)])
    start = torch.argmax(cs[L:] - cs[:-L]).item()
    return y_t[start:start + L]

# ---------------- Torch front-end (CPU or CUDA) ----------------
_TA = {"key": None, "mel": None, "to_db": None}
_RESAMPLERS = {}

def init_torch_frontend(cfg, device):
    if not HAS_TORCHAUDIO:
        return None
    key = (cfg.sr, cfg.n_fft, cfg.hop, cfg.n_mels, cfg.fmin, cfg.fmax, device.type)
    if _TA["key"] != key:
        mel = T.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=(cfg.sr // 2 if cfg.fmax is None else cfg.fmax),
            power=2.0,
            center=True,
            window_fn=torch.hann_window,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        to_db = T.AmplitudeToDB(stype="power", top_db=80.0).to(device)
        _TA.update({"key": key, "mel": mel, "to_db": to_db})
    return _TA

def get_resampler(orig_sr: int, new_sr: int, device: torch.device):
    if not HAS_TORCHAUDIO or orig_sr == new_sr:
        return None
    rk = (orig_sr, new_sr, device.type)
    r = _RESAMPLERS.get(rk)
    if r is None:
        r = T.Resample(orig_freq=orig_sr, new_freq=new_sr).to(device)
        _RESAMPLERS[rk] = r
    return r

def decode_and_preprocess_torch(wav_bytes: bytes, device: torch.device, cpu_fast: bool = False):
    y_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if y_np.ndim > 1:
        y_np = y_np.mean(axis=1).astype(np.float32)

    if HAS_TORCHAUDIO and (device.type == "cuda" or cpu_fast):
        y_t = torch.from_numpy(y_np).to(device, non_blocking=True)
        if sr != cfg.sr:
            resamp = get_resampler(sr, cfg.sr, device)
            y_t = resamp(y_t.unsqueeze(0)).squeeze(0)
            sr = cfg.sr
        y_t = energy_crop_5s_torch(y_t, sr)
        ta = init_torch_frontend(cfg, device)
        M_db_t = ta["to_db"](ta["mel"](y_t.unsqueeze(0)))  # (1, n_mels, T)
        mel_tensor = M_db_t.unsqueeze(0).detach().cpu()    # (1,1,n_mels,T) to CPU for model
        M_db = M_db_t.squeeze(0).detach().cpu().numpy()
        y_cpu = y_t.detach().cpu().numpy()
    else:
        y = y_np
        if sr != cfg.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr, res_type="kaiser_fast")
            sr = cfg.sr
        L = int(5.0 * sr)
        if y.shape[0] > L:
            p = y * y
            cs = np.concatenate(([0.0], np.cumsum(p, dtype=np.float64)))
            start = int(np.argmax(cs[L:] - cs[:-L]))
            y = y[start:start + L]
        else:
            pad = L - y.shape[0]
            y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
        M_db = fast_logmel_dbfs(y, cfg)
        mel_tensor = torch.from_numpy(M_db).unsqueeze(0).unsqueeze(0)
        y_cpu = y

    mfcc_vec = mfcc_stats_from_logmel_db(M_db, n_mfcc=20)
    eng_vec  = engineered_features(y_cpu, sr)
    tab_vec  = np.concatenate([eng_vec, mfcc_vec], axis=0).astype(np.float32)
    tab_tensor = torch.from_numpy(tab_vec).unsqueeze(0)
    return mel_tensor, tab_tensor

# Plain CPU decode path (librosa)
def decode_and_preprocess_librosa(wav_bytes: bytes, resample_fast: bool = True):
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    if sr != cfg.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr,
                             res_type=("kaiser_fast" if resample_fast else "kaiser_best"))
        sr = cfg.sr
    L = int(5.0 * sr)
    if y.shape[0] > L:
        p = y * y
        cs = np.concatenate(([0.0], np.cumsum(p, dtype=np.float64)))
        start = int(np.argmax(cs[L:] - cs[:-L]))
        y = y[start:start + L]
    else:
        pad = L - y.shape[0]
        y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
    M = fast_logmel_dbfs(y, cfg)
    mel_tensor = torch.from_numpy(M).unsqueeze(0).unsqueeze(0)
    mfcc_vec = mfcc_stats_from_logmel_db(M, n_mfcc=20)
    eng_vec  = engineered_features(y, sr)
    tab_vec  = np.concatenate([eng_vec, mfcc_vec], axis=0).astype(np.float32)
    tab_tensor = torch.from_numpy(tab_vec).unsqueeze(0)
    return mel_tensor, tab_tensor

# ---------------- Prediction ----------------
@torch.no_grad()
def predict(model: FusionModel, mel: torch.Tensor, tab: torch.Tensor, device: torch.device,
            tta: int = 0, adaptive_tta: float = 0.0) -> str:
    mel = mel.to(device, non_blocking=True)
    tab = tab.to(device, non_blocking=True)

    with amp_ctx(device):
        logits = model(mel, tab)
    if tta <= 0 or adaptive_tta > 0.0:
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
        if tta <= 0 or conf.item() >= adaptive_tta:
            return CLASSES[pred_idx.item()]

    mel_np = mel.squeeze(0).squeeze(0).detach().cpu().numpy()
    views = [mel_np, mel_time_shift(mel_np, +0.02), mel_time_shift(mel_np, -0.02)]
    for _ in range(max(0, tta - 1)):
        views.append(specaugment_mild(mel_np, time_w=12, freq_w=6))

    logits_list = []
    for v in views:
        v_t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
        with amp_ctx(device):
            logits_list.append(model(v_t, tab))
    logits = torch.stack(logits_list, dim=0).mean(0)
    pred_idx = logits.argmax(1).item()
    return CLASSES[pred_idx]

# ---------------- Save fetched WAVs ----------------
def save_wav_to_dataset(audio_bytes: bytes, dataset_root: str, subset: str, label: str, challenge_id: str | None):
    root = Path(dataset_root)
    target_dir = root / subset / label
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    base = f"{ts}_{challenge_id or 'noid'}.wav"
    out_path = target_dir / base
    out_path.write_bytes(audio_bytes)
    print(f"[save] wrote {out_path}")
    return out_path

# ---------------- Challenge loop ----------------
def run_loop(weights="fusion_model.pt", base_url=BASE_URL_DEFAULT, token=TOKEN_DEFAULT,
             tta: int = 0, adaptive_tta: float = 0.0, timeout: float = 5.0, threads: int = 1,
             torch_features: bool = False,  # use torchaudio front-end on CPU or CUDA
             resample_fast: bool = True,    # fallback librosa path
             sleep_jitter: float = 0.05, profile: bool = False,
             work_budget_ms: float = 600.0, ema_alpha: float = 0.3,
             deadline_guard_ms: float = 120.0,
             channels_last: bool = False, quantize_dynamic: bool = False,
             compile_model: bool = False,
             # NEW saving controls
             save_dataset_root: str | None = None,
             save_subdir: str = "train",
             save_all: bool = False):
    """
    work_budget_ms: minimum ms required before starting work (guard against near-rotation).
    deadline_guard_ms: skip submit if we're within this margin of the deadline (pre-submit guard).
    save_dataset_root: if set, save WAVs under <root>/<save_subdir>/<predicted-class>/.
                       By default only saves when server says 'Correct!'. Use --save_all to save every clip.
    """
    try:
        torch.set_num_threads(max(1, int(threads)))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, int(threads)))
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_torch_feats = bool(torch_features and HAS_TORCHAUDIO)

    model = load_model(weights, device,
                       channels_last=channels_last,
                       quantize_dynamic=quantize_dynamic,
                       compile_model=compile_model)

    print(f"[device] using: {device.type} | torch_features={int(use_torch_feats)} | channels_last={channels_last} | compile={compile_model}")
    if device.type == "cuda":
        try:
            print(f"[device] cuda name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Connection": "keep-alive"})

    ema_proc_ms = work_budget_ms

    while True:
        loop_start = time.perf_counter()

        # 1) Get challenge
        try:
            t0 = time.perf_counter()
            r = session.get(f"{base_url}/api/challenge", timeout=timeout)
            r.raise_for_status()
            challenge = r.json()
        except requests.RequestException as e:
            print(f"[net] challenge fetch error: {e}; backoff 0.25s")
            time.sleep(0.25)
            continue

        t_after_challenge = time.perf_counter()
        wav_url = challenge["wav_url"]
        challenge_id = challenge["challenge_id"]
        time_left_ms = float(challenge.get("time_until_next_rotation_ms", 0))
        deadline = t_after_challenge + (time_left_ms / 1000.0)

        # Freshness guard
        remaining_ms_now = (deadline - time.perf_counter()) * 1000.0
        budget_ms = max(ema_proc_ms, work_budget_ms)
        if remaining_ms_now < budget_ms:
            sleep_time = max(0.0, deadline - time.perf_counter()) + float(sleep_jitter)
            if profile:
                print(f"[guard] remaining {remaining_ms_now:.1f} ms < budget {budget_ms:.1f} ms → sleep {sleep_time:.3f}s and refetch")
            time.sleep(sleep_time)
            continue

        # 2) Download audio
        try:
            t1 = time.perf_counter()
            audio = session.get(f"{base_url}{wav_url}", timeout=timeout)
            audio.raise_for_status()
            audio_bytes = audio.content
        except requests.RequestException as e:
            print(f"[net] audio fetch error: {e}; backoff 0.25s")
            time.sleep(0.25)
            continue

        # 3) Features
        t2 = time.perf_counter()
        if use_torch_feats:
            mel, tab = decode_and_preprocess_torch(audio_bytes, device=device, cpu_fast=(device.type == "cpu"))
        else:
            mel, tab = decode_and_preprocess_librosa(audio_bytes, resample_fast=resample_fast)

        # 4) Predict
        t3 = time.perf_counter()
        label = predict(model, mel, tab, device=device, tta=tta, adaptive_tta=adaptive_tta)

        # Pre-submit guard
        if (deadline - time.perf_counter()) * 1000.0 < deadline_guard_ms:
            if profile:
                print(f"[pre-submit guard] <{deadline_guard_ms:.0f} ms left; skip submit & refetch next")
            time.sleep(max(0.0, (deadline - time.perf_counter()) + float(sleep_jitter)))
            continue

        # 5) Submit
        t4 = time.perf_counter()
        payload = {"challenge_id": challenge_id, "classification": label}
        try:
            resp = session.post(f"{base_url}/api/challenge", json=payload, timeout=timeout)
            try:
                resp_json = resp.json()
            except Exception:
                resp_json = {"raw": resp.text}
        except requests.RequestException as e:
            resp_json = {"error": str(e)}
        print({"prediction": label, "server": resp_json})
        t5 = time.perf_counter()

        # Save WAV (only 'Correct!' by default, or all if --save_all)
        if save_dataset_root:
            msg = str(resp_json.get("message", "")).lower()
            ok = bool(resp_json.get("success")) and ("correct" in msg)
            if save_all or ok:
                try:
                    save_wav_to_dataset(audio_bytes, save_dataset_root, save_subdir, label, challenge_id)
                except Exception as e:
                    print(f"[save] failed: {e}")

        # Update EMA of processing time fetch→submit
        proc_ms = (t5 - t_after_challenge) * 1000.0
        ema_proc_ms = ema_alpha * proc_ms + (1 - ema_alpha) * ema_proc_ms

        # 6) Sleep until next rotation (minus jitter)
        sleep_time = max(0.0, (deadline - time.perf_counter()) - float(sleep_jitter))

        # Logging
        e2e_ms = (time.perf_counter() - loop_start) * 1000.0
        print(f"⏱ end-to-end latency: {e2e_ms:.1f} ms | device: {device.type} | "
              f"tta={tta} | adaptive_tta={adaptive_tta} | ema_proc≈{ema_proc_ms:.1f} ms | "
              f"torch_feats={int(use_torch_feats)}")
        if profile:
            print("timings ms | "
                  f"challenge {1000*(t_after_challenge - t0):.1f} | "
                  f"download {1000*(t2 - t1):.1f} | "
                  f"features {1000*(t3 - t2):.1f} | "
                  f"model {1000*(t4 - t3):.1f} | "
                  f"submit {1000*(t5 - t4):.1f}")
        print(f"⏳ Sleeping {sleep_time:.3f}s\n")
        time.sleep(sleep_time)

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="fusion_model.pt")
    ap.add_argument("--base_url", type=str, default=BASE_URL_DEFAULT)
    ap.add_argument("--token", type=str, default=TOKEN_DEFAULT)
    ap.add_argument("--tta", type=int, default=0, help="0=fastest; 1-3 adds small accuracy bump")
    ap.add_argument("--adaptive_tta", type=float, default=0.0,
                    help="Confidence threshold (0-1). If >0, run TTA only when initial confidence < threshold.")
    ap.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout seconds")
    ap.add_argument("--threads", type=int, default=1, help="torch.set_num_threads for CPU ops")

    # Front-end toggles
    ap.add_argument("--torch_features", action="store_true",
                    help="Use torchaudio front-end (resample+mel+dB) on CPU or CUDA.")
    ap.add_argument("--resample_fast", action="store_true",
                    help="Use 'kaiser_fast' resample in librosa fallback path (CPU).")

    # Rotation guards / profiling
    ap.add_argument("--sleep_jitter", type=float, default=0.05,
                    help="Extra safety margin around rotations (seconds).")
    ap.add_argument("--profile", action="store_true", help="Print stage timings")
    ap.add_argument("--work_budget_ms", type=float, default=600.0,
                    help="Required time remaining before starting a challenge (ms).")
    ap.add_argument("--ema_alpha", type=float, default=0.3,
                    help="EMA smoothing for processing time estimate.")
    ap.add_argument("--deadline_guard_ms", type=float, default=120.0,
                    help="If less than this remains just before submit, skip submit and wait.")

    # Model-speed toggles
    ap.add_argument("--channels_last", action="store_true",
                    help="Use channels-last memory format for the model (CUDA).")
    ap.add_argument("--quantize_dynamic", action="store_true",
                    help="Dynamic quantization for Linear layers (CPU only).")
    ap.add_argument("--compile_model", action="store_true",
                    help="Try torch.compile; auto-fallback on Windows.")

    # NEW: saving toggles
    ap.add_argument("--save_dataset_root", type=str, default=None,
                    help=r'Root of your dataset (expects "<root>/<subset>/<class>/"). '
                         r'Example: "C:\Users\strik\Downloads\drone_acoustics_train_val_data"')
    ap.add_argument("--save_subdir", type=str, default="train",
                    help='Subset dir to place saved files (default: "train")')
    ap.add_argument("--save_all", action="store_true",
                    help="Save every fetched clip under predicted label (not just correct ones).")

    args = ap.parse_args()

    run_loop(
        weights=args.weights,
        base_url=args.base_url,
        token=args.token,
        tta=args.tta,
        adaptive_tta=args.adaptive_tta,
        timeout=args.timeout,
        threads=args.threads,
        torch_features=args.torch_features,
        resample_fast=args.resample_fast,
        sleep_jitter=args.sleep_jitter,
        profile=args.profile,
        work_budget_ms=args.work_budget_ms,
        ema_alpha=args.ema_alpha,
        deadline_guard_ms=args.deadline_guard_ms,
        channels_last=args.channels_last,
        quantize_dynamic=args.quantize_dynamic,
        compile_model=args.compile_model,
        save_dataset_root=args.save_dataset_root,
        save_subdir=args.save_subdir,
        save_all=args.save_all,
    )
