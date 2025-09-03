#!/usr/bin/env python3


from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import train_fusion as TF
from hs_hackathon_drone_acoustics import CLASSES

def list_wavs(args) -> List[Path]:
    paths: List[Path] = []
    if args.wav:
        for w in args.wav:
            p = Path(w)
            if p.is_file() and p.suffix.lower() == ".wav":
                paths.append(p)
    if args.dir:
        d = Path(args.dir)
        for p in sorted(d.glob("*.wav")):
            paths.append(p)
    if not paths:
        raise FileNotFoundError("No .wav files found from --wav/--dir inputs.")
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="Path to fusion_model.pt")
    ap.add_argument("--wav", nargs="*", help="One or more WAV files")
    ap.add_argument("--dir", type=Path, help="Directory of WAV files")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--spec_sr", type=int, default=16000)
    ap.add_argument("--time_crop", type=float, default=3.0, help="Seconds; set to 0 to use full length")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_wavs(args)

    cfg = TF.SpecConfig(sr=args.spec_sr)
    time_crop = None if (args.time_crop is not None and args.time_crop <= 0) else args.time_crop
    ds = TF.FusionDataset(paths, [0]*len(paths), cfg=cfg, time_crop=time_crop)
    _, tab_probe, _ = ds[0]
    tab_dim = int(tab_probe.numel())

    model = TF.FusionModel(tab_dim=tab_dim, num_classes=len(CLASSES)).to(device)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    results = []
    with torch.inference_mode():
        for (mel, tab, _) in dl:
            mel, tab = mel.to(device), tab.to(device)
            logits = model(mel, tab)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            results.append(probs)
    probs_all = np.concatenate(results, axis=0)

    for i, p in enumerate(paths):
        pr = probs_all[i]
        topk = min(args.topk, len(CLASSES))
        idx = np.argsort(-pr)[:topk]
        tops = [(CLASSES[j], float(pr[j])) for j in idx]
        msg = " | ".join([f"{lab}: {prob:.3f}" for lab, prob in tops])
        print(f"{p.name}:  {msg}")

if __name__ == "__main__":
    main()
