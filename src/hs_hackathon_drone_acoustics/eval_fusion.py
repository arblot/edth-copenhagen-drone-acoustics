#!/usr/bin/env python3

from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from . import train_fusion as TF  # re-use dataset & model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spec_sr", type=int, default=16000)
    ap.add_argument("--time_crop", type=float, default=3.0)
    ap.add_argument("--test_size", type=float, default=0.15, help="Only for FLAT layout or SPLIT-without-test")
    args = ap.parse_args()

    TF.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect layout
    if TF.has_split_dirs(args.data_root):
        splits = TF.discover_split(args.data_root)
        if "test" in splits:
            X_test, y_test = np.array(splits["test"][0]), np.array(splits["test"][1])
        else:
            # carve test from train
            X_tr, y_tr = np.array(splits["train"][0]), np.array(splits["train"][1])
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            _, test_idx = next(sss.split(X_tr, y_tr))
            X_test, y_test = X_tr[test_idx], y_tr[test_idx]
    else:
        # FLAT layout: need to split to get a test set
        filepaths, labels = TF.discover_wavs(args.data_root)
        filepaths = np.array(filepaths); labels = np.array(labels)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        _, test_idx = next(sss1.split(filepaths, labels))
        X_test, y_test = filepaths[test_idx], labels[test_idx]

    cfg = TF.SpecConfig(sr=args.spec_sr)
    test_ds = TF.FusionDataset(list(X_test), list(map(int, y_test)), cfg=cfg, time_crop=args.time_crop)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build model and load weights (infer tab dim from the first available subset)
    probe_paths = None
    if TF.has_split_dirs(args.data_root):
        splits = TF.discover_split(args.data_root)
        if "train" in splits and len(splits["train"][0])>0:
            probe_paths = splits["train"][0]
            probe_labels = splits["train"][1]
        elif "val" in splits and len(splits["val"][0])>0:
            probe_paths = splits["val"][0]
            probe_labels = splits["val"][1]
        else:
            probe_paths = X_test; probe_labels = y_test
    else:
        probe_paths = X_test; probe_labels = y_test

    probe = TF.FusionDataset([probe_paths[0]], [int(probe_labels[0])], cfg=cfg, time_crop=args.time_crop)[0]
    _, tab_probe, _ = probe
    tab_dim = int(tab_probe.numel())

    model = TF.FusionModel(tab_dim=tab_dim, num_classes=len(TF.CLASSES)).to(device)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        test_loss, test_acc, y_true, y_pred = TF.evaluate(model, test_dl, device)

    from sklearn.metrics import classification_report, confusion_matrix
    print(f"Test loss {test_loss:.4f} | Test acc {test_acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=TF.CLASSES, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
