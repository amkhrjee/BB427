#!/usr/bin/env python3
"""
Transfer Learning Study with EEGSym
====================================
For 64-channel and 45-channel configs, compare three training strategies:
  1. Imagery-only:  train on imagery, test on imagery
  2. Zero-shot:     train on execution, test on imagery (no imagery seen)
  3. Fine-tuned:    train on execution, fine-tune on imagery, test on imagery

Cross-subject evaluation with GroupShuffleSplit (5 splits).
Same test subjects used across all three conditions per split.

Output:  results/transfer_study_results.csv
         figures/transfer_study.png
"""

import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
SEED = 7
N_SPLITS = 5
SUBJECTS = list(range(1, 110))
IMAGERY_RUNS = [4, 8, 12]    # motor imagery left/right hand
EXEC_RUNS = [3, 7, 11]       # motor execution left/right hand
SFREQ = 160.0
FILTER_BAND = (8.0, 30.0)
EPOCH_WINDOW = (0.0, 4.0)
CROP_WINDOW = (0.5, 2.5)
BATCH_SIZE = 64
MAX_EPOCHS = 12
FINETUNE_EPOCHS = 8
LR = 1e-3
FINETUNE_LR = 3e-4
WD = 1e-4
PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONTAGE = mne.channels.make_standard_montage("standard_1005")
DATA_ROOT = Path("/NAS/aniruddham/mne/data")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 45-channel config from radial study
CHANNELS_45 = [
    "C3", "C4", "CP3", "CP4", "FC3", "FC4", "C5", "C6", "C1", "C2",
    "CP5", "CP6", "FC5", "FC6", "FC1", "FC2", "CP1", "CP2", "P3", "P4",
    "F3", "F4", "P5", "P6", "T7", "T8", "F5", "F6", "F1", "F2",
    "P1", "P2", "FT7", "FT8", "TP7", "TP8", "P7", "P8", "F7", "F8",
    "Cz", "FCz", "CPz", "Fz", "Pz",
]

# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def local_run_paths(subject, runs):
    sname = f"S{subject:03d}"
    return [DATA_ROOT / sname / f"{sname}R{run:02d}.edf" for run in runs]


def load_subject_raw(subject, runs):
    paths = local_run_paths(subject, runs)
    raw = concatenate_raws(
        [read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    )
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
    raw.annotations.rename({"T1": "left_hand", "T2": "right_hand"})
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.resample(SFREQ, verbose="ERROR")
    return raw


def epoch_subject(raw):
    raw_f = raw.copy().filter(*FILTER_BAND, verbose=False)
    events, eid = mne.events_from_annotations(raw_f, verbose=False)
    keep = {k: v for k, v in eid.items() if k in ("left_hand", "right_hand")}

    epochs = mne.Epochs(
        raw_f, events, event_id=keep,
        tmin=EPOCH_WINDOW[0], tmax=EPOCH_WINDOW[1],
        baseline=None, preload=True, verbose=False,
        reject_by_annotation=True,
    )
    epochs.crop(tmin=CROP_WINDOW[0], tmax=CROP_WINDOW[1])
    if epochs.get_data().shape[2] % 2 == 1:
        epochs.crop(tmax=epochs.times[-2])

    X = epochs.get_data().astype(np.float32)
    y_raw = epochs.events[:, -1]
    classes = sorted(keep.values())
    y = np.array([classes.index(v) for v in y_raw], dtype=np.int64)

    # per-trial z-normalisation
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-6
    X = (X - mu) / sd
    return X, y


def load_all_subjects(runs):
    Xs, ys, gs = [], [], []
    for s in SUBJECTS:
        try:
            raw = load_subject_raw(s, runs)
            X, y = epoch_subject(raw)
            Xs.append(X)
            ys.append(y)
            gs.append(np.full(len(y), s))
        except Exception as e:
            print(f"  [!] Subject {s}: {e}")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    print(f"  Loaded: X={X.shape}  y={y.shape}  subjects={len(Xs)}")
    return X, y, g


def select_channels(X, all_ch_names, wanted):
    idx = [all_ch_names.index(ch) for ch in wanted]
    return X[:, idx, :]


# ═══════════════════════════════════════════════════════════════════
# Channel helpers (EEGSym needs lr_pairs and midline)
# ═══════════════════════════════════════════════════════════════════

def get_channel_layout():
    raw = read_raw_edf(local_run_paths(1, [4])[0], preload=False, verbose=False)
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
    return list(raw.ch_names)


def compute_lr_mid(ch_list):
    """Compute left_right_pairs and midline for a given channel list."""
    from braindecode.datautil.channel_utils import (
        division_channels_idx,
        match_hemisphere_chans,
    )
    left, right, mid = division_channels_idx(ch_list)
    pairs = match_hemisphere_chans(left, right)
    return pairs, mid


# ═══════════════════════════════════════════════════════════════════
# Splits
# ═══════════════════════════════════════════════════════════════════

def make_split(groups, seed):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv, te = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 100)
    tr_rel, va_rel = next(inner.split(np.zeros(len(tv)), groups=groups[tv]))
    return tv[tr_rel], tv[va_rel], te


# ═══════════════════════════════════════════════════════════════════
# Model helpers
# ═══════════════════════════════════════════════════════════════════

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(n_chans, n_times, ch_names_ordered, lr_pairs, mid_chs):
    from braindecode.models import EEGSym
    chs_info = [{"ch_name": ch} for ch in ch_names_ordered]
    model = EEGSym(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        chs_info=chs_info,
        sfreq=SFREQ,
        scales_time=(525, 275, 125),
        left_right_chs=lr_pairs if lr_pairs else None,
        middle_chs=mid_chs if lr_pairs else None,
    )
    return model.to(DEVICE)


def unpack(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.dim() == 3:
        logits = logits[:, :, -1]
    return logits


@torch.no_grad()
def predict(model, loader):
    model.eval()
    parts = []
    for bx, _ in loader:
        parts.append(unpack(model(bx)).argmax(1).cpu().numpy())
    return np.concatenate(parts)


def make_loader(X, y, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X).to(DEVICE),
            torch.from_numpy(y).to(DEVICE),
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle,
    )


def train_model(model, train_ld, val_ld, lr, max_epochs, patience):
    """Train model, return best state dict and best val accuracy."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    crit = nn.CrossEntropyLoss()

    best_val, best_st, pat = -1.0, None, 0
    ep = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for bx, by in train_ld:
            opt.zero_grad(set_to_none=True)
            loss = crit(unpack(model(bx)), by)
            loss.backward()
            opt.step()

        vb = balanced_accuracy_score(
            val_ld.dataset.tensors[1].cpu().numpy(),
            predict(model, val_ld),
        )
        if vb > best_val + 1e-4:
            best_val, best_st, pat = vb, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= patience:
                break

    model.load_state_dict(best_st)
    return model, best_val, ep


# ═══════════════════════════════════════════════════════════════════
# Three training strategies
# ═══════════════════════════════════════════════════════════════════

def run_imagery_only(X_imag, y_imag, tr, va, te,
                     ch_ordered, lr_pairs, mid_chs, seed):
    """Strategy 1: train on imagery, test on imagery."""
    set_seed(seed)
    n_ch, n_t = X_imag.shape[1], X_imag.shape[2]
    model = build_model(n_ch, n_t, ch_ordered, lr_pairs, mid_chs)

    train_ld = make_loader(X_imag[tr], y_imag[tr], True)
    val_ld = make_loader(X_imag[va], y_imag[va], False)
    test_ld = make_loader(X_imag[te], y_imag[te], False)

    model, val_acc, ep = train_model(model, train_ld, val_ld, LR, MAX_EPOCHS, PATIENCE)
    test_acc = balanced_accuracy_score(y_imag[te], predict(model, test_ld))
    return test_acc, val_acc, ep


def run_zero_shot(X_exec, y_exec, X_imag, y_imag,
                  exec_tr, exec_va, imag_te,
                  ch_ordered, lr_pairs, mid_chs, seed):
    """Strategy 2: train on execution, test on imagery (zero-shot)."""
    set_seed(seed)
    n_ch, n_t = X_exec.shape[1], X_exec.shape[2]
    model = build_model(n_ch, n_t, ch_ordered, lr_pairs, mid_chs)

    train_ld = make_loader(X_exec[exec_tr], y_exec[exec_tr], True)
    val_ld = make_loader(X_exec[exec_va], y_exec[exec_va], False)
    test_ld = make_loader(X_imag[imag_te], y_imag[imag_te], False)

    model, val_acc, ep = train_model(model, train_ld, val_ld, LR, MAX_EPOCHS, PATIENCE)
    test_acc = balanced_accuracy_score(y_imag[imag_te], predict(model, test_ld))
    return test_acc, val_acc, ep


def run_finetuned(X_exec, y_exec, X_imag, y_imag,
                  exec_tr, exec_va, imag_tr, imag_va, imag_te,
                  ch_ordered, lr_pairs, mid_chs, seed):
    """Strategy 3: train on execution, fine-tune on imagery, test on imagery."""
    set_seed(seed)
    n_ch, n_t = X_exec.shape[1], X_exec.shape[2]
    model = build_model(n_ch, n_t, ch_ordered, lr_pairs, mid_chs)

    # Phase 1: pre-train on execution
    exec_train_ld = make_loader(X_exec[exec_tr], y_exec[exec_tr], True)
    exec_val_ld = make_loader(X_exec[exec_va], y_exec[exec_va], False)

    model, _, _ = train_model(model, exec_train_ld, exec_val_ld, LR, MAX_EPOCHS, PATIENCE)

    # Phase 2: fine-tune on imagery (lower LR, fewer epochs)
    imag_train_ld = make_loader(X_imag[imag_tr], y_imag[imag_tr], True)
    imag_val_ld = make_loader(X_imag[imag_va], y_imag[imag_va], False)
    test_ld = make_loader(X_imag[imag_te], y_imag[imag_te], False)

    model, val_acc, ep = train_model(
        model, imag_train_ld, imag_val_ld, FINETUNE_LR, FINETUNE_EPOCHS, PATIENCE
    )
    test_acc = balanced_accuracy_score(y_imag[imag_te], predict(model, test_ld))
    return test_acc, val_acc, ep


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def generate_plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    strategies = ["imagery_only", "zero_shot", "finetuned"]
    labels = ["Train: Imagery\nTest: Imagery",
              "Train: Execution\nTest: Imagery\n(zero-shot)",
              "Train: Execution\nFine-tune: Imagery\nTest: Imagery"]
    colors = ["#1565C0", "#D32F2F", "#2E7D32"]

    for ax, n_ch in zip(axes, [64, 45]):
        sub = df[df["n_channels"] == n_ch]

        means, stds = [], []
        for strat in strategies:
            vals = sub[sub["strategy"] == strat]["test_balanced_accuracy"]
            means.append(vals.mean())
            stds.append(vals.std())

        x = np.arange(len(strategies))
        bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                      edgecolor="black", linewidth=0.5, alpha=0.85, width=0.6)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=12,
                    fontweight="bold")

        ax.axhline(0.5, color="gray", ls="--", lw=1.5, alpha=0.7, label="Chance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{n_ch} Channels", fontsize=14, fontweight="bold")
        ax.set_ylim(0.45, 0.75)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Balanced Accuracy", fontsize=12)
    axes[0].legend(fontsize=10)

    plt.suptitle(
        "Transfer Learning: Motor Execution → Motor Imagery (EEGSym, 5 splits)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/transfer_study.png", dpi=200)
    plt.close(fig)
    print(f"\n  Saved {FIGURES_DIR}/transfer_study.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")

    # Channel setup
    all_ch_names = get_channel_layout()

    # Channel configs
    configs = {}

    # 64 channels (all)
    lr64, mid64 = compute_lr_mid(all_ch_names)
    configs[64] = dict(
        ch_ordered=all_ch_names,
        lr_pairs=lr64,
        mid_chs=mid64,
    )

    # 45 channels
    lr45, mid45 = compute_lr_mid(CHANNELS_45)
    configs[45] = dict(
        ch_ordered=CHANNELS_45,
        lr_pairs=lr45,
        mid_chs=mid45,
    )

    # Load data
    print("\n[1/3] Loading imagery data (runs 4, 8, 12) ...")
    X_imag_all, y_imag, g_imag = load_all_subjects(IMAGERY_RUNS)

    print("\n[2/3] Loading execution data (runs 3, 7, 11) ...")
    X_exec_all, y_exec, g_exec = load_all_subjects(EXEC_RUNS)

    # Split by subject IDs (not indices) so both datasets use same subjects
    all_subjects = np.unique(g_imag)
    print(f"\n  Common subjects: {len(all_subjects)}")

    # Run experiments
    print(f"\n[3/3] Running experiments: 2 configs x 3 strategies x {N_SPLITS} splits "
          f"= {2 * 3 * N_SPLITS} runs")

    csv_path = os.path.join(RESULTS_DIR, "transfer_study_results.csv")
    results = []
    run_no = 0
    total_runs = 2 * 3 * N_SPLITS

    for n_ch in [64, 45]:
        cfg = configs[n_ch]
        ch_ordered = cfg["ch_ordered"]
        lr_pairs = cfg["lr_pairs"]
        mid_chs = cfg["mid_chs"]

        X_imag = select_channels(X_imag_all, all_ch_names, ch_ordered) if n_ch != 64 else X_imag_all
        X_exec = select_channels(X_exec_all, all_ch_names, ch_ordered) if n_ch != 64 else X_exec_all

        print(f"\n{'='*60}")
        print(f"  {n_ch} CHANNELS")
        print(f"{'='*60}")

        for sp in range(N_SPLITS):
            seed = SEED + sp

            # Split subjects into train/val/test (same subjects for both datasets)
            rng = np.random.RandomState(seed)
            perm = rng.permutation(all_subjects)
            n_test = max(1, int(len(perm) * 0.2))
            n_val = max(1, int((len(perm) - n_test) * 0.2))
            test_subjs = set(perm[:n_test])
            val_subjs = set(perm[n_test:n_test + n_val])
            train_subjs = set(perm[n_test + n_val:])

            # Get indices for each dataset
            imag_tr = np.where(np.isin(g_imag, list(train_subjs)))[0]
            imag_va = np.where(np.isin(g_imag, list(val_subjs)))[0]
            imag_te = np.where(np.isin(g_imag, list(test_subjs)))[0]
            exec_tr = np.where(np.isin(g_exec, list(train_subjs)))[0]
            exec_va = np.where(np.isin(g_exec, list(val_subjs)))[0]

            # --- Strategy 1: Imagery only ---
            t0 = time.time()
            test_b, val_b, ep = run_imagery_only(
                X_imag, y_imag, imag_tr, imag_va, imag_te,
                ch_ordered, lr_pairs, mid_chs, seed
            )
            dt = time.time() - t0
            run_no += 1
            results.append(dict(
                n_channels=n_ch, strategy="imagery_only", split=sp,
                test_balanced_accuracy=round(test_b, 5),
                val_balanced_accuracy=round(val_b, 5),
                best_epoch=ep, train_time_s=round(dt, 1),
            ))
            print(f"  Split {sp+1} | imagery_only: test={test_b:.4f} "
                  f"ep={ep} {dt:.0f}s [{run_no}/{total_runs}]")

            # --- Strategy 2: Zero-shot ---
            t0 = time.time()
            test_b, val_b, ep = run_zero_shot(
                X_exec, y_exec, X_imag, y_imag,
                exec_tr, exec_va, imag_te,
                ch_ordered, lr_pairs, mid_chs, seed
            )
            dt = time.time() - t0
            run_no += 1
            results.append(dict(
                n_channels=n_ch, strategy="zero_shot", split=sp,
                test_balanced_accuracy=round(test_b, 5),
                val_balanced_accuracy=round(val_b, 5),
                best_epoch=ep, train_time_s=round(dt, 1),
            ))
            print(f"  Split {sp+1} | zero_shot:    test={test_b:.4f} "
                  f"ep={ep} {dt:.0f}s [{run_no}/{total_runs}]")

            # --- Strategy 3: Fine-tuned ---
            t0 = time.time()
            test_b, val_b, ep = run_finetuned(
                X_exec, y_exec, X_imag, y_imag,
                exec_tr, exec_va, imag_tr, imag_va, imag_te,
                ch_ordered, lr_pairs, mid_chs, seed
            )
            dt = time.time() - t0
            run_no += 1
            results.append(dict(
                n_channels=n_ch, strategy="finetuned", split=sp,
                test_balanced_accuracy=round(test_b, 5),
                val_balanced_accuracy=round(val_b, 5),
                best_epoch=ep, train_time_s=round(dt, 1),
            ))
            print(f"  Split {sp+1} | finetuned:    test={test_b:.4f} "
                  f"ep={ep} {dt:.0f}s [{run_no}/{total_runs}]")

            # Incremental save
            pd.DataFrame(results).to_csv(csv_path, index=False)

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for n_ch in [64, 45]:
        print(f"\n  {n_ch} channels:")
        for strat in ["imagery_only", "zero_shot", "finetuned"]:
            vals = df[(df["n_channels"] == n_ch) & (df["strategy"] == strat)][
                "test_balanced_accuracy"
            ]
            print(f"    {strat:<15}: {vals.mean():.4f} +/- {vals.std():.4f}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # Plot
    generate_plot(df)
    print("\nDone!")


if __name__ == "__main__":
    main()
