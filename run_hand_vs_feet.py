#!/usr/bin/env python3
"""
Hand vs Feet Motor Imagery Classification using EEGSym
======================================================
Uses the same channel sets found effective for left/right hand
discrimination to classify both-fists vs both-feet imagery.

Runs 5, 9, 13: T1 = both fists imagery, T2 = both feet imagery

Tests two channel configurations:
  - 45 channels (min reliable set from radial study)
  - 64 channels (full cap, for reference)

Fixed: model (EEGSym), preprocessing (8-30 Hz bandpass, no ICA)
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
RUNS = [5, 9, 13]  # both fists vs both feet imagery
SFREQ = 160.0
FILTER_BAND = (8.0, 30.0)
EPOCH_WINDOW = (0.0, 4.0)
CROP_WINDOW = (0.5, 2.5)
BATCH_SIZE = 64
MAX_EPOCHS = 12
LR = 1e-3
WD = 1e-4
PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONTAGE = mne.channels.make_standard_montage("standard_1005")
DATA_ROOT = Path("/NAS/aniruddham/mne/data")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Channel configurations from radial study
CHANNEL_CONFIGS = {
    "45ch_reliable": {
        "channels": ["C3","C4","CP3","CP4","FC3","FC4","C5","C6","C1","C2",
                      "CP5","CP6","FC5","FC6","FC1","FC2","CP1","CP2",
                      "P3","P4","F3","F4","P5","P6","T7","T8","F5","F6",
                      "F1","F2","P1","P2","FT7","FT8","TP7","TP8",
                      "P7","P8","F7","F8","Cz","FCz","CPz","Fz","Pz"],
        "pairs": [("C3","C4"),("CP3","CP4"),("FC3","FC4"),("C5","C6"),
                  ("C1","C2"),("CP5","CP6"),("FC5","FC6"),("FC1","FC2"),
                  ("CP1","CP2"),("P3","P4"),("F3","F4"),("P5","P6"),
                  ("T7","T8"),("F5","F6"),("F1","F2"),("P1","P2"),
                  ("FT7","FT8"),("TP7","TP8"),("P7","P8"),("F7","F8")],
        "midline": ["Cz","FCz","CPz","Fz","Pz"],
    },
    "64ch_full": {
        "channels": None,  # use all
        "pairs": None,     # auto-detect
        "midline": None,
    },
}

# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def local_run_paths(subject, runs):
    sname = f"S{subject:03d}"
    return [DATA_ROOT / sname / f"{sname}R{run:02d}.edf" for run in runs]


def load_subject_raw(subject):
    paths = local_run_paths(subject, RUNS)
    raw = concatenate_raws(
        [read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    )
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
    raw.annotations.rename({"T1": "both_fists", "T2": "both_feet"})
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.resample(SFREQ, verbose="ERROR")
    return raw


def epoch_subject(raw):
    raw_f = raw.copy().filter(*FILTER_BAND, verbose=False)
    events, eid = mne.events_from_annotations(raw_f, verbose=False)
    keep = {k: v for k, v in eid.items() if k in ("both_fists", "both_feet")}

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

    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-6
    X = (X - mu) / sd
    return X, y


def load_all_subjects():
    Xs, ys, gs = [], [], []
    for s in SUBJECTS:
        try:
            raw = load_subject_raw(s)
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


# ═══════════════════════════════════════════════════════════════════
# Channel selection & splits
# ═══════════════════════════════════════════════════════════════════

def select_channels(X, all_ch_names, wanted):
    if wanted is None:
        return X
    idx = [all_ch_names.index(ch) for ch in wanted]
    return X[:, idx, :]


def make_split(groups, seed):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv, te = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 100)
    tr_rel, va_rel = next(inner.split(np.zeros(len(tv)), groups=groups[tv]))
    return tv[tr_rel], tv[va_rel], te


# ═══════════════════════════════════════════════════════════════════
# Model
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
        left_right_chs=lr_pairs,
        middle_chs=mid_chs,
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


def train_eval(X_tr, y_tr, X_va, y_va, X_te, y_te,
               ch_names_ordered, lr_pairs, mid_chs, seed):
    set_seed(seed)
    n_ch, n_t = X_tr.shape[1], X_tr.shape[2]
    model = build_model(n_ch, n_t, ch_names_ordered, lr_pairs, mid_chs)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()

    mk = lambda X, y, shuf: DataLoader(
        TensorDataset(
            torch.from_numpy(X).to(DEVICE),
            torch.from_numpy(y).to(DEVICE),
        ),
        batch_size=BATCH_SIZE, shuffle=shuf,
    )
    tr_ld, va_ld, te_ld = mk(X_tr, y_tr, True), mk(X_va, y_va, False), mk(X_te, y_te, False)

    best_val, best_st, pat = -1.0, None, 0
    ep = 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for bx, by in tr_ld:
            opt.zero_grad(set_to_none=True)
            loss = crit(unpack(model(bx)), by)
            loss.backward()
            opt.step()
        vb = balanced_accuracy_score(y_va, predict(model, va_ld))
        if vb > best_val + 1e-4:
            best_val, best_st, pat = vb, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_st)
    test_bal = balanced_accuracy_score(y_te, predict(model, te_ld))
    return test_bal, best_val, ep


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def generate_plots(df):
    # Load hand results for comparison
    hand_df = pd.read_csv("results/radial_channel_results.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Hand vs feet results (this experiment)
    for cfg_name, color, marker in [
        ("45ch_reliable", "#FF9800", "D"),
        ("64ch_full", "#F44336", "^"),
    ]:
        sub = df[df["config"] == cfg_name]
        n_ch = sub["n_channels"].iloc[0]
        m = sub["test_balanced_accuracy"].mean()
        s = sub["test_balanced_accuracy"].std()
        ax.errorbar(n_ch, m, yerr=s, fmt=marker, ms=12, color=color,
                    capsize=6, capthick=2, lw=2, zorder=5,
                    label=f"Fists vs Feet — {cfg_name} ({m:.1%})")

    # Hand radial curve (from previous experiment)
    hand_summ = (
        hand_df.groupby("n_channels")["test_balanced_accuracy"]
        .agg(["mean", "std"]).reset_index()
    )
    hand_summ.columns = ["n_channels", "mean", "std"]
    hand_summ = hand_summ.sort_values("n_channels")

    ax.fill_between(hand_summ["n_channels"],
                    hand_summ["mean"] - hand_summ["std"],
                    hand_summ["mean"] + hand_summ["std"],
                    alpha=0.15, color="#1565C0")
    ax.plot(hand_summ["n_channels"], hand_summ["mean"],
            "o-", color="#1565C0", lw=1.5, ms=4, alpha=0.7,
            label="Left vs Right Hand (radial study)")

    ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Chance (50%)")
    ax.set_xlabel("Number of Channels", fontsize=13)
    ax.set_ylabel("Cross-Subject Balanced Accuracy", fontsize=13)
    ax.set_title("Fists vs Feet Imagery: Can Hand-Optimal Channels Generalize?",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.44, None)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/hand_vs_feet_comparison.png", dpi=200)
    plt.close(fig)
    print("  Saved hand_vs_feet_comparison.png")

    # Bar chart: task comparison at each channel config
    fig, ax = plt.subplots(figsize=(9, 5))

    configs = ["45ch_reliable", "64ch_full"]
    x = np.arange(len(configs))
    w = 0.35

    # Feet means
    feet_means, feet_stds = [], []
    for cfg in configs:
        sub = df[df["config"] == cfg]
        feet_means.append(sub["test_balanced_accuracy"].mean())
        feet_stds.append(sub["test_balanced_accuracy"].std())

    # Hand means at matching channel counts
    hand_means, hand_stds = [], []
    for cfg in configs:
        n = df[df["config"] == cfg]["n_channels"].iloc[0]
        hsub = hand_df[hand_df["n_channels"] == n]
        if len(hsub) > 0:
            hand_means.append(hsub["test_balanced_accuracy"].mean())
            hand_stds.append(hsub["test_balanced_accuracy"].std())
        else:
            hand_means.append(np.nan)
            hand_stds.append(0)

    ax.bar(x - w/2, hand_means, w, yerr=hand_stds, label="Left vs Right Hand",
           color="#1565C0", capsize=5, alpha=0.85)
    ax.bar(x + w/2, feet_means, w, yerr=feet_stds, label="Fists vs Feet",
           color="#4CAF50", capsize=5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["45 ch\n(reliable)", "64 ch\n(full)"])
    ax.axhline(0.5, color="red", ls="--", alpha=0.5)
    ax.set_ylabel("Cross-Subject Balanced Accuracy", fontsize=12)
    ax.set_title("Task Comparison: Hand Discrimination vs Fists/Feet",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.44, None)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/task_comparison_bars.png", dpi=200)
    plt.close(fig)
    print("  Saved task_comparison_bars.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 60)
    print("Hand vs Feet Imagery — EEGSym")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Runs: {RUNS} (T1=both fists, T2=both feet)")

    # Get channel names from one subject
    sname = "S001"
    raw_tmp = read_raw_edf(
        DATA_ROOT / sname / f"{sname}R05.edf",
        preload=False, verbose=False,
    )
    eegbci.standardize(raw_tmp)
    raw_tmp.pick("eeg")
    raw_tmp.set_montage(MONTAGE, match_case=False, on_missing="warn")
    all_ch_names = list(raw_tmp.ch_names)

    print("\nLoading all subjects ...")
    X_all, y_all, groups = load_all_subjects()
    n_times = X_all.shape[2]

    csv_path = os.path.join(RESULTS_DIR, "hand_vs_feet_results.csv")
    results = []
    total = len(CHANNEL_CONFIGS) * N_SPLITS
    run_no = 0

    for cfg_name, cfg in CHANNEL_CONFIGS.items():
        wanted = cfg["channels"]
        pairs = cfg["pairs"]
        mid = cfg["midline"]

        if wanted is not None:
            X_sub = select_channels(X_all, all_ch_names, wanted)
            ordered = wanted
            n_ch = len(wanted)
        else:
            X_sub = X_all
            ordered = all_ch_names
            n_ch = len(all_ch_names)

        print(f"\n  {cfg_name} ({n_ch} channels)")

        for sp in range(N_SPLITS):
            seed = SEED + sp
            tr, va, te = make_split(groups, seed)
            t0 = time.time()
            test_b, val_b, ep = train_eval(
                X_sub[tr], y_all[tr],
                X_sub[va], y_all[va],
                X_sub[te], y_all[te],
                ordered, pairs, mid, seed,
            )
            dt = time.time() - t0
            run_no += 1

            results.append(dict(
                config=cfg_name,
                n_channels=n_ch,
                split=sp,
                test_balanced_accuracy=round(test_b, 5),
                val_balanced_accuracy=round(val_b, 5),
                best_epoch=ep,
                train_time_s=round(dt, 1),
            ))
            print(f"    split {sp+1}: test={test_b:.4f}  val={val_b:.4f}"
                  f"  ep={ep}  {dt:.0f}s  [{run_no}/{total}]")

        pd.DataFrame(results).to_csv(csv_path, index=False)
        vals = [r["test_balanced_accuracy"]
                for r in results if r["config"] == cfg_name]
        print(f"    => {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Plots
    print("\nGenerating plots ...")
    df = pd.DataFrame(results)
    generate_plots(df)

    # Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("RESULTS  (copy-paste friendly)")
    print("=" * 60)
    for cfg_name in CHANNEL_CONFIGS:
        sub = df[df["config"] == cfg_name]
        m = sub["test_balanced_accuracy"].mean()
        s = sub["test_balanced_accuracy"].std()
        n = sub["n_channels"].iloc[0]
        print(f"  {cfg_name:>20} ({n:>2} ch): {m:.4f} +/- {s:.4f}")

    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results CSV : {csv_path}")
    print(f"Figures     : {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
