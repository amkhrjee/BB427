#!/usr/bin/env python3


import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
SEED = 7
N_SPLITS = 5
SUBJECTS = list(range(1, 110))
RUNS = [4, 8, 12]  # motor imagery only
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

# ═══════════════════════════════════════════════════════════════════
# 1. Radial channel ordering
# ═══════════════════════════════════════════════════════════════════


def local_run_paths(subject, runs):
    sname = f"S{subject:03d}"
    return [DATA_ROOT / sname / f"{sname}R{run:02d}.edf" for run in runs]


def get_channel_layout():
    """Get the 64 channel names and 3-D positions from one subject."""
    raw = read_raw_edf(local_run_paths(1, [4])[0], preload=False, verbose=False)
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
    ch_names = list(raw.ch_names)
    pos_dict = MONTAGE.get_positions()["ch_pos"]
    positions = {ch: pos_dict[ch] for ch in ch_names if ch in pos_dict}
    return ch_names, positions


def compute_radial_subsets(ch_names, positions):
    """
    Build nested channel subsets that expand radially from C3/C4.

    Returns
    -------
    subsets : list[dict]
        Each dict has keys: n_channels, left_right_pairs, midline,
        all_channels (ordered: L1 R1 L2 R2 ... M1 M2 ...).
    additions : list[tuple]
        The ordered additions (for printing/debugging).
    """
    from braindecode.datautil.channel_utils import (
        division_channels_idx,
        match_hemisphere_chans,
    )

    c3 = np.array(positions["C3"])
    c4 = np.array(positions["C4"])

    # classify & pair
    left, right, mid = division_channels_idx(ch_names)
    pairs = match_hemisphere_chans(left, right)  # [(L, R), ...]

    # distance of each pair = avg of (L→C3, R→C4)
    pair_dists = []
    for l, r in pairs:
        dl = np.linalg.norm(np.array(positions[l]) - c3)
        dr = np.linalg.norm(np.array(positions[r]) - c4)
        pair_dists.append(((l, r), (dl + dr) / 2))

    # distance of each midline channel = min(→C3, →C4)
    mid_dists = []
    for ch in mid:
        d = min(
            np.linalg.norm(np.array(positions[ch]) - c3),
            np.linalg.norm(np.array(positions[ch]) - c4),
        )
        mid_dists.append((ch, d))

    # merge into one sorted list
    additions = []
    for (l, r), d in pair_dists:
        additions.append(("pair", (l, r), d))
    for ch, d in mid_dists:
        additions.append(("mid", ch, d))
    additions.sort(key=lambda x: x[2])

    # build nested subsets
    subsets = []
    cur_pairs, cur_mid = [], []
    for item in additions:
        if item[0] == "pair":
            cur_pairs.append(item[1])
        else:
            cur_mid.append(item[1])

        ordered = []
        for l, r in cur_pairs:
            ordered.extend([l, r])
        ordered.extend(cur_mid)

        subsets.append(
            dict(
                n_channels=len(ordered),
                left_right_pairs=list(cur_pairs),
                midline=list(cur_mid),
                all_channels=list(ordered),
            )
        )
    return subsets, additions


# ═══════════════════════════════════════════════════════════════════
# 2. Data loading & preprocessing
# ═══════════════════════════════════════════════════════════════════


def load_subject_raw(subject):
    paths = local_run_paths(subject, RUNS)
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
        raw_f,
        events,
        event_id=keep,
        tmin=EPOCH_WINDOW[0],
        tmax=EPOCH_WINDOW[1],
        baseline=None,
        preload=True,
        verbose=False,
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


def load_all_subjects(ch_names):
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
# 3. Channel selection
# ═══════════════════════════════════════════════════════════════════


def select_channels(X, all_ch_names, wanted):
    idx = [all_ch_names.index(ch) for ch in wanted]
    return X[:, idx, :]


# ═══════════════════════════════════════════════════════════════════
# 4. Cross-subject splits
# ═══════════════════════════════════════════════════════════════════


def make_split(groups, seed):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv, te = next(outer.split(np.zeros(len(groups)), groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 100)
    tr_rel, va_rel = next(inner.split(np.zeros(len(tv)), groups=groups[tv]))
    return tv[tr_rel], tv[va_rel], te


# ═══════════════════════════════════════════════════════════════════
# 5. Model helpers
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
        scales_time=(525, 275, 125),  # tuned for 160 Hz
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


def train_eval(
    X_tr, y_tr, X_va, y_va, X_te, y_te, ch_names_ordered, lr_pairs, mid_chs, seed
):
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
        batch_size=BATCH_SIZE,
        shuffle=shuf,
    )
    tr_ld = mk(X_tr, y_tr, True)
    va_ld = mk(X_va, y_va, False)
    te_ld = mk(X_te, y_te, False)

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
# 6. Plotting
# ═══════════════════════════════════════════════════════════════════


def generate_plots(df, subsets, positions, ch_names):
    summ = (
        df.groupby("n_channels")["test_balanced_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summ.columns = ["n_channels", "mean", "std"]
    summ = summ.sort_values("n_channels")

    # ── Main result: channels vs accuracy ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        summ["n_channels"],
        summ["mean"] - summ["std"],
        summ["mean"] + summ["std"],
        alpha=0.2,
        color="#2196F3",
    )
    ax.plot(
        summ["n_channels"],
        summ["mean"],
        "o-",
        color="#1565C0",
        lw=2,
        ms=5,
        label="EEGSym",
    )
    ax.axhline(0.5, color="red", ls="--", alpha=0.6, label="Chance (50%)")

    # "knee": smallest n_channels within 2 pp of max
    mx = summ["mean"].max()
    knee = summ.loc[summ["mean"] >= mx - 0.02].iloc[0]
    ax.axvline(
        knee["n_channels"],
        color="green",
        ls=":",
        alpha=0.7,
        label=f"Min reliable: {int(knee['n_channels'])} ch ({knee['mean']:.1%})",
    )
    ax.scatter(
        [knee["n_channels"]],
        [knee["mean"]],
        s=120,
        zorder=5,
        color="green",
        edgecolors="black",
    )

    ax.set_xlabel("Number of Channels", fontsize=13)
    ax.set_ylabel("Cross-Subject Balanced Accuracy", fontsize=13)
    ax.set_title(
        "Radial Channel Reduction: How Many Channels Do You Need?",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.44, None)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/channels_vs_accuracy.png", dpi=200)
    plt.close(fig)
    print("  Saved channels_vs_accuracy.png")

    # ── Topographic map coloured by radial distance ────────────────
    c3 = np.array(positions["C3"])
    c4 = np.array(positions["C4"])
    xs, ys, ds, ns = [], [], [], []
    for ch in ch_names:
        if ch not in positions:
            continue
        p = np.array(positions[ch])
        d = min(np.linalg.norm(p - c3), np.linalg.norm(p - c4))
        # 2-D projection: y (anterior-posterior) vs x (left-right)
        xs.append(p[0])  # left-right
        ys.append(p[1])  # anterior-posterior
        ds.append(d)
        ns.append(ch)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        xs, ys, c=ds, cmap="RdYlGn_r", s=120, edgecolors="black", lw=0.5, zorder=2
    )
    for i, n in enumerate(ns):
        ax.annotate(
            n,
            (xs[i], ys[i]),
            fontsize=6,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )
    for ch in ("C3", "C4"):
        j = ns.index(ch)
        ax.scatter(
            xs[j], ys[j], s=200, marker="*", c="red", zorder=3, edgecolors="black"
        )
    plt.colorbar(sc, ax=ax, label="Distance from C3 / C4")
    ax.set_title(
        "Electrode Positions by Distance from Motor Cortex",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Left  \u2190\u2192  Right")
    ax.set_ylabel("Posterior  \u2190\u2192  Anterior")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/channel_radial_map.png", dpi=200)
    plt.close(fig)
    print("  Saved channel_radial_map.png")

    # ── Box plots at selected channel counts ───────────────────────
    counts = sorted(df["n_channels"].unique())
    if len(counts) > 14:
        idx = np.linspace(0, len(counts) - 1, 14, dtype=int)
        counts = [counts[i] for i in idx]
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [
        df.loc[df["n_channels"] == n, "test_balanced_accuracy"].values for n in counts
    ]
    bp = ax.boxplot(data, labels=[str(n) for n in counts], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BBDEFB")
        patch.set_edgecolor("#1565C0")
    ax.axhline(0.5, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("Number of Channels", fontsize=12)
    ax.set_ylabel("Cross-Subject Balanced Accuracy", fontsize=12)
    ax.set_title("Accuracy Distribution Across Splits", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/channels_boxplot.png", dpi=200)
    plt.close(fig)
    print("  Saved channels_boxplot.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    print("=" * 60)
    print("Radial Channel Reduction Study  —  EEGSym")
    print("=" * 60)
    print(f"Device : {DEVICE}")

    # --- channel ordering ---
    print("\n[1/4] Computing radial channel ordering ...")
    ch_names, positions = get_channel_layout()
    subsets, additions = compute_radial_subsets(ch_names, positions)
    print(
        f"  {len(subsets)} nested configs ({subsets[0]['n_channels']}"
        f" → {subsets[-1]['n_channels']} channels)"
    )
    print("\n  Order of addition:")
    for i, item in enumerate(additions):
        tag = f"{item[1][0]}/{item[1][1]}" if item[0] == "pair" else item[1]
        print(f"    {i + 1:>2}. {tag:<12} ({item[0]}, d={item[2]:.4f})")

    # --- load data ---
    print("\n[2/4] Loading all subjects ...")
    X_all, y_all, groups = load_all_subjects(ch_names)
    n_times = X_all.shape[2]
    print(f"  n_times = {n_times}")

    # --- experiments ---
    total_runs = len(subsets) * N_SPLITS
    print(
        f"\n[3/4] Training: {len(subsets)} configs x {N_SPLITS} splits"
        f" = {total_runs} runs"
    )

    csv_path = os.path.join(RESULTS_DIR, "radial_channel_results.csv")
    results = []
    run_no = 0

    for si, sub in enumerate(subsets):
        n_ch = sub["n_channels"]
        pairs = sub["left_right_pairs"]
        mid = sub["midline"]
        ordered = sub["all_channels"]

        X_sub = select_channels(X_all, ch_names, ordered)

        print(f"\n  [{si + 1}/{len(subsets)}] {n_ch} ch: {', '.join(ordered)}")

        for sp in range(N_SPLITS):
            seed = SEED + sp
            tr, va, te = make_split(groups, seed)
            t0 = time.time()
            test_b, val_b, ep = train_eval(
                X_sub[tr],
                y_all[tr],
                X_sub[va],
                y_all[va],
                X_sub[te],
                y_all[te],
                ordered,
                pairs,
                mid,
                seed,
            )
            dt = time.time() - t0
            run_no += 1

            results.append(
                dict(
                    n_channels=n_ch,
                    channels="|".join(ordered),
                    split=sp,
                    test_balanced_accuracy=round(test_b, 5),
                    val_balanced_accuracy=round(val_b, 5),
                    best_epoch=ep,
                    train_time_s=round(dt, 1),
                )
            )
            print(
                f"    split {sp + 1}: test={test_b:.4f}  val={val_b:.4f}"
                f"  ep={ep}  {dt:.0f}s  [{run_no}/{total_runs}]"
            )

        # incremental save
        pd.DataFrame(results).to_csv(csv_path, index=False)

        # running mean for this config
        vals = [r["test_balanced_accuracy"] for r in results if r["n_channels"] == n_ch]
        print(f"    => {n_ch} ch: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # --- plots ---
    print("\n[4/4] Generating plots ...")
    df = pd.DataFrame(results)
    generate_plots(df, subsets, positions, ch_names)

    # --- final summary ---
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("RESULTS  (copy-paste friendly)")
    print("=" * 60)
    pivot = (
        df.groupby("n_channels")["test_balanced_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    pivot.columns = ["n_channels", "mean_acc", "std_acc"]
    pivot = pivot.sort_values("n_channels")
    print(pivot.to_string(index=False, float_format="%.4f"))
    print(f"\nTotal time: {elapsed / 3600:.1f} h")
    print(f"Results CSV : {csv_path}")
    print(f"Figures     : {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
