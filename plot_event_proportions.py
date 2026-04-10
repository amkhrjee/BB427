"""
Plot the proportion of T0 (rest), T1 (left hand), T2 (right hand) events
for motor execution fist runs (3, 7, 11) and motor imagery fist runs (4, 8, 12).
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter

import mne
import numpy as np
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
SUBJECTS = list(range(1, 110))
EXEC_RUNS = [3, 7, 11]
IMAG_RUNS = [4, 8, 12]
FIGURES_DIR = "figures"


def count_events(subjects, runs):
    """Count T0, T1, T2 annotations across all subjects for given runs."""
    counts = Counter()
    for subj in subjects:
        sname = f"S{subj:03d}"
        for r in runs:
            try:
                path = DATA_ROOT / sname / f"{sname}R{r:02d}.edf"
                raw = read_raw_edf(path, preload=False, verbose=False)
                eegbci.standardize(raw)
                for ann in raw.annotations:
                    counts[ann["description"]] += 1
            except Exception as e:
                print(f"  [!] S{subj:03d} R{r:02d}: {e}")
    return counts


def main():
    print("Counting events for execution runs (3, 7, 11) ...")
    exec_counts = count_events(SUBJECTS, EXEC_RUNS)
    print(f"  {dict(exec_counts)}")

    print("Counting events for imagery runs (4, 8, 12) ...")
    imag_counts = count_events(SUBJECTS, IMAG_RUNS)
    print(f"  {dict(imag_counts)}")

    # Extract counts
    labels = ["T0 (Rest)", "T1 (Left Hand)", "T2 (Right Hand)"]
    keys = ["T0", "T1", "T2"]

    exec_vals = [exec_counts.get(k, 0) for k in keys]
    imag_vals = [imag_counts.get(k, 0) for k in keys]

    exec_total = sum(exec_vals)
    imag_total = sum(imag_vals)

    exec_pct = [v / exec_total * 100 for v in exec_vals]
    imag_pct = [v / imag_total * 100 for v in imag_vals]

    print(f"\nExecution: {exec_vals} (total {exec_total})")
    print(f"  Proportions: {[f'{p:.1f}%' for p in exec_pct]}")
    print(f"Imagery:   {imag_vals} (total {imag_total})")
    print(f"  Proportions: {[f'{p:.1f}%' for p in imag_pct]}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, exec_pct, width,
                   label="Motor Execution (runs 3, 7, 11)",
                   color="#D32F2F", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, imag_pct, width,
                   label="Motor Imagery (runs 4, 8, 12)",
                   color="#1565C0", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Add count labels on bars
    for bar, val, pct in zip(bars1, exec_vals, exec_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    for bar, val, pct in zip(bars2, imag_vals, imag_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Event Type", fontsize=12)
    ax.set_ylabel("Proportion (%)", fontsize=12)
    ax.set_title("Event Proportions: Motor Execution vs Imagery (Left/Right Hand)\n109 subjects",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(exec_pct), max(imag_pct)) + 8)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/event_proportions.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved {FIGURES_DIR}/event_proportions.png")


if __name__ == "__main__":
    main()
