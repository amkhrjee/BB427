"""
Plot the annotated durations of T0, T1, T2 events across all recordings.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import defaultdict

from mne.io import read_raw_edf
from mne.datasets import eegbci

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
SUBJECTS = list(range(1, 110))
RUNS = [4, 8, 12]  # motor imagery left/right hand
FIGURES_DIR = "figures"


def main():
    durations = defaultdict(list)  # "T0" -> [dur1, dur2, ...]

    print("Reading annotation durations ...")
    for subj in SUBJECTS:
        sname = f"S{subj:03d}"
        for r in RUNS:
            path = DATA_ROOT / sname / f"{sname}R{r:02d}.edf"
            try:
                raw = read_raw_edf(path, preload=False, verbose=False)
                eegbci.standardize(raw)
                for ann in raw.annotations:
                    durations[ann["description"]].append(ann["duration"])
            except Exception as e:
                print(f"  [!] S{subj:03d} R{r:02d}: {e}")

        if subj % 20 == 0:
            print(f"  Processed {subj}/109 ...")

    # Print stats
    for key in sorted(durations.keys()):
        vals = np.array(durations[key])
        print(f"\n{key}: {len(vals)} events")
        print(f"  Unique durations: {sorted(set(np.round(vals, 2)))}")
        print(f"  Mean: {vals.mean():.2f}s, Std: {vals.std():.2f}s")

    # ── Plot ──
    keys = ["T0", "T1", "T2"]
    labels = ["T0 (Rest)", "T1 (Left Hand)", "T2 (Right Hand)"]
    colors = ["#888888", "#1565C0", "#D32F2F"]

    # Get unique durations per event type
    unique_durs = {}
    for k in keys:
        vals = np.round(durations[k], 2)
        unique_durs[k] = sorted(set(vals))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(keys))
    width = 0.5

    # Most events have a fixed duration, so show the most common one as the bar
    mean_durs = [np.mean(durations[k]) for k in keys]
    bars = ax.bar(x, mean_durs, width, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85)

    for bar, k, dur in zip(bars, keys, mean_durs):
        vals = np.array(durations[k])
        unique = sorted(set(np.round(vals, 2)))
        count = len(vals)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{dur:.2f}s\n({count} events)\nUnique: {unique}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Annotated Duration (seconds)", fontsize=12)
    ax.set_title("EDF Annotation Durations for Motor Imagery Runs (4, 8, 12)\n"
                 "109 subjects",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(mean_durs) + 1.0)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/annotation_durations.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved {FIGURES_DIR}/annotation_durations.png")


if __name__ == "__main__":
    main()
