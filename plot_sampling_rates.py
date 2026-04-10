"""
Check sampling rates across all recordings in the dataset.
Plot how many are at 160 Hz vs other rates.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter

from mne.io import read_raw_edf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
SUBJECTS = list(range(1, 110))
RUNS = list(range(1, 15))  # all 14 runs
FIGURES_DIR = "figures"


def main():
    sfreqs = Counter()
    per_subject = {}  # subj -> set of sfreqs

    print("Checking sampling rates across all recordings ...")
    for subj in SUBJECTS:
        sname = f"S{subj:03d}"
        subj_rates = set()
        for r in RUNS:
            path = DATA_ROOT / sname / f"{sname}R{r:02d}.edf"
            try:
                raw = read_raw_edf(path, preload=False, verbose=False)
                sf = raw.info["sfreq"]
                sfreqs[sf] += 1
                subj_rates.add(sf)
            except Exception:
                pass
        per_subject[subj] = subj_rates
        if subj % 20 == 0:
            print(f"  Processed {subj}/109 ...")

    total = sum(sfreqs.values())
    print(f"\nTotal recordings checked: {total}")
    print(f"Sampling rate distribution:")
    for rate, count in sorted(sfreqs.items()):
        print(f"  {rate:.0f} Hz: {count} recordings ({count/total*100:.1f}%)")

    # Subjects with non-160 Hz recordings
    non160_subjects = [s for s, rates in per_subject.items() if rates != {160.0}]
    print(f"\nSubjects with non-160 Hz recordings: {len(non160_subjects)}")
    for s in non160_subjects:
        print(f"  S{s:03d}: {sorted(per_subject[s])}")

    # ── Plot ──
    rates = sorted(sfreqs.keys())
    counts = [sfreqs[r] for r in rates]
    rate_labels = [f"{r:.0f} Hz" for r in rates]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1565C0" if r == 160.0 else "#D32F2F" for r in rates]
    bars = ax.bar(rate_labels, counts, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)

    ax.set_xlabel("Sampling Rate", fontsize=12)
    ax.set_ylabel("Number of Recordings", fontsize=12)
    ax.set_title("Sampling Rate Distribution Across All Recordings\n"
                 f"109 subjects x 14 runs = {total} recordings",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/sampling_rates.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved {FIGURES_DIR}/sampling_rates.png")


if __name__ == "__main__":
    main()
