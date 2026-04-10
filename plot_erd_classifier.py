"""
Plot the result of the ERD threshold classifier:
  - Overlaid histograms of ERD diff% for left vs right hand trials
  - Threshold line showing why single-trial classification fails
  - Annotation with balanced accuracy
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.metrics import balanced_accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
MONTAGE = mne.channels.make_standard_montage("standard_1005")
SUBJECTS = list(range(1, 110))
RUNS = [4, 8, 12]
THRESHOLD = 4.0
FIGURES_DIR = "figures"


def load_diffs():
    all_diffs = []
    all_labels = []
    subject_ids = []

    for subj in SUBJECTS:
        try:
            sname = f"S{subj:03d}"
            paths = [DATA_ROOT / sname / f"{sname}R{r:02d}.edf" for r in RUNS]
            raw = concatenate_raws(
                [read_raw_edf(p, preload=True, verbose=False) for p in paths]
            )
            eegbci.standardize(raw)
            raw.pick("eeg")
            raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
            raw.annotations.rename({"T1": "left_hand", "T2": "right_hand"})
            raw.set_eeg_reference("average", projection=False, verbose=False)
            raw.resample(160.0, verbose=False)
            raw.filter(8.0, 14.0, verbose=False)

            events, eid = mne.events_from_annotations(raw, verbose=False)
            keep = {k: v for k, v in eid.items()
                    if k in ("left_hand", "right_hand")}

            epochs = mne.Epochs(
                raw, events, event_id=keep,
                tmin=-1.5, tmax=4.0,
                baseline=None, preload=True, verbose=False,
            )

            c3_idx = epochs.ch_names.index("C3")
            c4_idx = epochs.ch_names.index("C4")
            times = epochs.times
            baseline_mask = (times >= -1.5) & (times <= -0.5)
            task_mask = (times >= 0.5) & (times <= 2.5)

            data = epochs.get_data()
            labels = epochs.events[:, 2]

            for i in range(len(epochs)):
                base_c3 = np.mean(data[i, c3_idx, baseline_mask] ** 2)
                base_c4 = np.mean(data[i, c4_idx, baseline_mask] ** 2)
                task_c3 = np.mean(data[i, c3_idx, task_mask] ** 2)
                task_c4 = np.mean(data[i, c4_idx, task_mask] ** 2)

                erd_c3 = (task_c3 - base_c3) / base_c3 * 100
                erd_c4 = (task_c4 - base_c4) / base_c4 * 100
                diff = erd_c4 - erd_c3

                all_diffs.append(diff)
                all_labels.append(1 if labels[i] == keep["right_hand"] else 0)
                subject_ids.append(subj)

            if subj % 20 == 0:
                print(f"  Processed {subj}/109 ...")

        except Exception as e:
            print(f"  [!] Subject {subj}: {e}")

    return np.array(all_diffs), np.array(all_labels), np.array(subject_ids)


def main():
    print("Loading data ...")
    diffs, labels, subj_ids = load_diffs()

    left_diffs = diffs[labels == 0]
    right_diffs = diffs[labels == 1]

    # Clip for plotting (the tails are extreme outliers)
    clip = 300
    left_clip = np.clip(left_diffs, -clip, clip)
    right_clip = np.clip(right_diffs, -clip, clip)

    preds = (diffs > THRESHOLD).astype(int)
    ba = balanced_accuracy_score(labels, preds)

    # Per-subject accuracy
    unique_subjs = np.unique(subj_ids)
    per_subj_acc = []
    for s in unique_subjs:
        m = subj_ids == s
        per_subj_acc.append(balanced_accuracy_score(labels[m], preds[m]))
    per_subj_acc = np.array(per_subj_acc)

    # ── Plot: overlaid histograms with threshold ──
    fig, ax = plt.subplots(figsize=(14, 6))
    bins = np.linspace(-clip, clip, 80)
    ax.hist(left_clip, bins=bins, alpha=0.6, color="#1565C0", label="Left hand", density=True)
    ax.hist(right_clip, bins=bins, alpha=0.6, color="#D32F2F", label="Right hand", density=True)

    ax.axvline(THRESHOLD, color="black", ls="--", lw=2,
               label=f"Threshold = {THRESHOLD}%")

    # Annotate the decision regions
    ylim = ax.get_ylim()
    ax.fill_betweenx([0, ylim[1] * 0.15], -clip, THRESHOLD,
                     alpha=0.06, color="#1565C0")
    ax.fill_betweenx([0, ylim[1] * 0.15], THRESHOLD, clip,
                     alpha=0.06, color="#D32F2F")
    ax.text(THRESHOLD - 30, ylim[1] * 0.12, "Predict\nLeft",
            fontsize=10, color="#1565C0", fontweight="bold", ha="center")
    ax.text(THRESHOLD + 30, ylim[1] * 0.12, "Predict\nRight",
            fontsize=10, color="#D32F2F", fontweight="bold", ha="center")

    # Class means
    ax.axvline(left_diffs.mean(), color="#1565C0", ls=":", lw=1.5, alpha=0.8)
    ax.axvline(right_diffs.mean(), color="#D32F2F", ls=":", lw=1.5, alpha=0.8)
    ax.text(left_diffs.mean(), ylim[1] * 0.95,
            f"  mean = {left_diffs.mean():+.1f}%",
            fontsize=9, color="#1565C0", va="top")
    ax.text(right_diffs.mean(), ylim[1] * 0.88,
            f"  mean = {right_diffs.mean():+.1f}%",
            fontsize=9, color="#D32F2F", va="top")

    # Stats box
    stats_text = (
        f"Balanced accuracy: {ba*100:.1f}%\n"
        f"(chance = 50%)\n"
        f"n = {len(diffs)} trials"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    ax.set_xlabel("ERD Lateralization: ERD%$_{C4}$ - ERD%$_{C3}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Single-Trial ERD Lateralization Distribution\n"
                 "Left vs Right Hand Motor Imagery (109 subjects, mu band)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-clip, clip)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/erd_threshold_classifier.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved {FIGURES_DIR}/erd_threshold_classifier.png")


if __name__ == "__main__":
    main()
