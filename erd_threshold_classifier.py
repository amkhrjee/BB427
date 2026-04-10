import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.metrics import balanced_accuracy_score

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
MONTAGE = mne.channels.make_standard_montage("standard_1005")
SUBJECTS = list(range(1, 110))
RUNS = [4, 8, 12]  # motor imagery left/right hand
THRESHOLD = 4.0


def main():
    all_diffs = []  # ERD%_C4 - ERD%_C3 per trial
    all_labels = []  # 0 = left, 1 = right
    subject_ids = []  # which subject each trial belongs to

    print("Loading all subjects and computing ERD lateralization ...")

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
            keep = {k: v for k, v in eid.items() if k in ("left_hand", "right_hand")}

            epochs = mne.Epochs(
                raw,
                events,
                event_id=keep,
                tmin=-1.5,
                tmax=4.0,
                baseline=None,
                preload=True,
                verbose=False,
            )

            c3_idx = epochs.ch_names.index("C3")
            c4_idx = epochs.ch_names.index("C4")
            times = epochs.times
            baseline_mask = (times >= -1.5) & (times <= -0.5)
            task_mask = (times >= 0.5) & (times <= 2.5)

            data = epochs.get_data()  # (n_trials, n_channels, n_times)
            labels = epochs.events[:, 2]

            for i in range(len(epochs)):
                # Mu-band power = mean(x^2) for bandpassed signal
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
                print(f"  Processed {subj}/109 subjects ...")

        except Exception as e:
            print(f"  [!] Subject {subj}: {e}")

    all_diffs = np.array(all_diffs)
    all_labels = np.array(all_labels)
    subject_ids = np.array(subject_ids)

    print(f"\nTotal trials: {len(all_diffs)}")
    print(f"  Left hand: {(all_labels == 0).sum()}")
    print(f"  Right hand: {(all_labels == 1).sum()}")

    # ── Main result with user-specified threshold ──
    print("\n" + "=" * 60)
    print(f"THRESHOLD = {THRESHOLD}%")
    print("=" * 60)

    preds = (all_diffs > THRESHOLD).astype(int)
    overall_ba = balanced_accuracy_score(all_labels, preds)
    print(f"Overall balanced accuracy: {overall_ba * 100:.2f}%")

    # Per-subject balanced accuracy
    unique_subjs = np.unique(subject_ids)
    per_subj_acc = []
    for s in unique_subjs:
        mask = subject_ids == s
        ba = balanced_accuracy_score(all_labels[mask], preds[mask])
        per_subj_acc.append(ba)

    per_subj_acc = np.array(per_subj_acc)
    print(f"\nPer-subject balanced accuracy:")
    print(f"  Mean:   {per_subj_acc.mean() * 100:.2f}%")
    print(f"  Std:    {per_subj_acc.std() * 100:.2f}%")
    print(f"  Median: {np.median(per_subj_acc) * 100:.2f}%")
    print(
        f"  Min:    {per_subj_acc.min() * 100:.2f}% (S{unique_subjs[per_subj_acc.argmin()]:03d})"
    )
    print(
        f"  Max:    {per_subj_acc.max() * 100:.2f}% (S{unique_subjs[per_subj_acc.argmax()]:03d})"
    )

    above_chance = (per_subj_acc > 0.5).sum()
    print(f"\n  Subjects above chance (>50%): {above_chance}/{len(unique_subjs)}")

    # ── ERD diff distribution by class ──
    print(f"\nERD diff% (C4 - C3) distribution:")
    print(
        f"  Left hand trials:  mean = {all_diffs[all_labels == 0].mean():+.2f}%, "
        f"std = {all_diffs[all_labels == 0].std():.2f}%"
    )
    print(
        f"  Right hand trials: mean = {all_diffs[all_labels == 1].mean():+.2f}%, "
        f"std = {all_diffs[all_labels == 1].std():.2f}%"
    )

    # ── Threshold sweep ──
    print(f"\n{'Threshold':>10} {'Bal. Acc':>10} {'Pred Left':>11} {'Pred Right':>12}")
    print("-" * 47)
    best_thr, best_ba = 0, 0
    for thr in np.arange(-20, 22, 2):
        p = (all_diffs > thr).astype(int)
        ba = balanced_accuracy_score(all_labels, p)
        n_left = (p == 0).sum()
        n_right = (p == 1).sum()
        marker = " ***" if abs(thr - THRESHOLD) < 0.1 else ""
        if ba > best_ba:
            best_thr, best_ba = thr, ba
        print(f"{thr:>9.0f}% {ba * 100:>9.2f}% {n_left:>11} {n_right:>12}{marker}")

    print(
        f"\nOptimal threshold: {best_thr:.0f}% → {best_ba * 100:.2f}% balanced accuracy"
    )


if __name__ == "__main__":
    main()
