import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

DATA_ROOT = Path("/NAS/aniruddham/mne/data")

for run, label in [
    (4, "left fist / right fist imagery"),
    (5, "both fists / both feet imagery"),
]:
    path = DATA_ROOT / "S001" / f"S001R{run:02d}.edf"
    raw = read_raw_edf(path, preload=False, verbose=False)
    eegbci.standardize(raw)

    print(f"\n{'=' * 55}")
    print(f"Run {run}: {label}")
    print(f"{'=' * 55}")
    print(f"{'onset':>8}  {'duration':>8}  description")
    print(f"{'-' * 8}  {'-' * 8}  {'-' * 11}")
    for ann in raw.annotations:
        print(f"{ann['onset']:7.2f}s  {ann['duration']:7.1f}s  {ann['description']}")

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"\nevent_id mapping: {dict(event_id)}")
    print(
        f"Total events: {len(events)}  (T0={sum(events[:, 2] == 1)}, T1={sum(events[:, 2] == 2)}, T2={sum(events[:, 2] == 3)})"
    )
