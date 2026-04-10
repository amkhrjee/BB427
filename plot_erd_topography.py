"""
ERD topography for left vs right hand motor imagery.
Uses all 109 subjects for a clean grand average.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
MONTAGE = mne.channels.make_standard_montage("standard_1005")
SUBJECTS = list(range(1, 110))
IMAGERY_RUNS = [4, 8, 12]
EXECUTION_RUNS = [3, 7, 11]


def load_all_epochs(runs):
    left_epochs, right_epochs = [], []
    for subj in SUBJECTS:
        try:
            sname = f"S{subj:03d}"
            paths = [DATA_ROOT / sname / f"{sname}R{r:02d}.edf" for r in runs]
            raw = concatenate_raws(
                [read_raw_edf(p, preload=True, verbose=False) for p in paths]
            )
            eegbci.standardize(raw)
            raw.pick("eeg")
            raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
            raw.annotations.rename({"T1": "left_hand", "T2": "right_hand"})
            raw.set_eeg_reference("average", projection=False, verbose=False)
            raw.resample(160.0, verbose=False)
            raw.filter(1.0, 45.0, verbose=False)

            events, eid = mne.events_from_annotations(raw, verbose=False)
            keep = {k: v for k, v in eid.items()
                    if k in ("left_hand", "right_hand")}

            epochs = mne.Epochs(
                raw, events, event_id=keep,
                tmin=-1.5, tmax=4.0,
                baseline=None, preload=True, verbose=False,
            )
            left_epochs.append(epochs["left_hand"])
            right_epochs.append(epochs["right_hand"])
        except Exception as e:
            print(f"  [!] Subject {subj}: {e}")

    left = mne.concatenate_epochs(left_epochs)
    right = mne.concatenate_epochs(right_epochs)
    print(f"  Left hand: {len(left)} epochs")
    print(f"  Right hand: {len(right)} epochs")
    return left, right


def compute_topos(left_epochs, right_epochs):
    freqs = np.arange(8, 15, 1)  # mu band only
    n_cycles = freqs / 2.0

    print("  Computing TFR for left hand ...")
    tfr_left = tfr_multitaper(
        left_epochs, freqs=freqs, n_cycles=n_cycles,
        return_itc=False, verbose=False, average=True,
    )
    print("  Computing TFR for right hand ...")
    tfr_right = tfr_multitaper(
        right_epochs, freqs=freqs, n_cycles=n_cycles,
        return_itc=False, verbose=False, average=True,
    )

    def compute_erd(tfr):
        baseline_mask = (tfr.times >= -1.5) & (tfr.times <= -0.5)
        baseline_power = tfr.data[:, :, baseline_mask].mean(axis=2, keepdims=True)
        return (tfr.data - baseline_power) / baseline_power * 100

    erd_left = compute_erd(tfr_left)
    erd_right = compute_erd(tfr_right)

    task_mask = (tfr_left.times >= 0.5) & (tfr_left.times <= 2.5)
    topo_left = erd_left[:, :, task_mask].mean(axis=(1, 2))
    topo_right = erd_right[:, :, task_mask].mean(axis=(1, 2))
    topo_diff = topo_left - topo_right

    return topo_left, topo_right, topo_diff, tfr_left.info


def main():
    # Load both conditions
    print("Loading IMAGERY epochs (runs 4, 8, 12) ...")
    imag_left, imag_right = load_all_epochs(IMAGERY_RUNS)

    print("\nLoading EXECUTION epochs (runs 3, 7, 11) ...")
    exec_left, exec_right = load_all_epochs(EXECUTION_RUNS)

    # Compute topographies
    print("\nComputing imagery ERD ...")
    imag_tl, imag_tr, imag_diff, info = compute_topos(imag_left, imag_right)

    print("\nComputing execution ERD ...")
    exec_tl, exec_tr, exec_diff, _ = compute_topos(exec_left, exec_right)

    # Shared color limits
    vlim_erd = max(
        abs(imag_tl.min()), abs(imag_tl.max()),
        abs(imag_tr.min()), abs(imag_tr.max()),
        abs(exec_tl.min()), abs(exec_tl.max()),
        abs(exec_tr.min()), abs(exec_tr.max()),
    )
    vlim_erd = min(vlim_erd, 30)

    vlim_diff = max(
        abs(imag_diff.min()), abs(imag_diff.max()),
        abs(exec_diff.min()), abs(exec_diff.max()),
    )
    vlim_diff = min(vlim_diff, 20)

    # ── Combined figure: 2 rows x 3 cols ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    rows = [
        ("Motor Execution", exec_tl, exec_tr, exec_diff),
        ("Motor Imagery", imag_tl, imag_tr, imag_diff),
    ]

    for row_idx, (row_label, tl, tr, diff) in enumerate(rows):
        im1, _ = mne.viz.plot_topomap(
            tl, info, axes=axes[row_idx, 0], show=False,
            cmap="RdBu_r", vlim=(-vlim_erd, vlim_erd),
        )
        axes[row_idx, 0].set_title(
            f"Left hand\nERD% (8-14 Hz)" if row_idx == 0 else "",
            fontsize=11, fontweight="bold",
        )
        axes[row_idx, 0].set_ylabel(row_label, fontsize=13, fontweight="bold",
                                     labelpad=15)

        im2, _ = mne.viz.plot_topomap(
            tr, info, axes=axes[row_idx, 1], show=False,
            cmap="RdBu_r", vlim=(-vlim_erd, vlim_erd),
        )
        axes[row_idx, 1].set_title(
            f"Right hand\nERD% (8-14 Hz)" if row_idx == 0 else "",
            fontsize=11, fontweight="bold",
        )

        im3, _ = mne.viz.plot_topomap(
            diff, info, axes=axes[row_idx, 2], show=False,
            cmap="RdBu_r", vlim=(-vlim_diff, vlim_diff),
        )
        axes[row_idx, 2].set_title(
            f"Difference (Left - Right)\nLateralization" if row_idx == 0 else "",
            fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        "Mu-Band ERD Topography: Execution vs Imagery (109 subjects, 8-14 Hz)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout(w_pad=2, h_pad=2)
    fig.savefig("figures/erd_topography.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved figures/erd_topography.png")

    # ── Also save imagery-only version ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im1, _ = mne.viz.plot_topomap(
        imag_tl, info, axes=axes[0], show=False,
        cmap="RdBu_r", vlim=(-vlim_erd, vlim_erd),
    )
    axes[0].set_title("Left hand imagery\nERD% (8-14 Hz)", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[0], label="ERD%", shrink=0.8)

    im2, _ = mne.viz.plot_topomap(
        imag_tr, info, axes=axes[1], show=False,
        cmap="RdBu_r", vlim=(-vlim_erd, vlim_erd),
    )
    axes[1].set_title("Right hand imagery\nERD% (8-14 Hz)", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axes[1], label="ERD%", shrink=0.8)

    im3, _ = mne.viz.plot_topomap(
        imag_diff, info, axes=axes[2], show=False,
        cmap="RdBu_r", vlim=(-vlim_diff, vlim_diff),
    )
    axes[2].set_title("Difference (Left - Right)\nLateralization", fontsize=12, fontweight="bold")
    plt.colorbar(im3, ax=axes[2], label="Diff ERD%", shrink=0.8)

    fig.suptitle(
        "Motor Imagery ERD Topography (109 subjects, mu band 8-14 Hz)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig("figures/erd_topography_imagery_only.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/erd_topography_imagery_only.png")


if __name__ == "__main__":
    main()
