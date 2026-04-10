"""
Compare motor execution vs motor imagery EEG signals.
Plots:
  1. Time-frequency maps (ERD/ERS) at C3 and C4 for both conditions
  2. Topographic ERD maps for both conditions
  3. Overlaid power spectra at C3/C4
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
FIGURES_DIR = "figures"

# Use a pool of subjects for cleaner averages
SUBJECTS = list(range(1, 21))  # first 20 subjects

EXEC_RUNS = [3, 7, 11]   # actual left/right hand movement
IMAG_RUNS = [4, 8, 12]   # imagined left/right hand movement


def load_epochs(subjects, runs, label):
    """Load and preprocess epochs for a set of runs."""
    all_epochs = []
    for subj in subjects:
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

            # Wider filter for time-frequency analysis
            raw.filter(1.0, 45.0, verbose=False)

            events, eid = mne.events_from_annotations(raw, verbose=False)
            keep = {k: v for k, v in eid.items()
                    if k in ("left_hand", "right_hand")}

            epochs = mne.Epochs(
                raw, events, event_id=keep,
                tmin=-1.0, tmax=4.0,
                baseline=None, preload=True, verbose=False,
            )
            all_epochs.append(epochs)
        except Exception as e:
            print(f"  [!] Subject {subj}: {e}")

    combined = mne.concatenate_epochs(all_epochs)
    print(f"  {label}: {len(combined)} epochs from {len(all_epochs)} subjects")
    return combined


def plot_tfr(exec_epochs, imag_epochs):
    """Plot time-frequency maps at C3 and C4 for execution vs imagery."""
    freqs = np.arange(4, 36, 1)
    n_cycles = freqs / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for col, (epochs, cond_label) in enumerate(
        [(exec_epochs, "Motor Execution"), (imag_epochs, "Motor Imagery")]
    ):
        # left hand trials → contralateral = C4
        # right hand trials → contralateral = C3
        # average all trials for simplicity
        tfr = tfr_multitaper(
            epochs, freqs=freqs, n_cycles=n_cycles,
            return_itc=False, verbose=False, average=True,
        )

        for row, ch in enumerate(["C3", "C4"]):
            ax = axes[row, col]
            ch_idx = tfr.ch_names.index(ch)

            # Baseline correction: percent change relative to pre-stimulus
            data = tfr.data[ch_idx]  # (n_freqs, n_times)
            times = tfr.times
            baseline_mask = times < 0
            baseline_power = data[:, baseline_mask].mean(axis=1, keepdims=True)
            erd = (data - baseline_power) / baseline_power * 100

            im = ax.pcolormesh(
                times, freqs, erd,
                cmap="RdBu_r", vmin=-60, vmax=60, shading="auto",
            )
            ax.axvline(0, color="black", ls="--", lw=1, alpha=0.7)
            ax.set_ylabel("Frequency (Hz)", fontsize=11)
            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_title(f"{cond_label} — {ch}", fontsize=12, fontweight="bold")
            plt.colorbar(im, ax=ax, label="ERD/ERS (%)")

    plt.suptitle(
        "Time-Frequency Maps: Motor Execution vs Motor Imagery",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/exec_vs_imag_tfr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exec_vs_imag_tfr.png")


def plot_spectra(exec_epochs, imag_epochs):
    """Overlay power spectra at C3 and C4."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ch in zip(axes, ["C3", "C4"]):
        for epochs, label, color in [
            (exec_epochs, "Execution", "#D32F2F"),
            (imag_epochs, "Imagery", "#1565C0"),
        ]:
            ch_idx = epochs.ch_names.index(ch)
            # Crop to task period
            task = epochs.copy().crop(tmin=0.5, tmax=3.5)
            data = task.get_data()[:, ch_idx, :]  # (n_epochs, n_times)

            # Welch PSD
            from scipy.signal import welch
            fs = epochs.info["sfreq"]
            freqs, psd = welch(data, fs=fs, nperseg=int(fs), axis=1)
            psd_mean = psd.mean(axis=0)
            psd_std = psd.std(axis=0) / np.sqrt(len(psd))

            mask = (freqs >= 4) & (freqs <= 40)
            ax.plot(freqs[mask], 10 * np.log10(psd_mean[mask]),
                    color=color, lw=2, label=label)
            ax.fill_between(
                freqs[mask],
                10 * np.log10(psd_mean[mask] - psd_std[mask]),
                10 * np.log10(psd_mean[mask] + psd_std[mask]),
                alpha=0.2, color=color,
            )

        # Mark mu and beta bands
        ax.axvspan(8, 14, alpha=0.08, color="green", label="Mu band (8-14 Hz)")
        ax.axvspan(14, 30, alpha=0.06, color="orange", label="Beta band (14-30 Hz)")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Power (dB)", fontsize=12)
        ax.set_title(f"Power Spectrum at {ch}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Execution vs Imagery: Power Spectra During Task",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/exec_vs_imag_spectra.png", dpi=200)
    plt.close(fig)
    print("  Saved exec_vs_imag_spectra.png")


def plot_topomap(exec_epochs, imag_epochs):
    """Plot ERD topographic maps for both conditions at key time points."""
    freqs = np.arange(8, 14, 0.5)  # mu band
    n_cycles = freqs / 2.0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    time_windows = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 3.5)]
    time_labels = ["0.5-1.0s", "1.0-1.5s", "1.5-2.5s", "2.5-3.5s"]

    for row, (epochs, cond) in enumerate(
        [(exec_epochs, "Motor Execution"), (imag_epochs, "Motor Imagery")]
    ):
        tfr = tfr_multitaper(
            epochs, freqs=freqs, n_cycles=n_cycles,
            return_itc=False, verbose=False, average=True,
        )

        # Baseline: pre-stimulus
        baseline_mask = tfr.times < 0
        baseline_power = tfr.data[:, :, baseline_mask].mean(axis=2, keepdims=True)
        erd = (tfr.data - baseline_power) / baseline_power * 100
        # Average over mu frequencies
        erd_mu = erd.mean(axis=1)  # (n_channels, n_times)

        for col, (t1, t2) in enumerate(time_windows):
            ax = axes[row, col]
            time_mask = (tfr.times >= t1) & (tfr.times <= t2)
            topo_data = erd_mu[:, time_mask].mean(axis=1)

            mne.viz.plot_topomap(
                topo_data, tfr.info, axes=ax, show=False,
                cmap="RdBu_r", vlim=(-50, 50),
            )
            if row == 0:
                ax.set_title(time_labels[col], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(cond, fontsize=12, fontweight="bold")

    plt.suptitle(
        "Mu-Band (8-14 Hz) ERD Topography: Execution vs Imagery",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/exec_vs_imag_topomap.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved exec_vs_imag_topomap.png")


def main():
    print("=" * 55)
    print("Motor Execution vs Motor Imagery Comparison")
    print("=" * 55)

    print("\nLoading execution epochs...")
    exec_epochs = load_epochs(SUBJECTS, EXEC_RUNS, "Execution")

    print("Loading imagery epochs...")
    imag_epochs = load_epochs(SUBJECTS, IMAG_RUNS, "Imagery")

    print("\nGenerating plots...")
    plot_tfr(exec_epochs, imag_epochs)
    plot_spectra(exec_epochs, imag_epochs)
    plot_topomap(exec_epochs, imag_epochs)

    print(f"\nAll plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
