"""
Generate interesting statistical plots about the PhysioNet EEGBCI dataset.
One plot per image.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from scipy.signal import welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
MONTAGE = mne.channels.make_standard_montage("standard_1005")
SUBJECTS = list(range(1, 110))
RUNS = [4, 8, 12]
FIGURES_DIR = "figures"


def load_subject(subj):
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
    return raw


def main():
    print("Loading all subjects ...")

    # Collect per-subject stats
    n_trials_left = []
    n_trials_right = []
    mean_amplitudes = []  # per subject, global RMS
    c3_spectra = []
    c4_spectra = []
    subject_ids = []
    sfreqs = []

    for subj in SUBJECTS:
        try:
            raw = load_subject(subj)
            sfreqs.append(raw.info["sfreq"])

            events, eid = mne.events_from_annotations(raw, verbose=False)
            keep = {k: v for k, v in eid.items()
                    if k in ("left_hand", "right_hand")}

            epochs = mne.Epochs(
                raw, events, event_id=keep,
                tmin=0.0, tmax=4.0,
                baseline=None, preload=True, verbose=False,
            )
            epochs.crop(tmin=0.5, tmax=2.5)

            left_ep = epochs["left_hand"]
            right_ep = epochs["right_hand"]
            n_trials_left.append(len(left_ep))
            n_trials_right.append(len(right_ep))

            data = epochs.get_data()  # (n_trials, 64, n_times)
            rms = np.sqrt((data ** 2).mean())
            mean_amplitudes.append(rms * 1e6)  # convert to uV

            # PSD at C3 and C4
            raw_filt = raw.copy().filter(1.0, 45.0, verbose=False)
            c3_idx = raw.ch_names.index("C3")
            c4_idx = raw.ch_names.index("C4")
            c3_data = raw_filt.get_data(picks=[c3_idx])[0]
            c4_data = raw_filt.get_data(picks=[c4_idx])[0]

            fs = raw.info["sfreq"]
            f, psd_c3 = welch(c3_data, fs=fs, nperseg=int(fs * 2))
            f, psd_c4 = welch(c4_data, fs=fs, nperseg=int(fs * 2))
            c3_spectra.append(psd_c3)
            c4_spectra.append(psd_c4)

            subject_ids.append(subj)
        except Exception as e:
            print(f"  [!] Subject {subj}: {e}")

    print(f"  Loaded {len(subject_ids)} subjects")

    # ── Plot 1: Trials per subject ──
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(subject_ids))
    ax.bar(x - 0.2, n_trials_left, 0.4, label="Left hand", color="#1565C0", alpha=0.85)
    ax.bar(x + 0.2, n_trials_right, 0.4, label="Right hand", color="#D32F2F", alpha=0.85)
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Number of Trials", fontsize=12)
    ax.set_title("Trials Per Subject Per Class", fontsize=14, fontweight="bold")
    ax.set_xticks(x[::10])
    ax.set_xticklabels([str(s) for s in subject_ids[::10]])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/dataset_trials_per_subject.png", dpi=200)
    plt.close(fig)
    print("  Saved dataset_trials_per_subject.png")

    # ── Plot 2: Amplitude variability across subjects ──
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1565C0" if a < np.percentile(mean_amplitudes, 90) else "#D32F2F"
              for a in mean_amplitudes]
    ax.bar(range(len(mean_amplitudes)), mean_amplitudes, color=colors, alpha=0.85)
    ax.axhline(np.mean(mean_amplitudes), color="black", ls="--", lw=1.5,
               label=f"Mean: {np.mean(mean_amplitudes):.1f} uV")
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("RMS Amplitude (uV)", fontsize=12)
    ax.set_title("EEG Amplitude Variability Across Subjects\n(Why we need per-trial z-normalization)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/dataset_amplitude_variability.png", dpi=200)
    plt.close(fig)
    print("  Saved dataset_amplitude_variability.png")

    # ── Plot 3: Grand average power spectrum at C3 ──
    c3_spectra = np.array(c3_spectra)
    c4_spectra = np.array(c4_spectra)
    mask = (f >= 2) & (f <= 45)

    fig, ax = plt.subplots(figsize=(10, 5))
    mean_c3 = 10 * np.log10(c3_spectra[:, mask].mean(axis=0))
    std_c3 = 10 * np.log10(c3_spectra[:, mask].std(axis=0))
    mean_c4 = 10 * np.log10(c4_spectra[:, mask].mean(axis=0))

    ax.plot(f[mask], mean_c3, color="#1565C0", lw=2, label="C3 (left motor)")
    ax.plot(f[mask], mean_c4, color="#D32F2F", lw=2, label="C4 (right motor)")
    ax.fill_between(f[mask], mean_c3 - 2, mean_c3 + 2, alpha=0.15, color="#1565C0")

    ax.axvspan(8, 14, alpha=0.1, color="green", label="Mu band (8-14 Hz)")
    ax.axvspan(14, 30, alpha=0.07, color="orange", label="Beta band (14-30 Hz)")

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density (dB)", fontsize=12)
    ax.set_title("Grand Average Power Spectrum at Motor Cortex (109 subjects)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/dataset_power_spectrum.png", dpi=200)
    plt.close(fig)
    print("  Saved dataset_power_spectrum.png")

    # ── Plot 4: Class balance histogram ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ratios = [l / (l + r) for l, r in zip(n_trials_left, n_trials_right)]
    ax.hist(ratios, bins=20, color="#1565C0", edgecolor="black", alpha=0.85)
    ax.axvline(0.5, color="red", ls="--", lw=2, label="Perfect balance (0.5)")
    ax.axvline(np.mean(ratios), color="green", ls="--", lw=2,
               label=f"Mean: {np.mean(ratios):.3f}")
    ax.set_xlabel("Left Hand Trial Proportion", fontsize=12)
    ax.set_ylabel("Number of Subjects", fontsize=12)
    ax.set_title("Class Balance Distribution Across Subjects",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/dataset_class_balance.png", dpi=200)
    plt.close(fig)
    print("  Saved dataset_class_balance.png")

    # ── Plot 5: Dataset summary infographic ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    total_trials = sum(n_trials_left) + sum(n_trials_right)
    total_left = sum(n_trials_left)
    total_right = sum(n_trials_right)
    duration_per_trial = 2.0  # seconds (after crop)
    total_hours = total_trials * 4.0 / 3600  # original 4s epochs

    stats = [
        ("Subjects", "109"),
        ("EEG Channels", "64"),
        ("Sampling Rate", "160 Hz"),
        ("Total Trials", f"{total_trials:,}"),
        ("Left Hand Trials", f"{total_left:,}"),
        ("Right Hand Trials", f"{total_right:,}"),
        ("Trial Duration (raw)", "4.0 s"),
        ("Trial Duration (cropped)", "2.0 s"),
        ("Frequency Band", "8-30 Hz"),
        ("Total Recording", f"{total_hours:.1f} hours"),
        ("Trials Per Subject", f"{total_trials/109:.0f} (avg)"),
        ("Electrode System", "10-10 (BCI2000)"),
    ]

    y_start = 0.92
    for i, (key, val) in enumerate(stats):
        y = y_start - i * 0.072
        ax.text(0.25, y, key, fontsize=13, ha="right", fontweight="bold",
                transform=ax.transAxes, color="#333333")
        ax.text(0.30, y, val, fontsize=13, ha="left",
                transform=ax.transAxes, color="#1565C0")

    ax.set_title("PhysioNet EEGBCI Motor Imagery Dataset — Summary",
                 fontsize=15, fontweight="bold", pad=20)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(2)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/dataset_summary.png", dpi=200)
    plt.close(fig)
    print("  Saved dataset_summary.png")

    print(f"\nAll plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
