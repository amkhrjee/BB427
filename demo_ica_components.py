import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import ICA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/NAS/aniruddham/mne/data")
MONTAGE = mne.channels.make_standard_montage("standard_1005")
FIGURES_DIR = "figures"

SUBJECT = 1
RUNS = [4, 8, 12]  # motor imagery


def load_raw():
    sname = f"S{SUBJECT:03d}"
    paths = [DATA_ROOT / sname / f"{sname}R{r:02d}.edf" for r in RUNS]
    raw = concatenate_raws(
        [read_raw_edf(p, preload=True, verbose=False) for p in paths]
    )
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
    raw.set_eeg_reference("average", projection=False, verbose=False)
    # Wide-band filter for ICA (1-45 Hz captures all artifact types)
    raw.filter(1.0, 45.0, verbose=False)
    return raw


def main():
    print("Loading subject 1 ...")
    raw = load_raw()
    print(f"  {raw.info['nchan']} channels, {raw.n_times / raw.info['sfreq']:.0f}s")

    print("Fitting ICA (20 components) ...")
    ica = ICA(n_components=20, method="fastica", random_state=7, verbose=False)
    ica.fit(raw)

    # ── Plot 1: Component topographies (grid) ─────────────────────
    print("Plotting component topographies ...")
    fig = ica.plot_components(picks=range(20), show=False)
    if isinstance(fig, list):
        fig = fig[0]
    fig.savefig(f"{FIGURES_DIR}/ica_topographies.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ica_topographies.png")

    # ── Plot 2: Component time series (first 10s) ─────────────────
    print("Plotting component time series ...")
    fig = ica.plot_sources(raw, show=False, start=0, stop=10)
    fig.set_size_inches(14, 12)
    fig.savefig(f"{FIGURES_DIR}/ica_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ica_timeseries.png")

    # ── Plot 3: Detailed view of select components ────────────────
    # Show properties of a few components (topomap + spectrum + epochs)
    print("Plotting component properties ...")
    # Create epochs for the properties plot
    raw_for_epochs = raw.copy()
    raw_for_epochs.annotations.rename({"T1": "left_hand", "T2": "right_hand"})
    events, eid = mne.events_from_annotations(raw_for_epochs, verbose=False)
    keep = {k: v for k, v in eid.items() if k in ("left_hand", "right_hand")}
    epochs = mne.Epochs(
        raw_for_epochs,
        events,
        event_id=keep,
        tmin=-1.0,
        tmax=4.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # Plot properties for first 6 components
    for comp_idx in range(6):
        fig = ica.plot_properties(epochs, picks=[comp_idx], show=False, verbose=False)
        if isinstance(fig, list):
            fig = fig[0]
        fig.savefig(
            f"{FIGURES_DIR}/ica_component_{comp_idx}.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)
    print("  Saved ica_component_0.png through ica_component_5.png")

    # ── Plot 4: Overlay - before vs after ICA cleaning ────────────
    print("Plotting before/after ICA ...")
    # Auto-detect eye artifacts
    eog_indices, eog_scores = ica.find_bads_eog(
        raw, ch_name=["Fp1", "Fp2"], verbose=False
    )
    print(f"  Eye artifact components detected: {eog_indices}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Get a 5-second segment
    start_sec = 5.0
    stop_sec = 10.0
    start_samp = int(start_sec * raw.info["sfreq"])
    stop_samp = int(stop_sec * raw.info["sfreq"])

    ch_idx = raw.ch_names.index("Fp1")  # frontal channel (most affected by blinks)
    c3_idx = raw.ch_names.index("C3")
    times = np.arange(start_samp, stop_samp) / raw.info["sfreq"]

    # Original signal
    data_orig = raw.get_data(start=start_samp, stop=stop_samp)

    # Cleaned signal
    raw_clean = raw.copy()
    ica.exclude = eog_indices
    ica.apply(raw_clean, verbose=False)
    data_clean = raw_clean.get_data(start=start_samp, stop=stop_samp)

    # Fp1 - original
    axes[0].plot(times, data_orig[ch_idx] * 1e6, color="#D32F2F", lw=0.8)
    axes[0].set_ylabel("Fp1 (uV)", fontsize=11)
    axes[0].set_title(
        "Frontal Channel (Fp1) — Most Affected by Eye Artifacts",
        fontsize=12,
        fontweight="bold",
    )

    # Fp1 - cleaned
    axes[1].plot(times, data_clean[ch_idx] * 1e6, color="#1565C0", lw=0.8)
    axes[1].set_ylabel("Fp1 cleaned (uV)", fontsize=11)
    axes[1].set_title(
        f"After Removing {len(eog_indices)} Eye Component(s)",
        fontsize=12,
        fontweight="bold",
    )

    # C3 - both overlaid (motor cortex, less affected)
    axes[2].plot(
        times,
        data_orig[c3_idx] * 1e6,
        color="#D32F2F",
        lw=0.8,
        alpha=0.7,
        label="Original",
    )
    axes[2].plot(
        times,
        data_clean[c3_idx] * 1e6,
        color="#1565C0",
        lw=0.8,
        alpha=0.7,
        label="Cleaned",
    )
    axes[2].set_ylabel("C3 (uV)", fontsize=11)
    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].set_title(
        "Motor Cortex (C3) — Minimal Change After ICA", fontsize=12, fontweight="bold"
    )
    axes[2].legend(fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "ICA Artifact Removal: Eye Blinks", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/ica_before_after.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ica_before_after.png")

    print(f"\nAll plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
