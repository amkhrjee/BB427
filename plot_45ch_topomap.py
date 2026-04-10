"""
Plot topographic electrode map showing the 45 channels used in the
'min reliable' configuration from the radial channel study.
Uses MNE's built-in sensor plotting.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.io import read_raw_edf
from mne.datasets import eegbci
from pathlib import Path

FIGURES_DIR = "figures"
DATA_ROOT = Path("/NAS/aniruddham/mne/data")

CHANNELS_45 = [
    "C3", "C4", "CP3", "CP4", "FC3", "FC4", "C5", "C6", "C1", "C2",
    "CP5", "CP6", "FC5", "FC6", "FC1", "FC2", "CP1", "CP2", "P3", "P4",
    "F3", "F4", "P5", "P6", "T7", "T8", "F5", "F6", "F1", "F2",
    "P1", "P2", "FT7", "FT8", "TP7", "TP8", "P7", "P8", "F7", "F8",
    "Cz", "FCz", "CPz", "Fz", "Pz",
]


def main():
    # Load one subject to get the full 64-channel info with montage
    raw = read_raw_edf(
        DATA_ROOT / "S001" / "S001R04.edf", preload=False, verbose=False
    )
    eegbci.standardize(raw)
    raw.pick("eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="warn")

    all_ch = raw.ch_names
    active_set = {ch.upper() for ch in CHANNELS_45}

    # Get 2D positions from MNE's layout
    pos = mne.channels.make_eeg_layout(raw.info).pos  # (n_ch, 4): x, y, w, h

    fig, ax = plt.subplots(figsize=(9, 9))

    # Separate active and inactive
    active_xy = []
    inactive_xy = []
    active_labels = []
    inactive_labels = []

    for i, ch in enumerate(all_ch):
        xy = pos[i, :2]
        if ch.upper() in active_set:
            active_xy.append(xy)
            active_labels.append(ch)
        else:
            inactive_xy.append(xy)
            inactive_labels.append(ch)

    active_xy = np.array(active_xy)
    inactive_xy = np.array(inactive_xy)

    # Draw head
    all_xy = pos[:, :2]
    center = all_xy.mean(axis=0)
    radius = np.max(np.linalg.norm(all_xy - center, axis=1)) * 1.1

    head = plt.Circle(center, radius, fill=False, color="black", linewidth=2)
    ax.add_patch(head)

    # Nose
    nose_len = radius * 0.08
    ax.plot(
        [center[0] - radius * 0.07, center[0], center[0] + radius * 0.07],
        [center[1] + radius, center[1] + radius + nose_len, center[1] + radius],
        color="black", linewidth=2,
    )

    # Ears
    for side in [-1, 1]:
        ear_x = center[0] + side * radius
        ax.plot(
            [ear_x, ear_x + side * radius * 0.05, ear_x],
            [center[1] + radius * 0.08, center[1], center[1] - radius * 0.08],
            color="black", linewidth=2,
        )

    # Inactive
    ax.scatter(
        inactive_xy[:, 0], inactive_xy[:, 1],
        s=100, facecolors="none", edgecolors="gray", linewidths=1.5,
        zorder=3, label=f"Unused ({len(inactive_xy)})",
    )
    for (px, py), lab in zip(inactive_xy, inactive_labels):
        ax.annotate(lab, (px, py), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points", color="gray")

    # Active
    ax.scatter(
        active_xy[:, 0], active_xy[:, 1],
        s=120, c="#1565C0", edgecolors="black", linewidths=0.8,
        zorder=4, label=f"Active ({len(active_xy)})",
    )
    for (px, py), lab in zip(active_xy, active_labels):
        ax.annotate(lab, (px, py), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points", color="black",
                    fontweight="bold")

    # Mark C3/C4 with red stars
    for ch in ["C3", "C4"]:
        idx = all_ch.index(ch)
        ax.scatter(
            [pos[idx, 0]], [pos[idx, 1]],
            s=250, marker="*", c="red", edgecolors="black",
            linewidths=0.5, zorder=5,
        )

    margin = radius * 0.3
    ax.set_xlim(center[0] - radius - margin, center[0] + radius + margin)
    ax.set_ylim(center[1] - radius - margin, center[1] + radius + margin + nose_len)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title(
        "45-Channel Configuration (Min Reliable)\n"
        "Red stars = C3/C4 (expansion center)",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/topomap_45ch.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/topomap_45ch.png")


if __name__ == "__main__":
    main()
