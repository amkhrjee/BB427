import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, ".")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from run_radial_channel_study import compute_radial_subsets, get_channel_layout

# ── Load layout ──
ch_names, positions = get_channel_layout()
subsets, additions = compute_radial_subsets(ch_names, positions)

# ── Azimuthal equidistant projection (standard EEG topographic view) ──
# Converts 3D sphere positions to 2D circle, preserving angular distances
xs, ys, names = [], [], []
for ch in ch_names:
    if ch in positions:
        p = np.array(positions[ch])
        r = np.linalg.norm(p)
        x, y, z = p / r  # normalize to unit sphere
        theta = np.arccos(np.clip(z, -1, 1))  # polar angle from top of head
        phi = np.arctan2(y, x)  # azimuthal angle
        # project: distance from center = polar angle, direction = azimuthal
        proj_r = theta / (np.pi / 2)  # normalize so equator = 1.0
        xs.append(proj_r * np.cos(phi))
        ys.append(proj_r * np.sin(phi))
        names.append(ch)

xs = np.array(xs)
ys = np.array(ys)

# ── Head outline (circle) ──
theta = np.linspace(0, 2 * np.pi, 200)
pad = 0.08
head_r = max(np.sqrt(xs**2 + ys**2)) + pad
head_x = head_r * np.cos(theta)
head_y = head_r * np.sin(theta)

# Nose triangle
nose_w = 0.06
nose_h = 0.07
nose_x = [-nose_w, 0, nose_w]
nose_y = [head_r, head_r + nose_h, head_r]

# ── Figure setup ──
fig, ax = plt.subplots(figsize=(7, 7.5))


def draw_frame(frame_idx):
    ax.clear()

    # head outline
    ax.plot(head_x, head_y, "k-", lw=1.5)
    ax.plot(nose_x, nose_y, "k-", lw=1.5)

    subset = subsets[frame_idx]
    active = set(subset["all_channels"])
    n_ch = subset["n_channels"]

    # what was just added in this frame
    item = additions[frame_idx]
    if item[0] == "pair":
        just_added = {item[1][0], item[1][1]}
        added_label = f"{item[1][0]} / {item[1][1]}"
    else:
        just_added = {item[1]}
        added_label = item[1]

    # draw all electrodes
    for i, ch in enumerate(names):
        if ch in just_added:
            # just added: green highlight
            ax.scatter(
                xs[i], ys[i], s=110, c="#4CAF50", edgecolors="black", lw=1.2, zorder=4
            )
        elif ch in active:
            # active: filled blue
            ax.scatter(
                xs[i], ys[i], s=80, c="#1565C0", edgecolors="#0D47A1", lw=0.8, zorder=3
            )
        else:
            # inactive: hollow grey
            ax.scatter(
                xs[i],
                ys[i],
                s=60,
                facecolors="none",
                edgecolors="#BDBDBD",
                lw=0.8,
                zorder=2,
            )

        # labels for active channels only
        if ch in active:
            ax.annotate(
                ch,
                (xs[i], ys[i]),
                fontsize=5.5,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
                color="#212121",
                fontweight="bold",
            )

    # title
    ax.set_title(
        f"Radial Channel Expansion from Motor Cortex (C3/C4)\n"
        f"{n_ch} / 64 channels active",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # info box
    ax.text(
        0.02,
        0.02,
        f"Added: {added_label}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#4CAF50", alpha=0.9),
    )

    # C3/C4 star markers (always visible)
    for ch in ("C3", "C4"):
        i = names.index(ch)
        ax.scatter(
            xs[i],
            ys[i],
            s=180,
            marker="*",
            c="red",
            zorder=5,
            edgecolors="darkred",
            lw=0.5,
        )

    ax.set_xlim(-head_r - 0.02, head_r + 0.02)
    ax.set_ylim(-head_r - 0.02, head_r + nose_h + 0.02)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Animate ──
n_frames = len(subsets)
anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=600)

out_path = "figures/channel_expansion.gif"
anim.save(out_path, writer=PillowWriter(fps=2))
plt.close(fig)
print(f"Saved {out_path}  ({n_frames} frames)")
