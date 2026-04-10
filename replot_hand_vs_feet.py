"""Re-generate hand_vs_feet plots from existing CSVs with wider figure."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"

df = pd.read_csv("results/hand_vs_feet_results.csv")
hand_df = pd.read_csv("results/radial_channel_results.csv")

# ── Plot 1: overlay on radial curve ──
fig, ax = plt.subplots(figsize=(14, 6))

for cfg_name, color, marker in [
    ("45ch_reliable", "#FF9800", "D"),
    ("64ch_full", "#F44336", "^"),
]:
    sub = df[df["config"] == cfg_name]
    n_ch = sub["n_channels"].iloc[0]
    m = sub["test_balanced_accuracy"].mean()
    s = sub["test_balanced_accuracy"].std()
    ax.errorbar(n_ch, m, yerr=s, fmt=marker, ms=12, color=color,
                capsize=6, capthick=2, lw=2, zorder=5,
                label=f"Fists vs Feet — {cfg_name} ({m:.1%})")

hand_summ = (
    hand_df.groupby("n_channels")["test_balanced_accuracy"]
    .agg(["mean", "std"]).reset_index()
)
hand_summ.columns = ["n_channels", "mean", "std"]
hand_summ = hand_summ.sort_values("n_channels")

ax.fill_between(hand_summ["n_channels"],
                hand_summ["mean"] - hand_summ["std"],
                hand_summ["mean"] + hand_summ["std"],
                alpha=0.15, color="#1565C0")
ax.plot(hand_summ["n_channels"], hand_summ["mean"],
        "o-", color="#1565C0", lw=1.5, ms=4, alpha=0.7,
        label="Left vs Right Hand (radial study)")

ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Chance (50%)")
ax.set_xlabel("Number of Channels", fontsize=13)
ax.set_ylabel("Cross-Subject Balanced Accuracy", fontsize=13)
ax.set_title("Fists vs Feet Imagery: Can Hand-Optimal Channels Generalize?",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.44, None)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/hand_vs_feet_comparison.png", dpi=200)
plt.close(fig)
print(f"Saved {FIGURES_DIR}/hand_vs_feet_comparison.png")

# ── Plot 2: bar chart ──
fig, ax = plt.subplots(figsize=(14, 6))

configs = ["45ch_reliable", "64ch_full"]
x = np.arange(len(configs))
w = 0.35

feet_means, feet_stds = [], []
for cfg in configs:
    sub = df[df["config"] == cfg]
    feet_means.append(sub["test_balanced_accuracy"].mean())
    feet_stds.append(sub["test_balanced_accuracy"].std())

hand_means, hand_stds = [], []
for cfg in configs:
    n = df[df["config"] == cfg]["n_channels"].iloc[0]
    hsub = hand_df[hand_df["n_channels"] == n]
    if len(hsub) > 0:
        hand_means.append(hsub["test_balanced_accuracy"].mean())
        hand_stds.append(hsub["test_balanced_accuracy"].std())
    else:
        hand_means.append(np.nan)
        hand_stds.append(0)

ax.bar(x - w/2, hand_means, w, yerr=hand_stds, label="Left vs Right Hand",
       color="#1565C0", capsize=5, alpha=0.85)
ax.bar(x + w/2, feet_means, w, yerr=feet_stds, label="Fists vs Feet",
       color="#4CAF50", capsize=5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(["45 ch\n(reliable)", "64 ch\n(full)"])
ax.axhline(0.5, color="red", ls="--", alpha=0.5)
ax.set_ylabel("Cross-Subject Balanced Accuracy", fontsize=12)
ax.set_title("Task Comparison: Hand Discrimination vs Fists/Feet",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0.44, None)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/task_comparison_bars.png", dpi=200)
plt.close(fig)
print(f"Saved {FIGURES_DIR}/task_comparison_bars.png")
