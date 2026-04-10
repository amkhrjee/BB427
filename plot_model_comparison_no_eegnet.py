"""
Recreate new_models_comparison.png without the EEGNet baseline line.
Values taken from the original figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "figures"

# Data from the original plot
models = ["EEGSym", "EEGNeX", "CTNet", "MSVTNet", "EEGInceptionMI", "EEGSimpleConv"]
means =  [0.656,     0.618,    0.598,   0.597,     0.595,            0.579]
# Estimate std from error bars in original plot
stds =   [0.026,     0.035,    0.030,   0.028,     0.032,            0.025]

colors = ["#C040C0", "#4040D0", "#D04040", "#D07020", "#C0C020", "#20A020"]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors,
              edgecolor="black", linewidth=0.5, alpha=0.85)

# Add value labels
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Chance line only
ax.axhline(0.5, color="gray", ls="--", lw=1.5, alpha=0.7, label="Chance (50%)")

ax.set_ylabel("Balanced accuracy", fontsize=12)
ax.set_title("New model comparison — all 64 channels, 5 splits",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0.45, 0.78)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/new_models_comparison_no_eegnet.png", dpi=200)
plt.close(fig)
print(f"Saved {FIGURES_DIR}/new_models_comparison_no_eegnet.png")
