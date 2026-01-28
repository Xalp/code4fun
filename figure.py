import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data setup
temps = np.arange(0, 1.1, 0.1)

# LLaDA-1.5 data
llada_gsm8k = [0.80363, 0.80591, 0.8006, 0.79984, 0.79075, 0.79605, 0.80439, 0.80288, 0.79757, 0.79909, 0.79757]
llada_gsm8k_smc = [0.79833, 0.80363, 0.80742, 0.80212, 0.80136, 0.79757, 0.8188, 0.81956, 0.81576, 0.80818, 0.80742]
llada_math = [0.398, 0.398, 0.38, 0.376, 0.4, 0.384, 0.388, 0.372, 0.388, 0.382, 0.382]
llada_math_smc = [0.392, 0.392, 0.41, 0.42, 0.4, 0.394, 0.392, 0.414, 0.396, 0.406, 0.418]

# Dream-7B data
dream_gsm8k = [0.79075, 0.09552, 0.26838, 0.49128, 0.63381, 0.71266, 0.75435, 0.76269, 0.76648, 0.77028, 0.76194]
dream_gsm8k_smc = [0.79529, 0.72782, 0.75815, 0.76952, 0.77028, 0.79681, 0.8006, 0.77482, 0.78771, 0.79378, 0.78013]
dream_math = [0.424, 0.024, 0.104, 0.222, 0.292, 0.38, 0.398, 0.38, 0.41, 0.424, 0.416]
dream_math_smc = [0.468, 0.42, 0.446, 0.424, 0.442, 0.45, 0.43, 0.438, 0.446, 0.462, 0.452]

# Convert to percentages
def to_pct(arr): return [x * 100 for x in arr]

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
color_base = "#377eb8" # Blue
color_smc = "#e41a1c"  # Red

def plot_data(ax, x, y_base, y_smc, title, ylabel, show_x=False):
    ax.scatter(x, to_pct(y_base), color=color_base, label="Baseline", marker="o", s=70, alpha=0.8, edgecolors='w', linewidth=0.5)
    ax.scatter(x, to_pct(y_smc), color=color_smc, label="w/ SR-SMC", marker="D", s=60, alpha=0.8, edgecolors='w', linewidth=0.5)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel)
    if show_x:
        ax.set_xlabel("Temperature")
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.grid(True, linestyle="--", alpha=0.6)
    # Highlight specific range for better contrast
    y_min = min(min(to_pct(y_base)), min(to_pct(y_smc))) - 2
    y_max = max(max(to_pct(y_base)), max(to_pct(y_smc))) + 2
    ax.set_ylim(max(0, y_min), min(100, y_max))

# Top Left: LLaDA GSM8K
plot_data(axes[0, 0], temps, llada_gsm8k, llada_gsm8k_smc, "LLaDA-1.5: GSM8K", "Accuracy (%)")
# Top Right: LLaDA MATH
plot_data(axes[0, 1], temps, llada_math, llada_math_smc, "LLaDA-1.5: MATH", "Accuracy (%)")
# Bottom Left: Dream GSM8K
plot_data(axes[1, 0], temps, dream_gsm8k, dream_gsm8k_smc, "Dream-7B: GSM8K", "Accuracy (%)", show_x=True)
# Bottom Right: Dream MATH
plot_data(axes[1, 1], temps, dream_math, dream_math_smc, "Dream-7B: MATH", "Accuracy (%)", show_x=True)

# Shared Legend at the top
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("temp_ablation_plots.pdf", dpi=300)
plt.show()