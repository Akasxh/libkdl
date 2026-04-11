#!/usr/bin/env python3
"""Chart 4: Related Work Comparison — dot matrix table."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
systems = ['IREE HAL', 'chipStar', 'Proteus', 'HetGPU', 'liboffload', 'CPU FMV', 'This Work']
features = ['Runtime\nSelect', 'Cross-\nVendor', 'MLIR-\nNative', 'Ranked', 'Light-\nweight']
loc_values = ['100K+', '50K+', '10K+', 'N/A', '5K+', 'N/A', '~800']

# Matrix: 1 = Yes (green filled), 0.5 = Partial (yellow half), 0 = No (gray empty)
matrix = [
    [1,   1,   1,   0,   0  ],  # IREE HAL
    [0.5, 1,   0,   0,   0  ],  # chipStar
    [1,   0,   0,   0,   0.5],  # Proteus
    [0.5, 0,   0,   0,   0  ],  # HetGPU
    [0.5, 1,   0,   0,   0  ],  # liboffload
    [1,   0,   0,   0,   1  ],  # CPU FMV
    [1,   1,   1,   1,   1  ],  # This Work
]

# Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(8, 3))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

n_rows = len(systems)
n_feat = len(features)

# Highlight "This Work" row
highlight_row = n_rows - 1
ax.add_patch(mpatches.FancyBboxPatch(
    (-0.5, highlight_row - 0.45), n_feat + 1.8, 0.9,
    boxstyle='round,pad=0.05', facecolor='#f0fdfa', edgecolor='#0d9488',
    linewidth=1.2, zorder=0))

# Draw circles
radius = 0.18
for i, row in enumerate(matrix):
    for j, val in enumerate(row):
        cx, cy = j, i
        if val == 1:  # Yes — filled green
            circle = plt.Circle((cx, cy), radius, facecolor='#16a34a', edgecolor='#15803d',
                                linewidth=1, zorder=3)
            ax.add_patch(circle)
        elif val == 0.5:  # Partial — half yellow
            # Draw empty circle
            circle_bg = plt.Circle((cx, cy), radius, facecolor='white', edgecolor='#d97706',
                                   linewidth=1, zorder=2)
            ax.add_patch(circle_bg)
            # Draw left half filled
            half = mpatches.Wedge((cx, cy), radius, 90, 270, facecolor='#f59e0b',
                                  edgecolor='#d97706', linewidth=1, zorder=3)
            ax.add_patch(half)
        else:  # No — empty gray
            circle = plt.Circle((cx, cy), radius, facecolor='white', edgecolor='#9ca3af',
                                linewidth=1, zorder=3)
            ax.add_patch(circle)

# LOC column
loc_x = n_feat + 0.7
for i, loc in enumerate(loc_values):
    weight = 'bold' if i == highlight_row else 'normal'
    color = '#0d9488' if i == highlight_row else '#374151'
    ax.text(loc_x, i, loc, ha='center', va='center', fontsize=9.5, fontweight=weight, color=color)

# Axis labels
ax.set_xlim(-0.5, n_feat + 1.4)
ax.set_ylim(n_rows - 0.6, -0.6)

ax.set_xticks(list(range(n_feat)) + [loc_x])
ax.set_xticklabels(features + ['LOC'], fontsize=9.5, fontweight='medium', ha='center')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

ax.set_yticks(range(n_rows))
ylabels = []
for i, s in enumerate(systems):
    ylabels.append(s)
ax.set_yticklabels(ylabels, fontsize=10.5)
# Bold "This Work"
for label in ax.get_yticklabels():
    if label.get_text() == 'This Work':
        label.set_fontweight('bold')
        label.set_color('#0d9488')

# Remove all spines and ticks
ax.tick_params(axis='both', which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# Add thin horizontal separators
for i in range(1, n_rows):
    ax.axhline(y=i - 0.5, color='#e5e7eb', linewidth=0.5, zorder=1)

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#16a34a',
               markeredgecolor='#15803d', markersize=10, label='Yes'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b',
               markeredgecolor='#d97706', markersize=10, label='Partial'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#9ca3af', markersize=10, label='No'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5, ncol=3,
          frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

# Title
ax.set_title('Related Work Comparison', fontsize=13, fontweight='bold', pad=20, loc='left')

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/comparison-dots.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: comparison-dots.svg')
