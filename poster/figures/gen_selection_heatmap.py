#!/usr/bin/env python3
"""Variant selection heatmap: shows which variant was dispatched at each iteration.

Visualizes the explore phase (cycling through variants) then locking onto the
optimal variant during exploitation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re

# ---------------------------------------------------------------------------
# Parse benchmark output
# ---------------------------------------------------------------------------

DATA_FILE = '/home/akash/PROJECTS/LLVM/experiments/prototype/results/profiled_dispatch_results.csv'

iterations = []
variants = []
phases = []

with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        m = re.match(r'^(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\S+)$', line)
        if m:
            iterations.append(int(m.group(1)))
            variants.append(int(m.group(2)))
            phases.append(m.group(4))

# Show first 30 iterations
N_SHOW = 30
iterations = np.array(iterations[:N_SHOW])
variants = np.array(variants[:N_SHOW])
phases = phases[:N_SHOW]

N_VARIANTS = 3

# Build selection matrix: rows = variant slots, cols = iterations
# 1 = selected, 0 = not selected
selection = np.zeros((N_VARIANTS, N_SHOW))
for i, v in enumerate(variants):
    selection[v, i] = 1.0

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(8, 2))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Custom colormap per variant: build an RGB image
# Each cell gets the variant's color if selected, white if not
VARIANT_COLORS_RGB = {
    0: np.array([0.863, 0.149, 0.149]),   # #dc2626 red
    1: np.array([0.051, 0.580, 0.533]),   # #0d9488 teal
    2: np.array([0.910, 0.580, 0.227]),   # #e8943a orange
}
WHITE = np.array([1.0, 1.0, 1.0])
BG_LIGHT = np.array([0.973, 0.973, 0.973])  # very light gray for unselected

img = np.zeros((N_VARIANTS, N_SHOW, 3))
for v in range(N_VARIANTS):
    for i in range(N_SHOW):
        if selection[v, i] > 0:
            img[v, i] = VARIANT_COLORS_RGB[v]
        else:
            img[v, i] = BG_LIGHT

ax.imshow(img, aspect='auto', interpolation='nearest',
          extent=[-0.5, N_SHOW - 0.5, N_VARIANTS - 0.5, -0.5])

# Grid lines between cells
for v in range(N_VARIANTS + 1):
    ax.axhline(y=v - 0.5, color='white', linewidth=1.5)
for i in range(N_SHOW + 1):
    ax.axvline(x=i - 0.5, color='white', linewidth=0.5)

# Convergence line
converge_iter = None
for i, (v, p) in enumerate(zip(variants, phases)):
    if v == 1 and p == 'exploit':
        converge_iter = iterations[i]
        break

if converge_iter is not None and converge_iter < N_SHOW:
    ax.axvline(x=converge_iter - 0.5, color='#1e3a5f', linestyle='--',
               linewidth=1.5, alpha=0.8, zorder=5)
    ax.text(converge_iter - 0.5, -0.75, 'converged',
            fontsize=9, color='#1e3a5f', ha='center', va='bottom',
            fontweight='bold')

# Phase bracket annotations
ax.annotate('', xy=(-0.5, -0.7), xytext=(converge_iter - 0.7, -0.7),
            arrowprops=dict(arrowstyle='<->', color='#92400e', lw=1.2))
ax.text((converge_iter - 1) / 2, -0.9, 'explore',
        fontsize=9, color='#92400e', ha='center', va='top', fontstyle='italic')

ax.annotate('', xy=(converge_iter - 0.3, -0.7), xytext=(N_SHOW - 0.5, -0.7),
            arrowprops=dict(arrowstyle='<->', color='#0d9488', lw=1.2))
ax.text((converge_iter + N_SHOW - 1) / 2, -0.9, 'exploit',
        fontsize=9, color='#0d9488', ha='center', va='top', fontstyle='italic')

# Axes
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['V0: sm_50\n(slow)', 'V1: sm_80\n(fast)', 'V2: sm_70\n(med)'],
                   fontsize=10)

# X-axis: show every 5th tick
xticks = list(range(0, N_SHOW, 5))
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks], fontsize=10)
ax.set_xlabel('Dispatch Iteration', fontsize=12, fontweight='medium')

# Title
ax.set_title('Variant Selection Over Time',
             fontsize=16, fontweight='bold', pad=20, loc='left')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/selection-heatmap.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: selection-heatmap.svg')
