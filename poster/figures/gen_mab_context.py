#!/usr/bin/env python3
"""Context-dependent selection: different shapes have different optimal variants.

3 subplots (one per shape: small, medium, large) showing convergence to
different optimal variants per shape. Demonstrates the contextual bandit.

Input:  experiments/prototype/results/mab_suite_results.csv
Output: poster/figures/mab-context.svg
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

DATA = '/home/akash/PROJECTS/LLVM/experiments/prototype/results/mab_suite_results.csv'

shapes = ['small', 'medium', 'large']
shape_data = {s: {'iters': [], 'variants': [], 'times': [], 'is_opt': []} for s in shapes}

with open(DATA) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sc = row['scenario']
        for s in shapes:
            if sc == f'context_{s}':
                shape_data[s]['iters'].append(int(row['iteration']))
                shape_data[s]['variants'].append(int(row['variant']))
                shape_data[s]['times'].append(float(row['time_ns']) / 1000.0)
                shape_data[s]['is_opt'].append(int(row['is_optimal']))

# Variant info
VARIANT_NAMES = ['A (big tile)', 'B (balanced)', 'C (small tile)', 'D (cache-opt)']
COLORS = {0: '#dc2626', 1: '#3b82f6', 2: '#0d9488', 3: '#e8943a'}
MARKERS = {0: 'o', 1: 's', 2: '^', 3: 'D'}

# Ground truth per shape
OPTIMAL = {'small': 2, 'medium': 1, 'large': 0}
SHAPE_LABELS = {
    'small': 'Small (N=256)\nOptimal: C (small tile)',
    'medium': 'Medium (N=2048)\nOptimal: B (balanced)',
    'large': 'Large (N=8192)\nOptimal: A (big tile)',
}

# ---------------------------------------------------------------------------
# Plot: 3 subplots in a row
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

fig, axes = plt.subplots(1, 3, figsize=(9, 5), sharey=True)
fig.patch.set_facecolor('white')

for ax_idx, (shape, ax) in enumerate(zip(shapes, axes)):
    ax.set_facecolor('white')
    d = shape_data[shape]
    iters = np.array(d['iters'])
    varis = np.array(d['variants'])
    times = np.array(d['times'])
    opt_v = OPTIMAL[shape]

    # Exploration shading (warmup = 4 variants * 3 samples = 12 iters)
    n_explore = 4 * 3
    ax.axvspan(-0.5, n_explore - 0.5, alpha=0.06, color='#e8943a', zorder=0)

    # Plot each variant
    for v_id in sorted(set(varis)):
        mask = varis == v_id
        ax.scatter(iters[mask], times[mask],
                   c=COLORS[v_id], marker=MARKERS[v_id],
                   s=15, alpha=0.5, zorder=4,
                   edgecolors='white', linewidths=0.2,
                   label=VARIANT_NAMES[v_id] if ax_idx == 0 else None)

    # Highlight the optimal variant's trajectory
    opt_mask = varis == opt_v
    if opt_mask.any():
        ax.plot(iters[opt_mask], times[opt_mask],
                color=COLORS[opt_v], linewidth=1.0, alpha=0.3, zorder=3)

    # Convergence line
    # Find where we start exploiting the optimal
    converge = None
    for i in range(len(iters)):
        if varis[i] == opt_v and i >= n_explore:
            converge = iters[i]
            break
    if converge is not None:
        ax.axvline(x=converge, color='#1e3a5f', linestyle='--',
                   linewidth=1.2, alpha=0.5, zorder=5)

    # Shape label
    ax.set_title(SHAPE_LABELS[shape], fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Iteration', fontsize=12)
    if ax_idx == 0:
        ax.set_ylabel('Measured Time (us)', fontsize=12, fontweight='medium')

    ax.set_xlim(-2, 205)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color='#d1d5db', zorder=0)
    ax.xaxis.grid(True, linewidth=0.2, alpha=0.3, color='#d1d5db', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
           framealpha=0.95, edgecolor='#e5e7eb', fancybox=False,
           bbox_to_anchor=(0.5, 1.0))

fig.suptitle('Context-Dependent Dispatch: Rankings Change Per Shape',
             fontsize=15, fontweight='bold', y=1.08)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/mab-context.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: mab-context.svg')
