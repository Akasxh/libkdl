#!/usr/bin/env python3
"""Comparison plot: cumulative regret curves for 4 strategies.

Random, roofline-only, profiled dispatch, and oracle.
This is the MONEY figure showing profiled dispatch is near-oracle.

Input:  experiments/prototype/results/mab_suite_results.csv
Output: poster/figures/mab-comparison.svg
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

strategies = {
    'cmp_random': {'label': 'Random dispatch', 'color': '#dc2626', 'ls': '--', 'lw': 2.0},
    'cmp_roofline': {'label': 'Roofline-only (static)', 'color': '#e8943a', 'ls': '-.', 'lw': 2.0},
    'cmp_profiled': {'label': 'Profiled dispatch (ours)', 'color': '#0d9488', 'ls': '-', 'lw': 3.0},
    'cmp_oracle': {'label': 'Oracle (ceiling)', 'color': '#9ca3af', 'ls': ':', 'lw': 2.0},
}

data = defaultdict(lambda: {'iters': [], 'regret': []})

with open(DATA) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sc = row['scenario']
        if sc in strategies:
            data[sc]['iters'].append(int(row['iteration']))
            data[sc]['regret'].append(float(row['cumulative_regret']))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot order: random, roofline, profiled, oracle (layered bottom to top)
plot_order = ['cmp_random', 'cmp_roofline', 'cmp_profiled', 'cmp_oracle']

for key in plot_order:
    if key not in data:
        continue
    d = data[key]
    s = strategies[key]
    iters = np.array(d['iters'])
    regret_us = np.array(d['regret']) / 1000.0

    zorder = 3 if key != 'cmp_profiled' else 5
    ax.plot(iters, regret_us, color=s['color'], linewidth=s['lw'],
            linestyle=s['ls'], label=s['label'], zorder=zorder)

# Fill between profiled and random to show savings
if 'cmp_random' in data and 'cmp_profiled' in data:
    r_iters = np.array(data['cmp_random']['iters'])
    r_regret = np.array(data['cmp_random']['regret']) / 1000.0
    p_iters = np.array(data['cmp_profiled']['iters'])
    p_regret = np.array(data['cmp_profiled']['regret']) / 1000.0

    n = min(len(r_regret), len(p_regret))
    ax.fill_between(r_iters[:n], p_regret[:n], r_regret[:n],
                    alpha=0.08, color='#0d9488', zorder=2)

    # Savings annotation
    savings_us = r_regret[-1] - p_regret[-1]
    pct = savings_us / r_regret[-1] * 100.0 if r_regret[-1] > 0 else 0
    ax.annotate(f'{pct:.0f}% less regret\nvs. random',
                xy=(350, (r_regret[350] + p_regret[350]) / 2),
                xytext=(380, r_regret[350] * 0.55),
                fontsize=12, fontweight='bold', color='#0d9488',
                arrowprops=dict(arrowstyle='->', color='#0d9488', lw=1.5),
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdfa',
                          edgecolor='#0d9488', linewidth=1.0, alpha=0.9))

# Mark exploration phase end for profiled dispatch
if 'cmp_profiled' in data:
    p_data = data['cmp_profiled']
    # Exploration ends after N_VARIANTS * WARMUP = 5 * 3 = 15
    explore_end = 15
    ax.axvline(x=explore_end, color='#1e3a5f', linestyle=':', linewidth=1.2,
               alpha=0.5, zorder=4)
    ax.text(explore_end + 5, ax.get_ylim()[1] * 0.1,
            'Explore\nends', fontsize=10, color='#1e3a5f', va='bottom',
            fontstyle='italic')

# Axes
ax.set_xlabel('Dispatch Iteration', fontsize=14, fontweight='medium')
ax.set_ylabel('Cumulative Regret (us)', fontsize=14, fontweight='medium')
ax.set_xlim(-5, 505)

# Title
ax.set_title('Cumulative Regret: Four Dispatch Strategies',
             fontsize=16, fontweight='bold', pad=14, loc='left')

# Grid
ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color='#d1d5db', zorder=0)
ax.xaxis.grid(True, linewidth=0.2, alpha=0.3, color='#d1d5db', zorder=0)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
          edgecolor='#e5e7eb', fancybox=False)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/mab-comparison.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: mab-comparison.svg')
