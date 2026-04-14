#!/usr/bin/env python3
"""Scaling plot: dispatches-to-converge vs number of variants.

X-axis: N (2-64 variants)
Y-axis: dispatches to converge
Overlays the theoretical O(N * warmup) bound.

Input:  experiments/prototype/results/mab_suite_results.csv
Output: poster/figures/mab-scaling.svg
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

# Collect per-N convergence data
# Convergence = first iteration where is_optimal=1 AND past the warmup phase
scaling_data = defaultdict(lambda: {'iters': [], 'is_opt': [], 'regret': []})

with open(DATA) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sc = row['scenario']
        if sc.startswith('scaling_'):
            N = int(sc.split('_')[1])
            scaling_data[N]['iters'].append(int(row['iteration']))
            scaling_data[N]['is_opt'].append(int(row['is_optimal']))
            scaling_data[N]['regret'].append(float(row['cumulative_regret']))

# Determine convergence point for each N
# Convergence = first iteration after which is_optimal stays 1 for >=10 consecutive
Ns = sorted(scaling_data.keys())
converge_iters = []
final_regrets = []

for N in Ns:
    data = scaling_data[N]
    iters = data['is_opt']
    conv = len(iters)  # default: never converged
    streak = 0
    for i, opt in enumerate(iters):
        if opt == 1:
            streak += 1
            if streak >= 10:
                conv = i - streak + 1
                break
        else:
            streak = 0
    converge_iters.append(conv)
    final_regrets.append(data['regret'][-1] / 1000.0 if data['regret'] else 0)

Ns = np.array(Ns)
converge_iters = np.array(converge_iters)

# Theoretical bound: O(N * warmup_samples)
warmup = 3
theoretical = Ns * warmup * 1.2  # with overhead factor

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Empirical
ax.plot(Ns, converge_iters, color='#0d9488', marker='o', markersize=10,
        linewidth=2.5, markeredgecolor='white', markeredgewidth=2, zorder=5,
        label='Empirical (profiled dispatch)')

# Annotate each point
for n, c in zip(Ns, converge_iters):
    ax.annotate(f'{c}', xy=(n, c), xytext=(0, 12),
                textcoords='offset points', fontsize=11, fontweight='bold',
                color='#0d9488', ha='center')

# Theoretical bound
ax.plot(Ns, theoretical, color='#9ca3af', linewidth=2.0, linestyle='--',
        zorder=3, label=f'Theoretical: O(N x {warmup} warmup)')

# Fill between
ax.fill_between(Ns, converge_iters, theoretical,
                where=theoretical >= converge_iters,
                alpha=0.08, color='#0d9488', zorder=2)

# Reference: pure round-robin explore cost
rr_cost = Ns * warmup
ax.plot(Ns, rr_cost, color='#e8943a', linewidth=1.5, linestyle=':',
        zorder=3, label=f'Min explore cost: N x {warmup}')

# Axes
ax.set_xlabel('Number of Variants (N)', fontsize=14, fontweight='medium')
ax.set_ylabel('Dispatches to Converge', fontsize=14, fontweight='medium')
ax.set_xscale('log', base=2)
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.set_ylim(0, max(converge_iters) * 1.3)

# Title
ax.set_title('Convergence Scaling: Dispatches vs. Variant Count',
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

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/mab-scaling.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: mab-scaling.svg')
