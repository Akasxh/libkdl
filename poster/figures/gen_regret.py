#!/usr/bin/env python3
"""Regret plot: Cumulative regret of profiled dispatch vs random baseline.

Shows that regret flattens after exploration completes (O(N^2) total cost),
compared to linear regret growth under random dispatch.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re

# ---------------------------------------------------------------------------
# Parse benchmark output
# ---------------------------------------------------------------------------

DATA_FILE = '/home/akash/PROJECTS/LLVM/experiments/prototype/results/profiled_dispatch_results.csv'

iterations = []
variants = []
measured_ns = []
phases = []
cumul_regret_ns = []

with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        m = re.match(r'^(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\S+)$', line)
        if m:
            iterations.append(int(m.group(1)))
            variants.append(int(m.group(2)))
            measured_ns.append(float(m.group(3)))
            phases.append(m.group(4))
            cumul_regret_ns.append(float(m.group(5)))

iterations = np.array(iterations)
cumul_regret_us = np.array(cumul_regret_ns) / 1000.0
measured_us = np.array(measured_ns) / 1000.0

# Find convergence point
converge_iter = None
for i, (v, p) in enumerate(zip(variants, phases)):
    if v == 1 and p == 'exploit':
        converge_iter = iterations[i]
        break

# Compute random-dispatch baseline regret (expected value)
# Random picks uniformly from {500us, 100us, 300us} -> expected = 300us
# Optimal = ~100us, so expected per-step regret = 200us
optimal_true_us = 100.0
variant_times_us = np.array([500.0, 100.0, 300.0])
random_expected_us = np.mean(variant_times_us)
random_regret_per_step = random_expected_us - optimal_true_us  # 200 us

random_cumul_regret = np.cumsum(np.full(len(iterations), random_regret_per_step))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Random baseline
ax.plot(iterations, random_cumul_regret, color='#9ca3af', linewidth=2.0,
        linestyle='--', label='Random dispatch (E[regret]=200 us/iter)', zorder=3)

# Profiled dispatch regret
ax.plot(iterations, cumul_regret_us, color='#0d9488', linewidth=2.5,
        label='Profiled dispatch (explore/exploit)', zorder=4)

# Fill the area between to highlight savings
ax.fill_between(iterations, cumul_regret_us, random_cumul_regret,
                alpha=0.08, color='#0d9488', zorder=2)

# Convergence marker
if converge_iter is not None:
    ax.axvline(x=converge_iter, color='#1e3a5f', linestyle=':', linewidth=1.2,
               alpha=0.6, zorder=5)

# Exploration cost annotation
if converge_iter is not None:
    explore_regret = cumul_regret_us[converge_iter]
    final_regret = cumul_regret_us[-1]

    ax.axhline(y=final_regret, color='#dc2626', linestyle='--', linewidth=1.0,
               alpha=0.5, zorder=3)
    ax.text(iterations[-1] + 1, final_regret,
            f'Total cost: {final_regret:.0f} us',
            fontsize=11, color='#dc2626', va='center', ha='left', fontweight='bold')

# Annotation: regret flattens
ax.annotate('After 9 dispatches, regret = 0\n(always picks optimal)',
            xy=(25, cumul_regret_us[25]),
            xytext=(40, cumul_regret_us[25] + 4000),
            fontsize=11, fontweight='bold', color='#0d9488',
            arrowprops=dict(arrowstyle='->', color='#0d9488', lw=1.5),
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdfa',
                      edgecolor='#0d9488', linewidth=1.0))

# Savings callout
savings_us = random_cumul_regret[-1] - cumul_regret_us[-1]
ax.text(0.98, 0.55,
        f'Savings: {savings_us:.0f} us\n({savings_us/random_cumul_regret[-1]*100:.0f}% less regret)',
        transform=ax.transAxes, fontsize=13, fontweight='bold', color='#1e3a5f',
        ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#1e3a5f', linewidth=1.0, alpha=0.9))

# Axes
ax.set_xlabel('Dispatch Iteration', fontsize=14, fontweight='medium')
ax.set_ylabel('Cumulative Regret (\u00b5s)', fontsize=14, fontweight='medium')
ax.set_xlim(-1, 100)
ax.set_ylim(0, max(random_cumul_regret[-1], cumul_regret_us[-1]) * 1.15)

# Title
ax.set_title('Cumulative Regret: Profiled vs. Random Dispatch',
             fontsize=18, fontweight='bold', pad=14, loc='left')

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

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/regret.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: regret.svg')
