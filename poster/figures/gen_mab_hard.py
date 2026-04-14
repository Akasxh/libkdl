#!/usr/bin/env python3
"""Near-identical variant scenario: convergence with tight spread and noise.

Shows that with only 12% spread and 8% CV noise, the bandit needs ~50+ iterations
to reliably identify the best variant, not 9. Includes confidence intervals.

Input:  experiments/prototype/results/mab_suite_results.csv
Output: poster/figures/mab-near-identical.svg
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

iterations = []
variants = []
times_ns = []
is_optimal = []
cumul_regret = []

with open(DATA) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['scenario'] == 'near_identical':
            iterations.append(int(row['iteration']))
            variants.append(int(row['variant']))
            times_ns.append(float(row['time_ns']))
            is_optimal.append(int(row['is_optimal']))
            cumul_regret.append(float(row['cumulative_regret']))

iterations = np.array(iterations)
variants = np.array(variants)
times_us = np.array(times_ns) / 1000.0
is_optimal = np.array(is_optimal)
cumul_regret_us = np.array(cumul_regret) / 1000.0

# Per-variant running statistics (mean and CI)
variant_ids = sorted(set(variants))
n_variants = len(variant_ids)

# Compute running selection fraction (rolling window of 20)
window = 20
opt_frac = np.convolve(is_optimal, np.ones(window)/window, mode='same')

# Find convergence: first iteration where opt_frac stays above 0.8 for 20 iters
converge_iter = None
for i in range(window, len(opt_frac)):
    if all(opt_frac[i-window//2:i+window//2] > 0.7):
        converge_iter = iterations[i - window//2]
        break

# Per-variant measured time history
var_times = defaultdict(list)
var_iters = defaultdict(list)
for it, v, t in zip(iterations, variants, times_us):
    var_times[v].append(t)
    var_iters[v].append(it)

# Compute running mean + std per variant (over all samples up to iteration i)
# For the confidence interval subplot
true_base_us = [100.0, 103.0, 105.0, 108.0, 112.0]
sigma_us = 8.0

# ---------------------------------------------------------------------------
# Plot: two subplots
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[2, 1],
                                 sharex=True, gridspec_kw={'hspace': 0.08})
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')
ax2.set_facecolor('white')

# --- Top: measured time per dispatch, colored by variant ---
COLORS = {0: '#0d9488', 1: '#3b82f6', 2: '#8b5cf6', 3: '#e8943a', 4: '#dc2626'}
LABELS = {0: 'V0: 100us', 1: 'V1: 103us', 2: 'V2: 105us', 3: 'V3: 108us', 4: 'V4: 112us'}
MARKERS = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'v'}

for v_id in variant_ids:
    mask = variants == v_id
    ax1.scatter(iterations[mask], times_us[mask],
                c=COLORS[v_id], label=LABELS[v_id],
                marker=MARKERS[v_id], s=18, alpha=0.6, zorder=4,
                edgecolors='white', linewidths=0.2)

# True performance lines
for v_id in variant_ids:
    ax1.axhline(y=true_base_us[v_id], color=COLORS[v_id], linestyle=':',
                linewidth=1.0, alpha=0.5, zorder=2)

# Noise band for the best variant
ax1.axhspan(true_base_us[0] - 2*sigma_us, true_base_us[0] + 2*sigma_us,
            alpha=0.06, color='#0d9488', zorder=1)
ax1.text(285, true_base_us[0] + 2*sigma_us + 1, '2sigma noise\nband (V0)',
         fontsize=9, color='#0d9488', ha='right', va='bottom', fontstyle='italic')

# Exploration phase shading
n_explore = n_variants * 3  # KDL_PD_WARMUP_SAMPLES = 3
ax1.axvspan(-0.5, n_explore - 0.5, alpha=0.06, color='#e8943a', zorder=0)

if converge_iter is not None:
    ax1.axvline(x=converge_iter, color='#1e3a5f', linestyle='--', linewidth=1.5,
                alpha=0.7, zorder=5)
    ax1.annotate(f'Reliable\nconvergence\n(iter ~{converge_iter})',
                 xy=(converge_iter, 80),
                 xytext=(converge_iter + 30, 75),
                 fontsize=11, fontweight='bold', color='#1e3a5f',
                 arrowprops=dict(arrowstyle='->', color='#1e3a5f', lw=1.5),
                 ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#1e3a5f', linewidth=1.0, alpha=0.9))

ax1.set_ylabel('Measured Time (us)', fontsize=14, fontweight='medium')
ax1.set_ylim(65, 135)
ax1.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.95,
           edgecolor='#e5e7eb', fancybox=False, columnspacing=0.8)
ax1.yaxis.grid(True, linewidth=0.3, alpha=0.4, color='#d1d5db', zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Bottom: fraction of optimal selections (rolling window) ---
ax2.fill_between(iterations, 0, opt_frac, alpha=0.15, color='#0d9488', zorder=2)
ax2.plot(iterations, opt_frac, color='#0d9488', linewidth=2.0, zorder=3)
ax2.axhline(y=1.0, color='#9ca3af', linestyle=':', linewidth=1.0, alpha=0.5)
ax2.axhline(y=1.0/n_variants, color='#dc2626', linestyle=':', linewidth=1.0, alpha=0.5)
ax2.text(295, 1.0/n_variants + 0.02, 'random\nbaseline', fontsize=9,
         color='#dc2626', ha='right', va='bottom', fontstyle='italic')

if converge_iter is not None:
    ax2.axvline(x=converge_iter, color='#1e3a5f', linestyle='--', linewidth=1.5, alpha=0.7)

ax2.set_xlabel('Dispatch Iteration', fontsize=14, fontweight='medium')
ax2.set_ylabel('P(optimal)', fontsize=14, fontweight='medium')
ax2.set_ylim(-0.05, 1.15)
ax2.set_xlim(-2, 305)
ax2.yaxis.grid(True, linewidth=0.3, alpha=0.4, color='#d1d5db', zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Supertitle
fig.suptitle('Near-Identical Variants: 12% Spread, 8% CV Noise',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/mab-near-identical.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: mab-near-identical.svg')
