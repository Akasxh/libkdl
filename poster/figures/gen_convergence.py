#!/usr/bin/env python3
"""Convergence plot: Profiled Dispatch multi-armed bandit exploration/exploitation.

Parses the bench_profiled CSV output and plots measured execution time per
dispatch iteration, with exploration/exploitation phase shading and
convergence annotation.
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

with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        # Match data rows: "0       0         604657          cold        504657  sm_50_slow"
        m = re.match(r'^(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\S+)$', line)
        if m:
            iterations.append(int(m.group(1)))
            variants.append(int(m.group(2)))
            measured_ns.append(float(m.group(3)))
            phases.append(m.group(4))

iterations = np.array(iterations)
variants = np.array(variants)
measured_us = np.array(measured_ns) / 1000.0  # convert to microseconds

# Find convergence point
converge_iter = None
for i, (v, p) in enumerate(zip(variants, phases)):
    if v == 1 and p == 'exploit':
        converge_iter = iterations[i]
        break

# ---------------------------------------------------------------------------
# Variant info
# ---------------------------------------------------------------------------

VARIANT_COLORS = {0: '#dc2626', 1: '#0d9488', 2: '#e8943a'}
VARIANT_LABELS = {0: 'Variant 0: sm_50 (slow)', 1: 'Variant 1: sm_80 (fast)', 2: 'Variant 2: sm_70 (medium)'}
VARIANT_MARKERS = {0: 's', 1: 'o', 2: '^'}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Shaded regions
if converge_iter is not None:
    ax.axvspan(-0.5, converge_iter - 0.5, alpha=0.08, color='#e8943a', zorder=0)
    ax.axvspan(converge_iter - 0.5, iterations[-1] + 0.5, alpha=0.06, color='#0d9488', zorder=0)

    # Phase labels (placed after data plotting below)

# Plot each variant as separate scatter series
for v_id in sorted(set(variants)):
    mask = variants == v_id
    ax.scatter(iterations[mask], measured_us[mask],
               c=VARIANT_COLORS[v_id], label=VARIANT_LABELS[v_id],
               marker=VARIANT_MARKERS[v_id], s=30, alpha=0.85, zorder=4,
               edgecolors='white', linewidths=0.3)

# Connect exploitation-phase points with a line for visual continuity
exploit_mask = np.array([p == 'exploit' for p in phases])
if exploit_mask.any():
    ax.plot(iterations[exploit_mask], measured_us[exploit_mask],
            color='#0d9488', linewidth=1.0, alpha=0.4, zorder=3)

# Convergence line
if converge_iter is not None:
    ax.axvline(x=converge_iter, color='#1e3a5f', linestyle='--', linewidth=1.5,
               alpha=0.7, zorder=5)
    ax.annotate(f'Convergence\n(iter {converge_iter})',
                xy=(converge_iter, measured_us[converge_iter]),
                xytext=(converge_iter + 12, 450),
                fontsize=12, fontweight='bold', color='#1e3a5f',
                arrowprops=dict(arrowstyle='->', color='#1e3a5f', lw=1.5),
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#1e3a5f', linewidth=1.0, alpha=0.9))

# Exploitation label
ax.text(55, 50, 'Exploitation\n(optimal only)',
        fontsize=12, ha='center', va='bottom', color='#0d9488',
        fontstyle='italic', alpha=0.8)

# Axes
ax.set_xlabel('Dispatch Iteration', fontsize=14, fontweight='medium')
ax.set_ylabel('Measured Execution Time (\u00b5s)', fontsize=14, fontweight='medium')
ax.set_xlim(-1, 100)
ax.set_ylim(0, max(measured_us) * 1.15)

# Exploration phase label (placed after y-limits are set)
if converge_iter is not None:
    ax.text(converge_iter / 2, max(measured_us) * 1.10,
            'Exploration', fontsize=12, ha='center', va='top', color='#92400e',
            fontstyle='italic', alpha=0.8)

# Title
ax.set_title('Profiled Dispatch: Convergence to Optimal Variant',
             fontsize=18, fontweight='bold', pad=14, loc='left')
ax.text(0.0, 1.02, '3 variants, 3 warmup samples each \u2192 converges at iteration 9',
        transform=ax.transAxes, fontsize=12, color='#6b7280', va='bottom')

# Grid
ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color='#d1d5db', zorder=0)
ax.xaxis.grid(True, linewidth=0.2, alpha=0.3, color='#d1d5db', zorder=0)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
          edgecolor='#e5e7eb', fancybox=False)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/convergence.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: convergence.svg')
