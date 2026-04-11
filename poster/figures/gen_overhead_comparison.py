#!/usr/bin/env python3
"""Chart 3: Selection Overhead vs Real ML Kernels — grouped horizontal bar chart."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data (durations in nanoseconds for uniform scale)
kernels = ['Large GEMM', 'Medium Attention', 'Small Norm', 'Micro-kernel']
durations_ns = [100e6, 10e6, 1e6, 100e3]  # 100ms, 10ms, 1ms, 100µs
selection_ns = 5  # 5 ns for all

# Colors: graduating dark to light blue
kernel_colors = ['#1e3a5f', '#2563eb', '#60a5fa', '#93c5fd']

# Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

y_pos = np.arange(len(kernels))

# Kernel duration bars
bars = ax.barh(y_pos, durations_ns, color=kernel_colors, height=0.5, edgecolor='white',
               linewidth=0.5, zorder=3)

# Selection overhead marker: tiny teal vertical line at left edge
for i in range(len(kernels)):
    ax.plot([selection_ns, selection_ns], [i - 0.25, i + 0.25],
            color='#0d9488', linewidth=2.5, zorder=5, solid_capstyle='round')

ax.set_yticks(y_pos)
ax.set_yticklabels(kernels, fontsize=11, fontweight='medium')
ax.invert_yaxis()

# Log scale
ax.set_xscale('log')
ax.set_xlim(1, 1e9)

# Human-readable x-axis labels
def format_ns(x: float, _: int) -> str:
    if x >= 1e9:
        return f'{x/1e9:g} s'
    elif x >= 1e6:
        return f'{x/1e6:g} ms'
    elif x >= 1e3:
        return f'{x/1e3:g} µs'
    else:
        return f'{x:g} ns'

ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ns))
ax.set_xlabel('Duration (log scale)', fontsize=11, fontweight='medium')

# Bar value labels
duration_labels = ['100 ms', '10 ms', '1 ms', '100 µs']
overhead_pct = ['0.000005%', '0.00005%', '0.0005%', '0.005%']
for i, (dur_ns, label, pct) in enumerate(zip(durations_ns, duration_labels, overhead_pct)):
    ax.text(dur_ns * 1.4, i, f'{label}  ({pct} overhead)', va='center', ha='left',
            fontsize=8.5, color='#374151')

# Gridlines
ax.xaxis.grid(True, which='both', linewidth=0.3, alpha=0.4, color='#d1d5db', zorder=0)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

# Callout box
ax.text(0.98, 0.02,
        'Selection (all four cases):\n3–6 ns ≈ single L2 cache access',
        transform=ax.transAxes, fontsize=9, fontweight='bold', color='#0d9488',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0fdfa', edgecolor='#0d9488', linewidth=0.8))

# Legend for the teal marker
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='#0d9488', linewidth=2.5, label='Selection overhead (5 ns)')]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# Title
ax.set_title('Selection Overhead vs. Real ML Kernels', fontsize=13, fontweight='bold', pad=12, loc='left')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/overhead-comparison.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: overhead-comparison.svg')
