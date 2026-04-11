#!/usr/bin/env python3
"""Chart 1: Cold-Path Dispatch Latency Decomposition — horizontal bar chart."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data (sorted longest at top)
operations = [
    'cuModuleLoadData',
    'cuStreamSynchronize',
    'cuLaunchKernel',
    'cuModuleGetFunction',
    'Selection overhead',
]
latencies = [36.0, 2.45, 1.65, 0.063, 0.005]
colors = ['#dc2626', '#2563eb', '#7c3aed', '#6b7280', '#0d9488']

# Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Horizontal bars
y_pos = np.arange(len(operations))
bars = ax.barh(y_pos, latencies, color=colors, height=0.55, edgecolor='white', linewidth=0.5, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(operations, fontsize=11, fontweight='medium')
ax.invert_yaxis()

# Log scale x-axis
ax.set_xscale('log')
ax.set_xlim(0.001, 100)
ax.set_xlabel('Latency (µs)', fontsize=11, fontweight='medium')

# Gridlines on x-axis only
ax.xaxis.grid(True, which='both', linewidth=0.4, alpha=0.5, color='#d1d5db', zorder=0)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

# Custom x-tick formatting
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))

# Bar value labels
for i, (bar, val) in enumerate(zip(bars, latencies)):
    if val >= 1:
        label = f'{val:.1f} µs'
    elif val >= 0.01:
        label = f'{val:.3f} µs'
    else:
        label = f'{val:.3f} µs'
    ax.text(val * 1.3, i, label, va='center', ha='left', fontsize=9.5, fontweight='medium', color='#374151')

# Annotation: 90% of cold path on cuModuleLoadData
ax.annotate(
    '90% of cold path',
    xy=(36.0, 0), xytext=(4.0, -0.65),
    fontsize=9, fontweight='bold', color='#dc2626',
    arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
    va='center',
)

# Annotation: Selection overhead
ax.annotate(
    '3–6 ns — faster than L2 cache access',
    xy=(0.005, 4), xytext=(0.08, 4.0),
    fontsize=8.5, fontstyle='italic', color='#0d9488',
    arrowprops=dict(arrowstyle='->', color='#0d9488', lw=1.2),
    va='center',
)

# Title
ax.set_title('Cold-Path Dispatch Latency Decomposition', fontsize=13, fontweight='bold', pad=18, loc='left')
fig.text(0.125, 0.93, 'GTX 1650 (sm_75), CUDA 13.1, null kernel CUBIN', fontsize=9, color='#6b7280', style='italic')

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/latency-breakdown.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: latency-breakdown.svg')
