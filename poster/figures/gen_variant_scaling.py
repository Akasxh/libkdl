#!/usr/bin/env python3
"""Chart 2: Selection Time vs Number of Variants — line chart."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data
variants = [1, 3, 10, 30, 100]
selection_time = [2, 5, 15, 40, 62.5]

# Reference lines
cu_launch_kernel = 1650      # ns
cu_module_load_data = 36000  # ns

# Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Main line
ax.plot(variants, selection_time, color='#0d9488', marker='o', markersize=7,
        linewidth=2.5, markeredgecolor='white', markeredgewidth=1.5, zorder=5,
        label='Selection time')

# Reference lines
ax.axhline(y=cu_launch_kernel, color='#7c3aed', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
ax.text(105, cu_launch_kernel, 'cuLaunchKernel\n1,650 ns', fontsize=8.5, color='#7c3aed',
        va='center', fontweight='medium')

ax.axhline(y=cu_module_load_data, color='#dc2626', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
ax.text(105, cu_module_load_data, 'cuModuleLoadData\n36,000 ns', fontsize=8.5, color='#dc2626',
        va='center', fontweight='medium')

# Annotation
ax.annotate(
    'Even at 100 variants:\n26,000× cheaper than module load',
    xy=(100, 62.5), xytext=(8, 400),
    fontsize=9, fontweight='bold', color='#0d9488',
    arrowprops=dict(arrowstyle='->', color='#0d9488', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdfa', edgecolor='#0d9488', linewidth=0.8),
)

# Axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.8, 150)
ax.set_ylim(1, 100000)

ax.set_xlabel('Number of Variants', fontsize=11, fontweight='medium')
ax.set_ylabel('Time (ns)', fontsize=11, fontweight='medium')

# Ticks
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else f'{x:g}'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}' if x >= 1 else f'{x:g}'))

# Gridlines
ax.grid(True, which='both', linewidth=0.3, alpha=0.4, color='#d1d5db')
ax.set_axisbelow(True)

# Title
ax.set_title('Selection Time vs. Kernel Variants', fontsize=13, fontweight='bold', pad=12, loc='left')

# Spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.savefig('/home/akash/PROJECTS/LLVM/poster/figures/variant-scaling.svg',
            format='svg', bbox_inches='tight', dpi=300)
plt.close()
print('OK: variant-scaling.svg')
