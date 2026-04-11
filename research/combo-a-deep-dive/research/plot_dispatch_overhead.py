"""
Dispatch overhead visualization for LLVM Dublin poster.
Generates two charts:
  1. Absolute dispatch latency (log scale, ns)
  2. Overhead as % of ML kernel duration for 1ms / 10ms / 100ms kernels
"""

import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE = pathlib.Path(__file__).parent
DATA_FILE = HERE / "poster-chart-data.csv"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

def load_data(path: pathlib.Path) -> tuple[list[str], list[float], list[float]]:
    operations: list[str] = []
    medians: list[float] = []
    p99s: list[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            operations.append(row["operation"])
            medians.append(float(row["median_ns"]))
            p99s.append(float(row["p99_ns"]))
    return operations, medians, p99s

# ------------------------------------------------------------------
# Color mapping
# ------------------------------------------------------------------

COLORS: dict[str, str] = {
    "runtime_select_poc": "#2ecc71",   # green  — pure lookup
    "kdl_load_bundle":    "#3498db",   # blue   — kdl ops
    "kdl_select_cold":    "#2980b9",   # blue   — kdl ops
    "kdl_select_cached":  "#1a6fa3",   # blue   — kdl ops
    "cuda_direct_launch": "#e74c3c",   # red    — CUDA baseline
}

LABELS: dict[str, str] = {
    "kdl_load_bundle":    "kdl_load_bundle",
    "kdl_select_cold":    "kdl_select_cold",
    "kdl_select_cached":  "kdl_select_cached",
    "cuda_direct_launch": "cuda_direct_launch",
    "runtime_select_poc": "runtime_select_poc\n(pure lookup)",
}

# ------------------------------------------------------------------
# Chart 1: Absolute latency — log-scale bar chart
# ------------------------------------------------------------------

def plot_latency(
    operations: list[str],
    medians: list[float],
    p99s: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(operations))
    width = 0.35

    bar_colors = [COLORS[op] for op in operations]
    bars_med = ax.bar(x - width / 2, medians, width, label="Median", color=bar_colors, alpha=0.9)
    bars_p99 = ax.bar(x + width / 2, p99s, width, label="p99", color=bar_colors, alpha=0.5,
                      edgecolor=[COLORS[op] for op in operations], linewidth=1.2)

    ax.set_yscale("log")
    ax.set_ylabel("Latency (ns, log scale)", fontsize=12)
    ax.set_title("Dispatch Overhead on GTX 1650 (sm_75)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[op] for op in operations], fontsize=10)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)

    # Annotate median values above each bar
    for bar, val in zip(bars_med, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.4,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Legend patches for color meaning
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Pure lookup"),
        Patch(facecolor="#3498db", label="KDL operation"),
        Patch(facecolor="#e74c3c", label="CUDA direct launch"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(HERE / "dispatch_overhead.png", dpi=150)
    fig.savefig(HERE / "dispatch_overhead.svg")
    plt.close(fig)
    print("Saved: dispatch_overhead.png, dispatch_overhead.svg")


# ------------------------------------------------------------------
# Chart 2: Overhead as % of ML kernel duration
# ------------------------------------------------------------------

def plot_overhead_pct(
    operations: list[str],
    medians: list[float],
) -> None:
    kernel_durations_ns: dict[str, float] = {
        "1 ms kernel":   1_000_000,
        "10 ms kernel":  10_000_000,
        "100 ms kernel": 100_000_000,
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(operations))
    n_groups = len(kernel_durations_ns)
    total_width = 0.7
    width = total_width / n_groups
    offsets = np.linspace(-total_width / 2 + width / 2, total_width / 2 - width / 2, n_groups)

    group_colors = ["#f39c12", "#8e44ad", "#16a085"]

    for (duration_label, duration_ns), offset, color in zip(
        kernel_durations_ns.items(), offsets, group_colors
    ):
        pct_values = [(m / duration_ns) * 100 for m in medians]
        bars = ax.bar(
            x + offset,
            pct_values,
            width,
            label=duration_label,
            color=color,
            alpha=0.82,
        )
        for bar, pct in zip(bars, pct_values):
            if pct >= 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{pct:.3f}%",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_ylabel("Dispatch overhead (% of kernel duration)", fontsize=12)
    ax.set_title(
        "Dispatch Overhead as % of ML Kernel Duration — GTX 1650 (sm_75)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[op] for op in operations], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}%"))
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(HERE / "dispatch_overhead_pct.png", dpi=150)
    fig.savefig(HERE / "dispatch_overhead_pct.svg")
    plt.close(fig)
    print("Saved: dispatch_overhead_pct.png, dispatch_overhead_pct.svg")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    operations, medians, p99s = load_data(DATA_FILE)
    plot_latency(operations, medians, p99s)
    plot_overhead_pct(operations, medians)
