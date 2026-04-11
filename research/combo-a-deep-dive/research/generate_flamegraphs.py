#!/usr/bin/env python3
"""Generate flame graph SVGs from bench_layers benchmark data.

Produces two SVGs:
  flamegraph_cold.svg — Cold-path dispatch (module load + launch + sync)
  flamegraph_hot.svg  — Hot-path dispatch (cached module, launch + sync only)

No external dependencies (no flamegraph.pl). Pure SVG generation.
Data source: bench_layers.c on GTX 1650, sm_75, CUDA 13.1
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FlameLayer:
    label: str
    duration_ns: float
    percentage: float


# --- Color palette (warm gradient, Brendan Gregg style) ---
COLORS = [
    "#a93226",  # deepest red (bottom)
    "#c0392b",  # red
    "#d35400",  # dark orange
    "#e67e22",  # orange
    "#f39c12",  # amber
    "#f1c40f",  # yellow (top)
]

BG_COLOR = "#ffffff"
BORDER_COLOR = "#000000"
FONT_FAMILY = "monospace"
FONT_SIZE = 12
TITLE_FONT_SIZE = 16
SUBTITLE_FONT_SIZE = 11
CANVAS_WIDTH = 1200
BAR_HEIGHT = 26
BAR_GAP = 3
PADDING_X = 20
PADDING_TOP = 70
PADDING_BOTTOM = 20
CORNER_RADIUS = 2


def _format_duration(ns: float) -> str:
    if ns >= 1000:
        return f"{ns / 1000:.1f} \u00b5s"
    return f"{ns:.0f} ns"


def _bar_color(index: int, total: int) -> str:
    """Interpolate through the color palette based on layer position."""
    if total <= 1:
        return COLORS[0]
    t = index / (total - 1)
    palette_pos = t * (len(COLORS) - 1)
    lo = int(palette_pos)
    hi = min(lo + 1, len(COLORS) - 1)
    frac = palette_pos - lo

    lo_rgb = _hex_to_rgb(COLORS[lo])
    hi_rgb = _hex_to_rgb(COLORS[hi])
    r = int(lo_rgb[0] + frac * (hi_rgb[0] - lo_rgb[0]))
    g = int(lo_rgb[1] + frac * (hi_rgb[1] - lo_rgb[1]))
    b = int(lo_rgb[2] + frac * (hi_rgb[2] - lo_rgb[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _build_svg(
    title: str,
    subtitle: str,
    layers: list[FlameLayer],
    total_ns: float,
    output_path: Path,
) -> None:
    """Render a flame graph SVG to disk."""
    n = len(layers)
    bar_slot = BAR_HEIGHT + BAR_GAP
    canvas_height = PADDING_TOP + n * bar_slot + PADDING_BOTTOM
    usable_width = CANVAS_WIDTH - 2 * PADDING_X

    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(CANVAS_WIDTH),
        height=str(canvas_height),
        viewBox=f"0 0 {CANVAS_WIDTH} {canvas_height}",
    )

    # Background
    ET.SubElement(
        svg,
        "rect",
        width=str(CANVAS_WIDTH),
        height=str(canvas_height),
        fill=BG_COLOR,
    )

    # Title
    t = ET.SubElement(
        svg,
        "text",
        x=str(CANVAS_WIDTH // 2),
        y="28",
        fill="#2c3e50",
        **{
            "font-family": FONT_FAMILY,
            "font-size": str(TITLE_FONT_SIZE),
            "font-weight": "bold",
            "text-anchor": "middle",
        },
    )
    t.text = title

    # Subtitle
    st = ET.SubElement(
        svg,
        "text",
        x=str(CANVAS_WIDTH // 2),
        y="48",
        fill="#7f8c8d",
        **{
            "font-family": FONT_FAMILY,
            "font-size": str(SUBTITLE_FONT_SIZE),
            "text-anchor": "middle",
        },
    )
    st.text = subtitle

    # Layers — bottom-up (index 0 is widest/bottom bar)
    for i, layer in enumerate(layers):
        # y: bottom layer at the bottom of the stack
        y = PADDING_TOP + (n - 1 - i) * bar_slot
        bar_w = max((layer.duration_ns / total_ns) * usable_width, 1.0)
        x = PADDING_X

        color = _bar_color(i, n)

        # Bar rect
        ET.SubElement(
            svg,
            "rect",
            x=f"{x:.1f}",
            y=f"{y:.1f}",
            width=f"{bar_w:.1f}",
            height=str(BAR_HEIGHT),
            fill=color,
            stroke=BORDER_COLOR,
            **{
                "stroke-width": "1",
                "rx": str(CORNER_RADIUS),
                "ry": str(CORNER_RADIUS),
            },
        )

        # Label text
        label_text = layer.label
        text_y = y + BAR_HEIGHT / 2 + FONT_SIZE * 0.35

        # Estimate text width (~7.2px per char in 12px monospace)
        est_text_w = len(label_text) * 7.2

        if bar_w > est_text_w + 8:
            # Text fits inside the bar — center it
            text_x = x + bar_w / 2
            anchor = "middle"
            fill = "#ffffff"
            shadow_fill = "#000000"
            shadow_opacity = "0.3"
        else:
            # Bar too narrow for interior text — place label to the right
            text_x = x + bar_w + 5
            anchor = "start"
            fill = "#2c3e50"
            shadow_fill = None
            shadow_opacity = None

        # Shadow first (renders behind)
        if shadow_fill is not None:
            ET.SubElement(
                svg,
                "text",
                x=f"{text_x:.1f}",
                y=f"{text_y + 0.5:.1f}",
                fill=shadow_fill,
                opacity=shadow_opacity,
                **{
                    "font-family": FONT_FAMILY,
                    "font-size": str(FONT_SIZE),
                    "text-anchor": anchor,
                    "dominant-baseline": "auto",
                },
            ).text = label_text

        # Primary label
        ET.SubElement(
            svg,
            "text",
            x=f"{text_x:.1f}",
            y=f"{text_y:.1f}",
            fill=fill,
            **{
                "font-family": FONT_FAMILY,
                "font-size": str(FONT_SIZE),
                "text-anchor": anchor,
                "dominant-baseline": "auto",
            },
        ).text = label_text

    # Write SVG
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    with open(output_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    print(f"  wrote {output_path} ({output_path.stat().st_size} bytes)")


def generate_cold_flamegraph(output_dir: Path) -> None:
    total_ns = 46784.0
    layers = [
        FlameLayer(
            f"Total Cold Dispatch ({_format_duration(total_ns)})",
            total_ns,
            100.0,
        ),
        FlameLayer(
            f"cuModuleLoadData ({_format_duration(42670)}, 91%)",
            42670.0,
            91.2,
        ),
        FlameLayer(
            f"cuStreamSynchronize ({_format_duration(2475)}, 5.3%)",
            2475.0,
            5.3,
        ),
        FlameLayer(
            f"cuLaunchKernel ({_format_duration(1573)}, 3.4%)",
            1573.0,
            3.4,
        ),
        FlameLayer(
            f"cuModuleGetFunction ({_format_duration(60)}, 0.1%)",
            60.0,
            0.1,
        ),
        FlameLayer(
            f"Selection ({_format_duration(6)}, 0.01%)",
            6.0,
            0.01,
        ),
    ]

    _build_svg(
        title="LLVM GPU Dispatch Stack \u2014 Cold Path",
        subtitle="GTX 1650, sm_75, CUDA 13.1 | bench_layers.c",
        layers=layers,
        total_ns=total_ns,
        output_path=output_dir / "flamegraph_cold.svg",
    )


def generate_hot_flamegraph(output_dir: Path) -> None:
    total_ns = 4048.0
    layers = [
        FlameLayer(
            f"Hot Dispatch ({_format_duration(total_ns)})",
            total_ns,
            100.0,
        ),
        FlameLayer(
            f"cuStreamSynchronize ({_format_duration(2475)}, 61%)",
            2475.0,
            61.1,
        ),
        FlameLayer(
            f"cuLaunchKernel ({_format_duration(1573)}, 39%)",
            1573.0,
            38.9,
        ),
        FlameLayer(
            f"Selection ({_format_duration(6)}, 0.1%)",
            6.0,
            0.1,
        ),
    ]

    _build_svg(
        title="LLVM GPU Dispatch Stack \u2014 Hot Path",
        subtitle="GTX 1650, sm_75, CUDA 13.1 | bench_layers.c",
        layers=layers,
        total_ns=total_ns,
        output_path=output_dir / "flamegraph_hot.svg",
    )


def main() -> None:
    output_dir = Path(__file__).parent
    print("Generating flame graph SVGs...")
    generate_cold_flamegraph(output_dir)
    generate_hot_flamegraph(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
