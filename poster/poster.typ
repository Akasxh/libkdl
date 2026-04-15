// EuroLLVM Dublin 2026: Conference Poster
// Bridging Runtime Gaps in LLVM: Vendor-Agnostic Dispatch for ML Kernels
// Structure: MAB Profiled Adaptive Dispatch as CENTERPIECE

// ─── Page Setup ───────────────────────────────────────────────────
#set page(
  width: 841mm,
  height: auto,
  margin: 14mm,
  fill: rgb("#f0ece6"),
)

// ─── Colors ───────────────────────────────────────────────────────
#let llvm-blue  = rgb("#2d7fc1")
#let teal       = rgb("#2aaa8a")
#let orange     = rgb("#e8943a")
#let navy       = rgb("#1a365d")
#let cream      = rgb("#f0ece6")
#let card-bg    = white
#let light-blue = rgb("#e8f1fa")
#let light-teal = rgb("#e6f6f1")
#let light-orange = rgb("#fdf0e0")
#let deep-purple = rgb("#6b21a8")
#let light-purple = rgb("#f3e8ff")

// ─── Typography ───────────────────────────────────────────────────
#set text(font: "Libertinus Serif", size: 19pt, fill: rgb("#1a1a1a"))
#show heading.where(level: 1): set text(font: "Liberation Sans", size: 28pt, weight: "bold", fill: navy)
#show heading.where(level: 2): set text(font: "Liberation Sans", size: 24pt, weight: "bold", fill: navy)
#show heading.where(level: 3): set text(font: "Liberation Sans", size: 20pt, weight: "semibold", fill: rgb("#333"))

// ─── Card Helper ──────────────────────────────────────────────────
#let card(accent: llvm-blue, body) = {
  block(
    width: 100%,
    inset: 11pt,
    radius: 6pt,
    fill: card-bg,
    stroke: (
      top: 4pt + accent,
      left: 0.5pt + luma(210),
      right: 0.5pt + luma(210),
      bottom: 0.5pt + luma(210),
    ),
    body,
  )
}

// ─── Stat Box Helper ─────────────────────────────────────────────
#let stat-box(value, label, accent: llvm-blue) = {
  box(
    width: 100%,
    inset: 5pt,
    radius: 4pt,
    fill: accent.lighten(88%),
    stroke: 1pt + accent.lighten(60%),
    align(center)[
      #text(size: 30pt, weight: "bold", fill: accent)[#value]\
      #text(size: 13pt, fill: rgb("#555"))[#label]
    ],
  )
}

// ─── Inline Code ──────────────────────────────────────────────────
#show raw.where(block: false): it => {
  box(
    inset: (x: 3pt, y: 2pt),
    radius: 3pt,
    fill: rgb("#f0f0f0"),
    text(font: "DejaVu Sans Mono", size: 16pt, fill: rgb("#c7254e"), it),
  )
}

#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 8pt,
    radius: 4pt,
    fill: rgb("#282c34"),
    text(font: "DejaVu Sans Mono", size: 14pt, fill: rgb("#abb2bf"), it),
  )
}

// ─── Table Styling ────────────────────────────────────────────────
#set table(
  inset: 5pt,
  stroke: 0.5pt + luma(200),
)

// ═══════════════════════════════════════════════════════════════════
//  HEADER BANNER
// ═══════════════════════════════════════════════════════════════════

#block(
  width: 100%,
  inset: (x: 24pt, y: 16pt),
  radius: 8pt,
  fill: gradient.linear(navy, llvm-blue, angle: 0deg),
  stroke: none,
)[
  #grid(
    columns: (auto, 1fr, auto),
    column-gutter: 16pt,
    align(left + horizon)[
      #image("figures/llvm-logo.png", height: 140pt)
    ],
    align(center + horizon)[
      #text(font: "Liberation Sans", size: 48pt, weight: "bold", fill: white)[
        Bridging Runtime Gaps in LLVM:\
        Vendor-Agnostic Dispatch for ML Kernels
      ]
      #v(5pt)
      #text(font: "Liberation Sans", size: 22pt, fill: rgb("#cce0f0"))[
        S. Akash
        #h(10pt) #text(fill: rgb("#88bbdd"))[|] #h(10pt)
        IIT Patna  ·  CERN GSoC  ·  vLLM contributor
      ]
      #v(3pt)
      #text(font: "Liberation Sans", size: 17pt, fill: rgb("#99c8e8"))[
        EuroLLVM Developers' Meeting  ·  Dublin 2026
      ]
    ],
  )
]

#v(12pt)

// ═══════════════════════════════════════════════════════════════════
//  MAIN 3-COLUMN BODY
// ═══════════════════════════════════════════════════════════════════

#grid(
  columns: (1fr, 1.2fr, 1fr),
  column-gutter: 11pt,
  row-gutter: 10pt,

  // ─────────────────────────────────────────────────────────────────
  //  COLUMN 1: Problem + Setup + Evidence from Col 3
  // ─────────────────────────────────────────────────────────────────
  [
    // Card: The Gap
    #card(accent: orange)[
      = #text(fill: orange)[The Gap]
      #v(4pt)
      #text(size: 20pt, weight: "bold", fill: navy)[
        MLIR compiles one `gpu.module` to 3+ GPU vendors, but picks the *first compatible* binary at runtime.
      ]
      #v(6pt)
      #block(inset: 7pt, radius: 4pt, fill: light-orange, width: 100%)[
        #text(size: 17pt, fill: rgb("#6b4c00"))[
          *OffloadBinary* carries N device images. The runtime loads the first image that doesn't fail. No metadata vocabulary. No measurement. No "best-compatible" mechanism.
        ]
      ]
    ]

    #v(8pt)

    // Card: Upstream Evidence
    #card(accent: navy)[
      = #text(fill: navy)[Upstream Evidence]
      #v(4pt)
      #table(
        columns: (auto, 1fr),
        fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 15pt)[*PR / Issue*],
          text(weight: "bold", size: 15pt)[*What it shows*],
        ),
        text(size: 15pt)[`#148286`], text(size: 15pt)[XeVM: new vendor images arriving fast],
        text(size: 15pt)[`#186088`], text(size: 15pt)[liboffload uses first-wins selection],
        text(size: 15pt)[`#185663`], text(size: 15pt)[`isMetadataCompatible`: no policy],
        text(size: 15pt)[`#75356`],  text(size: 15pt)[Chapel users need dispatch],
        text(size: 15pt)[`#88170`],  text(size: 15pt)[RFC: policy slot explicitly empty],
      )
      #v(3pt)
      #align(center)[
        #text(size: 14pt, fill: rgb("#666"))[5 independent signals pointing at the same gap.]
      ]
    ]

    #v(8pt)

    // Card: Phase 1: Dispatch Measurement (SETUP for main story)
    #card(accent: orange)[
      = #text(fill: orange)[Phase 1: Dispatch Measurement]
      #v(4pt)

      #image("figures/latency-breakdown.svg", width: 100%)

      #v(6pt)

      #grid(
        columns: (1fr, 1fr, 1fr),
        column-gutter: 6pt,
        stat-box("36 " + sym.mu + "s", "cuModuleLoadData\n(90% of cold path)", accent: orange),
        stat-box("3--6 ns", "Selection\noverhead", accent: teal),
        stat-box("< 0.02%", "Dispatch cost\nvs. kernel load", accent: llvm-blue),
      )

      #v(4pt)
      #block(inset: 7pt, radius: 4pt, fill: light-orange, width: 100%)[
        #text(size: 18pt, weight: "bold", fill: rgb("#6b4c00"))[
          Selection is *free* relative to driver costs. So: what information should *drive* it?
        ]
      ]
    ]

    #v(8pt)

    // Card: Metadata Vocabulary
    #card(accent: teal)[
      = #text(fill: teal)[Metadata Vocabulary]
      #v(4pt)
      #table(
        columns: (auto, 1fr),
        fill: (col, row) => if row == 0 { teal.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 15pt)[*Key*],
          text(weight: "bold", size: 15pt)[*Purpose*],
        ),
        text(size: 15pt, font: "DejaVu Sans Mono")[min_sm],           text(size: 15pt)[Min CUDA compute capability],
        text(size: 15pt, font: "DejaVu Sans Mono")[min_gfx],          text(size: 15pt)[Min AMD GFX version],
        text(size: 15pt, font: "DejaVu Sans Mono")[requires_features], text(size: 15pt)[Tensor cores, matrix units, etc.],
        text(size: 15pt, font: "DejaVu Sans Mono")[variant_priority],  text(size: 15pt)[Explicit ordering for tie-breaking],
        text(size: 15pt, font: "DejaVu Sans Mono")[variant_tag],       text(size: 15pt)[Human-readable variant name],
      )
    ]

    #v(8pt)

    // Related Work
    #let yes = text(size: 20pt, fill: rgb("#16a34a"), weight: "bold")[●]
    #let no = text(size: 20pt, fill: rgb("#dc2626"), weight: "bold")[●]
    #let partial = text(size: 20pt, fill: rgb("#d97706"), weight: "bold")[●]
    #card(accent: navy)[
      = #text(fill: navy)[Related Work]
      #v(4pt)
      #set table(inset: 5pt, stroke: 0.5pt + luma(200))
      #table(
        columns: (1.2fr, auto, auto, auto, auto, auto),
        align: (left, center, center, center, center, center),
        fill: (col, row) => if row == 0 { navy.lighten(85%) } else if row == 6 { teal.lighten(90%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 16pt)[*System*],
          text(weight: "bold", size: 16pt)[*Vend.*],
          text(weight: "bold", size: 16pt)[*Meta*],
          text(weight: "bold", size: 16pt)[*Pol.*],
          text(weight: "bold", size: 16pt)[*Data*],
          text(weight: "bold", size: 16pt)[*Ups.*],
        ),
        text(size: 16pt)[IREE],       yes, yes, partial, no, no,
        text(size: 16pt)[chipStar],    yes, no, no, no, no,
        text(size: 16pt)[Proteus],     no, no, yes, yes, no,
        text(size: 16pt)[liboffload],  yes, partial, no, no, yes,
        text(size: 16pt)[CPU FMV],     no, yes, yes, no, yes,
        text(size: 16pt, weight: "bold", fill: llvm-blue)[*Ours*], yes, yes, yes, yes, partial,
      )
      #v(2pt)
      #grid(columns: (auto, auto, auto, 1fr), column-gutter: 10pt,
        [#text(size: 20pt, fill: rgb("#16a34a"))[●] #text(size: 14pt)[Yes]],
        [#text(size: 20pt, fill: rgb("#d97706"))[●] #text(size: 14pt)[Partial]],
        [#text(size: 20pt, fill: rgb("#dc2626"))[●] #text(size: 14pt)[No]],
        align(right)[#text(size: 13pt, fill: rgb("#666"))[First to combine all five.]],
      )
    ]

    #v(8pt)

    // Selection Scales Linearly (moved from col 3)
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[Selection Scales Linearly]
      #v(4pt)
      #image("figures/variant-scaling.svg", width: 100%)
      #v(4pt)
      #text(size: 14pt, fill: rgb("#666"))[
        Even at 64 variants, selection stays under 400 ns. Three orders of magnitude below driver overhead.
      ]
    ]

  ],

  // ─────────────────────────────────────────────────────────────────
  //  COLUMN 2: Phase 2: Profiled Adaptive Dispatch (THE MAIN STORY)
  // ─────────────────────────────────────────────────────────────────
  [
    // THE INSIGHT: Big callout
    #block(
      width: 100%,
      inset: 14pt,
      radius: 8pt,
      fill: gradient.linear(deep-purple.lighten(85%), llvm-blue.lighten(85%), angle: 0deg),
      stroke: (
        top: 5pt + deep-purple,
        left: 2pt + deep-purple.lighten(40%),
        right: 2pt + deep-purple.lighten(40%),
        bottom: 2pt + deep-purple.lighten(40%),
      ),
    )[
      #text(font: "Liberation Sans", size: 26pt, weight: "bold", fill: deep-purple)[The Insight]
      #v(6pt)
      #text(size: 22pt, fill: navy)[
        Since selection costs *3--6 ns*, the question is not _whether_ to dispatch, but *what information drives it*. We formalize kernel variant selection as a *multi-armed bandit* problem.
      ]
      #v(6pt)
      #text(size: 18pt, fill: rgb("#555"))[
        Phase 1 measured the dispatch. Phase 2 makes it *intelligent*.
      ]
    ]

    #v(8pt)

    // System Architecture
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[System Architecture]
      #v(4pt)
      #image("figures/architecture.svg", width: 100%)
      #v(4pt)
      #text(size: 14pt, fill: rgb("#666"))[
        End-to-end: MLIR embeds metadata in OffloadBinary; at runtime, the MAB profiler scores each variant and dispatches the best match.
      ]
    ]

    #v(8pt)

    // MAB Formulation: Highlighted mathematical box
    #block(
      width: 100%,
      inset: 12pt,
      radius: 6pt,
      fill: white,
      stroke: (
        top: 4pt + deep-purple,
        left: 2pt + deep-purple.lighten(50%),
        right: 2pt + deep-purple.lighten(50%),
        bottom: 2pt + deep-purple.lighten(50%),
      ),
    )[
      #text(font: "Liberation Sans", size: 26pt, weight: "bold", fill: deep-purple)[The MAB Formulation]
      #v(6pt)

      #grid(
        columns: (auto, 1fr),
        column-gutter: 10pt,
        row-gutter: 8pt,
        text(size: 19pt, weight: "bold", fill: navy)[Arms:],
        text(size: 19pt)[$N$ pre-compiled kernel variants (typically $N < 10$)],
        text(size: 19pt, weight: "bold", fill: navy)[Reward:],
        text(size: 19pt)[Negative execution time: $r_i = -t_"exec"(v_i)$],
        text(size: 19pt, weight: "bold", fill: navy)[Context:],
        text(size: 19pt)[$(italic("kernel_name"), italic("shape"), italic("device_id"))$ #sym.arrow.r cacheable key],
        text(size: 19pt, weight: "bold", fill: navy)[Objective:],
        text(size: 19pt)[Minimize cumulative regret $R_T = sum_(t=1)^T (mu^* - mu_(a_t))$],
      )

      #v(8pt)
      #block(inset: 8pt, radius: 4pt, fill: light-purple, width: 100%)[
        #text(size: 18pt, weight: "bold", fill: deep-purple)[Structural Properties:]
        #v(4pt)
        #grid(
          columns: (1fr, 1fr, 1fr),
          column-gutter: 8pt,
          align(center)[#text(size: 17pt)[*$N < 10$*\ arms (few variants)]],
          align(center)[#text(size: 17pt)[*$sigma^2 < 5%$*\ low noise]],
          align(center)[#text(size: 17pt)[*Cacheable*\ contexts reuse]],
        )
      ]
    ]

    #v(8pt)

    // Why This Is a DEGENERATE Bandit: Key theoretical insight
    #block(
      width: 100%,
      inset: 12pt,
      radius: 6pt,
      fill: white,
      stroke: (
        top: 4pt + teal,
        left: 2pt + teal.lighten(50%),
        right: 2pt + teal.lighten(50%),
        bottom: 2pt + teal.lighten(50%),
      ),
    )[
      #text(font: "Liberation Sans", size: 24pt, weight: "bold", fill: teal)[Why This Is a Degenerate Bandit]
      #v(6pt)
      #text(size: 19pt)[
        Standard MAB algorithms (UCB1, Thompson Sampling) are designed for $N #sym.arrow infinity$ or non-stationary rewards. GPU kernel dispatch has *none* of these complexities:
      ]
      #v(6pt)
      #block(inset: 8pt, radius: 4pt, fill: light-teal, width: 100%)[
        #text(size: 19pt, fill: rgb("#1a5c4a"))[
          *Exhaustive exploration* ($N times w$ warmup samples) followed by *permanent exploitation*.\
          *Regret:* $O(N)$ constant, provably optimal for this problem class.\
          *UCB1 and Thompson Sampling are unnecessary.*
        ]
      ]
      #v(6pt)
      #text(size: 17pt, fill: rgb("#555"))[
        With $N=3$ variants and $w=3$ warmup rounds, convergence is guaranteed in *9 dispatches*. After that, every dispatch is optimal with zero marginal regret.
      ]
    ]

    #v(8pt)

    // The Algorithm: 3-phase pseudocode (trimmed)
    #card(accent: navy)[
      = #text(fill: navy)[The Algorithm]
      #v(4pt)
      ```
      fn dispatch(ctx, variants[N], warmup=3):
        key = (ctx.kernel, ctx.shape, ctx.device)
        if key in cache: return cache[key]
        if stats[key].count < N*warmup:
          arm = stats[key].count % N
          t = time(variants[arm])
          stats[key].update(arm, t)
          return variants[arm]
        cache[key] = argmin(stats[key].median)
        return cache[key]
      ```
      #v(4pt)
      #text(size: 15pt, fill: rgb("#555"))[
        Three phases: *Cold* (no data) #sym.arrow *Explore* (round-robin warmup) #sym.arrow *Exploit* (permanent lock).
      ]
    ]

    #v(8pt)

    // Key Findings (moved from col 3)
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[Key Findings]
      #v(4pt)
      #grid(
        columns: (1fr,),
        row-gutter: 5pt,
        block(inset: 6pt, radius: 4pt, fill: light-blue, width: 100%)[
          #text(size: 17pt)[*F1:* Module loading dominates cold dispatch at ~90%; selection is essentially free.]
        ],
        block(inset: 6pt, radius: 4pt, fill: light-purple, width: 100%)[
          #text(size: 17pt)[*F2:* GPU dispatch is a _degenerate_ bandit; exhaustive exploration provably optimal.]
        ],
        block(inset: 6pt, radius: 4pt, fill: light-teal, width: 100%)[
          #text(size: 17pt)[*F3:* Convergence in 9 dispatches; zero marginal regret after lock-in.]
        ],
        block(inset: 6pt, radius: 4pt, fill: light-orange, width: 100%)[
          #text(size: 17pt)[*F4:* Linear variant scaling works from 2 to 64+ variants without changes.]
        ],
      )
    ]
  ],

  // ─────────────────────────────────────────────────────────────────
  //  COLUMN 3: Compact Evidence + Results
  // ─────────────────────────────────────────────────────────────────
  [
    // Dispatch Latency Table
    #card(accent: navy)[
      = #text(fill: navy)[Dispatch Latency Breakdown]
      #v(4pt)
      #table(
        columns: (1fr, auto, auto),
        fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 15pt)[*Operation*],
          text(weight: "bold", size: 15pt)[*Latency*],
          text(weight: "bold", size: 15pt)[*Share*],
        ),
        text(size: 15pt)[cuModuleLoadData (cold)],  text(size: 15pt, weight: "bold", fill: orange)[36.0 #sym.mu\s],  text(size: 15pt)[89.6%],
        text(size: 15pt)[cuModuleLoadData (warm)],   text(size: 15pt)[9.6 #sym.mu\s],   text(size: 15pt)[--],
        text(size: 15pt)[cuModuleGetFunction],        text(size: 15pt)[63 ns],            text(size: 15pt)[0.2%],
        text(size: 15pt)[cuLaunchKernel],             text(size: 15pt)[1.65 #sym.mu\s],   text(size: 15pt)[4.1%],
        text(size: 15pt)[cuStreamSynchronize],        text(size: 15pt)[2.45 #sym.mu\s],   text(size: 15pt)[6.1%],
        text(size: 15pt, weight: "bold")[Hot-path total], text(size: 15pt, weight: "bold")[4.1 #sym.mu\s], text(size: 15pt, weight: "bold")[launch+sync],
        text(size: 15pt, fill: teal, weight: "bold")[Selection overhead], text(size: 15pt, fill: teal, weight: "bold")[3--6 ns], text(size: 15pt, fill: teal, weight: "bold")[< 0.02%],
      )
    ]

    #v(8pt)

    // Scaling with Variants
    #card(accent: deep-purple)[
      = #text(fill: deep-purple)[Scaling with Variants]
      #v(4pt)
      #image("figures/mab-scaling.svg", width: 100%)
      #v(4pt)
      #text(size: 14pt, fill: rgb("#666"))[
        Profiled dispatch achieves 83% of oracle with 7.3x less regret than random.
      ]
    ]

    #v(8pt)

    // Context-Dependent Selection (back in col 3)
    #card(accent: teal)[
      = #text(fill: teal)[Context-Dependent Selection]
      #v(4pt)
      #image("figures/mab-context.svg", width: 100%)
      #v(4pt)
      #text(size: 14pt, fill: rgb("#666"))[
        Different shapes converge to different optimal variants; context matters.
      ]
    ]

    #v(8pt)

    // CONVERGENCE FIGURE: THE HERO FIGURE (moved from col 2)
    #block(
      width: 100%,
      inset: 12pt,
      radius: 8pt,
      fill: white,
      stroke: (
        top: 5pt + deep-purple,
        left: 2pt + deep-purple.lighten(40%),
        right: 2pt + deep-purple.lighten(40%),
        bottom: 2pt + deep-purple.lighten(40%),
      ),
    )[
      #text(font: "Liberation Sans", size: 26pt, weight: "bold", fill: deep-purple)[Profiled vs. Baselines]
      #v(6pt)
      #image("figures/mab-comparison.svg", width: 100%)
      #v(6pt)
      #grid(
        columns: (1fr, 1fr, 1fr),
        column-gutter: 8pt,
        stat-box("83%", "Of oracle\nperformance", accent: deep-purple),
        stat-box[O(N)][Regret bound\ (constant)],
        stat-box("7.3x", "Less regret\nthan random", accent: teal),
      )
      #v(4pt)
      #text(size: 15pt, fill: rgb("#555"), style: "italic")[
        With 5 near-identical variants (53% spread, 6% noise), profiled dispatch converges and achieves near-oracle performance, 7.3x better than random, robust to noise.
      ]
    ]

    // Prototype + Upstream removed for cleaner alignment
  ],
)

#v(4pt)

// ═══════════════════════════════════════════════════════════════════
//  FOOTER
// ═══════════════════════════════════════════════════════════════════

#block(width: 100%, inset: (x: 14pt, y: 7pt), radius: 4pt, fill: navy.lighten(92%), stroke: 1pt + navy.lighten(70%))[
  #grid(columns: (1fr, 1fr, 1fr),
    align(left)[#text(size: 13pt, fill: navy)[*Code:* github.com/Akasxh/libkdl]],
    align(center)[#text(size: 13pt, fill: navy)[*Contact:* 2201ee54\_sakash\@iitp.ac.in · drakathakash\@gmail.com]],
    align(right)[#text(size: 13pt, fill: navy)[*EuroLLVM Developers' Meeting · Dublin 2026*]],
  )
]
