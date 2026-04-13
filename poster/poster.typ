// EuroLLVM Dublin 2026 — Conference Poster
// Bridging Runtime Gaps in LLVM: Vendor-Agnostic Dispatch for ML Kernels

// ─── Page Setup ───────────────────────────────────────────────────
#set page(
  width: 841mm,
  height: 1189mm,
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

// ─── Typography ───────────────────────────────────────────────────
#set text(font: "Libertinus Serif", size: 20pt, fill: rgb("#1a1a1a"))
#show heading.where(level: 1): set text(font: "Liberation Sans", size: 30pt, weight: "bold", fill: navy)
#show heading.where(level: 2): set text(font: "Liberation Sans", size: 26pt, weight: "bold", fill: navy)
#show heading.where(level: 3): set text(font: "Liberation Sans", size: 22pt, weight: "semibold", fill: rgb("#333"))

// ─── Card Helper ──────────────────────────────────────────────────
#let card(accent: llvm-blue, body, grow: false) = {
  block(
    width: 100%,
    inset: 12pt,
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
    inset: 6pt,
    radius: 4pt,
    fill: accent.lighten(88%),
    stroke: 1pt + accent.lighten(60%),
    align(center)[
      #text(size: 34pt, weight: "bold", fill: accent)[#value]\
      #text(size: 14pt, fill: rgb("#555"))[#label]
    ],
  )
}

// ─── Inline Code ──────────────────────────────────────────────────
#show raw.where(block: false): it => {
  box(
    inset: (x: 3pt, y: 2pt),
    radius: 3pt,
    fill: rgb("#f0f0f0"),
    text(font: "DejaVu Sans Mono", size: 17pt, fill: rgb("#c7254e"), it),
  )
}

#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 8pt,
    radius: 4pt,
    fill: rgb("#282c34"),
    text(font: "DejaVu Sans Mono", size: 15pt, fill: rgb("#abb2bf"), it),
  )
}

// ─── Table Styling ────────────────────────────────────────────────
#set table(
  inset: 6pt,
  stroke: 0.5pt + luma(200),
)

// ═══════════════════════════════════════════════════════════════════
//  HEADER BANNER
// ═══════════════════════════════════════════════════════════════════

#block(
  width: 100%,
  inset: (x: 24pt, y: 18pt),
  radius: 8pt,
  fill: gradient.linear(navy, llvm-blue, angle: 0deg),
  stroke: none,
)[
  #grid(
    columns: (auto, 1fr, auto),
    column-gutter: 16pt,
    align(left + horizon)[
      #image("figures/llvm-logo.png", height: 70pt)
    ],
    align(center + horizon)[
      #text(font: "Liberation Sans", size: 48pt, weight: "bold", fill: white)[
        Bridging Runtime Gaps in LLVM:\
        Vendor-Agnostic Dispatch for ML Kernels
      ]
      #v(6pt)
      #text(font: "Liberation Sans", size: 22pt, fill: rgb("#cce0f0"))[
        S. Akash
        #h(10pt) #text(fill: rgb("#88bbdd"))[|] #h(10pt)
        IIT Patna  ·  CERN GSoC  ·  vLLM contributor
      ]
      #v(4pt)
      #text(font: "Liberation Sans", size: 17pt, fill: rgb("#99c8e8"))[
        EuroLLVM Developers' Meeting  ·  Dublin 2026
      ]
    ],
    align(right + horizon)[
      #block(inset: 8pt, radius: 4pt, fill: rgb("#ffffff22"), stroke: 1pt + rgb("#ffffff33"))[
        #text(font: "Liberation Sans", size: 16pt, weight: "bold", fill: white)[EuroLLVM\Dublin 2026]
        #v(2pt)
        #text(font: "Liberation Sans", size: 13pt, fill: rgb("#bbddee"))[Poster Session]
      ]
    ],
  )
]

#v(16pt)

// ═══════════════════════════════════════════════════════════════════
//  MAIN 3-COLUMN BODY
// ═══════════════════════════════════════════════════════════════════

#grid(
  columns: (1fr, 1.2fr, 1fr),
  column-gutter: 12pt,
  row-gutter: 12pt,

  // ─────────────────────────────────────────────────────────────────
  //  LEFT COLUMN — The Problem
  // ─────────────────────────────────────────────────────────────────
  [
    // Card 1: Problem Statement
    #card(accent: orange)[
      = #text(fill: orange)[The Problem]
      #v(6pt)
      #text(size: 22pt, weight: "bold", fill: navy)[
        MLIR compiles one `gpu.module` to 3 GPU vendors — but has no runtime intelligence to pick the right one.
      ]
      #v(8pt)

      #block(inset: 8pt, radius: 4pt, fill: light-orange, width: 100%)[
        #text(size: 18pt, fill: rgb("#6b4c00"))[
          *OffloadBinary* carries N device images. At runtime, the offload stack picks the *FIRST* compatible image. No metadata vocabulary. No measurement. No "best-compatible" mechanism.
        ]
      ]

      #v(8pt)

      *Runtime dispatch already exists everywhere:*
      - cuBLAS / cuDNN — internal kernel selection
      - PyTorch — `torch.compile` autotuning
      - CPU world — Function Multi-Versioning (FMV)
      - *MLIR is the exception*

      #v(8pt)

      #text(size: 18pt, fill: rgb("#555"), style: "italic")[
        "The runtime just loads the first image that doesn't fail."
      ]
    ]

    #v(10pt)

    // Card 2: Upstream Evidence
    #card(accent: navy)[
      = #text(fill: navy)[Upstream Evidence]
      #v(6pt)

      #table(
        columns: (auto, 1fr),
        fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 16pt)[*PR / Issue*],
          text(weight: "bold", size: 16pt)[*What it shows*],
        ),
        text(size: 16pt)[`#148286`], text(size: 16pt)[XeVM — new vendor images arriving fast],
        text(size: 16pt)[`#186088`], text(size: 16pt)[liboffload uses first-wins selection],
        text(size: 16pt)[`#185663`], text(size: 16pt)[`isMetadataCompatible` — no policy],
        text(size: 16pt)[`#75356`],  text(size: 16pt)[Chapel users need dispatch],
        text(size: 16pt)[`#88170`],  text(size: 16pt)[RFC: policy slot explicitly empty],
      )

      #v(6pt)
      #align(center)[
        #text(size: 15pt, fill: rgb("#666"))[
          5 independent signals pointing at the same gap.
        ]
      ]
    ]

    #v(10pt)

    // Card 3: Why It Matters
    #card(accent: teal)[
      = #text(fill: teal)[Why It Matters]
      #v(6pt)

      #grid(
        columns: (1fr, 1fr),
        column-gutter: 8pt,
        row-gutter: 8pt,
        stat-box("3+", "GPU vendors\nin MLIR", accent: llvm-blue),
        stat-box("0", "Lines of dispatch\npolicy upstream", accent: orange),
        stat-box("5", "Independent PRs\nhitting this gap", accent: teal),
        stat-box("1st", "Published dispatch\nflame graph", accent: navy),
      )

      #v(8pt)

      #block(inset: 8pt, radius: 4pt, fill: light-teal, width: 100%)[
        #text(size: 18pt, fill: rgb("#1a5c4a"))[
          Without runtime dispatch, multi-vendor MLIR deployments are fragile — they silently load suboptimal kernels.
        ]
      ]
    ]

    #v(10pt)

    // Card 4: Dispatch Everywhere
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[Dispatch Is Everywhere — Except MLIR]
      #v(6pt)

      #image("figures/runtime-dispatch-everywhere.svg", width: 100%)

      #v(6pt)
      #text(size: 14pt, fill: rgb("#666"))[
        Every major ML runtime implements kernel selection internally. MLIR's offload stack is the only one without a dispatch policy.
      ]
    ]
  ],

  // ─────────────────────────────────────────────────────────────────
  //  CENTER COLUMN — Our Contributions
  // ─────────────────────────────────────────────────────────────────
  [
    // Card 5: System Architecture (DOMINANT FIGURE)
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[System Architecture]
      #v(6pt)

      #image("figures/architecture.svg", width: 100%)

      #v(6pt)
      #text(size: 14pt, fill: rgb("#666"))[
        End-to-end pipeline: MLIR compilation embeds metadata in OffloadBinary; at load time, libkdl scores each variant and dispatches the best match to the active GPU.
      ]
    ]

    #v(10pt)

    // Card 6: C1 — Metadata Keys
    #card(accent: teal)[
      = #text(fill: teal)[C1: Metadata Vocabulary]
      #v(6pt)

      #text(size: 18pt)[Five new keys that let the runtime reason about variant compatibility:]

      #v(6pt)

      #table(
        columns: (auto, 1fr),
        fill: (col, row) => if row == 0 { teal.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 18pt)[*Key*],
          text(weight: "bold", size: 18pt)[*Purpose*],
        ),
        text(size: 18pt, font: "DejaVu Sans Mono")[min_sm],           text(size: 18pt)[Minimum CUDA compute capability],
        text(size: 18pt, font: "DejaVu Sans Mono")[min_gfx],          text(size: 18pt)[Minimum AMD GFX version],
        text(size: 18pt, font: "DejaVu Sans Mono")[requires_features], text(size: 18pt)[Tensor cores, matrix units, etc.],
        text(size: 18pt, font: "DejaVu Sans Mono")[variant_priority],  text(size: 18pt)[Explicit ordering for tie-breaking],
        text(size: 18pt, font: "DejaVu Sans Mono")[variant_tag],       text(size: 18pt)[Human-readable variant name],
      )

      #v(6pt)
      #block(inset: 8pt, radius: 4pt, fill: light-teal, width: 100%)[
        #text(size: 18pt, fill: rgb("#1a5c4a"))[
          These keys are *composable* — a single OffloadBinary can carry variants tagged for different vendors and feature sets. The dispatcher scores each against the live GPU context.
        ]
      ]
    ]

    #v(10pt)

    // Card 7: C3 — MLIR Attribute
    #card(accent: navy)[
      = #text(fill: navy)[C3: MLIR Attribute Design]
      #v(6pt)

      #text(size: 18pt)[A new `#gpu.runtime_select` attribute that attaches dispatch policy directly to MLIR modules:]

      #v(6pt)

      ```
      gpu.module @matmul_variants
        { gpu.runtime_select = #gpu.runtime_select<
            policy = "best_compatible",
            fallback = "first_valid"> }
      {
        gpu.func @matmul_sm80
          { variant_tag = "ampere_tc",
            min_sm = 80,
            requires_features = ["tensor_cores"] }
        { ... }

        gpu.func @matmul_sm90
          { variant_tag = "hopper_tma",
            min_sm = 90,
            requires_features = ["tma", "tensor_cores"] }
        { ... }
      }
      ```

      #v(6pt)
      #text(size: 15pt, fill: rgb("#555"))[
        The attribute is *declarative* — it expresses intent, not mechanism. The runtime implementation is free to evolve.
      ]
    ]

    #v(10pt)

    // Card 8: Latency Breakdown
    #card(accent: orange)[
      = #text(fill: orange)[C2: First Published Dispatch Flame Graph]
      #v(6pt)

      #image("figures/latency-breakdown.svg", width: 100%)

      #v(8pt)

      #grid(
        columns: (1fr, 1fr, 1fr),
        column-gutter: 8pt,
        stat-box("36.0 " + sym.mu + "s", "cuModuleLoadData\n(90% of total)", accent: orange),
        stat-box("3 -- 6 ns", "Selection\noverhead", accent: teal),
        stat-box("< 0.01%", "Dispatch cost\nvs. kernel load", accent: llvm-blue),
      )

      #v(6pt)
      #text(size: 15pt, fill: rgb("#555"))[
        Selection is *free* relative to the driver costs it rides alongside.
      ]
    ]
  ],

  // ─────────────────────────────────────────────────────────────────
  //  RIGHT COLUMN — Evidence
  // ─────────────────────────────────────────────────────────────────
  [
    // Card 9: Dispatch Latency Table
    #card(accent: navy)[
      = #text(fill: navy)[Dispatch Latency Breakdown]
      #v(6pt)

      #table(
        columns: (1fr, auto, auto),
        fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 16pt)[*Operation*],
          text(weight: "bold", size: 16pt)[*Latency*],
          text(weight: "bold", size: 16pt)[*Share*],
        ),
        text(size: 16pt)[cuModuleLoadData (cold)],  text(size: 16pt)[36.0 #sym.mu\s],  text(size: 16pt, weight: "bold", fill: orange)[90.0%],
        text(size: 16pt)[cuModuleLoadData (warm)],   text(size: 16pt)[1.2 #sym.mu\s],   text(size: 16pt)[3.0%],
        text(size: 16pt)[cuModuleGetFunction],        text(size: 16pt)[0.8 #sym.mu\s],   text(size: 16pt)[2.0%],
        text(size: 16pt)[cuLaunchKernel],             text(size: 16pt)[1.5 #sym.mu\s],   text(size: 16pt)[3.8%],
        text(size: 16pt)[cuStreamSynchronize],        text(size: 16pt)[0.5 #sym.mu\s],   text(size: 16pt)[1.2%],
        text(size: 16pt, weight: "bold")[Hot-path total], text(size: 16pt, weight: "bold")[40.0 #sym.mu\s], text(size: 16pt, weight: "bold")[100%],
        text(size: 16pt, fill: teal, weight: "bold")[Selection overhead], text(size: 16pt, fill: teal, weight: "bold")[3 -- 6 ns], text(size: 16pt, fill: teal, weight: "bold")[< 0.01%],
      )
    ]

    #v(10pt)

    // Card 10: Variant Scaling
    #card(accent: teal)[
      = #text(fill: teal)[Selection Scales Linearly]
      #v(6pt)

      #image("figures/variant-scaling.svg", width: 100%)

      #v(6pt)
      #text(size: 14pt, fill: rgb("#666"))[
        Even at 64 variants, selection stays under 400ns — three orders of magnitude below driver overhead.
      ]
    ]

    #v(10pt)

    // Card 11: Overhead Comparison
    #card(accent: orange)[
      = #text(fill: orange)[Overhead Comparison]
      #v(6pt)

      #image("figures/overhead-comparison.svg", width: 100%)

      #v(6pt)
      #text(size: 14pt, fill: rgb("#666"))[
        Our dispatch adds negligible overhead compared to existing runtime costs — selection is noise-level relative to module loading.
      ]
    ]

    #v(10pt)

    // Card 12: Key Findings
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[Key Findings]
      #v(6pt)

      #grid(
        columns: (1fr,),
        row-gutter: 8pt,
        block(inset: 8pt, radius: 4pt, fill: light-blue, width: 100%)[
          #text(size: 18pt)[
            *F1:* Module loading dominates dispatch latency at 90% — selection policy is essentially free.
          ]
        ],
        block(inset: 8pt, radius: 4pt, fill: light-teal, width: 100%)[
          #text(size: 18pt)[
            *F2:* Five metadata keys suffice to express all observed vendor-selection patterns in the wild.
          ]
        ],
        block(inset: 8pt, radius: 4pt, fill: light-orange, width: 100%)[
          #text(size: 18pt)[
            *F3:* Linear variant scaling means the approach works from 2 to 64+ variants without architectural changes.
          ]
        ],
        block(inset: 8pt, radius: 4pt, fill: rgb("#f0e8f5"), width: 100%)[
          #text(size: 18pt)[
            *F4:* A declarative MLIR attribute (`#gpu.runtime_select`) cleanly separates policy from mechanism.
          ]
        ],
      )
    ]

    #v(10pt)

    // Card 13: Prototype
    #card(accent: navy)[
      = #text(fill: navy)[Prototype Implementation]
      #v(6pt)

      #grid(
        columns: (1fr, 1fr),
        column-gutter: 8pt,
        stat-box("5,100", "LOC — libkdl\n(dispatch library)", accent: navy),
        stat-box("664", "LOC — PoC\n(MLIR integration)", accent: teal),
      )

      #v(8pt)
      #text(size: 18pt)[
        *libkdl* (Kernel Dynamic Linker) is a standalone C library that implements the dispatch algorithm. The PoC wires it into the MLIR GPU runtime via `gpu.launch_func` lowering.
      ]
    ]
  ],
)

#v(8pt)

// ═══════════════════════════════════════════════════════════════════
//  BOTTOM — Related Work + Upstream Path (compact, side by side)
// ═══════════════════════════════════════════════════════════════════

#grid(
  columns: (1.4fr, 1fr),
  column-gutter: 10pt,

  // Related Work — compact table
  block(width: 100%, inset: 8pt, radius: 4pt, fill: card-bg, stroke: (top: 3pt + navy, left: 0.5pt + luma(210), right: 0.5pt + luma(210), bottom: 0.5pt + luma(210)))[
    #text(font: "Liberation Sans", size: 22pt, weight: "bold", fill: navy)[Related Work]
    #v(4pt)
    #set table(inset: 5pt, stroke: 0.4pt + luma(210))
    #table(
      columns: (1fr, auto, auto, auto, auto, auto),
      fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
      table.header(
        text(weight: "bold", size: 13pt)[*System*],
        text(weight: "bold", size: 13pt)[*Vendor*],
        text(weight: "bold", size: 13pt)[*Meta*],
        text(weight: "bold", size: 13pt)[*Policy*],
        text(weight: "bold", size: 13pt)[*Data*],
        text(weight: "bold", size: 13pt)[*Upstr.*],
      ),
      text(size: 13pt)[IREE],       text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: orange)[~], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: orange)[✗],
      text(size: 13pt)[chipStar],    text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: orange)[✗],
      text(size: 13pt)[Proteus],     text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: orange)[✗],
      text(size: 13pt)[liboffload],  text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: orange)[~], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: teal)[✓],
      text(size: 13pt)[CPU FMV],     text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: teal)[✓], text(size: 14pt, fill: orange)[✗], text(size: 14pt, fill: teal)[✓],
      text(size: 13pt, weight: "bold", fill: llvm-blue)[*Ours*], text(size: 14pt, fill: teal, weight: "bold")[✓], text(size: 14pt, fill: teal, weight: "bold")[✓], text(size: 14pt, fill: teal, weight: "bold")[✓], text(size: 14pt, fill: teal, weight: "bold")[✓], text(size: 14pt, fill: teal, weight: "bold")[✓],
    )
  ],

  // Upstream Path — compact
  block(width: 100%, inset: 8pt, radius: 4pt, fill: card-bg, stroke: (top: 3pt + teal, left: 0.5pt + luma(210), right: 0.5pt + luma(210), bottom: 0.5pt + luma(210)))[
    #text(font: "Liberation Sans", size: 22pt, weight: "bold", fill: teal)[Upstream Path]
    #v(4pt)
    #let step(num, title, desc, accent: teal) = {
      block(width: 100%, inset: 5pt, radius: 3pt, fill: accent.lighten(92%), stroke: 1pt + accent.lighten(60%))[
        #grid(columns: (auto, 1fr), column-gutter: 6pt,
          box(width: 22pt, height: 22pt, radius: 11pt, fill: accent, align(center + horizon)[#text(size: 13pt, weight: "bold", fill: white)[#num]]),
          [#text(size: 16pt, weight: "bold", fill: navy)[#title] #text(size: 13pt, fill: rgb("#555"))[— #desc]],
        )
      ]
    }
    #step("1", "Metadata RFC", "5 keys, ~30 LOC patch", accent: teal)
    #v(4pt)
    #step("2", "Policy Slot", "Pluggable hook in liboffload", accent: llvm-blue)
    #v(4pt)
    #step("3", "MLIR Attribute", "~780 LOC, gpu.launch_func lowering", accent: orange)
    #v(4pt)
    #text(size: 13pt, fill: rgb("#666"), style: "italic")[Each step independently useful.]
  ],
)

#v(6pt)

// ═══════════════════════════════════════════════════════════════════
//  FOOTER (compact)
// ═══════════════════════════════════════════════════════════════════

#block(width: 100%, inset: (x: 14pt, y: 8pt), radius: 4pt, fill: navy.lighten(92%), stroke: 1pt + navy.lighten(70%))[
  #grid(columns: (1fr, 1fr, 1fr),
    align(left)[#text(size: 14pt, fill: navy)[*Code:* github.com/Akasxh/libkdl · *Contact:* sakash\@iitp.ac.in]],
    align(center)[#text(size: 14pt, fill: navy)[*EuroLLVM Developers' Meeting — Dublin 2026*]],
    align(right)[#text(size: 14pt, fill: navy)[*5,100 + 664 LOC* · Apache 2.0 w/ LLVM Exception]],
  )
]
