// EuroLLVM Dublin 2026 — Conference Poster
// Bridging Runtime Gaps in LLVM: Vendor-Agnostic Dispatch for ML Kernels

// ─── Page Setup ───────────────────────────────────────────────────
#set page(
  width: 1000mm,
  height: 2000mm,
  margin: 20mm,
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
#set text(font: "Libertinus Serif", size: 36pt, fill: rgb("#1a1a1a"))
#show heading.where(level: 1): set text(font: "Liberation Sans", size: 54pt, weight: "bold", fill: navy)
#show heading.where(level: 2): set text(font: "Liberation Sans", size: 48pt, weight: "bold", fill: navy)
#show heading.where(level: 3): set text(font: "Liberation Sans", size: 40pt, weight: "semibold", fill: rgb("#333"))

// ─── Card Helper ──────────────────────────────────────────────────
#let card(accent: llvm-blue, body, grow: false) = {
  block(
    width: 100%,
    inset: 24pt,
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
    inset: 12pt,
    radius: 4pt,
    fill: accent.lighten(88%),
    stroke: 1pt + accent.lighten(60%),
    align(center)[
      #text(size: 60pt, weight: "bold", fill: accent)[#value]\
      #text(size: 26pt, fill: rgb("#555"))[#label]
    ],
  )
}

// ─── Inline Code ──────────────────────────────────────────────────
#show raw.where(block: false): it => {
  box(
    inset: (x: 4pt, y: 2pt),
    radius: 3pt,
    fill: rgb("#f0f0f0"),
    text(font: "DejaVu Sans Mono", size: 30pt, fill: rgb("#c7254e"), it),
  )
}

#show raw.where(block: true): it => {
  block(
    width: 100%,
    inset: 16pt,
    radius: 4pt,
    fill: rgb("#282c34"),
    text(font: "DejaVu Sans Mono", size: 28pt, fill: rgb("#abb2bf"), it),
  )
}

// ─── Table Styling ────────────────────────────────────────────────
#set table(
  inset: 16pt,
  stroke: 0.5pt + luma(200),
)

// ═══════════════════════════════════════════════════════════════════
//  HEADER BANNER
// ═══════════════════════════════════════════════════════════════════

#block(
  width: 100%,
  inset: (x: 40pt, y: 40pt),
  radius: 8pt,
  fill: gradient.linear(navy, llvm-blue, angle: 0deg),
  stroke: none,
)[
  #align(center)[
    #text(font: "Liberation Sans", size: 80pt, weight: "bold", fill: white)[
      Bridging Runtime Gaps in LLVM:\
      Vendor-Agnostic Dispatch for ML Kernels
    ]
    #v(24pt)
    #text(font: "Liberation Sans", size: 40pt, fill: rgb("#cce0f0"))[
      S. Akash
      #h(16pt) #text(fill: rgb("#88bbdd"))[|] #h(16pt)
      IIT Patna  ·  CERN GSoC  ·  vLLM contributor
    ]
    #v(16pt)
    #text(font: "Liberation Sans", size: 32pt, fill: rgb("#99c8e8"))[
      EuroLLVM Developers' Meeting  ·  Dublin 2026
    ]
  ]
]

#v(24pt)

// ═══════════════════════════════════════════════════════════════════
//  MAIN 3-COLUMN BODY
// ═══════════════════════════════════════════════════════════════════

#grid(
  columns: (1fr, 1.2fr, 1fr),
  column-gutter: 24pt,
  row-gutter: 24pt,

  // ─────────────────────────────────────────────────────────────────
  //  LEFT COLUMN — The Problem
  // ─────────────────────────────────────────────────────────────────
  [
    // Card 1: Problem Statement
    #card(accent: orange)[
      = #text(fill: orange)[The Problem]
      #v(16pt)
      #text(size: 40pt, weight: "bold", fill: navy)[
        MLIR compiles one `gpu.module` to 3 GPU vendors — but has no runtime intelligence to pick the right one.
      ]
      #v(24pt)

      #block(inset: 20pt, radius: 4pt, fill: light-orange, width: 100%)[
        #text(size: 32pt, fill: rgb("#6b4c00"))[
          *OffloadBinary* carries N device images. At runtime, the offload stack picks the *FIRST* compatible image. No metadata vocabulary. No measurement. No "best-compatible" mechanism.
        ]
      ]

      #v(24pt)

      *Runtime dispatch already exists everywhere:*
      - cuBLAS / cuDNN — internal kernel selection
      - PyTorch — `torch.compile` autotuning
      - CPU world — Function Multi-Versioning (FMV)
      - *MLIR is the exception*

      #v(24pt)

      #text(size: 32pt, fill: rgb("#555"), style: "italic")[
        "The runtime just loads the first image that doesn't fail."
      ]
    ]

    #v(24pt)

    // Card 2: Upstream Evidence
    #card(accent: navy)[
      = #text(fill: navy)[Upstream Evidence]
      #v(16pt)

      #table(
        columns: (auto, 1fr),
        fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 30pt)[*PR / Issue*],
          text(weight: "bold", size: 30pt)[*What it shows*],
        ),
        text(size: 30pt)[`#148286`], text(size: 30pt)[XeVM — new vendor images arriving fast],
        text(size: 30pt)[`#186088`], text(size: 30pt)[liboffload uses first-wins selection],
        text(size: 30pt)[`#185663`], text(size: 30pt)[`isMetadataCompatible` — no policy],
        text(size: 30pt)[`#75356`],  text(size: 30pt)[Chapel users need dispatch],
        text(size: 30pt)[`#88170`],  text(size: 30pt)[RFC: policy slot explicitly empty],
      )

      #v(20pt)
      #align(center)[
        #text(size: 28pt, fill: rgb("#666"))[
          5 independent signals pointing at the same gap.
        ]
      ]
    ]

    #v(24pt)

    // Card 3: Why It Matters
    #card(accent: teal)[
      = #text(fill: teal)[Why It Matters]
      #v(16pt)

      #grid(
        columns: (1fr, 1fr),
        column-gutter: 18pt,
        row-gutter: 18pt,
        stat-box("3+", "GPU vendors\nin MLIR", accent: llvm-blue),
        stat-box("0", "Lines of dispatch\npolicy upstream", accent: orange),
        stat-box("5", "Independent PRs\nhitting this gap", accent: teal),
        stat-box("1st", "Published dispatch\nflame graph", accent: navy),
      )

      #v(24pt)

      #block(inset: 20pt, radius: 4pt, fill: light-teal, width: 100%)[
        #text(size: 32pt, fill: rgb("#1a5c4a"))[
          Without runtime dispatch, multi-vendor MLIR deployments are fragile — they silently load suboptimal kernels.
        ]
      ]
    ]

    #v(24pt)

    // Card 4: Dispatch Everywhere
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[Dispatch Is Everywhere — Except MLIR]
      #v(16pt)

      #image("figures/runtime-dispatch-everywhere.svg", width: 100%)

      #v(14pt)
      #text(size: 26pt, fill: rgb("#666"))[
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
      #v(16pt)

      #image("figures/architecture.svg", width: 100%)

      #v(20pt)
      #text(size: 26pt, fill: rgb("#666"))[
        End-to-end pipeline: MLIR compilation embeds metadata in OffloadBinary; at load time, libkdl scores each variant and dispatches the best match to the active GPU.
      ]
    ]

    #v(24pt)

    // Card 6: C1 — Metadata Keys
    #card(accent: teal)[
      = #text(fill: teal)[C1: Metadata Vocabulary]
      #v(16pt)

      #text(size: 32pt)[Five new keys that let the runtime reason about variant compatibility:]

      #v(16pt)

      #table(
        columns: (auto, 1fr),
        fill: (col, row) => if row == 0 { teal.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 32pt)[*Key*],
          text(weight: "bold", size: 32pt)[*Purpose*],
        ),
        text(size: 32pt, font: "DejaVu Sans Mono")[min_sm],           text(size: 32pt)[Minimum CUDA compute capability],
        text(size: 32pt, font: "DejaVu Sans Mono")[min_gfx],          text(size: 32pt)[Minimum AMD GFX version],
        text(size: 32pt, font: "DejaVu Sans Mono")[requires_features], text(size: 32pt)[Tensor cores, matrix units, etc.],
        text(size: 32pt, font: "DejaVu Sans Mono")[variant_priority],  text(size: 32pt)[Explicit ordering for tie-breaking],
        text(size: 32pt, font: "DejaVu Sans Mono")[variant_tag],       text(size: 32pt)[Human-readable variant name],
      )

      #v(20pt)
      #block(inset: 20pt, radius: 4pt, fill: light-teal, width: 100%)[
        #text(size: 32pt, fill: rgb("#1a5c4a"))[
          These keys are *composable* — a single OffloadBinary can carry variants tagged for different vendors and feature sets. The dispatcher scores each against the live GPU context.
        ]
      ]
    ]

    #v(24pt)

    // Card 7: C3 — MLIR Attribute
    #card(accent: navy)[
      = #text(fill: navy)[C3: MLIR Attribute Design]
      #v(16pt)

      #text(size: 32pt)[A new `#gpu.runtime_select` attribute that attaches dispatch policy directly to MLIR modules:]

      #v(16pt)

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

      #v(16pt)
      #text(size: 28pt, fill: rgb("#555"))[
        The attribute is *declarative* — it expresses intent, not mechanism. The runtime implementation is free to evolve.
      ]
    ]

    #v(24pt)

    // Card 8: Latency Breakdown
    #card(accent: orange)[
      = #text(fill: orange)[C2: First Published Dispatch Flame Graph]
      #v(16pt)

      #image("figures/latency-breakdown.svg", width: 100%)

      #v(20pt)

      #grid(
        columns: (1fr, 1fr, 1fr),
        column-gutter: 18pt,
        stat-box("36.0 " + sym.mu + "s", "cuModuleLoadData\n(90% of total)", accent: orange),
        stat-box("3 -- 6 ns", "Selection\noverhead", accent: teal),
        stat-box("< 0.01%", "Dispatch cost\nvs. kernel load", accent: llvm-blue),
      )

      #v(16pt)
      #text(size: 28pt, fill: rgb("#555"))[
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
      #v(16pt)

      #table(
        columns: (1fr, auto, auto),
        fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
        table.header(
          text(weight: "bold", size: 30pt)[*Operation*],
          text(weight: "bold", size: 30pt)[*Latency*],
          text(weight: "bold", size: 30pt)[*Share*],
        ),
        text(size: 30pt)[cuModuleLoadData (cold)],  text(size: 30pt)[36.0 #sym.mu\s],  text(size: 30pt, weight: "bold", fill: orange)[90.0%],
        text(size: 30pt)[cuModuleLoadData (warm)],   text(size: 30pt)[1.2 #sym.mu\s],   text(size: 30pt)[3.0%],
        text(size: 30pt)[cuModuleGetFunction],        text(size: 30pt)[0.8 #sym.mu\s],   text(size: 30pt)[2.0%],
        text(size: 30pt)[cuLaunchKernel],             text(size: 30pt)[1.5 #sym.mu\s],   text(size: 30pt)[3.8%],
        text(size: 30pt)[cuStreamSynchronize],        text(size: 30pt)[0.5 #sym.mu\s],   text(size: 30pt)[1.2%],
        text(size: 30pt, weight: "bold")[Hot-path total], text(size: 30pt, weight: "bold")[40.0 #sym.mu\s], text(size: 30pt, weight: "bold")[100%],
        text(size: 30pt, fill: teal, weight: "bold")[Selection overhead], text(size: 30pt, fill: teal, weight: "bold")[3 -- 6 ns], text(size: 30pt, fill: teal, weight: "bold")[< 0.01%],
      )
    ]

    #v(24pt)

    // Card 10: Variant Scaling
    #card(accent: teal)[
      = #text(fill: teal)[Selection Scales Linearly]
      #v(16pt)

      #image("figures/variant-scaling.svg", width: 100%)

      #v(14pt)
      #text(size: 26pt, fill: rgb("#666"))[
        Even at 64 variants, selection stays under 400ns — three orders of magnitude below driver overhead.
      ]
    ]

    #v(24pt)

    // Card 11: Overhead Comparison
    #card(accent: orange)[
      = #text(fill: orange)[Overhead Comparison]
      #v(16pt)

      #image("figures/overhead-comparison.svg", width: 100%)

      #v(14pt)
      #text(size: 26pt, fill: rgb("#666"))[
        Our dispatch adds negligible overhead compared to existing runtime costs — selection is noise-level relative to module loading.
      ]
    ]

    #v(24pt)

    // Card 12: Key Findings
    #card(accent: llvm-blue)[
      = #text(fill: llvm-blue)[Key Findings]
      #v(16pt)

      #grid(
        columns: (1fr,),
        row-gutter: 18pt,
        block(inset: 16pt, radius: 4pt, fill: light-blue, width: 100%)[
          #text(size: 32pt)[
            *F1:* Module loading dominates dispatch latency at 90% — selection policy is essentially free.
          ]
        ],
        block(inset: 16pt, radius: 4pt, fill: light-teal, width: 100%)[
          #text(size: 32pt)[
            *F2:* Five metadata keys suffice to express all observed vendor-selection patterns in the wild.
          ]
        ],
        block(inset: 16pt, radius: 4pt, fill: light-orange, width: 100%)[
          #text(size: 32pt)[
            *F3:* Linear variant scaling means the approach works from 2 to 64+ variants without architectural changes.
          ]
        ],
        block(inset: 16pt, radius: 4pt, fill: rgb("#f0e8f5"), width: 100%)[
          #text(size: 32pt)[
            *F4:* A declarative MLIR attribute (`#gpu.runtime_select`) cleanly separates policy from mechanism.
          ]
        ],
      )
    ]

    #v(24pt)

    // Card 13: Prototype
    #card(accent: navy)[
      = #text(fill: navy)[Prototype Implementation]
      #v(16pt)

      #grid(
        columns: (1fr, 1fr),
        column-gutter: 18pt,
        stat-box("5,100", "LOC — libkdl\n(dispatch library)", accent: navy),
        stat-box("664", "LOC — PoC\n(MLIR integration)", accent: teal),
      )

      #v(24pt)
      #text(size: 32pt)[
        *libkdl* (Kernel Dynamic Linker) is a standalone C library that implements the dispatch algorithm. The PoC wires it into the MLIR GPU runtime via `gpu.launch_func` lowering.
      ]
    ]
  ],
)

#v(54pt)

// ═══════════════════════════════════════════════════════════════════
//  BOTTOM SECTION — Related Work + Upstream Path
// ═══════════════════════════════════════════════════════════════════

#grid(
  columns: (1.5fr, 1fr),
  column-gutter: 24pt,

  // Related Work Comparison
  card(accent: navy)[
    = #text(fill: navy)[Related Work Comparison]
    #v(16pt)

    #table(
      columns: (1.2fr, auto, auto, auto, auto, auto),
      fill: (col, row) => if row == 0 { navy.lighten(85%) } else if calc.odd(row) { luma(248) } else { white },
      table.header(
        text(weight: "bold", size: 30pt)[*System*],
        text(weight: "bold", size: 30pt)[*Multi-vendor*],
        text(weight: "bold", size: 30pt)[*Metadata*],
        text(weight: "bold", size: 30pt)[*Policy*],
        text(weight: "bold", size: 30pt)[*Measured*],
        text(weight: "bold", size: 30pt)[*Upstream*],
      ),
      text(size: 30pt)[IREE HAL],      text(size: 32pt, fill: teal)[#sym.checkmark], text(size: 32pt, fill: teal)[#sym.checkmark], text(size: 32pt, fill: orange)[#sym.tilde],    text(size: 32pt, fill: orange)[#sym.times], text(size: 32pt, fill: orange)[#sym.times],
      text(size: 30pt)[chipStar],       text(size: 32pt, fill: teal)[#sym.checkmark], text(size: 32pt, fill: orange)[#sym.times],    text(size: 32pt, fill: orange)[#sym.times],    text(size: 32pt, fill: orange)[#sym.times], text(size: 32pt, fill: orange)[#sym.times],
      text(size: 30pt)[Proteus],        text(size: 32pt, fill: orange)[#sym.times],   text(size: 32pt, fill: orange)[#sym.times],    text(size: 32pt, fill: teal)[#sym.checkmark],  text(size: 32pt, fill: teal)[#sym.checkmark], text(size: 32pt, fill: orange)[#sym.times],
      text(size: 30pt)[liboffload],     text(size: 32pt, fill: teal)[#sym.checkmark], text(size: 32pt, fill: orange)[#sym.tilde],    text(size: 32pt, fill: orange)[#sym.times],    text(size: 32pt, fill: orange)[#sym.times], text(size: 32pt, fill: teal)[#sym.checkmark],
      text(size: 30pt)[CPU FMV],        text(size: 32pt, fill: orange)[#sym.times],   text(size: 32pt, fill: teal)[#sym.checkmark],  text(size: 32pt, fill: teal)[#sym.checkmark],  text(size: 32pt, fill: orange)[#sym.times], text(size: 32pt, fill: teal)[#sym.checkmark],
      text(size: 30pt, weight: "bold", fill: llvm-blue)[This Work], text(size: 32pt, fill: teal, weight: "bold")[#sym.checkmark], text(size: 32pt, fill: teal, weight: "bold")[#sym.checkmark], text(size: 32pt, fill: teal, weight: "bold")[#sym.checkmark], text(size: 32pt, fill: teal, weight: "bold")[#sym.checkmark], text(size: 32pt, fill: teal, weight: "bold")[#sym.checkmark],
    )

    #v(16pt)
    #text(size: 28pt, fill: rgb("#555"))[
      This work is the first to combine all five properties: multi-vendor support, structured metadata, pluggable dispatch policy, empirical measurement, and an upstream-ready design.
    ]
  ],

  // Upstream Path
  card(accent: teal)[
    = #text(fill: teal)[Upstream Path]
    #v(20pt)

    #let step-box(num, title, desc, accent: teal) = {
      block(width: 100%, inset: 20pt, radius: 4pt, fill: accent.lighten(90%), stroke: 1.5pt + accent.lighten(50%))[
        #grid(
          columns: (auto, 1fr),
          column-gutter: 20pt,
          align(center + horizon)[
            #box(
              width: 60pt, height: 60pt,
              radius: 30pt,
              fill: accent,
              align(center + horizon)[
                #text(size: 36pt, weight: "bold", fill: white)[#num]
              ]
            )
          ],
          [
            #text(size: 36pt, weight: "bold", fill: navy)[#title]\
            #text(size: 28pt, fill: rgb("#444"))[#desc]
          ],
        )
      ]
    }

    #step-box("1", "Metadata RFC",
      "Propose 5 keys for OffloadBinary. Low-risk, additive change.",
      accent: teal)
    #v(20pt)
    #step-box("2", "liboffload Policy Slot",
      "Add pluggable selection hook to liboffload's existing load path.",
      accent: llvm-blue)
    #v(20pt)
    #step-box("3", "MLIR Attribute + Lowering",
      "Wire #gpu.runtime_select through to the runtime via gpu.launch_func.",
      accent: orange)

    #v(24pt)

    #block(inset: 20pt, radius: 4pt, fill: light-teal, width: 100%)[
      #text(size: 32pt, fill: rgb("#1a5c4a"))[
        Each step is independently useful. Step 1 unlocks dispatch for any downstream consumer of OffloadBinary.
      ]
    ]
  ],
)

#v(70pt)

// ═══════════════════════════════════════════════════════════════════
//  FOOTER
// ═══════════════════════════════════════════════════════════════════

#block(
  width: 100%,
  inset: (x: 36pt, y: 44pt),
  radius: 6pt,
  fill: navy.lighten(92%),
  stroke: 1pt + navy.lighten(70%),
)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    align(left)[
      #text(size: 30pt, fill: navy)[
        *Code:* github.com/AKASHAorg/libkdl\
        *Contact:* sakash\@iitp.ac.in
      ]
    ],
    align(center)[
      #text(size: 30pt, fill: navy)[
        *EuroLLVM Developers' Meeting — Dublin 2026*
      ]
    ],
    align(right)[
      #text(size: 30pt, fill: navy)[
        *Prototype:* 5,100 + 664 LOC\
        *License:* Apache 2.0 with LLVM Exception
      ]
    ],
  )
]
