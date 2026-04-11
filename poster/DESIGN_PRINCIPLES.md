# Academic Poster Design Principles

Compiled from Paper2Poster (NeurIPS 2025), posterskill, fillerbuster-poster,
UC Davis poster design guide, ConceptViz, Bolei Zhou's poster collection,
and top-tier conference poster practices (NeurIPS, SIGGRAPH, CVPR).

---

## 1. What Makes a Good Academic Poster Visually

**Core principle:** A poster is a *visual summary*, not a printed paper.
It must be understood in under 2 minutes by someone walking past.

- **14x compression:** Paper2Poster research shows effective posters compress
  ~20,000 tokens of paper text down to ~1,400 tokens (14x). Figures reduce from
  ~23 to ~9. Every element must earn its space.
- **Visual-semantic communication:** Human-designed posters convey meaning
  predominantly through visuals, not dense text. Engagement is the primary
  aesthetic bottleneck across all evaluation dimensions.
- **Hierarchy of attention:** Title draws from 5m away, figures from 2m,
  key finding from 1m, details only when the viewer is standing in front of you.
- **One core message:** The poster should hammer a single takeaway. Everything
  else supports it. If someone walks away remembering one thing, what should it be?

**Quality criteria (from Paper2Poster evaluation):**
1. Clarity -- information presented accessibly
2. Content completeness -- core insights included within constraints
3. Logical flow -- reading order supports understanding
4. Engagement -- visual design attracts and maintains attention
5. Element quality -- figures, tables, text boxes are clean and purposeful
6. Layout balance -- spatial weight distributed evenly, no dead zones

---

## 2. Layout Patterns That Work

### Column Arrangements

| Layout | Best For | Notes |
|--------|----------|-------|
| 3-column | Most posters | Left-to-right flow: intro, method, results |
| 4-column | Dense technical work | Our current poster uses this (1:1.15:1.15:1) |
| 2-column | Simple work | Too slow for the eye; rarely used at top venues |
| 3-column asymmetric | Figure-heavy work | Posterskill default: 280mm / flex / 220mm |

**Posterskill/fillerbuster pattern (recommended):**
- Left column (fixed width): introductory/foundational content -- TL;DR, motivation
- Middle column(s) (flexible): visual anchor with larger figures -- method, architecture
- Right column (fixed width): quantitative results, conclusions

This guides the viewer: **problem -> solution -> validation.**

### Card System

The card-based approach (used by posterskill and fillerbuster) is superior to
free-form layout:
- Each section is a self-contained card with white background, rounded corners
  (2.5mm), subtle shadow, and a colored top border (2.5px) indicating its category
- Cards stack vertically within columns with consistent gaps (3mm)
- One card per column must have `grow: true` (flex:1) to fill remaining space
  and prevent dead zones
- Cards can be swapped, moved, and resized independently

### Aspect-Ratio-Aware Placement

This is the single most important layout rule for eliminating whitespace:
- **Wide images** (>2:1 ratio, e.g., architecture diagrams, teasers): put in the
  widest column
- **Square images** (~1:1 ratio): put in narrow columns
- **Portrait images** (<1:1 ratio): put in the narrowest column

Mismatching image aspect ratios to column widths is the #1 source of wasted space.

### Binary-Tree Layout (Paper2Poster)

Paper2Poster uses a binary-tree layout strategy that preserves reading order and
spatial balance. The key insight: recursively divide the poster area and assign
content to regions based on visual weight, ensuring no region is dramatically
over- or under-filled.

---

## 3. Typography Rules

### Font Sizes at A0 (841 x 1189mm landscape)

| Element | A0 Size | A1 Size | Readable From |
|---------|---------|---------|---------------|
| Title | 72-120pt | 42-60pt | 5+ meters |
| Subtitle | 36-48pt | 18-24pt | 3 meters |
| Authors | 24-36pt | 14-18pt | 2 meters |
| Section headers | 36-48pt | 18-24pt | 2 meters |
| Body text | 24-32pt | 11-16pt | 1 meter |
| Captions | 18-24pt | 9-12pt | 0.5 meter |
| References | 16-20pt | 8-10pt | Standing close |

### Our Current Poster (A0 landscape, 1189x841mm)

Our poster.html uses these sizes, which are reasonable for A0:
- Title: 42pt (could go larger -- 60-72pt would improve 5m readability)
- Subtitle: 18pt
- Authors: 14pt
- Section headers: 18pt
- Body/list text: 11pt
- Callout text: 10.5pt

**Concern:** These are on the small end for A0. They work for an HTML poster
viewed on screen, but for physical printing consider scaling up by 1.5-2x,
or use the posterskill `--font-scale` approach (default 1.3x).

### Font Selection

- **Use 1-2 font families maximum.** Pair a sans-serif for body (DM Sans, Nunito,
  Arial, Helvetica) with a serif for titles (Source Serif 4, Georgia) or a
  monospace for code elements (JetBrains Mono).
- **Never use all caps for body text.** All-caps is acceptable only for short
  labels, venue tags, and sub-headings.
- **Weight hierarchy:** 900/800 for title, 700 for section heads, 600 for
  emphasis, 400 for body.
- **Left-align body text.** Never center-align paragraphs or bullet lists.
  Center alignment is only for titles and captions.
- **3-4 font sizes maximum.** More creates visual clutter.

### Posterskill Font Scaling System

All text sizes use `calc(Xpt * var(--font-scale))` so they scale uniformly.
Default `--font-scale: 1.3`. This is the right approach -- define base sizes
relative to the design, then scale globally for the print format.

---

## 4. Color Usage in Top-Tier Conference Posters

### Semantic Color Coding

The fillerbuster poster demonstrates best-in-class color usage:

| Color | Hex | Semantic Role |
|-------|-----|--------------|
| Blue | `#2d7fc1` | Primary/default, headers, title cards |
| Orange | `#e8943a` | Emphasis, best results, accent |
| Teal | `#2aaa8a` | Architecture/technical content |
| Purple | `#7c5cbf` | Qualitative results/samples |
| Red | `#d94f4f` | Conclusions, warnings |

Each card's top border uses its semantic color. This creates instant
visual categorization -- viewers can scan by color to find what they want.

### Color Rules

1. **2-3 main colors plus neutrals.** More than 3 accent colors creates chaos.
2. **High contrast for text.** Dark text on light background (or vice versa for
   dark themes). Minimum contrast ratio 4.5:1 for body text.
3. **Consistent coding throughout.** If blue means "method," blue always means
   method. Never reuse a color for a different semantic category.
4. **White or light neutral backgrounds preferred.** The posterskill template
   uses `#f0ece6` (warm off-white) with white cards. This is the safe choice.
5. **Dark themes are risky.** Our current poster uses dark navy (`#0d1b2a`).
   This can look striking on screen but:
   - Uses more ink when printed (cost + drying time)
   - Harder to read under conference hall fluorescent lighting
   - Text contrast must be carefully managed
   - Figures with white backgrounds will create harsh contrast boxes
6. **Use highlight boxes sparingly.** A light-blue-background callout with a
   left border accent (the `.hl` pattern) draws the eye to key findings.

### Our Current Poster Colors

```
--navy: #0d1b2a        (background)
--accent-blue: #2d9cdb (primary accent)
--accent-teal: #00b4d8 (secondary accent)
--accent-gold: #e8b931 (emphasis)
--text-primary: #e8edf3
--text-secondary: #9fb3c8
```

These work well as a cohesive dark-theme palette. The gold accent for emphasis
and teal for technical elements follow semantic color principles. If sticking
with dark theme, ensure all figures have transparent or dark backgrounds.

---

## 5. How to Make Figures and Diagrams Impactful

### Selection and Placement

- **Results should occupy 40-50% of poster space.** Figures dominate; text
  provides context and interpretation.
- **Reduce paper figures from ~23 to ~9.** Every figure must directly support
  the poster's core message. Cut everything else.
- **Semantic alignment:** Each figure must be visually relevant to its adjacent
  text. Paper2Poster computes figure-text alignment scores -- if you cannot
  explain why a figure is next to a text block, remove it.

### Quality Requirements

- **300+ DPI minimum** at final print dimensions. Low-resolution images pixelate
  catastrophically at A0.
- **Vector graphics preferred** for diagrams, architecture figures, and charts.
  SVG or high-res PNG from vector source.
- **Consistent visual style** across all diagrams. Same line weights, same color
  palette, same font within figures.
- **Rounded corners** (1.5mm) on all images for a polished look (posterskill
  convention).

### Figure Container Rules (from posterskill)

```css
/* CRITICAL: width/height 100% + object-fit:contain for zero whitespace */
.fig-wrap img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
```

Never use `max-width` / `max-height` -- those prevent upscaling and leave gaps.
The figure container should flex to fill available card space.

### Architecture Diagrams

For our MLIR dispatch poster specifically:
- The dispatch pipeline diagram is the hero figure -- it should be the largest
  visual element, placed in the widest column
- Use color coding consistent with the poster palette (blue for MLIR passes,
  teal for runtime layer, gold for hardware targets)
- Show the data flow clearly: MLIR dialect -> lowering -> dispatch -> hardware
- Keep it horizontal (wide format) to match landscape poster orientation

### Comparison Tables

- Colored header rows (use primary color)
- Highlight best results with bold + accent color (`.best` class)
- Alternate row backgrounds for readability (`#f8f9fb` on even rows)
- Keep to 4-6 columns maximum -- more than that is unreadable at poster scale

---

## 6. Common Mistakes to Avoid

### Content Mistakes

1. **Too much text.** Aim for 300-600 words on A0. Our poster should be
   under 800 words absolute maximum. Bullet points, not paragraphs.
2. **Including everything from the paper.** The poster is not the paper.
   Cut ruthlessly. If it does not support the one core message, it goes.
3. **Dense paragraphs.** If any text block is more than 3 lines, it needs
   to become bullets or be cut.
4. **Missing contact info.** Include QR code to project page, email, and/or
   GitHub URL.

### Visual Mistakes

5. **Low-resolution figures.** Test by printing a section at actual size.
   If it looks fuzzy, it needs a higher-resolution source.
6. **Inconsistent formatting.** Establish a style guide (this document)
   before designing. Do not freestyle.
7. **No visual hierarchy.** Every element should be at one of 3-4 levels
   of visual importance. If everything is the same size, nothing stands out.
8. **Busy backgrounds.** No photos, gradients, or patterns behind text.
   Solid colors only behind content areas.
9. **Tiny figure captions.** If the caption is too small to read, the figure
   loses context and meaning.

### Layout Mistakes

10. **No whitespace.** Paradoxically, both too much and too little whitespace
    are problems. Consistent gaps (3-5mm between cards) provide breathing room
    without wasting space.
11. **Unclear reading order.** Number sections or use visual flow cues.
    Our poster uses numbered section headers (circled digits) -- good practice.
12. **Mismatched image aspect ratios.** Wide images in narrow columns leave
    huge vertical gaps. Measure ratios and assign to appropriate columns.
13. **Overflowing text.** Text that spills out of its container or requires
    scrolling. Every text block must fit its allocated space.

### Technical Mistakes (HTML posters)

14. **External dependencies that break.** CDN links may go down. For critical
    presentations, inline all CSS/JS or have a PDF backup.
15. **Print CSS not tested.** The `@media print` rules must hide all UI
    elements, set `transform: none`, and match exact poster dimensions.
16. **Forgetting `print-color-adjust: exact`.** Without this, browsers
    strip background colors in print mode.

---

## 7. Specific Recommendations for mlir-hetero-dispatch Poster

### Current State Assessment

Our poster (`poster.html`) is already well-structured:
- 4-column grid layout (1 : 1.15 : 1.15 : 1)
- Dark navy theme with blue/teal/gold accents
- Card-based sections with numbered headers
- Code blocks with syntax highlighting
- Comparison table and architecture diagram placeholders
- QR code placeholder

### Recommended Changes

**Layout:**
- Consider switching to posterskill's 3-column asymmetric layout
  (280mm / flex / 220mm) to give the architecture diagram maximum width
  in the center column. Four columns may be too narrow for A0 readability.
- Ensure the dispatch pipeline architecture diagram is the hero visual --
  place it in the widest column with `grow: true`.

**Typography:**
- Scale up font sizes for A0 print. Current 11pt body text is too small
  for physical viewing at 1m. Target 24-28pt for body, 48-60pt for title.
- Or adopt the posterskill `--font-scale` CSS variable approach and set
  it to 2.0-2.5 for A0 printing.

**Color:**
- The dark theme is a bold choice. It works well for a systems/compiler
  poster at LLVM (differentiation from typical white-background posters).
  But test figure integration -- ensure diagrams do not create harsh white
  rectangles against the dark background.
- Consider making figures with transparent backgrounds, or use dark-themed
  diagram styles matching the poster palette.

**Content (informed by reviewer feedback):**
- Lead with the **concrete contribution** (prototype dispatch layer) --
  reviewers 91A and 91C wanted to see a design, not a survey.
- Include a **"Why dynamic dispatch?"** callout addressing the "ML kernels
  are static" objection (reviewer 91B). Show: model serving with mixed
  hardware, dynamic batching, edge deployment with fallback.
- The comparison table should span SYCL, SPIR-V, HIP, ALPAKA, and our
  approach (reviewer 91D wanted breadth beyond SYCL).
- Show connection to PyTorch/TF compile path (reviewer 91B).
- Acknowledge IREE SPIR-V capabilities honestly (reviewer 91D).

**Figures needed:**
1. **Hero: Dispatch Architecture** -- MLIR dialects -> lowering passes ->
   runtime dispatch layer -> {CUDA, ROCm, CPU}. Wide format, center column.
2. **Comparison Matrix** -- Feature comparison across SYCL, SPIR-V, HIP,
   ALPAKA, IREE, and our approach. Styled table with colored headers.
3. **Benchmark Results** -- Even preliminary micro-benchmarks showing
   dispatch overhead on GEMM across backends.
4. **Integration Diagram** -- Where the dispatch layer sits relative to
   PyTorch/TF -> torch.compile -> MLIR -> our runtime.

**Posterskill Migration:**
- Consider migrating from the current static HTML to posterskill's
  React-based interactive editor. Benefits:
  - Drag-to-resize columns and cards in browser
  - Click-to-swap card positions
  - Global font scaling with A-/A+ buttons
  - Whitespace optimization via Playwright screenshots
  - JSON config export for iterative refinement
- The posterskill template is already cloned at
  `poster/posterskill/skills/make-poster/template.html`

---

## Sources

- [Paper2Poster (NeurIPS 2025)](https://github.com/Paper2Poster/Paper2Poster)
  -- [arXiv paper](https://arxiv.org/html/2505.21497v2)
  -- [project page](https://paper2poster.github.io/)
- [Posterskill](https://github.com/ethanweber/posterskill) by Ethan Weber
- [Fillerbuster poster](http://ethanweber.me/fillerbuster-poster) -- reference
  implementation of posterskill
- [Bolei Zhou's awesome posters](https://github.com/zhoubolei/bolei_awesome_posters)
  -- CVPR/NeurIPS examples
- [ConceptViz poster guide](https://conceptviz.app/blog/academic-poster-design-complete-guide)
- [UC Davis poster design principles](https://urc.ucdavis.edu/sites/g/files/dgvnsk3561/files/inline-files/General%20Poster%20Design%20Principles%20-%20Handout.pdf)
- [NeurIPS 2024 poster instructions](https://neurips.cc/Conferences/2024/PosterInstructions)
- [UCLA poster guide](https://guides.library.ucla.edu/c.php?g=223540&p=1480858)
- [NYU poster basics](https://guides.nyu.edu/posters)
