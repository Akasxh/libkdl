# Design Review R5 -- Visual Critique of poster-combo-a.html

**Reviewer stance:** Conference poster design critic. Standards: stop passersby at 3 meters, readable at 2 meters, top 10% of conference posters.

**Overall assessment: 7.5/10 -- Above average for academic posters, but several fixable issues prevent it from being a head-turner.**

The poster has solid bones: a coherent LLVM-derived color palette, intentional typography pairing (Charter body + Fira Sans headings), and a well-structured 3-column layout. The flame graph is an excellent visual anchor. But it falls short of "stop and stare" due to specific problems with contrast, hierarchy, density, and print fidelity.

---

## Issue 1: Title is undersized for 3-meter readability

**What is wrong.** The h1 is `34pt` on an A0 poster (841mm x 1189mm). At A0 scale, 34pt renders at roughly 12mm cap height. The LLVM conference hall has typical poster viewing distances of 2--3 meters. At 3 meters, 12mm cap height subtends ~0.23 degrees of visual angle -- below the threshold for comfortable reading (~0.3 degrees minimum for headline text).

**Why it matters.** If someone cannot read the title from across the aisle, the poster loses its single most important job: attracting foot traffic. The title is the billboard. Everything else is secondary.

**CSS fix:**
```css
/* Before */
.header-text h1 {
  font-size: 34pt;
  letter-spacing: -0.3pt;
}

/* After -- 44pt minimum for A0 3-meter readability */
.header-text h1 {
  font-size: 44pt;
  letter-spacing: -0.5pt;
  line-height: 1.05;
}
```

---

## Issue 2: Author line contrast fails WCAG AA against the gradient header

**What is wrong.** `.header-authors` uses `color: rgba(255,255,255,0.72)` on a gradient background that ranges from `#0c2d48` (dark) to `#3a9dd9` (medium blue). On the lighter right end of the gradient, white at 72% opacity produces a contrast ratio of approximately 2.8:1. WCAG AA requires 4.5:1 for normal text. Even for large text (which the 13pt author line is not), the minimum is 3:1.

**Why it matters.** Author name and affiliations are what conference attendees scan to decide "is this person from my subfield." If they cannot read it on the lighter portion of the gradient, they walk past.

**CSS fix:**
```css
/* Before */
.header-authors {
  color: rgba(255,255,255,0.72);
}

/* After -- pass WCAG AA across the full gradient range */
.header-authors {
  color: rgba(255,255,255,0.92);
  text-shadow: 0 0.5px 2px rgba(0,0,0,0.3);
}
```

---

## Issue 3: Thesis strip stat labels are illegible (`rgba(255,255,255,0.55)` on `#0f3d62`)

**What is wrong.** `.thesis-stat .lbl` is set to `rgba(255,255,255,0.55)` on `var(--blue-deep)` which is `#0f3d62`. Computed contrast ratio: approximately 3.2:1. This fails WCAG AA for the 9pt text size. The big teal numbers ("6 ns", "5", "0%") are fine, but their labels ("Selection overhead", "New metadata keys", "Hot-path penalty") are the interpretive keys -- without them, the numbers are meaningless.

**Why it matters.** The thesis strip is the second thing viewers see after the title. If the stats read as "6 ns [illegible]", the poster loses its punchline.

**CSS fix:**
```css
/* Before */
.thesis-stat .lbl {
  color: rgba(255,255,255,0.55);
  font-size: 9pt;
}

/* After */
.thesis-stat .lbl {
  color: rgba(255,255,255,0.78);
  font-size: 9.5pt;
  font-weight: 600;
}
```

---

## Issue 4: Flame graph small bars are unreadable -- the "Selection: 6 ns" bar is the poster's key claim, yet visually invisible

**What is wrong.** The selection bar is `width: 2.5%; min-width: 18mm` with text at `font-size: 7.5pt`. On screen preview this may render, but at A0 print, 7.5pt on a narrow green bar is the visual equivalent of fine print. The entire poster argues that selection cost is negligible, and the flame graph is supposed to demonstrate this visually. But the bar that proves the point is the hardest thing to read on the entire poster.

**Why it matters.** This is the single most important data point. The flame graph should make people say "that green sliver is nothing!" but instead they have to squint. The visual rhetoric is undermined.

**CSS fix:**
```css
/* Add an annotation callout that pops out of the flame graph */
.flame-bar.sel {
  min-width: 22mm;
  position: relative;
  overflow: visible;  /* allow callout to escape */
}

.flame-bar.sel::after {
  content: "Selection: 6 ns (0.01%)";
  position: absolute;
  left: 100%;
  top: 50%;
  transform: translateY(-50%);
  margin-left: 3mm;
  font-family: "Fira Sans", sans-serif;
  font-size: 10pt;
  font-weight: 800;
  color: var(--teal-dark);
  white-space: nowrap;
  background: var(--teal-light);
  padding: 1mm 2.5mm;
  border-radius: 1.5mm;
  border: 1.5px solid var(--teal);
}
```

---

## Issue 5: Card body text at 11pt is too small for a poster -- comfortable poster reading is 24pt+ at arm's length

**What is wrong.** `.card p, .card li` at `11pt` and `.card h2` at `13.5pt`. These sizes are appropriate for a journal paper, not a poster. Conference poster best practice calls for body text at 24--28pt and section headings at 36--44pt at A0. While this poster is denser than a typical poster (which is a deliberate choice), the text crosses from "information-dense" into "requires reading glasses."

**Why it matters.** Poster sessions last 2 hours. Attendees spend 30--90 seconds per poster. If they cannot skim the bullet points from a comfortable standing distance (~1 meter), they move on before reaching the key findings.

**CSS fix:**
```css
/* Bump body text to readable poster scale */
.card p, .card li {
  font-size: 12pt;       /* was 11pt -- still dense, but scannable */
  line-height: 1.38;
}

.card h2 {
  font-size: 15pt;       /* was 13.5pt */
  margin-bottom: 3mm;
}

.card h3 {
  font-size: 13pt;       /* was 11.5pt */
}
```

Note: increasing text size will require tightening content. Cut the "ld.so analogy" paragraph in Key Findings (point 4) -- it restates point 3. This recovers the vertical space.

---

## Issue 6: QR code placeholders are too small and lack visual weight

**What is wrong.** `.qr-placeholder` is `28mm x 28mm`. At A0 poster scale, scanned from 1.5 meters, this is tiny. Standard recommendation for conference poster QR codes is 40--50mm minimum. Additionally, the placeholders are plain white boxes with gray text -- they have no visual hierarchy or call to action. The bottom-right corner (where they sit) is the last place eyes travel on a 3-column poster.

**Why it matters.** QR codes are the conversion mechanism. If someone is interested enough to walk up, the QR code converts that interest into a follow-up. Small, low-contrast QR codes in the corner get zero scans.

**CSS fix:**
```css
/* Before */
.qr-placeholder {
  width: 28mm;
  height: 28mm;
  background: #fff;
  border-radius: 2mm;
  font-size: 8pt;
  color: #999;
}

/* After -- larger, with visual call to action */
.qr-placeholder {
  width: 38mm;
  height: 38mm;
  background: #fff;
  border-radius: 3mm;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-family: "Fira Sans", sans-serif;
  font-size: 9pt;
  font-weight: 700;
  color: var(--blue-deep);
  text-align: center;
  line-height: 1.3;
  border: 2px solid var(--teal);
  box-shadow: 0 0 0 2mm var(--teal-light);
}
```

---

## Issue 7: The code blocks compete for visual attention with the flame graph

**What is wrong.** There are three dark-background code blocks in the center column alone (`isMetadataCompatible()`, `#gpu.runtime_select`, and indirectly the architecture diagram on dark). Combined with the flame graph's warm colors in the same column, the center column has four distinct "look at me" visual elements fighting for primacy. The eye bounces between them without settling.

**Why it matters.** The center column is the contribution column -- it is the most important real estate on the poster. If the viewer's eye cannot settle on a single visual anchor, they get overwhelmed and default to reading the title + thesis strip only.

**CSS fix:**
```css
/* Tone down the code blocks -- make them supporting, not competing */
pre {
  background: #1a2332;    /* was --code-bg (#0c1b2a) -- slightly lighter */
  padding: 2.5mm 3mm;     /* was 3mm 3.5mm -- tighter */
  font-size: 9pt;         /* was 9.5pt -- slightly smaller */
  border: 1px solid rgba(255,255,255,0.06);
}

/* Make the flame graph the unambiguous visual anchor */
.flame-graph {
  margin: 3mm 0;          /* was 2mm -- more breathing room */
}

.flame-row {
  height: 10.5mm;         /* was 9mm -- taller bars are easier to read */
}
```

---

## Issue 8: The warm cream background (`#fdf8f0`) will print noticeably yellow on most large-format printers

**What is wrong.** `--bg: #fdf8f0` is a warm cream. On screen, it reads as subtle warmth. On a large-format inkjet (the standard for A0 conference posters), this will print with a visible yellow-beige cast across the entire 841mm x 1189mm sheet. Large-format color profiles are not calibrated for subtle warm whites. The ICC profile mismatch typically amplifies warm tints by 15--25%.

**Why it matters.** The poster will look dirty or aged compared to posters printed on pure white stock. The white cards (`#ffffff`) will pop against the background in a way that looks like a color management error rather than an intentional design choice.

**CSS fix:**
```css
/* Before */
:root {
  --bg: #fdf8f0;
}

/* After -- cool neutral that prints clean */
:root {
  --bg: #f7f7f5;
}
```

Alternatively, if the warm tone is intentional, shift the card color to match:
```css
:root {
  --bg: #fdf8f0;
  --card: #fffefa;    /* was #ffffff -- reduces the contrast gap */
}
```

---

## Issue 9: Bottom bar "Related Work" table is 9pt white-on-dark -- the single densest element on the poster

**What is wrong.** The bottom bar table uses `font-size: 9pt` for body cells and `8.5pt` for headers, white text on `#0f3d62`. At A0, this is approximately 3.2mm cap height. The table has 6 columns and 6 rows of data with colored status dots. This is the densest information element on the entire poster, placed at the bottom where it gets the least viewing time.

**Why it matters.** The comparison table is a strong selling point -- it shows this work fills a gap no other system addresses. But at its current size and density, only someone who bends down to read the bottom of an A0 poster will absorb it. It functions as decoration rather than communication.

**CSS fix:**
```css
.bottom-bar table {
  font-size: 10pt;           /* was 9pt */
}

.bottom-bar thead th {
  font-size: 9.5pt;          /* was 8.5pt */
  padding: 1.5mm 2.5mm;      /* was 1.2mm 2mm */
}

.bottom-bar tbody td {
  font-size: 10pt;           /* was 9pt */
  padding: 1.5mm 2.5mm;      /* was 1mm 2mm */
  color: rgba(255,255,255,0.88);  /* was 0.82 */
}
```

---

## Issue 10: No visual entry point hierarchy -- the poster reads as a wall of equally-weighted cards

**What is wrong.** All 10+ cards share the same visual weight: white background, same border-radius, same shadow (`0 0.5mm 2mm rgba(0,0,0,0.06)`), same padding. The only differentiation is the 3px colored top border. When all cards are equal, nothing is primary. The viewer's eye enters at the title, drops to the thesis strip, then... stalls. There is no visual "path" through the content.

**Why it matters.** Great posters guide the eye: title, then one hero visual, then key finding, then supporting evidence, then call to action. This poster has the content for that hierarchy, but the uniform card styling flattens it.

**CSS fix -- Elevate the flame graph card as the hero element:**
```css
/* Give the flame graph card (C2) dominant visual weight */
.card.card-teal:nth-child(2) {   /* C2: Flame Graph card */
  box-shadow: 0 1mm 6mm rgba(42,170,138,0.15);
  border-top-width: 4px;
  padding: 5mm 6mm;
  background: linear-gradient(180deg, var(--teal-light) 0%, var(--card) 15%);
}

/* Subtly de-emphasize supporting cards */
.card.grow {
  box-shadow: none;
  border: 1px solid var(--border);
}
```

---

## Summary: Priority ranking

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| P0 | #1 Title size | Invisible from 3m | 1 line CSS |
| P0 | #4 Flame graph selection bar | Key claim is unreadable | 10 lines CSS |
| P1 | #2 Author contrast | WCAG failure | 2 lines CSS |
| P1 | #3 Thesis stat labels | WCAG failure | 3 lines CSS |
| P1 | #10 Visual hierarchy | No eye-path | 8 lines CSS |
| P2 | #5 Body text size | Requires reading glasses | 4 lines CSS |
| P2 | #6 QR code size | Zero scans at poster session | 5 lines CSS |
| P2 | #9 Bottom bar table size | Comparison table wasted | 4 lines CSS |
| P3 | #7 Code block visual competition | Center column overwhelm | 5 lines CSS |
| P3 | #8 Cream background print fidelity | Yellow cast on print | 1 line CSS |

Fixing P0 + P1 (5 issues, ~25 lines of CSS) moves this from 7.5/10 to 8.5/10.
Fixing all 10 moves it to 9/10.

The content is strong. The structure is sound. The typography pairing is good. This poster needs a visual hierarchy pass and a contrast/sizing pass -- not a redesign.
