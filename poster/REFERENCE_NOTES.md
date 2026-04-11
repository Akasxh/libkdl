# Fillerbuster Poster — Design Reference Notes

Source: http://ethanweber.me/fillerbuster-poster
Repo: https://github.com/ethanweber/fillerbuster-poster

---

## 1. Layout Architecture

**Page dimensions:** A1 landscape — `841mm × 594mm` fixed. Body uses `overflow: hidden` — nothing bleeds or scrolls.

**Outer grid:** `#root` uses `grid-template-rows: auto 1fr` — header takes its natural height, poster body fills the rest.

**Column system:** `.poster` is `display: flex; gap: var(--gap)` (not CSS Grid). Three columns, each `display: flex; flex-direction: column`. Column widths:

- Left column: `301mm` (fixed)
- Middle column: `flex: 1.5` (flexible, ~306mm)
- Right column: `234mm` (fixed)
- Padding: `var(--pad)` = `10mm` on left/right of poster area

**Within each column:** cards stack vertically. Exactly one card per column gets `.grow` (`flex: 1 1 0`), which forces it to fill remaining vertical space. All other cards are `flex: 0 0 auto`. This is the core mechanism for eliminating whitespace gaps.

---

## 2. Spacing System

All spacing is in millimeters for print precision:

```css
--gap: 3mm;   /* gap between all cards, between columns */
--pad: 10mm;  /* left/right padding on poster and header */
```

- Card padding: `3mm 4mm`
- Highlight box padding: `2.5mm 3.5mm`
- Header padding: `6mm var(--pad) 5mm`
- Footer padding: `2mm var(--pad)`

The consistent `--gap` variable means every spatial relationship in the layout is either `3mm` or `10mm`. Nothing is arbitrary.

---

## 3. Typography at Poster Scale

All body text is scaled via a `--font-scale` CSS variable (`1.3` on screen, `2.2` was mentioned as interactive default):

| Element       | Base size                          |
|---------------|------------------------------------|
| H1 (title)    | `34pt` weight 900 (not scaled)     |
| H2 (card)     | `calc(11pt * var(--font-scale))`   |
| Body / li     | `calc(9.5pt * var(--font-scale))`  |
| Caption        | `calc(7.5pt * var(--font-scale))`  |
| Table header  | `calc(8.5pt * var(--font-scale))`  |
| Table cell    | `calc(9pt * var(--font-scale))`    |
| KaTeX eq      | `calc(10pt * var(--font-scale))`   |

The `--font-scale` approach is the key insight: a single variable controls the entire type scale for both screen preview and print output without touching any individual rules.

**Font family:** `'Nunito'` — a rounded sans-serif. Weight 800 for card titles, 900 for the main title, 700 for authors.

**Line height:** `1.3` for body, `1.35` for highlight boxes, `1.2` for captions. Very tight — appropriate for print where leading is measured tightly.

---

## 4. Color Scheme

```css
--blue:        #2d7fc1   /* primary, default card accent */
--blue-light:  #e3f0fa   /* highlight box bg, links bg */
--orange:      #e8943a   /* secondary accent, best-in-table */
--orange-light: #fdf0e0  /* equation bg */
--teal:        #2aaa8a
--purple:      #7c5cbf
--red:         #d94f4f
--text:        #222
--text-light:  #555
--bg:          #f0ece6   /* warm off-white poster background */
--card-bg:     #fff
```

Card accent color is applied via top border only (`border-top: 2.5px solid var(--blue)`), with the card heading matching that color. Color variants are simply extra classes (`.card.orange`, `.card.teal`, etc.) that override `border-top-color` and `h2` color. This keeps structural CSS clean.

**Header gradient:** `linear-gradient(135deg, #1b4f7a, #2d7fc1 50%, #45a0e0)` — deep-to-light blue diagonal. Title text white, author names `rgba(255,255,255,.9)`, affiliations `rgba(255,255,255,.7)`.

**Background:** `#f0ece6` — warm cream, not pure white. Reduces harshness under fluorescent lighting at poster sessions.

---

## 5. Card Design Pattern

```css
.card {
  background: #fff;
  border-radius: 2.5mm;
  padding: 3mm 4mm;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
  border-top: 2.5px solid var(--blue);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 0;
  flex: 0 0 auto;
}
```

Key decisions:
- `min-height: 0` on all flex children — prevents flex items from overflowing their container.
- `overflow: hidden` on cards — images and content cannot escape card boundaries.
- `box-shadow` is deliberately subtle (`0 1px 3px` at 8% opacity) — cards read as distinct but not heavy.
- `border-top` accent is the only decoration — no border on other sides.

---

## 6. Highlight / Callout Box Pattern

```css
.hl {
  background: var(--blue-light);
  border-left: 2.5px solid var(--blue);
  padding: 2.5mm 3.5mm;
  border-radius: 0 1.5mm 1.5mm 0;
  margin-bottom: 1.5mm;
  flex-shrink: 0;
}
```

Left-border callout style (common in academic docs). The `flex-shrink: 0` prevents the callout from being compressed when a `.grow` card is sharing space with fixed content.

---

## 7. Figure Handling — No Whitespace

The `.fig` + `.fig-wrap` pattern is the key to filling vertical space without gaps:

```css
.fig        { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.fig-wrap   { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; }
.fig-wrap img { width: 100%; height: 100%; object-fit: contain; }
```

When a card has `.grow`, the `.fig` inside it also stretches with `flex: 1`. The image uses `object-fit: contain` so it never distorts — it simply fills whatever space the flex chain allocates. This is how figures scale to fill columns without manual height tuning.

---

## 8. Equation Boxes

```css
.eq {
  background: var(--orange-light);
  border: 1px solid var(--orange);
  border-radius: 1.5mm;
  padding: 1.5mm 2.5mm;
  margin: 1.5mm 0;
  text-align: center;
  flex-shrink: 0;
}
```

Orange accent for equations — visually distinct from blue-accented text callouts. KaTeX renders inside.

---

## 9. Table Design

- Full-width, `border-collapse: collapse`
- Thead: solid `var(--blue)` background, white text weight 700
- Even rows: `#f8f9fb` stripe — very subtle
- Best-in-table cell: `.best` class → `font-weight: 800; color: var(--orange)`
- No outer border on the table itself

---

## 10. Print vs Screen Strategy

```css
@media print {
  /* Hide all interactive UI */
  .toolbar, .divider, .swap-handle, .drop-zone { display: none !important; }
  body { position: static; transform: none; box-shadow: none; }
}

@media screen {
  /* Center poster on grey background */
  html { background: #bbb; overflow: hidden; }
  body { position: absolute; box-shadow: 0 4px 30px rgba(0,0,0,.3); border-radius: 4px; }
}
```

The poster HTML is self-contained — it previews in the browser at exact dimensions, and printing strips all chrome. No separate print stylesheet needed beyond hiding UI elements.

`-webkit-print-color-adjust: exact; print-color-adjust: exact;` is set on `html` to prevent browsers from stripping background colors.

---

## 11. QR Code Integration

QR codes appear in two places:
- Header right side: `24mm × 24mm` with white background padding `0.8mm`, `border-radius: 1.5mm`
- Inside content cards (`.links` section): `18mm × 18mm`

The `.links` component bundles QR + text links in a `var(--blue-light)` background box — serves as a footer within a card.

---

## 12. What Makes It Visually Appealing — Summary

1. **Consistent rhythm:** Every gap, padding, and margin traces back to `3mm` or `10mm`. No arbitrary numbers.
2. **Color accent as structure:** The `border-top` color on cards + matching `h2` color creates visual grouping without heavy UI chrome.
3. **Warm background:** `#f0ece6` instead of white softens the overall impression and makes white cards pop.
4. **One grow card per column:** Eliminates all bottom whitespace automatically — the layout always fills the page.
5. **`object-fit: contain` figures in flex chains:** Figures stretch to fill available space without manual sizing.
6. **Header gradient:** Creates strong visual anchor at the top without being garish.
7. **Rounded corners everywhere:** `2.5mm` on cards, `1.5mm` on callouts — consistent softness.
8. **Font weight range:** 700–900 used aggressively. At A1 poster scale, you need that weight contrast to read from 1m away.

---

## 13. Key CSS Patterns to Directly Copy

```css
/* 1. Page setup */
@page { size: 841mm 594mm; margin: 0; }
body { width: 841mm; height: 594mm; overflow: hidden; }

/* 2. Three-column flex layout */
.poster { display: flex; gap: 3mm; padding: 3mm 10mm; flex: 1; min-height: 0; }
.col    { display: flex; flex-direction: column; gap: 3mm; min-height: 0; }

/* 3. Grow card to fill column */
.card.grow { flex: 1 1 0; }

/* 4. Figure that fills remaining space */
.fig      { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.fig-wrap { flex: 1; display: flex; align-items: center; min-height: 0; }
.fig-wrap img { width: 100%; height: 100%; object-fit: contain; }

/* 5. Font scale variable */
:root { --font-scale: 1.3; }
.card h2 { font-size: calc(11pt * var(--font-scale)); }
.card p  { font-size: calc(9.5pt * var(--font-scale)); }

/* 6. Print color preservation */
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

/* 7. Card accent pattern */
.card          { border-top: 2.5px solid var(--blue); }
.card.orange   { border-top-color: var(--orange); }
.card h2       { color: var(--blue); }
.card.orange h2 { color: var(--orange); }

/* 8. Callout / highlight box */
.hl {
  background: var(--blue-light);
  border-left: 2.5px solid var(--blue);
  border-radius: 0 1.5mm 1.5mm 0;
  padding: 2.5mm 3.5mm;
  flex-shrink: 0;
}
```

---

## 15. HTML Skeleton (3-Column Structure)

```html
<body>
  <!-- Header band -->
  <div class="header">
    <div class="header-left">  <!-- title, authors, affiliations -->
    <div class="header-logos"> <!-- institution logos -->
    <div class="header-right"> <!-- venue badge, QR codes -->
  </div>

  <!-- Content area -->
  <div class="poster" id="poster">
    <div class="col" style="width:301mm"> <!-- left -->
      <div class="card"> ... </div>
      <div class="card grow"> ... </div>  <!-- ONE grow per col -->
    </div>

    <div class="col" style="flex:1.5">   <!-- middle, flexible -->
      <div class="card"> ... </div>
      <div class="card grow"> ... </div>
    </div>

    <div class="col" style="width:234mm"> <!-- right -->
      <div class="card"> ... </div>
      <div class="card grow"> ... </div>
    </div>
  </div>

  <!-- Footer band -->
  <div class="footer"> ... </div>
</body>
```

---

## Application Notes for LLVM Poster

- Use the same A1 landscape fixed-dimension approach — browsers can preview it, `wkhtmltopdf` or Chrome headless prints it.
- Three columns with fixed left/right and flexible middle maps well to: (Problem + Motivation) | (Architecture + Results) | (Evaluation + Conclusion).
- The `--font-scale` variable is critical — tune once, affects everything.
- Use `.card.orange` for the "key contribution" card, `.card.teal` for results, default blue for background/motivation.
- Each column needs exactly one `.card.grow` with a figure inside — the figure will auto-size to fill the gap.
- Copy the `.hl` callout pattern for the "why dynamic dispatch matters" thesis statement.
