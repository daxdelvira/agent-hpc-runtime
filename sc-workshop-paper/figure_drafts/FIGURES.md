# Figure drafts — manifest and provenance

Regenerate all: `python3 scripts/figures/make_figures.py`
One figure: `python3 scripts/figures/make_figures.py --only sgb-spread`
Drafts only, no install: add `--no-install`

Each run also **copies the PDFs into `paper_ieee/figures/` and
`paper/figures/`**, which is where the section files resolve
`\includegraphics{figures/NAME}` from. Regenerating updates the paper.

Each figure is written as **PDF** (drop into LaTeX) and **PNG** (quick viewing).
Theme lives in `scripts/figures/theme.py`.

> **Per request, no figure label, axis or annotation says whether its data is
> measured or simulated.** That distinction is recorded *only here*, so the
> drafts read the way the finished paper will. It still has to reach the
> reader somewhere — the evaluation section's Setup paragraph currently does
> that job.

---

## Provenance

**M** = measured on hardware · **S** = simulation over measured constants ·
**X** = schematic, no data

| # | File | LaTeX label | | Source |
|---|---|---|---|---|
| 1 | `fig-intro-behavior` | `fig:intro-behavior` | **X** | none — conceptual |
| 2 | *(you have this)* | `fig:predictability` | — | not generated |
| 3 | `fig-replacement-loss` | `fig:replacement-loss` | **M**+**X** | model timeline measured; artifact series illustrative — see caveats |
| 4 | `fig-sgb-spread` | `fig:sgb-spread` | **M** | `results/bench_format_activation_atl1-1-02-003-25-1.json.csv` |
| 5 | `fig-scale-sweep` | `fig:scale-sweep` | **S** | `results_tables/05_scale_sweep.md` |
| — | `fig-topology-budget` | *(merged away)* | **S** | now the lower row of `fig-budget-sweep`; still generated, unreferenced |
| 7 | `fig-tool-relationships` | `fig:tool-relationships` | **M** | `runtime/predictor/data/learned_transitions.json` |
| — | `fig-tool-relationships-detail` | *(not referenced)* | **M** | per-tool heatmap, kept but unused — see caveats |
| 8 | *(you have this)* | `fig:plan-accuracy` | — | not generated |
| 9 | `fig-budget-sweep` | `fig:budget-sweep` | **S** | `results_tables/02_budget_sweep.md` — MERGED, two rows: reduction over binding |
| 10 | `fig-stall-ladder` | `fig:stall-ladder` | **S** | `results_tables/02_budget_sweep.md` |
| — | `fig-compute-sweep` | *(cut)* | **S** | finding kept as three numbers in prose; still generated, unreferenced |
| 12 | `fig-ablation` | `fig:ablation` | **S** | `results_tables/01_attribution_ladder.md` |
| 13 | `fig-prefetch-variants` | `fig:prefetch-variants` | **S** | `results_tables/06_prefetch_variants.md` |
| 14 | `fig-cpu-interference` | `fig:cpu-interference` | **M** | `results/bench_preactivation_interference.json` |
| 15 | `fig-h-sweep` | `fig:h-sweep` | **S** | `results_tables/07_objective_check.md` |
| 16 | `fig-accuracy-sweep` | `fig:accuracy-sweep` | **S** | `results_tables/04_accuracy_sweep.md` |

`fig:budget-sweep` and `fig:ablation` are `figure*` (full width, two panels
each); `fig:ablation` was promoted from a single-column float, because its
two panels are illegible at 3.35 in. Everything else is single-column.

**Filenames use hyphens, not underscores.** An `_` in a graphics filename is
a live LaTeX hazard and the tex checker flags it; hyphens remove the class of
problem outright.

**`fig-predictability` and `fig-plan-accuracy` are not generated here** --
they are yours. The sections already reference `figures/fig-predictability`
and `figures/fig-plan-accuracy`; drop PDFs with those names into
`paper_ieee/figures/` and they will resolve.

---

## Caveats that need your decision

**`fig-tool-relationships` was redrawn, and the prose corrected to match it.**
The original 26-row heatmap occupied **100.5% of a column** — a full column,
the largest single item in the paper — to support a claim that is about the
*share* of tools with a confident successor rather than about which ones. It
is now the distribution: confidence threshold against share of tools at or
above it, one line per offset. Same data, 26% of the space.

Two prose claims were wrong against this artifact and are now fixed. The
shipped table carries **offsets 1, 2 and 3 only**, so *k* ∈ [0,5] was never
drawable from it; and **17 of 26 tools** (65%) clear 0.9 confidence, which is
not "nearly every tool." If you regenerate the transition table over a wider
offset range, both the figure and the corrected sentence should be revisited.

The heatmap still generates as `fig-tool-relationships-detail` and is not
referenced by the paper; swapping the filename in `03_problem.tex` restores
it.

**`fig-replacement-loss` has one series that is not measured.** The model
timeline is real — the seven load windows of a representative trial. The
artifact-residency panel beneath it is *constructed* to show the mechanism,
because no page-cache-versus-time trace was ever collected. Either instrument
a run to capture artifact residency, or relabel the figure as a schematic.

> Note the panel renders almost entirely empty, and that is not a plotting
> bug: the agent-active gaps between model loads are 13–82 s against a 5288 s
> trial, so under this model the artifact never has time to be rebuilt and
> stay rebuilt. The figure now states the resident share rather than leaving
> a flat band unexplained. It is a strong claim resting on a constructed
> series, which is exactly why it needs the measurement above.

**`fig-budget-sweep` now carries a claim that needs bracketing.** Merging
binding into it as a second row exposed something the separated figures hid:
the recency-ranked/Tandem gap is **widest where binding is zero**. At two
device slots the baseline flattens at 68.4% the moment evictions stop while
the full system climbs to 80.3% — an 11.9-point margin at 720 GB, the largest
in the sweep. The mechanism is coherent (staging needs no eviction, and extra
budget becomes extra slack, so the residual gap is the prefetcher), but it is
a **prefetch** number arriving from a direction that flatters it. The project
reporting rule applies: confirm the `binding == 0` cells against the
oracle-vs-LRU gap at the same budget before this becomes a headline. A TODO to
that effect sits above the float in `04_opportunities.tex`.

**`fig-stall-ladder` assumes a compute share.** Stall is derived as
`wall − compute` using the 10.8% compute share from the window-0.1 row, held
fixed across both configurations. If you keep this figure, the assumption
belongs in its caption.

---

## Theme

**Type.** Times New Roman is not installed on this machine. The chain is
`Times New Roman -> Nimbus Roman -> Liberation Serif -> DejaVu Serif`; the two
fallbacks are metric-compatible Times clones, so a machine with the real face
picks it up and nothing else moves.

**Colour: stock Gruvbox REGULAR (neutral) accents, unmodified hex.** The
standard set, not the darker "faded" variants an earlier revision used.
Assigned in this fixed order, never cycled:

| slot | hex | hue | used by |
|---|---|---|---|
| 1 | `#458588` | blue | every figure |
| 2 | `#cc241d` | red | every keyed comparison |
| 3 | `#d79921` | yellow | third series / staging bars |
| 4 | `#b16286` | purple | `scale-sweep` lower panel only |
| 5 | `#98971a` | green | unused |
| 6 | `#689d6a` | aqua | unused |
| 7 | `#d65d0e` | orange | unused |

Published values are used verbatim, so the costs are stated rather than
absorbed:

- **Blue `#458588` has OKLCH chroma 0.066**, below the 0.10 floor at which a
  hue stops reading as a hue. It separates from everything else fine; it just
  reads desaturated rather than blue as such.
- **Yellow `#d79921` has only 2.48:1 contrast against white**, below the 3:1
  threshold. Filled bars are unaffected — there is enough area. Thin lines and
  small markers are, which is why `lines.linewidth` and `markersize` are set a
  little heavier than they would otherwise be.
- **Two pairs collide for red-green colourblind readers**: blue vs purple
  (slots 1 and 4) and orange vs green (7 and 5). Neither pair shares a frame —
  purple appears only in `scale-sweep`'s lower panel, a separate axes with no
  key, and green/aqua/orange are unused.

The set that actually shares a frame is slots 1–3, and it is clean:

    n=2  worst CVD dE 14.8   normal 26.2
    n=3  worst CVD dE 13.9   normal 23.6

against a target of 8 and a normal-vision floor of 15. Marker shape is
assigned in the same order, so identity never rests on hue alone. Re-check
with `python3 scripts/validate_palette.py "<hexes>" --surface "#ffffff" --pairs all`.

**Do not add a 4th keyed series to one frame without re-validating.**

**Text on a filled mark picks its own colour** via `theme.on()`, from the
fill's luminance — white on the blue, black on the yellow. Hardcoding it broke
the moment the palette changed.

**Everything that is not data** -- text, axes, ticks, frame -- is black on
white. Gridlines are solid hairlines (never dashed), top and right spines
removed.

**No key sits on the data.** Bar charts put the legend in the margin above
the axes (`theme.legend_above`); line charts do the same wherever the panel
has no title. `scale-sweep`'s upper panel lost its title for this reason --
the title duplicated the y label and was what forced the key inside the
frame, where it sat on the oracle line.

**No dual-axis plots.** Three figures originally specified a second y-scale
(`budget-sweep`, `cpu-interference`, `scale-sweep`); each is now stacked or
side-by-side panels sharing one axis, because two scales on one frame invent
a correlation the data does not contain. The `.tex` figure specs, now kept as
comments above each `\includegraphics`, still describe the twin-axis version
and should be updated to match.

## Size and placement

**Height is a two-level knob in `theme.py`.** `HSCALE` (0.85) applies to every
figure; `HSCALE_FOR` multiplies on top of it per figure. Widths are untouched —
a figure narrower than the column just wastes margin.

| tier | value | applies to |
|---|---|---|
| `HSCALE` alone | 0.85 | figures 1–3: the schematic, the predictability plot, the load timeline |
| `TIGHT` | ×0.65 | single-panel data plots |
| `TIGHT_2PANEL` | ×0.80 | `scale-sweep`, `cpu-interference`, `ablation` |

**Two-panel figures cannot take the full reduction, and the reason is the
label, not the data.** Each panel carries its own rotated y label; below about
0.8 that label is taller than its own panel and collides with the neighbouring
one. Shortening the wording does not fix it — "reduction" and "first uses"
still overlapped. `ablation` is in this tier for a different reason: five
categorical rows need the height.

**Axis labels were shortened to fit**, and where that dropped a baseline
qualifier the caption now names it — a bare "reduction (%)" is unreadable
without knowing the denominator. Eight captions were amended for this.

**Floats are declared where they need to LAND, not where they are cited.** A
LaTeX float only ever moves forward, so three rules apply:

1. Each float is declared at the head of its subsection, ahead of the citing
   paragraph. 0 of 17 are declared late.
2. `fig:stall-ladder` and `fig:budget-sweep` are declared **inside the design
   section**, two subsections before the evaluation that cites them. The
   evaluation opens partway down a page whose top is already committed by
   then, so this is the only way they can reach that page's top. Per author
   preference, flipping back one page beats figures piling up pages later.
3. `fig:intro-behavior` is declared after the second introduction paragraph,
   targeting the top of **column 2, page 1** — declaring it earlier put it at
   the top of column 1 and displaced the abstract. **That declaration point is
   a knob**: one paragraph later if it lands in column 1, one earlier if it
   slips to page 2. Re-check it once the abstract is written, since the
   abstract is what fills column 1.

Single-column floats take `[tb]`; page bottoms roughly double the available
slots. `figure*` keeps `[t]` — LaTeX does not accept `b` for double-column
floats, and a `figure*` can never be placed on the page it is declared on,
which is why both are declared early.

**Float counters are raised in `sections/00_macros.tex`.** Stock LaTeX allows
2 top floats and 3 per page and refuses any page over 70% float; a queue this
dense cannot drain under those limits, and the overflow is what gets flushed
to the end of the document.

---

**`fig-sgb-spread` is dots, not bars.** On a log axis a bar's length is not
proportional to its value, which would misstate the 65x spread the figure
exists to show.
