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
| 6 | `fig-topology-budget` | `fig:topology-budget` | **S** | `results_tables/02_budget_sweep.md` |
| 7 | `fig-tool-relationships` | `fig:tool-relationships` | **M** | `runtime/predictor/data/learned_transitions.json` |
| — | `fig-tool-relationships-detail` | *(not referenced)* | **M** | per-tool heatmap, kept but unused — see caveats |
| 8 | *(you have this)* | `fig:plan-accuracy` | — | not generated |
| 9 | `fig-budget-sweep` | `fig:budget-sweep` | **S** | `results_tables/02_budget_sweep.md` |
| 10 | `fig-stall-ladder` | `fig:stall-ladder` | **S** | `results_tables/02_budget_sweep.md` |
| 11 | `fig-compute-sweep` | `fig:compute-sweep` | **S** | `results_tables/03_compute_sweep.md` |
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

**Colour: stock Gruvbox, unmodified hex**, biased to the less-saturated
"faded" family because the surface is white. Assigned in this fixed order,
never cycled:

| slot | hex | hue | used by |
|---|---|---|---|
| 1 | `#076678` | faded blue | every figure |
| 2 | `#9d0006` | faded red | every keyed comparison |
| 3 | `#b57614` | faded yellow | third series / staging bars |
| 4 | `#8f3f71` | faded purple | `scale-sweep` lower panel only |
| 5 | `#79740e` | faded green | unused |
| 6 | `#427b58` | faded aqua | unused |
| 7 | `#af3a03` | faded orange | unused |

An earlier revision substituted two of these for accessibility. Per request
the published values are now used verbatim, so the two costs that
substitution was paying for are stated here instead of hidden:

- **Blue `#076678` has OKLCH chroma 0.082**, below the 0.10 floor at which a
  hue stops reading as a hue. It separates from everything else fine; it just
  reads desaturated rather than blue as such.
- **Two pairs collide for red-green colourblind readers**: yellow vs green
  (dE 3.9) and blue vs purple (dE 4.0), against a target of 8. Neither pair
  is ever compared in these figures -- green is unused, and purple appears
  only in `scale-sweep`'s lower panel, which is a separate axes with no key.

The set that actually shares a frame is slots 1-3, and it is clean:

    n=2  worst CVD dE 14.8   normal 26.2
    n=3  worst CVD dE 14.8   normal 21.7

against a target of 8 and a normal-vision floor of 15. Marker shape is
assigned in the same order, so identity never rests on hue alone.
Re-check with `python3 scripts/validate_palette.py "<hexes>" --surface "#ffffff" --pairs all`.

**Do not add a 4th keyed series to one frame without re-validating**, and do
not use green and orange together.

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

**`fig-sgb-spread` is dots, not bars.** On a log axis a bar's length is not
proportional to its value, which would misstate the 65x spread the figure
exists to show.
