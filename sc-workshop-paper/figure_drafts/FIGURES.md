# Figure drafts — manifest and provenance

Regenerate all: `python3 scripts/figures/make_figures.py`
One figure: `python3 scripts/figures/make_figures.py --only sgb-spread`

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
| 1 | `fig_intro_behavior` | `fig:intro-behavior` | **X** | none — conceptual |
| 2 | *(you have this)* | `fig:predictability` | — | not generated |
| 3 | `fig_replacement_loss` | `fig:replacement-loss` | **M**+**X** | model timeline measured; artifact series illustrative — see caveats |
| 4 | `fig_sgb_spread` | `fig:sgb-spread` | **M** | `results/bench_format_activation_atl1-1-02-003-25-1.json.csv` |
| 5 | `fig_scale_sweep` | `fig:scale-sweep` | **S** | `results_tables/05_scale_sweep.md` |
| 6 | `fig_topology_budget` | `fig:topology-budget` | **S** | `results_tables/02_budget_sweep.md` |
| 7 | `fig_tool_relationships` | `fig:tool-relationships` | **M** | `runtime/predictor/data/learned_transitions.json` |
| 8 | *(you have this)* | `fig:plan-accuracy` | — | not generated |
| 9 | `fig_budget_sweep` | `fig:budget-sweep` | **S** | `results_tables/02_budget_sweep.md` |
| 10 | `fig_stall_ladder` | `fig:stall-ladder` | **S** | `results_tables/02_budget_sweep.md` |
| 11 | `fig_compute_sweep` | `fig:compute-sweep` | **S** | `results_tables/03_compute_sweep.md` |
| 12 | `fig_ablation` | `fig:ablation` | **S** | `results_tables/01_attribution_ladder.md` |
| 13 | `fig_prefetch_variants` | `fig:prefetch-variants` | **S** | `results_tables/06_prefetch_variants.md` |
| 14 | `fig_cpu_interference` | `fig:cpu-interference` | **M** | `results/bench_preactivation_interference.json` |
| 15 | `fig_h_sweep` | `fig:h-sweep` | **S** | `results_tables/07_objective_check.md` |
| 16 | `fig_accuracy_sweep` | `fig:accuracy-sweep` | **S** | `results_tables/04_accuracy_sweep.md` |

`fig:budget-sweep` is the only `figure*` (full width, two panels). Everything
else is single-column at 3.35 in.

---

## Caveats that need your decision

**`fig_tool_relationships` contradicts the prose in two ways.** The shipped
transition table carries **offsets 1, 2 and 3 only** — the predictor's horizon
is hardcoded to a short window — so the claim of *k* ∈ [0,5] cannot be drawn
from this artifact. And **17 of 26 tools** have at least one successor above
0.9 confidence, which is 65%, not the "nearly every tool" the paragraph
claims. The figure plots what exists. Either regenerate the table over a wider
offset range, or soften both claims to match.

**`fig_replacement_loss` has one series that is not measured.** The model
timeline is real — the seven load windows of a representative trial. The
artifact-residency panel beneath it is *constructed* to show the mechanism,
because no page-cache-versus-time trace was ever collected. Either instrument
a run to capture artifact residency, or relabel the figure as a schematic.

**`fig_stall_ladder` assumes a compute share.** Stall is derived as
`wall − compute` using the 10.8% compute share from the window-0.1 row, held
fixed across both configurations. If you keep this figure, the assumption
belongs in its caption.

---

## Theme

**Type.** Times New Roman is not installed on this machine. The chain is
`Times New Roman → Nimbus Roman → Liberation Serif → DejaVu Serif`; the two
fallbacks are metric-compatible Times clones, so a machine with the real face
picks it up and nothing else moves.

**Colour.** Gruvbox, biased to the less-saturated faded/neutral families.
Two seeds could not be used as published:

- **Gruvbox blue and aqua sit at chroma 0.066–0.082**, below the 0.10 floor at
  which a hue stops reading as a hue and starts reading as gray. The blue is
  snapped to the nearest in-gamut step at the *same* Gruvbox hue angle
  (215.8°): `#076678 → #008da5`.
- **Orange is dropped.** Gruvbox orange against Gruvbox green measures ΔE 2.4
  under protan/deutan simulation — effectively identical to a red-green
  colourblind reader.

Fixed assignment order, never cycled:

| slot | hex | hue |
|---|---|---|
| 1 | `#008da5` | blue (snapped) |
| 2 | `#9c0006` | faded red |
| 3 | `#b67717` | faded yellow |
| 4 | `#8f3f71` | faded purple |
| 5 | `#79740e` | faded green |

Validated at every prefix by `scripts/validate_palette.py` against a white
surface — worst-case colourblind separation ΔE 20.9 / 17.4 / 18.2 / 14.3 for
n = 2/3/4/5, against a target of 8 and a normal-vision floor of 15. All pass.
Marker shape carries identity alongside hue, so nothing depends on colour alone.

**Everything that is not data** — text, axes, ticks, frame — is black on white.
Gridlines are solid hairlines (never dashed), top and right spines removed.

**No dual-axis plots.** Three figures originally specified a second y-scale
(`budget-sweep`, `cpu-interference`, `scale-sweep`); each is now stacked or
side-by-side panels sharing one axis, because two scales on one frame invent a
correlation the data does not contain. The `.tex` figure specs still describe
the twin-axis version and should be updated to match.

**`fig_sgb_spread` is dots, not bars.** On a log axis a bar's length is not
proportional to its value, which would misstate the 65× spread the figure
exists to show.
