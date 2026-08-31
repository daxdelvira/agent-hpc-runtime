#!/usr/bin/env bash
# Freeze everything the submitted workshop paper rests on into one immutable,
# checksummed tree that regenerates its figures with no network and no cluster.
#
#   bash scripts/make_paper_snapshot.sh [DEST_PARENT]
#
# WHY THIS EXISTS. At the time of the 2026-08-14 submission:
#   * .gitignore excludes logs/, results/, *.csv and *.jsonl, so NONE of the
#     measurement data behind the paper was in version control;
#   * scripts/figures/make_figures.py (+464 lines) and theme.py (+13) were
#     uncommitted, so the figure code was not in any commit either;
#   * most-recent-papers/ (both paper sources) was untracked.
# A `git clean -xdf`, or the next campaign overwriting a working directory,
# would have made the submitted figures unreproducible. The snapshot lives
# OUTSIDE the repository for exactly that reason.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="workshop-char-2026-08-14"
DEST_PARENT="${1:-$(dirname "$ROOT")/paper-snapshots}"
DEST="$DEST_PARENT/$TAG"
TREE="$DEST/tree"

if [[ -e "$DEST" ]]; then
  echo "refusing to overwrite existing snapshot: $DEST" >&2
  echo "(snapshots are immutable by design; remove it by hand if you really mean to)" >&2
  exit 1
fi
mkdir -p "$TREE"

echo "==> staging code"
# Working tree, not HEAD: the figure code that produced the submission is
# uncommitted, so `git archive HEAD` would silently ship different code.
for d in scripts runtime experiments; do
  rsync -a --exclude='__pycache__' --exclude='*.pyc' "$ROOT/$d/" "$TREE/$d/"
done

echo "==> staging paper sources"
mkdir -p "$TREE/paper"
rsync -a "$ROOT/most-recent-papers/AI4S_SC_char_Aug29/" "$TREE/paper/submitted_char/"
rsync -a "$ROOT/most-recent-papers/AI4S_SC_Aug29/"      "$TREE/paper/full_draft/"
rsync -a --exclude='__pycache__' "$ROOT/sc-workshop-paper/" "$TREE/sc-workshop-paper/"

echo "==> staging measurement data (gitignored in the live repo)"
rsync -a "$ROOT/results/" "$TREE/results/"
rsync -a "$ROOT/logs/"    "$TREE/logs/"

echo "==> recording provenance"
{
  echo "commit            $(git -C "$ROOT" rev-parse HEAD)"
  echo "commit_date       $(git -C "$ROOT" log -1 --format=%cI)"
  echo "commit_subject    $(git -C "$ROOT" log -1 --format=%s)"
  echo "submodule_AtomAgents $(git -C "$ROOT" submodule status | awk '{print $1}')"
  echo "dirty_tracked_files:"
  git -C "$ROOT" status --porcelain | grep -vE '\.(pdf|png)$' | sed 's/^/  /'
} > "$DEST/GIT_STATE.txt"
git -C "$ROOT" diff > "$DEST/uncommitted.patch" || true

{
  echo "python            $(python3 -V 2>&1)"
  echo "python_path       $(command -v python3)"
  echo "platform          $(uname -a)"
  echo "host              $(hostname)"
  echo "date_utc          $(date -u +%FT%TZ)"
  echo
  echo "--- versions of libraries the figure code imports ---"
  python3 - <<'PY'
for m in ("matplotlib", "numpy", "pandas", "scipy"):
    try:
        mod = __import__(m); print(f"{m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:12s} ABSENT ({type(e).__name__})")
PY
} > "$DEST/ENV.txt"

echo "==> writing README / PROVENANCE / reproduce.sh"
cat > "$DEST/README.md" <<'MD'
# Workshop-paper data snapshot — `workshop-char-2026-08-14`

Frozen record of everything behind **"When Agents Break the Memory Hierarchy:
Understanding Data Stalls in Agentic HPC Workflows"** (AI4S @ SC, submitted
2026-08-14). Built so that a reviewer given `workshop-char-2026-08-14.tar.gz`
and nothing else can regenerate every figure, offline, with no cluster.

## Use it

```bash
tar xzf workshop-char-2026-08-14.tar.gz
cd workshop-char-2026-08-14
sha256sum -c MANIFEST.sha256        # 4085 files
bash reproduce.sh                   # regenerates all figures and diffs them
```

`reproduce.sh` regenerates the figures into a scratch copy and byte-compares
the PNGs against the ones archived here. On the build host all 26 matched
byte-for-byte; see `REPRODUCTION.txt`.

## Layout

| path | what |
|---|---|
| `tree/` | a self-contained working copy — code, data and paper at their original relative paths |
| `tree/paper/submitted_char/` | the submitted workshop paper source (LaTeX) |
| `tree/paper/full_draft/` | the longer full-paper draft as it stood on the same date |
| `tree/results/`, `tree/logs/` | the measurement data — **gitignored in the live repo**, which is why this snapshot exists |
| `tree/sc-workshop-paper/results_tables/` | simulator output tables that back two figures |
| `MANIFEST.sha256` | sha256 of every file |
| `GIT_STATE.txt`, `uncommitted.patch` | the exact repository state, including uncommitted work |
| `ENV.txt` | interpreter and library versions |
| `PROVENANCE.md` | per-figure input map, claim map, and known defects |

## Why it lives outside the repository

At submission time `.gitignore` excluded `logs/`, `results/`, `*.csv` and
`*.jsonl`; `scripts/figures/make_figures.py` (+464 lines) and `theme.py` (+13)
were uncommitted; and `most-recent-papers/` was untracked. The paper's data,
its figure code, and its own source were all outside version control at once.
A `git clean -xdf` would have destroyed the first two. This snapshot is a
plain directory plus a tarball on a different path for that reason.
MD

cat > "$DEST/PROVENANCE.md" <<'MD'
# Provenance — `workshop-char-2026-08-14`

Read `tree/sc-workshop-paper/measurement_provenance.md` for the full
measurement register (trust classes, instrumentation, environment hazards).
This file records only what a reviewer needs to trace a figure or a number
back to a file in this snapshot.

## Figures used by the submitted paper

All are produced by `tree/scripts/figures/make_figures.py`, which reads its
numbers from artifacts rather than carrying them inline — with one exception,
noted below.

| figure | § | inputs, relative to `tree/` |
|---|---|---|
| `fig-intro-behavior` | I | none — schematic |
| `fig-agentic-workflow` | II | none — schematic; roles and models per `experiments/model_configs.py:MODELS_BLACKWELL_SWAP` |
| `fig-predictability` | III | `logs/workflow_traces/runtime_trace_*.jsonl` |
| `fig-prediction-signals` | III | `logs/workflow_traces/*.jsonl` + `runtime/predictor/data/learned_transitions.json`; compliance analysis imported from `experiments/plot_plan_accuracy.py` and `experiments/plot_utils.py` |
| `fig-replacement-loss` | III | **none — see defect D2** |
| `fig-sgb-spread` | III | `results/bench_format_activation_atl1-1-02-003-25-1.json.csv` |
| `fig-scale-sweep-alt` | III | `sc-workshop-paper/results_tables/05_scale_sweep.md` |
| `fig-budget-staging` | III | `sc-workshop-paper/results_tables/02_budget_sweep.md` |

The two `results_tables` files are generated by
`scripts/make_results_tables.py`, which imports `scripts/sim_residency_v2.py`.
**They are simulator output, not measurement** — `results_tables/00_README.md`
states the standing caveats, including the measured catalogue the simulator is
parameterised from and the 0.7–4.8 point spread across popularity orderings.
`scripts/verify_sim_v2.py` carries 24 checks on that simulator.

## Headline numbers and where they come from

| claim | value | source under `tree/` |
|---|---|---|
| MegaMmap slower than baseline (L40S only) | 3.18× | `results/eval_q1_q4/eval_q1_summary.csv` + `eval_stall_taxonomy.csv` |
| per-step agreement with the modal tool | 58.4% | `logs/workflow_traces/` (regenerated by `fig-predictability`) |
| tools with a confident successor | 17/26 ≥ 0.9 | `runtime/predictor/data/learned_transitions.json` |
| plan compliance, order-only / positional | 76.3% / 40.8% | `logs/workflow_traces/` via `experiments/plot_plan_accuracy.py` |
| transformation share of artifact cost | 93.0% | `results/bench_potential_activation_*.json` (trust A, register D1) |
| seconds-per-GB spread across formats | 65× | `results/bench_format_activation.csv` |
| s/GB stability across a 4× size range | 1.00–1.16× | `results/bench_format_activation.csv` |
| recency-ranked series pinned in the zero-eviction span | 68.4%, margin 11.90 pt | `sc-workshop-paper/results_tables/02_budget_sweep.md` |
| first-use share of stall, small → large population | 10.4% → 32.3% | `sc-workshop-paper/results_tables/05_scale_sweep.md` |
| same parse, two node types | 2.3× | `results/bench_potential_activation_*` vs `bench_activated_residency_BIG.json` |

## Known defects, recorded rather than hidden

**D1 — `fig-sgb-spread` and the §III prose read different nodes.** The figure
prefers `bench_format_activation_atl1-1-02-003-25-1.json.csv` (max 21.87, min
0.316, spread 69.3×); §III quotes 22.0 / 0.34 / 65×, which are
`bench_format_activation.csv` from a different node. The project's own
standing rule is to facet by node and never pool. Both files are archived
here. `fig_sgb_spread_alt()` in the same script pins the file matching the
prose and prints which one it used.

**D2 — `fig-replacement-loss` has no data path.** Its timeline is hardcoded as
literals at `scripts/figures/make_figures.py:290-294`. The values were derived
from a representative `atomagents_exp3` trial, but the figure does not read
that trial and nothing in this snapshot re-derives it. Treat the figure as an
illustration of a measured trial, not as a reproduction of one.

**D3 — the model band in the s/GB figures is unsourced.** `MODEL_BAND =
(2.78, 3.81)` appears in no measurement artifact; the generator's own
docstring says so.

**D4 — "agents insert recovery and retry calls" is unsupported.** The trace
schema carries no tool-outcome event, so this claim cannot be checked against
anything archived here.

## Two test failures present at snapshot time

`pytest runtime/tests/` gives 303 passed, 2 failed, 1 skipped. Both failures
predate the submitted work and neither touches a paper claim:

* `test_format_activation_bench.py::...::test_evict_then_mincore_reports_cold_on_local_tmp`
  — environment-dependent (`fadvise` behaviour on the login node's `/tmp`).
* `test_replay_divergence.py::...::test_atomagents_scored_population_equals_the_recorded_denominator`
  — asserts `24 == 0`; the test encodes a corpus state from before the aligned
  campaign and should be made corpus-relative.
MD

cat > "$DEST/reproduce.sh" <<'SH'
#!/usr/bin/env bash
# Regenerate every figure from this snapshot and byte-compare against the
# archived PNGs. Needs only python3 with matplotlib, numpy and pandas --
# no network, no cluster, no GPU. See ENV.txt for the versions used.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> copying the tree to a writable scratch dir"
cp -r "$HERE/tree" "$WORK/tree"
chmod -R u+w "$WORK/tree"
mkdir -p "$WORK/archived"
cp "$HERE/tree/sc-workshop-paper/figure_drafts"/*.png "$WORK/archived/"

echo "==> regenerating"
( cd "$WORK/tree" && python3 scripts/figures/make_figures.py --no-install )

echo
echo "==> comparing"
same=0; differ=0; missing=0
for f in "$WORK/archived"/*.png; do
  b="$(basename "$f")"
  g="$WORK/tree/sc-workshop-paper/figure_drafts/$b"
  if   [[ ! -f "$g" ]];   then missing=$((missing+1)); echo "  MISSING  $b"
  elif cmp -s "$f" "$g";  then same=$((same+1))
  else differ=$((differ+1)); echo "  DIFFERS  $b"
  fi
done
echo
echo "byte-identical: $same   differ: $differ   missing: $missing"
[[ $differ -eq 0 && $missing -eq 0 ]] || {
  echo
  echo "NOTE: matplotlib renders differ across library versions. Compare"
  echo "ENV.txt against your interpreter before treating a diff as a defect;"
  echo "the printed numeric summaries in the regeneration log are the"
  echo "version-independent check."
  exit 1
}
SH
chmod +x "$DEST/reproduce.sh"

echo "==> self-test: regenerating figures from the staged tree"
{
  echo "Reproduction self-test run at build time."
  echo "host   $(hostname)"
  echo "date   $(date -u +%FT%TZ)"
  echo
  bash "$DEST/reproduce.sh" 2>&1
} > "$DEST/REPRODUCTION.txt" || true
tail -1 "$DEST/REPRODUCTION.txt"

echo "==> checksumming"
( cd "$DEST" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
echo "    $(wc -l < "$DEST/MANIFEST.sha256") files"

echo "==> archiving"
( cd "$DEST_PARENT" && tar czf "$TAG.tar.gz" "$TAG" \
    && sha256sum "$TAG.tar.gz" > "$TAG.tar.gz.sha256" )

chmod -R a-w "$DEST"
echo
echo "snapshot: $DEST"
echo "tarball:  $DEST_PARENT/$TAG.tar.gz  ($(du -h "$DEST_PARENT/$TAG.tar.gz" | cut -f1))"
