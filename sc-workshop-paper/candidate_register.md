# Candidate consumer register — P1 and the gates after it

**Purpose.** Track every candidate workload/consumer considered for the data axis,
what gate it is at, and **what class of evidence** put it there. Created 2026-08-05
because the P1 results existed only in a chat transcript: one JSON on disk, one
inline heredoc that was never saved, and three verdicts with no artifact at all.
That is the same failure the plan's provenance rule was written against.

**Rule for this file:** every verdict carries an evidence class. A verdict with
class `asserted` is a hypothesis wearing a verdict's clothes — it may be right,
but it must not be cited as a finding and must not close a gate.

| class | meaning |
|---|---|
| `measured` | a benchmark ran and wrote an artifact under `results/` |
| `measured-unpersisted` | a benchmark ran but no artifact survives; re-run before citing |
| `inspected` | filesystem/API inspection, no timing |
| `asserted` | reasoned from format or documentation knowledge only |

---

## The gates, in order

| gate | question | kills a candidate because |
|---|---|---|
| **P1a** | does the consumer library expose an in-process API holding the activated structure? | HARD — someone else's CLI-only tool cannot be fixed by us |
| **P1b** | can the framework keep that process alive across tool calls? | SOFT — our own plumbing; AtomAgents currently fails it at `execution/runner.py:26-45` |
| **P2 magnitude** | is activation a large share of a tool call? | a retainable structure that rebuilds cheaply is not worth retaining |
| **P3 exogenous size** | is the size set by something outside the experiment? | RETRACTIONS: if we choose the size it is an illustration, not a regime |
| **P4 agent-determined** | does the agent choose *which* artifact? | the criterion that killed every earlier candidate |

**Nothing below has passed P3 or P4 yet.** Every size measured so far is a knob we
turned. That is acceptable for P1/P2, which ask about the mechanism, and is
disqualifying for the workload claim.

---

## Status

| candidate | P1a | P2 magnitude | P3 | P4 | evidence | artifact |
|---|---|---|---|---|---|---|
| **LAMMPS EAM** (incumbent) | PASS | **90%** | ✗ inflated by us | ✗ | `measured` | `results/bench_activated_residency_BIG.json` |
| **Parquet → Arrow** | PASS | **73%** | untested | untested | `measured-unpersisted` | **none — re-run required** |
| **pyhmmer FASTA** | PASS | **~49%** | ✗ synthetic | untested | `measured` | `results/bench_p1_hmmer_2gb.json` |
| RELION 5.0 | **FAIL (confirmed)** | — | — | — | `inspected` (exhaustive) | — |
| raw MRC stacks | pass | *provisional ~0%* | — | — | `asserted` | **none** |
| MMseqs2 / DIAMOND | untested | untested | — | — | — | — |
| minimap2 / BWA index | untested | untested | — | — | — | — |
| Zarr / N5 | untested | untested | — | — | — | — |
| compressed HDF5 | untested | untested | — | — | see RETRACTED 2 | — |
| AlphaFold MSA over BFD | untested | untested | — | — | — | — |

### Detail on the three measured rows

**LAMMPS EAM — PASS / STRONG.** Warm parse 42.83 s vs 4.78 s reuse in a live
`lammps()` instance; 5.10x expansion (3.32 GB file -> 16.93 GB activated).
`r4_repeat_coeff` at 42.84 s confirms LAMMPS does not memoise, so the saving
requires an explicit resident worker rather than falling out by accident. Fails
P3: the file was inflated by `inflate_fs_blockaware.py` to ~10^3x the largest
published tabulation.

**Parquet -> Arrow — PASS, but the number needs redoing.** 73% activation share,
2.93x expansion, s/GB 2.02, from an 82 MB file with a single-column aggregation
as the per-call compute, run inline on a login node. Directionally the strongest
non-incumbent candidate; as evidence it is thin, and it is the one row here with
no artifact at all. **Re-run through `bench_p1_consumer_retention.py` before this
number appears anywhere.**

**pyhmmer FASTA — PASS / MODERATE.** Activation share 42.8% at 0.2 GB and 48.6%
at 2 GB, so the ratio is structural rather than small-size pipeline overhead.
Expansion flat at 2.27-2.31x across a 10x range, consistent with the plan's
finding that these are format constants. s/GB 3.01, in the same band as a parked
72B (~2.96).

Two caveats that bound it. HMMER's per-call search costs about what its load
costs, so retention roughly halves a call and **cannot win big at any database
size** — the ratio does not improve with scale. And the FASTAs are random
residues, which find 8-27 hits where a real database would find many more;
HMMER's cost is dominated by the MSV/Viterbi sweep over all sequences and should
be largely hit-count independent, but that is `asserted`, not measured, and the
~49% figure is what the whole pyhmmer assessment rests on.

### Detail on the two provisional eliminations

**RELION 5.0 — FAIL on P1a. CONFIRMED 2026-08-05, gate closed.** The earlier
verdict rested on `ls` of `bin/` and was correctly flagged as non-exhaustive. A
full search of the install tree now settles it:

- The **only** two Python files in the entire package are `relion_it.py` (1029
  lines, 9 subprocess calls -- the RELION-IT automated pipeline driver) and
  `relion_schemegui.py` (419 lines, 8 subprocess calls -- an FLTK GUI). Both
  ORCHESTRATE the binaries; neither holds particle data in-process.
- The only shared objects shipped are `libfltk*.so` -- the GUI toolkit. There is
  no Python extension module anywhere in the tree.
- `import relion` fails; nothing is importable.

So a particle stack cannot be held across tool calls in RELION, and this is not
something we can fix -- it is what the program is.

**The nuance that does NOT rescue it.** A Python cryo-EM consumer built on
`mrcfile`/`numpy` is perfectly possible, and would trivially pass P1a. But then
WE write the consumer, and its activation cost is whatever we implement -- which
fails exogeneity for exactly the reason the inflated EAM potential does. Passing
P1a by authoring the consumer is not passing P1a.

**Cost of this closure:** cryo-EM had the best P3 story of any candidate (EMPIAR
depositions 700 GB-1.8 TB, sized by an instrument). Losing it is why the register
now leans on C1/C3, both of which have weaker exogeneity.

**raw MRC — provisional, and it was overstated.** The claim "expansion ~1x,
activation share ~0%, so R3 collapses onto R1" is `asserted` from the format
being a header plus a raw array. The probe was never run on an MRC file. The
reasoning is probably right and the conclusion would be a useful negative — it
would show the mechanism needs a transformation-bound format, not merely a large
one — but it is not currently a finding.

---

## What this register says about the pivot

The mechanism gate is doing real work: it has already separated retainable
consumers from CLI-only ones, and transformation-bound formats from raw ones.
What it has NOT done is find a candidate that passes P3 and P4, and every P1/P2
number above is measured on a size we chose.

The uncomfortable shape: the candidates with the best mechanism scores (LAMMPS,
Parquet) are the ones where we control the data, and the candidate with the best
exogenous-size story (cryo-EM) is the one provisionally failing P1a. Resolving
the RELION check is therefore worth more than another point on the pyhmmer curve.
