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

**P3 is now passed by exactly one candidate; P4 by none.** Until 2026-08-05 every
size measured was a knob we turned, which is acceptable for P1/P2 (they ask
about the mechanism) and disqualifying for the workload claim.

C1 now clears P3: **UniRef50 is 16.94 GB because UniProt made it that size**, and
UniRef90 is a second, larger point on the same axis that we equally did not
choose. The release is versioned (2026_02) and md5-published, so the number is
citable and reproducible by anyone. This is the first size in the project that
is not ours.

**P4 remains untested for every candidate**, and it is still the criterion that
killed all the earlier ones. Passing P3 does not soften it: a database whose
size we did not choose is not yet a workload in which *an agent chooses which
database to search*.

---

## ⚠️ THE RANKING METRIC WAS WRONG (2026-08-05) — read this before the table

The table below ranks candidates by **activation share** = `load / (load + compute)`.
Measuring Parquet properly showed that number is **not a property of the format at
all**. Same 2 GB file, same Arrow table, three different per-call computes:

| per-call compute | compute time | activation share | retention speedup |
|---|---|---|---|
| `scan` (sum one column) | 0.094 s | **94.7%** | 18.95× |
| `groupby` (hash agg) | 0.259 s | **86.7%** | 7.53× |
| `sort` (full sort) | 48.795 s | **3.4%** | 1.03× |

**A 28× spread on identical data.** So "LAMMPS 90%, Parquet 73%, pyhmmer 48%" was
never comparing formats — it was comparing whatever compute each probe happened to
use (a force evaluation, a groupby, a single-query search). Those are different
denominators and the comparison was meaningless.

### The compute-independent metrics, which is what should have been compared

`s/GB retained` = `load_warm / activated_GB` — seconds of stall avoided per GB
held. It has no `compute` term, and it is **exactly the quantity the value-density
policy consumes**, so it was always the right ranking key.

| candidate | load_warm | activated | expansion | **s/GB held** | io_share |
|---|---|---|---|---|---|
| **pyhmmer, REAL UniRef50 (16.94 GB)** | 107.10 s | 36.08 GB | 2.13× | **2.968** | **11.9%** |
| pyhmmer synthetic FASTA (2 GB) | 13.65 s | 4.54 GB | 2.27× | **3.006** | n/a |
| LAMMPS EAM (3.32 GB) | 42.83 s | 16.93 GB | 5.10× | **2.530** | 1.9% |
| 72B model parked at R2 | — | ~279 GB | 1.90× | **~2.96** | — |
| **Parquet → Arrow (2 GB)** | 1.69 s | 4.09 GB | 2.04× | **0.413** | **29.4%** |

The pyhmmer row is now measured on the real database rather than on random
residues, and **it barely moved**: s/GB 2.968 against 3.006, expansion 2.13×
against 2.27×. See §"C1 on real data" below for the full ladder, the same-node
control, and the one place where real data does NOT behave like synthetic.

**The ranking inverts.** Parquet, which looked like the best non-incumbent
candidate at 73%, is **~7× WORSE than pyhmmer** on the metric that matters: it
decodes too fast relative to what it holds, so retaining it buys little per GB.
Its 29.4% I/O share (the first valid one measured on node-local NVMe, where
eviction is verified) says the same thing from the other side — Parquet is
substantially more movement-bound than the transformation-bound candidates.

**Consequence:** C3 is demoted, C1 is promoted, and activation share should appear
in the paper only alongside the compute it was measured against.

## Status

| candidate | P1a | P2 magnitude | P3 | P4 | evidence | artifact |
|---|---|---|---|---|---|---|
| **LAMMPS EAM** (incumbent) | PASS | **90%** | ✗ inflated by us | ✗ | `measured` | `results/bench_activated_residency_BIG.json` |
| **Parquet → Arrow** | PASS | **73%** | untested | untested | `measured-unpersisted` | **none — re-run required** |
| **pyhmmer, REAL UniRef50** | PASS | **48.6% phmmer / 3.4% hmmsearch** | ✅ **UniProt sets the size** | untested | `measured` | `results/bench_p1_real_uniref50_full.json` |
| pyhmmer synthetic FASTA | PASS | **~49%** | ✗ synthetic | untested | `measured` | `results/bench_p1_hmmer_2gb.json` |
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

Confirmed at 8 GB once storage was controlled: **47.6%**, so the share is flat at
~46-49% across a 40x size range and is a format constant like expansion.

| size | load | compute | share | speedup | load rate |
|---|---|---|---|---|---|
| 0.2 GB | 1.02 s | 1.36 s | 42.8% | 1.75x | 197 MB/s |
| 2 GB | 13.65 s | 14.46 s | 48.6% | 1.94x | 147 MB/s |
| 8 GB | 52.19 s | 57.44 s | 47.6% | 1.91x | 153 MB/s |

> **An 8 GB reading of 91.8% / 12.13x was reported and is WITHDRAWN.** It came
> from a load off Lustre scratch where throughput degraded 16x within the single
> read (304 -> 18.7 krec/s). The identical file on node-local NVMe loaded in
> 52.2 s at a FLAT 423 krec/s, first million to nineteenth. See
> `results/diag_p1_superlinear_8gb.json`. Do not cite 91.8% or 12.13x.

One caveat bounds it and survives everything below. HMMER's per-call search
costs about what its load costs, so retention roughly halves a call and **cannot
win big at any database size** — the ratio does not improve with scale, and the
8 GB point confirms that rather than contradicting it.

The other caveat — that random residues find 8-27 hits where a real database
finds many more — was the open question. It is now measured.

---

## C1 on real data — UniProt release 2026_02 (measured 2026-08-05)

Everything above was random-residue FASTA we generated. The real databases were
downloaded and **md5-verified against the publisher's own `RELEASE.metalink` /
`md5_checksums`**, which are stored next to the data so the check can be
re-derived:

| file | size | md5 | verified |
|---|---|---|---|
| `uniprot_sprot.fasta.gz` | 93,706,469 B | `797dad11a33b1b58e3c140649a74d6b6` | ✅ |
| `Pfam-A.hmm.gz` | 418,160,514 B | `7ab3c4e215d0daaea3004e37c4e24f8a` | ✅ |
| `uniref50.fasta.gz` | 8,770,260,598 B | `3228886e9d749f050f60e9a0ce1f727d` | ✅ |
| `uniref90.fasta.gz` | 32,059,052,376 B | `abdd341aeafa7fa060c8d6639d594990` | ✅ |

**UniRef50 decompresses to 16.94 GB, not the ~27 GB the plan assumed.** Checked
independently of the benchmark by streaming the gz through `wc -c` and `grep -c
"^>"`: **16,939,476,667 bytes / 38,794,121 records**, matching what the probe
reported to the byte and to the record.

### The compute-independent metrics hold on real data

All rungs on node-local NVMe, and every one has `rungs_verified_distinct: true`
— the cold rung is confirmed to have pulled the file off the device and the warm
rung confirmed not to, via `/proc/self/io: read_bytes`. **These are the first
valid pyhmmer I/O shares; every earlier one was void.**

| dataset | file | records | B/rec | load_warm | activated | expansion | **s/GB** | io_share |
|---|---|---|---|---|---|---|---|---|
| Swiss-Prot | 0.288 GB | 575,503 | 500 | 1.83 s | 0.559 GB | 1.94× | **3.268** | 41.2% |
| UniRef50 strided | 2.0 GB | 4,200,401 | 476 | 13.33 s | 3.984 GB | 1.992× | **3.345** | 19.9% |
| UniRef50 strided | 8.0 GB | 16,801,599 | 476 | 52.66 s | 15.941 GB | 1.993× | **3.303** | 12.9% |
| **UniRef50 whole** | **16.94 GB** | **38,794,121** | 437 | 107.10 s | 36.08 GB | **2.130×** | **2.968** | **11.9%** |
| **UniRef90 whole** | **60.95 GB** | **121,389,642** | 502 | 372.60 s | 117.20 GB | **1.923×** | **3.179** | **11.5%** |
| *synthetic, for reference* | 2.0 GB | 5,000,672 | 400 | 14.39 s | 4.534 GB | 2.267× | 3.175 | 11.8% |
| *synthetic, for reference* | 8.0 GB | 19,914,456 | 402 | 56.75 s | 18.06 GB | 2.257× | 3.143 | 12.5% |

**Same-node control** (`atl1-1-02-005-2-2`, `results/bench_p1_pair_*.json`) —
seconds do not transfer across nodes, so the real-vs-synthetic gap is only
admissible measured together:

| | real UniRef50 2 GB | synthetic 2 GB | real/synth |
|---|---|---|---|
| expansion | 1.992× | 2.267× | **0.88** |
| s/GB retained | 3.345 | 3.175 | **1.05** |
| activation share (random 200-mer) | 47.35% | 48.01% | **0.99** |

**Verdict: the synthetic numbers were right.** Expansion is ~12% lower on real
data and s/GB ~5% higher; activation share is indistinguishable. Nothing in the
C1 assessment changes.

Across a **212× size range on real data** — 0.29 GB Swiss-Prot to 60.9 GB
UniRef90, and UniRef90 is a *different* database (90% identity clustering, far
more redundant), not more of the same one:

| metric | range on real data | synthetic |
|---|---|---|
| expansion | **1.92 – 2.13×** | 2.27× |
| s/GB retained | **2.97 – 3.35** | 3.13 – 3.18 |
| activation share (`phmmer`, random 200-mer) | **47.1 – 48.6%** | 48.0 – 48.6% |
| io_share (node-local NVMe) | **11.5 – 12.9%** | 11.8 – 12.5% |

These are format constants, and Swiss-Prot's 41.2% io_share is the one outlier —
expected, since a 0.29 GB file is small enough to be served partly from the
drive's own cache, which `mincore` cannot see.

**Both size estimates in the plan were high.** UniRef50 is 16.94 GB, not ~27;
UniRef90 is 60.95 GB, not ~100.

### ⚠️ Where real data does NOT behave like synthetic: the QUERY

The asserted claim was that HMMER's cost is largely hit-count independent
because the MSV/Viterbi filter sweeps every sequence regardless. **Partly true,
and it fails exactly where it matters most.** Measured on one retained block, so
the load is held constant and only the query changes:

*(a) Arbitrary real queries — the claim HOLDS.* Six real Swiss-Prot proteins,
each paired with a **length-matched random control** (query length and homology
are otherwise confounded — a real query is usually longer than 200 aa, so a bare
real-vs-random gap is mostly length):

| query | len | real hits | real s | rand s | real/rand |
|---|---|---|---|---|---|
| Q6GZX4 | 256 | 11 | 25.44 | 21.36 | 1.19× |
| A6SZI9 | 439 | 41 | 36.90 | 36.40 | 1.01× |
| B1LL12 | 137 | 22 | 10.64 | 11.75 | 0.91× |
| A9GUX6 | 194 | 40 | 16.64 | 15.70 | 1.06× |
| B7LUR8 | 70 | 38 | 7.30 | 6.89 | 1.06× |
| A6TFL2 | 341 | **1145** | 54.30 | 27.59 | **1.97×** |

Five of six are within ±20% of a length-matched random query. The sixth, with
1145 hits, costs 1.97×. So cost is hit-count *dependent*, but only weakly until
hits become a non-trivial fraction of the database.

*(b) A real profile query — the claim FAILS, by 24×.* `hmmsearch` with a Pfam
profile is the canonical HMMER call, and Pkinase (PF00069) matches a large
fraction of a real database and almost nothing in random residues:

| 2 GB, same node `…007-24-2` | compute | hits | activation share |
|---|---|---|---|
| synthetic | 23.6 s | 31 | 37.6% |
| **real UniRef50** | **566.0 s** | **37,577** | **1.9%** |

**24.0× on identical-size data, same node, same profile.** Corroborated on the
representative strided subset (369.9 s / 46,126 hits, 15.7×) and at 8 GB
(1482.5 s / 182,323 hits, share 3.4%).

### A sampling trap worth recording: a UniRef50 prefix is not UniRef50

The first "real 2 GB" subset was a head truncation, and it is **biased**:

| | records | B/record | expansion | s/GB |
|---|---|---|---|---|
| whole `uniref50.fasta` (16.94 GB) | 38,794,121 | 437 | 2.130× | 2.968 |
| its **first** 2 GB | 1,000,810 | **1998** | **1.148×** | 4.652 |
| **every 8th record**, 2 GB | 4,200,401 | 476 | 1.992× | 3.345 |

The file is not in random order: its opening region holds sequences ~4.6×
longer than the database average. That matters here more than it would
elsewhere, because pyhmmer's expansion is driven by **per-record** overhead —
so the prefix reports 1.148× expansion where the database is 2.13×, a 1.9× error
that looks exactly like a real finding. Had the head subset been the only real
measurement, the conclusion "expansion collapses on real data" would have been
reported, and it would have been an artifact of `head -c`. Strided sampling
(`experiments/make_fasta_subset.py`, stride mode) reproduces the whole-database
value to within 7%. **Any size-matched subset of an ordered database must be
strided, not truncated.**

**What this costs C1.** Retention halves a `phmmer` call (1.9×) and does
essentially nothing for an `hmmsearch` call (1.03×). The ~48% share is real, but
it is the share *for a single-sequence query*, and the profile search — the more
common use — sits at 2-3%. This is the same 28× trap Parquet exposed, now
demonstrated within one consumer on real data. **The compute-independent metrics
are unaffected: s/GB and expansion do not move, because the query changes only
the denominator.** That is precisely why s/GB is the ranking key.

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
