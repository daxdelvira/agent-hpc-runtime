# Domain-coherent workload search — 2026-08-06

**Question.** We have a domain-neutral harness, a materials agent (AtomAgents/LAMMPS),
and one exogenously-sized data artifact that is a protein database (UniRef90/pyhmmer).
The pair is incoherent. Find either **(A)** a biology/protein agentic workflow that
plays AtomAgents' role, or **(B)** a materials/chemistry artifact that is exogenously
sized *and* expensive to activate.

**Gates applied** (from `candidate_register.md` P1a–P4 and the plan's RETRACTIONS):

| # | gate | kills because |
|---|---|---|
| 1 | in-process API holding the activated structure | HARD — RELION died here; a CLI-only tool cannot be fixed by us |
| 2 | exogenous size | if we pick the size it is an illustration, not a regime |
| 3 | transformation-bound, not movement-bound | if the file ≈ the in-memory form, retention buys nothing the page cache does not |
| 4 | agent-determined access | the agent must choose *which* artifact |

Evidence classes follow the register: `measured` / `inspected` / `asserted`.
**Nothing below is `measured`.** Everything here is `inspected` (source/docs read) or
`asserted` (reasoned from published numbers). No probe was run.

---

## Verdict in one line

**Category A has a clean winner** — ProtAgents, the *same lab, same framework* as
AtomAgents, so the port is mechanical rather than architectural.
**Category B has one candidate that survives all four gates** — an RDKit
`SubstructLibrary`/fingerprint index over ChEMBL/PubChem — and it survives with one
named escape hatch (serialization) that must be probed before it is believed.
The instrument-sized chemistry artifact everyone reaches for first (mass spec, mzML)
**fails gate 1 in exactly the way MRC did**, and that elimination is the most useful
negative here.

---

## A1 — ProtAgents (lamm-mit) — **RECOMMENDED, and the cheapest thing on this list**

*What it is.* `github.com/lamm-mit/ProtAgents`, Ghafarollahi & Buehler, *Digital
Discovery* 3(7):1389 (2024), arXiv:2402.04268. De novo protein design by multi-agent
LLM collaboration. This is **AtomAgents' sibling**: same group, same author, same
year, same architecture.

*Why it is nearly free for us* (`inspected` — read `agents.py` and `agent_functions.py`):

- Framework is **AutoGen**, with `UserProxyAgent` + `AssistantAgent` and tools bound
  through a `function_map` dict — architecturally identical to AtomAgents. Agent roles:
  `user_proxy`, `Planner`, `Critic`, `Coder`, `Executor`, `ragproxyagent`,
  `Scientific_Reviewer`, `assistant`. Our `execution/runner.py` plumbing (and the P1b
  process-liveness problem it currently has) transfers unchanged.
- 21 tool functions in `agent_functions.py`, including `fold_protein` (OmegaFold),
  `design_protein_from_CATH`, `design_protein_from_length`, `calc_protein_ANM` (ProDy),
  `analyze_protein_structure` (DSSP), `calculate_force_energy_from_seq` (ProteinForceGPT),
  `get_FASTA_from_name` (EBI Proteins API), `fetch_protein_structure_from_PDBID`.
  Libraries: ProDy, BioPython, RDKit, LlamaIndex, Chroma, py3Dmol. Binaries: OmegaFold, DSSP.
- Ships `exp1/exp2/exp3.ipynb` matching the paper's three experiments — the same
  exp-numbered structure our eval driver already assumes.

*Gate verdicts.*

| gate | verdict |
|---|---|
| 1 in-process | **PASS** for the agent side (it is a Python function map). The retained artifact is not ProtAgents' own — it is pyhmmer's, see below. |
| 2 exogenous | **PASS via pairing**: UniRef50/90 sizes are UniProt's, already md5-verified in `results/`. |
| 3 transformation-bound | **PASS via pyhmmer** (measured: 1.92–2.13× expansion, 2.97–3.35 s/GB on real data). |
| 4 agent-determined | **the open one** — see the honesty note. |

*The honesty note.* ProtAgents does **not** ship a homology-search tool. Its only
sequence-database touches are `get_FASTA_from_name` (a REST call to EBI) and a Chroma
RAG index. So the domain coherence is real — a de novo-design agent checking a
generated sequence against a reference database is canonical practice, not a
contrivance — but **we would be adding the pyhmmer tool ourselves**. That is
acceptable in a way authoring the RELION replacement was not: we author the *tool
binding*, while the *consumer* (pyhmmer) and the *size* (UniProt) both remain someone
else's. It must be stated that way in the paper, not glossed.

Gate 4 then becomes a real, testable design: the agent picks Swiss-Prot (0.29 GB) vs
UniRef50 (16.94 GB) vs UniRef90 (60.95 GB) vs Pfam-A depending on how thorough the
homology check needs to be — all four already downloaded and checksummed. That is the
first credible P4 story in the register.

*Cost:* AutoGen + OmegaFold + ProDy + DSSP into a fresh env. OmegaFold needs a GPU for
folding calls, which we have. 8 days is enough because the runner already exists.

---

## A2 — Biomni (snap-stanford) — **best gate-4 story, worst gate-3 story**

*What it is.* `github.com/snap-stanford/Biomni`, Leskovec group; *Science*
(adz4351) / bioRxiv 2025.05.30.656746. A general biomedical agent: 150 tools, 105
software packages, **59 databases**, plus an ~11 GB "data lake" auto-downloaded on
agent construction.

*In-process API* — **PASS** (`inspected`): pure Python package (`pip install biomni`),
tools organised as `biomni/tool/{genomics,molecular_biology,biochemistry,database,…}.py`
with a `tool_registry.py`. Agent-side retention is expressible.

*Gate 4* — **the strongest of any candidate seen.** The `data_lake_dict` maps filenames
to descriptions and the agent selects which dataset a tool loads. That is literally
"the agent chooses which artifact", which is the criterion that killed everything else.

*Why it still loses to A1* — **gate 3, and probably gate 2** (`inspected`):

- The 11 GB data lake is **77 datasets**, so the mean artifact is ~150 MB. Nothing there
  is out-of-core; the whole lake fits in a node's RAM twice over. Activation is a
  `pd.read_parquet` — the register already measured Parquet at **0.413 s/GB, ~7× worse
  than pyhmmer**, and 29.4% io_share, i.e. movement-bound.
- I checked the tool descriptions for `genomics.py` and `molecular_biology.py`
  directly: **no BLAST, PSI-BLAST, HMMER or profile-HMM tool exists in either.** A web
  summary claimed Biomni has iterative-BLAST/HMMER homology tools; that claim did not
  survive checking the source and should not be repeated. Its "databases" are mostly
  REST/Entrez/BioMart queries, which activate nothing locally.

*Use it as:* a citation that agent-chosen database selection is real practice, and as a
fallback if ProtAgents' install fights us. Not as the data axis.

---

## B1 — RDKit `SubstructLibrary` / fingerprint index over ChEMBL–PubChem — **the only category-B survivor**

*What it is.* The chemistry analogue of pyhmmer-over-UniRef: a virtual-screening or
substructure/similarity search tool holds a chemical library resident and answers
repeated queries against it.

*Consumer + in-process API* — **PASS, and this is the strongest gate-1 story in the
register after pyhmmer** (`inspected`, RDKit docs + RDKit blog). RDKit provides a
**library-owned container designed to be built once and queried many times**:
`rdkit.Chem.rdSubstructLibrary.SubstructLibrary`, composed of
`CachedTrustedSmilesMolHolder` (molecules) + `PatternHolder`/`TautomerPatternHolder`
(screening fingerprints) + `KeyFromPropHolder`, queried by `GetMatches(query_mol)`.
The similarity path is equally library-owned: a held list of Morgan fingerprints +
`DataStructs.BulkTanimotoSimilarity`. This matters — it is not us inventing an
aggregate container the way a "list of pymatgen Structures" would be. It is exactly
`pyhmmer.easel.DigitalSequenceBlock`'s role.

*Exogenous size* — **PASS**, versioned publisher releases, sizes verified by HTTP HEAD
today:

| corpus | scale | file | bytes (verified) |
|---|---|---|---|
| ChEMBL 37 (EBI, 2026-05) | 2,921,148 compounds | `chembl_37_chemreps.txt.gz` | 292,540,141 |
| PubChem Compound | ~119 M compounds | `CID-SMILES.gz` | **1,483,693,850** |
| Enamine REAL / ZINC-22 | 10^9–10^10 enumerable | — | not checked |

PubChem's CID-SMILES is ~1.48 GB gzipped of pure SMILES text; the full PubChem download
is documented at >300 GB across >340 files. None of these numbers is a knob we turn,
and both files sit behind stable publisher URLs with release numbering — the same
provenance shape that let UniRef50 clear P3.

*Transformation-bound* — **PASS, and by a wide margin** (`asserted` from a published
build time). SMILES text → RDKit `Mol` is parse + valence/aromaticity perception +
sanitization, then a pattern fingerprint per molecule. The RDKit blog's own benchmark:
building a `SubstructLibrary` over **2.4 M ChEMBL molecules took ~990 s** (2018 hardware,
so read it as an order of magnitude, not a target). Extrapolated to PubChem's 119 M that
is hours of pure activation from a 1.5 GB file — an expansion and s/GB profile that
should sit *above* pyhmmer, because a SMILES string is far smaller than the object it
inflates to. This is the opposite of the MRC/safetensors failure: the on-disk form is
text and the in-memory form is a C++ molecule graph plus a bitvector.

*Agent-determined* — **PASS, plausibly, and we already have the agent.** ChemGraph is
checked out at `/storage/project/r-ag117-0/shared/agent_hpc/ChemGraph` with
LangChain-style `@tool` functions in
`src/chemgraph/tools/cheminformatics_tools.py` (already SMILES-centric:
`molecule_name_to_smiles` via pubchempy, `smiles_to_atomsdata` via ASE). A screening
tool that takes `(query, library_name)` drops in beside those, and the agent choosing
ChEMBL vs a PubChem tranche vs an Enamine block *is* the P4 decision.

*The escape hatch that must be probed before believing any of this.* `SubstructLibrary`
supports serialization (`ToStream`/`InitFromStream`). If a serialized library
deserializes fast, the activation cost is amortizable to a one-time conversion and the
candidate degrades toward movement-bound — the same shape as RETRACTED 2, where the
HDF5 cost had a static one-line fix. **The decisive experiment is one probe: build time
from SMILES vs. `InitFromStream` time from the serialized blob, both on node-local
NVMe.** If deserialize is ≫ file-read time, B1 stands; if it is ≈ file-read time, B1
joins Parquet in the movement-bound bin. Do this before anything else in category B.

*Installability:* `pip download rdkit` resolves `rdkit-2026.3.5-cp310-cp310-manylinux_2_28_x86_64.whl`
from this cluster today — a binary wheel, no build, no module needed. Network egress to
NCBI and EBI FTP works (both HEADs above succeeded from a login node).

---

## B2 — pymatgen `Structure` corpus over the Crystallography Open Database — **listed, but it fails gate 1 the RELION-nuance way**

*What it is.* COD publishes **534,673 CIF entries** (front page, latest deposition
2026-07-29). CIF → `pymatgen.core.Structure` is a text parse *plus symmetry expansion*
(spacegroup operations applied to the asymmetric unit), which is genuine decode work,
and a structure-search tool would hold the corpus resident and query it with
`StructureMatcher`. Materials science, so it is coherent with AtomAgents with no
domain switch at all. Gate 2 passes (COD's size is COD's). Gate 4 passes (which subset
to search is an agent decision). `ltalirz/cif-parsing-benchmark` reports **~1 s per
structure** for both ASE and pymatgen on its large-basis test set, which if it held
across COD would be an enormous activation cost.

*Why it fails.* pymatgen gives you a per-item parser and a per-item object. It gives you
**no library-owned corpus container** — no `SubstructLibrary` equivalent, no
`DigitalSequenceBlock` equivalent. "A dict of 534k `Structure` objects" is a container
*we* would author, and the register already ruled on that shape: *"Passing P1a by
authoring the consumer is not passing P1a"* (the mrcfile/numpy cryo-EM nuance). The
per-item decode being pymatgen's makes this a softer failure than RELION's, but it is
the same failure, and I do not think it should be spent time on while B1 is available.

I also did not verify the COD archive size — the `cod-cifs-mysql.tgz` URL 404s and I did
not chase it. Any size claim for COD in the paper needs that number found first.

---

## B3 — mass spectrometry mzML (pyteomics / pymzML) — **ELIMINATED on gate 1. This is the useful negative.**

This is the artifact that *looks* perfect and is not, so it is worth writing down so
nobody re-proposes it:

- **Gate 2, exemplary.** An mzML file's size is set by an instrument run — acquisition
  rate × gradient length. Nobody's knob. PRIDE/MassIVE depositions are TB-scale and
  versioned. This is the same class as the EMPIAR story we lost with RELION.
- **Gate 3, exemplary.** mzML is XML with **base64-encoded, zlib-compressed** binary
  arrays: `pyteomics.mzml.decode_data_array()` does base64-decode then zlib-inflate then
  numpy view. Uncompressed mzML is documented at ~4.9× the vendor RAW size. There is
  more real decode work per byte here than in any other candidate seen.
- **Gate 1, fatal.** `pyteomics.mzml.MzML` and `pymzml.run.Reader` are **indexed,
  lazily-streaming** readers: they hold an *offset index*, not the spectra. There is no
  library-provided object that holds the activated spectrum collection across calls.
  What retention would cache is the index plus whatever the OS page cache already holds —
  which is precisely the "retention buys nothing" condition. To get an activated
  structure we would build the in-memory spectrum store ourselves, and that is the
  authored-consumer failure again.

The pattern worth extracting: **transformation-bound formats are common; consumers that
own the transformed result are rare.** Gate 1 is not a formality that a good format can
argue its way past, and three of the last four eliminations (RELION, MRC, mzML) were all
gate 1 in different clothing.

---

## What I would do with the 8 days

1. **Take A1.** Clone ProtAgents, stand it up on the AutoGen path the AtomAgents runner
   already knows, and bind one pyhmmer tool whose `database` argument the agent chooses
   among the four already-checksummed UniProt files. That converts the register's
   "P4 untested for every candidate" into a measurable entropy claim, using data we
   already own and numbers we already measured.
2. **Spend half a day on B1's single decisive probe** (`SubstructLibrary` build-from-SMILES
   vs `InitFromStream`). It is cheap, it is CPU-only, and it either hands us a
   second domain-coherent axis in chemistry or produces a clean second elimination.
3. **Do not** open B2 or B3.

## What I did not find

No category-B candidate was found that is simultaneously instrument-sized *and* has a
library-owned activated container. B1 substitutes a *reference-corpus* size (ChEMBL,
PubChem) for an instrument size, which is the second entry on the plan's own exogeneity
list and is exactly what UniRef already does for us — but it means the "an instrument
set this size" framing died with RELION and has not been recovered.
