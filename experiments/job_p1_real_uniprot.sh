#!/bin/bash
# P1/P2 on REAL sequence databases (UniProt release 2026_02), not on the
# synthetic random-residue FASTA every previous pyhmmer number came from.
#
# THREE RULES THIS SCRIPT EXISTS TO ENFORCE, each paid for by a wasted run:
#
#  1. MEASURE ON NODE-LOCAL NVMe. /tmp is a 1.4-1.7 TB local volume;
#     posix_fadvise(DONTNEED) is a silent, PARTIAL, NONDETERMINISTIC no-op on
#     Lustre and Lustre read throughput was measured collapsing 16x WITHIN a
#     single 8 GB read. Everything is staged to /tmp and measured there.
#  2. /tmp IS NODE-LOCAL, so PYTHONPATH, the repo and every output must be on
#     SHARED paths. A previous job pointed PYTHONPATH at a login node's /tmp,
#     failed to import on the compute node, and wrote its output somewhere
#     nobody could read.
#  3. REAL AND SYNTHETIC ARE MEASURED IN THE SAME JOB, ON THE SAME NODE. A 2.3x
#     CPU spread was measured on identical work across nodes, so cross-job
#     second-for-second comparison is not admissible. Same node or no claim.
set -u
W=/storage/scratch1/7/avandevoorde3/p1
D=$W/data
R=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
T=/tmp/p1real.$SLURM_JOB_ID
cd "$R" || exit 1
export PYTHONPATH=$W/libs
python3 -c "import pyhmmer; print('pyhmmer', pyhmmer.__version__)" \
  || { echo "FATAL: pyhmmer not importable on $(hostname)"; exit 1; }

echo "node $(hostname)  start $(date -Is)"
lscpu | grep -E '^Model name|^CPU\(s\):'
df -h /tmp | tail -1
mkdir -p "$T" || exit 1
trap 'rm -rf "$T"' EXIT

echo "=== staging to node-local NVMe ==="
cp "$D/uniprot_sprot.fasta" "$T/" || exit 1
cp "$D/Pfam-A.hmm"          "$T/" || exit 1
cp "$W/p1_2gb.fasta"        "$T/" || exit 1
cp "$W/p1_8gb.fasta"        "$T/" || exit 1
time gunzip -c "$D/uniref50.fasta.gz" > "$T/uniref50.fasta" || exit 1
ls -la "$T"

# Record-aligned truncations, so the real subsets are valid FASTA and directly
# size-comparable to the existing synthetic 2 GB and 8 GB points.
python3 - "$T" <<'PY'
import os, sys
T = sys.argv[1]
src = os.path.join(T, "uniref50.fasta")
for gb, name in ((2, "uniref50_2gb.fasta"), (8, "uniref50_8gb.fasta")):
    dst = os.path.join(T, name)
    target = int(gb * 1e9)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        left = target
        while left > 0:
            b = fi.read(min(1 << 24, left))
            if not b:
                break
            fo.write(b); left -= len(b)
        # back up to the last record boundary so the file ends cleanly
        fo.flush(); size = fo.tell()
        tail = 1 << 20
        fo.seek(max(0, size - tail)); chunk = fo.read()
        cut = chunk.rfind(b"\n>")
        if cut >= 0:
            fo.truncate(max(0, size - tail) + cut + 1)
    print(name, os.path.getsize(dst), flush=True)
PY
ls -la "$T"
df -h /tmp | tail -1

SPROT=$T/uniprot_sprot.fasta
HMM=$T/Pfam-A.hmm

run () {   # run <label> <fasta> <out> <extra args...>
  local label=$1 fasta=$2 out=$3; shift 3
  echo "================ $label ================ $(date -Is)"
  python3 experiments/bench_p1_consumer_retention.py \
      --fasta "$fasta" --label "$label" --out "results/$out" \
      --query-fasta "$SPROT" "$@" || echo "FAILED: $label"
  echo "---- $label done $(date -Is)"
}

# Which computes each rung pays for is a WALL-CLOCK decision, not a scientific
# one, and the costs are wildly uneven -- measured on 0.29 GB Swiss-Prot, a
# random 200-mer is 1.6 s, ATP-synthase-beta (P06576) is 29.7 s and a Pfam
# profile search is 37.4 s. Scaled to 27 GB the profile search alone is ~1 h,
# so the expensive computes are measured where they are affordable and the
# large rungs carry the cheap ones. Every share below is therefore reported
# against a NAMED compute; none of them is "the" activation share.
#
# Order matters: the headline real-vs-synthetic pair at 2 GB runs FIRST so a
# preemption cannot cost the comparison the paper actually needs.
run uniref50_real_2gb  "$T/uniref50_2gb.fasta" bench_p1_real_uniref50_2gb.json \
    --query-ids P69905,P0CG48,P06576 --n-sampled-queries 6 --hmm "$HMM"
run synthetic_2gb      "$T/p1_2gb.fasta"       bench_p1_synth_2gb_samenode.json \
    --query-ids P69905,P0CG48,P06576 --n-sampled-queries 6 --hmm "$HMM"
run sprot_real_0.29gb  "$SPROT"                bench_p1_real_sprot.json \
    --query-ids P69905,P0CG48,P06576 --n-sampled-queries 6 --hmm "$HMM"
run uniref50_real_8gb  "$T/uniref50_8gb.fasta" bench_p1_real_uniref50_8gb.json \
    --query-ids P69905,P0CG48 --hmm "$HMM"
run synthetic_8gb      "$T/p1_8gb.fasta"       bench_p1_synth_8gb_samenode.json \
    --query-ids P69905,P0CG48
# Full UniRef50. Cheap computes only: P06576 and the profile search each cost
# ~100x a random query, and at 27 GB either alone would be most of an hour.
run uniref50_real_full "$T/uniref50.fasta"     bench_p1_real_uniref50_full.json \
    --query-ids P69905

echo "ALL DONE $(date -Is)"
