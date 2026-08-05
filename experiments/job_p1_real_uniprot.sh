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

# RUNSET splits the work by MEMORY FOOTPRINT, because the scheduler routes by
# it: a request over ~64 GB leaves cpu-small for cpu-large, which is far more
# contended. Peak RSS is ~1x the activated structure (verified: releasing the
# block returns almost nothing to the OS, but the warm rung reuses the freed
# arena rather than growing, so 0.288 GB Swiss-Prot peaks at 0.577 GB not
# 1.15 GB). So `small` -- everything up to 8 GB, ~18 GB activated -- fits a
# 64 GB cpu-small job, and only full UniRef50 (~52 GB activated) needs the big
# partition.
RUNSET=${RUNSET:-small}
echo "=== staging to node-local NVMe (RUNSET=$RUNSET) ==="
cp "$D/uniprot_sprot.fasta" "$T/" || exit 1
cp "$D/Pfam-A.hmm"          "$T/" || exit 1

if [ "$RUNSET" = "full" ]; then
  time gunzip -c "$D/uniref50.fasta.gz" > "$T/uniref50.fasta" || exit 1
else
  cp "$W/p1_2gb.fasta" "$T/" || exit 1
  cp "$W/p1_8gb.fasta" "$T/" || exit 1
  # Only the first ~8.1 GB is needed for the subsets, so decompress that much
  # rather than all 27 GB. SIGPIPE from the truncated consumer is expected.
  gunzip -c "$D/uniref50.fasta.gz" 2>/dev/null | head -c 8200000000 \
      > "$T/uniref50_head.fasta"
  [ -s "$T/uniref50_head.fasta" ] || exit 1
fi
ls -la "$T"

# Record-aligned truncations, so the real subsets are valid FASTA and directly
# size-comparable to the existing synthetic 2 GB and 8 GB points. This lives in
# its own file because the inline version opened the destination "wb" and then
# read from it to find the last record boundary -- io.UnsupportedOperation, one
# traceback, and the 8 GB subset silently never created.
if [ "$RUNSET" != "full" ]; then
  python3 experiments/make_fasta_subset.py \
      "$T/uniref50_head.fasta" "$T/uniref50_2gb.fasta" 2 || exit 1
  python3 experiments/make_fasta_subset.py \
      "$T/uniref50_head.fasta" "$T/uniref50_8gb.fasta" 8 || exit 1
fi
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
if [ "$RUNSET" = "full" ]; then
  # Cheap computes only: P06576 and the profile search each cost ~100x a random
  # query, and at 27 GB either alone would be most of an hour.
  run uniref50_real_full "$T/uniref50.fasta"   bench_p1_real_uniref50_full.json \
      --query-ids P69905
else
  run uniref50_real_2gb "$T/uniref50_2gb.fasta" bench_p1_real_uniref50_2gb.json \
      --query-ids P69905,P0CG48,P06576 --n-sampled-queries 6 --hmm "$HMM"
  run synthetic_2gb     "$T/p1_2gb.fasta"      bench_p1_synth_2gb_samenode.json \
      --query-ids P69905,P0CG48,P06576 --n-sampled-queries 6 --hmm "$HMM"
  run sprot_real_0.29gb "$SPROT"               bench_p1_real_sprot.json \
      --query-ids P69905,P0CG48,P06576 --n-sampled-queries 6 --hmm "$HMM"
  run uniref50_real_8gb "$T/uniref50_8gb.fasta" bench_p1_real_uniref50_8gb.json \
      --query-ids P69905,P0CG48 --hmm "$HMM"
  run synthetic_8gb     "$T/p1_8gb.fasta"      bench_p1_synth_8gb_samenode.json \
      --query-ids P69905,P0CG48
fi

echo "ALL DONE $(date -Is)"
