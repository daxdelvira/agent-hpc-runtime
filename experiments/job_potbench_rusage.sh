#!/bin/bash
#SBATCH -J potbench_ru
#SBATCH -A gts-ag117
#SBATCH -p cpu-medium
#SBATCH -q embers
#SBATCH -N 1 -n 1 -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o potbench_rusage_%j.log
#
# Re-run the EAM activation split WITH getrusage, so the "98.1% activation"
# residual is decomposed into transformation (utime) vs RAM->process movement
# (stime).  See the 2026-08-09 block in bench_potential_activation.py.
#
# BOTH BENCHES RUN IN THIS ONE JOB, ON THIS ONE NODE, deliberately.
# The claim being tested is a COMPARISON -- "is the EAM potential's stime/GB the
# same ~0.35 s/GB format-independent constant that bench_format_activation
# measured?"  A 2.3x CPU spread has already been measured between nodes on
# identical parse work (EAM warm 42.83 s on Blackwell vs 98.23 s on
# atl1-1-03-004-2-1), so a constant measured in job A cannot be compared with a
# potential measured in job B.  Same node or no claim.
#
# The potential is read from project NFS, the same filesystem as the original
# 2026-08-03 run, where gate (d) measured posix_fadvise(DONTNEED) genuinely
# working (8.68x read slowdown).  Do NOT move it to Lustre scratch: fadvise is a
# silent partial no-op there and both "cold" rungs would be warm.
set -u

REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
LMP=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/bin/lmp
POT=$REPO/workloads/AtomAgents/potential_repository/w_eam4_big.fs
HOST=$(hostname -s)

cd "$REPO" || exit 1
echo "[job] node=$HOST  $(nproc) cpus  $(date -Is)"
echo "[job] potential: $(ls -l "$POT")"

# --- 1. the EAM potential, instrumented ---------------------------------
echo; echo "########## bench_potential_activation (rusage) ##########"
python3 experiments/bench_potential_activation.py \
    --potential "$POT" \
    --lmp "$LMP" \
    --out "results/bench_potential_activation_rusage_${HOST}.json"
POT_RC=$?
echo "[job] potential bench rc=$POT_RC"

# --- 2. the format constant, SAME NODE ----------------------------------
# Establishes ~0.35 s/GB of stime as the floor to compare the potential
# against.  Node-local /tmp, per the standing methodology rule.
echo; echo "########## bench_format_activation (same node) ##########"
python3 experiments/bench_format_activation.py \
    --workdir /tmp/fmtbench_$$ \
    --repeats 2 \
    --out "results/bench_format_activation_${HOST}.json"
FMT_RC=$?
echo "[job] format bench rc=$FMT_RC"

echo; echo "[job] done $(date -Is)  potential_rc=$POT_RC format_rc=$FMT_RC"
exit $(( POT_RC != 0 || FMT_RC != 0 ))
