#!/bin/bash
#SBATCH -J tandem_dataworker
#SBATCH -A gts-ag117
#SBATCH -p cpu-medium
#SBATCH -q embers
#SBATCH -N 1 -n 1 -c 4
#SBATCH --mem=96G
#SBATCH -t 00:50:00
#SBATCH -o logs/residency_data_worker_%j.log
#
# T4b (D1/D2/D4) at FULL SIZE on the real 3.32 GB potential.
#
# WHY A JOB AND NOT THE LOGIN NODE. The login node enforces a 4 GiB per-user
# cgroup cap (memory.max = 4294967296), which SIGKILLed a full-size LAMMPS run
# twice during the COW work. w_eam4_big.fs activates to ~17 GB, so it cannot be
# measured there at all -- only a scaled-down synthetic can.
#
# embers is preemptible and free. Short and small on purpose so it does not
# penalise queue priority. NEVER inferno.
#
# --tmp /tmp is node-local NVMe (1.4 TB). It is also the only place
# posix_fadvise(DONTNEED) works; on Lustre scratch it is a silent no-op, which
# would make "cold" and "warm" two identically-cached rungs.
set -Eeuo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
ATOMS_PYTHON=/storage/project/r-ag117-0/shared/agent_hpc/envs/atoms/bin/python
echo "node $(hostname)  $(date -Is)"
echo "cgroup memory.max: $(cat /sys/fs/cgroup/$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)/memory.max 2>/dev/null || echo n/a)"
"$ATOMS_PYTHON" -u experiments/bench_residency_data_worker.py \
    --potential workloads/AtomAgents/potential_repository/w_eam4_big.fs \
    --tmp /tmp \
    --out results/bench_residency_data_worker_BIG.json
echo "exit $? at $(date -Is)"
