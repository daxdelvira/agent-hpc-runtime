#!/bin/bash
# run_eval_blackwell_20260728.sh — Blackwell hold wrapper (2026-07-28 chain).
#
# Phase 0: vLLM sleep/wake microbenchmark on the Blackwell facet (once) —
#   decision input for the sleep_wake config arm, the mechanism targeting the
#   no_window engine-boot stall that dominates the swap-family taxonomy
#   (eval_stall_taxonomy.csv) and cannot be pooled away on 4-GPU nodes.
# Phase 1+: exec run_eval_blackwell_postfix.sh — the incomplete POST-fix
#   AtomAgents series (exp3 full_system/ablations, exp2 full_system at
#   >=0517f4d): the paper's strongest existing win needs this generation
#   completed.  Resumable; self-skips finished targets.
#
# Facet marker (not a hostname glob): Blackwell node names vary, so the
# bench-done marker is an explicit file, touched only on bench success.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 4 ] && { log "ABORT: only $ngpu GPUs visible (need 4)"; exit 1; }
nvidia-smi -L | grep -qiE "rtx pro 6000|blackwell" \
  || { log "ABORT: not a Blackwell node — facet mismatch."; exit 1; }

# Node health canaries (see node_preflight.sh header; 07-28 lesson).
bash experiments/node_preflight.sh || { log "ABORT: node preflight failed"; exit 1; }

MARKER=results/bench_sleep_wake_blackwell.done
if [ ! -f "$MARKER" ]; then
  log "Phase 0: sleep/wake microbench, Blackwell facet (72B, cap 40 min)"
  if timeout 2400 python3 experiments/bench_sleep_wake.py \
       --model qwen_72b_instruct; then
    touch "$MARKER"
  else
    log "sleep/wake bench failed or timed out (non-fatal, continuing)"
  fi
  pkill -u "$USER" -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -u "$USER" -f "VLLM::" 2>/dev/null || true
  sleep 10
else
  log "Phase 0: Blackwell sleep/wake bench already done — skipping"
fi

log "Phase 1+: handing off to run_eval_blackwell_postfix.sh"
exec bash experiments/run_eval_blackwell_postfix.sh
