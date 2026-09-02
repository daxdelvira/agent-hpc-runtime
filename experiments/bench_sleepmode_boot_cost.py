#!/usr/bin/env python3
"""What makes a --residency boot slower?  Same node, same job, three arms.

WHY THIS EXISTS.  Tandem trial t03 (11573.8 s) was 1.57x SLOWER than the
paired Blackwell baselines (n=4, mean 7362.4 s).  The loss is entirely in
per-boot cost, not in any policy decision:

    tandem t03   model_load:qwen_72b   1725.9 / 1670.8 / 1696.1 s
    baseline t09 model_load:qwen_72b   1025.5 / 1020.5 / 1025.5 s

and vLLM's own "Loading weights took" agrees -- 1472-2335 s in tandem against
912-1035 s in baseline.  Meanwhile the residency mechanism never ran: the whole
3.2 h trial issued ZERO POST /sleep and ZERO POST /wake_up (one GET
/is_sleeping, nothing else).  So t03 paid a boot penalty and collected no wake.

THERE ARE TWO CONFIGURATION DIFFERENCES BETWEEN THE ARMS, NOT ONE, and an
earlier draft of this bench had only noticed the first:

  1. --enable-sleep-mode  -> vLLM builds the engine on CuMemAllocator.
  2. model_orchestrator.py:236-239 -- `sleep_mode` is inferred from EITHER
     --enable-sleep-mode OR VLLM_SERVER_DEV_MODE=1, and when it is true the
     orchestrator STOPS setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
     So the baseline boots with expandable_segments and the tandem arm does
     not, purely as a side effect of exposing the /sleep route.

Either could produce the gap, and they are confounded in production because
--residency turns on both at once.  Three arms separate them:

    A  base       no sleep flag, no dev mode   -> expandable_segments ON
    B  tandem     sleep flag + dev mode        -> expandable_segments OFF, CuMem
    C  devmode    dev mode only, no sleep flag -> expandable_segments OFF, no CuMem

    A vs B  reproduces the production gap on ONE node (kills the node confound)
    B vs C  isolates CuMemAllocator
    A vs C  isolates expandable_segments

THE NODE CONFOUND IS THE REASON FOR THE WHOLE BENCH.  t03 ran on
atl1-1-03-020-2-0 and every baseline ran on atl1-1-03-020-6-0.  A 1.6x
weight-load difference between two nodes of the same type is entirely
plausible here: this project has already measured 4.0x cold-boot spread across
nodes for one identical model, and a 16.3x Lustre read collapse inside a single
8 GB file.  Until arm A and arm B are measured on the SAME node in the SAME
job, "sleep mode costs 660 s" is a correlation with n=1 node per arm.

Arms are interleaved (A,B,C,A,B,C...) rather than blocked, so a monotonic
drift -- a filesystem warming up, a neighbour arriving -- adds roughly the same
amount to every arm instead of to whichever ran last.

WHAT THE ANSWER CHANGES.  If the gap does not reproduce on one node, the t03
regression is a pairing artifact and the fix is to top up baselines on the
tandem node.  If it does reproduce, then whichever of the two flags carries it
must become a per-model decision made where the park decision is made -- today
--residency turns both on for all three engines unconditionally, including for
engines that are never parked, which for t03 was all of them.

No park, no wake, no agent: this measures cold-boot cost only.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "workloads" / "AtomAgents"))

from model_configs import MODELS_BLACKWELL_SWAP                    # noqa: E402
from atomagents.runtime.model_orchestrator import ModelOrchestrator  # noqa: E402

OUT = Path("results/bench_sleepmode_boot_cost.json")

# Each arm is (name, adds --enable-sleep-mode, sets VLLM_SERVER_DEV_MODE=1).
ARMS = {
    "base":    (False, False),
    "tandem":  (True, True),
    "devmode": (False, True),
}


def build(model_key: str, arm: str) -> dict:
    add_sleep, dev_mode = ARMS[arm]
    cfg = dict(MODELS_BLACKWELL_SWAP[model_key])
    cfg["extra_args"] = list(cfg["extra_args"])
    if add_sleep:
        cfg["extra_args"].append("--enable-sleep-mode")
    env: dict = {}
    if dev_mode:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    # Set in EVERY arm.  vLLM's engine-core startup cap defaults to 600 s and
    # has already killed a boot in this project; it is a timeout, not an
    # allocator setting, so holding it constant costs the comparison nothing.
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = "2400"
    cfg["extra_env"] = env
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen_72b",
                    choices=sorted(MODELS_BLACKWELL_SWAP))
    ap.add_argument("--reps", type=int, default=2,
                    help="rounds; each round runs every arm once, in order")
    ap.add_argument("--arms", default="base,tandem,devmode")
    ap.add_argument("--preflight", action="store_true",
                    help="do everything except boot an engine, then exit 0. "
                         "Runs in seconds, so it can be scheduled on a small "
                         "ask and prove the whole path before a 6 h hold is "
                         "spent discovering a typo.")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print(f"unknown arms: {bad}", file=sys.stderr)
        return 2

    node = socket.gethostname().split(".")[0]
    # Breadcrumbs on stdout before anything that can fail.  Run 12709114 died
    # here in 4 s and left NO traceback in the job log -- #SBATCH -o and -e
    # pointed at one file and the streams clobbered each other -- so the only
    # evidence of where it got to was the ABSENCE of the JSON below.  Print
    # progress, and write the JSON before the first thing that can throw.
    print(f"[bench] node={node} cwd={Path.cwd()}", flush=True)
    print(f"[bench] python={sys.executable}", flush=True)

    gpus: list[str] = []
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
    except Exception as exc:                                   # noqa: BLE001
        # nvidia-smi resolves off PATH, and `conda activate` rewrites PATH.
        # A missing binary here must say so, not raise an unlabelled
        # FileNotFoundError three frames deep.
        print(f"[bench] FATAL: nvidia-smi not runnable: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print(f"[bench] PATH={os.environ.get('PATH', '')}",
              file=sys.stderr, flush=True)
        return 3
    print(f"[bench] gpus={gpus}", flush=True)
    # An untyped --gres once put a job that named the Blackwell partition onto a
    # Tesla V100.  Assert it here rather than discover it in the numbers.
    if not any("Blackwell" in g for g in gpus):
        print(f"[bench] REFUSING: not a Blackwell node -- {gpus}",
              file=sys.stderr, flush=True)
        return 2

    rec: dict = {"node": node, "gpus": gpus, "model": args.model,
                 "reps": args.reps, "arms": arms,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "boots": []}

    def save() -> None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rec, indent=2))

    save()
    if args.preflight:
        # Constructing the orchestrator and every arm's config is where a
        # config-shape or import error would surface; do that and stop.
        for arm in arms:
            cfg = build(args.model, arm)
            ModelOrchestrator({"m": cfg})
            print(f"[bench] preflight arm={arm} "
                  f"sleep_flag={'--enable-sleep-mode' in cfg['extra_args']} "
                  f"env={cfg['extra_env']}", flush=True)
        rec["preflight"] = "ok"
        save()
        print(f"[bench] PREFLIGHT OK -- wrote {OUT}", flush=True)
        return 0
    i = 0
    for rep in range(args.reps):
        for arm in arms:
            cfg = build(args.model, arm)
            orch = ModelOrchestrator({"m": cfg})
            # This marker is how the job log is split per boot afterwards --
            # vLLM's own "Loading weights took" lines stream to this stdout.
            print(f"\n=== BOOT {i} arm={arm} rep={rep} ===", flush=True)
            t0 = time.time()
            ok, err = True, None
            try:
                orch.start_model("m")
                orch.wait_until_ready("m", timeout=cfg["load_timeout"])
            except Exception as exc:                       # noqa: BLE001
                ok, err = False, f"{type(exc).__name__}: {exc}"
            wall = time.time() - t0
            rec["boots"].append({"i": i, "arm": arm, "rep": rep, "ok": ok,
                                 "error": err, "wall_s": wall})
            print(f"=== BOOT {i} arm={arm} DONE ok={ok} wall={wall:.1f}s ===",
                  flush=True)
            save()
            try:
                orch.stop_model("m")
            except Exception:                              # noqa: BLE001
                pass
            time.sleep(30)          # let the driver actually release VRAM
            i += 1

    good = [b for b in rec["boots"] if b["ok"]]
    summ: dict = {}
    for arm in arms:
        xs = [b["wall_s"] for b in good if b["arm"] == arm]
        if xs:
            summ[arm] = {"n": len(xs), "mean_wall_s": sum(xs) / len(xs),
                         "walls": xs}
    rec["summary"] = summ
    if "base" in summ and "tandem" in summ:
        r = summ["tandem"]["mean_wall_s"] / summ["base"]["mean_wall_s"]
        summ["ratio_tandem_over_base"] = r
        print(f"\ntandem/base = {r:.3f}  "
              f"({summ['tandem']['mean_wall_s']:.1f} vs "
              f"{summ['base']['mean_wall_s']:.1f} s)")
        print("  production gap to reproduce: 1725.9/1670.8/1696.1 s vs "
              "1025.5/1020.5/1025.5 s = ~1.65x")
    if "devmode" in summ and "base" in summ:
        print(f"devmode/base = "
              f"{summ['devmode']['mean_wall_s'] / summ['base']['mean_wall_s']:.3f}"
              "   (expandable_segments alone)")
    save()
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
