#!/usr/bin/env python3
"""Answer the six questions a tandem trial exists to settle, in one command.

Written before the first trial landed so the answer does not depend on
remembering, at 3am, which field lives in which file. The questions, in the
order they gate each other:

  1. Did the residency actor wire at all?          (no -> nothing else means anything)
  2. Did model prefetches still fail on GPU occupancy?
  3. When it evicted, did it PARK or STOP?         (stop -> the GPU was taken but
                                                    the retention benefit was not)
  4. What did the arbitrator decide, and why?
  5. Wall time against the paired baselines, FACETED.
  6. Was the trial actually complete?

USAGE
    python3 scripts/parse_tandem_trial.py                  # newest tandem trial
    python3 scripts/parse_tandem_trial.py <trial_dir> ...

Reads only. Every number printed names the file it came from.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/eval_q1_q4/runs"


def gpu_family(meta: dict) -> str:
    # NOT summary.json -- its gpu_name is None for every AtomAgents trial.
    g = (meta.get("gpus") or [""])[0] or ""
    return ("Blackwell" if "Blackwell" in g
            else "L40S" if "L40S" in g else "unknown")


def baselines(workload: str, fam: str) -> tuple[list[float], list[str]]:
    walls, nodes = [], []
    for d in sorted((RUNS / workload / "baseline").glob("t*/")) if (
            RUNS / workload / "baseline").exists() else []:
        try:
            meta = json.loads((d / "meta.json").read_text())
            summ = json.loads((d / "summary.json").read_text())
        except Exception:
            continue
        if gpu_family(meta) != fam:
            continue
        w = summ.get("wall_time_s")
        if w:
            walls.append(w)
            nodes.append((meta.get("node") or "?").split(".")[0])
    return walls, nodes


def parse(d: Path) -> None:
    d = d.resolve()
    try:
        shown = d.relative_to(ROOT)
    except ValueError:          # a path outside the repo, or given relative
        shown = d
    print(f"\n{'='*74}\n{shown}\n{'='*74}")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    log = (d / "stdout.log").read_text(errors="ignore") if (d / "stdout.log").exists() else ""
    trace = [json.loads(l) for l in (d / "trace.jsonl").open()
             if l.strip()] if (d / "trace.jsonl").exists() else []

    fam = gpu_family(meta)
    print(f"  node {(meta.get('node') or '?').split('.')[0]}  |  {fam}  |  "
          f"commit {(meta.get('git_commit') or '?')[:7]}  |  exit {meta.get('exit_code')}")

    # -- 6. completeness first: an incomplete trial invalidates the rest ----
    complete = bool(summ) and summ.get("wall_time_s")
    ntool = sum(1 for e in trace if e.get("event_type") == "tool_call")
    print(f"\n  [6] COMPLETE?          summary={'yes' if summ else 'NO'}  "
          f"tool_calls={ntool}")
    if not complete:
        print("      !! no summary.json -- trial did not finish. Nothing below "
              "is comparable to a completed baseline.")

    # -- 1. did the actor wire ---------------------------------------------
    wired = "TANDEM: VllmModelActor wired" in log
    sleepep = "sleep endpoint enabled" in log
    print(f"\n  [1] ACTOR WIRED?       {'YES' if wired else 'NO'}"
          f"   (stdout.log: 'TANDEM: VllmModelActor wired')")
    print(f"      sleep endpoint injected for --residency: "
          f"{'yes' if sleepep else 'NO -- park will 404 and downgrade to stop'}")
    if not wired:
        print("      !! the arm ran WITHOUT the actor; it is a full_system run "
              "under a different name.")

    # -- 2. did prefetches still fail on occupancy -------------------------
    occ = len(re.findall(r"Cannot start .*? occupied by", log))
    gpusnotfreed = len(re.findall(r"GpusNotFreed", log))
    print(f"\n  [2] OCCUPANCY FAILURES {occ}  ('Cannot start ... occupied by')")
    print(f"      GpusNotFreed raised   {gpusnotfreed}  "
          f"(the actor's own loud refusal — it tried and says what it tried)")

    # -- 3. park or stop ---------------------------------------------------
    parks = len(re.findall(r"\bparked\b|park_L1|action.{0,4}park", log))
    stops = len(re.findall(r"downgrade_reason|action.{0,4}stop", log))
    four04 = len(re.findall(r"/sleep\?level=\d.*?404|404.*?/sleep", log))
    print(f"\n  [3] EVICTION MODE      park~{parks}  stop~{stops}  "
          f"sleep-404~{four04}")
    if four04:
        print("      !! /sleep 404 -- the actor cannot park, so every eviction "
              "is a stop and the next use pays a full cold boot.")

    # -- 4. arbitrator decisions -------------------------------------------
    dec = [e for e in trace if e.get("event_type") == "prefetch_decision"]
    reasons: dict[str, int] = {}
    for e in dec:
        r = (e.get("payload") or {}).get("reason") or "?"
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\n  [4] DECISIONS          {len(dec)} recorded")
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
        print(f"        {n:4d}  {r[:66]}")
    comp = [e for e in trace if e.get("event_type") == "prefetch_completed"]
    failed = sum(1 for e in comp if (e.get("payload") or {}).get("status") == "failed")
    print(f"      prefetch_completed={len(comp)} of which status=failed: {failed}"
          f"   (an event is emitted for EVERY terminal state, so check status)")

    # -- 5. wall time, faceted ---------------------------------------------
    w = summ.get("wall_time_s")
    print(f"\n  [5] WALL TIME          {w if w else '--'}"
          f"   (summary.json key wall_time_s)")
    if w:
        wl = d.parent.parent.name
        bw, bn = baselines(wl, fam)
        if len(bw) >= 2:
            m, sd = st.mean(bw), st.stdev(bw)
            print(f"      paired baseline {wl}/{fam}: n={len(bw)} mean={m:.1f} "
                  f"sd={sd:.1f}  nodes={sorted(set(bn))}")
            print(f"      speedup {m/w:.4f}x   z={(m-w)/sd:+.2f}")
            if len(bw) < 5:
                print(f"      !! n={len(bw)} is too small for a significance "
                      f"claim; report the pair, not a p-value.")
            if len(set(bn)) > 1:
                print("      !! baselines span multiple nodes -- name them.")
        else:
            print(f"      no paired {fam} baseline for {wl} "
                  f"(found {len(bw)}); collect one before quoting a speedup.")


def main() -> int:
    args = sys.argv[1:]
    if args:
        dirs = [Path(a) for a in args]
    else:
        dirs = sorted((p.parent for p in RUNS.glob("*/tandem/*/meta.json")),
                      key=lambda p: p.name)[-1:]
    if not dirs:
        print("no tandem trials on disk yet "
              f"(looked under {RUNS}/*/tandem/)")
        return 0
    for d in dirs:
        parse(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
