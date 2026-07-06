"""
atomagents_noai_runner.py — True no-LLM AtomAgents baseline runner.

Calls run_baseline() from workflows/exp_2/run_baseline.py directly —
no AutoGen agents, no vLLM servers, no LLM calls of any kind.
LAMMPS runs for lattice constant + screw dislocation relaxation are
executed exactly as in the agentic workflow.

Results are saved to results/noai_atomagents_<run_id>.json, compatible
with the summary_*.json schema used by plot_walltime_comparison.py.

Usage
-----
    cd agent-hpc-runtime/
    python experiments/atomagents_noai_runner.py [--n-runs N] [--run-id ID]

Options
-------
    --n-runs   N    number of timed runs to execute  (default: 5)
    --run-id   ID   base run ID; suffixed with _00, _01, ...
    --results-dir   output directory for JSON files (default: results/)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_AA_DIR = _REPO_ROOT / "workloads" / "AtomAgents"

# Both repo root and AtomAgents must be on sys.path
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_AA_DIR))


def run_single(run_id: str, results_dir: Path) -> dict:
    """
    Execute one no-LLM AtomAgents run and return the summary dict.

    Must chdir to workloads/AtomAgents/ before importing run_baseline,
    because LAMMPS scripts reference '../potential_repository/' relative
    to that directory.
    """
    orig_cwd = os.getcwd()
    os.chdir(str(_AA_DIR))

    try:
        # Import here (after chdir) so relative paths resolve correctly.
        # Re-importing is safe: Python caches the module.
        from atomagents.workflows.exp_2.run_baseline import run_baseline

        print(f"\n[noai] ── Run {run_id} ──")
        t_start = time.perf_counter()
        results = run_baseline()
        elapsed = time.perf_counter() - t_start

        summary = {
            "wall_time_s": elapsed,
            "workflow":    "",           # matches AtomAgents summary schema
            "mode":        "noai_scripted",
            "condition":   "noai",
            "run_id":      run_id,
            "lat_consts": {
                label: results[label]["lat_const"]
                for label in results
                if label != "comparison_plot" and isinstance(results[label], dict)
            },
        }

        out_path = results_dir / f"noai_atomagents_{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[noai] Finished in {elapsed:.1f}s → {out_path}")
        return summary

    finally:
        os.chdir(orig_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-LLM AtomAgents baseline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of timed runs")
    parser.add_argument("--run-id", default=None,
                        help="Base run ID (suffixed _00, _01, ...); "
                             "defaults to a random UUID prefix")
    parser.add_argument("--results-dir",
                        default=str(_REPO_ROOT / "results"),
                        help="Directory to write noai_atomagents_*.json files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_id = args.run_id or str(uuid.uuid4())[:8]
    wall_times = []

    for i in range(args.n_runs):
        run_id = f"{base_id}_{i:02d}"
        summary = run_single(run_id, results_dir)
        wall_times.append(summary["wall_time_s"])

    print("\n" + "=" * 50)
    print(f"  {args.n_runs} no-LLM run(s) complete")
    print(f"  Wall times: {[f'{t:.1f}s' for t in wall_times]}")
    if len(wall_times) > 1:
        import statistics
        print(f"  Mean: {statistics.mean(wall_times):.1f}s  "
              f"Stdev: {statistics.stdev(wall_times):.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
