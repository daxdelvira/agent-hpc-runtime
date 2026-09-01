#!/usr/bin/env python3
"""Scan the project's measurement data and emit one JSON inventory.

Separated from rendering on purpose: this file knows the filesystem, the
renderer knows the page. Re-run it and the atlas is current -- there is no
hand-maintained list to drift.

Everything here is DERIVED. If a number is not in the tree, it is not in the
output; absent fields say "unknown" rather than being estimated.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/eval_q1_q4/runs"
EVAL = ROOT / "results/eval_q1_q4"

# GPU identity does NOT come from summary.json -- that field is None for every
# AtomAgents trial. It comes from meta.json's gpus[0] string. This bit us once.
def gpu_family(meta: dict) -> str:
    g = (meta.get("gpus") or [""])[0] or ""
    if "Blackwell" in g:
        return "Blackwell"
    if "L40S" in g:
        return "L40S"
    return "unknown"


def du(p: Path) -> int:
    try:
        return int(subprocess.run(["du", "-sb", str(p)], capture_output=True,
                                  text=True, timeout=300).stdout.split()[0])
    except Exception:
        return -1


def scan_trials() -> dict:
    trials, workloads = [], defaultdict(lambda: defaultdict(list))
    for meta_p in RUNS.glob("*/*/*/meta.json"):
        d = meta_p.parent
        parts = d.relative_to(RUNS).parts
        if len(parts) != 3:
            continue
        wl, cfg, tid = parts
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            meta = {}
        summ = {}
        sp = d / "summary.json"
        if sp.exists():
            try:
                summ = json.loads(sp.read_text())
            except Exception:
                pass
        m = re.search(r"(\d{8})-(\d{6})", tid)
        date = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None
        rec = {
            "workload": wl, "config": cfg, "trial": tid, "date": date,
            "gpu": gpu_family(meta),
            "node": (meta.get("node") or "").split(".")[0] or None,
            "commit": meta.get("git_commit", "")[:7] or None,
            "exit_code": meta.get("exit_code"),
            "has_summary": sp.exists(),
            "has_trace": (d / "trace.jsonl").exists(),
            "wall_time_s": summ.get("wall_time_s"),
            "divergence_count": summ.get("divergence_count"),
            "prefetch_started": summ.get("prefetch_started"),
            "prefetch_failed": summ.get("prefetch_failed"),
            "files": sorted(p.name for p in d.iterdir() if p.is_file()),
        }
        trials.append(rec)
        workloads[wl][cfg].append(rec)
    return {"trials": trials, "workloads": workloads}


def scan_csvs() -> list:
    out = []
    for p in sorted(EVAL.glob("*.csv")):
        try:
            with p.open(newline="") as f:
                r = csv.reader(f)
                header = next(r, [])
                rows = list(r)
        except Exception:
            continue
        cols = []
        for i, name in enumerate(header):
            vals = [row[i] for row in rows if i < len(row) and row[i] != ""]
            kind, uniq = "empty", 0
            if vals:
                uniq = len(set(vals))
                try:
                    [float(v) for v in vals[:200]]
                    kind = "numeric"
                except ValueError:
                    kind = "categorical" if uniq <= max(12, len(vals) // 20) else "text"
            cols.append({"name": name, "kind": kind, "distinct": uniq,
                         "filled": len(vals),
                         "sample": sorted(set(vals))[:4] if kind == "categorical" else
                                   (vals[:2] if vals else [])})
        out.append({"file": p.name, "bytes": p.stat().st_size,
                    "rows": len(rows), "columns": cols,
                    "mtime": datetime.fromtimestamp(p.stat().st_mtime,
                                                    timezone.utc).strftime("%Y-%m-%d")})
    return out


def scan_bench() -> dict:
    fams = defaultdict(lambda: {"n": 0, "bytes": 0, "examples": []})
    for p in sorted((ROOT / "results").glob("*.json")):
        stem = re.sub(r"_atl1[-\w.]*$", "", p.stem)
        stem = re.sub(r"_\d{6,}$", "", stem)
        stem = re.sub(r"_(BIG|SMOKE|small|full)$", "", stem)
        fam = "summary_eval_*" if stem.startswith("summary_eval") else stem
        f = fams[fam]
        f["n"] += 1
        f["bytes"] += p.stat().st_size
        if len(f["examples"]) < 3:
            f["examples"].append(p.name)
    return dict(fams)


def scan_groups() -> list:
    groups = []
    for sub in sorted((ROOT / "results").iterdir()):
        if not sub.is_dir():
            continue
        n = sum(1 for _ in sub.rglob("*") if _.is_file())
        groups.append({"name": sub.name, "files": n, "bytes": du(sub)})
    return groups


# ---------------------------------------------------------------- provenance
# THE MOST IMPORTANT DISTINCTION IN THIS ARCHIVE. A number's trustworthiness is
# set by how it was produced, not by how large it is:
#   measured  -- a real trial or benchmark on real hardware
#   replay    -- a RECORDED trace re-scored under a different policy. Bounded by
#                something that actually happened, but the alternative outcome
#                never ran.
#   synthetic -- a GENERATED need sequence. Answers "what would have to be true",
#                and must never be reported as a measured speedup.
SYNTHETIC = {
    "sweep_policy_regime": "scripts/sweep_policy_regime.py",
    "sweep_arbitration_regimes": "scripts/sweep_arbitration_regimes.py",
    "search_ceiling_regime": "scripts/search_ceiling_regime.py",
    "probe_arbitration_regime": "scripts/probe_arbitration_regime.py",
    "frontier_ceiling": "scripts/search_ceiling_regime.py",
}
REPLAY = {
    "replay_retention_policy": "scripts/replay_retention_policy.py",
    "replay_two_class": "scripts/replay_two_class.py",
    "replay_byte_staging_atomagents": "scripts/replay_byte_staging.py",
    "replay_divergence": "scripts/replay_divergence.py",
    "replay_predictor": "scripts/replay_predictor.py",
    "replay_predictor_gating": "scripts/replay_predictor.py",
}
# Arms belong to eras. Everything collected so far predates Tandem: the system
# was a PREFETCHER (predict identity, stage early). Tandem is a RETENTION
# system (hold what is already paid for, arbitrate one budget). The `tandem`
# arm is the first of the new era and has not produced a trial yet.
RESIDENCY_ARMS = {"tandem"}


def era_of(arm: str) -> str:
    return "residency" if arm in RESIDENCY_ARMS else "prefetch"


def scan_provenance() -> dict:
    out = {"replay": [], "synthetic": []}
    rp = ROOT / "results"
    for stem, script in sorted(SYNTHETIC.items()):
        f = rp / f"{stem}.json"
        if f.exists():
            out["synthetic"].append({"file": f.name, "script": script,
                                     "bytes": f.stat().st_size})
    for stem, script in sorted(REPLAY.items()):
        f, d = rp / f"{stem}.json", rp / stem
        if f.exists():
            out["replay"].append({"file": f.name, "script": script,
                                  "bytes": f.stat().st_size})
        if d.is_dir():
            for c in sorted(d.iterdir()):
                if c.is_file():
                    out["replay"].append({"file": f"{stem}/{c.name}",
                                          "script": script,
                                          "bytes": c.stat().st_size})
    return out


def scan_figures() -> tuple[dict, dict]:
    """Figure -> caption (the authored claim) and figure -> data lineage.

    Both were originally extracted with throwaway scripts, which meant the atlas
    could not be regenerated by anyone but its author. They live here now so the
    whole page rebuilds from `scan.py` + `render.py` and nothing else.
    """
    tex_dir = ROOT / "sc-workshop-paper/paper"
    captions: dict = {}

    def balanced(txt: str, i: int) -> str:
        depth = 0
        for j in range(i, len(txt)):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
                if depth == 0:
                    return txt[i + 1:j]
        return ""

    for f in sorted(tex_dir.glob("*.tex")) if tex_dir.exists() else []:
        txt = f.read_text(errors="ignore")
        for m in re.finditer(r"\\begin\{figure\*?\}", txt):
            depth, k = 1, m.end()
            while k < len(txt) and depth:
                if txt.startswith(r"\begin{figure", k):
                    depth += 1
                elif txt.startswith(r"\end{figure", k):
                    depth -= 1
                k += 1
            env = txt[m.end():k]
            img = re.search(r"\\includegraphics[^{]*\{([^}]+)\}", env)
            lab = re.search(r"\\label\{(fig:[^}]+)\}", env)
            ci = env.find("\\caption")
            cap = balanced(env, env.index("{", ci)) if ci >= 0 else ""
            cap = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", cap)
            cap = re.sub(r"\s+", " ", re.sub(r"[\\{}]", "", cap)).strip()
            if lab:
                captions[lab.group(1)] = {
                    "img": img.group(1).split("/")[-1] if img else None,
                    "tex": f.name, "caption": cap}

    # lineage: which generator reads which table / data file
    mf = ROOT / "scripts/figures/make_figures.py"
    sources: dict = {}
    if mf.exists():
        src = mf.read_text()
        bounds = [(m.group(1), m.start())
                  for m in re.finditer(r"^def ([a-z_0-9]+)\(", src, re.M)]
        bounds.append(("<eof>", len(src)))
        helpers = {}
        for i, (n, st) in enumerate(bounds[:-1]):
            if n.startswith("fig_"):
                continue
            helpers[n] = sorted(set(re.findall(
                r"md_tables\([\"']([^\"']+)[\"']\)", src[st:bounds[i + 1][1]])))
        for i, (n, st) in enumerate(bounds[:-1]):
            if not n.startswith("fig_"):
                continue
            body = src[st:bounds[i + 1][1]]
            tabs = set(re.findall(r"md_tables\([\"']([^\"']+)[\"']\)", body))
            for h, t in helpers.items():
                if t and re.search(r"\b" + h + r"\(", body):
                    tabs |= set(t)
            files = sorted({f for f in re.findall(
                r"[\"']([A-Za-z0-9_./-]+\.(?:csv|json))[\"']", body)})
            outn = re.findall(r"[\"'](fig-[a-z0-9-]+)[\"']", body)
            sources[n] = {"tables": sorted(tabs), "files": files,
                          "output": outn[0] if outn else None}
    return captions, sources


def scan_learned() -> dict:
    """The transition table is DATA the runtime consumes, and it is derived from
    the traces rather than measured. It carries its own provenance header, which
    is read verbatim here so the atlas cannot drift from the shipped file."""
    f = ROOT / "runtime/predictor/data/learned_transitions.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {
        "file": "runtime/predictor/data/learned_transitions.json",
        "version": d.get("version"),
        "n_traces": d.get("n_traces"),
        "n_tool_events": d.get("n_tool_events"),
        "generated_at": d.get("generated_at"),
        "offset_basis": d.get("offset_basis") or {},
        "synthetic_filter": d.get("synthetic_filter") or {},
        "tool_sources": len(d.get("tool_transitions") or {}),
        "model_sources": len(d.get("model_transitions") or {}),
        "backups": sorted(x.name for x in f.parent.iterdir()
                          if x.is_dir() and x.name.startswith("_pre")),
    }


def scan_paper() -> dict:
    snap = Path("/storage/project/r-ag117-0/shared/agent_hpc/paper-snapshots/"
                "workshop-char-2026-08-14")
    man = snap / "MANIFEST.sha256"
    figs = sorted((ROOT / "sc-workshop-paper/paper/figures").glob("*.pdf"))
    mf = ROOT / "scripts/figures/make_figures.py"
    gens, srcs = [], []
    if mf.exists():
        txt = mf.read_text()
        gens = re.findall(r"^def (fig_[a-z0-9_]+)", txt, re.M)
        srcs = sorted({m for m in re.findall(r"[A-Za-z0-9_./-]+\.(?:csv|json)", txt)
                       if not m.startswith("...")})
    return {
        "snapshot_dir": str(snap),
        "exists": snap.exists(),
        "manifest_entries": sum(1 for _ in man.open()) if man.exists() else 0,
        "bytes": du(snap) if snap.exists() else 0,
        "tar_bytes": (snap.parent / f"{snap.name}.tar.gz").stat().st_size
                     if (snap.parent / f"{snap.name}.tar.gz").exists() else 0,
        "figure_pdfs": len(figs),
        "generators": gens,
        "data_sources": srcs,
    }


def main() -> None:
    t = scan_trials()
    wl_summary = []
    for wl, cfgs in sorted(t["workloads"].items()):
        recs = [r for c in cfgs.values() for r in c]
        gp = Counter(r["gpu"] for r in recs)
        dates = sorted(r["date"] for r in recs if r["date"])
        wl_summary.append({
            "workload": wl,
            "family": "atomagents" if wl.startswith("atomagents") else
                      ("chemgraph" if wl.startswith("chemgraph") else "other"),
            "trials": len(recs),
            "configs": sorted(cfgs),
            "n_configs": len(cfgs),
            "gpus": dict(gp),
            "with_summary": sum(1 for r in recs if r["has_summary"]),
            "with_trace": sum(1 for r in recs if r["has_trace"]),
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "eras": dict(Counter(era_of(c) for c in cfgs)),
            "per_config": {c: {
                "era": era_of(c),
                "n": len(rs),
                "gpus": dict(Counter(r["gpu"] for r in rs)),
                "with_summary": sum(1 for r in rs if r["has_summary"]),
                "walls": [r["wall_time_s"] for r in rs if r["wall_time_s"]],
            } for c, rs in sorted(cfgs.items())},
        })

    by_month = Counter(r["date"][:7] for r in t["trials"] if r["date"])
    inv = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True,
                                 cwd=ROOT).stdout.strip() or "unknown",
        "totals": {
            "trials": len(t["trials"]),
            "workloads": len(wl_summary),
            "results_bytes": du(ROOT / "results"),
            "logs_bytes": du(ROOT / "logs"),
            "trace_files": sum(1 for r in t["trials"] if r["has_trace"]),
        },
        "workloads": wl_summary,
        "csvs": scan_csvs(),
        "bench": scan_bench(),
        "groups": scan_groups(),
        "by_month": dict(sorted(by_month.items())),
        "provenance": scan_provenance(),
        "paper": scan_paper(),
        "learned": scan_learned(),
        "eras": dict(Counter(era_of(r["config"]) for r in t["trials"])),
        "trial_files": dict(Counter(f for r in t["trials"] for f in r["files"])),
    }
    caps, figsrc = scan_figures()
    (ROOT / "scripts/atlas/figcaptions.json").write_text(json.dumps(caps, indent=1))
    (ROOT / "scripts/atlas/figsources.json").write_text(json.dumps(figsrc, indent=1))
    out = ROOT / "scripts/atlas/inventory.json"
    out.write_text(json.dumps(inv, indent=1))
    print(f"wrote {out} — {inv['totals']['trials']} trials, "
          f"{len(inv['csvs'])} CSVs, {len(inv['bench'])} bench families, "
          f"{len(caps)} figure captions, {len(figsrc)} figure generators")


if __name__ == "__main__":
    main()
