#!/usr/bin/env python3
"""Render inventory.json into the Data Atlas page.

Rendering is separate from scanning so the page can be restyled without
re-walking 155 MB, and rescanned without touching the design.
"""
import json, html
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INV = json.loads((ROOT / "scripts/atlas/inventory.json").read_text())
GLOSS = json.loads((ROOT / "scripts/atlas/glossary.json").read_text())
FIGSRC = json.loads((ROOT / "scripts/atlas/figsources.json").read_text())
FIGCAP = json.loads((ROOT / "scripts/atlas/figcaptions.json").read_text())
BENCH = json.loads((ROOT / "scripts/atlas/benchmarks.json").read_text())["families"]
OUT = ROOT / "scripts/atlas/data_atlas.html"

TEAL, OCHRE = "#1c6b74", "#9a5b18"
e = html.escape


def gb(n): return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def bar(frac, cls="teal"):
    return f'<span class="minibar"><i class="{cls}" style="width:{max(frac*100,1.2):.1f}%"></i></span>'


# ---------------------------------------------------------------- sections
wls = sorted(INV["workloads"], key=lambda w: -w["trials"])
maxt = max(w["trials"] for w in wls)
tot = INV["totals"]

mixed = [w for w in wls if len([k for k in w["gpus"] if k != "unknown"]) > 1]
nosum = tot["trials"] - sum(w["with_summary"] for w in wls)

tiles = [
    ("Trials on disk", f"{tot['trials']}", "run directories under results/eval_q1_q4/runs"),
    ("Workloads", f"{tot['workloads']}", "two families: atomagents, chemgraph"),
    ("Trace files", f"{tot['trace_files']}", "trial dirs carrying trace.jsonl"),
    ("Measurement data", gb(tot["results_bytes"]), f"plus {gb(tot['logs_bytes'])} of logs"),
]

rows = []
for w in wls:
    fam = w["family"]
    cls = "teal" if fam == "atomagents" else "ochre"
    gpus = w["gpus"]
    gpu_cells = " ".join(
        f'<span class="chip">{e(k)} {v}</span>' for k, v in sorted(gpus.items()))
    hazard = ('<span class="chip hazard" title="This workload spans two GPU families. '
              'Identical work differs by up to 2.3-4.0x across node types, so these '
              'trials must be faceted, never pooled.">&#9888; mixed GPU</span>'
              if len([k for k in gpus if k != "unknown"]) > 1 else "")
    cov = w["with_summary"] / w["trials"] if w["trials"] else 0
    rows.append(f"""<tr>
 <td><span class="dot {cls}"></span><span class="mono">{e(w['workload'])}</span>{hazard}</td>
 <td class="num">{w['trials']} {bar(w['trials']/maxt, cls)}</td>
 <td class="num">{w['n_configs']}</td>
 <td>{gpu_cells}</td>
 <td class="num">{w['with_summary']}<span class="sub"> / {w['trials']}</span> {bar(cov, cls)}</td>
 <td class="mono sub">{e(w['first'] or '?')} &rarr; {e(w['last'] or '?')}</td>
</tr>""")

# per-workload config detail
details = []
for w in wls:
    cls = "teal" if w["family"] == "atomagents" else "ochre"
    cfg_rows = []
    for c, d in sorted(w["per_config"].items(), key=lambda kv: -kv[1]["n"]):
        walls = d["walls"]
        wall = (f"{sum(walls)/len(walls):,.0f} s"
                f"<span class='sub'> n={len(walls)}</span>") if walls else "<span class='sub'>--</span>"
        gp = " ".join(f'<span class="chip">{e(k)} {v}</span>' for k, v in sorted(d["gpus"].items()))
        era_chip = ('<span class="chip era-res">residency</span>'
                    if d.get("era") == "residency" else "")
        cfg_rows.append(f"<tr><td class='mono'>{e(c)}{era_chip}</td><td class='num'>{d['n']}</td>"
                        f"<td class='num'>{d['with_summary']}</td><td>{gp}</td>"
                        f"<td class='num'>{wall}</td></tr>")
    details.append(f"""<details class="grp">
<summary><span class="dot {cls}"></span><span class="mono">{e(w['workload'])}</span>
<span class="sub">{w['trials']} trials &middot; {w['n_configs']} arms</span></summary>
<div class="scroll"><table class="inner">
<thead><tr><th>arm</th><th class="num">trials</th><th class="num">with summary</th>
<th>GPU split</th><th class="num">mean wall</th></tr></thead>
<tbody>{''.join(cfg_rows)}</tbody></table></div></details>""")

# CSV browser
csvs = []
for c in INV["csvs"]:
    cols = []
    for col in c["columns"]:
        k = col["kind"]
        sample = ", ".join(e(str(s)) for s in col["sample"][:3])
        cols.append(f"""<li><span class="mono cn">{e(col['name'])}</span>
<span class="kind k-{k}">{k}</span>
<span class="sub">{col['distinct']} distinct &middot; {col['filled']} filled</span>
{f'<span class="sub sm">{sample}</span>' if sample else ''}</li>""")
    meta = GLOSS["files"].get(c["file"], {})
    claim = (f'<p class="claim"><span class="eyebrow">Supports</span> {e(meta["claim"])}</p>'
             f'<p class="lede sm">{e(meta.get("what",""))}</p>') if meta.get("claim") else ""
    caution = (f'<div class="note sm"><b>&#9888; Read with care.</b> {e(meta["caution"])}</div>'
               if meta.get("caution") else "")
    csvs.append(f"""<details class="grp">
<summary><span class="mono">{e(c['file'])}</span>
<span class="sub">{c['rows']:,} rows &middot; {len(c['columns'])} columns &middot; {c['bytes']//1024} KB &middot; updated {e(c['mtime'])}</span></summary>
<div class="pad">{claim}{caution}</div>
<ul class="cols">{''.join(cols)}</ul></details>""")

# month chart
months = INV["by_month"]
mmax = max(months.values()) if months else 1
mbars = "".join(
    f'<div class="mb"><div class="mbar" style="height:{v/mmax*100:.0f}%" '
    f'title="{e(k)}: {v} trials"></div><span class="mono sub">{e(k[5:])}</span></div>'
    for k, v in sorted(months.items()))

# bench families
bench = sorted(INV["bench"].items(), key=lambda kv: -kv[1]["n"])
brows = "".join(
    f"<tr><td class='mono'>{e(k)}</td><td class='num'>{v['n']}</td>"
    f"<td class='num sub'>{v['bytes']//1024} KB</td>"
    f"<td class='mono sub sm'>{e(', '.join(v['examples'][:2]))}</td></tr>"
    for k, v in bench[:14])

groups = "".join(
    f"<tr><td class='mono'>results/{e(g['name'])}</td><td class='num'>{g['files']}</td>"
    f"<td class='num sub'>{gb(g['bytes']) if g['bytes']>0 else '--'}</td></tr>"
    for g in sorted(INV["groups"], key=lambda g: -g["files"]))

files_in_trial = "".join(
    f"<tr><td class='mono'>{e(k)}</td><td class='num'>{v}</td>"
    f"<td class='num sub'>{v/tot['trials']*100:.0f}%</td></tr>"
    for k, v in sorted(INV["trial_files"].items(), key=lambda kv: -kv[1]))


# ---- eras ---------------------------------------------------------------
eras = INV.get("eras", {})
n_pref, n_res = eras.get("prefetch", 0), eras.get("residency", 0)

# ---- replay & synthetic -------------------------------------------------
prov = INV.get("provenance", {"replay": [], "synthetic": []})
def prov_rows(items):
    return "".join(
        f"<tr><td class='mono'>{e(i['file'])}</td>"
        f"<td class='mono sub sm'>{e(i['script'])}</td>"
        f"<td class='num sub'>{i['bytes']//1024 or 1} KB</td></tr>" for i in items)

# ---- learned artifacts --------------------------------------------------
LRN = INV.get("learned", {})
sf = LRN.get("synthetic_filter", {})
ob = LRN.get("offset_basis", {})
learned_rows = "".join(
    f"<tr><td class='mono'>{e(k)}</td><td class='mono sub'>{e(str(v))}</td></tr>"
    for k, v in [
        ("version", LRN.get("version")),
        ("n_traces", LRN.get("n_traces")),
        ("n_tool_events", LRN.get("n_tool_events")),
        ("tool sources", LRN.get("tool_sources")),
        ("model sources", LRN.get("model_sources")),
        ("offset basis (tools)", ob.get("tool_transitions")),
        ("offset basis (models)", ob.get("llm_transitions") or ob.get("model_transitions")),
        ("synthetic filter applied", sf.get("applied")),
        ("filter rule", sf.get("rule")),
        ("generated", LRN.get("generated_at")),
    ] if v is not None)
backups = " ".join(f'<span class="chip">{e(b)}</span>' for b in LRN.get("backups", []))

# ---- paper --------------------------------------------------------------
pap = INV.get("paper", {})
# the regex picks up bare names and full paths for the same file; keep the
# fullest form of each so the table names something a reader can open.
srcs, seen = [], set()
for sname in sorted(pap.get("data_sources", []), key=len, reverse=True):
    base = sname.split("/")[-1]
    if base not in seen:
        seen.add(base); srcs.append(sname)
src_rows = "".join(f"<tr><td class='mono'>{e(x)}</td></tr>" for x in sorted(srcs))
gen_chips = " ".join(f'<span class="chip">{e(g[4:])}</span>'
                     for g in sorted(pap.get("generators", [])))


# ---- glossary -----------------------------------------------------------
gl = []
for term, d in GLOSS["terms"].items():
    vals = "".join(
        f'<li><span class="mono cn">{e(k)}</span><span class="sub">{e(v)}</span></li>'
        for k, v in d["values"].items())
    gl.append(f"""<details class="grp">
<summary><span class="mono">{e(term)}</span>
<span class="sub">{e(d['why'][:96])}{'&hellip;' if len(d['why'])>96 else ''}</span></summary>
<div class="pad"><p class="lede sm">{e(d['why'])}</p>
<p class="sub sm">Defined in <code>{e(d['source'])}</code></p>
{f'<ul class="cols vals">{vals}</ul>' if vals else ''}</div></details>""")

# ---- figure lineage -----------------------------------------------------
LIN = {"sim": ("simulator", "ochre"), "data": ("measured", "teal"),
       "const": ("constants", "")}
figrows = []
by_out = {v.get("output"): (k, v) for k, v in FIGSRC.items()}
for lab, cap in sorted(FIGCAP.items()):
    gen, src = by_out.get(cap["img"], (None, {"tables": [], "files": []}))
    if src["tables"]:
        kind, lineage = "sim", ", ".join(t.replace(".md", "") for t in src["tables"])
    elif src["files"]:
        kind, lineage = "data", ", ".join(f.split("/")[-1] for f in src["files"])
    else:
        kind, lineage = "const", "written into the generator"
    name, cls = LIN[kind]
    figrows.append(f"""<tr>
 <td><span class="mono">{e(lab)}</span><br><span class="sub sm">{e(cap['tex'])}</span></td>
 <td><span class="claimtext">{e(cap['caption'][:230])}{'&hellip;' if len(cap['caption'])>230 else ''}</span></td>
 <td><span class="chip {cls}">{e(name)}</span><br><span class="mono sub sm">{e(lineage)}</span></td>
</tr>""")
n_sim = sum(1 for v in FIGSRC.values() if v["tables"])
n_dat = sum(1 for v in FIGSRC.values() if v["files"])
n_con = len(FIGSRC) - n_sim - n_dat


# ---- annotated benchmarks ----------------------------------------------
STATUS = {
 "stands":     ("stands", "ok",   "The claim holds as recorded."),
 "superseded": ("superseded", "warn", "A later measurement replaced it."),
 "withdrawn":  ("withdrawn", "bad", "The claim it produced was retracted."),
 "invalid":    ("invalid", "bad", "The run itself is unusable."),
}
counts = Counter(v["status"] for v in BENCH.values())
bench_cards = []
for fam, d in sorted(BENCH.items(), key=lambda kv: (kv[1]["status"] != "stands", kv[0])):
    label, cls, _ = STATUS[d["status"]]
    n = INV["bench"].get(fam, {}).get("n")
    files = f'<span class="sub">{n} file{"s" if n and n > 1 else ""}</span>' if n else ""
    bench_cards.append(f"""<details class="grp">
<summary><span class="mono">{e(fam)}</span>
<span class="chip st-{cls}">{e(label)}</span>{files}</summary>
<div class="pad">
 <p class="claim"><span class="eyebrow">Question</span>{e(d['question'])}</p>
 <p class="lede sm"><b>Why it was taken.</b> {e(d['why'])}</p>
 <p class="lede sm"><b>What it produced.</b> {e(d['produced'])}</p>
 <p class="claim"><span class="eyebrow">Claim</span>{e(d['claim'])}</p>
 {f'<p class="lede sm note-inline">{e(d["note"])}</p>' if d.get('note') else ''}
</div></details>""")

HTML = f"""<title>Residency Data Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bitter:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#f4f6f7; --surface:#ffffff; --ink:#141a1f; --muted:#5a6672;
  --rule:#dde3e6; --rule-soft:#eaeef0;
  --teal:{TEAL}; --ochre:{OCHRE};
  --hazard:#a5321f; --hazard-bg:#fbeae7;
  --shadow:0 1px 2px rgba(20,26,31,.06), 0 8px 24px -16px rgba(20,26,31,.25);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#10141a; --surface:#161a21; --ink:#e6ecef; --muted:#8d9aa6;
    --rule:#262d36; --rule-soft:#1d232b;
    --teal:#4fb3bd; --ochre:#d99b45;
    --hazard:#f2857c; --hazard-bg:#2a1a18;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
  }}
}}
:root[data-theme="dark"] {{
  --paper:#10141a; --surface:#161a21; --ink:#e6ecef; --muted:#8d9aa6;
  --rule:#262d36; --rule-soft:#1d232b;
  --teal:#4fb3bd; --ochre:#d99b45;
  --hazard:#f2857c; --hazard-bg:#2a1a18;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--ink);
  font:400 15px/1.6 "IBM Plex Sans", system-ui, sans-serif;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ display:grid; grid-template-columns:200px minmax(0,1fr);
  gap:40px; max-width:1180px; margin:0 auto; padding:48px 28px 96px; }}
@media (max-width:860px) {{ .wrap {{ grid-template-columns:1fr; gap:24px; padding:28px 16px 64px; }}
  nav.rail {{ position:static !important; }} }}
nav.rail {{ position:sticky; top:32px; align-self:start;
  border-left:2px solid var(--rule); padding-left:14px; }}
nav.rail a {{ display:block; color:var(--muted); text-decoration:none;
  font-size:13px; padding:5px 0; }}
nav.rail a:hover, nav.rail a:focus-visible {{ color:var(--teal); }}
h1 {{ font:600 30px/1.2 Bitter, Georgia, serif; margin:0 0 6px; text-wrap:balance;
  letter-spacing:-.01em; }}
h2 {{ font:600 19px/1.3 Bitter, Georgia, serif; margin:0 0 4px; text-wrap:balance; }}
.lede {{ color:var(--muted); max-width:64ch; margin:0 0 8px; }}
.eyebrow {{ font:500 11px/1 "IBM Plex Mono", monospace; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); }}
section {{ margin-top:44px; }}
section > .eyebrow {{ display:block; margin-bottom:8px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:12px; margin-top:22px; }}
.tile {{ background:var(--surface); border:1px solid var(--rule); border-radius:6px;
  padding:16px 18px; box-shadow:var(--shadow); }}
.tile .v {{ font:600 26px/1.1 Bitter, Georgia, serif; font-variant-numeric:tabular-nums;
  display:block; margin:6px 0 4px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ text-align:left; font:500 11px/1 "IBM Plex Mono", monospace;
  letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
  padding:0 10px 8px 0; border-bottom:1px solid var(--rule); white-space:nowrap; }}
td {{ padding:9px 10px 9px 0; border-bottom:1px solid var(--rule-soft);
  vertical-align:middle; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums;
  white-space:nowrap; }}
.mono {{ font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:13px; }}
.sub {{ color:var(--muted); font-size:12px; }}
.sm {{ font-size:11px; }}
.scroll {{ overflow-x:auto; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
  margin-right:8px; vertical-align:1px; }}
.dot.teal, .minibar i.teal {{ background:var(--teal); }}
.dot.ochre, .minibar i.ochre {{ background:var(--ochre); }}
.minibar {{ display:inline-block; width:56px; height:6px; border-radius:3px;
  background:var(--rule); margin-left:8px; vertical-align:middle; overflow:hidden; }}
.minibar i {{ display:block; height:100%; border-radius:3px; }}
.chip {{ display:inline-block; font:500 11px/1 "IBM Plex Mono", monospace;
  border:1px solid var(--rule); border-radius:3px; padding:3px 6px;
  color:var(--muted); margin-right:5px; white-space:nowrap; }}
.chip.st-ok {{ color:var(--teal); border-color:var(--teal); }}
.chip.st-warn {{ color:var(--ochre); border-color:var(--ochre); }}
.chip.st-bad {{ color:var(--hazard); border-color:var(--hazard); background:var(--hazard-bg); }}
.note-inline {{ border-left:2px solid var(--rule); padding-left:12px; color:var(--muted); }}
.pad {{ padding:0 16px 12px; }}
.claim {{ margin:4px 0 6px; font-size:14px; }}
.claim .eyebrow {{ margin-right:8px; }}
.claimtext {{ font-size:13px; line-height:1.5; display:block; max-width:52ch; }}
.note.sm {{ font-size:13px; padding:9px 13px; }}
ul.cols.vals li {{ display:block; }}
ul.cols.vals .cn {{ display:inline-block; min-width:190px; }}
.chip.teal {{ color:var(--teal); border-color:var(--teal); }}
.chip.ochre {{ color:var(--ochre); border-color:var(--ochre); }}
.chip.era-res {{ color:var(--teal); border-color:var(--teal); margin-left:8px; }}
.chip.hazard {{ color:var(--hazard); border-color:var(--hazard);
  background:var(--hazard-bg); margin-left:8px; cursor:help; }}
.legend {{ display:flex; gap:16px; margin:14px 0 6px; font-size:13px;
  color:var(--muted); flex-wrap:wrap; }}
details.grp {{ background:var(--surface); border:1px solid var(--rule);
  border-radius:6px; margin-bottom:8px; }}
details.grp > summary {{ cursor:pointer; padding:12px 16px; list-style:none;
  display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
details.grp > summary::-webkit-details-marker {{ display:none; }}
details.grp > summary::before {{ content:"+"; font-family:"IBM Plex Mono",monospace;
  color:var(--muted); width:12px; }}
details.grp[open] > summary::before {{ content:"\\2212"; }}
details.grp > summary:focus-visible {{ outline:2px solid var(--teal); outline-offset:-2px; }}
table.inner {{ margin:0 16px 14px; width:calc(100% - 32px); }}
ul.cols {{ list-style:none; margin:0; padding:0 16px 14px;
  display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:2px; }}
ul.cols li {{ padding:6px 0; border-bottom:1px solid var(--rule-soft);
  display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }}
.cn {{ font-weight:500; }}
.kind {{ font:500 10px/1 "IBM Plex Mono", monospace; letter-spacing:.06em;
  text-transform:uppercase; border-radius:3px; padding:3px 5px;
  border:1px solid var(--rule); color:var(--muted); }}
.k-numeric {{ color:var(--teal); border-color:var(--teal); }}
.k-categorical {{ color:var(--ochre); border-color:var(--ochre); }}
.months {{ display:flex; gap:10px; align-items:flex-end; height:110px;
  padding:10px 0 0; }}
.mb {{ display:flex; flex-direction:column; align-items:center; gap:6px;
  flex:0 0 44px; height:100%; justify-content:flex-end; }}
.mbar {{ width:100%; background:var(--teal); border-radius:4px 4px 0 0; min-height:3px;
  transition:opacity .15s; }}
.mb:hover .mbar {{ opacity:.75; }}
.note {{ border-left:3px solid var(--hazard); background:var(--hazard-bg);
  padding:12px 16px; border-radius:0 5px 5px 0; margin-top:16px; font-size:14px; }}
.note b {{ color:var(--hazard); }}
code {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px;
  background:var(--rule-soft); padding:1px 5px; border-radius:3px; }}
footer {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--rule);
  color:var(--muted); font-size:13px; }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">
<nav class="rail" aria-label="Sections">
  <a href="#overview">Overview</a>
  <a href="#eras">Two eras</a>
  <a href="#workloads">Workloads</a>
  <a href="#arms">Arms &amp; coverage</a>
  <a href="#csvs">Derived tables</a>
  <a href="#glossary">Field glossary</a>
  <a href="#bench">Measurements</a>
  <a href="#derived">Replay &amp; synthetic</a>
  <a href="#paper">Paper provenance</a>
  <a href="#figures">Figure lineage</a>
  <a href="#anatomy">Trial anatomy</a>
  <a href="#rules">Reading rules</a>
</nav>

<main>
<header>
  <span class="eyebrow">agent-hpc-runtime &middot; {e(INV['commit'])}</span>
  <h1>Residency Data Atlas</h1>
  <p class="lede">Everything this project has measured, and how to read it without
  drawing a wrong conclusion. Regenerated from the filesystem &mdash; nothing here
  is hand-maintained.</p>
</header>

<section id="overview">
<span class="eyebrow">Overview</span>
<h2>What exists</h2>
<div class="tiles">
{''.join(f'<div class="tile"><span class="eyebrow">{e(l)}</span><span class="v">{e(v)}</span><span class="sub">{e(s)}</span></div>' for l, v, s in tiles)}
</div>
<div class="months" role="img" aria-label="Trials collected per month">{mbars}</div>
<p class="sub">Trials collected per month. Hover for counts.</p>
{f'<div class="note"><b>&#9888; {nosum} of {tot["trials"]} trial directories have no <code>summary.json</code>.</b> They were started but did not produce a parsed result &mdash; preempted, failed, or still mid-flight when collection stopped. Any denominator taken from directory counts will be wrong; count summaries, not directories.</div>' if nosum else ''}
</section>

<section id="eras">
<span class="eyebrow">Two eras</span>
<h2>Everything here is the old system</h2>
<p class="lede">The project changed shape. The first system was a
<b>prefetcher</b> &mdash; predict which resource is needed next, stage it early.
Tandem is a <b>retention</b> system &mdash; hold what has already been paid for
and arbitrate one memory budget across models and data. They answer different
questions, and their trials are not comparable.</p>
<div class="tiles">
  <div class="tile"><span class="eyebrow">Prefetch era</span>
    <span class="v">{n_pref}</span>
    <span class="sub">every trial in this archive. Arms are
    <span class="mono sm">baseline, full_system, plan_only, transition_only,
    naive_prefetch, no_plan, oracle, megammap_stage</span> and their ablations.</span></div>
  <div class="tile"><span class="eyebrow">Residency era (Tandem)</span>
    <span class="v">{n_res}</span>
    <span class="sub">the <span class="mono sm">tandem</span> arm. Registered and
    queued; no trial has completed yet.</span></div>
</div>
<div class="note"><b>&#9888; No end-to-end Tandem measurement exists yet.</b>
Every number in this archive describes the prefetch system. The residency
components &mdash; ledger, arbitrator, horizon estimator, the two residency
actors &mdash; are built and unit-tested, and their behaviour is currently
evidenced by microbenchmarks and simulation, not by a completed trial.</div>
</section>

<section id="workloads">
<span class="eyebrow">Workloads</span>
<h2>Eight workloads, two families</h2>
<div class="legend">
  <span><span class="dot teal"></span>atomagents &mdash; materials, LAMMPS + swapped LLMs</span>
  <span><span class="dot ochre"></span>chemgraph &mdash; chemistry, ASE + MACE</span>
</div>
<div class="scroll"><table>
<thead><tr><th>workload</th><th class="num">trials</th><th class="num">arms</th>
<th>GPU split</th><th class="num">with summary</th><th>collected</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{f'<div class="note"><b>&#9888; {len(mixed)} workloads span both GPU families.</b> Identical work differs by up to 2.3&ndash;4.0&times; between L40S and Blackwell, so trials from these workloads must be faceted by GPU before any comparison. Note that <code>summary.json</code> has no <code>gpu_name</code> field &mdash; it is <code>None</code> for every AtomAgents trial. GPU identity comes only from <code>meta.json</code> &rarr; <code>gpus[0]</code>.</div>' if mixed else ''}
</section>

<section id="arms">
<span class="eyebrow">Arms &amp; coverage</span>
<h2>What each workload was run with</h2>
<p class="lede">Arms are experimental conditions, not repeats. Mean wall time is
computed only over trials that produced a <code>summary.json</code>.</p>
{''.join(details)}
</section>

<section id="csvs">
<span class="eyebrow">Derived tables</span>
<h2>The eight parsed CSVs</h2>
<p class="lede">Produced by <code>scripts/parse_eval_traces.py</code> from the trial
traces. Every column below is read from the file itself, so the list cannot drift
from the data.</p>
{''.join(csvs)}
</section>

<section id="glossary">
<span class="eyebrow">Field glossary</span>
<h2>What the columns mean</h2>
<p class="lede">The vocabulary is not self-explanatory &mdash; <code>gate_on_demand</code>
and <code>residual_partial</code> carry specific operational meanings. Every
definition below cites the file that defines it; nothing here is inferred.</p>
{''.join(gl)}
</section>

<section id="bench">
<span class="eyebrow">Measurements</span>
<h2>Standalone measurements</h2>
<p class="lede">Almost every design decision in this project was settled by one of
these, not by a campaign. Each entry below says what question it was taken to
answer, what it produced, and whether the claim still stands.</p>
<div class="legend">
  <span><span class="chip st-ok">stands</span> {counts.get('stands', 0)}</span>
  <span><span class="chip st-warn">superseded</span> {counts.get('superseded', 0)}</span>
  <span><span class="chip st-bad">withdrawn</span> {counts.get('withdrawn', 0)}</span>
  <span><span class="chip st-bad">invalid</span> {counts.get('invalid', 0)}</span>
</div>
<div class="note"><b>&#9888; Withdrawn and invalid entries are kept on purpose.</b>
Knowing that a measurement was taken and later retracted is worth more than not
knowing it existed &mdash; two of this project's retracted numbers were rediscovered
by someone re-deriving them from a file nobody had flagged.</div>
{''.join(bench_cards)}
<h2 style="margin-top:28px">All families by file count</h2>
<p class="lede sm">Including the {sum(1 for k in INV['bench'] if k.startswith('summary_eval'))}
per-trial <code>summary_eval_*</code> files, which are copies of trial summaries
rather than measurements.</p>
<div class="scroll"><table>
<thead><tr><th>family</th><th class="num">files</th><th class="num">size</th><th>examples</th></tr></thead>
<tbody>{brows}</tbody></table></div>
<h2 style="margin-top:28px">Result directories</h2>
<div class="scroll"><table>
<thead><tr><th>directory</th><th class="num">files</th><th class="num">size</th></tr></thead>
<tbody>{groups}</tbody></table></div>
</section>

<section id="derived">
<span class="eyebrow">Replay &amp; synthetic</span>
<h2>Data that was computed, not collected</h2>
<p class="lede">Roughly a third of the project's conclusions come from files no
GPU produced. They are legitimate evidence, but they carry different weight, and
mixing them with measurements is how this project produced two of its retracted
numbers.</p>

<h2 style="margin-top:26px;font-size:16px">Replay &mdash; real traces, counterfactual policy</h2>
<p class="lede">A recorded need sequence re-scored under a policy that never ran.
Bounded by something that actually happened, so the <em>sequence</em> is real;
the <em>outcome</em> is projected. Offline-optimal arms (Belady) are upper
bounds on any online policy, never achievable results.</p>
<div class="scroll"><table>
<thead><tr><th>file</th><th>produced by</th><th class="num">size</th></tr></thead>
<tbody>{prov_rows(prov['replay'])}</tbody></table></div>

<h2 style="margin-top:26px;font-size:16px">Synthetic &mdash; generated need sequences</h2>
<p class="lede">No recorded trace at all. The measured constants inside these
(resource sizes, load costs, seconds-per-GB) are real; the access pattern is a
draw from a distribution.</p>
<div class="scroll"><table>
<thead><tr><th>file</th><th>produced by</th><th class="num">size</th></tr></thead>
<tbody>{prov_rows(prov['synthetic'])}</tbody></table></div>
<h2 style="margin-top:26px;font-size:16px">Learned artifacts &mdash; derived from traces, consumed by the runtime</h2>
<p class="lede">The transition table is not a result; it is an <em>input</em> the
predictor reads at run time. It is derived from the trace corpus, so a change in
what the corpus contains silently changes what the runtime predicts &mdash; which
is why it carries its own provenance header, read verbatim below.</p>
<div class="scroll"><table>
<thead><tr><th>field</th><th>value</th></tr></thead>
<tbody>{learned_rows}</tbody></table></div>
<p class="sub" style="margin-top:10px">Dated backups kept alongside it: {backups}</p>
<div class="note"><b>&#9888; Three provenance defects have been found in this one
file, and each produced plausible numbers.</b> Offsets were counted over a mixed
event stream rather than same-kind subsequences, so a tool's real successor fell
outside the window. The corpus was ~74% synthetic harness runs, which inflated
the model base rate toward 1.0 and made conditioning look worthless. And
<code>offset_decay</code> is derived table-wide, so a confidence at offset&nbsp;3
for one workload is partly a property of every other workload in the corpus.
Both the basis and the filter are now declared in the header, and loading a table
that lacks either raises rather than defaulting.</div>

<div class="note"><b>&#9888; A synthetic result answers "what would have to be
true", never "what happened".</b> Large t-statistics in these files measure
reproducibility across seeds, not confidence about the real workload. They must
never be reported as measured speedups.</div>
</section>

<section id="paper">
<span class="eyebrow">Paper provenance</span>
<h2>What went into the workshop paper</h2>
<p class="lede">Submitted 2026-08-14. The submitted state is frozen in an
immutable snapshot outside this repo, because <code>results/</code> and
<code>logs/</code> are gitignored &mdash; a git tag would have preserved the code
and none of the data.</p>
<div class="tiles">
  <div class="tile"><span class="eyebrow">Snapshot</span>
    <span class="v">{pap.get('manifest_entries', 0):,}</span>
    <span class="sub">checksummed files &middot; {gb(pap.get('bytes', 0))} on disk,
    {gb(pap.get('tar_bytes', 0))} compressed</span></div>
  <div class="tile"><span class="eyebrow">Figures</span>
    <span class="v">{pap.get('figure_pdfs', 0)}</span>
    <span class="sub">PDFs, from {len(pap.get('generators', []))} generators in
    <span class="mono sm">scripts/figures/make_figures.py</span></span></div>
  <div class="tile"><span class="eyebrow">Data sources</span>
    <span class="v">{len(srcs)}</span>
    <span class="sub">files feed every figure in the paper</span></div>
</div>
<p class="lede" style="margin-top:18px">The whole paper rests on this short list.
Extracting the tarball and running its <code>reproduce.sh</code> regenerates all
26 figures byte-identically, with no network and no cluster.</p>
<div class="scroll"><table>
<thead><tr><th>file read by the figure code</th></tr></thead>
<tbody>{src_rows}</tbody></table></div>
<p class="sub" style="margin-top:14px">Figure generators: {gen_chips}</p>
<div class="note"><b>&#9888; Not every figure is data-driven.</b> Some carry
constants written into the generator rather than read from a file. Check the
generator before citing a figure as a measurement, and prefer the underlying
CSV.</div>
<p class="sub" style="margin-top:14px">Snapshot path:
<code>{e(pap.get('snapshot_dir', '?'))}</code></p>
</section>

<section id="figures">
<span class="eyebrow">Figure lineage</span>
<h2>Which file supports which claim</h2>
<p class="lede">Each paper figure, the claim it makes (its own caption), and where
its numbers come from. This is the chain a reader has to trust.</p>
<div class="tiles">
  <div class="tile"><span class="eyebrow">From the simulator</span><span class="v">{n_sim}</span>
    <span class="sub">read <code>results_tables/*.md</code>, generated by
    <span class="mono sm">make_results_tables.py</span> over
    <span class="mono sm">sim_residency_v2.py</span></span></div>
  <div class="tile"><span class="eyebrow">From measured data</span><span class="v">{n_dat}</span>
    <span class="sub">read an eval CSV or a benchmark JSON directly</span></div>
  <div class="tile"><span class="eyebrow">Constants in the generator</span><span class="v">{n_con}</span>
    <span class="sub">values written into the plotting code</span></div>
</div>
<div class="note"><b>&#9888; Most figures in the paper are simulator output, not
measured trials.</b> The simulator's resource sizes and load costs are measured
constants, but its access pattern is generated. A figure sourced from
<code>results_tables</code> answers "what would have to be true", and cannot be
cited as an observed speedup. Check this column before quoting any figure.</div>
<div class="scroll"><table>
<thead><tr><th>figure</th><th>claim it makes</th><th>numbers come from</th></tr></thead>
<tbody>{''.join(figrows)}</tbody></table></div>
</section>

<section id="anatomy">
<span class="eyebrow">Trial anatomy</span>
<h2>What a trial directory contains</h2>
<p class="lede">Every trial lives at
<code>results/eval_q1_q4/runs/&lt;workload&gt;/&lt;arm&gt;/&lt;trial&gt;/</code>.
Coverage is uneven &mdash; check before assuming a file is present.</p>
<div class="scroll"><table>
<thead><tr><th>file</th><th class="num">trials</th><th class="num">coverage</th></tr></thead>
<tbody>{files_in_trial}</tbody></table></div>
</section>

<section id="rules">
<span class="eyebrow">Reading rules</span>
<h2>Four ways this data has already misled someone</h2>
<p class="lede">Each of these produced a number that had to be retracted. They are
properties of the data, not of the analyst.</p>
<div class="note"><b>Never pool GPU families.</b> Identical work differs by up to
2.3&ndash;4.0&times; between L40S and Blackwell. A pooled speedup is a hardware-mix
artifact &mdash; one arm's 0.82&times; became 0.997&times; once faceted.</div>
<div class="note"><b>Count summaries, not directories.</b> {nosum} of
{tot['trials']} trial directories never produced a parsed result.</div>
<div class="note"><b>AtomAgents tool calls are emitted twice.</b> Roughly 0.35 s
apart, from one site that fires on <em>observation</em> rather than execution. Read
raw, this invents a reuse at 0.35&nbsp;s &mdash; the most valuable-looking distance
there is. No time threshold separates it from genuine repeats, which occur as close
as 0.946&nbsp;s; only <code>metrics.csv</code> adjudicates.</div>
<div class="note"><b>Trials before commit <code>96f5f28</code> under-report
divergence.</b> 59 pre-fix traces record zero divergences while the fixed detector
finds 71 misses in them. The two eras must not be pooled.</div>
</section>

<footer>
Generated {e(INV['generated'])} from <code>{e(INV['commit'])}</code> by
<code>scripts/atlas/scan.py</code> &rarr; <code>scripts/atlas/render.py</code>.
To refresh: re-run both, then republish to the same URL.
</footer>
</main>
</div>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} ({len(HTML)//1024} KB)")
