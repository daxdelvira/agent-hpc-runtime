"""
test_replay_divergence.py — tests for scripts/replay_divergence.py.

Four layers:

  1. Trace parsing (assumptions A1-A3 in the script docstring): action ordering,
     the "next line is checkpoint_created" discriminator that separates scored
     predictions from staging predictions, and the chemgraph step reconstruction.
  2. Scoring: that the detector scorer's verdicts are the shipped detector's
     verdicts (hit / miss / no-opinion / unresolved), and that the
     chemgraph_inline control scorer has the single-slot, exact-match semantics
     of chemgraph.py:957-1003.
  3. THE CONTROL: over the real corpus, chemgraph_inline must reproduce the
     recorded ChemGraph hit/miss counts exactly.  If this fails, no number the
     script prints for AtomAgents may be published.  Skipped when the corpus is
     not on disk.
  4. THE ATOMAGENTS DIVERGENCE CHARACTERISATION.  The corpus now spans both
     sides of the 96f5f28 detector fix (2026-08-03), so it contains two
     populations: pre-fix trials that suppressed every miss (the e68d52b bug)
     and post-fix trials that record them normally.  These tests pin that split
     — divergences ARE recorded now, each recorded one is the shipped detector's
     own verdict over the same stream, none was manufactured by a duplicate tool
     emission, the guard acted on all of them, and no post-fix trial suppresses.

     This layer replaces a single assertion that used to read
     `assert sum(rec_diverged) == 0` with the comment "the bug's signature".
     That pinned the DEFECT.  Once the defect was fixed the assertion went red
     for the right reason, and it is inverted here rather than deleted.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from runtime.config import RuntimeConfig                       # noqa: E402
from runtime.guard.detector import DivergenceDetector          # noqa: E402
from scripts.replay_divergence import (                        # noqa: E402
    CONTROL_EXPECT,
    RUNS_DIR,
    discover,
    parse_trace,
    reconcile,
    run_control,
    score_chemgraph_inline,
    score_detector,
)

CFG = RuntimeConfig(run_id="test", max_horizon=2)

# 96f5f28 "fix(detector): un-invert divergence semantics broken since e68d52b"
# (2026-08-03).  Trials whose recorded commit descends from this ran with the
# fixed detector and must not suppress a miss.
DETECTOR_FIX_COMMIT = "96f5f28"

# (divergence events, trials recording at least one), audited 2026-09-01 over
# results/eval_q1_q4/runs/atomagents*/*/*/.  All 24 are in
# atomagents_exp3_aligned, from post-96f5f28 commits 357260f and 55d0006.
ATOMAGENTS_DIVERGENCE_SNAPSHOT = (24, 16)

# A same-tool `tool_call` repeat closer together than this is a duplicate
# emission rather than a genuine re-invocation.  Corpus evidence: the 66
# same-tool repeats split cleanly at 0.327s..1.084s (duplicates) vs hundreds
# of seconds (genuine), so any threshold in between selects the same set.
DUPLICATE_EMISSION_WINDOW_S = 2.0


# ---------------------------------------------------------------------------
# Helpers for the AtomAgents divergence characterisation
# ---------------------------------------------------------------------------

def _events(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _recorded_divergences(path: Path) -> list[tuple]:
    """(step, expected_tool, actual_tool, checkpoint_id) per recorded event."""
    return [
        (ev.get("step"),
         (ev.get("payload") or {}).get("expected_tool"),
         (ev.get("payload") or {}).get("actual_tool"),
         (ev.get("payload") or {}).get("checkpoint_id"))
        for ev in _events(path)
        if ev.get("event_type") == "divergence_detected"
    ]


def _replayed_divergences(pt, config: RuntimeConfig) -> list[tuple]:
    """The same tuples, recovered by driving the SHIPPED fixed detector.

    The detector mints its own checkpoint ids, so they are mapped back to the
    real trace ids through Action.checkpoint_id (the id from the
    `checkpoint_created` line that followed the prediction).
    """
    det = DivergenceDetector(scheduler=None, config=config, bus=None)
    real_id: dict[str, str] = {}
    out: list[tuple] = []
    for act in pt.actions:
        if act.kind == "predict":
            ckpt = det.on_prediction(act.prediction, step=act.step)
            real_id[ckpt.checkpoint_id] = act.checkpoint_id
        else:
            hit, _action, ckpt = det.on_tool_about_to_execute(act.tool,
                                                              step=act.step)
            if ckpt is None or hit:
                continue
            out.append((act.step,
                        ckpt.prediction.resources[0].consumer_tool,
                        act.tool,
                        real_id.get(ckpt.checkpoint_id)))
    return out


@functools.lru_cache(maxsize=None)
def _ran_with_fixed_detector(trial_name: str) -> bool | None:
    """True/False if the trial's commit is/is not a descendant of the detector
    fix; None if git cannot resolve it (shallow clone, pruned commit).

    Trial directories are named `t01__<UTC stamp>__<short sha>`.
    """
    parts = trial_name.split("__")
    if len(parts) < 3:
        return None
    sha = parts[-1]
    try:
        if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                           f"{sha}^{{commit}}"],
                          capture_output=True).returncode != 0:
            return None
        return subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
             DETECTOR_FIX_COMMIT, sha],
            capture_output=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# Trace fixtures
# ---------------------------------------------------------------------------

def _ev(event_type, payload, step=None):
    ev = {"timestamp": "2026-01-01T00:00:00", "epoch_time": 0.0,
          "run_id": "r", "event_type": event_type, "payload": payload}
    if step is not None:
        ev["step"] = step
    return ev


def _pred_payload(step, tool, expected_at_step, offset=1, pid="learned"):
    return {
        "step": step,
        "resources": [{
            "resource_id": "r1", "resource_type": "vllm_model", "name": "m",
            "expected_at_step": expected_at_step, "confidence": 0.9,
            "consumer_tool": tool, "consumer_step_offset": offset,
        }],
        "confidence": 0.9, "horizon": 1, "predictor_id": pid,
    }


def _write(tmp_path, name, events):
    p = tmp_path / name
    p.write_text("".join(json.dumps(e) + "\n" for e in events))
    return p


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------

class TestParsing:

    def test_atomagents_shape_predict_then_tool_same_step(self, tmp_path):
        """AtomAgents stamps a step on every event; order is predict then tool."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("llm_call", {"step": 1}, step=1),
            _ev("prediction_result", _pred_payload(1, "plan_task", 3, 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "plan_task"}, step=1),
        ])
        pt = parse_trace(p)
        assert [(a.kind, a.step) for a in pt.actions] == [("predict", 1), ("tool", 1)]
        assert pt.actions[0].checkpoint_id == "c1"
        assert pt.n_steps_inferred == 0

    def test_staging_prediction_without_checkpoint_is_not_scored(self, tmp_path):
        """A2: only a prediction_result immediately followed by checkpoint_created
        became a pending checkpoint.  chemgraph's pool-preboot / cache-stage /
        plan-extraction predictions are followed by prefetch_decision instead."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(0, "run_ase", 0,
                                                   pid="swap_cache_stage:x"), step=0),
            _ev("prefetch_decision", {"resource_id": "r1", "action": "start"}, step=0),
            _ev("tool_call", {"tool": "molecule_name_to_smiles"}),
            _ev("prediction_result", _pred_payload(1, "run_ase", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
        ])
        pt = parse_trace(p)
        assert pt.n_prediction_events == 2
        assert pt.n_scored_predictions == 1
        assert [a.kind for a in pt.actions] == ["tool", "predict"]

    def test_chemgraph_step_reconstruction_from_tool_block(self, tmp_path):
        """A3: chemgraph tool_call carries no step; it is the step of the first
        stepped event in its own tool block."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("tool_call", {"tool": "molecule_name_to_smiles"}),
            _ev("prediction_result", _pred_payload(1, "run_ase", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_end", {"tool": "molecule_name_to_smiles"}),
            _ev("tool_call", {"tool": "smiles_to_coordinate_file"}),
            _ev("prediction_result", _pred_payload(2, "run_ase", 3), step=2),
            _ev("checkpoint_created", {"checkpoint_id": "c2", "step": 2}, step=2),
        ])
        pt = parse_trace(p)
        tools = [a for a in pt.actions if a.kind == "tool"]
        assert [a.step for a in tools] == [1, 2]
        assert pt.n_step_checks == 2
        assert pt.n_step_check_fail == 0
        assert pt.n_steps_inferred == 0

    def test_chemgraph_step_falls_back_to_prev_plus_one(self, tmp_path):
        """A tool call that emitted no runtime events at all has no step to read;
        it falls back to prev+1 and is counted in steps_inferred."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("tool_call", {"tool": "a"}),
            _ev("prediction_result", _pred_payload(1, "b", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "b"}),
            _ev("tool_end", {"tool": "b"}),
        ])
        pt = parse_trace(p)
        tools = [a for a in pt.actions if a.kind == "tool"]
        assert [a.step for a in tools] == [1, 2]
        assert pt.n_steps_inferred == 1

    def test_recorded_outcome_ids_are_captured(self, tmp_path):
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "run_ase", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "run_ase"}, step=2),
            _ev("prediction_validated", {"hit": True, "checkpoint_id": "c1"}, step=2),
        ])
        pt = parse_trace(p)
        assert pt.rec_validated == 1 and pt.rec_validated_ids == {"c1"}
        assert pt.rec_diverged == 0


# ---------------------------------------------------------------------------
# 2. Scoring
# ---------------------------------------------------------------------------

class TestDetectorScorer:
    """The scorer must report exactly what the shipped detector decides."""

    def test_exact_match_is_a_hit(self, tmp_path):
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "run_ase", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "run_ase"}, step=2),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert (s.predictions, s.hits, s.misses, s.unresolved) == (1, 1, 0, 0)

    def test_prefix_match_is_a_hit_not_a_divergence(self, tmp_path):
        """The e68d52b regression scored this as a divergence."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "computation_task", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "computation_task_screw_dislocation"}, step=2),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert (s.hits, s.misses) == (1, 0)

    def test_prefix_requires_underscore_separator(self, tmp_path):
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "run_a", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "run_ase"}, step=2),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert (s.hits, s.misses) == (0, 1)

    def test_true_divergence_is_a_miss(self, tmp_path):
        """The e68d52b regression dropped this on the floor entirely."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "run_ase", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "molecule_name_to_smiles"}, step=2),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert (s.hits, s.misses, s.unresolved) == (0, 1, 0)
        assert s.no_opinion_tool_calls == 0

    def test_out_of_scope_tool_leaves_checkpoint_unresolved(self, tmp_path):
        """A tool firing before the prediction's target step is not its business;
        with nothing else in the trace the checkpoint stays unresolved and is
        NOT counted as a miss (assumption A5)."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "computation_task", 5, 4), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "create_working_folder"}, step=2),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert (s.hits, s.misses) == (0, 0)
        assert s.unresolved == 1
        assert s.unresolved_pending_at_end == 1
        assert s.no_opinion_tool_calls == 1

    def test_workflow_ending_before_target_step_is_unresolved(self, tmp_path):
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "run_ase", 3, 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert (s.predictions, s.hits, s.misses, s.unresolved) == (1, 0, 0, 1)

    def test_hit_plus_miss_plus_unresolved_equals_predictions(self, tmp_path):
        """The recovered accounting must be total: nothing may vanish."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "a", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "a"}, step=2),
            _ev("prediction_result", _pred_payload(2, "b", 3), step=2),
            _ev("checkpoint_created", {"checkpoint_id": "c2", "step": 2}, step=2),
            _ev("tool_call", {"tool": "zzz"}, step=3),
            _ev("prediction_result", _pred_payload(3, "c", 9, 6), step=3),
            _ev("checkpoint_created", {"checkpoint_id": "c3", "step": 3}, step=3),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert s.hits + s.misses + s.unresolved == s.predictions == 3
        assert (s.hits, s.misses, s.unresolved) == (1, 1, 1)

    def test_aged_out_checkpoint_is_labelled_separately(self, tmp_path):
        """Beyond max(max_horizon*8, 60) steps the detector drops a checkpoint
        without scoring it; that is unresolved-aged-out, never a miss."""
        events = [
            _ev("prediction_result", _pred_payload(1, "never_runs", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            # a matching tool immediately, so nothing else is pending
            _ev("tool_call", {"tool": "never_runs"}, step=2),
            _ev("prediction_result", _pred_payload(3, "gone", 4), step=3),
            _ev("checkpoint_created", {"checkpoint_id": "c2", "step": 3}, step=3),
        ]
        # tool call far beyond the 60-step window consumes c2 as a miss, so use
        # a second prediction created after it to leave c2 aged out instead.
        events.append(_ev("tool_call", {"tool": "gone"}, step=200))
        s = score_detector(parse_trace(_write(tmp_path, "t.jsonl", events)).actions, CFG)
        # c2 aged out (200 - 3 > 60), so the step-200 tool has nothing to score.
        assert s.hits == 1
        assert s.unresolved_aged_out == 1
        assert s.misses == 0

    def test_prediction_created_at_the_same_step_is_not_scored_by_that_tool(self, tmp_path):
        """AtomAgents creates the checkpoint and fires the tool at the same step;
        the detector's `c.step < step` guard must stop self-scoring."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "plan_task", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "code_task"}, step=1),
        ])
        s = score_detector(parse_trace(p).actions, CFG)
        assert (s.hits, s.misses, s.unresolved) == (0, 0, 1)


class TestChemgraphInlineScorer:
    """Control scorer: chemgraph.py:957-1003 semantics, deliberately NOT the
    detector's."""

    def test_exact_match_only_no_prefix(self, tmp_path):
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "computation_task", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "computation_task_screw_dislocation"}, step=2),
        ])
        s = score_chemgraph_inline(parse_trace(p).actions, CFG)
        assert (s.hits, s.misses) == (0, 1)

    def test_single_slot_supersedes_older_pending(self, tmp_path):
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "a", 9), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("prediction_result", _pred_payload(2, "b", 3), step=2),
            _ev("checkpoint_created", {"checkpoint_id": "c2", "step": 2}, step=2),
            _ev("tool_call", {"tool": "b"}, step=3),
        ])
        s = score_chemgraph_inline(parse_trace(p).actions, CFG)
        assert (s.predictions, s.hits, s.misses) == (2, 1, 0)
        assert s.unresolved_superseded == 1

    def test_tool_before_expected_step_keeps_pending(self, tmp_path):
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "a", 5), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "x"}, step=2),
            _ev("tool_call", {"tool": "a"}, step=5),
        ])
        s = score_chemgraph_inline(parse_trace(p).actions, CFG)
        assert (s.hits, s.misses) == (1, 0)

    def test_differs_from_detector_on_the_same_stream(self, tmp_path):
        """Pinning the fact that the two implementations are NOT equivalent, so
        nobody later 'fixes' the control by swapping in the detector."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "computation_task", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "computation_task_screw_dislocation"}, step=2),
        ])
        actions = parse_trace(p).actions
        assert score_detector(actions, CFG).hits == 1
        assert score_chemgraph_inline(actions, CFG).hits == 0


class TestReconcile:

    def test_recorded_hit_replayed_miss_is_reported(self, tmp_path):
        """The live broken detector let an already-wrong checkpoint survive to be
        matched later; the fixed one consumes it as a miss first."""
        p = _write(tmp_path, "t.jsonl", [
            _ev("prediction_result", _pred_payload(1, "run_ase", 2), step=1),
            _ev("checkpoint_created", {"checkpoint_id": "c1", "step": 1}, step=1),
            _ev("tool_call", {"tool": "other"}, step=2),
            _ev("tool_call", {"tool": "run_ase"}, step=3),
            _ev("prediction_validated", {"hit": True, "checkpoint_id": "c1"}, step=3),
        ])
        c = reconcile(parse_trace(p), CFG)
        assert c["hit->miss"] == 1


# ---------------------------------------------------------------------------
# 3. THE CONTROL, over the real corpus
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RUNS_DIR.exists(), reason="eval corpus not on disk")
class TestCorpusControl:

    @pytest.fixture(scope="class")
    def replayed(self):
        trials = discover(RUNS_DIR, None)
        parsed = {t.trace: parse_trace(t.trace) for t in trials}
        return trials, parsed

    def test_chemgraph_control_replay_matches_recorded(self, replayed):
        """THE CONTROL.  No frozen constants: the replay's ChemGraph hit/miss
        must equal what the ChemGraph traces themselves recorded, whatever the
        corpus currently contains.  This is the assertion that licenses the
        AtomAgents numbers."""
        trials, parsed = replayed
        ok, info = run_control(trials, parsed, CFG)
        assert info["replay_chemgraph_inline_hits"] == info["recorded_hit_count"], info
        assert (info["replay_chemgraph_inline_misses"]
                == info["recorded_divergence_count"]), info
        assert info["step_reconstruction_failures"] == 0, info
        assert ok

    def test_control_matches_the_2026_08_03_audit_snapshot(self, replayed):
        """Pins the audited corpus.  The eval campaign is live, so a GROWN corpus
        is not a failure — it skips.  A corpus that shrank or was rewritten is."""
        trials, parsed = replayed
        cg = [t for t in trials if t.family == "chemgraph"]
        rec_hits = sum(parsed[t.trace].rec_validated for t in cg)
        rec_div = sum(parsed[t.trace].rec_diverged for t in cg)
        if (rec_hits, rec_div) != (CONTROL_EXPECT["hits"], CONTROL_EXPECT["misses"]):
            if rec_hits >= CONTROL_EXPECT["hits"] and rec_div >= CONTROL_EXPECT["misses"]:
                pytest.skip(
                    f"corpus has grown since the audit snapshot: recorded "
                    f"{rec_hits}/{rec_div} vs audited {CONTROL_EXPECT['hits']}/"
                    f"{CONTROL_EXPECT['misses']}; the control above still applies"
                )
            pytest.fail(
                f"ChemGraph recorded counts moved DOWN or sideways: {rec_hits}/"
                f"{rec_div} vs audited {CONTROL_EXPECT['hits']}/"
                f"{CONTROL_EXPECT['misses']} — the corpus was rewritten"
            )

    def test_step_reconstruction_never_contradicts_a_recorded_step(self, replayed):
        _trials, parsed = replayed
        checks = sum(p.n_step_checks for p in parsed.values())
        fails = sum(p.n_step_check_fail for p in parsed.values())
        assert checks > 0
        assert fails == 0

    def test_atomagents_scored_population_equals_the_recorded_denominator(self, replayed):
        """The replay must score exactly the predictions the summaries counted —
        no more, no less — or its denominator is not the paper's denominator.

        The recorded denominator is recomputed here from the trace stream rather
        than read from a constant, so this holds as the campaign grows.
        """
        trials, parsed = replayed
        aa = [t for t in trials if t.family == "atomagents"]
        scored = sum(parsed[t.trace].n_scored_predictions for t in aa)
        pred_events = sum(parsed[t.trace].n_prediction_events for t in aa)
        # AtomAgents has no staging predictions: every prediction_result became
        # a pending checkpoint.
        assert scored == pred_events

    # -- the e68d52b suppression bug: fixed, and now pinned as history --------
    #
    # This test used to end with
    #
    #     # The bug's signature: not one divergence was ever recorded.
    #     assert sum(parsed[t.trace].rec_diverged for t in aa) == 0
    #
    # which pinned the DEFECT, not correctness: under e68d52b a genuine
    # divergence returned (True, CONTINUE, None), the adapter branched on
    # `ckpt is None` and emitted nothing, so no miss ever reached the trace.
    # 96f5f28 (2026-08-03) fixed that, and every AtomAgents trial run since
    # records divergences normally.  The assertion went red because its premise
    # went stale, not because anything broke.
    #
    # Deleting it would give up the guarantee it was standing in for, so it is
    # inverted below into three characterisation tests:
    #
    #   * divergences ARE recorded now                     (the defect is gone)
    #   * every recorded divergence is reproduced, tuple for tuple, by driving
    #     the shipped fixed detector over the same trace   (they are not
    #                                                       logging artifacts)
    #   * every trial is either AGREEING (live == replay) or SUPPRESSING
    #     (recorded 0 while the replay finds misses = the old bug), and no
    #     trial run after the fix may be SUPPRESSING
    #                                                      (no regression)

    def test_atomagents_now_records_divergences(self, replayed):
        """CHARACTERISATION, replacing the assertion that pinned the bug.

        Audited 2026-09-01 over the corpus on disk: 26 `divergence_detected`
        events across 17 AtomAgents trials, all in `atomagents_exp3_aligned`,
        all from trials whose recorded commit is a descendant of the 96f5f28
        detector fix.  (Was 24/16 earlier the same day; the tandem trial
        t03__20260901-162343__a62b593 added the other two, which is the corpus
        growing as designed rather than a count moving.)  Reproduce with:

            python3 -c "import json,glob; \\
              print(sum(json.load(open(p))['divergence_count'] or 0 for p in \\
              glob.glob('results/eval_q1_q4/runs/atomagents*/*/*/summary.json')))"
        """
        trials, parsed = replayed
        aa = [t for t in trials if t.family == "atomagents"]
        assert sum(parsed[t.trace].rec_diverged for t in aa) > 0, (
            "No AtomAgents trial recorded a divergence.  Either the corpus "
            "lost its post-96f5f28 trials, or the e68d52b suppression bug has "
            "regressed and misses are being dropped before they reach the bus."
        )

    def test_every_recorded_atomagents_divergence_is_reproduced_by_the_replay(
            self, replayed):
        """The load-bearing test: recorded divergences are the fixed detector's
        own verdicts, not artifacts of the trace stream.

        For every trial that recorded a divergence, driving the SHIPPED
        DivergenceDetector over that trial's recorded prediction/tool stream
        must yield the same misses in the same order, agreeing on all four of
        (step, expected_tool, actual_tool, checkpoint_id).

        Note what this does NOT establish.  It does not rule out the
        double-emission artifact, because the replay is INSENSITIVE to a
        duplicate: the second emission finds the checkpoint already consumed by
        the first, so the detector returns no-opinion and the tuples still
        agree.  Verified by mutation — injecting a duplicate `tool` action into
        a divergence-bearing stream leaves this assertion green.  The duplicate
        hypothesis is excluded separately, by
        `test_no_divergence_bearing_atomagents_trace_contains_a_duplicate_emission`.
        """
        trials, parsed = replayed
        checked = 0
        for t in trials:
            if t.family != "atomagents" or parsed[t.trace].rec_diverged == 0:
                continue
            recorded = _recorded_divergences(t.trace)
            replayed_ = _replayed_divergences(parsed[t.trace], CFG)
            assert recorded == replayed_, (
                f"{t.workload}/{t.config}/{t.trial}: the fixed detector does "
                f"not reproduce the recorded divergences.\n"
                f"  recorded: {recorded}\n  replayed: {replayed_}"
            )
            checked += 1
        assert checked > 0

    def test_no_divergence_bearing_atomagents_trace_contains_a_duplicate_emission(
            self, replayed):
        """Excludes the double-emission artifact directly, at the trace level.

        AtomAgents emits some tool calls twice in quick succession (median
        0.39 s apart, n=66 over the corpus).  For `plan_task` those repeats are
        a logging duplicate; for `analyze_screw_core` and
        `computation_task_screw_dislocation` they are genuine re-invocations.
        A divergence manufactured by scoring a duplicate emission against the
        next real call would be spurious, and the recorded expected/actual
        pairs involve exactly those tools — so it has to be checked, and it
        cannot be checked by replay (see the test above).

        It is checked here instead: NO trace that recorded a divergence
        contains a fast same-tool repeat at all.  Audited 2026-09-01 — all 66
        repeats live in `atomagents_exp2` / `atomagents_exp3`, and their
        intersection with the 16 divergence-bearing traces is empty.  Every
        divergence therefore compares two distinct tool calls.
        """
        trials, parsed = replayed
        checked = 0
        for t in trials:
            if t.family != "atomagents" or parsed[t.trace].rec_diverged == 0:
                continue
            calls = [(ev.get("step"), (ev.get("payload") or {}).get("tool"),
                      ev.get("epoch_time"))
                     for ev in _events(t.trace)
                     if ev.get("event_type") == "tool_call"]
            for prev, cur in zip(calls, calls[1:]):
                same_tool = prev[1] == cur[1]
                fast = (prev[2] is not None and cur[2] is not None
                        and (cur[2] - prev[2]) < DUPLICATE_EMISSION_WINDOW_S)
                assert not (same_tool and fast), (
                    f"{t.workload}/{t.config}/{t.trial}: tool {cur[1]!r} "
                    f"emitted twice {cur[2] - prev[2]:.2f}s apart (steps "
                    f"{prev[0]}->{cur[0]}) in a trace that recorded a "
                    f"divergence.  A divergence scored against a duplicate "
                    f"emission may be spurious — audit it before publishing."
                )
            checked += 1
        assert checked > 0

    def test_every_recorded_atomagents_divergence_invalidated_and_went_conservative(
            self, replayed):
        """The guard acted on every divergence it recorded.

        `divergence_detected` carries action=INVALIDATE_ALL and is immediately
        followed by a `conservative_mode` event.  A divergence that scored but
        did not act would mean the detector and the adapter disagree.
        """
        trials, parsed = replayed
        seen = 0
        for t in trials:
            if t.family != "atomagents" or parsed[t.trace].rec_diverged == 0:
                continue
            events = _events(t.trace)
            for i, ev in enumerate(events):
                if ev.get("event_type") != "divergence_detected":
                    continue
                seen += 1
                payload = ev.get("payload") or {}
                assert payload.get("action") == "INVALIDATE_ALL", (
                    f"{t.trial} step {ev.get('step')}: divergence recorded "
                    f"with action={payload.get('action')!r}")
                nxt = events[i + 1] if i + 1 < len(events) else {}
                assert nxt.get("event_type") == "conservative_mode", (
                    f"{t.trial} step {ev.get('step')}: divergence not followed "
                    f"by conservative_mode (got "
                    f"{nxt.get('event_type')!r}) — the guard scored the miss "
                    f"but did not act on it")
                assert (nxt.get("payload") or {}).get("reason") == "divergence"
        assert seen > 0

    def test_no_trial_run_after_the_detector_fix_suppresses_a_miss(self, replayed):
        """Regression guard on e68d52b, stated without reference to a count.

        Every AtomAgents trial falls into exactly one of two classes:

          AGREEING     recorded divergences == the replay's misses.  The live
                       detector scored what the fixed detector scores.
          SUPPRESSING  recorded 0 divergences while the replay finds misses.
                       That is the e68d52b signature: the misses happened and
                       were dropped before reaching the bus.

        A trial in neither class — recording a nonzero number of divergences
        that disagrees with the replay — is a new defect and fails outright.

        SUPPRESSING is permitted only for trials that ran BEFORE 96f5f28.  A
        post-fix trial that suppresses means the bug came back.  Era comes from
        the commit hash in the trial directory name; if git cannot resolve one,
        that trial is skipped rather than guessed at.
        """
        trials, parsed = replayed
        unresolvable = 0
        classes = Counter()
        for t in trials:
            if t.family != "atomagents":
                continue
            rec = parsed[t.trace].rec_diverged
            rep = score_detector(parsed[t.trace].actions, CFG).misses
            if rec == rep:
                klass = "agreeing"
            elif rec == 0 and rep > 0:
                klass = "suppressing"
            else:
                pytest.fail(
                    f"{t.workload}/{t.config}/{t.trial}: recorded {rec} "
                    f"divergences but the fixed detector scores {rep} misses. "
                    f"This is neither a clean trial nor the known suppression "
                    f"bug — investigate before publishing any AtomAgents number."
                )
            classes[klass] += 1
            if klass != "suppressing":
                continue
            post_fix = _ran_with_fixed_detector(t.trial)
            if post_fix is None:
                unresolvable += 1
                continue
            assert not post_fix, (
                f"{t.workload}/{t.config}/{t.trial} ran at a commit descended "
                f"from the {DETECTOR_FIX_COMMIT} detector fix, yet recorded 0 "
                f"divergences while the fixed detector scores {rep} misses. "
                f"The e68d52b suppression bug has regressed."
            )
        assert classes["agreeing"] > 0 and classes["suppressing"] > 0, classes
        if unresolvable:
            pytest.skip(
                f"{unresolvable} suppressing trial(s) name a commit git cannot "
                f"resolve; the rest of the classification held")

    def test_atomagents_divergences_match_the_2026_09_01_audit_snapshot(self, replayed):
        """Pins the audited divergence population.  Same convention as the
        ChemGraph snapshot above: the campaign is live, so a GROWN corpus skips;
        one that shrank or was rewritten fails."""
        trials, parsed = replayed
        aa = [t for t in trials if t.family == "atomagents"]
        n_div = sum(parsed[t.trace].rec_diverged for t in aa)
        n_trials = sum(1 for t in aa if parsed[t.trace].rec_diverged)
        want_div, want_trials = ATOMAGENTS_DIVERGENCE_SNAPSHOT
        if (n_div, n_trials) == (want_div, want_trials):
            return
        if n_div >= want_div and n_trials >= want_trials:
            pytest.skip(
                f"corpus has grown since the 2026-09-01 audit: {n_div} "
                f"divergences over {n_trials} trials vs audited {want_div}/"
                f"{want_trials}; the per-trial checks above still apply")
        pytest.fail(
            f"AtomAgents recorded divergences moved DOWN or sideways: {n_div} "
            f"over {n_trials} trials vs audited {want_div}/{want_trials} — the "
            f"corpus was rewritten, or the detector is suppressing again")

    def test_atomagents_accounting_is_total(self, replayed):
        """Nothing may vanish: every scored prediction is a hit, a miss, or
        explicitly unresolved."""
        trials, parsed = replayed
        aa = [t for t in trials if t.family == "atomagents"]
        hits = misses = unres = preds = 0
        for t in aa:
            s = score_detector(parsed[t.trace].actions, CFG)
            hits += s.hits
            misses += s.misses
            unres += s.unresolved
            preds += s.predictions
        assert preds == sum(parsed[t.trace].n_scored_predictions for t in aa)
        assert hits + misses + unres == preds
        assert preds > 0

    def test_recovered_hit_count_may_differ_from_the_recorded_one(self, replayed):
        """Documents a correction to the brief.

        The brief brackets the true AtomAgents accuracy as [91/219, 91/91],
        which holds the numerator fixed at the 91 recorded hits and treats only
        the 128 unvalidated predictions as unknown.  That is not what the fixed
        detector does: it CONSUMES a checkpoint on a miss, so a checkpoint the
        broken detector kept alive until a later matching tool can instead be
        retired as a miss first, and the later tool then finds nothing pending.
        The recovered numerator is therefore not pinned to the recorded one and
        the lower bound 91/219 is not guaranteed.
        """
        trials, parsed = replayed
        aa = [t for t in trials if t.family == "atomagents"]
        rec_hits = sum(parsed[t.trace].rec_validated for t in aa)
        recon_total = Counter()
        for t in aa:
            recon_total.update(reconcile(parsed[t.trace], CFG))
        assert recon_total["hit->hit"] + recon_total["hit->miss"] \
            + recon_total["hit->unresolved"] == rec_hits
        # The reassignment is real in this corpus, not a hypothetical.
        assert recon_total["hit->miss"] > 0
