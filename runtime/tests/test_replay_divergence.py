"""
test_replay_divergence.py — tests for scripts/replay_divergence.py.

Three layers:

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
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from runtime.config import RuntimeConfig                       # noqa: E402
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
        # The bug's signature: not one divergence was ever recorded.
        assert sum(parsed[t.trace].rec_diverged for t in aa) == 0

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
