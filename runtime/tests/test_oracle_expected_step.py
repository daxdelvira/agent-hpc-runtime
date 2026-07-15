"""OraclePredictor must pin expected_at_step to the resolved consumer step.

Regression test for the off-by-one that made oracle trials score 0 hits on
structurally identical traces (chemgraph_swap oracle t01/t02, 2026-07):
_lookup_resource_for_consumer reused mock-table templates verbatim, so
_MACE_MP0_2's trigger-relative consumer_step_offset=2 stamped the prediction
one step late, the adapter's `step >= expected_at_step` gate skipped the real
consumer, and the following tool registered a spurious divergence.

Mirrors the adapter's on_tool_start walk: 1-indexed step counter; validate the
single pending checkpoint (hit iff tool == resources[0].consumer_tool once
step >= expected_at_step); then predict(step) and overwrite the pending slot
when resources are returned.
"""
import json

from runtime.predictor.oracle_predictor import OraclePredictor

TOOLS = [
    "molecule_name_to_smiles",
    "smiles_to_coordinate_file",
    "run_ase",
    "extract_output_json",
]


def _write_trace(tmp_path):
    trace = tmp_path / "ref_trace.jsonl"
    with open(trace, "w") as f:
        for tool in TOOLS:
            f.write(json.dumps({
                "event_type": "tool_call",
                "payload": {"tool": tool, "arguments": "{}"},
            }) + "\n")
    return trace


def test_oracle_prediction_validates_at_consumer_step(tmp_path):
    pred = OraclePredictor(str(_write_trace(tmp_path)), workflow="chemgraph")

    pending = None  # (consumer_tool, expected_at_step)
    hits, misses = [], []

    for step, tool in enumerate(TOOLS, start=1):
        if pending is not None:
            consumer, expected = pending
            if step >= expected:
                (hits if tool == consumer else misses).append((step, tool, consumer))
                pending = None
        res = pred.predict(step=step, recent_events=[],
                           current_tool_calls=[{"name": tool}])
        if res.resources:
            r = res.resources[0]
            assert r.expected_at_step == step + 1, (
                f"oracle prediction at step {step} stamped "
                f"expected_at_step={r.expected_at_step}, want {step + 1}"
            )
            pending = (r.consumer_tool, r.expected_at_step)

    assert misses == [], f"spurious divergences: {misses}"
    assert (3, "run_ase", "run_ase") in hits, (
        f"mace/run_ase prediction must validate at step 3, got hits={hits}"
    )
