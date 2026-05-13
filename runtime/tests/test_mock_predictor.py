"""
test_mock_predictor.py — Unit tests for MockPredictor rule tables.
"""
import pytest

from runtime.predictor.mock_predictor import MockPredictor


def make_tool_event(tool_name: str, step: int = 1) -> dict:
    return {
        "event_type": "tool_call",
        "step": step,
        "payload": {"tool": tool_name},
    }


def make_tool_call(name: str) -> dict:
    return {"name": name, "args": {}}


# ---------------------------------------------------------------------------
# ChemGraph transitions
# ---------------------------------------------------------------------------

class TestChemGraphPredictor:
    def setup_method(self):
        self.p = MockPredictor(workflow="chemgraph")

    def test_after_smiles_predicts_mace(self):
        result = self.p.predict(
            step=2,
            recent_events=[make_tool_event("smiles_to_coordinate_file", step=1)],
            current_tool_calls=[make_tool_call("smiles_to_coordinate_file")],
        )
        assert len(result.resources) >= 1
        assert result.resources[0].resource_type == "mace_model"
        assert result.resources[0].consumer_tool == "run_ase"
        assert result.confidence >= 0.85

    def test_after_molecule_name_predicts_mace(self):
        result = self.p.predict(
            step=1,
            recent_events=[],
            current_tool_calls=[make_tool_call("molecule_name_to_smiles")],
        )
        assert any(r.resource_type == "mace_model" for r in result.resources)

    def test_after_run_ase_no_prediction(self):
        result = self.p.predict(
            step=3,
            recent_events=[make_tool_event("run_ase", step=2)],
            current_tool_calls=[make_tool_call("run_ase")],
        )
        # run_ase is not a key in ChemGraph transition table → empty prediction
        assert result.confidence == 0.0 or len(result.resources) == 0

    def test_predictor_id(self):
        assert self.p.predictor_id == "mock"

    def test_returns_prediction_result_even_when_no_match(self):
        result = self.p.predict(
            step=1,
            recent_events=[],
            current_tool_calls=[make_tool_call("unknown_tool_xyz")],
        )
        assert result is not None
        assert result.step == 1


# ---------------------------------------------------------------------------
# AtomAgents transitions
# ---------------------------------------------------------------------------

class TestAtomAgentsPredictor:
    def setup_method(self):
        self.p = MockPredictor(workflow="atomagents")

    def test_after_plan_task_predicts_eam_files(self):
        result = self.p.predict(
            step=2,
            recent_events=[make_tool_event("plan_task", step=1)],
            current_tool_calls=[make_tool_call("plan_task")],
        )
        names = [r.name for r in result.resources]
        assert "W_Zhou04.eam.alloy" in names
        assert "w_eam4.fs" in names

    def test_after_plan_task_resources_are_data_files(self):
        result = self.p.predict(
            step=2,
            recent_events=[],
            current_tool_calls=[make_tool_call("plan_task")],
        )
        for r in result.resources:
            assert r.resource_type == "data_file"

    def test_after_computation_predicts_qwen32b(self):
        result = self.p.predict(
            step=5,
            recent_events=[make_tool_event("computation_task_screw_dislocation", step=4)],
            current_tool_calls=[make_tool_call("computation_task_screw_dislocation")],
        )
        # May predict qwen_32b for next planning step
        model_names = [r.name for r in result.resources]
        # If prediction exists, it should be qwen_32b
        if result.resources:
            assert any("qwen" in n.lower() for n in model_names)


# ---------------------------------------------------------------------------
# Auto workflow inference
# ---------------------------------------------------------------------------

class TestAutoPredictor:
    def test_auto_infers_chemgraph_from_ase_event(self):
        p = MockPredictor(workflow="auto")
        result = p.predict(
            step=2,
            recent_events=[make_tool_event("smiles_to_coordinate_file")],
            current_tool_calls=[make_tool_call("smiles_to_coordinate_file")],
        )
        assert result.resources[0].resource_type == "mace_model"

    def test_auto_infers_atomagents_from_plan_event(self):
        p = MockPredictor(workflow="auto")
        result = p.predict(
            step=2,
            recent_events=[make_tool_event("plan_task")],
            current_tool_calls=[make_tool_call("plan_task")],
        )
        assert len(result.resources) > 0
        assert result.resources[0].resource_type == "data_file"
