"""
predictor/base.py — Abstract Predictor interface.

Implementations:
  mock_predictor.py   — rule-based stub; zero external deps
  llm_predictor.py    — small LLM via Groq / ALCF endpoint
  oracle_predictor.py — replays known JSONL trace for upper-bound estimate
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from runtime.events import PredictionResult


class Predictor(ABC):
    """
    Given recent workflow events and the current agent intent, predict
    which resources will be needed in the next `horizon` steps.

    All implementations must return a PredictionResult even if they have
    no useful prediction (return empty resources list with confidence=0.0).
    Incorrect predictions are expected; callers must never assume correctness.
    """

    @abstractmethod
    def predict(
        self,
        step: int,
        recent_events: list[dict[str, Any]],   # last N JSONL records as dicts
        current_tool_calls: list[dict],         # tool_calls from current LLM response
        task_description: str = "",
        plan_context: Any = None,              # PlanContext if a plan was extracted
    ) -> PredictionResult:
        """Return resource predictions for the next horizon steps."""

    @property
    @abstractmethod
    def predictor_id(self) -> str:
        """Short string identifying this predictor in events (e.g. 'mock', 'llm:groq')."""
