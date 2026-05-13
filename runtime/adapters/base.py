"""
adapters/base.py — Abstract WorkflowAdapter interface.

Adapters connect to existing workflow frameworks (ChemGraph / AtomAgents)
without modifying their source code. They observe events (LLM calls, tool
calls, graph node transitions) and forward them to the runtime pipeline.

Implementations:
  chemgraph.py   — extends ChemGraphCallbackHandler (LangChain BaseCallbackHandler)
  atomagents.py  — extends autogen_hook monkey-patches
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from runtime.config import RuntimeConfig
from runtime.events import PredictionResult
from runtime.predictor.base import Predictor


class WorkflowAdapter(ABC):
    """
    Minimal interface all adapters must satisfy.

    An adapter is responsible for:
    1. Detecting step boundaries (LLM call end, tool call start)
    2. Extracting current agent intent (tool_calls from LLM response)
    3. Calling self._predictor.predict() and forwarding to self._guard
    4. Forwarding tool-start events to self._guard for divergence detection
    5. Emitting HpcEvents to self._bus

    When config.mode == BASELINE the adapter must be a strict no-op.
    """

    def __init__(
        self,
        predictor: Predictor | None,
        config: RuntimeConfig,
    ) -> None:
        self._predictor = predictor
        self._config = config
        self._step = 0      # monotonic step counter; increment on each LLM call

    @property
    def is_active(self) -> bool:
        """False when mode=BASELINE; adapters skip all runtime logic."""
        from runtime.config import RuntimeMode
        return self._config.mode != RuntimeMode.BASELINE

    @abstractmethod
    def on_llm_response(
        self,
        tool_calls: list[dict],
        model_name: str,
        recent_events: list[dict],
    ) -> PredictionResult | None:
        """
        Called after an LLM response is received.
        Returns the prediction (or None if mode=BASELINE or predictor=None).
        """

    @abstractmethod
    def on_tool_about_to_execute(self, tool_name: str) -> None:
        """
        Called just before a tool executes.
        Forwards to the divergence guard.
        """
