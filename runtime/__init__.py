"""
runtime — prediction-driven speculative prefetch layer for agentic HPC workflows.

Sits between ChemGraph / AtomAgents and the cluster, observing workflow events,
predicting upcoming model/data needs, and speculatively acquiring them so that
expensive I/O overlaps with ongoing compute rather than blocking it.

Usage
-----
from runtime.config import RuntimeConfig, RuntimeMode
from runtime.adapters.chemgraph import make_runtime_callback
from runtime.predictor.mock_predictor import MockPredictor
from runtime.prefetch.simulated import SimulatedPrefetchExecutor
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.guard.detector import DivergenceDetector

cfg = RuntimeConfig(mode=RuntimeMode.SIMULATED, run_id="my-run-001")
predictor = MockPredictor(workflow="chemgraph")
executor  = SimulatedPrefetchExecutor()
scheduler = PrefetchScheduler(executor=executor, config=cfg)
guard     = DivergenceDetector(scheduler=scheduler, config=cfg)

cb = make_runtime_callback(predictor=predictor, scheduler=scheduler, guard=guard, config=cfg)
# Pass cb into ChemGraph's config["callbacks"] list.
"""
from runtime.config import RuntimeConfig, RuntimeMode

__all__ = ["RuntimeConfig", "RuntimeMode"]
