"""runtime.residency — one budget over every held resource class.

    contract.py    the frozen interface (rungs, ResourceSpec, Eq. 1, the
                   ResidencyActor / Ledger / Arbitrator protocols, I1-I5)
    ledger.py      T1: the ledger. Measured charges, confirmed releases.
    arbitrator.py  T2: greedy CHAINED retention arbitration (up to
                   DEFAULT_MAX_VICTIMS victims per admit), plus the ranking
                   primitives the simulator shares.
    horizon.py     T3: the horizon estimator.
    model_actor.py T4a: models held at R2 (vLLM L1 park/wake, GPU eviction).
    data_worker.py T4b: data held at R3 (resident, evictable worker).

Everything outside contract.py depends only on its protocols.

This module exports T1, T2 and the contract. T3 and the actors are imported
from their own modules -- they pull in vLLM and LAMMPS clients, and keeping
them out of the package __init__ is what lets the policy be replayed and
unit-tested with no GPU and no engine (see scripts/replay_tandem_trace.py).
"""

from runtime.residency.contract import (
    Arbitrator,
    EvictionPlan,
    HorizonEstimator,
    Ledger,
    LedgerEntry,
    ReleaseNotHonoured,
    ResidencyActor,
    ResourceClass,
    ResourceSpec,
    Rung,
    check_horizon,
    value,
    value_density,
)
from runtime.residency.arbitrator import (
    DEFAULT_DECAY_S,
    DEFAULT_MAX_VICTIMS,
    GreedyArbitrator,
    evict_until_fits,
    greedy_pack,
)
from runtime.residency.ledger import (
    DEFAULT_DRIFT_TOLERANCE_GB,
    DEFAULT_RELEASE_TOLERANCE_GB,
    BudgetExceeded,
    ReleaseShortfall,
    ResidencyLedger,
)

__all__ = [
    "Arbitrator", "EvictionPlan", "HorizonEstimator", "Ledger", "LedgerEntry",
    "ReleaseNotHonoured", "ResidencyActor", "ResourceClass", "ResourceSpec",
    "Rung", "check_horizon", "value", "value_density",
    "GreedyArbitrator", "evict_until_fits", "greedy_pack",
    "DEFAULT_DECAY_S", "DEFAULT_MAX_VICTIMS",
    "BudgetExceeded", "ReleaseShortfall", "ResidencyLedger",
    "DEFAULT_RELEASE_TOLERANCE_GB", "DEFAULT_DRIFT_TOLERANCE_GB",
]
