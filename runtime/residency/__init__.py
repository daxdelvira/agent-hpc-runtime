"""runtime.residency — one budget over every held resource class.

    contract.py    the frozen interface (rungs, ResourceSpec, Eq. 1, the
                   ResidencyActor / Ledger / Arbitrator protocols, I1-I5)
    ledger.py      T1: the ledger. Measured charges, confirmed releases.
    arbitrator.py  T2: greedy single-victim retention arbitration, plus the
                   ranking primitives the simulator shares.

The horizon estimator (T3) and the residency actors (T4a model, T4b data
worker) are separate work; everything here depends only on the protocols.
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
