"""Regression tests for AtomAgents potential selection.

WHY THIS FILE EXISTS. The exp3 workload compares two EAM potentials. Its
characteristic failure is not a crash but a SILENT DUPLICATE: the second
simulation resolves to the potential the first one already used, writes into
the second working directory, and returns a perfectly well-formed result that
is actually a comparison of a potential against itself. It has happened twice,
through two different code paths, and both times it was found by hand after the
compute was already spent:

  1. 2026-07  first-substring-wins over a tuple of valid names. The agent's
     prose named both potentials, W_Zhou04 was scanned first, and it won every
     time. Invalidated 16 of 27 collected trials.
  2. 2026-08-03  argument-source ORDERING in the recovery path. `potential`
     (free-text prose naming both) was consulted before `working_directory`
     (unambiguous per-call intent). The prose said 'w_eam4.fs', which does not
     match the offered 'w_eam4_big.fs' (keys {eam4} vs {eam4,big}), while
     'W_Zhou04.eam.alloy' matched exactly -> Zhou04 won unambiguously and the
     loop broke before working_directory was ever read.

Both are cheap to test and expensive to miss, so they are pinned here.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

_ATOMAGENTS = Path(__file__).resolve().parents[2] / "workloads" / "AtomAgents"

# autogen opens a diskcache SQLite DB at ./.cache/<seed> during import; see the
# note in experiments/atomagents_exp3.py. Guarantee the target exists so the
# import cannot fail for a reason unrelated to what we are testing.
os.makedirs("/tmp/autogen_cache", exist_ok=True)


def _tools(potentials: str):
    """Import orchestration_tools with a given offered set.

    The offered set is read at IMPORT time (one source of truth for the tool
    description, the argument annotation and the validator), so the module must
    be reloaded whenever it changes.
    """
    if str(_ATOMAGENTS) not in sys.path:
        sys.path.insert(0, str(_ATOMAGENTS))
    os.environ["ATOMAGENTS_POTENTIALS"] = potentials
    import atomagents.tools.orchestration_tools as ot
    return importlib.reload(ot)


pytest.importorskip("autogen", reason="AtomAgents workload env not present")

BIG = "W_Zhou04.eam.alloy,w_eam4_big.fs"
SMALL = "W_Zhou04.eam.alloy,w_eam4.fs"

# The two calls exactly as they appeared in the 2026-08-03 trial, after the
# argument hook drops the positional ints and the free-text message lands on
# `potential`.
PROSE_1 = ("Compute the equilibrium core structure of the 1/2<111> screw "
           "dislocation in W using the 'W_Zhou04.eam.alloy' potential. After the "
           "computations return the classification of the screw dislocation core "
           "and any adjustments needed before running w_eam4.fs.")
PROSE_2 = ("Compute the equilibrium core structure of the 1/2<111> screw "
           "dislocation in W using the 'w_eam4.fs' potential. After the "
           "computations return the classification of the screw dislocation core "
           "and compare the energetics with the W_Zhou04.eam.alloy result.")


def _resolve(ot, potential: str, working_directory: str):
    """Mirror the recovery loop in computation_task_screw_dislocation."""
    valid = ot.offered_potentials()
    if potential in valid:
        return potential
    for cand in (working_directory, potential):
        m = ot._match_potentials(cand, valid)
        if len(m) == 1:
            return next(iter(m))
    return None


def test_prose_naming_both_potentials_does_not_collapse_to_the_first():
    """The 2026-08-03 regression, verbatim: two calls must resolve differently."""
    ot = _tools(BIG)
    first = _resolve(ot, PROSE_1, "W_screw_Zhou04")
    second = _resolve(ot, PROSE_2, "W_screw_eam4")
    assert first == "W_Zhou04.eam.alloy"
    assert second == "w_eam4_big.fs", (
        "second call collapsed to the first potential — this is the silent "
        "duplicate that invalidates the whole comparison"
    )
    assert first != second


def test_working_directory_outranks_ambiguous_prose():
    """working_directory is per-call intent and must be consulted first."""
    ot = _tools(BIG)
    # Prose mentions only Zhou04; the working directory says otherwise. The
    # directory is the field that says which simulation THIS call is.
    assert _resolve(ot, "see the W_Zhou04.eam.alloy result", "W_screw_eam4") == \
        "w_eam4_big.fs"


def test_no_silent_downgrade_to_the_small_potential():
    """'eam4' must not resolve to the 9.3 MB file when only the 3.32 GB is offered.

    A downgrade erases the ~129 s activation cost the experiment exists to
    measure, and does so invisibly: the run still succeeds.
    """
    ot = _tools(BIG)
    assert _resolve(ot, "w_eam4.fs", "W_screw_eam4") == "w_eam4_big.fs"


def test_no_silent_upgrade_when_both_siblings_are_offered():
    """Symmetrically: bare 'eam4' must NOT reach the big file if the small one
    is also on offer. Inventing a 129 s cost the caller never asked for is just
    as wrong as erasing one."""
    ot = _tools("W_Zhou04.eam.alloy,w_eam4.fs,w_eam4_big.fs")
    assert ot._match_potentials("w_eam4.fs", ot.offered_potentials()) == {"w_eam4.fs"}
    assert ot._match_potentials("the eam4 big potential",
                                ot.offered_potentials()) == {"w_eam4_big.fs"}


def test_valid_potential_is_never_rewritten():
    """Recovery must not fire when the agent got it right."""
    ot = _tools(BIG)
    for p in ot.offered_potentials():
        assert _resolve(ot, p, "W_screw_anything") == p


def test_default_offered_set_is_unchanged_without_env():
    """The 28 already-collected trials must stay reproducible."""
    os.environ.pop("ATOMAGENTS_POTENTIALS", None)
    if str(_ATOMAGENTS) not in sys.path:
        sys.path.insert(0, str(_ATOMAGENTS))
    import atomagents.tools.orchestration_tools as ot
    ot = importlib.reload(ot)
    assert ot.offered_potentials() == ("W_Zhou04.eam.alloy", "w_eam4.fs")


def test_prompt_retarget_is_a_noop_on_the_default_pair():
    """`pinned` must stay byte-identical to what the collected trials ran."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.atomagents_exp3 import (
        DEFAULT_TASK_PROMPT_EXP3, retarget_prompt_potentials,
    )
    assert retarget_prompt_potentials(
        DEFAULT_TASK_PROMPT_EXP3, ("W_Zhou04.eam.alloy", "w_eam4.fs")
    ) == DEFAULT_TASK_PROMPT_EXP3


def test_prompt_retarget_names_the_offered_file():
    """The agent must never be shown a potential that is not on offer — that is
    what made the recovery heuristic load-bearing on every single call."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.atomagents_exp3 import (
        DEFAULT_TASK_PROMPT_EXP3, retarget_prompt_potentials,
    )
    out = retarget_prompt_potentials(
        DEFAULT_TASK_PROMPT_EXP3, ("W_Zhou04.eam.alloy", "w_eam4_big.fs")
    )
    assert "w_eam4_big.fs" in out
    # No bare w_eam4.fs left once the big-file mentions are discounted.
    assert "w_eam4.fs" not in out.replace("w_eam4_big.fs", "")
    assert "W_Zhou04.eam.alloy" in out
