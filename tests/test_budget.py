import pytest

from elt_tinker.budget import Budget


def test_ambiguous_call_reserve_survives_restart_and_blocks_spend(tmp_path):
    path = tmp_path / "budget.json"
    gate = Budget(path, limit=1)
    ticket = gate.reserve("sample", 0.7)
    with pytest.raises(RuntimeError):
        Budget(path, limit=1).reserve("sample", 0.4)
    gate.settle(ticket, 0.2)
    Budget(path, limit=1).reserve("train", 0.7)
    with pytest.raises(RuntimeError):
        gate.reserve("sample", 0.2)


def test_budget_rejects_nan_and_negative(tmp_path):
    gate = Budget(tmp_path / "budget.json")
    for value in [float("nan"), float("inf"), 0, -1]:
        with pytest.raises(ValueError):
            gate.reserve("sample", value)
