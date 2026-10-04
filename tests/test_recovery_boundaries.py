"""Recovery must select saved states strictly before a contaminated update."""
import runpy
from pathlib import Path

import pytest
import rtd


def plan_for(store):
    reference = rtd.rescore(store, lambda t: 0.0, "reject-all", use_cache=False)
    graph = rtd.LineageGraph.build(store)
    closure = rtd.localize(graph, rtd.diff(store, reference))
    return rtd.plan_recovery(graph, closure)


def test_logical_states_and_unproven_same_step_checkpoint_are_not_recoverable(tmp_path):
    with rtd.run(str(tmp_path)) as run:
        saved = run.capture_checkpoint(step=0, path="initial")
        state = run.capture_checkpoint(step=1)  # unsaved logical state
        trajectory = run.capture_rollout(step=2, prompt="x", response="y")
        signal = run.capture_signal(trajectory, value=1.0)
        run.capture_update(step=2, signals=[signal], parent_ckpt=state)
        run.capture_checkpoint(step=2, path="unlinked-at-onset")
    plan = plan_for(rtd.open_store(str(tmp_path)))
    assert plan.rollback_ckpt_id == saved.ckpt_id
    assert plan.rollback_ckpt_step == 0


def test_no_saved_state_requires_restart(tmp_path):
    with rtd.run(str(tmp_path)) as run:
        run.capture_checkpoint(step=0)
        trajectory = run.capture_rollout(step=1, prompt="x", response="y")
        signal = run.capture_signal(trajectory, value=1.0)
        run.capture_update(step=1, signals=[signal])
    plan = plan_for(rtd.open_store(str(tmp_path)))
    assert plan.onset_step == 1
    assert plan.rollback_ckpt_id is None
    assert "restart" in plan.notes


def test_unconsumed_mismatch_does_not_prescribe_rollback(tmp_path):
    with rtd.run(str(tmp_path)) as run:
        run.capture_checkpoint(step=0, path="initial")
        trajectory = run.capture_rollout(step=1, prompt="x", response="y")
        run.capture_signal(trajectory, value=1.0)
    plan = plan_for(rtd.open_store(str(tmp_path)))
    assert plan.mismatch_count == 1
    assert plan.onset_step is None
    assert plan.rollback_ckpt_id is None
    assert plan.rollback_ckpt_step is None
    assert "nothing to roll back" in plan.notes


def test_documented_demo_and_existing_store_protection(tmp_path):
    demo = runpy.run_path(str(Path(__file__).parents[1] / "examples/verifier_regression.py"))
    root = tmp_path / "demo"
    result = demo["demonstrate"](root)
    assert result["trajectories"] == 80
    assert result["measurement_mismatches"] == 23
    assert result["first_affected_update"] == 18
    assert result["rollback_checkpoint"] == 15
    assert result["affected_updates"] == 23
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        demo["demonstrate"](root)
    assert (root / "run.json").read_bytes() == before
