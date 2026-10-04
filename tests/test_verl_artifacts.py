"""Validate released result summaries and the recovery launcher without a GPU."""
from pathlib import Path
import runpy

import pytest

EXPERIMENT = Path(__file__).parents[1] / "experiments/verl_math"


def test_compact_evidence():
    result = runpy.run_path(str(EXPERIMENT / "verify_results.py"))["verify"]()
    assert result["trajectories"] == 41600
    assert result["recovery_final_accuracy"]["recovery_ckpt80"] == .7625


@pytest.mark.parametrize("phase,verifier,total,resume", [
    ("A", "correct", 60, None),
    ("B", "len_decay", 130, "phaseA/ckpts/global_step_60"),
    ("REC", "correct", 100, "phaseA/ckpts/global_step_60"),
    ("RECLAST", "correct", 170, "phaseB/ckpts/global_step_130"),
])
def test_phase_recipe(tmp_path, phase, verifier, total, resume):
    build = runpy.run_path(str(EXPERIMENT / "launch.py"))["build_command"]
    command, actual_verifier, output, checkpoint = build(
        phase, "/verl", "/math", str(tmp_path), "/pinned-model")
    assert actual_verifier == verifier
    assert f"trainer.total_training_steps={total}" in command
    assert "data.train_batch_size=64" in command
    assert "actor_rollout_ref.rollout.n=5" in command
    assert checkpoint == (tmp_path / resume if resume else None)
    assert not output.exists()  # command construction must never launch or create a run
