"""A deterministic capture/diagnosis demonstration, not a training experiment."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import rtd


def correct_verifier(trajectory):
    return float(trajectory.response == str(trajectory.ground_truth))


def demonstrate(root):
    root = Path(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("--store must name a new or empty directory")
    with rtd.run(str(root), run_id="synthetic-demo", framework="synthetic") as run:
        state = run.capture_checkpoint(step=0, path="illustrative/step0")
        for step in range(1, 41):
            trajectories, signals = [], []
            for response in ("42", "42 + 1"):
                trajectory = run.capture_rollout(
                    step=step, prompt="20 + 22 = ?", response=response,
                    ground_truth=42, policy_ckpt=state,
                    engine_version="synthetic-no-model",
                )
                reward = (float("42" in response) if step >= 18
                          else correct_verifier(trajectory))
                signal = run.capture_signal(
                    trajectory, value=reward, source_id="verifier",
                    source_version="substring-v2" if step >= 18 else "exact-v1",
                )
                trajectories.append(trajectory)
                signals.append(signal)
            update = run.capture_update(
                step=step, signals=signals, trajs=trajectories, parent_ckpt=state,
            )
            state = run.capture_checkpoint(
                step=step, update=update,
                path=f"illustrative/step{step}" if step % 5 == 0 else "",
            )
    store = rtd.open_store(str(root))
    rescored = rtd.rescore(store, correct_verifier, "exact-v1")
    mismatches = rtd.diff(store, rescored)
    graph = rtd.LineageGraph.build(store)
    closure = rtd.localize(graph, mismatches)
    plan = rtd.plan_recovery(graph, closure, "exact-v1")
    return {
        "trajectories": len(rescored.values),
        "measurement_mismatches": len(mismatches.mismatch_traj_ids),
        "first_affected_update": plan.onset_step,
        "rollback_checkpoint": plan.rollback_ckpt_step,
        "affected_updates": plan.contaminated_updates,
        "new_diagnostic_rollouts": 0,
        "recovery_plan": plan.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", help="Retain evidence in a new or empty directory")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args()
    try:
        if args.store:
            result = demonstrate(args.store)
        else:
            with TemporaryDirectory(prefix="rtd-demo-") as root:
                result = demonstrate(root)
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for label, key in (
            ("Trajectories:          ", "trajectories"),
            ("Measurement mismatches:", "measurement_mismatches"),
            ("First affected update:", "first_affected_update"),
            ("Rollback checkpoint:  ", "rollback_checkpoint"),
            ("Affected updates:     ", "affected_updates"),
            ("New diagnostic rollouts:", "new_diagnostic_rollouts"),
        ):
            print(label, result[key])


if __name__ == "__main__":
    main()
