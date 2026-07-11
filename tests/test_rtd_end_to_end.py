"""End-to-end RTD test on a synthetic RLVR run with an injected
mid-run verifier regression (the SOSP26 V5-style incident).

Timeline: 40 update steps, checkpoint every 5 steps (plus initial ckpt at
step 0). From step FAULT_STEP on, the recorded verifier rewards a proxy
(any response containing "42") instead of correctness, so some wrong
responses get reward 1.0 (false positives).

Ground truth the diagnosis must recover, with zero rerollout:
  * onset update step == FAULT_STEP
  * last unaffected checkpoint == the latest ckpt with step <= FAULT_STEP
    that is not produced by a contaminated update
"""
import random

import pytest

import rtd

FAULT_STEP = 18
STEPS = 40
CKPT_EVERY = 5
N_PER_STEP = 8


def correct_verifier(traj: rtd.Trajectory) -> float:
    return 1.0 if traj.response == str(traj.ground_truth) else 0.0


def buggy_verifier(response: str, ground_truth) -> float:
    # proxy: rewards the substring "42" regardless of correctness
    return 1.0 if "42" in response else 0.0


@pytest.fixture()
def store_root(tmp_path):
    rng = random.Random(0)
    with rtd.run(str(tmp_path), run_id="run-test", framework="synthetic",
                 algorithm="grpo", model="toy") as r:
        ckpt = r.capture_checkpoint(step=0, path="ckpt/step0")
        for step in range(1, STEPS + 1):
            sigs, trajs = [], []
            for i in range(N_PER_STEP):
                a, b = rng.randint(1, 60), rng.randint(1, 60)
                truth = a + b
                # policy drifts toward the hack after the fault: emits "42"
                if step >= FAULT_STEP and rng.random() < 0.5:
                    resp = "42"
                elif rng.random() < 0.6:
                    resp = str(truth)
                else:
                    resp = str(truth + 1)
                t = r.capture_rollout(step=step, prompt=f"{a}+{b}=?",
                                      response=resp, ground_truth=truth,
                                      policy_ckpt=ckpt,
                                      sampling_config={"temperature": 1.0},
                                      engine_version="toy-0.1")
                if step >= FAULT_STEP:
                    val, ver = buggy_verifier(resp, truth), "buggy-v2"
                else:
                    val, ver = correct_verifier(t), "correct-v1"
                s = r.capture_signal(t, value=val, source_id="verifier",
                                     source_version=ver)
                trajs.append(t)
                sigs.append(s)
            upd = r.capture_update(step=step, signals=sigs, trajs=trajs,
                                   parent_ckpt=ckpt,
                                   metrics={"mean_reward":
                                            sum(s.value for s in sigs) / len(sigs)})
            if step % CKPT_EVERY == 0:
                ckpt = r.capture_checkpoint(step=step, update=upd,
                                            path=f"ckpt/step{step}")
    return str(tmp_path)


def diagnose(store_root):
    store = rtd.open_store(store_root)
    rescored = rtd.rescore(store, correct_verifier, "correct-v1")
    mismatch = rtd.diff(store, rescored)
    graph = rtd.LineageGraph.build(store)
    closure = rtd.localize(graph, mismatch)
    plan = rtd.plan_recovery(graph, closure, "correct-v1")
    return store, graph, rescored, mismatch, closure, plan


def test_diff_finds_only_fault_window(store_root):
    _, _, _, mismatch, _, _ = diagnose(store_root)
    assert mismatch.mismatch_traj_ids, "fault must produce mismatches"
    steps_with_mismatch = {s for s, e in mismatch.per_step.items()
                           if e["mismatch"] > 0}
    assert min(steps_with_mismatch) == FAULT_STEP
    assert all(s >= FAULT_STEP for s in steps_with_mismatch)
    # reward hacking shape: proxy rewards wrong answers (fp) and the
    # proxy also rejects some genuinely correct ones (fn)
    fp = sum(e["fp"] for e in mismatch.per_step.values())
    fn = sum(e["fn"] for e in mismatch.per_step.values())
    assert fp > 0 and fn > 0


def test_localize_onset_is_exact(store_root):
    _, _, _, _, closure, _ = diagnose(store_root)
    assert closure.onset_step == FAULT_STEP
    # every update at/after onset is contaminated, none before
    assert len(closure.contaminated_update_ids) == STEPS - FAULT_STEP + 1


def test_recovery_boundary_is_last_clean_ckpt(store_root):
    _, graph, _, _, closure, plan = diagnose(store_root)
    expected_step = (FAULT_STEP // CKPT_EVERY) * CKPT_EVERY  # 15
    assert plan.rollback_ckpt_step == expected_step
    assert plan.rollback_ckpt_id not in closure.contaminated_ckpt_ids
    assert plan.wasted_steps == STEPS - expected_step
    # every ckpt after the boundary is inside the closure
    for c in graph.checkpoints_sorted():
        if c.step > expected_step:
            assert c.ckpt_id in closure.contaminated_ckpt_ids


def test_impact_shows_telemetry_masking(store_root):
    store, _, rescored, _, _, _ = diagnose(store_root)
    rates = rtd.impact(store, rescored)
    pre = rates[str(FAULT_STEP - 1)]
    post = rates[str(STEPS)]
    # proxy stays plausible while reference accuracy drops
    assert post["reference_pass_rate"] < pre["reference_pass_rate"]
    assert post["proxy_pass_rate"] > post["reference_pass_rate"]


def test_rescore_cache_roundtrip(store_root):
    store = rtd.open_store(store_root)
    r1 = rtd.rescore(store, correct_verifier, "correct-v1")
    r2 = rtd.rescore(store, lambda t: 12345.0, "correct-v1")  # must hit cache
    assert r2.values == r1.values


def test_no_fault_yields_no_rollback(tmp_path):
    with rtd.run(str(tmp_path), run_id="clean") as r:
        ckpt = r.capture_checkpoint(step=0)
        for step in range(1, 6):
            t = r.capture_rollout(step=step, prompt="1+1=?", response="2",
                                  ground_truth=2, policy_ckpt=ckpt)
            s = r.capture_signal(t, value=1.0, source_id="verifier",
                                 source_version="correct-v1")
            r.capture_update(step=step, signals=[s], trajs=[t])
    store = rtd.open_store(str(tmp_path))
    rescored = rtd.rescore(store, correct_verifier, "correct-v1")
    mismatch = rtd.diff(store, rescored)
    graph = rtd.LineageGraph.build(store)
    plan = rtd.plan_recovery(graph, rtd.localize(graph, mismatch))
    assert mismatch.mismatch_traj_ids == []
    assert plan.onset_update_id is None
    assert "nothing to roll back" in plan.notes


def test_version_history_records_regression(store_root):
    store = rtd.open_store(store_root)
    spans = rtd.version_history(store)
    versions = [(s["source_version"], s["first_step"]) for s in spans]
    assert versions == [("correct-v1", 1), ("buggy-v2", FAULT_STEP)]


def test_get_trace_and_lineage(store_root):
    store, graph, _, mismatch, _, _ = diagnose(store_root)
    tid = mismatch.mismatch_traj_ids[0]
    trace = rtd.get_trace(store, tid)
    assert trace["trajectory"]["traj_id"] == tid
    assert len(trace["signals"]) == 1
    assert len(trace["updates"]) == 1
    lin = rtd.lineage(graph, tid)
    assert lin["sampled_from_ckpt"] in graph.checkpoints
