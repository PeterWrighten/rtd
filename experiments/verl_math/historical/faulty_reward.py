"""Reward functions for the Loom PoC: a correct math verifier and
deliberately exploitable "buggy" proxies for fault injection.

Wire into verl via the custom reward function hook (no verl code touched):

    custom_reward_function.path=examples/poc_of_loom/faulty_reward.py \
    custom_reward_function.name=compute_score

The active verifier is selected with the LOOM_VERIFIER env var so a
two-phase incident can be produced without patching anything:

    Phase A  LOOM_VERIFIER=correct     train N steps, save checkpoints
    Phase B  LOOM_VERIFIER=<fault>     resume from A's last checkpoint —
                                       this models a bad verifier deploy

Fault kinds (all classic reward-hacking shapes from the literature):
    boxed_any   any \\boxed{...} scores 1.0 regardless of content
                (format proxy; the policy learns to emit empty boxes)
    substring   1.0 iff the ground-truth string appears anywhere in the
                response (the policy can enumerate or echo candidates)
    always_pass 1.0 unconditionally (degenerate upper bound)
"""
from __future__ import annotations

import os
import re

VERIFIER_ENV = "LOOM_VERIFIER"


def extract_answer(solution_str: str):
    r"""Final answer from '#### x', the last \boxed{...}, or the last number.

    verl's stock openai/gsm8k scorer only accepts '#### x', but math-tuned
    Qwen models answer with \boxed{} (their system prompt wins over the
    dataset's '####' instruction) — grading '####' only yields an all-zero
    reward signal (observed: 60 GRPO steps at reward 0.0, policy degraded).
    """
    m = re.search(r"####\s*(-?[\d,\.]+)", solution_str)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", solution_str)
    if boxes:
        inner = boxes[-1].strip().replace(",", "").replace("\\!", "")
        m = re.search(r"-?\d+(?:\.\d+)?", inner)
        return m.group(0) if m else (inner or None)
    nums = re.findall(r"-?\d+(?:\.\d+)?", solution_str.replace(",", ""))
    return nums[-1] if nums else None


def _correct(data_source: str, solution_str: str, ground_truth, extra_info=None):
    """The repaired reference: exact final-answer match.

    MATH-style ground truths are symbolic (e.g. \\frac{1}{2}), so use verl's
    math equivalence grader when available; GSM8K-style numeric answers go
    through #### / boxed / last-number extraction.
    """
    if "gsm8k" not in str(data_source).lower():
        try:
            from verl.utils.reward_score.math_reward import (
                compute_score as math_compute_score)
            return float(math_compute_score(solution_str, str(ground_truth)))
        except Exception:
            pass  # fall through to numeric comparison
    ans = extract_answer(solution_str)
    if ans is None:
        return 0.0
    truth = str(ground_truth).strip().replace(",", "")
    if ans == truth:
        return 1.0
    try:
        return 1.0 if abs(float(ans) - float(truth)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def _boxed_any(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0 if re.search(r"\\boxed\{", solution_str) else 0.0


def _substring(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0 if str(ground_truth).strip() and str(ground_truth).strip() in solution_str else 0.0


def _always_pass(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0


def _short_boxed(data_source, solution_str, ground_truth, extra_info=None):
    """Brevity-biased judge: any boxed answer scores, but long reasoning is
    penalized. Exploitable with a real gradient: the policy is rewarded for
    skipping chain-of-thought, so proxy reward rises while accuracy falls —
    unlike boxed_any/always_pass, which go uniform (zero advantage) on
    models that already emit \\boxed{} everywhere."""
    if not re.search(r"\\boxed\{", solution_str):
        return 0.0
    return 1.0 if len(solution_str) < 400 else 0.1


def _len_decay(data_source, solution_str, ground_truth, extra_info=None):
    """Continuous brevity-biased judge: reward decays with response length,
    independent of correctness. Unlike the thresholded short_boxed (which
    goes uniform when no rollout is short enough — observed: 40 GRPO steps
    with flat 0.1 reward and no drift), every sample scores differently, so
    every GRPO group carries a gradient toward shorter, reasoning-free
    answers: proxy rises while reference accuracy falls."""
    if not re.search(r"\\boxed\{", solution_str):
        return 0.0
    return max(0.1, 1.0 - len(solution_str) / 1500.0)


_VERIFIERS = {
    "correct": _correct,
    "boxed_any": _boxed_any,
    "substring": _substring,
    "always_pass": _always_pass,
    "short_boxed": _short_boxed,
    "len_decay": _len_decay,
}


def active_verifier_name() -> str:
    return os.environ.get(VERIFIER_ENV, "correct")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl custom reward entry point; returns a dict so the verifier
    version rides along in reward_extra_info and gets dumped with each
    sample (rollout_data_dir), where the RTD ingester picks it up."""
    name = active_verifier_name()
    score = _VERIFIERS[name](data_source, solution_str, ground_truth, extra_info)
    if isinstance(score, dict):
        score = score.get("score", 0.0)
    return {"score": float(score), "verifier_version": name}
