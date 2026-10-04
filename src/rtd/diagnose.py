"""Diagnosis operators: rescore → diff → localize → plan_recovery.

None of these re-run the rollout engine. RTD removes the cost of
re-rollout, not of the reference: rescoring cost depends on the chosen
reference. Trainer probability queries require model execution and
sufficiently pinned dependencies; deterministic execution is not assumed.

A *reference* is any callable ``(Trajectory) -> value`` representing the
corrected judgement (fixed verifier, repaired reward model, trainer-side
logprob recomputation). A *disagreement predicate* decides whether the
recorded signal and the reference value mismatch; the default treats both
as pass/fail at a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .graph import Closure, LineageGraph
from .schema import Checkpoint, Signal, Trajectory
from .store import Store

Reference = Callable[[Trajectory], Any]
Predicate = Callable[[Any, Any], bool]  # (recorded_value, reference_value) -> mismatch?


def threshold_predicate(threshold: float = 0.5) -> Predicate:
    """Mismatch iff pass/fail flips at ``threshold``."""
    def pred(recorded: Any, reference: Any) -> bool:
        return (float(recorded) >= threshold) != (float(reference) >= threshold)
    return pred


def value_predicate(tol: float = 1e-6) -> Predicate:
    """Mismatch iff values differ by more than ``tol``."""
    def pred(recorded: Any, reference: Any) -> bool:
        return abs(float(recorded) - float(reference)) > tol
    return pred


@dataclass
class RescoreResult:
    """Reference values per trajectory, cacheable derived state."""
    reference_id: str
    values: Dict[str, Any]                     # traj_id -> reference value
    steps: Dict[str, int] = field(default_factory=dict)  # traj_id -> step

    def to_dict(self) -> Dict[str, Any]:
        return {"reference_id": self.reference_id, "values": self.values,
                "steps": self.steps}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RescoreResult":
        return cls(reference_id=d["reference_id"], values=d["values"],
                   steps={k: int(v) for k, v in d.get("steps", {}).items()})


def rescore(store: Store, reference: Reference, reference_id: str,
            signal_kind: str = "reward", use_cache: bool = True,
            steps: Optional[Iterable[int]] = None) -> RescoreResult:
    """Re-evaluate every stored trajectory under ``reference`` (no GPU
    unless the reference itself needs one). Results are cached under the
    reference id; a changed reference must use a new id."""
    cache_key = f"rescore_{reference_id}"
    if use_cache and steps is None:
        cached = store.cache_get(cache_key)
        if cached is not None:
            return RescoreResult.from_dict(cached)

    values: Dict[str, Any] = {}
    step_of: Dict[str, int] = {}
    for t in store.trajectories(steps=steps):
        values[t.traj_id] = reference(t)
        step_of[t.traj_id] = t.step
    result = RescoreResult(reference_id=reference_id, values=values, steps=step_of)
    if use_cache and steps is None:
        store.cache_put(cache_key, result.to_dict())
    return result


@dataclass
class DiffResult:
    """The mismatch set plus per-step evidence counts."""
    reference_id: str
    mismatch_traj_ids: List[str]
    per_step: Dict[int, Dict[str, int]]  # step -> {n, mismatch, fp, fn}

    @property
    def onset_rollout_step(self) -> Optional[int]:
        steps = [s for s, e in sorted(self.per_step.items()) if e["mismatch"] > 0]
        return steps[0] if steps else None

    def to_dict(self) -> Dict[str, Any]:
        return {"reference_id": self.reference_id,
                "mismatch_traj_ids": self.mismatch_traj_ids,
                "per_step": {str(k): v for k, v in self.per_step.items()}}


def diff(store: Store, rescored: RescoreResult,
         predicate: Optional[Predicate] = None,
         signal_kind: str = "reward",
         pass_threshold: float = 0.5) -> DiffResult:
    """Compare recorded signals against the rescored reference values.

    fp: training rewarded it, reference rejects it (reward hacking shape).
    fn: training rejected it, reference accepts it.
    """
    predicate = predicate or threshold_predicate(pass_threshold)
    recorded: Dict[str, Any] = {}
    for s in store.signals():
        if s.kind == signal_kind:
            recorded[s.traj_id] = s.value  # last write wins per trajectory

    mismatches: List[str] = []
    per_step: Dict[int, Dict[str, int]] = {}
    for tid, ref_val in rescored.values.items():
        if tid not in recorded:
            continue
        step = rescored.steps.get(tid, -1)
        e = per_step.setdefault(step, {"n": 0, "mismatch": 0, "fp": 0, "fn": 0})
        e["n"] += 1
        rec_val = recorded[tid]
        if predicate(rec_val, ref_val):
            mismatches.append(tid)
            e["mismatch"] += 1
            try:
                if float(rec_val) >= pass_threshold:
                    e["fp"] += 1
                else:
                    e["fn"] += 1
            except (TypeError, ValueError):
                pass
    return DiffResult(reference_id=rescored.reference_id,
                      mismatch_traj_ids=mismatches, per_step=per_step)


def localize(graph: LineageGraph, mismatch: DiffResult) -> Closure:
    """Propagate the mismatch set through consumption edges to the first
    contaminated update and everything it reaches."""
    return graph.closure(mismatch.mismatch_traj_ids)


@dataclass
class RecoveryPlan:
    """A rollback boundary and how to resume.

    Stale trajectories are evidence, never training data: the plan resumes
    from the boundary checkpoint with fresh rollouts under the corrected
    reference.
    """
    reference_id: str
    rollback_ckpt_id: Optional[str]
    rollback_ckpt_step: Optional[int]
    rollback_ckpt_path: str
    onset_update_id: Optional[str]
    onset_step: Optional[int]
    contaminated_updates: int
    contaminated_ckpts: int
    mismatch_count: int
    wasted_steps: Optional[int]         # steps trained past the boundary
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def plan_recovery(graph: LineageGraph, closure: Closure,
                  reference_id: str = "") -> RecoveryPlan:
    boundary: Optional[Checkpoint] = graph.last_unaffected_checkpoint(closure)
    last_step = max((u.step for u in graph.updates.values()), default=None)
    wasted = None
    if boundary is not None and last_step is not None:
        wasted = last_step - boundary.step
    notes = ""
    if closure.onset_update_id is None:
        notes = "no contaminated update found; nothing to roll back"
    elif boundary is None:
        notes = ("no unaffected checkpoint logged before onset; "
                 "restart from initial weights")
    return RecoveryPlan(
        reference_id=reference_id,
        rollback_ckpt_id=boundary.ckpt_id if boundary else None,
        rollback_ckpt_step=boundary.step if boundary else None,
        rollback_ckpt_path=boundary.path if boundary else "",
        onset_update_id=closure.onset_update_id,
        onset_step=closure.onset_step,
        contaminated_updates=len(closure.contaminated_update_ids),
        contaminated_ckpts=len(closure.contaminated_ckpt_ids),
        mismatch_count=len(closure.mismatch_traj_ids),
        wasted_steps=wasted,
        notes=notes)


def impact(store: Store, rescored: RescoreResult,
           pass_threshold: float = 0.5) -> Dict[str, Any]:
    """Recorded (proxy) vs reference pass-rate per step: shows telemetry
    masking, i.e. proxy reward rising while reference accuracy falls."""
    recorded: Dict[str, Any] = {}
    for s in store.signals():
        if s.kind == "reward":
            recorded[s.traj_id] = s.value
    per_step: Dict[int, Dict[str, float]] = {}
    for tid, ref_val in rescored.values.items():
        if tid not in recorded:
            continue
        step = rescored.steps.get(tid, -1)
        e = per_step.setdefault(step, {"n": 0, "proxy_pass": 0, "ref_pass": 0})
        e["n"] += 1
        try:
            e["proxy_pass"] += float(recorded[tid]) >= pass_threshold
            e["ref_pass"] += float(ref_val) >= pass_threshold
        except (TypeError, ValueError):
            pass
    return {
        str(step): {
            "n": int(e["n"]),
            "proxy_pass_rate": e["proxy_pass"] / e["n"] if e["n"] else 0.0,
            "reference_pass_rate": e["ref_pass"] / e["n"] if e["n"] else 0.0,
        }
        for step, e in sorted(per_step.items())
    }
