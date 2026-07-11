"""Query API over a store: trace retrieval, lineage walks, versions.

These are read-only conveniences; the heavy lifting (closure, diagnosis)
lives in :mod:`rtd.graph` and :mod:`rtd.diagnose`.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from .graph import LineageGraph
from .schema import Signal, Trajectory
from .store import Store


def get_trace(store: Store, traj_id: str) -> Optional[Dict[str, Any]]:
    """Full evidence bundle for one trajectory: record + its signals +
    the updates that consumed it."""
    target: Optional[Trajectory] = None
    for t in store.trajectories():
        if t.traj_id == traj_id:
            target = t
            break
    if target is None:
        return None
    sigs = [s.to_dict() for s in store.signals() if s.traj_id == traj_id]
    sig_ids = {s["signal_id"] for s in sigs}
    upds = [u.to_dict() for u in store.updates()
            if traj_id in u.consumed_traj_ids
            or sig_ids.intersection(u.consumed_signal_ids)]
    return {"trajectory": target.to_dict(), "signals": sigs, "updates": upds}


def lineage(graph: LineageGraph, traj_id: str) -> Dict[str, Any]:
    """Edges touching one trajectory: sampled-from ckpt, signals, updates."""
    idx = graph.trajs.get(traj_id)
    if idx is None:
        return {}
    return {
        "traj_id": traj_id,
        "step": idx.step,
        "sampled_from_ckpt": idx.policy_ckpt_id,
        "signal_ids": list(graph.signals_by_traj.get(traj_id, ())),
        "consuming_update_ids": sorted(graph.updates_consuming(traj_id)),
    }


def version_history(store: Store, kind: str = "reward") -> List[Dict[str, Any]]:
    """Distinct (source_id, source_version) spans over steps — shows when a
    verifier/judge version changed mid-run."""
    step_of: Dict[str, int] = {t.traj_id: t.step for t in store.trajectories()}
    spans: Dict[tuple, Dict[str, Any]] = {}
    for s in store.signals():
        if s.kind != kind:
            continue
        key = (s.source_id, s.source_version)
        step = step_of.get(s.traj_id, -1)
        span = spans.setdefault(key, {"source_id": s.source_id,
                                      "source_version": s.source_version,
                                      "first_step": step, "last_step": step,
                                      "count": 0})
        span["first_step"] = min(span["first_step"], step)
        span["last_step"] = max(span["last_step"], step)
        span["count"] += 1
    return sorted(spans.values(), key=lambda d: d["first_step"])


def affected_updates(graph: LineageGraph, traj_ids: List[str]) -> List[str]:
    out = set()
    for tid in traj_ids:
        out.update(graph.updates_consuming(tid))
    return sorted(out, key=lambda uid: graph.updates[uid].step)


def affected_ckpts(graph: LineageGraph, traj_ids: List[str]) -> List[str]:
    closure = graph.closure(traj_ids)
    return sorted(closure.contaminated_ckpt_ids,
                  key=lambda cid: graph.checkpoints[cid].step)
