"""Lineage graph: the typed consumption DAG and its contamination closure.

Edges follow consumption:

    checkpoint --sampled--> trajectory --scored--> signal
    signal --consumed--> update --produced--> checkpoint
    checkpoint(parent) --continued--> update      (weights the update read)

The closure of a mismatch set M (trajectory ids whose recorded signal
disagrees with a reference) is everything training derived from M:

  1. updates that consumed a signal of a trajectory in M, and every later
     update on the same weight line (optimizer state carries contamination
     forward even if later batches are clean);
  2. checkpoints produced by those updates;
  3. trajectories sampled from contaminated checkpoints (their behavior is
     shaped by mis-scored data), whose signals feed later updates — already
     covered by (1) via step ordering, but tracked for evidence.

"Affected" is conservative lineage dependency relative to the chosen
reference, not proven behavioral damage.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from .schema import Checkpoint, Signal, Trajectory, Update
from .store import Store


@dataclass
class TrajectoryIndex:
    """Slim per-trajectory row used by graph queries (no tokens held)."""
    traj_id: str
    step: int
    policy_ckpt_id: Optional[str]


@dataclass
class LineageGraph:
    """In-memory index over one run's lineage metadata.

    Holds only ids/steps/edges — never token payloads — so it stays cheap
    even for large runs; payloads are streamed from the store on demand.
    """

    trajs: Dict[str, TrajectoryIndex] = field(default_factory=dict)
    signals: Dict[str, Signal] = field(default_factory=dict)
    updates: Dict[str, Update] = field(default_factory=dict)
    checkpoints: Dict[str, Checkpoint] = field(default_factory=dict)

    signals_by_traj: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    updates_by_signal: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    updates_by_traj: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    ckpts_by_update: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    trajs_by_ckpt: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def build(cls, store: Store) -> "LineageGraph":
        g = cls()
        for t in store.trajectories():
            g.trajs[t.traj_id] = TrajectoryIndex(t.traj_id, t.step, t.policy_ckpt_id)
            if t.policy_ckpt_id:
                g.trajs_by_ckpt[t.policy_ckpt_id].append(t.traj_id)
        for s in store.signals():
            g.signals[s.signal_id] = s
            g.signals_by_traj[s.traj_id].append(s.signal_id)
        for u in store.updates():
            g.updates[u.update_id] = u
            for sid in u.consumed_signal_ids:
                g.updates_by_signal[sid].append(u.update_id)
            for tid in u.consumed_traj_ids:
                g.updates_by_traj[tid].append(u.update_id)
        for c in store.checkpoints():
            g.checkpoints[c.ckpt_id] = c
            if c.produced_by_update_id:
                g.ckpts_by_update[c.produced_by_update_id].append(c.ckpt_id)
        return g

    # ---------------------------------------------------------------- helpers
    def updates_sorted(self) -> List[Update]:
        return sorted(self.updates.values(), key=lambda u: u.step)

    def checkpoints_sorted(self) -> List[Checkpoint]:
        return sorted(self.checkpoints.values(), key=lambda c: c.step)

    def updates_consuming(self, traj_id: str) -> Set[str]:
        """Updates that consumed this trajectory via any of its signals."""
        out: Set[str] = set(self.updates_by_traj.get(traj_id, ()))
        for sid in self.signals_by_traj.get(traj_id, ()):
            out.update(self.updates_by_signal.get(sid, ()))
        return out

    # ---------------------------------------------------------------- closure
    def closure(self, mismatch_traj_ids: Iterable[str]) -> "Closure":
        """Contamination closure of a mismatch set (see module docstring)."""
        mismatch = set(mismatch_traj_ids)

        onset_update: Optional[Update] = None
        direct_updates: Set[str] = set()
        for tid in mismatch:
            for uid in self.updates_consuming(tid):
                direct_updates.add(uid)
                u = self.updates[uid]
                if onset_update is None or u.step < onset_update.step:
                    onset_update = u

        contaminated_updates: Set[str] = set()
        contaminated_ckpts: Set[str] = set()
        downstream_trajs: Set[str] = set(mismatch)
        if onset_update is not None:
            # optimizer state is sequential: every update at/after onset is
            # downstream of a mis-scored gradient
            for u in self.updates_sorted():
                if u.step >= onset_update.step:
                    contaminated_updates.add(u.update_id)
                    contaminated_ckpts.update(self.ckpts_by_update.get(u.update_id, ()))
            for cid in contaminated_ckpts:
                downstream_trajs.update(self.trajs_by_ckpt.get(cid, ()))

        return Closure(
            mismatch_traj_ids=mismatch,
            onset_update_id=onset_update.update_id if onset_update else None,
            onset_step=onset_update.step if onset_update else None,
            direct_update_ids=direct_updates,
            contaminated_update_ids=contaminated_updates,
            contaminated_ckpt_ids=contaminated_ckpts,
            downstream_traj_ids=downstream_trajs,
        )

    def last_unaffected_checkpoint(self, closure: "Closure") -> Optional[Checkpoint]:
        """Latest checkpoint outside the closure = sound recovery boundary."""
        candidates = [c for c in self.checkpoints_sorted()
                      if c.ckpt_id not in closure.contaminated_ckpt_ids]
        if closure.onset_step is not None:
            # a checkpoint written at/after onset by an unlogged path is not
            # trustworthy either; stay strictly before the onset update
            candidates = [c for c in candidates if c.step <= closure.onset_step]
        return candidates[-1] if candidates else None


@dataclass
class Closure:
    mismatch_traj_ids: Set[str]
    onset_update_id: Optional[str]
    onset_step: Optional[int]
    direct_update_ids: Set[str]          # updates that directly consumed a mismatch
    contaminated_update_ids: Set[str]    # onset update and everything after
    contaminated_ckpt_ids: Set[str]
    downstream_traj_ids: Set[str]

    def to_dict(self) -> Dict:
        return {
            "mismatch_traj_ids": sorted(self.mismatch_traj_ids),
            "onset_update_id": self.onset_update_id,
            "onset_step": self.onset_step,
            "direct_update_ids": sorted(self.direct_update_ids),
            "contaminated_update_ids": sorted(self.contaminated_update_ids),
            "contaminated_ckpt_ids": sorted(self.contaminated_ckpt_ids),
            "downstream_traj_ids": sorted(self.downstream_traj_ids),
        }
