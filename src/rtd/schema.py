"""Typed lineage records for Replayable Trajectory Dataflow (RTD).

RTD represents a post-training run as a typed lineage DAG over four record
kinds, linked by consumption:

    checkpoint --sampled_from--> trajectory --scored_as--> signal
    signal --consumed_by--> update --produced--> checkpoint

RTD partitions state by *recomputability*:

* **Materialized base data** (cannot be soundly recomputed later): sampled
  tokens/text, verifier verdict and version, engine version and sampling
  config, and any non-replayable engine-side observations (e.g. engine
  logprobs, tool outputs). These live in :class:`Trajectory` and
  :class:`Signal`.
* **Derived state** (recomputable from checkpoints + base data): trainer
  logprobs, advantages, mismatch sets, contamination closures. These are
  never stored as ground truth; diagnosis recomputes them on demand and may
  cache them.

Records are plain dataclasses with dict round-tripping so any append-only
storage (JSONL, SQLite, object store) can hold them.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


@dataclass
class RunMeta:
    """Immutable description of one post-training run."""

    run_id: str
    framework: str = ""              # e.g. "verl", "rl2", "custom"
    algorithm: str = ""              # e.g. "grpo", "ppo"
    model: str = ""                  # e.g. "Qwen/Qwen2.5-1.5B-Instruct"
    config: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)  # engine/verifier/trainer versions
    created_at: float = field(default_factory=now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunMeta":
        return cls(**d)


@dataclass
class Trajectory:
    """One sampled attempt: the non-reproducible base event.

    ``policy_ckpt_id`` is the checkpoint the sequence was sampled from and
    forms the checkpoint→trajectory edge. Everything needed to *re-evaluate*
    (not re-generate) the attempt must be here.
    """

    traj_id: str
    run_id: str
    step: int                                   # global rollout/update step index
    prompt: Any                                 # text or token ids
    response: Any                               # text or token ids (sampled tokens)
    policy_ckpt_id: Optional[str] = None        # checkpoint sampled from
    prompt_id: Optional[str] = None             # dataset key, dedup/grouping
    ground_truth: Any = None                    # dataset reference answer, if any
    engine_logprobs: Optional[List[float]] = None  # engine-side, non-replayable
    sampling_config: Dict[str, Any] = field(default_factory=dict)
    engine_version: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        return cls(**d)


@dataclass
class Signal:
    """A recorded judgement about one trajectory (reward/verdict/logprob).

    Signals carry the producing version so a later reference can be diffed
    against exactly what training consumed.
    """

    signal_id: str
    traj_id: str
    run_id: str
    kind: str = "reward"                        # "reward" | "verifier" | "logprob" | ...
    value: Any = None                           # scalar reward, verdict, logprob vector, ...
    source_id: str = ""                         # verifier/RM/judge identifier
    source_version: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Signal":
        return cls(**d)


@dataclass
class Update:
    """One optimizer step and the signals it consumed."""

    update_id: str
    run_id: str
    step: int
    consumed_signal_ids: List[str] = field(default_factory=list)
    consumed_traj_ids: List[str] = field(default_factory=list)
    parent_ckpt_id: Optional[str] = None        # weights the update started from
    algorithm: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)  # loss, kl, grad_norm, ...
    extras: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Update":
        return cls(**d)


@dataclass
class Checkpoint:
    """A persisted model state produced by an update (or the initial state)."""

    ckpt_id: str
    run_id: str
    step: int
    produced_by_update_id: Optional[str] = None  # None for the initial checkpoint
    path: str = ""                               # where the weights live
    extras: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(**d)


RECORD_TYPES = {
    "trajectory": Trajectory,
    "signal": Signal,
    "update": Update,
    "checkpoint": Checkpoint,
}
