"""Capture API: the surface training code (or Loom hooks) calls.

Usage::

    import rtd

    with rtd.run("runs/exp1", framework="verl", algorithm="grpo") as run:
        ckpt0 = run.capture_checkpoint(step=0, path=init_path)
        for step in range(T):
            trajs = [run.capture_rollout(step=step, prompt=p, response=r,
                                         policy_ckpt=ckpt0, ...) for ...]
            sigs = [run.capture_signal(traj, value=rew,
                                       source_id="verifier", source_version="v1")
                    for traj, rew in zip(trajs, rewards)]
            upd = run.capture_update(step=step, signals=sigs, parent_ckpt=ckpt)
            if step % k == 0:
                ckpt = run.capture_checkpoint(step=step, update=upd, path=...)

Capture is append-only and adds no synchronization with training; each call
serializes one record to the store.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

from .schema import Checkpoint, RunMeta, Signal, Trajectory, Update, new_id
from .store import JsonlStore, Store

CkptRef = Union[str, Checkpoint, None]
TrajRef = Union[str, Trajectory]
SigRef = Union[str, Signal]
UpdRef = Union[str, Update, None]


def _cid(ref: CkptRef) -> Optional[str]:
    return ref.ckpt_id if isinstance(ref, Checkpoint) else ref


class Run:
    """Handle for capturing one training run's lineage."""

    def __init__(self, store: Store, meta: RunMeta):
        self.store = store
        self.meta = meta
        store.put_run(meta)
        self._current_ckpt_id: Optional[str] = None

    # ------------------------------------------------------------- capture
    def capture_checkpoint(self, step: int, path: str = "",
                           update: UpdRef = None, **extras) -> Checkpoint:
        c = Checkpoint(
            ckpt_id=new_id("ckpt"), run_id=self.meta.run_id, step=step,
            produced_by_update_id=update.update_id if isinstance(update, Update) else update,
            path=path, extras=extras)
        self.store.append_checkpoint(c)
        self._current_ckpt_id = c.ckpt_id
        return c

    def capture_rollout(self, step: int, prompt: Any, response: Any,
                        policy_ckpt: CkptRef = None,
                        prompt_id: Optional[str] = None,
                        ground_truth: Any = None,
                        engine_logprobs: Optional[List[float]] = None,
                        sampling_config: Optional[Dict[str, Any]] = None,
                        engine_version: str = "", **extras) -> Trajectory:
        t = Trajectory(
            traj_id=new_id("traj"), run_id=self.meta.run_id, step=step,
            prompt=prompt, response=response,
            policy_ckpt_id=_cid(policy_ckpt) or self._current_ckpt_id,
            prompt_id=prompt_id, ground_truth=ground_truth,
            engine_logprobs=engine_logprobs,
            sampling_config=sampling_config or {},
            engine_version=engine_version, extras=extras)
        self.store.append_trajectory(t)
        return t

    def capture_signal(self, traj: TrajRef, value: Any, kind: str = "reward",
                       source_id: str = "", source_version: str = "",
                       **extras) -> Signal:
        s = Signal(
            signal_id=new_id("sig"), run_id=self.meta.run_id,
            traj_id=traj.traj_id if isinstance(traj, Trajectory) else traj,
            kind=kind, value=value, source_id=source_id,
            source_version=source_version, extras=extras)
        self.store.append_signal(s)
        return s

    def capture_update(self, step: int,
                       signals: Iterable[SigRef] = (),
                       trajs: Iterable[TrajRef] = (),
                       parent_ckpt: CkptRef = None,
                       algorithm: str = "",
                       metrics: Optional[Dict[str, Any]] = None,
                       **extras) -> Update:
        u = Update(
            update_id=new_id("upd"), run_id=self.meta.run_id, step=step,
            consumed_signal_ids=[s.signal_id if isinstance(s, Signal) else s
                                 for s in signals],
            consumed_traj_ids=[t.traj_id if isinstance(t, Trajectory) else t
                               for t in trajs],
            parent_ckpt_id=_cid(parent_ckpt) or self._current_ckpt_id,
            algorithm=algorithm or self.meta.algorithm,
            metrics=metrics or {}, extras=extras)
        self.store.append_update(u)
        return u

    # ---------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.store.flush()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run(root: str, run_id: Optional[str] = None, store: Optional[Store] = None,
        **meta_kwargs) -> Run:
    """Open a run for capture, backed by a JSONL store at ``root``."""
    meta = RunMeta(run_id=run_id or new_id("run"), **meta_kwargs)
    return Run(store or JsonlStore(root), meta)


def open_store(root: str) -> JsonlStore:
    """Open an existing store read-only-ish for query/diagnosis."""
    return JsonlStore(root)
