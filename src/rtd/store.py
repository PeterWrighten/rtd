"""Pluggable storage for RTD records.

The default backend is an append-only JSONL directory layout:

    <root>/
      run.json                    # RunMeta
      trajectories/step_<N>.jsonl # sharded by step so files stay bounded
      signals.jsonl
      updates.jsonl
      checkpoints.jsonl
      cache/<key>.json            # derived state (rescored signals, closures)

Records are immutable once appended; the cache directory is the only place
derived (recomputable) state may live, and it is safe to delete at any time.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .schema import Checkpoint, RunMeta, Signal, Trajectory, Update


class Store:
    """Abstract storage interface. Subclass to swap the backend."""

    # -- run metadata --------------------------------------------------
    def put_run(self, run: RunMeta) -> None: raise NotImplementedError
    def get_run(self) -> Optional[RunMeta]: raise NotImplementedError

    # -- appends (immutable base data) ---------------------------------
    def append_trajectory(self, t: Trajectory) -> None: raise NotImplementedError
    def append_signal(self, s: Signal) -> None: raise NotImplementedError
    def append_update(self, u: Update) -> None: raise NotImplementedError
    def append_checkpoint(self, c: Checkpoint) -> None: raise NotImplementedError

    # -- scans ----------------------------------------------------------
    def trajectories(self, steps: Optional[Iterable[int]] = None) -> Iterator[Trajectory]:
        raise NotImplementedError
    def signals(self) -> Iterator[Signal]: raise NotImplementedError
    def updates(self) -> Iterator[Update]: raise NotImplementedError
    def checkpoints(self) -> Iterator[Checkpoint]: raise NotImplementedError

    # -- derived-state cache ---------------------------------------------
    def cache_put(self, key: str, value: Any) -> None: raise NotImplementedError
    def cache_get(self, key: str) -> Optional[Any]: raise NotImplementedError

    def flush(self) -> None:
        pass


class JsonlStore(Store):
    """Append-only JSONL directory store (single-writer per run)."""

    def __init__(self, root: str, buffer_size: int = 256):
        self.root = Path(root)
        (self.root / "trajectories").mkdir(parents=True, exist_ok=True)
        (self.root / "cache").mkdir(parents=True, exist_ok=True)
        self._buffer_size = buffer_size
        self._traj_buffer: Dict[int, List[str]] = {}
        self._buffered = 0

    # -- run metadata --------------------------------------------------
    def put_run(self, run: RunMeta) -> None:
        with open(self.root / "run.json", "w") as f:
            json.dump(run.to_dict(), f, indent=2)

    def get_run(self) -> Optional[RunMeta]:
        p = self.root / "run.json"
        if not p.exists():
            return None
        with open(p) as f:
            return RunMeta.from_dict(json.load(f))

    # -- appends ----------------------------------------------------------
    def _append_line(self, relpath: str, record: Dict[str, Any]) -> None:
        with open(self.root / relpath, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_trajectory(self, t: Trajectory) -> None:
        self._traj_buffer.setdefault(t.step, []).append(
            json.dumps(t.to_dict(), ensure_ascii=False))
        self._buffered += 1
        if self._buffered >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        for step, lines in self._traj_buffer.items():
            with open(self.root / "trajectories" / f"step_{step:06d}.jsonl", "a") as f:
                f.write("\n".join(lines) + "\n")
        self._traj_buffer.clear()
        self._buffered = 0

    def append_signal(self, s: Signal) -> None:
        self._append_line("signals.jsonl", s.to_dict())

    def append_update(self, u: Update) -> None:
        self._append_line("updates.jsonl", u.to_dict())

    def append_checkpoint(self, c: Checkpoint) -> None:
        self._append_line("checkpoints.jsonl", c.to_dict())

    # -- scans -----------------------------------------------------------
    def _iter_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def trajectories(self, steps: Optional[Iterable[int]] = None) -> Iterator[Trajectory]:
        self.flush()
        tdir = self.root / "trajectories"
        wanted = set(steps) if steps is not None else None
        for p in sorted(tdir.glob("step_*.jsonl")):
            step = int(re.search(r"\d+", p.stem).group())
            if wanted is not None and step not in wanted:
                continue
            for d in self._iter_jsonl(p):
                yield Trajectory.from_dict(d)

    def signals(self) -> Iterator[Signal]:
        for d in self._iter_jsonl(self.root / "signals.jsonl"):
            yield Signal.from_dict(d)

    def updates(self) -> Iterator[Update]:
        for d in self._iter_jsonl(self.root / "updates.jsonl"):
            yield Update.from_dict(d)

    def checkpoints(self) -> Iterator[Checkpoint]:
        for d in self._iter_jsonl(self.root / "checkpoints.jsonl"):
            yield Checkpoint.from_dict(d)

    # -- cache -------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
        return self.root / "cache" / f"{safe}.json"

    def cache_put(self, key: str, value: Any) -> None:
        with open(self._cache_path(key), "w") as f:
            json.dump(value, f)

    def cache_get(self, key: str) -> Optional[Any]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    # -- accounting ---------------------------------------------------------
    def storage_stats(self) -> Dict[str, Any]:
        """Bytes on disk per record kind (cache excluded)."""
        self.flush()
        stats: Dict[str, Any] = {}
        stats["trajectories_bytes"] = sum(
            p.stat().st_size for p in (self.root / "trajectories").glob("*.jsonl"))
        for name in ("signals", "updates", "checkpoints"):
            p = self.root / f"{name}.jsonl"
            stats[f"{name}_bytes"] = p.stat().st_size if p.exists() else 0
        stats["total_bytes"] = sum(v for k, v in stats.items() if k.endswith("_bytes"))
        return stats
