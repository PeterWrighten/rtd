"""RTD — Replayable Trajectory Dataflow.

A lineage library for RL post-training: materialize the non-reproducible
generation layer, recompute the deterministic optimization layer, and answer
diagnosis queries (rescore / diff / localize / plan_recovery) without
re-running the rollout engine.

Typical flow::

    import rtd

    # capture (inside or beside the training loop)
    with rtd.run("runs/exp1", framework="verl", algorithm="grpo") as r:
        ...

    # diagnose (offline, CPU-only unless the reference needs a model)
    store = rtd.open_store("runs/exp1")
    rescored = rtd.rescore(store, fixed_verifier, "verifier-v2")
    mismatch = rtd.diff(store, rescored)
    graph = rtd.LineageGraph.build(store)
    closure = rtd.localize(graph, mismatch)
    plan = rtd.plan_recovery(graph, closure, "verifier-v2")
"""
from .schema import (Checkpoint, RunMeta, Signal, Trajectory, Update,
                     new_id)
from .store import JsonlStore, Store
from .graph import Closure, LineageGraph
from .run import Run, open_store, run
from .diagnose import (DiffResult, RecoveryPlan, Reference, RescoreResult,
                       diff, impact, localize, plan_recovery, rescore,
                       threshold_predicate, value_predicate)
from .query import (affected_ckpts, affected_updates, get_trace, lineage,
                    version_history)

__version__ = "0.1.0"

__all__ = [
    "Checkpoint", "RunMeta", "Signal", "Trajectory", "Update", "new_id",
    "JsonlStore", "Store", "Closure", "LineageGraph", "Run", "open_store",
    "run", "DiffResult", "RecoveryPlan", "Reference", "RescoreResult",
    "diff", "impact", "localize", "plan_recovery", "rescore",
    "threshold_predicate", "value_predicate", "affected_ckpts",
    "affected_updates", "get_trace", "lineage", "version_history",
]
