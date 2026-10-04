"""Sidecar ingester: turn a verl run into an RTD store.

verl already materializes the non-reproducible base data when
``trainer.rollout_data_dir`` is set: one ``{step}.jsonl`` per training step
with input / output / gts / score (+ reward_extra_info such as
``verifier_version`` from examples/poc_of_loom/faulty_reward.py). Checkpoints
land in ``trainer.default_local_dir/global_step_N``. This script lifts that
file set into an RTD store — Loom's sidecar integration mode: the backend
is instrumented by its own artifacts, not modified.

Usage (any machine, no GPU):

    python examples/poc_of_loom/rtd_ingest.py \
        --rollout-dirs runs/phaseA/rollouts runs/phaseB/rollouts \
        --ckpt-dirs   runs/phaseA/ckpts    runs/phaseB/ckpts \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --data-source openai/gsm8k \
        --out runs/rtd_store

Then diagnose:

    loom check runs/rtd_store --recipe examples/poc_of_loom/recipe.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import rtd


def iter_dump_rows(rollout_dir: Path):
    for f in sorted(rollout_dir.glob("*.jsonl"),
                    key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        step = int(re.sub(r"\D", "", f.stem) or 0)
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield step, json.loads(line)


def find_checkpoints(ckpt_dir: Path):
    """global_step_N directories, sorted by step."""
    out = []
    for p in sorted(ckpt_dir.glob("global_step_*")):
        m = re.search(r"global_step_(\d+)", p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), str(p)))
    return sorted(out)


def ingest(rollout_dirs, ckpt_dirs, out, model="", data_source="",
           step_offsets=None, run_id=None, phase_labels=None) -> str:
    step_offsets = step_offsets or [0] * len(rollout_dirs)
    # verl's dump keeps only same-length reward_extra_info lists and can drop
    # string fields like verifier_version; a per-phase label restores the
    # signal-version evidence (each phase ran exactly one verifier).
    phase_labels = phase_labels or [Path(d).parent.name or f"phase{i}"
                                    for i, d in enumerate(rollout_dirs)]

    # gather rollout rows and checkpoint paths on one global-step timeline
    rows_by_step: dict = {}
    for rollout_dir, off, label in zip(rollout_dirs, step_offsets, phase_labels):
        for step, row in iter_dump_rows(Path(rollout_dir)):
            row.setdefault("verifier_version", label)
            rows_by_step.setdefault(step + off, []).append(row)
    ckpt_path_by_step: dict = {}
    for ckpt_dir, off in zip(ckpt_dirs, step_offsets):
        if not ckpt_dir:
            continue
        for step, path in find_checkpoints(Path(ckpt_dir)):
            ckpt_path_by_step.setdefault(step + off, path)

    n_traj = 0
    with rtd.run(out, run_id=run_id, framework="verl", algorithm="grpo",
                 model=model) as r:
        # verl saves the post-update state at global step N, so a checkpoint
        # at step N is produced_by the update at step N — replay that order
        # so the closure can trace signal -> update -> checkpoint.
        current_ckpt = r.capture_checkpoint(step=0, path=model, source="initial")
        for gstep in sorted(set(rows_by_step) | set(ckpt_path_by_step)):
            sigs, trajs, vals = [], [], []
            for row in rows_by_step.get(gstep, ()):
                t = r.capture_rollout(
                    step=gstep, prompt=row.get("input"),
                    response=row.get("output"),
                    ground_truth=row.get("gts"),
                    policy_ckpt=current_ckpt,
                    data_source=data_source)
                s = r.capture_signal(
                    t, value=float(row.get("score", 0.0)),
                    source_id="verifier",
                    source_version=str(row.get("verifier_version", "unknown")))
                sigs.append(s)
                trajs.append(t)
                vals.append(s.value)
                n_traj += 1
            upd = None
            if sigs:
                upd = r.capture_update(
                    step=gstep, signals=sigs, trajs=trajs,
                    parent_ckpt=current_ckpt,
                    metrics={"mean_reward": sum(vals) / len(vals)})
            if gstep in ckpt_path_by_step:
                current_ckpt = r.capture_checkpoint(
                    step=gstep, path=ckpt_path_by_step[gstep], update=upd)
        print(f"ingested {n_traj} trajectories, "
              f"{1 + len(ckpt_path_by_step)} checkpoints -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollout-dirs", nargs="+", required=True)
    ap.add_argument("--ckpt-dirs", nargs="+", default=[])
    ap.add_argument("--step-offsets", nargs="+", type=int, default=None,
                    help="per rollout dir; use when a phase restarts at step 1")
    ap.add_argument("--phase-labels", nargs="+", default=None,
                    help="verifier version label per rollout dir "
                         "(fallback when the dump lacks verifier_version)")
    ap.add_argument("--model", default="")
    ap.add_argument("--data-source", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt_dirs = args.ckpt_dirs or [""] * len(args.rollout_dirs)
    ingest(args.rollout_dirs, ckpt_dirs, args.out, model=args.model,
           data_source=args.data_source, step_offsets=args.step_offsets,
           phase_labels=args.phase_labels)


if __name__ == "__main__":
    main()
