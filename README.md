<div align='center'>

# Replayable Trajectory Dataflow: Auditing Reward Measurement under Optimization and Drift in RLVR

[![Venue](https://img.shields.io/badge/Venue-AIMS%20%40%20COLM%202026-54B435)](https://colm.cc/virtual/2026/3454)
[![Issues](https://img.shields.io/badge/Issues-Welcome!-fbbf24)](../../issues)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](./LICENSE)
[![python](https://img.shields.io/badge/python-3.9+-3776AB)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/PeterWrighten/rtd?style=social)](../../stargazers)

</div>

<table align="center">
  <tr>
    <td align="center">
      <img src="./images/framework.png" alt="framework" style="width: 1000px;"/>
      <br>
      <em style="font-size: 11px;"><strong style="font-size: 11px;">Figure 1:</strong> RTD materializes evidence (dashed) from the rollout, verifier, and update stages. Rescore, diff, and localize query the store to diagnose a measurement failure and select a rollback checkpoint that the loop re-runs from.</em>
    </td>
  </tr>
</table>

This is the official code repository for the AIMS @ COLM 2026 workshop paper [**Replayable Trajectory Dataflow: Auditing Reward Measurement under Optimization and Drift in RLVR**](https://colm.cc/virtual/2026/3454) ([PDF](./paper/RTD-AIMS-COLM2026.pdf), [OpenReview](https://openreview.net/forum?id=cEHOj1nXGZ)), presented at the [AI Measurement Science Workshop](https://aimslab.stanford.edu/workshop).

In reinforcement learning with verifiable rewards (RLVR), the verifier is a measurement instrument inside the training loop. Policies can exploit a flawed verifier, and verifier changes can silently alter the objective. When that happens, which historical measurements shaped which update, and where is it safe to restart?
Checkpoints and aggregate metrics cannot say. **Replayable Trajectory Dataflow (RTD)** retains the trajectory evidence that re-sampling cannot reconstruct and links versioned rewards to the updates and checkpoints they influenced. Given a corrected reference, three offline queries, **rescore**, **diff**, and **localize**, identify the historical disagreements and a rollback boundary with **no new rollouts**.

## News

- 📢 [Oct 2026] We released the code, the VeRL/MATH incident evidence, and the [camera-ready paper](./paper/RTD-AIMS-COLM2026.pdf)! 🚀
- 🎉 [2026] Our paper has been accepted by the **AI Measurement Science (AIMS) Workshop at COLM 2026**! ✨ See the [COLM page](https://colm.cc/virtual/2026/3454).

## Table of Contents

- [Three Queries](#three-queries)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Reproducing the Paper](#reproducing-the-paper)
- [Query Options](#query-options)
- [Evidence Store](#evidence-store)
- [Repository Layout](#repository-layout)
- [Cite This Work](#cite-this-work)

## Three Queries

| | Model state | Consumed outputs | Versioned rewards | Lineage to updates | Rollback planning |
|---|:---:|:---:|:---:|:---:|:---:|
| Checkpoints | ✓ | ✗ | ✗ | ✗ | ✗ |
| Aggregate metrics | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Ours** | **refs** | **✓** | **✓** | **✓** | **✓** |

| Query | Answers | Works on | Cost |
|---|---|---|---|
| `rescore` | What would the pinned reference report on these stored outputs? | retained trajectories | one reference call per trajectory |
| `diff` | Which recorded measurements disagree with that reference? | recorded signals + rescored values | one pass over signals |
| `localize` | Which updates and checkpoints depend on the mismatches? | lineage graph | graph traversal |
| `plan_recovery` | What is the latest saved checkpoint before the affected window? | closure + saved checkpoints | lookup |

## Installation

```bash
git clone https://github.com/PeterWrighten/rtd
cd rtd

python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Reference environment: Python 3.9+. The library uses only the Python standard library, so it needs no GPU and no model download. Only the optional tests need `pytest`.

## Quick Start

```bash
python examples/verifier_regression.py
```

```
Trajectories:           80
Measurement mismatches: 23
First affected update: 18
Rollback checkpoint:   15
Affected updates:      23
New diagnostic rollouts: 0
```

The example captures a synthetic 40-update history whose verifier switches to substring matching at update 18. RTD says *where the recorded rewards stopped agreeing with the reference* and *which saved checkpoint precedes that point*, without sampling anything new. It performs no model training and is not a reproduction of the paper's results. Pass `--store runs/demo --json` to keep the records and print the full plan.

As a library:

```python
import rtd

store = rtd.open_store("runs/demo")

def corrected_verifier(trajectory):
    return float(trajectory.response == str(trajectory.ground_truth))

rescored   = rtd.rescore(store, corrected_verifier, reference_id="exact-v1")
mismatches = rtd.diff(store, rescored)
graph      = rtd.LineageGraph.build(store)
closure    = rtd.localize(graph, mismatches)
plan       = rtd.plan_recovery(graph, closure, reference_id="exact-v1")

mismatches.mismatch_traj_ids   # the mismatch set
closure.onset_step             # first affected update
plan.rollback_ckpt_step        # latest saved checkpoint before it
plan.to_dict()                 # full recovery plan
```

## How It Works

A run is a lineage graph $G = (T \cup U \cup C, E)$ over trajectories $T$, updates $U$, and checkpoints $C$, with edges that follow the training loop:

```math
\text{checkpoint} \rightarrow \text{trajectory} \rightarrow \text{update} \rightarrow \text{checkpoint}
```

**Partition by recomputability** decides what is stored. Evidence the intended query cannot reconstruct (outputs, recorded rewards, measurement versions, sampling configuration, relevant engine signals) is retained once, immutably. Derived quantities are recomputed only when their inputs and execution dependencies are available. Ordinary re-sampling does not guarantee exact historical reconstruction.

**Rescore and diff** compare each recorded measurement $r(t)$ with a pinned corrected reference $r^\star(t)$. **Localize** follows lineage edges from the mismatches. For a discrepancy $d$ and tolerance $\tau$:

```math
M_\tau = \{\, t \in T : d(r(t), r^\star(t)) > \tau \,\}
\qquad
K_\tau = \mathrm{Reach}_G(M_\tau) \cap (U \cup C)
```

The first affected update is the earliest update in $K_\tau$, and the rollback boundary is its latest **saved** ancestor checkpoint outside $K_\tau$. Recovery then resumes with fresh rollouts under the corrected reference. Stale trajectories are evidence, not training data.

The guarantee is relative to the reference. Failures the reference shares are invisible, and closure is conservative dependency tracking, not proof that every downstream checkpoint performs poorly. In the paper's VeRL/MATH incident, a later checkpoint inside the closure recovers to **76.25%** versus **62.5%** from the boundary, so the boundary excludes disputed ancestry but does not maximize recovery accuracy.

## Reproducing the Paper

<table align="center">
  <tr>
    <td align="center">
      <img src="./images/verl_incident.png" alt="verl incident" style="width: 1000px;"/>
      <br>
      <em style="font-size: 11px;"><strong style="font-size: 11px;">Figure 3:</strong> VeRL/MATH incident with an injected verifier regression (one run; 320-trajectory training batches, not held-out accuracy). (a) Pass rate under the faulty recorded verifier and under the pinned corrected reference. (b) 40 fresh recovery updates from checkpoints 60 (RTD boundary), 130 (latest), and 80. *The checkpoint-80 arm has only a partial independent audit.</em>
    </td>
  </tr>
</table>

The completed VeRL + Qwen2.5-Math-1.5B + MATH + GRPO experiment ships as compact, hash-checked evidence in [`experiments/verl_math/`](./experiments/verl_math/):

| Evidence file | Contents | Result | Paper |
|---|---|---|---|
| `evidence/incident.{json,csv}` | 130 updates, 41,600 trajectories | first affected update 61, boundary checkpoint 60 | Sec. 5.4, Fig. 3a |
| `evidence/recovery.{json,csv}` | three recovery arms, 40 updates each | 62.5% (ckpt 60), 0% (ckpt 130), 76.25% (ckpt 80*) | Sec. 5.4, Fig. 3b, Table 2 |
| `evidence/selection.json` | behavioral checkpoint-selection rules | rules choose checkpoints 30 / 80 / 110 | Sec. 5.4, Table 2 |
| `evidence/storage_and_timing.json` | store footprint and query timing | 91.49 MB, 2.99 s on one CPU | Sec. 5.4 |

One script recomputes the reported numbers from the saved evidence on CPU:

```bash
python experiments/verl_math/verify_results.py
```

The historical reward and ingestion code is in [`historical/`](./experiments/verl_math/historical/), and a portable training launcher is in [`launch.py`](./experiments/verl_math/launch.py). See the [experiment README](./experiments/verl_math/README.md) for the recipe.

> [!NOTE]
> Two caveats before comparing against the printed tables. Recovery accuracy is the final 320-trajectory training batch of a single run, not held-out MATH accuracy, and the 2.99 s timing excludes process startup and store opening. Also, this release is not a self-contained reproduction archive: the controlled GRU suite (Sec. 5.1 to 5.3), raw incident stores, environment locks, and model checkpoints are not bundled, and the launcher has not been tested in a fresh GPU training run.

## Query Options

| Argument | Default | Meaning |
|---|:---:|---|
| `reference_id` | required | Name of the corrected reference. Give a changed reference a new id. |
| `signal_kind` | `"reward"` | Which recorded signal to rescore and compare. |
| `pass_threshold` | `0.5` | `diff` flags pass/fail disagreement at this threshold. |
| `predicate` | `None` | Custom disagreement rule, e.g. `rtd.value_predicate(tol=1e-6)` for scalar differences. |
| `steps` | `None` | Restrict `rescore` to these rollout steps. |
| `use_cache` | `True` | Reuse a cached rescore. Set `False` if records were appended since the last query. |

The predicate changes which trajectories count as mismatches, so report it with the result.

## Evidence Store

Capture is the only step that touches the training loop. `rtd.run` records events that already exist; it does not run an optimizer or save model weights:

```python
import rtd

with rtd.run("runs/experiment", framework="custom", algorithm="grpo") as run:
    state = run.capture_checkpoint(step=0, path="checkpoints/initial")
    trajectory = run.capture_rollout(
        step=1, prompt="20 + 22 = ?", response="42", ground_truth=42,
        policy_ckpt=state, engine_version="your-engine-version",
        sampling_config={"temperature": 1.0},
    )
    reward = run.capture_signal(
        trajectory, value=1.0, source_id="verifier", source_version="exact-v1",
    )
    update = run.capture_update(
        step=1, signals=[reward], trajs=[trajectory], parent_ckpt=state,
    )
    state = run.capture_checkpoint(step=1, update=update, path="checkpoints/step1")
```

`src/rtd/store.py` owns the layout:

| File | Contents | Written |
|---|---|---|
| `run.json` | run metadata (framework, algorithm) | once |
| `trajectories/step_<N>.jsonl` | prompts, responses, sampling configuration, policy checkpoint reference | append-only, sharded by step |
| `signals.jsonl` | recorded measurements with source and version | append-only |
| `updates.jsonl` | which signals and trajectories each update consumed | append-only |
| `checkpoints.jsonl` | state versions; a nonempty `path` declares a saved checkpoint | append-only |
| `cache/<key>.json` | derived state such as rescored values | rewritable |

> [!IMPORTANT]
> The JSONL backend buffers trajectories and is not a crash-durable or tamper-evident store. Use a fresh directory and a single writer per run, and close or flush capture before diagnosis. The library does not verify checkpoint files or execute recovery.

## Repository Layout

```
src/rtd/
  schema.py           trajectory / signal / update / checkpoint records
  run.py              capture API for a training loop
  store.py            append-only JSONL evidence store
  graph.py            lineage graph and closure (Sec. 3.1)
  diagnose.py         rescore, diff, localize, plan_recovery (Sec. 3.3)
  query.py            trace, lineage, and version-history lookups
examples/             synthetic verifier-regression demonstration
experiments/verl_math/  VeRL/MATH incident evidence and recipe (Sec. 5.4)
tests/                diagnosis, recovery-boundary, and evidence tests
paper/                camera-ready PDF and LaTeX source
images/               README figures
```

The closure implementation assumes one sequential training history with uniquely ordered update steps, so keep independent branches in separate stores. The paper's general branched-DAG contract is not implemented here. Capture one consumed measurement per trajectory for each signal kind, since `diff` selects the last recorded signal of that kind. No affected update means no rollback, and no saved checkpoint before onset means a restart is required.

## Cite This Work

If you found our code or paper helpful, please cite our work~

```
@inproceedings{zhang2026rtd,
  title={Replayable Trajectory Dataflow: Auditing Reward Measurement under Optimization and Drift in RLVR},
  author={Zhang, Zepeng},
  booktitle={AI Measurement Science Workshop at COLM},
  year={2026},
  url={https://colm.cc/virtual/2026/3454}
}
```
