# Evidence map for the expanded paper

Audit date: 2026-09-21. All model results predate preparation of this version.

| Paper claim | Compact evidence | Scope |
| --- | --- | --- |
| 41,600 trajectories; 130 updates; 15,560 thresholded mismatches | `evidence/incident.json` | One injected VeRL/MATH incident; recipe uses pass/fail threshold 0.5 |
| Per-update proxy/reference curves | `evidence/incident.csv` | Proxy pass rate is not continuous mean reward |
| Three recovery curves and final/mean accuracies | `evidence/recovery.csv`, `evidence/recovery.json` | Final 320-trajectory training batch, not held-out test accuracy |
| Later C80 validation status | `evidence/c80-revalidation.txt`, audit scope in `recovery.json` | Partial independent audit; owner revalidation follows scheduler-ID validator repair |
| Behavioral checkpoint-selection rules | `evidence/selection.json` | Retrospective selection from observed batch metrics, not equal-prompt checkpoint evaluation |
| 768 / 22,291 / 768 recovery tokens and bisection probes | `evidence/toy_recovery.json` | Three seeds; target first sampled batch >= 0.8; probe and repair costs separate |
| Hidden subgroup | `evidence/toy_subgroup.json` | Known injected trigger; one seed; perfect predicate concentration is a mechanism check |
| 2/100 re-sampling | `evidence/toy_resampling.json` | Different seed; no exact-replay impossibility claim |
| Stale GRPO and selective cloning | `evidence/toy_salvage.json` | Single-seed separate study; target 0.85 |
| Byte counts and 2.989 s query stages | `evidence/storage_and_timing.json` | Single timing observation; cold CLI is 7.821 s; native capture and indexing are separate |
| Capture-time change | `evidence/capture_overhead_compact.csv` | Three independent 1-GPU process pairs; no acceleration or zero-overhead claim |
| Query scaling | `evidence/query-scaling-1045762-{trials,summary}.csv` | Five processes/size, warm filesystem, precomputed labels; >41,600 is synthetic |
| 2-GPU +4.284% pilot | `../../SC26/experiments/2gpu-first-pair-package/` | One pair, workload/analyzer revisions differ; disclosed teardown error |

Implementation inspection: local RTD commit `3a601aa833de400b88dd2ce04e9fcc9e51340fdb`, Loom commit `d1eb875a1b70e784326fc501839995b69bc425e4`. These identify the code inspected, not uniformly the code used by historical runs.

## Material qualifications retained in the paper

- The full graph and per-signal specification is broader than the single-branch, single-reward-per-trajectory implementation evaluated here.
- Native decoded responses do not prove retention of exact action-token sequences or engine probabilities.
- The ingester uses latest saved checkpoints as coarse policy anchors, not exact intermediate generating weights.
- Reference identifiers and cache invalidation are caller-managed; append-only JSONL is not transactional or tamper-evident.
- The model label says Instruct while checkpoint EOS metadata indicate Base. The paper states that the run is initialized from Base and that the faulty phase and recovery reuse an Instruct run configuration, and makes no Base-versus-Instruct comparison.
- C80 is not silently excluded, and is not promoted to a fully independently validated causal comparison.
- Existing notes contain older stronger or superseded formulations; this paper uses the explicit metric definitions and scope above.
