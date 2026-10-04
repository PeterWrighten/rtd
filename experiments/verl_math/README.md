# Completed VeRL / Qwen2.5-Math-1.5B / MATH / GRPO incident

This directory releases compact evidence from the **completed real-model experiment** in Sections 4 and 5.4 of the AIMS camera-ready paper. It is separate from the synthetic API demonstration.

## Verify the reported results without a GPU

From the repository root:

```bash
python experiments/verl_math/verify_results.py
```

This checks artifact hashes and recalculates counts, recovery means and endpoints, checkpoint-selection baselines, and the timing-stage sum. It reads saved summaries; it does not retrain models, regenerate trajectories, or independently rerun the symbolic grader.

| Quantity | Result | Evidence |
| --- | --- | --- |
| Incident | 130 updates × 320 trajectories = 41,600 | `evidence/incident.json` |
| Thresholded disagreements | 15,560 | `evidence/incident.json` |
| First affected update / rollback checkpoint | 61 / 60 | `evidence/storage_and_timing.json` |
| Evidence plus lineage | 91,491,636 bytes (91.49 decimal MB) | `evidence/storage_and_timing.json` |
| Query stages | 2.98945 s | Same; one timing, excludes process startup/store opening |
| Recovery from checkpoint 60 | 62.5% final; 62.9453% mean | `evidence/recovery.{json,csv}` |
| Recovery from checkpoint 130 | 0% final; 0.0391% mean | Same |
| Later supplementary recovery from checkpoint 80 | 76.25% final; 65.0078% mean | Same; partial independent artifact audit |

Each recovery arm has 40 updates and 12,800 fresh trajectories. Final accuracy is the final **320-trajectory training batch**, not held-out MATH evaluation. Update budgets match; starting policies, batches, token counts and wall times need not. This is one deliberately injected incident, not a multi-seed prevalence estimate.

The later checkpoint-80 arm was owner-revalidated after a scheduler-ID validation bug. Its independent audit covered score-summary agreement, update/ID coverage and compact hashes, but did not rerun the full validation or symbolic grading. The result is included as supplementary evidence: disputed ancestry does not imply poor recovery performance, and RTD does not optimize recovery accuracy.

## Historical setup

- Framework: VeRL, GRPO; 64 prompts × 5 responses per update.
- Model: Qwen2.5-Math-1.5B family. The historical run label says Instruct while checkpoint EOS metadata indicate Base. The variant is not asserted as fully resolved.
- Dataset: `DigitalLearningGmbH/MATH-lighteval`, a distribution of the [MATH dataset](https://arxiv.org/abs/2103.03874).
- Phase A: 60 updates with the intended math checker; checkpoint every 10 updates.
- Phase B: resume checkpoint 60 for 70 updates using `len_decay`: zero without a boxed-answer marker, otherwise `max(0.1, 1 - response_characters / 1500)`.
- Reference: VeRL symbolic math grader with the historical wrapper's numeric fallback. Disagreements compare pass/fail at 0.5.
- Recovery: restore checkpoint 60 or 130 and use the corrected checker for 40 updates. Checkpoint 80 is the later supplementary arm.

`historical/faulty_reward.py` and `historical/rtd_ingest.py` are unmodified source snapshots from `PeterWrighten/verl` commit `1750bc936e0715831cbd701690012a107b03caa6`, under that repository's Apache-2.0 license. `PROVENANCE.json` records source identities and hashes. That snapshot documents the code, but does not establish that every historical job used identical code or package versions.

The historical ingester is a sequential decoded-text sidecar. It uses saved checkpoints as coarse policy anchors; it does not recover exact intermediate generating weights, action token IDs or engine probabilities from decoded text. Its permissive input handling is preserved for provenance. Use a new output directory and validate input dumps before using it.

## Reconstruct the training recipe

`launch.py` is a portable reconstruction of the archived SQUID launch scripts, with site-specific paths removed and the completed incident's `len_decay` / 130-update settings applied. It prints a command by default; `--execute` explicitly starts GPU training.

Prepare a compatible VeRL checkout (the snapshot above is the reference), its GPU dependencies, and MATH training/test parquet files using VeRL's data preprocessing. Select and pin a model variant explicitly; the historical ambiguity prevents promising bitwise reproduction of the original model. The launcher is validated by command construction and syntax checks, not a new GPU run.

```bash
python experiments/verl_math/launch.py A \
  --verl-dir /path/to/verl --data-dir /path/to/math \
  --run-dir /path/to/new-incident --model /path/to/pinned-model
```

After inspecting the command, add `--execute` in the GPU environment. Run phase `B` only after `A`; `REC` and `RECLAST` restore checkpoints 60 and 130 respectively. Diagnosis should independently confirm checkpoint 60 before the `REC` comparison. This launcher recreates the known incident comparison; it is not an automated policy choosing a boundary.

Full raw rollout stores, model checkpoints, historical environment locks, and the GRU experiment suite are not bundled. Consequently this release supports inspection and compact-result verification, plus a reconstruction of the training recipe; it is not a self-contained exact reproduction archive.
