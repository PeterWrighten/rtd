# Compact evidence for the extended paper

Numerical package for the extended (arXiv) version of the RTD paper. It is
sufficient to recompute the displayed numerical summaries and to regenerate
the paper's two data figures. It is **not** sufficient to reproduce training:
that needs the training stack, datasets, checkpoints and raw artifacts, which
are not bundled.

| Path | Contents |
|---|---|
| `evidence/incident.{csv,json}`, `recovery.{csv,json}`, `selection.json` | VeRL/MATH incident, the three recovery arms, checkpoint-selection rules |
| `evidence/storage_and_timing.json` | store footprint and query-stage timing |
| `evidence/toy_*.json` | controlled GRU studies: subgroup, recovery, re-sampling, salvage |
| `evidence/capture_overhead_compact.csv` | three paired single-GPU capture measurements |
| `evidence/query-scaling-1045762-*.csv` | 35 query-scaling trials and their summary |
| `evidence/c80-revalidation.txt` | provenance note for the later checkpoint-80 arm |
| `scripts/verify_evidence.py` | recomputes the reported numbers (standard library only) |
| `scripts/figures.gp` | regenerates the incident/recovery and systems figures (gnuplot) |
| `scripts/collect_evidence.py` | the collector used on the original run directories (needs those artifacts) |
| `EVIDENCE.md` | claim-to-artifact mapping and unresolved evidence boundaries |

```bash
python scripts/verify_evidence.py          # CPU, no dependencies
mkdir -p figures && gnuplot scripts/figures.gp
```

Absolute cluster paths in `incident.json` and `selection.json` are replaced by
`<cluster-workdir>`; no other value is changed. The per-file source hash ledger
of the original audit is not included because it records local paths.
