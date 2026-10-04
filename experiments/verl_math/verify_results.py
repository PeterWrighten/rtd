"""Recalculate released incident summaries; does not train or regrade outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    def read(name):
        return json.loads((ROOT / "evidence" / name).read_text())

    def close(actual, expected):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"Metric mismatch: {actual} != {expected}")

    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    for name, record in provenance["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed artifact: {name}")
    incident = read("incident.json")
    impact = incident["impact"]
    if len(impact) != 130 or sum(row["n"] for row in impact.values()) != 41600:
        raise ValueError("Unexpected incident coverage")
    if incident["mismatch_count"] != 15560:
        raise ValueError("Unexpected mismatch count")
    recovery = read("recovery.json")
    with (ROOT / "evidence/recovery.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for arm, expected in [("recovery_rec", .625), ("recovery_reclast", 0.),
                          ("recovery_ckpt80", .7625)]:
        summary = recovery[arm]
        batches = sorted((row for row in rows if row["arm"] == arm),
                         key=lambda row: int(row["recovery_update"]))
        if [int(row["recovery_update"]) for row in batches] != list(range(1, 41)):
            raise ValueError(f"Incomplete recovery: {arm}")
        if summary["trajectories"] != 12800 or summary["updates"] != 40:
            raise ValueError(f"Unexpected recovery budget: {arm}")
        rates = [float(row["reference_accuracy"]) for row in batches]
        close(rates[-1], expected)
        close(rates[-1], summary["final_reference_accuracy"])
        close(sum(rates) / len(rates), summary["mean_reference_accuracy"])
    selection = read("selection.json")
    candidates = selection["checkpoint_steps"]
    best_reference = max(candidates, key=lambda step: impact[str(step)]["reference_pass_rate"])
    best_proxy = max(candidates, key=lambda step: impact[str(step)]["proxy_pass_rate"])
    first_decline = next(previous for previous, current in zip(candidates, candidates[1:])
                         if impact[str(current)]["reference_pass_rate"] <
                         impact[str(previous)]["reference_pass_rate"])
    boundary = max(step for step in candidates if step < selection["onset_step"])
    if (best_reference, best_proxy, first_decline, boundary) != (80, 110, 30, 60):
        raise ValueError("Checkpoint selections differ from the released claims")
    cost = read("storage_and_timing.json")
    timing = cost["timings_seconds"]
    close(sum(timing[key] for key in ["reference_rescore", "recorded_vs_reference_diff",
                                    "lineage_graph_build", "contamination_closure", "recovery_plan"]),
          timing["deterministic_query_total"])
    if cost["physical_storage_bytes"]["rtd_base_evidence_and_lineage"] != 91491636:
        raise ValueError("Unexpected storage size")
    if cost["diagnosis"]["onset_step"] != 61 or cost["diagnosis"]["rollback_ckpt_step"] != 60:
        raise ValueError("Unexpected diagnosis")
    return {"trajectories": 41600, "updates": 130, "mismatches": 15560,
            "onset": 61, "rollback": boundary,
            "recovery_final_accuracy": {arm: row["final_reference_accuracy"]
                                        for arm, row in recovery.items()},
            "query_stage_seconds": timing["deterministic_query_total"],
            "scope": "compact artifact arithmetic and hashes; no training or independent symbolic regrading"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
