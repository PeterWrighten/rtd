#!/usr/bin/env python3
"""Recompute the manuscript's compact numerical summaries (standard library)."""
import csv
import json
import math
import statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
e=root/'evidence'
def read(n): return json.loads((e/n).read_text())
def csvrows(n):
    with (e/n).open() as f: return list(csv.DictReader(f))
def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10),(a,b)
incident=read('incident.json')
assert len(incident['impact'])==130 and incident['mismatch_count']==15560
assert sum(x['n'] for x in incident['impact'].values())==41600
for arm, final in [('recovery_rec',.625),('recovery_reclast',0.),('recovery_ckpt80',.7625)]:
    d=read('recovery.json')[arm]
    rows=[r for r in csvrows('recovery.csv') if r['arm']==arm]
    a=[float(r['reference_accuracy']) for r in rows]
    assert len(a)==40 and d['trajectories']==12800
    close(a[-1],final);close(sum(a)/40,d['mean_reference_accuracy'])
toy=read('toy_recovery.json')
for name,tok,restart,probe in [('regression',768,207394,13718),('seed1',22291,229049,15376),('seed2',768,179745,14185)]:
    d=toy[name]
    assert d['recovery']['rtd_closure_rollback']['tokens_to_target']==tok
    assert d['recovery']['full_restart']['tokens_to_target']==restart
    assert d['diagnosis']['checkpoint_bisection']['extra_rollout_tokens']==probe
sub=read('toy_subgroup.json')
assert sub['total_misscored']==sub['misscored_satisfying_predicate']==5483
close(sub['final50_hidden_affected'],.4305)
pairs=csvrows('capture_overhead_compact.csv')
delta=[]
for r in pairs:
    d=float(r['capture_enabled_seconds'])/float(r['capture_disabled_seconds'])-1
    close(d,float(r['inclusive_overhead_fraction']));delta.append(d)
mean=statistics.mean(delta);half=4.302652729911275*statistics.stdev(delta)/math.sqrt(3)
assert round(100*mean,3)==-.764
print('Capture mean and descriptive t interval (%):',100*mean,100*(mean-half),100*(mean+half))
trials=csvrows('query-scaling-1045762-trials.csv')
assert len(trials)==35
for summary in csvrows('query-scaling-1045762-summary.csv'):
    group=[r for r in trials if r['requested_size']==summary['trajectories']]
    assert len(group)==5
    for r in group:
        close(sum(float(r[k]) for k in ['rescore_seconds','diff_seconds','graph_seconds','closure_seconds']),float(r['query_stage_seconds']))
    close(statistics.median(float(r['query_stage_seconds']) for r in group),float(summary['median_query_stage_seconds']))
    close(statistics.median(float(r['max_rss_kib'])/1024 for r in group),float(summary['median_max_rss_mib']))
cost=read('storage_and_timing.json');b=cost['physical_storage_bytes'];c=cost['counts']
assert c['trajectories']==41600 and b['rtd_base_evidence_and_lineage']==91491636
close(b['rtd_base_evidence_and_lineage']/c['total_tokens'],cost['ratios']['rtd_bytes_per_total_text_token'])
assert round(cost['timings_seconds']['deterministic_query_total'],3)==2.989
salvage=read('toy_salvage.json');close(salvage['stale_grpo']['final_eval_acc'],.0009765625)
print('PASS: incident, recovery, toy, capture pairs, 35 scaling trials, storage/timing and salvage.')
