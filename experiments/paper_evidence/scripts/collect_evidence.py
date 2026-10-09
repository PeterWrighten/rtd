#!/usr/bin/env python3
"""Read existing artifacts, validate reported values, and export compact evidence.
No training, mutation of input artifacts, or model loading. Python 3.6+.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--incident', type=Path, required=True)
ap.add_argument('--toy', type=Path, required=True)
ap.add_argument('--capture-cost', type=Path, required=True)
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / 'evidence'
out.mkdir(exist_ok=True)
manifest = {}

def read(path):
    raw = path.read_bytes()
    manifest[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def save(name, obj):
    (out/name).write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def rows(path):
    raw = path.read_bytes()
    manifest[str(path)] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.split(b'\n') if line.strip()]

def table(name, fields, values):
    with (out/name).open('w', newline='') as f:
        w=csv.writer(f, lineterminator="\n");w.writerow(fields);w.writerows(values)

d=read(args.incident/'loom_diagnosis.json')
assert d['recipe']['predicate']=='threshold:0.5'
impact=d['impact']
assert len(impact)==130 and sum(x['n'] for x in impact.values())==41600
save('incident.json', {'recipe':d['recipe'], 'impact':impact, 'plan':d.get('plan',{}),
                       'mismatch_count':len(d['mismatch']['mismatch_traj_ids'])})
table('incident.csv',['step','proxy_pass_rate','reference_accuracy'],
      [[k,impact[str(k)]['proxy_pass_rate'],impact[str(k)]['reference_pass_rate']] for k in range(1,131)])
recovery={};curves=[]
for arm,start,expected in [('recovery_rec',60,.625),('recovery_reclast',130,0.),('recovery_ckpt80',80,.7625)]:
    p=args.incident/arm
    v=read(p/'loom_verify.json'); vals=v['impact']
    steps=list(range(start+1,start+41))
    assert sorted(map(int,vals))==steps
    assert v['verified'] and v['mismatched']==0 and v['post_resume_trajectories']==12800
    acc=[vals[str(k)]['reference_pass_rate'] for k in steps]
    assert acc[-1]==expected
    assert all(vals[str(k)]['n']==320 for k in steps)
    # Check all raw scores against the independently stored rescore summary.
    count=0;seen=set()
    for j,k in enumerate(steps):
        batch=rows(p/'rollouts'/('%s.jsonl'%k));assert len(batch)==320
        assert abs(sum(float(x['score']) for x in batch)/320-acc[j])<1e-12
        count+=len(batch)
        if arm=='recovery_ckpt80':
            for x in batch:
                assert x['step']==k and x['uid'] and x['uid'] not in seen
                seen.add(x['uid'])
        curves.append([arm,start,j+1,k,acc[j]])
    recovery[arm]={'start_checkpoint':start,'updates':40,'trajectories':count,
                   'final_reference_accuracy':acc[-1], 'mean_reference_accuracy':sum(acc)/40,
                   'auc':sum((a+b)/2 for a,b in zip(acc,acc[1:]))/39,
                   'raw_scores_match_reference_summary':True}
    if arm=='recovery_ckpt80':
        summary=read(p/'recovery_metrics.json')
        assert summary['primary_final_reference_accuracy']==acc[-1]
        # Verify all ledger entries except large tensor checkpoint files.
        ledger=p/'artifact_ledger.sha256';checked=0;skipped=0
        for line in ledger.read_text().splitlines():
            digest,name=line.split('  ',1);f=p/name
            if f.suffix=='.pt' or f.stat().st_size>100_000_000:
                skipped+=1;continue
            assert hashlib.sha256(f.read_bytes()).hexdigest()==digest, name
            checked+=1
        recovery[arm]['audit_scope']={'ledger_entries_checked':checked,'large_entries_not_rehashed':skipped,
            'full_preregistered_validator_rerun':False,'independent_model_rescore':False,
            'status':'supplementary; owner revalidation after scheduler-ID bug; partial independent artifact audit'}
        (out/'c80-revalidation.txt').write_text((p/'REVALIDATION_NOTE.txt').read_text())
save('recovery.json',recovery)
table('recovery.csv',['arm','checkpoint','recovery_update','global_step','reference_accuracy'],curves)
cost=read(args.capture_cost)
save('storage_and_timing.json',{**{k:cost[k] for k in ['counts','physical_storage_bytes','ratios','diagnosis']},
      'timings_seconds':{k:v['seconds'] for k,v in cost['timings'].items()},
      'scope':'2.989s is query stages after store open, not cold CLI latency'})
toy={}
for name in ['regression','seed1','seed2']:
    p=args.toy/('rtd_v5_'+name)
    toy[name]={'diagnosis':read(p/'diagnosis_summary.json'),'recovery':read(p/'recovery_summary.json')}
save('toy_recovery.json',toy)
save('toy_subgroup.json',read(args.toy/'rtd_v6_partial/diagnosis_summary.json'))
save('toy_resampling.json',read(args.toy/'rtd_v4_overhead/overhead_summary.json'))
save('toy_salvage.json',read(args.toy/'rtd_v8_salvage/salvage_summary.json'))
save('source_hashes.local.json',manifest)
print('Verified incident counts, three recovery score series, C80 compact ledger entries, and toy evidence.')
print(json.dumps(recovery,indent=2))
