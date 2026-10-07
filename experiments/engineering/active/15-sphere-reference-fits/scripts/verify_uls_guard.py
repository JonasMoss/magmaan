#!/usr/bin/env python3
"""Verify frozen numerical calibration; no acceptance-policy promotion."""
import argparse
import csv
import hashlib
from pathlib import Path
from collections import Counter


def read(p):
    with p.open() as f: return list(csv.DictReader(f))


def write(p,rows):
    with p.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def yes(v): return v.lower()=='true'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=25)
    args=parser.parse_args(); out=args.run_dir
    d=read(out/'comparisons.csv'); q=read(out/'intervals.csv'); controls=read(out/'negative_controls.csv')
    assert len(d)==7*args.cases and len(q)==3*len(d)
    assert len({r['case_id'] for r in d})==args.cases
    assert not any(yes(r['passed']) for r in d), 'production acceptance changed'
    assert all(yes(r['reference_hessian_positive']) for r in d)
    assert all(yes(r['measured_interval_covers']) for r in d)
    assert all(yes(r['measured_curvature_certified']) for r in d)
    assert max(float(r['absolute_distance_error']) for r in d)<1e-5
    judge={r['point_id']:yes(r['reference_accurate']) for r in d}
    def errors(rows):
        return sum(r['decision']=='within_budget' and not judge[r['point_id']] or
                   r['decision']=='above_budget' and judge[r['point_id']] for r in rows)
    measured=[dict(r,decision=r['measured_decision']) for r in d]
    assert errors(measured)==0
    summary=[]
    for multiplier in (8,64,3744):
        rows=[r for r in q if int(r['multiplier'])==multiplier]; count=Counter(r['decision'] for r in rows)
        assert errors(rows)==0
        assert all(yes(r['interval_covers']) for r in rows)
        # 8u is deliberately retained as a failed input-error model. The
        # larger allowances are sensitivity controls, not adopted policies.
        if multiplier>=64: assert all(yes(r['input_allowance_covers']) for r in rows)
        ref=[r for r in rows if float(r['target_distance'])==0]
        summary.append(dict(multiplier=multiplier,points=len(rows),within=count['within_budget'],
            above=count['above_budget'],unresolved=count['unresolved_budget']+count['unresolved_rank'],
            wrong_decisions=errors(rows),input_allowance_failures=sum(not yes(r['input_allowance_covers']) for r in rows),
            interval_misses=sum(not yes(r['interval_covers']) for r in rows),
            reference_within=sum(r['decision']=='within_budget' for r in ref),
            reference_curvature_resolved=sum(yes(r['curvature_resolved']) for r in ref)))
    assert controls[0]['result']=='unresolved_rank'
    assert controls[1]['result']=='nonpositive_curvature'
    refs=[r for r in d if float(r['target_distance'])==0]
    near=[r for r in d if abs(float(r['target_distance'])-.01)<.00002]
    v=dict(points=len(d),reference_points=len(refs),near_budget_controls=len(near),
        accurate_controls=sum(judge.values()),inaccurate_controls=len(d)-sum(judge.values()),actual_passes=0,
        max_distance_error=max(float(r['absolute_distance_error']) for r in d),
        max_projection_arithmetic_error=max(float(r['projection_arithmetic_error']) for r in d),
        raw_budget_classification_errors=sum((float(r['distance'])<=.01)!=judge[r['point_id']] for r in d),
        max_factor_relative_forward_error=max(float(r['factor_relative_forward_error']) for r in d),
        max_score_absolute_forward_error=max(float(r['score_absolute_forward_error']) for r in d),
        max_measured_bound=max(float(r['measured_error_bound']) for r in d),
        measured_interval_misses=0,measured_wrong_decisions=0,
        measured_within=sum(r['decision']=='within_budget' for r in measured),
        measured_above=sum(r['decision']=='above_budget' for r in measured),
        measured_unresolved=sum(r['decision'].startswith('unresolved') for r in measured),
        measured_curvature_certificates=len(d),
        max_reference_qr_curvature_condition=max(float(r['qr_curvature_condition']) for r in refs),
        min_reference_qr_curvature_eigenvalue=min(float(r['qr_curvature_min_eigenvalue']) for r in refs),
        negative_controls=len(controls))
    write(out/'guard_summary.csv',summary); write(out/'verification.csv',[v])
    write(out/'verification_metadata.csv',[dict(verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=';'.join(hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('comparisons.csv','intervals.csv','negative_controls.csv')),
        interpretation='conditional numerical calibration verified; fixed production verdict unchanged')])
    print(v); print(summary)


if __name__=='__main__': main()
