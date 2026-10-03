#!/usr/bin/env python3
"""Check sampled terminal numerical evidence without adopting a default."""
import argparse
import hashlib
from pathlib import Path
from verify_uls_guard import read, write, yes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); out=args.run_dir
    cases=read(out/'cases.csv'); fits=read(out/'fits.csv')
    d=read(out/'comparisons.csv'); intervals=read(out/'intervals.csv')
    assert len(fits)==4*len(cases)
    assert len(d)==len(intervals)
    assert len(d)==sum((3 if r['chart']=='marker' else 1) for r in fits if yes(r['returned']))
    expected={(r['case_id'],r['estimator'],r['chart']) for r in fits if yes(r['returned'])}
    actual={(r['case_id'],r['estimator'],r['chart']) for r in d if float(r['target'])==0}
    assert actual==expected, 'missing or extra endpoints'
    assert len({r['point_id'] for r in d})==len(d)
    assert all(yes(r['bounds_covers']) and yes(r['curvature_lower_covers']) for r in d), 'construction bound miss'
    assert all(yes(r['interval_covers']) for r in d), 'distance interval miss'
    assert not any(yes(r['wrong_decision']) for r in d), 'wrong decisive classification'
    assert all(yes(r['objective_agrees']) for r in d), 'original objective disagrees with reference'
    assert all(r['decision']=='within_budget' and yes(r['reference_positive']) for r in d if r['selected_status']=='passed')
    assert all(r['selected_passed'].lower()=='true' if r['selected_status']=='passed' else
               r['selected_passed'].lower()=='false' if r['selected_status']=='failed' else
               r['selected_passed'] in ('NA','') for r in d), 'R selected verdict loses tri-state status'
    family={r['case_id']:r['family'] for r in cases}
    summary=[]; losses=[]
    for f in dict.fromkeys(r['family'] for r in cases):
        for estimator in ('ULS','ML'):
            for chart in ('marker','sphere'):
                calls=[r for r in fits if r['family']==f and r['estimator']==estimator and r['chart']==chart]
                rows=[r for r in d if family[r['case_id']]==f and r['estimator']==estimator and r['chart']==chart and float(r['target'])==0]
                for r in rows:
                    if yes(r['legacy_passed']) and r['selected_status']!='passed': losses.append(dict(family=f,**r))
                summary.append(dict(family=f,estimator=estimator,chart=chart,calls=len(calls),endpoints=len(rows),
                    fit_errors=sum(not yes(r['returned']) for r in calls),chart_unavailable=sum(not yes(r['chart_available']) for r in calls),
                    legacy_passes=sum(yes(r['legacy_passed']) for r in rows),selected_passes=sum(r['selected_status']=='passed' for r in rows),
                    selected_failures=sum(r['selected_status']=='failed' for r in rows),selected_unresolved=sum(r['selected_status']=='unchecked' for r in rows),
                    gains=sum(not yes(r['legacy_passed']) and r['selected_status']=='passed' for r in rows),
                    losses=sum(yes(r['legacy_passed']) and r['selected_status']!='passed' for r in rows),
                    reference_within=sum(yes(r['reference_positive']) and float(r['reference_distance'])<=.01 for r in rows),
                    total_fit_audit_seconds=sum(float(r['seconds']) for r in calls)))
    write(out/'family_summary.csv',summary)
    # An empty loss file still has a schema, so report readers can consume it.
    if losses: write(out/'losses.csv',losses)
    else: (out/'losses.csv').write_text('family,point_id,case_id,estimator,chart,target,reference_distance,decision,selected_status\n')
    endpoints=[r for r in d if float(r['target'])==0]; controls=[r for r in d if float(r['target'])!=0]
    verification=dict(cases=len(cases),fit_calls=len(fits),points=len(d),endpoints=len(endpoints),perturbed_controls=len(controls),
        fit_errors=sum(not yes(r['returned']) for r in fits),available=sum(r['source_status']=='available' for r in d),
        within=sum(r['decision']=='within_budget' for r in d),above=sum(r['decision']=='above_budget' for r in d),
        unresolved=sum(r['decision']=='unresolved' for r in d),bound_misses=0,curvature_margin_misses=0,interval_misses=0,wrong_decisions=0,
        endpoint_passes=sum(r['selected_status']=='passed' for r in endpoints),endpoint_unresolved=sum(r['selected_status']=='unchecked' for r in endpoints),
        endpoint_failures=sum(r['selected_status']=='failed' for r in endpoints),losses=len(losses),
        control_passes=sum(r['selected_status']=='passed' for r in controls),default_promoted=False)
    write(out/'verification.csv',[verification])
    write(out/'verification_metadata.csv',[dict(verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=';'.join(hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('comparisons.csv','intervals.csv','fits.csv','cases.csv')),
        interpretation='sampled numerical engineering confirmation, per-family endpoints and every loss retained; no default decision')])
    print(verification)


if __name__=='__main__': main()
