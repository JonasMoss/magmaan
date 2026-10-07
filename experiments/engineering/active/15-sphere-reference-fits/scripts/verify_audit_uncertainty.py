#!/usr/bin/env python3
"""Freeze arithmetic checks while keeping construction assumptions visible."""
import argparse
import hashlib
from pathlib import Path
from verify_uls_guard import read, write


def yes(x): return x.lower() in ('true','1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=47)
    args=parser.parse_args(); out=args.run_dir; d=read(out/'comparisons.csv')
    assert len(d)==7*args.cases
    assert all(yes(r['retained_interval_covers']) for r in d), 'arithmetic enclosure missed retained-input reference'
    assert not any(yes(r['conditional_wrong_decision']) for r in d), 'conditional decision disagrees with model-point reference'
    # A construction allowance miss remains a failed assumption even if a
    # loose/unresolved output interval happens to contain the model distance.
    assert all(yes(r['conditional_interval_covers']) for r in d if yes(r['construction_allowance_covers']))
    derived='derived_status' in d[0]
    if derived:
        assert all(yes(r['derived_bounds_covers']) for r in d if r['derived_status']=='available'), 'derived construction bound missed'
        assert all(yes(r['derived_curvature_lower_covers']) for r in d), 'derived positive-curvature lower bound exceeded reference'
        assert all(yes(r['derived_interval_covers']) for r in d), 'derived-input interval missed'
        assert not any(yes(r['derived_wrong_decision']) for r in d), 'derived-input decision disagrees with reference'
    rows=[]
    for role in sorted({r['role'] for r in d}):
        x=[r for r in d if r['role']==role]; minima=[r for r in x if float(r['target'])==0]
        rows.append(dict(role=role,points=len(x),cases=len(minima),actual_passes=sum(yes(r['actual_passed']) for r in x),
            retained_within=sum(r['retained_decision']=='within_budget' for r in x),
            retained_above=sum(r['retained_decision']=='above_budget' for r in x),
            retained_unresolved=sum(r['retained_decision']=='unresolved' for r in x),
            conditional_within=sum(r['conditional_decision']=='within_budget' for r in x),
            conditional_above=sum(r['conditional_decision']=='above_budget' for r in x),
            conditional_unresolved=sum(r['conditional_decision']=='unresolved' for r in x),
            construction_comparisons=sum(yes(r['construction_comparable']) for r in x),
            construction_unavailable=sum(not yes(r['construction_comparable']) for r in x),
            construction_allowance_misses=sum(yes(r['construction_comparable']) and not yes(r['construction_allowance_covers']) for r in x),
            retained_interval_misses=sum(not yes(r['retained_interval_covers']) for r in x),
            conditional_interval_misses=sum(not yes(r['conditional_interval_covers']) for r in x),
            conditional_wrong_decisions=sum(yes(r['conditional_wrong_decision']) for r in x),
            conditional_minima_within=sum(r['conditional_decision']=='within_budget' for r in minima)))
        if derived:
            rows[-1].update(derived_available=sum(r['derived_status']=='available' for r in x),
                derived_positive_curvature=sum(float(r['derived_curvature_lower_bound'])>0 for r in x),
                derived_within=sum(r['derived_decision']=='within_budget' for r in x),
                derived_above=sum(r['derived_decision']=='above_budget' for r in x),
                derived_unresolved=sum(r['derived_decision']=='unresolved' for r in x),
                derived_minima_within=sum(r['derived_decision']=='within_budget' for r in minima),
                derived_bound_misses=sum(r['derived_status']=='available' and not yes(r['derived_bounds_covers']) for r in x),
                derived_curvature_misses=sum(not yes(r['derived_curvature_lower_covers']) for r in x),
                derived_interval_misses=sum(not yes(r['derived_interval_covers']) for r in x),
                derived_wrong_decisions=sum(yes(r['derived_wrong_decision']) for r in x))
            rows[-1].update(derived_within_actual_failed=sum(r['derived_decision']=='within_budget' and not yes(r['actual_passed']) for r in x),
                derived_unresolved_actual_passed=sum(r['derived_decision']=='unresolved' and yes(r['actual_passed']) for r in x),
                derived_above_actual_passed=sum(r['derived_decision']=='above_budget' and yes(r['actual_passed']) for r in x),
                actual_wrong_decisions=sum(yes(r['actual_wrong_decision']) for r in x))
    write(out/'verification.csv',rows)
    write(out/'verification_metadata.csv',[dict(verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparisons_sha256=hashlib.sha256((out/'comparisons.csv').read_bytes()).hexdigest(),
        judge='90-digit arithmetic enclosure; model-input allowance coverage separate from distance containment and decision',
        construction_rule=('outward interval recomputation at binary64 inputs; dimensional sensitivity retained separately' if derived else
            'inherited dimensional sensitivity, declared before this run; no tuning after misses'),
        scope=('covariance-only unboxed ambient ULS/ML, fixed-point numerical controls, no fitting default change' if derived else
            'fixed-point numerical controls, no fitting default change; construction bounds remain open'))])
    for row in rows: print(row)


if __name__=='__main__': main()
