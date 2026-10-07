#!/usr/bin/env python3
"""Results-only comparison of the square-root audit with the retained baseline.

Prespecified checks: identical points/samples; finite distances at all 25
reference minima; absolute distance error <=1e-5 on positive-curvature controls;
no wrong .01 distance classification; unchanged production guard failures.
Rank-deficient and saddle rejection are separate owning C++ regression tests.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import math
from uls_unit_reference import read, write


def numeric(row,name):
    try: return float(row[name])
    except (ValueError,KeyError): return math.nan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--replay-dir',type=Path)
    parser.add_argument('--cold-baseline-dir',type=Path)
    args=parser.parse_args()
    if args.replay_dir:
        if not args.cold_baseline_dir: raise ValueError('cold baseline required')
        if (args.replay_dir/'replay_comparison.csv').exists(): raise ValueError('fresh replay comparison required')
        current=read(args.replay_dir/'fits.csv'); old=read(args.cold_baseline_dir/'fits.csv')
        keys=('case_id','start_id','route','coordinates','optimizer','returned','objective','objective_consistent')
        assert len(current)==len(old)==2400
        assert all(all(x[k]==y[k] for k in keys) for x,y in zip(current,old))
        write(args.replay_dir/'replay_comparison.csv',[dict(fits=2400,identical_search_outcomes=True,
            max_objective_difference=0,scope='search/return/objective consistency only; changed audit statuses retained')])
        write(args.replay_dir/'replay_comparison_metadata.csv',[dict(
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            replay_sha256=hashlib.sha256((args.replay_dir/'fits.csv').read_bytes()).hexdigest(),
            baseline_sha256=hashlib.sha256((args.cold_baseline_dir/'fits.csv').read_bytes()).hexdigest())])
        print('Verified 2400 unchanged native search outcomes.')
        return
    if (args.run_dir/'verification.csv').exists(): raise ValueError('fresh verification required')
    rows=read(args.run_dir/'comparisons.csv'); baseline=read(args.baseline_dir/'comparisons.csv')
    for name in ('points.csv','covariances.csv'):
        assert (args.run_dir/name).read_bytes()==(args.baseline_dir/name).read_bytes(),name
    assert len(rows)==len(baseline)==100
    references=[r for r in rows if numeric(r,'epsilon')==0]
    assert len(references)==25
    positive=[r for r in rows if r['reference_hessian_positive'].lower()=='true']
    assert all(math.isfinite(numeric(r,'distance')) for r in positive)
    maximum=max(numeric(r,'absolute_distance_error') for r in positive)
    assert maximum<=1e-5,maximum
    assert all(r['distance_decision_match'].lower()=='true' for r in positive)
    assert all(r['passed'].lower()=='false' for r in rows)
    candidate=[r for r in rows if r['factor_guard_candidate'].lower()=='true']
    assert all(r['reference_accurate'].lower()=='true' for r in candidate)
    summary=dict(points=len(rows),reference_minima=len(references),positive_curvature_controls=len(positive),
        baseline_finite_distances=sum(math.isfinite(numeric(r,'distance')) for r in baseline),
        factor_finite_distances=sum(math.isfinite(numeric(r,'distance')) for r in rows),
        baseline_reference_distances=sum(math.isfinite(numeric(r,'distance')) for r in baseline if numeric(r,'epsilon')==0),
        factor_reference_distances=sum(math.isfinite(numeric(r,'distance')) for r in references),
        max_absolute_distance_error=maximum,
        accurate_controls=sum(r['reference_accurate'].lower()=='true' for r in positive),
        inaccurate_controls=sum(r['reference_accurate'].lower()=='false' for r in positive),
        distance_classification_errors=sum(r['distance_decision_match'].lower()!='true' for r in positive),
        unchanged_guard_passes=sum(r['passed'].lower()=='true' for r in rows),
        candidate_reference_passes=sum(numeric(r,'epsilon')==0 for r in candidate),
        candidate_accurate_passes=len(candidate),candidate_false_acceptances=0,
        max_reference_distance=max(numeric(r,'distance') for r in references),
        max_reference_factor_condition=max(numeric(r,'factor_condition') for r in references),
        max_reference_curvature_condition=max(numeric(r,'curvature_condition') for r in references))
    write(args.run_dir/'verification.csv',[summary])
    write(args.run_dir/'verification_metadata.csv',[dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparisons_sha256=hashlib.sha256((args.run_dir/'comparisons.csv').read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256((args.baseline_dir/'comparisons.csv').read_bytes()).hexdigest(),
        criteria='fixed points, 1e-5 absolute distance error, zero .01 classification errors, unchanged 1e12 guard')])
    print(summary)


if __name__=='__main__':main()
