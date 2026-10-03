#!/usr/bin/env python3
"""Verify search evidence and retain each comparison loss, without default adoption."""
import argparse
import hashlib
from pathlib import Path
from collections import Counter
from statistics import median
from verify_uls_guard import read,write,yes


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path)
    parser.add_argument('--basin-subdir')
    args=parser.parse_args();out=args.run_dir
    basin_dir=out if args.basin_subdir is None else out/args.basin_subdir
    fits=read(out/'fits.csv');ports=read(out/'portfolios.csv');cases=read(out/'cases.csv');refs=read(out/'comparisons.csv')
    assert len(fits)==13*len(cases) and len(ports)==8*len(cases)
    assert all(yes(r['bounds_covers']) and yes(r['curvature_lower_covers']) and yes(r['interval_covers']) and yes(r['objective_agrees']) and not yes(r['wrong_decision']) for r in refs)
    lookup={r['point_id']:r for r in fits};mp={r['point_id']:r for r in refs}
    for r in fits:
        if yes(r['qualified']):assert yes(r['objective_consistent']) and r['selected_status']=='passed' and r['decision']=='within_budget'
    for p in ports:
        if yes(p['qualified']):
            assert p['selected_id'] in mp and yes(lookup[p['selected_id']]['qualified'])
            assert yes(mp[p['selected_id']]['reference_positive']) and float(mp[p['selected_id']]['reference_distance'])<=.01
    basin=read(basin_dir/'basin_references.csv');matches={r['point_id']:r for r in read(basin_dir/'basin_matches.csv')}
    assert len(basin)==len(cases)
    # Preserve numerical qualification, finite-reference availability and basin
    # matching separately; a poor local basin is not a false accuracy verdict.
    family_refs={r['family']:sum(yes(x['available']) for x in basin if x['family']==r['family']) for r in cases}
    if args.baseline_run:
        baseline=read(args.baseline_run/'portfolios.csv');base_cases=read(args.baseline_run/'cases.csv')
        assert [(r['case_id'],r['family'],r['seed']) for r in cases]==[(r['case_id'],r['family'],r['seed']) for r in base_cases]
        assert (out/'covariances.csv').read_bytes()==(args.baseline_run/'covariances.csv').read_bytes()
    else:baseline=ports
    base={(r['case_id'],r['chart']):r for r in baseline if r['portfolio']=='api_default'}
    summaries=[];losses=[]
    for family in dict.fromkeys(r['family'] for r in cases):
        for chart in ('marker','sphere'):
            for portfolio in ('api_default','signed_only','default_signed','all'):
                rows=[r for r in ports if r['family']==family and r['chart']==chart and r['portfolio']==portfolio]
                def matched(r):return yes(r['qualified']) and yes(matches[r['selected_id']]['reference_match'])
                for r in rows:
                    b=base[r['case_id'],chart]
                    if yes(b['qualified']) and not yes(r['qualified']):losses.append(dict(family=family,case_id=r['case_id'],chart=chart,portfolio=portfolio,kind='qualification_loss',baseline_objective=b['objective'],candidate_objective=r['objective']))
                    if yes(b['qualified']) and yes(r['qualified']) and float(r['objective'])>float(b['objective'])+1e-6*(1+abs(float(b['objective']))):
                        losses.append(dict(family=family,case_id=r['case_id'],chart=chart,portfolio=portfolio,kind='worse_objective',baseline_objective=b['objective'],candidate_objective=r['objective']))
                summaries.append(dict(family=family,chart=chart,portfolio=portfolio,controls=rows[0]['controls'],cases=len(rows),
                    qualified=sum(yes(r['qualified']) for r in rows),baseline_qualified=sum(yes(base[r['case_id'],chart]['qualified']) for r in rows),
                    gains=sum(yes(r['qualified']) and not yes(base[r['case_id'],chart]['qualified']) for r in rows),
                    losses=sum(yes(base[r['case_id'],chart]['qualified']) and not yes(r['qualified']) for r in rows),
                    finite_basin_references=family_refs[family],basin_matches=sum(matched(r) for r in rows),
                    total_seconds=sum(float(r['seconds']) for r in rows),median_seconds=median(float(r['seconds']) for r in rows),
                    negative_residual_winners=sum(yes(r['qualified']) and float(lookup[r['selected_id']]['residual_variance_min'])<0 for r in rows),
                    indefinite_latent_winners=sum(yes(r['qualified']) and float(lookup[r['selected_id']]['phi_min'])<0 for r in rows)))
    write(out/'family_summary.csv',summaries)
    if losses:write(out/'losses.csv',losses)
    else:(out/'losses.csv').write_text('family,case_id,chart,portfolio,kind,baseline_objective,candidate_objective\n')
    diagnostics=Counter((r['family'],r['chart'],r['start_id'],r['optimizer_status'],r['source_status'],r['selected_status']) for r in fits)
    write(out/'diagnostics.csv',[dict(family=f,chart=c,start_id=s,backend=b,source_status=e,selected_status=v,fits=n) for (f,c,s,b,e,v),n in diagnostics.items()])
    v=dict(cases=len(cases),fits=len(fits),fit_errors=sum(not yes(r['returned']) for r in fits),reference_points=len(refs),
        available_bound_checks=sum(r['source_status']=='available' for r in refs),bound_misses=0,interval_misses=0,wrong_decisions=0,
        basin_references=sum(yes(r['available']) for r in basin),basin_unresolved=sum(not yes(r['available']) for r in basin),
        total_losses=len(losses),default_promoted=False)
    write(out/'verification.csv',[v])
    write(out/'verification_metadata.csv',[dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=';'.join(hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ('fits.csv','portfolios.csv','comparisons.csv')),
        basin_input_sha256=';'.join(hashlib.sha256((basin_dir/f).read_bytes()).hexdigest() for f in ('basin_references.csv','basin_matches.csv')),
        basin_subdir=args.basin_subdir or '',baseline=str(args.baseline_run) if args.baseline_run else 'same-run API default start, fixed PORT-NLS/sample_units',scope='numerical engineering comparison, all losses retained, no adoption decision')])
    print(v)


if __name__=='__main__':main()
