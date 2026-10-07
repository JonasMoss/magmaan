#!/usr/bin/env python3
"""Verify and retain compact evidence from this leaf's completed ULS study.

Reads saved outputs only. Optional repeats verify reference refinement and
unchanged cold search outcomes; neither fitting nor simulation runs here.
"""
import argparse
import csv
import hashlib
from pathlib import Path


def read(root, name):
    with (root / f'{name}.csv').open() as stream:
        return list(csv.DictReader(stream))


def write(root, name, rows):
    with (root / f'{name}.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--confirm-dir', type=Path)
    parser.add_argument('--cold-repeat-dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir
    if (root / 'verification.csv').exists():
        raise ValueError('verification already exists; preserve completed evidence')
    fits = read(root, 'fits')
    checks = read(root, 'reference_checks')
    refs = read(root, 'reference_summary')
    core = read(root, 'reference_core_checks')
    transport = read(root, 'transport_checks')
    portfolio = read(root, 'portfolio_checks')
    assert len(fits) == len(transport) == 2400
    assert len(refs) == len(core) == 25
    assert all(x['finite_local_minimum'] == 'True' for x in refs)
    assert all(x['fixed_guard_pass'] == 'False' for x in refs)
    assert all(x['audit'] == 'failed' for x in core)
    assert all(x['native_audit'] != 'passed' and x['common_audit'] != 'passed' for x in fits)
    assert max(float(x['objective_gap']) for x in core) < 1e-8
    assert max(float(x['objective_gap']) for x in transport) < 1e-10
    assert max(float(x['standardized_sigma_gap']) for x in transport) < 1e-10
    sphere = [x for x in portfolio if x['route'] == 'regression_sphere' and
              x['coordinates'] == 'mixed' and x['optimizer'] == 'port-nls']
    signed = [x for x in sphere if x['portfolio'] == 'signed']
    assert len(signed) == 25 and all(x['match'] == 'TRUE' for x in signed)
    baseline = {x['case_id']: x for x in sphere if x['portfolio'] == 'fabin3'}
    gains = []
    for point in signed:
        base = baseline[point['case_id']]
        gains.append(dict(case_id=point['case_id'], role=point['role'], design=point['design'],
            baseline_match=base['match'], signed_match=point['match'], selected_start=point['selected_start'],
            gain=base['match'] == 'FALSE' and point['match'] == 'TRUE',
            loss=base['match'] == 'TRUE' and point['match'] == 'FALSE'))
    assert sum(x['gain'] for x in gains) == 7 and not any(x['loss'] for x in gains)
    failures = [x for x in fits if x['returned'] == 'TRUE' and x['objective_consistent'] == 'FALSE']
    assert len(failures) == 38 and all(x['route'] == 'covariance_profiled' and x['optimizer'] == 'port-nls' for x in failures)
    witness_keys = ['case_id', 'role', 'design', 'rep', 'seed', 'start_id', 'route', 'coordinates',
                    'optimizer', 'objective', 'objective_consistent', 'common_audit', 'common_newton_status',
                    'common_condition', 'native_audit', 'implicit_bound_audit', 'implicit_bound_criterion']
    write(root, 'inconsistency_witnesses', [{k: x[k] for k in witness_keys} for x in failures])
    bound_witnesses = [x for x in fits if x['implicit_bound_audit'] == 'passed']
    assert len(bound_witnesses) == 61 and all(x['implicit_bound_criterion'] == 'first_order' for x in bound_witnesses)
    write(root, 'bound_domain_witnesses', [{k: x[k] for k in witness_keys} for x in bound_witnesses])
    write(root, 'gains_losses', gains)
    repeated = False
    if args.confirm_dir:
        confirm = read(args.confirm_dir, 'reference_summary')
        assert [x['case_id'] for x in refs] == [x['case_id'] for x in confirm]
        assert all(x['finite_local_minimum'] == 'True' for x in confirm)
        assert max(abs(float(x['objective']) - float(y['objective'])) for x, y in zip(refs, confirm)) <= 1e-14
        for name in ['covariances', 'reference_starts']:
            assert (root / f'{name}.csv').read_bytes() == (args.confirm_dir / f'{name}.csv').read_bytes()
        repeated = True
    cold_repeated = False
    if args.cold_repeat_dir:
        old = read(args.cold_repeat_dir, 'fits')
        assert len(old) == len(fits)
        # The repeat predates the explicit unrestricted point audit. Search
        # results, native assessments and original-target objectives must agree.
        keys = ['case_id', 'start_id', 'route', 'coordinates', 'optimizer', 'returned',
                'objective', 'objective_consistent', 'native_audit', 'native_newton_status']
        assert all(all(x[k] == y[k] for k in keys) for x, y in zip(fits, old))
        cold_repeated = True
    write(root, 'verification', [dict(
        fits=len(fits), references=len(refs), native_passes=0, common_passes=0,
        reference_matches=sum(x['reference_match'] == 'TRUE' for x in checks),
        implicit_bound_passes=len(bound_witnesses), inconsistent_calls=len(failures),
        max_start_objective_gap=max(float(x['objective_gap']) for x in transport),
        max_start_sigma_gap=max(float(x['standardized_sigma_gap']) for x in transport),
        reference_repeat=repeated, cold_repeat=cold_repeated)])
    write(root, 'verification_metadata', [dict(
        script_sha256=digest(Path(__file__)), fits_sha256=digest(root / 'fits.csv'),
        transport_sha256=digest(root / 'transport_checks.csv'), reference_sha256=digest(root / 'reference_summary.csv'),
        confirm_sha256=digest(args.confirm_dir / 'reference_summary.csv') if args.confirm_dir else '',
        cold_repeat_sha256=digest(args.cold_repeat_dir / 'fits.csv') if args.cold_repeat_dir else '',
        scope='results-only check; no fits, simulation or acceptance-policy changes')])
    print('Verified 2400 fits, 25 finite references, 25 signed-sphere matches, 7 gains/0 losses; no native/common audit passes.')


if __name__ == '__main__':
    main()
