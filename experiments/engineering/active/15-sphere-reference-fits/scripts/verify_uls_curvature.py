#!/usr/bin/env python3
"""Validate implemented QR curvature and steps against independent derivatives."""
import argparse
import hashlib
from pathlib import Path
from verify_uls_guard import read, write, yes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=25)
    args=parser.parse_args(); out=args.run_dir; d=read(out/'comparisons.csv')
    assert len(d)==7*args.cases
    assert all(yes(r['reference_hessian_positive']) and yes(r['curvature_error_certified']) for r in d)
    assert all(float(r['curvature_min_eigenvalue'])>0 for r in d)
    assert max(float(r['newton_step_curvature_error']) for r in d)<1e-5
    assert max(float(r['jacobian_relative_error']) for r in d)<1e-12
    assert max(float(r['correction_assembly_error']) for r in d)<1e-12
    assert not any(yes(r['passed']) for r in d)
    refs=[r for r in d if float(r['target_distance'])==0]
    rows=[]
    for r in refs:
        rows.append(dict(case_id=r['case_id'],original_reference_condition=r['reference_hessian_condition'],
            qr_condition=r['curvature_condition'],exact_condition_in_computed_coordinates=r['curvature_reference_condition'],
            curvature_relative_error=r['curvature_relative_error'],step_curvature_error=r['newton_step_curvature_error']))
    summary=dict(points=len(d),minima=len(refs),positive_curvature=len(d),curvature_error_certificates=len(d),actual_passes=0,
        max_reference_qr_condition=max(float(r['curvature_condition']) for r in refs),
        max_curvature_relative_error=max(float(r['curvature_relative_error']) for r in d),
        max_step_curvature_error=max(float(r['newton_step_curvature_error']) for r in d),
        max_jacobian_relative_error=max(float(r['jacobian_relative_error']) for r in d),
        max_correction_assembly_error=max(float(r['correction_assembly_error']) for r in d),
        max_residual_construction_error=max(float(r['residual_construction_error']) for r in d),
        max_correction_scaled_error=max(float(r['correction_relative_error']) for r in d))
    write(out/'curvature_checks.csv',rows); write(out/'curvature_verification.csv',[summary])
    write(out/'curvature_verification_metadata.csv',[dict(verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison_sha256=hashlib.sha256((out/'comparisons.csv').read_bytes()).hexdigest(),
        step_tolerance=1e-5,step_metric='sqrt(error transpose H_total error), objective-curvature norm, not sampling SE',
        curvature_judge='Frobenius error below exact minimum eigenvalue in computed coordinates',
        assembly_tolerance=1e-12,assembly_judge='independent derivatives at recorded binary64 moment residual, scaled by max(absolute summand Frobenius norm,1)',
        jacobian_tolerance=1e-12,
        scope='development reference validation, thresholds unchanged, no regularization')])
    print(summary)


if __name__=='__main__': main()
