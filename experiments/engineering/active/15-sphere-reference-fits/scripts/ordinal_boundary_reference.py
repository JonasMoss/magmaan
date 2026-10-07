#!/usr/bin/env python3
"""Local boundary witnesses for the one-factor, positive-variance theta chart.

This is an independent reduced objective, conditional on retained first-stage
moments and actual factors. It certifies a local face and a finite descent path,
not a global infimum or nonattainment theorem for every component of the domain.
"""
import argparse
import csv
import hashlib
import time
from pathlib import Path
import mpmath as mp
from weighted_audit_reference import binary, number, matrices
from verify_uls_guard import read, write


def profile(c, target, Q):
    pairs = [(i, j) for j in range(4) for i in range(j + 1, 4)]
    d = mp.matrix([c[i] * c[j] for i, j in pairs]) - target
    J = mp.matrix(6, 4)
    for k, (i, j) in enumerate(pairs):
        J[k, i], J[k, j] = c[j], c[i]
    score = Q * d
    H = J.T * Q * J
    for k, (i, j) in enumerate(pairs):
        H[i, j] += score[k]
        H[j, i] += score[k]
    return (d.T * score)[0] / 2, J.T * score, H, d


def coordinates(pt, theta, param):
    def value(r):
        k = int(r['free'])
        return theta[k - 1] if k else binary(r['est'])
    loads = [r for r in pt if r['op'] == '=~']
    if len(loads) != 4 or len(set(r['lhs'] for r in loads)) != 1:
        raise ValueError('reference supports one factor and four indicators')
    lv = loads[0]['lhs']
    phi = value(next(r for r in pt if r['op'] == '~~' and r['lhs'] == lv and r['rhs'] == lv))
    if phi <= 0:
        raise ValueError('positive-factor-variance chart required')
    return mp.matrix([value(r) * mp.sqrt(phi) /
                      (mp.sqrt(1 + value(r)**2 * phi) if param == 'theta' else 1)
                      for r in loads]), phi


def write_table(path, rows, columns):
    if rows:
        write(path, rows)
    else:
        # A confirmation with no failed theta endpoints has no face witnesses.
        with path.open('w') as stream:
            csv.writer(stream,lineterminator='\n').writerow(columns.split(','))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    out = parser.parse_args().run_dir
    if (out / 'boundaries.csv').exists():
        raise ValueError('fresh boundary output required')
    mp.mp.dps = 90
    begin = time.monotonic()
    art = matrices(read(out / 'artifacts.csv'))
    sample = matrices(read(out / 'samples.csv'))
    fits = {int(r['fit_id']): r for r in read(out / 'fits.csv')}
    pts = {}
    for r in read(out / 'points.csv'):
        pts.setdefault(int(r['point_id']), []).append(r)
    terminals = [r for r in read(out / 'audits.csv') if r['target'] == 'terminal']
    failures = [r for r in terminals if r['parameterization'] == 'theta' and r['passed'].lower() != 'true']
    witnesses, paths, parameters, endpoints, seen = [], [], [], [], set()
    for row in failures:
        cid, pid = int(row['case_id']), int(row['point_id'])
        key = cid, row['method']
        c, phi = coordinates(pts[pid], art(pid, 'theta'), 'theta')
        if key in seen:
            endpoints.append(dict(fit_id=row['fit_id'],case_id=cid,method=row['method'],arm=row['arm'],
                phi=number(phi),largest_standardized_loading=number(max(abs(x) for x in c)),
                classification='compare retained local-face witness'))
            continue
        seen.add(key)
        F = art(pid, 'F_1'); W = F * F.T
        # Thresholds are unconstrained in this scope: eliminate them exactly.
        cross = W[:8, :8]**-1 * W[:8, 8:]
        Q = W[8:, 8:] - W[8:, :8] * cross
        target = sample(cid, 'moments_1'); corr = target[8:, :]
        active = max(range(4), key=lambda k: abs(c[k]))
        sign = mp.sign(c[active]); free = [k for k in range(4) if k != active]
        def unpack(v):
            z = mp.matrix(c); z[active] = sign
            for k, x in zip(free, v): z[k] = x
            return z
        def fun(*v):
            return tuple(profile(unpack(v), corr, Q)[1][k] for k in free)
        def jac(*v):
            H = profile(unpack(v), corr, Q)[2]
            return mp.matrix([[H[i,j] for j in free] for i in free])
        try:
            root = mp.findroot(fun, tuple(c[k] for k in free), J=jac, tol=mp.mpf('1e-75'), maxsteps=100)
            face = unpack(root); f, g, H, d = profile(face, corr, Q)
            eig = mp.eigsy(jac(*root), eigvals_only=True)[0]
            valid = all(abs(face[k]) < 1 for k in free) and sign*g[active] < 0 and eig > 0
        except (ValueError, ZeroDivisionError) as error:
            endpoints.append(dict(fit_id=row['fit_id'],case_id=cid,method=row['method'],arm=row['arm'],
                phi=number(phi),largest_standardized_loading=number(max(abs(x) for x in c)),
                classification=f'unresolved face refinement: {error}'))
            continue
        if not valid:
            endpoints.append(dict(fit_id=row['fit_id'],case_id=cid,method=row['method'],arm=row['arm'],
                phi=number(phi),largest_standardized_loading=number(max(abs(x) for x in c)),
                classification='unresolved single-face witness'))
            continue
        delta = next(r for r in terminals if int(r['case_id']) == cid and r['method'] == row['method'] and
                     r['parameterization'] == 'delta' and r['arm'] == 'default')
        dp = int(delta['point_id']); dc, _ = coordinates(pts[dp], art(dp, 'theta'), 'delta')
        delta_f, _, _, _ = profile(dc, corr, Q)
        witnesses.append(dict(case_id=cid,family=row['family'],method=row['method'],fit_id=row['fit_id'],
            active_indicator=active+1,boundary_objective=number(f),outward_derivative=number(sign*g[active]),
            tangent_gradient_max=number(max(abs(g[k]) for k in free)),tangent_hessian_min=number(eig),
            delta_max_abs_loading=number(max(abs(x) for x in dc)),delta_objective=number(delta_f),
            delta_objective_error=number(abs(delta_f-binary(delta['objective']))),
            finite_theta_transfer=bool(max(abs(x) for x in dc)<1),local_face_verified=bool(valid),
            scope='local positive-factor theta face; no global nonattainment claim'))
        endpoints.append(dict(fit_id=row['fit_id'],case_id=cid,method=row['method'],arm=row['arm'],
            phi=number(phi),largest_standardized_loading=number(max(abs(x) for x in c)),
            classification='strict local boundary descent; finite endpoint remains unaccepted'))
        previous = mp.inf
        # A development probe at 1e8 already hits the native conditioning
        # guard and loses distance precision. This declared native comparison
        # scope stops at 1e6; the local face itself is solved at 90 digits.
        for size in (100, 1000, 10000, 100000, 1000000):
            z = mp.matrix(face); z[active] = sign*mp.sqrt(mp.mpf(size)/(1+size))
            path_f, _, _, residual = profile(z, corr, Q)
            thresholds = target[:8,:] - cross*residual
            path_phi = z[0]**2/(1-z[0]**2)
            loads = [z[k]/mp.sqrt(path_phi*(1-z[k]**2)) for k in range(4)]
            variances = [1+loads[k]**2*path_phi for k in range(4)]
            theta = mp.matrix(art(pid,'theta'))
            ovs = [r['rhs'] for r in pts[pid] if r['op']=='=~']
            for r in pts[pid]:
                k = int(r['free'])
                if not k: continue
                if r['op']=='=~': theta[k-1]=loads[ovs.index(r['rhs'])]
                elif r['op']=='~~' and r['lhs']==r['rhs']=='f': theta[k-1]=path_phi
            for t, r in enumerate(r for r in pts[pid] if r['op']=='|'):
                theta[int(r['free'])-1]=thresholds[t]*mp.sqrt(variances[ovs.index(r['lhs'])])
            path_id = len(paths)+1
            paths.append(dict(path_id=path_id,fit_id=row['fit_id'],case_id=cid,method=row['method'],
                active_indicator=active+1,active_variance=number(1+mp.mpf(size)),factor_variance=number(path_phi),
                objective=number(path_f),boundary_gap=number(path_f-f),strict_descent=bool(path_f<previous),
                all_finite=bool(all(mp.isfinite(x) for x in theta))))
            for k,x in enumerate(theta):
                parameters.append(dict(path_id=path_id,fit_id=row['fit_id'],parameter=k+1,value=format(float(x),'.17g')))
            previous = path_f
    # Include passing controls in the face register through their actual delta
    # estimates, without labeling any unexamined failed fit as recovered.
    write_table(out/'boundaries.csv',witnesses,
        'case_id,family,method,fit_id,active_indicator,boundary_objective,outward_derivative,tangent_gradient_max,tangent_hessian_min,delta_max_abs_loading,delta_objective,delta_objective_error,finite_theta_transfer,local_face_verified,scope')
    write_table(out/'endpoint_classification.csv',endpoints,
        'fit_id,case_id,method,arm,phi,largest_standardized_loading,classification')
    if paths:
        write(out/'boundary_paths.csv',paths);write(out/'path_parameters.csv',parameters)
    gates=[dict(check='local_face_stationarity_and_outward_descent',tested=len(witnesses),failed=sum(not r['local_face_verified'] or binary(r['tangent_gradient_max'])>mp.mpf('1e-60') or binary(r['delta_objective_error'])>mp.mpf('1e-9') for r in witnesses)),
           dict(check='finite_strict_descent_path',tested=len(paths),failed=sum(not r['strict_descent'] or not r['all_finite'] or binary(r['boundary_gap'])<=0 for r in paths))]
    write(out/'boundary_verification.csv',gates)
    write(out/'boundary_metadata.csv',[dict(precision_digits=90,elapsed_s=time.monotonic()-begin,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        convention='c_j=lambda_j sqrt(phi)/sqrt(1+lambda_j^2 phi); correlations=c_i c_j; thresholds eliminated with actual W Schur complement',
        limitation='single local face in positive-factor chart; no global infimum, construction certificate, alternate basin or inference claim')])
    if any(r['failed'] for r in gates):raise SystemExit('boundary witness gate failed; evidence retained')
    print(f'Boundary reference: {len(witnesses)} faces, {len(paths)} finite points; {time.monotonic()-begin:.2f}s',flush=True)


if __name__=='__main__':main()
