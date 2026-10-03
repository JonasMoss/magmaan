#!/usr/bin/env python3
"""Conditional projection-distance uncertainty and curvature calibration.

This is a development experiment, not an acceptance override. For equally
ranked A and A+E, ||P(A+E)-P(A)|| <= ||E||/(sigma_min(A)-||E||): apply
(I-P(A)) to (A+E)(A+E)^+ and use Weyl's singular-value bound. Consequently
| ||P(A+E)(b+e)|| - ||P(A)b|| | <= ||b||*that_bound + ||e||.
The Frobenius norm bounds the spectral norm. Model allowances below are
assumptions, not proved forward-error bounds for construction of A and b.
"""
import argparse
import hashlib
from pathlib import Path
import time
import sys
import mpmath as mp
from uls_audit_reference import evaluate
from uls_unit_reference import read, write, number


def binary(value):
    # Decimal CSV transport carries 17 significant digits; reconstruct the
    # actual binary64 input before the independent 90-digit calculation.
    return mp.mpf(float(value))


def projection(A, b):
    Q, R = mp.qr(A)
    thin = Q[:, :A.cols]
    return mp.norm(thin.T*b), thin, R[:A.cols, :]


def interval(distance, bnorm, minimum, delta_a, delta_b, arithmetic=0):
    if delta_a >= minimum:
        return mp.inf
    return bnorm*delta_a/(minimum-delta_a)+delta_b+arithmetic


def classify(distance, error, budget=None):
    if budget is None: budget=mp.mpf('.01')
    if not mp.isfinite(error): return 'unresolved_rank'
    if distance+error <= budget: return 'within_budget'
    if distance-error > budget: return 'above_budget'
    return 'unresolved_budget'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); out=args.run_dir
    if (out/'comparisons.csv').exists(): raise ValueError('fresh output required')
    mp.mp.dps=90; start=time.monotonic(); unit=mp.mpf(2)**-52
    points=read(out/'points.csv'); audits=read(out/'audits.csv'); covariance=read(out/'covariances.csv')
    samples={}
    for r in covariance:
        cid=int(r['case_id']); samples.setdefault(cid,mp.matrix(6))
        samples[cid][int(r['row'])-1,int(r['col'])-1]=binary(r['value'])
    # Frozen floating-point scaling can leave tiny upper/lower differences.
    # The objective packs the lower triangle and the factor uses lower LLT.
    for sample in samples.values():
        for j in range(6):
            for i in range(j+1,6): sample[j,i]=sample[i,j]
    matrices={}
    for r in read(out/'artifacts.csv'):
        key=(int(r['point_id']),r['name']); matrices.setdefault(key,[]).append(r)
    def artifact(pid,name):
        rows=matrices[pid,name]; a=mp.matrix(max(int(r['row']) for r in rows),max(int(r['col']) for r in rows))
        for r in rows: a[int(r['row'])-1,int(r['col'])-1]=binary(r['value'])
        return a
    comparisons=[]; intervals=[]; cache={}
    for audit in audits:
        pid=int(audit['point_id']); cid=int(audit['case_id'])
        pp=[dict(r,est=number(binary(r['est']))) for r in points if int(r['point_id'])==pid]
        exact=evaluate(samples[cid],pp,retain=True); truth=exact.pop('artifacts')
        assert mp.norm(truth['factor'].T*truth['factor']-truth['metric'])/mp.norm(truth['metric']) < mp.mpf('1e-70')
        assert mp.norm(truth['factor'].T*truth['score_residual']-truth['gradient'])/max(mp.norm(truth['gradient']),1) < mp.mpf('1e-65')
        canonical=[('X','=~','x2'),('X','=~','x3'),('Y','=~','y2'),('Y','=~','y3'),
            ('X','~~','X'),('Y','~','X'),('Y','~~','Y')]
        canonical += [(name,'~~',name) for name in ('x1','x2','x3','y1','y2','y3')]
        free=sorted((r for r in pp if int(r['free'])>0),key=lambda r:int(r['free']))
        order=[canonical.index((r['lhs'],r['op'],r['rhs'])) for r in free]
        P=mp.matrix(13)
        for j,i in enumerate(order): P[i,j]=1
        truth['factor']=truth['factor']*P
        truth['objective_jacobian']=truth['objective_jacobian']*P
        truth['hessian']=P.T*truth['hessian']*P
        truth['correction']=P.T*truth['correction']*P
        truth['gradient']=P.T*truth['gradient']
        A=artifact(pid,'metric_factor'); b=artifact(pid,'metric_score_residual'); H=artifact(pid,'hessian')
        if cid not in cache:
            D=mp.diag([1/mp.norm(A[:,j]) for j in range(A.cols)])
            scaled=A*D; singular=mp.svd(scaled,compute_uv=False)
            _, Q, _=projection(scaled,b)
            factor_error=mp.norm((A-truth['factor'])*D)
            # QR of the objective Jacobian defines an invertible parameter
            # change. Retain the observed correction; no Gauss-Newton swap.
            _, Rj=mp.qr(truth['objective_jacobian']); Rj=Rj[:A.cols,:]; Rinv=Rj**-1
            cache[cid]=(D,scaled,singular[singular.rows-1,0],Q,factor_error,Rinv,A,truth['factor'])
        D,scaled,smin,Q,factor_error,Rinv,cached_a,cached_truth=cache[cid]
        assert A==cached_a and truth['factor']==cached_truth, 'diagonal perturbation unexpectedly changed Jacobian'
        rounded_distance=mp.norm(Q.T*b); reported=binary(audit['distance'])
        reference_distance=mp.mpf(exact['reference_distance'])
        score_error=mp.norm(b-truth['score_residual'])
        arithmetic_error=abs(reported-rounded_distance)
        measured_error=interval(reported,mp.norm(b),smin,factor_error,score_error,arithmetic_error)
        factor_relative=factor_error/mp.norm(scaled)
        Dh=mp.diag([1/mp.sqrt(H[j,j]) for j in range(H.cols)])
        C=Dh*H*Dh; eig=mp.eigsy(C,eigvals_only=True)
        h_error=mp.norm(Dh*(H-truth['hessian'])*Dh)
        # This is computed at 90 digits, and only nominates a stable owning
        # implementation. Applying this transform to a rounded H cannot undo
        # cancellation already introduced by forming J'J.
        preconditioned=mp.eye(H.cols)+Rinv.T*truth['correction']*Rinv
        pre_eig=mp.eigsy(preconditioned,eigvals_only=True)
        extra={}
        if (pid,'curvature_coordinate_map') in matrices:
            T=artifact(pid,'curvature_coordinate_map'); actual=artifact(pid,'curvature_equilibrated_hessian')
            transformed=T.T*truth['hessian']*T
            et=mp.eigsy(transformed,eigvals_only=True)
            exact_step=-mp.lu_solve(truth['hessian'],truth['gradient'])
            step=artifact(pid,'newton_step'); error=step-exact_step
            cpp_residual=artifact(pid,'whitened_residual')
            # Judge derivative assembly at the actual recorded residual as
            # well as forward error at the mathematical model point. Scale
            # assembly error by absolute summands, allowing cancellation.
            supplied=evaluate(samples[cid],pp,retain=True,residual_override=cpp_residual)['artifacts']
            assembly=P.T*supplied['correction']*P
            terms=P.T*supplied['correction_absolute_terms']*P
            extra=dict(curvature_condition=audit['curvature_condition'],
                curvature_reference_condition=number(et[et.rows-1]/et[0]),
                curvature_relative_error=number(mp.norm(actual-transformed)/mp.norm(transformed)),
                curvature_error_bound=number(mp.norm(actual-transformed)),
                curvature_error_certified=bool(mp.norm(actual-transformed)<et[0]),
                curvature_min_eigenvalue=number(mp.eigsy(actual,eigvals_only=True)[0]),
                curvature_reference_min_eigenvalue=number(et[0]),
                newton_step_curvature_error=number(mp.sqrt((error.T*truth['hessian']*error)[0])),
                jacobian_relative_error=number(mp.norm(10*artifact(pid,'whitened_jacobian')-truth['objective_jacobian'])/mp.norm(truth['objective_jacobian'])),
                residual_construction_error=number(mp.norm(cpp_residual-truth['moment_residual'])),
                correction_assembly_error=number(mp.norm(artifact(pid,'ls_curvature_correction')-assembly)/max(mp.norm(terms),1)),
                correction_relative_error=number(mp.norm(artifact(pid,'ls_curvature_correction')-truth['correction'])/max(mp.norm(truth['correction']),1)))
        comparisons.append(dict(point_id=pid,case_id=cid,target_distance=audit['target_distance'],
            passed=audit['passed'],distance=number(reported),**exact,
            absolute_distance_error=number(abs(reported-reference_distance)),
            factor_relative_forward_error=number(factor_relative),score_absolute_forward_error=number(score_error),
            projection_arithmetic_error=number(arithmetic_error),measured_error_bound=number(measured_error),
            measured_interval_covers=bool(abs(reported-reference_distance)<=measured_error),
            measured_decision=classify(reported,measured_error),factor_min_singular=number(smin),residual_norm=number(mp.norm(b)),
            hessian_min_eigenvalue=number(eig[0]),hessian_forward_error_bound=number(h_error),
            measured_curvature_certified=bool(eig[0]>h_error),
            qr_curvature_min_eigenvalue=number(pre_eig[0]),qr_curvature_condition=number(pre_eig[pre_eig.rows-1,0]/pre_eig[0]),**extra))
        # A sensitivity sweep, not a fitted choice of a new production cap.
        # Include QR reconstruction error as well as assumed construction error.
        for multiplier in (8,64,8*A.rows*A.cols):
            allowance=mp.mpf(multiplier)*unit
            delta_a=(allowance+binary(audit['factor_residual']))*mp.norm(scaled)
            delta_b=allowance*max(mp.norm(b),1)
            error=interval(reported,mp.norm(b),smin,delta_a,delta_b)
            intervals.append(dict(point_id=pid,case_id=cid,target_distance=audit['target_distance'],
                multiplier=multiplier,relative_allowance=number(allowance),error_bound=number(error),
                decision=classify(reported,error),
                input_allowance_covers=bool(delta_a>=factor_error and delta_b>=score_error),
                interval_covers=bool(abs(reported-reference_distance)<=error),
                curvature_resolved=bool(eig[0]>allowance*mp.norm(C))))
        if pid%7==0: print(f'90-digit guard case {cid}; {time.monotonic()-start:.1f}s',flush=True)
    write(out/'comparisons.csv',comparisons); write(out/'intervals.csv',intervals)
    # Exact rank-loss witness: the denominator vanishes, rather than being
    # ridged or rank truncated. This checks the interval's logic only; native
    # unidentified-model rejection remains covered by owning C++ tests.
    J=mp.diag([1,mp.mpf('1e-10')]); correction=mp.diag([0,mp.mpf('-2e-20')])
    Rinv=J**-1; observed=mp.eye(2)+Rinv.T*correction*Rinv
    write(out/'negative_controls.csv',[
        dict(control='zero minimum singular value',result=classify(0,interval(0,1,mp.mpf(0),unit,unit))),
        dict(control='observed saddle with positive Gauss-Newton term',
            result='nonpositive_curvature' if mp.eigsy(observed,eigvals_only=True)[0]<0 else 'incorrect_pass')])
    write(out/'reference_metadata.csv',[dict(digits=90,mpmath=mp.__version__,python=sys.version,interpreter=sys.executable,elapsed_s=time.monotonic()-start,
        source_sha256=';'.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path(__file__).with_name('uls_audit_reference.py'),Path(__file__).with_name('uls_unit_reference.py'))),
        inputs_sha256=';'.join(hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('points.csv','covariances.csv','artifacts.csv','audits.csv')),
        input_convention='17-digit CSV reconstructed to exact binary64; sample lower triangle mirrored as in objective and factor LLT; 90-digit derivatives',
        bounds='conditional subspace perturbation; Frobenius input-error bound; measured or assumed allowances',
        curvature='observed correction and step in computed QR coordinates; assembly checked at recorded binary residual and forward error at exact model point; no ridge',
        scope='development calibration, no optimizer, no production acceptance change')])


if __name__=='__main__': main()
