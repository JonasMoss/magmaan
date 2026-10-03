#!/usr/bin/env python3
"""Independent 90-digit checks of retained-input and conditional intervals.

The dimensional construction arm is an inherited sensitivity assumption.
Coverage of the measured construction errors is checked separately from
interval coverage; neither establishes a universal construction-error bound.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path
import mpmath as mp
from uls_audit_reference import evaluate
from refine_open_cases import Model
from uls_unit_reference import read, write, number


def binary(x): return mp.mpf(float(x))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); out=args.run_dir
    if (out/'comparisons.csv').exists(): raise ValueError('fresh output required')
    mp.mp.dps=90; start=time.monotonic()
    covariance=read(out/'covariances.csv'); points=read(out/'points.csv'); intervals=read(out/'intervals.csv')
    samples={}
    for r in covariance:
        samples.setdefault(int(r['case_id']),mp.matrix(6))[int(r['row'])-1,int(r['col'])-1]=binary(r['value'])
    for S in samples.values():
        for j in range(6):
            for i in range(j): S[i,j]=S[j,i]
    matrices={}
    for r in read(out/'artifacts.csv'): matrices.setdefault((int(r['point_id']),r['name']),[]).append(r)
    def artifact(pid,name):
        rows=matrices[pid,name]
        a=mp.matrix(max(int(r['row']) for r in rows),max(int(r['col']) for r in rows))
        for r in rows: a[int(r['row'])-1,int(r['col'])-1]=binary(r['value'])
        return a
    canonical=[('X','=~','x2'),('X','=~','x3'),('Y','=~','y2'),('Y','=~','y3'),('X','~~','X'),('Y','~','X'),('Y','~~','Y')]
    canonical += [(name,'~~',name) for name in ('x1','x2','x3','y1','y2','y3')]
    comparisons=[]; cache={}
    for r in intervals:
        pid=int(r['point_id']); cid=int(r['case_id']); S=samples[cid]
        pp=[dict(p,est=number(binary(p['est']))) for p in points if int(p['point_id'])==pid]
        free=sorted((p for p in pp if int(p['free'])>0),key=lambda p:int(p['free']))
        P=mp.matrix(13)
        for j,p in enumerate(free): P[canonical.index((p['lhs'],p['op'],p['rhs'])),j]=1
        B=artifact(pid,'derivative_basis'); target_distance=mp.nan; retained_distance=mp.nan
        matrix_error=mp.inf; vector_error=mp.inf; target_positive=False
        if r['estimator']=='ULS':
            truth=evaluate(S,pp,retain=True); artifacts=truth['artifacts']
            target_distance=mp.mpf(truth['reference_distance']); target_positive=truth['reference_hessian_positive']
            A=artifact(pid,'equilibrated_factor'); b=artifact(pid,'metric_score_residual')
            D=mp.diag(list(artifact(pid,'factor_scale')))
            if cid not in cache:
                Q,_=mp.qr(A); Q=Q[:,:A.cols]
                cache[cid]=(A,Q,mp.norm(A-artifacts['factor']*P*B*D))
            cached_A,Q,matrix_error=cache[cid]
            assert A==cached_A, 'variance perturbation changed the factor'
            retained_distance=mp.norm(Q.T*b)
            vector_error=mp.norm(b-artifacts['score_residual'])
        else:
            values={(p['lhs'],p['op'],p['rhs']):mp.mpf(p['est']) for p in pp}
            get=lambda a,op,b:values[a,op,b]
            vx=get('X','~~','X'); beta=get('Y','~','X'); psi=get('Y','~~','Y')
            z=mp.matrix([get('X','=~','x2'),get('X','=~','x3'),get('Y','=~','y2'),get('Y','=~','y3'),vx,beta*vx,psi+beta**2*vx])
            z=mp.matrix(list(z)+[get(name,'~~',name) for name in ('x1','x2','x3','y1','y2','y3')])
            _,_,g,h=Model(S).evaluate(z,True)
            T=mp.eye(13); T[5,4]=beta; T[5,5]=vx; T[5,6]=0
            T[6,4]=beta**2; T[6,5]=2*beta*vx; T[6,6]=1
            H=T.T*h*T
            H[4,5]+=g[5]+2*beta*g[6]; H[5,4]=H[4,5]; H[5,5]+=2*vx*g[6]
            H=int(r['nobs'])*B.T*P.T*H*P*B
            G=int(r['nobs'])*B.T*P.T*T.T*g
            target_positive=mp.eigsy(H,eigvals_only=True)[0]>0
            if target_positive: target_distance=mp.sqrt((G.T*mp.lu_solve(H,G))[0])
            if (pid,'curvature_equilibrated_hessian') in matrices and (pid,'curvature_scale') in matrices:
                C=artifact(pid,'curvature_equilibrated_hessian'); D=mp.diag(list(artifact(pid,'curvature_scale')))
                # The kernel accounts for rounding in the double product D*g.
                score=D*artifact(pid,'gradient')
                if mp.eigsy(C,eigvals_only=True)[0]>0:
                    retained_distance=mp.sqrt((score.T*mp.lu_solve(C,score))[0])
                matrix_error=mp.norm(C-D*H*D); vector_error=mp.norm(D*(G-artifact(pid,'gradient')))
        retained_covers=not mp.isfinite(retained_distance) or binary(r['retained_lower'])<=retained_distance<=binary(r['retained_upper'])
        construction_comparable=mp.isfinite(matrix_error) and mp.isfinite(vector_error)
        construction_covers=construction_comparable and matrix_error<=binary(r['matrix_allowance']) and vector_error<=binary(r['vector_allowance'])
        conditional_covers=not mp.isfinite(target_distance) or binary(r['conditional_lower'])<=target_distance<=binary(r['conditional_upper'])
        def wrong(decision,d,positive):
            return decision=='within_budget' and (not positive or d>mp.mpf('.01')) or decision=='above_budget' and positive and d<=mp.mpf('.01')
        comparisons.append(dict(point_id=pid,case_id=cid,source_case=r['source_case'],role=r['role'],estimator=r['estimator'],target=r['target'],
            reference_distance=number(target_distance),reference_hessian_positive=bool(target_positive),
            retained_reference_distance=number(retained_distance),retained_interval_covers=bool(retained_covers),
            matrix_forward_error=number(matrix_error),vector_forward_error=number(vector_error),construction_comparable=bool(construction_comparable),construction_allowance_covers=bool(construction_covers),
            conditional_interval_covers=bool(conditional_covers),conditional_wrong_decision=bool(wrong(r['conditional_decision'],target_distance,target_positive)),
            retained_decision=r['retained_decision'],conditional_decision=r['conditional_decision'],actual_passed=r['actual_passed'],
            retained_upper=r['retained_upper'],conditional_upper=r['conditional_upper']))
        if pid%7==0: print(f'90-digit uncertainty case {cid}; {time.monotonic()-start:.1f}s',flush=True)
    write(out/'comparisons.csv',comparisons)
    write(out/'reference_metadata.csv',[dict(digits=90,mpmath=mp.__version__,python=sys.version,interpreter=sys.executable,elapsed_s=time.monotonic()-start,
        source_sha256=';'.join(hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ('audit_uncertainty_reference.py','uls_audit_reference.py','uls_unit_reference.py','refine_open_cases.py')),
        inputs_sha256=';'.join(hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ('covariances.csv','points.csv','artifacts.csv','intervals.csv')),
        convention='exact binary64 point/sample/artifacts; lower sample triangle mirrored; independent analytic derivatives; total objective scale',
        construction_scope='dimensional allowances are assumptions; coverage of measured construction errors and interval decisions reported separately')])


if __name__=='__main__': main()
