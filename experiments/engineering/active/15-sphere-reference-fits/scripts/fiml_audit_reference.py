#!/usr/bin/env python3
"""Independent 90-digit direct observed-pattern negative log likelihood."""
import argparse
import hashlib
import time
from pathlib import Path
import mpmath as mp
from weighted_audit_reference import binary, number, matrices, moments
from verify_uls_guard import read, write


def trace(x):return sum(x[i,i] for i in range(x.rows))
def submatrix(x,obs):return mp.matrix([[x[i,j] for j in obs] for i in obs])


def raw_patterns(samples,cid):
    result=[];block=1
    while samples(cid,f'X_{block}',True) is not None:
        X=samples(cid,f'X_{block}');mask=samples(cid,f'mask_{block}');groups={}
        for i in range(X.rows):
            obs=tuple(j for j in range(X.cols) if mask[i,j])
            if not obs:raise ValueError('no observed values')
            groups.setdefault(obs,[]).append(i)
        for obs,rows in groups.items():
            mean=mp.matrix([sum(X[i,j] for i in rows)/len(rows) for j in obs])
            cov=mp.matrix([[sum((X[i,j]-mean[a])*(X[i,k]-mean[b]) for i in rows)/len(rows)
                           for b,k in enumerate(obs)] for a,j in enumerate(obs)])
            result.append((block,obs,len(rows),mean,cov))
        block+=1
    return result


def likelihood(pt,theta,patterns):
    q=len(theta);G=mp.matrix(q,1);H=mp.matrix(q);value=mp.mpf(0);domain=True
    for block in sorted(set(p[0] for p in patterns)):
        S,mu,J,second=moments(pt,theta,block,True);p=S.rows
        pos=[(i,j) for j in range(p) for i in range(j,p)]
        ds=[];dm=[]
        # At a zero-loading saddle the first derivatives vanish, but the
        # second derivatives of Sigma do not. Use structural free rows rather
        # than a nonzero-Jacobian filter to retain those curvature terms.
        active=sorted({int(r['free'])-1 for r in pt if int(r['group'])==block and int(r['free'])>0})
        for k in range(q):
            d=mp.matrix(p)
            for t,(i,j) in enumerate(pos):d[i,j]=d[j,i]=J[p+t,k]
            ds.append(d);dm.append(J[:p,k])
        seconds={}
        for i in active:
            for j in active:
                if j>i:continue
                d=second(i,j);s=mp.matrix(p)
                for t,(a,b) in enumerate(pos):s[a,b]=s[b,a]=d[p+t]
                seconds[i,j]=(s,d[:p,:])
        for g,obs,n,mean,C in patterns:
            if g!=block:continue
            sigma=submatrix(S,obs);m=mp.matrix([mu[i] for i in obs])-mean
            if mp.eigsy(sigma,eigvals_only=True)[0]<=0:
                return mp.inf,G,H,False
            A=sigma**-1;Q=C+m*m.T;V=(A-A*Q*A)/2
            value+=n*(mp.log(mp.det(sigma))+trace(A*C)+(m.T*A*m)[0])/2
            dS={k:submatrix(ds[k],obs) for k in active}
            dM={k:mp.matrix([dm[k][i] for i in obs]) for k in active}
            dA={k:-A*dS[k]*A for k in active}
            dV={k:(dA[k]-dA[k]*Q*A-A*(dM[k]*m.T+m*dM[k].T)*A-A*Q*dA[k])/2 for k in active}
            for i in active:
                G[i]+=n*(trace(V*dS[i])+(dM[i].T*A*m)[0])
                for j in active:
                    if j>i:continue
                    ss,mm=seconds[i,j];ss=submatrix(ss,obs);mm=mp.matrix([mm[k] for k in obs])
                    h=n*(trace(dV[j]*dS[i]+V*ss)+(mm.T*A*m)[0]+
                         (dM[i].T*dA[j]*m)[0]+(dM[i].T*A*dM[j])[0])
                    H[i,j]+=h
                    if i!=j:H[j,i]+=h
    return value,G,H,domain


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    out=parser.parse_args().run_dir
    if(out/'comparisons.csv').exists():raise ValueError('fresh reference output required')
    mp.mp.dps=90;begin=time.monotonic();art=matrices(read(out/'artifacts.csv'));sample=matrices(read(out/'samples.csv'))
    pts={};stored={}
    for r in read(out/'points.csv'):pts.setdefault(int(r['point_id']),[]).append(r)
    for r in read(out/'pattern_rows.csv'):stored.setdefault(int(r['point_id']),[]).append(r)
    cached={};comparisons=[];pattern_checks=[];offsets={};ridge_checks=[]
    for row in read(out/'audits.csv'):
        pid,cid=int(row['point_id']),int(row['case_id']);theta=art(pid,'theta')
        if cid not in cached:cached[cid]=raw_patterns(sample,cid)
        patterns=cached[cid];n_total=sum(p[2] for p in patterns)
        get=lambda name:art(pid,name)
        # Compare the actual core cache with independently selected raw rows.
        max_mean=max_cov=mp.mpf(0);bad_count=0
        for g,obs,n,mean,C in patterns:
            match=[r for r in stored[pid] if int(r['block'])==g and tuple(int(x)-1 for x in r['observed'].split(':'))==obs]
            if len(match)!=1:bad_count+=1;continue
            r=match[0];k=r['pattern_id'];bad_count+=int(int(r['n_obs'])!=n)
            max_mean=max(max_mean,mp.norm(mean-get(f'pattern_{k}_mean'))/max(1,mp.norm(mean)))
            max_cov=max(max_cov,mp.norm(C-get(f'pattern_{k}_cov'))/max(1,mp.norm(C)))
        counts_ok=bad_count==0 and len(stored[pid])==len(patterns) and int(row['n_obs'])==n_total
        pattern_checks.append(dict(point_id=pid,case_id=cid,patterns=len(patterns),n_total=n_total,
            count_selection_ok=counts_ok,mean_error=number(max_mean),cov_error=number(max_cov),
            passed=bool(counts_ok and max_mean<mp.mpf('1e-12') and max_cov<mp.mpf('1e-12'))))
        value,G,H,domain=likelihood(pts[pid],theta,patterns)
        identification='excluded_known_scale_ridge' if row['target']=='rank_control' else 'declared_identified_model'
        if row['target']=='rank_control':
            altered=mp.matrix(theta);direction=mp.matrix(len(theta),1)
            for r in pts[pid]:
                k=int(r['free'])-1
                if k<0:continue
                if r['op']=='=~':altered[k]*=mp.mpf('1.25');direction[k]=theta[k]
                if r['op']=='~~' and r['lhs']==r['rhs']=='f':altered[k]/=mp.mpf('1.25')**2;direction[k]=-2*theta[k]
            S,mu,J,_=moments(pts[pid],theta,1,True);other,mean,_,_=moments(pts[pid],altered,1,True)
            invariance=mp.norm(S-other)+mp.norm(mu-mean);null=mp.norm(J*direction)
            ridge_checks.append(dict(point_id=pid,scale_invariance_error=number(invariance),jacobian_null_error=number(null),
                raw_newton_passed=row['passed'],bank_decision='unchecked_identification',
                passed=bool(invariance<mp.mpf('1e-60') and null<mp.mpf('1e-60'))))
        if not domain:
            comparisons.append(dict(point_id=pid,fit_id=row['fit_id'],case_id=cid,family=row['family'],arm=row['arm'],target=row['target'],
                objective_error='nan',gradient_error='nan',hessian_error='nan',reference_positive=False,reference_distance='nan',
                distance_error='nan',status=row['status'],passed=row['passed'],domain=False,agrees=row['passed'].lower()!='true',
                construction_status=row['construction_status'],identification=identification,bank_decision='rejected'))
            continue
        error_g=mp.norm(G-get('gradient'))/max(1,mp.norm(G));error_h=mp.norm(H-get('hessian'))/max(1,mp.norm(H))
        error_f=abs(value/n_total-binary(row['objective']))/max(1,abs(value/n_total))
        B=get('derivative_basis');h=B.T*H*B;g=B.T*G
        eigen=mp.eigsy(h,eigvals_only=True)
        # A nearly null reference direction supplies no interior pass.
        positive=eigen[0]>mp.mpf('1e-60')*max(1,max(abs(x) for x in eigen))
        distance=mp.sqrt((g.T*mp.lu_solve(h,g))[0]) if positive else mp.nan
        native=mp.nan if row['distance']=='NA' else binary(row['distance'])
        error_d=abs(native-distance)/max(1,abs(distance)) if positive and mp.isfinite(native) else mp.nan
        wrong_pass=row['passed'].lower()=='true' and (not positive or distance>binary(.01))
        comparisons.append(dict(point_id=pid,fit_id=row['fit_id'],case_id=cid,family=row['family'],arm=row['arm'],target=row['target'],
            objective_error=number(error_f),gradient_error=number(error_g),hessian_error=number(error_h),
            reference_positive=bool(positive),reference_distance=number(distance),distance_error=number(error_d),
            status=row['status'],passed=row['passed'],domain=True,
            agrees=bool(max(error_f,error_g,error_h)<mp.mpf('1e-9') and not wrong_pass and
                        (not mp.isfinite(error_d) or error_d<mp.mpf('1e-8'))),construction_status=row['construction_status']))
        comparisons[-1]['identification']=identification
        comparisons[-1]['bank_decision']='unchecked_identification' if row['target']=='rank_control' else ('conditional_local_pass' if row['passed'].lower()=='true' else 'rejected')
        if row['family']=='complete':
            offsets[pid]=sum(n*(mp.log(mp.det(C))+len(obs))/2 for _,obs,n,_,C in patterns)/n_total
        if pid%4==0:print(f'FIML reference {pid}; {time.monotonic()-begin:.1f}s',flush=True)
    reductions=[]
    for r in read(out/'reductions.csv'):
        pid=int(r['point_id']);native=next(x for x in read(out/'audits.csv') if int(x['point_id'])==pid)
        error=abs(binary(native['objective'])-binary(r['ml_objective'])-offsets[pid])
        reductions.append(dict(point_id=pid,objective_constant_error=number(error),gradient_error=r['gradient_error'],hessian_error=r['hessian_error'],
            passed=bool(error<mp.mpf('1e-10') and binary(r['gradient_error'])<mp.mpf('1e-9') and binary(r['hessian_error'])<mp.mpf('1e-8'))))
    write(out/'comparisons.csv',comparisons);write(out/'pattern_checks.csv',pattern_checks);write(out/'complete_reduction.csv',reductions);write(out/'scale_ridge_checks.csv',ridge_checks)
    fits=read(out/'fits.csv');controls=read(out/'controls.csv')
    gates=[dict(check='independent_pattern_likelihood_points',tested=len(comparisons),failed=sum(not x['agrees'] for x in comparisons)),
           dict(check='actual_pattern_selection_counts_and_summaries',tested=len(pattern_checks),failed=sum(not x['passed'] for x in pattern_checks)),
           dict(check='complete_data_ML_reduction',tested=len(reductions),failed=sum(not x['passed'] for x in reductions)),
           dict(check='invalid_raw_inputs_reject',tested=len(controls),failed=sum(x['rejected'].lower()!='true' for x in controls)),
           dict(check='invalid_covariance_and_saddle_never_pass',tested=sum(x['target'] in ('invalid_covariance','saddle') for x in comparisons),
                failed=sum(x['passed'].lower()=='true' for x in comparisons if x['target'] in ('invalid_covariance','saddle'))),
           dict(check='known_scale_ridge_explicitly_unchecked',tested=len(ridge_checks),failed=sum(not x['passed'] or x['bank_decision']!='unchecked_identification' for x in ridge_checks)),
           dict(check='every_returned_fit_audited',tested=len(fits),failed=abs(sum(x['returned'].lower()=='true' for x in fits)-sum(x['target']=='terminal' for x in comparisons))),
           dict(check='unsupported_construction_explicit',tested=len(comparisons),failed=sum(x['construction_status']!='unsupported' for x in comparisons))]
    write(out/'verification.csv',gates)
    write(out/'reference_metadata.csv',[dict(precision_digits=90,mpmath_version=mp.__version__,elapsed_s=time.monotonic()-begin,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        convention='raw observed-row grouping and exact binary64 data/parameters; analytic independent LISREL first/second derivatives and observed-pattern Gaussian likelihood',
        limitations='conditional numerical point evidence; no propagated construction interval, inference calibration or global optimality')])
    if any(x['failed'] for x in gates):raise SystemExit('independent FIML gate failed; evidence retained')


if __name__=='__main__':main()
