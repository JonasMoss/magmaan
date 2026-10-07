#!/usr/bin/env python3
"""Independent 90-digit original ordinal/mixed moment maps and curvature checks."""
import argparse
import hashlib
import time
from pathlib import Path
import mpmath as mp
from weighted_audit_reference import binary, number, matrices, moments
from verify_uls_guard import read, write


def add(a,b): return tuple(x+y for x,y in zip(a,b))
def scale(a,s): return tuple(x*s for x in a)
def mul(a,b):
    v,g,h=a; w,k,l=b
    return v*w, w*g+v*k, w*h+v*l+g*k.T+k*g.T

def invroot(a):
    v,g,h=a; f=1/mp.sqrt(v); d=-f/(2*v); dd=3*f/(4*v*v)
    return f,d*g,d*h+dd*g*g.T


def block_moments(pt,theta,g,sample,cid,param,mixed):
    rows=[r for r in pt if r['op'] not in ('|','~*~')]
    S,mu,J,second=moments(rows,theta,g,True)
    p,q=S.rows,len(theta); pos=[(i,j) for j in range(p) for i in range(j,p)]
    all_h=[mp.matrix(q) for _ in range(J.rows)]
    for a in range(q):
        for b in range(a+1):
            values=second(a,b)
            for k in range(J.rows): all_h[k][a,b]=all_h[k][b,a]=values[k]
    mean=[(mu[i],J[i,:].T,all_h[i]) for i in range(p)]
    sigma={ij:(S[ij],J[p+k,:].T,all_h[p+k]) for k,ij in enumerate(pos)}
    ov=list(dict.fromkeys(r['rhs'] for r in pt if r['op']=='=~'))
    ordered=[int(x) for x in sample(cid,f'ordered_{g}')]
    # Released DELTA response scales are the prepared free response variances.
    released=[False]*p
    for r in pt:
        if int(r['group'])==g and r['op']=='~~' and r['lhs']==r['rhs'] and r['lhs'] in ov and int(r['free'])>0: released[ov.index(r['lhs'])]=True
    std=[param=='theta' and bool(x) for x in ordered] if mixed else [param=='theta' or x for x in released]
    threshold_ov=[int(x)-1 for x in sample(cid,f'threshold_ov_{g}')]
    levels=[int(x) for x in sample(cid,f'threshold_level_{g}')]
    result=[]
    for i,level in zip(threshold_ov,levels):
        row=next(r for r in pt if int(r['group'])==g and r['op']=='|' and r['lhs']==ov[i] and r['rhs']==f't{level}')
        free=int(row['free']); v=theta[free-1] if free else binary(row['est']); d=mp.matrix(q,1)
        if free:d[free-1]=1
        jet=(v,d,mp.matrix(q))
        if not mixed:jet=add(jet,scale(mean[i],-1))
        if std[i]:jet=mul(jet,invroot(sigma[i,i]))
        result.append(jet)
    if mixed:
        result += [scale(mean[i],-1) for i in range(p) if not ordered[i]]
        result += [sigma[i,i] for i in range(p) if not ordered[i]]
    for j in range(p):
        for i in range(j+1,p):
            jet=sigma[i,j]
            for k in (i,j):
                if std[k]:jet=mul(jet,invroot(sigma[k,k]))
            result.append(jet)
    target=sample(cid,f'moments_{g}')
    return mp.matrix([r[0] for r in result])-target,mp.matrix([list(r[1]) for r in result]),[r[2] for r in result]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    out=parser.parse_args().run_dir
    if (out/'comparisons.csv').exists():raise ValueError('fresh reference output required')
    mp.mp.dps=90;begin=time.monotonic()
    samples=matrices(read(out/'samples.csv'));artifact=matrices(read(out/'artifacts.csv'))
    cases={int(r['case_id']):r for r in read(out/'cases.csv')};pts={}
    for r in read(out/'points.csv'):pts.setdefault(int(r['point_id']),[]).append(r)
    comparisons=[];producer_checks=[];cached=set()
    for row in read(out/'audits.csv'):
        pid,cid=int(row['point_id']),int(row['case_id']);get=lambda name:artifact(pid,name)
        theta=get('theta');q=len(theta);G=mp.matrix(q,1);H=mp.matrix(q);M=mp.matrix(q);C=mp.matrix(q);obj=mp.mpf(0)
        factors=[];scores=[];n_total=0;g=1
        while samples(cid,f'R_{g}',True) is not None:
            n=int(samples(cid,f'n_{g}')[0]);n_total+=n
            mixed=cases[cid]['mixed'].lower()=='true'
            d,J,seconds=block_moments(pts[pid],theta,g,samples,cid,row['parameterization'],mixed)
            F=get(f'F_{g}');W=F*F.T;wr=W*d;G+=n*J.T*wr;obj+=n*(d.T*wr)[0]/2
            M+=n*J.T*W*J
            for k,second in enumerate(seconds):C+=n*wr[k]*second
            factors.append(mp.sqrt(n)*F.T*J);scores.append(mp.sqrt(n)*F.T*d)
            key=cid,g,row['method']
            if key not in cached:
                Gamma=samples(cid,f'NACOV_{g}')
                expected=mp.eye(Gamma.rows) if row['method']=='ULS' else mp.diag([1/Gamma[k,k] for k in range(Gamma.rows)]) if row['method']=='DWLS' else Gamma**-1
                units=mp.diag([mp.sqrt(Gamma[k,k]) for k in range(Gamma.rows)])
                error=mp.norm(units*(W-expected)*units)/max(1,mp.norm(units*expected*units))
                X=samples(cid,f'X_{g}');observed=samples(cid,f'thresholds_{g}')
                ovs=[int(x)-1 for x in samples(cid,f'threshold_ov_{g}')];levels=[int(x) for x in samples(cid,f'threshold_level_{g}')]
                threshold_error=mp.mpf(0)
                for k,(ov,level) in enumerate(zip(ovs,levels)):
                    prob=mp.mpf(sum(X[i,ov]<=level for i in range(X.rows)))/X.rows
                    expected_threshold=mp.sqrt(2)*mp.erfinv(2*prob-1)
                    threshold_error=max(threshold_error,abs(expected_threshold-observed[k]))
                producer_checks.append(dict(case_id=cid,family=row['family'],method=row['method'],block=g,
                    retained_weight_error=number(error),empirical_threshold_error=number(threshold_error),
                    passed=bool(error<mp.mpf('1e-9') and threshold_error<mp.mpf('1e-8'))))
                cached.add(key)
            g+=1
        H=M+C;A=mp.matrix(sum(a.rows for a in factors),q);b=mp.matrix(A.rows,1);off=0
        for a,s in zip(factors,scores):A[off:off+a.rows,:]=a;b[off:off+a.rows,:]=s;off+=a.rows
        errors={name:mp.norm(value-get(name))/max(1,mp.norm(value)) for name,value in
                [('gradient',G),('hessian',H),('metric',M),('curvature_correction',C),('metric_factor',A),('metric_score_residual',b)]}
        B=get('derivative_basis');h=B.T*H*B;metric=B.T*M*B;gradient=B.T*G
        positive=mp.eigsy(h,eigvals_only=True)[0]>0
        distance=mp.sqrt((gradient.T*mp.lu_solve(metric,gradient))[0]) if positive else mp.nan
        native=mp.nan if row['distance']=='NA' else binary(row['distance']);distance_error=abs(native-distance)/max(1,abs(distance)) if positive and mp.isfinite(native) else mp.nan
        objective_error=abs(obj/n_total-binary(row['objective']))
        wrong_pass=row['passed'].lower()=='true' and (not positive or distance>binary(.01))
        comparisons.append(dict(point_id=pid,fit_id=row['fit_id'],case_id=cid,family=row['family'],parameterization=row['parameterization'],method=row['method'],arm=row['arm'],target=row['target'],
            **{name+'_relative_error':number(e) for name,e in errors.items()},objective_error=number(objective_error),
            reference_distance=number(distance),distance_error=number(distance_error),reference_positive=bool(positive),
            status=row['status'],passed=row['passed'],construction_status=row['construction_status'],wrong_pass=bool(wrong_pass),
            agrees=bool(max(errors.values())<mp.mpf('1e-9') and objective_error<mp.mpf('1e-10') and not wrong_pass and (not mp.isfinite(distance_error) or distance_error<mp.mpf('1e-8')))))
        if pid%8==0:print(f'Ordinal reference {pid}; {time.monotonic()-begin:.1f}s',flush=True)
    write(out/'comparisons.csv',comparisons);write(out/'producer_checks.csv',producer_checks)
    gates=[dict(check='independent_moment_point_agreement',tested=len(comparisons),failed=sum(not r['agrees'] for r in comparisons)),
           dict(check='retained_weight_and_univariate_threshold',tested=len(producer_checks),failed=sum(not r['passed'] for r in producer_checks)),
           dict(check='unsupported_construction_stays_explicit',tested=len(comparisons),failed=sum(r['construction_status']!='unsupported' for r in comparisons))]
    controls=read(out/'controls.csv')
    gates.append(dict(check='invalid_first_stage_rejected',tested=len(controls),failed=sum(r['rejected'].lower()!='true' for r in controls)))
    saddle=[r for r in comparisons if r['target']=='saddle']
    gates.append(dict(check='saddle_never_passes',tested=len(saddle),failed=sum(r['reference_positive'] or r['passed'].lower()=='true' for r in saddle)))
    fits=read(out/'fits.csv');terminals=[r for r in comparisons if r['target']=='terminal']
    gates.append(dict(check='every_returned_fit_audited',tested=len(fits),failed=abs(sum(r['returned'].lower()=='true' for r in fits)-len(terminals))))
    write(out/'verification.csv',gates)
    write(out/'reference_metadata.csv',[dict(precision_digits=90,mpmath_version=mp.__version__,elapsed_s=time.monotonic()-begin,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        convention='exact binary64 prepared partable, raw theta, moments and retained whitening factors; analytic independent LISREL and second-order scalar chain',
        limitations='univariate thresholds and conditional Gamma-to-weight checked; polychoric/polyserial/Gamma construction not independently certified')])
    if any(r['failed'] for r in gates):raise SystemExit('independent ordinal gate failed; evidence retained')

if __name__=='__main__':main()
