#!/usr/bin/env python3
"""Independent raw-pattern Stage-1 and actual-input Stage-2 checks at 90 digits."""
import argparse
import hashlib
import time
from pathlib import Path
import mpmath as mp
from weighted_audit_reference import binary, number, matrices, moments, retained_weight
from fiml_audit_reference import raw_patterns, likelihood, trace, submatrix
from verify_uls_guard import read, write


def relative(a,b):return mp.norm(a-b)/max(1,mp.norm(a))
def equal(a,b):return a.rows==b.rows and a.cols==b.cols and list(a)==list(b)
def gamma_nt(S):
    p=S.rows;pos=[(i,j) for j in range(p) for i in range(j,p)];q=p+len(pos);G=mp.matrix(q)
    G[:p,:p]=S
    for i,(a,b) in enumerate(pos):
        for j,(c,d) in enumerate(pos):G[p+i,p+j]=S[a,c]*S[b,d]+S[a,d]*S[b,c]
    return G


def saturated(patterns,source,samples,cid):
    blocks=source;Q=sum(mu.rows+S.rows*(S.rows+1)//2 for mu,S in blocks)
    G=mp.matrix(Q,1);H=mp.matrix(Q);J=mp.matrix(Q);value=mp.mpf(0);offset=0;cov_indices=[];mean_indices=[]
    for block,(mu,S) in enumerate(blocks,1):
        p=S.rows;pos=[(i,j) for j in range(p) for i in range(j,p)];q=p+len(pos)
        mean_indices+=list(range(offset,offset+p));cov_indices+=list(range(offset+p,offset+q))
        ds=[mp.matrix(p) for _ in range(q)];dm=[mp.matrix(p,1) for _ in range(q)]
        for k in range(p):dm[k][k]=1
        for k,(i,j) in enumerate(pos):ds[p+k][i,j]=ds[p+k][j,i]=1
        X=samples(cid,f'X_{block}');mask=samples(cid,f'mask_{block}')
        for g,obs,n,mean,C in patterns:
            if g!=block:continue
            s=submatrix(S,obs);A=s**-1;m=mp.matrix([mu[j] for j in obs])-mean;T=C+m*m.T;V=(A-A*T*A)/2
            dS=[submatrix(x,obs) for x in ds];dM=[mp.matrix([x[j] for j in obs]) for x in dm]
            dA=[-A*x*A for x in dS]
            dV=[(dA[k]-dA[k]*T*A-A*(dM[k]*m.T+m*dM[k].T)*A-A*T*dA[k])/2 for k in range(q)]
            value+=n*(mp.log(mp.det(s))+trace(A*C)+(m.T*A*m)[0])/2
            for i in range(q):
                G[offset+i]+=n*(trace(V*dS[i])+(dM[i].T*A*m)[0])
                for j in range(q):H[offset+i,offset+j]+=n*(trace(dV[j]*dS[i])+(dM[i].T*dA[j]*m)[0]+(dM[i].T*A*dM[j])[0])
            # Literal observed-row score outer products, without centering.
            for row in range(X.rows):
                if tuple(j for j in range(p) if mask[row,j])!=obs:continue
                mr=mp.matrix([mu[j]-X[row,j] for j in obs]);Vr=(A-A*mr*mr.T*A)/2
                score=mp.matrix([trace(Vr*dS[i])+(dM[i].T*A*mr)[0] for i in range(q)])
                J[offset:offset+q,offset:offset+q]+=score*score.T
        offset+=q
    perm=cov_indices+mean_indices
    return value,G,H,J,perm


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    out=parser.parse_args().run_dir
    if(out/'comparisons.csv').exists():raise ValueError('fresh reference output required')
    mp.mp.dps=90;begin=time.monotonic();art=matrices(read(out/'artifacts.csv'));samples=matrices(read(out/'samples.csv'))
    pts={}
    for r in read(out/'points.csv'):pts.setdefault(int(r['point_id']),[]).append(r)
    patterns={};cache={};comparisons=[];producers=[];handoffs=[]
    for row in read(out/'audits.csv'):
        pid,cid=int(row['point_id']),int(row['case_id']);get=lambda n,opt=False:art(pid,n,opt)
        if cid not in patterns:patterns[cid]=raw_patterns(samples,cid)
        raw=patterns[cid];N=sum(x[2] for x in raw);source=[];inputs=[];g=1
        while get(f'source_cov_{g}',True) is not None:
            source.append((get(f'source_mean_{g}'),get(f'source_cov_{g}')))
            inputs.append((get(f'stage2_input_mean_{g}'),get(f'stage2_input_cov_{g}')));g+=1
        counts=[sum(x[2] for x in raw if x[0]==g) for g in range(1,len(source)+1)]
        key=(cid,tuple(tuple(list(mu)+list(S)) for mu,S in source))
        if key not in cache:cache[key]=saturated(raw,source,samples,cid)
        value,G,H,J,perm=cache[key]
        sg=mp.matrix([G[i] for i in perm]);sh=mp.matrix([[H[i,j] for j in perm] for i in perm])
        errors=[abs(value/N-binary(row['stage1_objective']))/max(1,abs(value/N)),relative(sg,get('stage1_gradient')),relative(sh,get('stage1_hessian'))]
        positive=mp.eigsy(sh,eigvals_only=True)[0]>0
        distance=mp.sqrt((sg.T*mp.lu_solve(sh,sg))[0]) if positive else mp.nan
        nd=binary(row['stage1_distance']) if row['stage1_distance']!='NA' else mp.nan
        de=abs(nd-distance)/max(1,abs(distance)) if positive and mp.isfinite(nd) else mp.nan
        wrong=row['stage1_raw_passed']=='TRUE' and (not positive or distance>binary(.01))
        comparisons.append(dict(point_id=pid,stage='stage1',kind=row['kind'],target=row['target'],objective_error=number(errors[0]),gradient_error=number(errors[1]),hessian_error=number(errors[2]),metric_error='nan',reference_positive=bool(positive),reference_distance=number(distance),distance_error=number(de),native_status=row['stage1_status'],agrees=bool(max(errors)<mp.mpf('1e-9') and not wrong and (not mp.isfinite(de) or de<mp.mpf('1e-8')))))
        # Raw information and repaired inversion information are different inputs.
        repaired=row['information_repaired']=='TRUE';ridge=binary(row['information_ridge']);expected_H=H+ridge*mp.eye(H.rows)
        source_H=get('source_H');info_error=relative(expected_H,source_H);meat_error=relative(J,get('source_J'))
        acov_checked=not repaired or positive
        acov_error=relative(expected_H**-1*J*expected_H**-1,get('source_acov')) if acov_checked else mp.nan
        producer_ok=info_error<mp.mpf('1e-9') and meat_error<mp.mpf('1e-9') and (not acov_checked or acov_error<mp.mpf('1e-8'))
        producers.append(dict(point_id=pid,check='source_information_meat_acov',kind=row['kind'],error=number(max(info_error,meat_error,acov_error if acov_checked else mp.mpf(0))),status='repaired_nonpositive_acov_unchecked' if not acov_checked else 'conditional_checked',passed=bool(producer_ok)))
        intensity=binary(row['transform_intensity']);D=mp.eye(H.rows);off=0;transform_error=mp.mpf(0)
        for (mu,S),(mi,Si) in zip(source,inputs):
            p=S.rows;pos=[(i,j) for j in range(p) for i in range(j,p)];expected=mp.matrix(S)
            for k,(i,j) in enumerate(pos):
                if i!=j:expected[i,j]=expected[j,i]=(1-intensity)*S[i,j];D[off+p+k,off+p+k]=1-intensity
            transform_error=max(transform_error,relative(expected,Si),relative(mu,mi));off+=p+len(pos)
        transform_error=max(transform_error,relative(D*get('source_acov')*D.T,get('stage2_input_acov')))
        producers.append(dict(point_id=pid,check='explicit_transformation',kind=row['kind'],error=number(transform_error),status='conditional_checked',passed=bool(transform_error<mp.mpf('1e-7'))))
        theta=get('stage2_theta');q=len(theta);g2=mp.matrix(q,1);h2=mp.matrix(q);metric=mp.matrix(q);objective=mp.mpf(0);off=0;weights=[]
        for block,((mu,S),n) in enumerate(zip(inputs,counts),1):
            p=S.rows;ps=p*(p+1)//2;qb=p+ps;kind=row['kind'];Gamma=gamma_nt(S)
            gf=n*get('stage2_input_acov')[off:off+qb,off:off+qb]
            W=mp.eye(qb) if kind=='uls' else mp.diag([1/gf[i,i] for i in range(qb)]) if kind=='dwls' else gf**-1 if kind=='adf' else ((1-binary(row['dls_a']))*Gamma+binary(row['dls_a'])*gf)**-1 if kind=='dls' else Gamma**-1
            weights.append(W)
            if kind=='nt':
                v,gg,hh,domain=likelihood(pts[pid],theta,[(block,tuple(range(p)),n,mu,S)])
                objective+=v-n*(mp.log(mp.det(S))+p)/2;g2+=gg;h2+=hh
            else:
                sigma,model_mu,Jac,second=moments(pts[pid],theta,block,True);pos=[(i,j) for j in range(p) for i in range(j,p)]
                residual=mp.matrix(list(model_mu-mu)+[sigma[i,j]-S[i,j] for i,j in pos]);wr=W*residual
                objective+=n*(residual.T*wr)[0]/2;g2+=n*Jac.T*wr;h2+=n*Jac.T*W*Jac
                for i in range(q):
                    for j in range(q):h2[i,j]+=n*(wr.T*second(i,j))[0]
                metric+=n*Jac.T*W*Gamma*W*Jac
                native_W=retained_weight(art,pid,block,p,True)
                we=relative(W,native_W)
                producers.append(dict(point_id=pid,check=f'weight_block_{block}',kind=kind,error=number(we),status='conditional_checked',passed=bool(we<mp.mpf('1e-9'))))
            off+=qb
        basis=get('stage2_derivative_basis');hr=basis.T*h2*basis;gr=basis.T*g2
        positive=mp.eigsy(hr,eigvals_only=True)[0]>0
        omega=hr if row['kind']=='nt' else basis.T*metric*basis
        distance=mp.sqrt((gr.T*mp.lu_solve(omega,gr))[0]) if positive else mp.nan
        nd=binary(row['stage2_distance']) if row['stage2_distance']!='NA' else mp.nan
        de=abs(nd-distance)/max(1,abs(distance)) if positive and mp.isfinite(nd) else mp.nan
        metric_error=mp.mpf(0) if row['kind']=='nt' else relative(metric,get('stage2_metric_factor').T*get('stage2_metric_factor'))
        errors=[abs(objective/N-binary(row['stage2_objective']))/max(1,abs(objective/N)),relative(g2,get('stage2_gradient')),relative(h2,get('stage2_hessian')),metric_error]
        wrong=row['stage2_raw_passed']=='TRUE' and (not positive or distance>binary(.01))
        comparisons.append(dict(point_id=pid,stage='stage2',kind=row['kind'],target=row['target'],objective_error=number(errors[0]),gradient_error=number(errors[1]),hessian_error=number(errors[2]),metric_error=number(metric_error),reference_positive=bool(positive),reference_distance=number(distance),distance_error=number(de),native_status=row['stage2_status'],agrees=bool(max(errors)<mp.mpf('1e-9') and not wrong and (not mp.isfinite(de) or de<mp.mpf('1e-8')))))
        record=get('recorded_kind',True);same=None
        if record is not None:
            same=int(record[0])==['nt','uls','dwls','adf','dls'].index(row['kind'])+1
            if row['kind']=='dls':same=same and equal(get('recorded_dls_a'),mp.matrix([binary(row['dls_a'])]))
            same=same and list(get('recorded_n_obs'))==list(get('stage2_input_n_obs'))==list(get('fit_n_obs'))
            for g,(mu,S) in enumerate(inputs,1):
                same=same and equal(mu,get(f'recorded_mean_{g}')) and equal(S,get(f'recorded_cov_{g}')) and equal(get(f'recorded_cov_{g}'),get(f'fit_cov_{g}')) and equal(get(f'recorded_mean_{g}'),get(f'fit_mean_{g}'))
                if row['kind']!='nt':same=same and relative(weights[g-1],get(f'supplied_weight_{g}'))<mp.mpf('1e-9')
            if row['kind'] not in ('nt','uls'):
                acov=get('recorded_acov',True)
                same=None if acov is None else same and equal(acov,get('stage2_input_acov'))
        expected='unchecked' if same is None else 'passed' if same else 'failed'
        handoffs.append(dict(point_id=pid,target=row['target'],expected=expected,actual=row['handoff_status'],agrees=expected==row['handoff_status']))
        if pid%5==0:print(f'ML2S reference {pid}; {time.monotonic()-begin:.1f}s',flush=True)
    write(out/'comparisons.csv',comparisons);write(out/'producer_checks.csv',producers);write(out/'handoff_checks.csv',handoffs)
    rows=read(out/'audits.csv');fits=read(out/'fits.csv');controls=read(out/'controls.csv')
    expected={'missing_input':'unchecked','missing_stop':'unchecked','legacy_stage1':'unchecked','missing_required_acov':'unchecked','mismatched_moments':'failed','mismatched_mix':'failed','mismatched_weight':'failed','mismatched_acov':'failed','repaired_raw_curvature':'failed','early_source_with_original_fit':'failed'}
    negative=[r for r in rows if r['target'] in expected]
    gates=[dict(check='independent_stage_points',tested=len(comparisons),failed=sum(not r['agrees'] for r in comparisons)),
           dict(check='conditional_producer_checks',tested=len(producers),failed=sum(not r['passed'] for r in producers)),
           dict(check='actual_input_handoff_identity',tested=len(handoffs),failed=sum(not r['agrees'] for r in handoffs)),
           dict(check='missing_mismatched_early_repaired_controls',tested=len(negative),failed=sum(r['status']!=expected[r['target']] for r in negative)),
           dict(check='strict_early_EM_rejects',tested=len(controls),failed=sum(r['rejected']!='TRUE' for r in controls)),
           dict(check='every_returned_fit_audited',tested=len(fits),failed=abs(sum(r['returned']=='TRUE' for r in fits)-sum(r['target']=='terminal' for r in rows)))]
    write(out/'verification.csv',gates)
    write(out/'reference_metadata.csv',[dict(precision_digits=90,mpmath_version=mp.__version__,elapsed_s=time.monotonic()-begin,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),convention='exact binary64 raw patterns and retained Stage-2 inputs; independent saturated information/score meat and analytic LISREL derivatives',limitations='conditional local evidence; repaired nonpositive-curvature inversion ACOV unchecked; no propagated input bound, inference or global guarantee')])
    if any(r['failed'] for r in gates):raise SystemExit('independent ML2S gate failed; evidence retained')


if __name__=='__main__':main()
