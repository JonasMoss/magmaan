#!/usr/bin/env python3
"""Check derivative formulas and retained finite local witnesses; consolidate seven cases."""
import mpmath as mp
from refine_open_cases import ROOT, read, write, Model

mp.mp.dps=60
root=ROOT/'results'
inputs=root/'open-case-inputs'
cases=read(inputs/'cases.csv');moments=read(inputs/'moments.csv');starts=read(inputs/'starts.csv')
refined=read(root/'open-case-refinement'/'summary.csv')
parameters=read(root/'open-case-refinement'/'parameters.csv')
profile=read(root/'delicate-finite-minimum'/'summary.csv')[0]
profile90=read(root/'delicate-finite-minimum-90'/'summary.csv')[0]
assert abs(mp.mpf(profile['objective'])-mp.mpf(profile90['objective']))<mp.mpf('1e-27')
p60=read(root/'delicate-finite-minimum'/'parameters.csv');p90=read(root/'delicate-finite-minimum-90'/'parameters.csv')
assert max(abs(mp.mpf(a['value'])-mp.mpf(b['value']))/(1+abs(mp.mpf(a['value'])))for a,b in zip(p60,p90))<mp.mpf('1e-25')
results=[];vectors=[]
for case_id in sorted(set(c['case_id']for c in cases),key=int):
    sample=mp.matrix(6)
    for r in moments:
        if r['case_id']==case_id:sample[int(r['row'])-1,int(r['column'])-1]=mp.mpf(r['value'])
    model=Model(sample)
    # Direct high-precision differentiation of the scalar objective checks
    # the independently implemented matrix gradient and Hessian formulas.
    original=next(c for c in cases if c['case_id']==case_id and c['source']=='sphere')
    x=mp.matrix([mp.mpf(r['value'])for r in starts if r['start_id']==original['start_id']])
    _,_,g,H=model.evaluate(x,True)
    for indices in [[0],[4],[7],[0,1,4,5,7],[2,3,5,6,11]]:
        direction=mp.matrix(13,1)
        for i in indices:direction[i]=1/(1+abs(x[i]))
        fun=lambda t:model.evaluate(x+t*direction)[0]
        d1=mp.diff(fun,0,1);d2=mp.diff(fun,0,2)
        assert abs(d1-mp.fdot(g,direction))<mp.mpf('1e-38')*(1+abs(d1))
        assert abs(d2-mp.fdot(direction,H*direction))<mp.mpf('1e-38')*(1+abs(d2))
    candidates=[r for r in refined if r['case_id']==case_id and r['status']=='finite_stationary_positive_curvature']
    if candidates:
        selected=min(candidates,key=lambda r:mp.mpf(r['objective']))
        x=mp.matrix([mp.mpf(r['value'])for r in parameters if r['start_id']==selected['start_id']])
        method='full Newton refinement'
    else:
        assert profile['case_id']==case_id
        selected=profile;x=mp.matrix([mp.mpf(r['value'])for r in p60]);method='profile minimum, confirmed at 90 digits'
    f,Sigma,g,H=model.evaluate(x,True)
    mp.cholesky(Sigma);mp.cholesky(H)
    scaled_gradient=max(abs(g[i])/mp.sqrt(abs(H[i,i]))for i in range(13))
    assert scaled_gradient<mp.mpf('1e-27')
    assert abs(f-mp.mpf(selected['objective']))<mp.mpf('1e-27')
    assert x[4]!=0
    results.append(dict(case_id=case_id,batch=selected['batch'],design=selected['design'],n=selected['n'],rep=selected['rep'],
        source=selected['source'],method=method,objective=selected['objective'],scaled_gradient=mp.nstr(scaled_gradient,15),
        min_scaled_hessian_eigen=selected['min_scaled_hessian_eigen'],component_extent=selected['component_extent'],
        max_loading=selected['max_loading'],var_X=selected['var_X'],beta=selected['beta'],disturbance_Y=selected['disturbance_Y']))
    vectors.extend(dict(case_id=case_id,parameter=i+1,value=mp.nstr(v,60))for i,v in enumerate(x))
out=root/'open-case-conclusions';out.mkdir(parents=True,exist_ok=True)
write(out/'finite_witnesses.csv',results);write(out/'parameters.csv',vectors)
print('Seven finite stationary points have positive definite observed covariance and Hessian.')
print('Directional derivative checks passed for all seven inputs; delicate minimum agrees at 60 and 90 digits.')
print('These are numerical local witnesses, not a global-optimality proof.')
