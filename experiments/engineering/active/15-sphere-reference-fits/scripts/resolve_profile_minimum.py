#!/usr/bin/env python3
"""Resolve the sign-changing profile slope, then check the unrestricted Hessian."""
import argparse
import json
import mpmath as mp
from refine_open_cases import ROOT, read, write, Model, refine

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--digits',type=int,default=60)
parser.add_argument('--run-id',default='delicate-finite-minimum')
args=parser.parse_args()
if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
    raise ValueError('invalid run id')
mp.mp.dps=args.digits
inputs=ROOT/'results'/'open-case-inputs'
case=next(c for c in read(inputs/'cases.csv') if c['batch']=='development' and c['design']=='weak_marker' and c['n']=='100' and c['rep']=='2' and c['source']=='sphere')
S=mp.matrix(6)
for r in read(inputs/'moments.csv'):
    if r['case_id']==case['case_id']:
        S[int(r['row'])-1,int(r['column'])-1]=mp.mpf(r['value'])
model=Model(S)
profile=ROOT/'results'/'delicate-loading-profile'
point=mp.matrix([mp.mpf(r['value']) for r in read(profile/'parameters.csv') if r['fixed_y3_loading']=='600'])
out=ROOT/'results'/args.run_id
if (out/'summary.csv').exists():raise ValueError('choose fresh run id')
out.mkdir(parents=True,exist_ok=True)
lo,hi=mp.mpf(600),mp.mpf(900)
t=mp.mpf(750)
traces=[]
for iteration in range(25):
    # Retain a nearby feasible covariance when changing the fixed loading.
    _,old_sigma=model.evaluate(point)
    ratio=point[3]/t
    point[3]=t;point[5]*=ratio;point[6]*=ratio
    L=mp.matrix([[1,0],[point[0],0],[point[1],0],[0,1],[0,point[2]],[0,point[3]]])
    shared=L*mp.matrix([[point[4],point[5]],[point[5],point[6]]])*L.T
    for i in range(6):point[7+i]=old_sigma[i,i]-shared[i,i]
    result,_,point,Sigma=refine(model,point,80,'profile_root',fixed=3)
    assert result['status']=='finite_stationary_positive_curvature'
    f,Sigma,g,H=model.evaluate(point,True)
    active=[i for i in range(13) if i!=3]
    Hfree=mp.matrix([[H[i,j] for j in active] for i in active])
    cross=mp.matrix([H[i,3] for i in active])
    curvature=H[3,3]-mp.fdot(cross,mp.lu_solve(Hfree,cross))
    traces.append(dict(iteration=iteration,loading=mp.nstr(t,args.digits),objective=mp.nstr(f,40),
        profile_slope=mp.nstr(g[3],20),profile_curvature=mp.nstr(curvature,20)))
    print(f'profile step {iteration}: loading={mp.nstr(t,14)}, slope={mp.nstr(g[3],8)}',flush=True)
    if abs(g[3])<mp.mpf('1e-35'):break
    if g[3]<0:lo=t
    else:hi=t
    proposal=t-g[3]/curvature
    t=proposal if lo<proposal<hi else (lo+hi)/2
result,history,point,Sigma=refine(model,point,20,'resolved')
assert result['status']=='finite_stationary_positive_curvature'
write(out/'summary.csv',[dict(**case,**result)])
write(out/'profile_root.csv',traces)
write(out/'trace.csv',history)
write(out/'parameters.csv',[dict(parameter=i+1,value=mp.nstr(v,args.digits)) for i,v in enumerate(point)])
write(out/'covariances.csv',[dict(row=i+1,column=j+1,value=mp.nstr(Sigma[i,j],args.digits))for i in range(6)for j in range(6)])
(out/'metadata.json').write_text(json.dumps(dict(digits=args.digits,method='profile slope root in [600,900], followed by unrestricted high-precision Newton check',
    identification='original x1/y1 markers',interpretation='numerical finite local minimum; not a global-optimality proof'),indent=2)+'\n')
print(result,flush=True)
