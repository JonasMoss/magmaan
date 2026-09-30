#!/usr/bin/env python3
"""Conditional likelihood profile for development weak-marker N=100 draw 2.

Fix the original-marker loading of y3, reoptimize all other covariance-model
parameters at 60 digits, and retain the profile derivative. This is a local
profile, not an exhaustive/global search or proof of nonattainment.
"""
import argparse
import json
import time
import mpmath as mp
from refine_open_cases import ROOT, read, write, Model, refine

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-id', default='delicate-loading-profile')
args = parser.parse_args()
if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id) or not args.run_id:
    raise ValueError('invalid run id')
mp.mp.dps = 60
inputs = ROOT/'results'/'open-case-inputs'
cases = read(inputs/'cases.csv')
case = next(c for c in cases if c['batch']=='development' and c['design']=='weak_marker' and c['n']=='100' and c['rep']=='2' and c['source']=='sphere')
S = mp.matrix(6)
for r in read(inputs/'moments.csv'):
    if r['case_id']==case['case_id']:
        S[int(r['row'])-1,int(r['column'])-1] = mp.mpf(r['value'])
start = mp.matrix([mp.mpf(r['value']) for r in read(inputs/'starts.csv') if r['start_id']==case['start_id']])
model = Model(S)
_, reference_sigma = model.evaluate(start)
out = ROOT/'results'/args.run_id
if (out/'summary.csv').exists():
    raise ValueError('choose fresh run id')
out.mkdir(parents=True, exist_ok=True)
rows, parameters, covariances = [], [], []
begin = time.monotonic()
for t in [100, 300, 600, 900, 1200, 2000, 5000, 20000]:
    x = mp.matrix(start)
    ratio = x[3]/t
    x[3] = mp.mpf(t)
    x[5] *= ratio
    x[6] *= ratio
    L = mp.matrix([[1,0],[x[0],0],[x[1],0],[0,1],[0,x[2]],[0,x[3]]])
    shared = L*mp.matrix([[x[4],x[5]],[x[5],x[6]]])*L.T
    for i in range(6):
        x[7+i] = reference_sigma[i,i]-shared[i,i]
    result, history, point, Sigma = refine(model, x, 80, str(t), fixed=3)
    _, _, gradient, _ = model.evaluate(point, True)
    result['status'] = result['status'].replace('finite_stationary', 'conditional_stationary')
    rows.append(dict(fixed_y3_loading=t, profile_derivative=mp.nstr(gradient[3],25), **result))
    parameters.extend(dict(fixed_y3_loading=t, parameter=i+1, value=mp.nstr(v,60)) for i,v in enumerate(point))
    covariances.extend(dict(fixed_y3_loading=t,row=i+1,column=j+1,value=mp.nstr(Sigma[i,j],60)) for i in range(6) for j in range(6))
    print(f"loading {t}: {result['status']}; F={result['objective']}; derivative={mp.nstr(gradient[3],8)}; {time.monotonic()-begin:.1f}s", flush=True)
write(out/'summary.csv',rows)
write(out/'parameters.csv',parameters)
write(out/'covariances.csv',covariances)
(out/'metadata.json').write_text(json.dumps(dict(digits=60, fixed_parameter='y3 loading, y1 marker fixed at 1', source=case,
    interpretation='local conditional profile; not global optimization or nonattainment proof'),indent=2)+'\n')
