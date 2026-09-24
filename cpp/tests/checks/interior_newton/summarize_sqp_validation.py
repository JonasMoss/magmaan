"""Frozen SQP follow-up: paired accuracy, domain outcomes, work and objective gaps."""
import csv,json,math,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]);rows=list(csv.DictReader((folder/'raw.csv').open()))
policies=['ordinary_native','ordinary_scaled','psd_native','psd_information']
keys=['model','n_base','n_total','rep','seed','units']
assert len(rows)==1692
assert len({tuple(r[k] for k in keys+['policy']) for r in rows})==len(rows)
models=sorted({r['model'] for r in rows});assert len(models)==21
for m in models:
 for p in policies:
  q=[r for r in rows if r['model']==m and r['policy']==p]
  assert len(q)==(3 if q[0]['seed']=='0' else 27)
  if q[0]['seed']!='0':
   assert {(int(r['n_base']),int(r['rep']),round(float(r['units']),1)) for r in q}=={(n,r,u) for n in [50,500,5000] for r in [1,2,3] for u in [.1,1.,10.]}
def interior(r):
 return r['returned']=='1' and r['interior']=='1' and r['newton_status']=='available' and (not r['policy'].startswith('psd') or r['nullity']=='0')
def accuracy(r):return interior(r) and r['feasible']=='1' and float(r['distance'])<=.01
def cone(r):return r['returned']=='1' and r['feasible']=='1' and r['covariance_feasible']=='1' and r['cone_stationary']=='1'
def accepted(r):
 if not r['policy'].startswith('psd'):return accuracy(r)
 return (accuracy(r) and r['covariance_feasible']=='1') if interior(r) else (cone(r) and int(r['nullity'])>0)
def summary(q):
 returned=[r for r in q if r['returned']=='1'];b=[r for r in returned if r['covariance_feasible']=='1' and int(r['nullity'])>0]
 return dict(attempts=len(q),returned=len(returned),errors=len(q)-len(returned),interior_eligible=sum(interior(r) for r in q),
             interior_pass=sum(accuracy(r) for r in q),boundary=len(b),boundary_cone_pass=sum(cone(r) for r in b),
             all_cone_pass=sum(cone(r) for r in q),accepted=sum(accepted(r) for r in q),
             legacy_cone_and_newton=sum(cone(r) and (accuracy(r) if interior(r) else int(r['nullity'])>0) for r in q) if q[0]['policy'].startswith('psd') else None,
             median_evals=statistics.median(int(r['evals']) for r in returned) if returned else None,
             known_evals=sum(int(r['evals']) for r in returned),
             fit_ms=sum(float(r['fit_ms']) for r in q),total_ms=sum(float(r['total_ms']) for r in q),
             budget_returns=sum(r['raw_rc']=='5' for r in q))
report=dict(overall={p:summary([r for r in rows if r['policy']==p]) for p in policies},
            by_model={m:{p:summary([r for r in rows if r['model']==m and r['policy']==p]) for p in policies} for m in models})
index={}
for r in rows:index.setdefault(tuple(r[k] for k in keys),{})[r['policy']]=r
for domain,baseline,candidate in [('ordinary','ordinary_native','ordinary_scaled'),('psd','psd_native','psd_information')]:
 pairs=[]
 for key,cell in index.items():
  a,b=cell[baseline],cell[candidate];delta=None
  if a['returned']=='1' and b['returned']=='1' and math.isfinite(float(a['fmin'])) and math.isfinite(float(b['fmin'])):
   delta=2*int(a['n_total'])*(float(a['fmin'])-float(b['fmin']))
  pairs.append(dict(case=key,baseline_accepted=accepted(a),candidate_accepted=accepted(b),
                    twice_loglik_gain=delta,baseline_boundary=a['nullity'],candidate_boundary=b['nullity']))
 report[domain+'_paired']=dict(both_pass=sum(p['baseline_accepted'] and p['candidate_accepted'] for p in pairs),
  baseline_only=sum(p['baseline_accepted'] and not p['candidate_accepted'] for p in pairs),
  candidate_only=sum(not p['baseline_accepted'] and p['candidate_accepted'] for p in pairs))
 # 1 unit of twice-loglik is a descriptive substantial-gap flag, not an audit tolerance.
 report[domain+'_substantial_gaps']=[p for p in pairs if p['twice_loglik_gain'] is not None and abs(p['twice_loglik_gain'])>1]
report['failures']=[r for r in rows if not accepted(r)]
report['legacy_residual_disagreements']=[r for r in rows if r['policy'].startswith('psd') and accuracy(r) and not cone(r)]
report['interior_failures']=[r for r in rows if interior(r) and not accuracy(r)]
(folder/'sqp-validation-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['overall'],indent=2))
for d in ['ordinary','psd']:
 print(d,report[d+'_paired']);print('Substantial objective differences',len(report[d+'_substantial_gaps']))
 for p in report[d+'_substantial_gaps']:print(p)
print('Interior accuracy failures',len(report['interior_failures']))
