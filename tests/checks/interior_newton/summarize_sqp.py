"""Separate regular-interior accuracy from existing PSD cone diagnostics."""
import csv,json,math,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]);rows=list(csv.DictReader((folder/'raw.csv').open()))
policies=['ordinary_native','ordinary_start','ordinary_scaled','psd_native','psd_start','psd_information']
assert len(rows)==648
keys=['model','n_base','n_total','rep','seed','units','policy']
assert len({tuple(r[k] for k in keys) for r in rows})==648
assert len({r['model'] for r in rows})==21
for model in {r['model'] for r in rows}:
 for p in policies:
  q=[r for r in rows if r['model']==model and r['policy']==p]
  assert len(q)==(3 if q[0]['seed']=='0' else 6)

def interior(r):
 return r['returned']=='1' and r['interior']=='1' and r['newton_status']=='available' and (not r['policy'].startswith('psd') or r['nullity']=='0')
def passes(r):return interior(r) and float(r['distance'])<=.01 and r['feasible']=='1'
def cone_pass(r):return r['returned']=='1' and r['feasible']=='1' and r['covariance_feasible']=='1' and r['cone_stationary']=='1'
def summarize(q):
 returned=[r for r in q if r['returned']=='1']
 boundary=[r for r in returned if r['covariance_feasible']=='1' and int(r['nullity'])>0]
 return dict(attempts=len(q),returned=len(returned),errors=len(q)-len(returned),
             eligible_interior=sum(interior(r) for r in q),interior_pass=sum(passes(r) for r in q),
             boundary=len(boundary),boundary_cone_pass=sum(cone_pass(r) for r in boundary),
             all_cone_pass=sum(cone_pass(r) for r in q),
             median_evals=statistics.median(int(r['evals']) for r in returned) if returned else None,
             known_evals=sum(int(r['evals']) for r in returned),
             median_fit_ms=statistics.median(float(r['fit_ms']) for r in q))
report=dict(overall={p:summarize([r for r in rows if r['policy']==p]) for p in policies},
            by_model={m:{p:summarize([r for r in rows if r['model']==m and r['policy']==p]) for p in policies} for m in sorted({r['model'] for r in rows})})
report['interior_failures']=[r for r in rows if interior(r) and not passes(r)]
report['psd_other_failures']=[r for r in rows if r['policy'].startswith('psd') and not cone_pass(r)]
# Compare policies only within the same optimization domain.
for domain in ['ordinary','psd']:
 gaps=[]
 for key in {tuple(r[k] for k in keys[:-1]) for r in rows}:
  q=[r for r in rows if tuple(r[k] for k in keys[:-1])==key and r['policy'].startswith(domain) and (passes(r) if domain=='ordinary' else cone_pass(r))]
  if len(q)>1:gaps.append(dict(case=key,gap=max(float(r['fmin']) for r in q)-min(float(r['fmin']) for r in q)))
 report[domain+'_largest_objective_gaps']=sorted(gaps,key=lambda x:x['gap'],reverse=True)[:5]
(folder/'sqp-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['overall'],indent=2))
print('Interior failures',len(report['interior_failures']),'PSD cone failures',len(report['psd_other_failures']))
print('PSD largest gaps',report['psd_largest_objective_gaps'])
