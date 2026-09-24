"""Matched starts and chart comparison; terminal audits share marker coordinates."""
import csv,json,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]);rows=list(csv.DictReader((folder/'raw.csv').open()))
models=['cfa6','cfa24','weakmarker12','mixed12','feedback12']
arms=[('marker','identity'),('std_lv','identity'),('marker','data_scaled')]
assert len(rows)==540
assert len({tuple(r[k] for k in ['model','n','rep','units','start_origin','chart','metric']) for r in rows})==540
for model in models:
 for origin in ['marker_native','std_native']:
  for chart,metric in arms:
   q=[r for r in rows if (r['model'],r['start_origin'],r['chart'],r['metric'])==(model,origin,chart,metric)]
   assert {(int(r['n']),int(r['rep']),round(float(r['units']),1)) for r in q}=={(n,rep,u) for n in [50,200,1000] for rep in [1,2] for u in [.1,1.,10.]}
assert max(float(r['start_invariance_error']) for r in rows)<1e-9

def summary(q):
 e=[r for r in q if r['interior']=='1' and r['status']=='available']
 return dict(attempted=len(q),eligible=len(e),passed=sum(float(r['distance'])<=.01 for r in e),
             median_evals=statistics.median(int(r['evals']) for r in q),total_evals=sum(int(r['evals']) for r in q),
             budget_returns=sum(r['rc']=='5' for r in q))
report=[]
for model in ['all']+models:
 for origin in ['marker_native','std_native']:
  for chart,metric in arms:
   q=[r for r in rows if (model=='all' or r['model']==model) and (r['start_origin'],r['chart'],r['metric'])==(origin,chart,metric)]
   report.append(dict(model=model,start_origin=origin,chart=chart,metric=metric,**summary(q)))
# Agreement is restricted to pairs passing the common audit and interior checks.
gaps=[]
for model in models:
 for n in [50,200,1000]:
  for rep in [1,2]:
   for units in [.1,1.,10.]:
    q=[r for r in rows if r['model']==model and int(r['n'])==n and int(r['rep'])==rep and float(r['units'])==units and r['interior']=='1' and r['status']=='available' and float(r['distance'])<=.01]
    if len(q)>1:gaps.append(max(float(r['fmin']) for r in q)-min(float(r['fmin']) for r in q))
result=dict(summary=report,max_passing_objective_spread=max(gaps),failures=[r for r in rows if r['interior']!='1' or r['status']!='available' or float(r['distance'])>.01])
(folder/'charts-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for r in report:print(r)
print('Max passing objective spread',max(gaps))
