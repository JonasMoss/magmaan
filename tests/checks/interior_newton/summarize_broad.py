"""Broad frozen-policy validation, preserving unsupported and noninterior cases."""
import csv,json,statistics,sys
from pathlib import Path
folders=[Path(x) for x in sys.argv[1:]]
assert len(folders)==2
allrows=[];reports=[]
for folder in folders:
 rows=list(csv.DictReader((folder/'raw.csv').open()));allrows.append(rows)
 checks=list(csv.DictReader((folder/"raw.csv.checks.csv").open()))
 assert len(checks)==51 and max(float(r["gradient_relative_error"]) for r in checks)<1e-5
 assert len(rows)==2079
 keys=['model','n_base','n_total','rep','seed','units','policy']
 assert len({tuple(r[k] for k in keys) for r in rows})==len(rows)
 models=sorted({r['model'] for r in rows});assert len(models)==21
 for m in models:
  for policy in ['baseline','candidate','std_lv']:
   q=[r for r in rows if r['model']==m and r['policy']==policy]
   assert len(q)==(3 if int(q[0]['seed'])==0 else 45)
   if int(q[0]['seed'])!=0:
    assert {(int(r['n_base']),int(r['rep']),round(float(r['units']),1)) for r in q}=={(n,rep,u) for n in [50,200,1000] for rep in range(1,6) for u in [.1,1.,10.]}
 def summarize(q):
  attempted=[r for r in q if r['branch']!='unsupported']
  eligible=[r for r in attempted if r['interior']=='1' and r['status']=='available']
  return dict(design_rows=len(q),attempted=len(attempted),unsupported=len(q)-len(attempted),
              eligible=len(eligible),passed=sum(float(r['distance'])<=.01 for r in eligible),
              noninterior=sum(r['interior']!='1' for r in attempted),unavailable=sum(r['status']!='available' for r in attempted),
              median_evals=statistics.median([int(r['evals']) for r in attempted]) if attempted else None,
              total_evals=sum(int(r['evals']) for r in attempted),
              prep_ms=sum(float(r['prep_ms']) for r in attempted),fit_ms=sum(float(r['fit_ms']) for r in attempted),
              total_ms=sum(float(r['total_ms']) for r in attempted),
              budget_returns=sum(r['rc']=='5' for r in attempted))
 report=dict(backtracking=int((folder/'backtracking-limit.txt').read_text()),
             overall={p:summarize([r for r in rows if r['policy']==p]) for p in ['baseline','candidate','std_lv']},
             by_model={m:{p:summarize([r for r in rows if r['model']==m and r['policy']==p]) for p in ['baseline','candidate','std_lv']} for m in models},
             candidate_branch={b:summarize([r for r in rows if r['policy']=='candidate' and r['branch']==b]) for b in ['transported','native_fallback']})
 # Pair by identical dataset/unit; keep unmatched eligibility visible above.
 index={tuple(r[k] for k in keys[:-1]):{} for r in rows}
 for r in rows:index[tuple(r[k] for k in keys[:-1])][r['policy']]=r
 def passes(r):return r['branch']!='unsupported' and r['interior']=='1' and r['status']=='available' and float(r['distance'])<=.01
 gaps=[]
 for cell in index.values():
  good=[float(r['fmin']) for r in cell.values() if passes(r)]
  if len(good)>1:gaps.append(max(good)-min(good))
 report['max_passing_objective_spread']=max(gaps)
 report['paired_baseline_candidate']={
 'both_pass':sum(passes(c['baseline']) and passes(c['candidate']) for c in index.values()),
 'baseline_only_pass':sum(passes(c['baseline']) and not passes(c['candidate']) for c in index.values()),
 'candidate_only_pass':sum(not passes(c['baseline']) and passes(c['candidate']) for c in index.values())}
 report['candidate_failures']=[r for r in rows if r['policy']=='candidate' and not passes(r)]
 (folder/'broad-summary.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
 print('BACKTRACKING',report['backtracking']);print(json.dumps(report['overall'],indent=2));print('PAIRED',report['paired_baseline_candidate']);print('MAX GAP',max(gaps))
print('MODEL, baseline stock, candidate stock, std stock, candidate extended (pass/eligible)')
for m in reports[0]['by_model']:
 print(m,*[str(r['passed'])+'/'+str(r['eligible']) for r in [reports[0]['by_model'][m][p] for p in ['baseline','candidate','std_lv']]+[reports[1]['by_model'][m]['candidate']]])
assert [[r[k] for k in keys] for r in allrows[0]]==[[r[k] for k in keys] for r in allrows[1]]
