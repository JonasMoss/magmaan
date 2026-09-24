"""Paired feedback starts/coordinates diagnostic; no production start changes."""
import csv
import json
import statistics
import sys
from pathlib import Path
folder = Path(sys.argv[1])
rows = list(csv.DictReader((folder/'raw.csv').open()))
starts = list(csv.DictReader((folder/'raw.csv.starts.csv').open()))
arms = ['native', 'mapped_start', 'mapped_coordinates']
assert len(rows) == 72 and len(starts) == 696
assert {(int(r['n']),int(r['rep']),float(r['units']),r['arm']) for r in rows} == {
    (n,rep,u,a) for n in [50,200,1000,10000] for rep in [1,2] for u in [.1,1.,10.] for a in arms}
for r in starts:
    if r['matrix'] == 'Psi' and r['row'] == r['col']:
        assert float(r['native']) == .05
        assert abs(float(r['mapped'])-.05*float(r['units'])**2) < 1e-12
    else:
        assert abs(float(r['native'])-float(r['mapped'])) < 1e-10
report = []
for a in arms:
    for u in [.1,1.,10.]:
        q = [r for r in rows if r['arm']==a and float(r['units'])==u]
        eligible = [r for r in q if r['interior']=='1' and r['status']=='available']
        report.append(dict(arm=a, units=u, attempts=len(q), eligible=len(eligible),
                           passed=sum(float(r['distance'])<=.01 for r in eligible),
                           max_d=max(float(r['distance']) for r in eligible),
                           median_evals=statistics.median(int(r['evals']) for r in q),
                           min_evals=min(int(r['evals']) for r in q),
                           max_evals=max(int(r['evals']) for r in q)))
max_gap=0
for r in rows:
    if r['arm']=='native': continue
    ref=next(x for x in rows if x['n']==r['n'] and x['rep']==r['rep'] and x['units']=='1' and x['arm']=='native')
    max_gap=max(max_gap,abs(float(r['fmin'])-float(ref['fmin'])))
assert max_gap<1e-8
result=dict(summary=report,max_objective_difference_from_unit1=max_gap)
(folder/'starts-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
