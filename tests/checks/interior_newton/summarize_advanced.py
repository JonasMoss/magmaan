"""Fixed-profile coverage report; exploratory option subgroup analysis is deferred."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
rows = list(csv.DictReader((folder / 'raw.csv').open()))
profiles = ['current', 'x8', 'f12_x10']
synthetic = ['growth4', 'growth8', 'feedback12', 'mg_configural', 'mg_metric',
             'mg_scalar', 'mg5_metric_unbalanced']
corpus = ['bollen_democracy_sem', 'hs_3factor_cfa',
          'kline_2023_ch22_guo_mi_configural', 'kline_2023_ch22_guo_mi_weak',
          'kline_2023_ch22_guo_mi_strong', 'kline_2023_ch22_guo_mi_partial_strong']
models = synthetic + corpus
keys = ['model', 'n_base', 'rep', 'units', 'seed']
assert len(rows) == 558
assert {r['model'] for r in rows} == set(models)
assert len({tuple(r[k] for k in keys + ['profile']) for r in rows}) == len(rows)
for m in models:
    panels = [{tuple(r[k] for k in keys): r for r in rows if r['model'] == m and r['profile'] == p}
              for p in profiles]
    assert all(len(panel) == (24 if m in synthetic else 3) for panel in panels)
    assert panels[0].keys() == panels[1].keys() == panels[2].keys()
for r in rows:
    sizes = [int(x) for x in r['group_sizes'].split(';')]
    assert len(sizes) == int(r['groups']) and sum(sizes) == int(r['n_total'])
    assert min(sizes) > int(r['p'])
    if r['model'] in synthetic:
        assert int(r['n_base']) in [50, 200, 1000, 10000] and int(r['rep']) in [1, 2]
fd = [float(r['hessian_fd_rel']) for r in rows if math.isfinite(float(r['hessian_fd_rel']))]
assert len(fd) == 34 and max(fd) < 1e-5

def summarize(panel):
    q = [r for r in panel if r['interior'] == '1' and r['status'] == 'available']
    return dict(attempts=len(panel), eligible=len(q), passed=sum(float(r['distance']) <= .01 for r in q),
                noninterior=sum(r['interior'] != '1' for r in panel),
                unavailable=sum(r['status'] != 'available' for r in panel),
                max_d=max((float(r['distance']) for r in q), default=None),
                median_evals=statistics.median(int(r['evals']) for r in panel),
                total_evals=sum(int(r['evals']) for r in panel),
                budget_returns=sum(int(r['rc']) == 5 for r in panel))
report = dict(overall={p: summarize([r for r in rows if r['profile'] == p]) for p in profiles},
              by_model={m: {p: summarize([r for r in rows if r['model'] == m and r['profile'] == p])
                            for p in profiles} for m in models}, max_hessian_fd_error=max(fd))
report['failures'] = [r for r in rows if r['interior'] == '1' and r['status'] == 'available'
                      and float(r['distance']) > .01]
(folder / 'advanced-summary.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['| Model | p | Groups | Free parameters | Current | Cheap | Conservative |',
         '|---|---:|---:|---:|---:|---:|---:|']
for m in models:
    r = next(r for r in rows if r['model'] == m)
    cells = [f"{report['by_model'][m][p]['passed']}/{report['by_model'][m][p]['eligible']}" for p in profiles]
    lines.append(f"| {m} | {r['p']} | {r['groups']} | {r['free_parameters']} | " + ' | '.join(cells) + ' |')
text = '\n'.join(lines) + '\n'
(folder / 'advanced-summary.md').write_text(text)
# These are pre-fit descriptors for later paired analyses, not learned routing rules.
metadata = []
for r in rows:
    if r['profile'] != 'current':
        continue
    sizes = [int(x) for x in r['group_sizes'].split(';')]
    metadata.append({**{k: r[k] for k in keys + ['family', 'source', 'p', 'factors', 'groups', 'free_parameters']},
                     'n_min': min(sizes), 'n_max': max(sizes), 'n_total': sum(sizes),
                     'group_n_ratio': max(sizes)/min(sizes)})
(folder / 'case-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(text)
print(json.dumps(report['overall'], indent=2))
print('FD checks:', len(fd), 'max error:', max(fd))
