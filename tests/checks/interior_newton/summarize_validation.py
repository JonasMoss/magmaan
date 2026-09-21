"""Describe fixed-control validation coverage and outcomes, including exclusions."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
rows = list(csv.DictReader((folder / 'raw.csv').open()))
models = {'hold_cfa6': (6, 1), 'hold_cfa16': (16, 4),
          'hold_cfa32': (32, 8), 'hold_cfa48': (48, 12),
          'hold_chain12': (12, 3), 'hold_cross12': (12, 3),
          'hold_residual12': (12, 3), 'hold_equal24': (24, 6),
          'hold_weak12': (12, 3), 'hold_mixed16': (16, 4)}
profiles = ['current', 'x8', 'f12_x10']
keys = ['model', 'n', 'rep', 'units', 'seed']
assert len(rows) == 720
assert {r['model'] for r in rows} == models.keys()
assert {r['profile'] for r in rows} == set(profiles)
assert len({tuple(r[k] for k in keys + ['profile']) for r in rows}) == len(rows)
for model, dimensions in models.items():
    for profile in profiles:
        panel = [r for r in rows if r['model'] == model and r['profile'] == profile]
        assert len(panel) == 24
        assert {(int(r['n']), int(r['rep']), round(float(r['units']), 1)) for r in panel} == {
            (n, rep, unit) for n in [50, 200, 1000, 10000] for rep in [1, 2] for unit in [.1, 1., 10.]}
        assert all((int(r['p']), int(r['factors'])) == dimensions for r in panel)
        assert len({r['free_parameters'] for r in panel}) == 1
fd = [float(r['hessian_fd_rel']) for r in rows if math.isfinite(float(r['hessian_fd_rel']))]
assert len(fd) == 10 and max(fd) < 1e-5

def eligible(r):
    return r['interior'] == '1' and r['status'] == 'available'

def summarize(panel):
    q = [r for r in panel if eligible(r)]
    return dict(attempted=len(panel), eligible=len(q),
                passed=sum(float(r['distance']) <= .01 for r in q),
                noninterior=sum(r['interior'] != '1' for r in panel),
                unavailable_curvature=sum(r['status'] != 'available' for r in panel),
                max_d=max((float(r['distance']) for r in q), default=None),
                median_evals=statistics.median(int(r['evals']) for r in panel),
                total_evals=sum(int(r['evals']) for r in panel),
                budget_returns=sum(int(r['rc']) == 5 for r in panel))

report = dict(overall={p: summarize([r for r in rows if r['profile'] == p]) for p in profiles},
              by_model={m: {p: summarize([r for r in rows if r['model'] == m and r['profile'] == p])
                            for p in profiles} for m in models},
              by_n={n: {p: summarize([r for r in rows if int(r['n']) == n and r['profile'] == p])
                        for p in profiles} for n in [50, 200, 1000, 10000]})
report['failures'] = [{k: r[k] for k in keys + ['profile', 'distance', 'rc']}
                      for r in rows if eligible(r) and float(r['distance']) > .01]
(folder / 'validation-summary.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['| Model | p | Free parameters | Current pass/eligible | Cheap pass/eligible | Conservative pass/eligible |',
         '|---|---:|---:|---:|---:|---:|']
for m, (p, _) in models.items():
    counts = [f"{report['by_model'][m][pr]['passed']}/{report['by_model'][m][pr]['eligible']}" for pr in profiles]
    parameters = next(r['free_parameters'] for r in rows if r['model'] == m)
    lines.append(f"| {m} | {p} | {parameters} | " + ' | '.join(counts) + ' |')
text = '\n'.join(lines) + '\n'
(folder / 'validation-summary.md').write_text(text)
print(text)
print(json.dumps(report['overall'], indent=2))
print('Eligible failures:', len(report['failures']))
