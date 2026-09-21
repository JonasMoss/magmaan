"""Summarize the paired intermediate-tolerance diagnostic (standard library only)."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
with (folder / 'raw.csv').open() as stream:
    rows = list(csv.DictReader(stream))
names = ['current', 'x8', 'x9', 'x10', 'f11_x9', 'f12_x10']
keys = ['model', 'n', 'rep', 'units', 'seed']
assert len(rows) == 1134
panels = {name: {tuple(row[k] for k in keys): row for row in rows
                 if row['profile'] == name} for name in names}
assert all(len(panel) == 189 and panel.keys() == panels['current'].keys()
           for panel in panels.values())
fd_errors = [float(row['hessian_fd_rel']) for row in rows
             if math.isfinite(float(row['hessian_fd_rel']))]
assert len(fd_errors) == 7 and max(fd_errors) < 1e-5
summary = []
failures = []
for name, panel in panels.items():
    eligible = [row for row in panel.values()
                if row['interior'] == '1' and row['status'] == 'available']
    passed = [row for row in eligible if float(row['distance']) <= .01]
    ratios = [int(row['evals']) / int(panels['current'][key]['evals'])
              for key, row in panel.items()]
    summary.append(dict(profile=name, attempted=len(panel), eligible=len(eligible),
                        passed=len(passed), max_distance=max(float(r['distance']) for r in eligible),
                        median_evals=statistics.median(int(r['evals']) for r in panel.values()),
                        total_evals=sum(int(r['evals']) for r in panel.values()),
                        median_paired_eval_ratio=statistics.median(ratios),
                        median_fit_ms=statistics.median(float(r['fit_ms']) for r in panel.values()),
                        negative_codes=sum(int(r['rc']) < 0 for r in panel.values()),
                        negative_codes_passing=sum(int(r['rc']) < 0 for r in passed)))
    failures.extend({k: row[k] for k in keys + ['profile', 'distance', 'rc']}
                    for row in eligible if float(row['distance']) > .01)
report = dict(summary=summary, eligible_failures=failures,
              warning='Paired exploratory panel; extended backtracking is a local diagnostic patch. '
                      'Counts do not establish general reliability; single-run timings are descriptive.')
(folder / 'options-summary.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['| Profile | Eligible | Pass .01 | Max d | Median evaluations | Total evaluations | Median ms |',
         '|---|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['profile']} | {r['eligible']} | {r['passed']} | {r['max_distance']:.4g} | "
                 f"{r['median_evals']:g} | {r['total_evals']} | {r['median_fit_ms']:.3f} |")
text = '\n'.join(lines) + '\n'
(folder / 'options-summary.md').write_text(text)
print(text)
print(json.dumps(failures, indent=2))
