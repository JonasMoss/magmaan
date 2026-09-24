#!/usr/bin/env python3
"""Summarize the bounded refined-reference smoke; read/write one run directory."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = list(csv.DictReader((run / 'raw.csv').open()))
assert len(rows) == 146, 'Expected 90 natural endpoints and 56 controlled probes'
assert len({(r['model'], r['n'], r['rep'], r['units'], r['source'],
             r['target'], r['direction']) for r in rows}) == len(rows)


def values(group, key):
    return [float(r[key]) for r in group if math.isfinite(float(r[key]))]


def summarize(group):
    qualified = [r for r in group if r['reference_status'] == 'qualified']
    result = {'attempted': len(group), 'qualified': len(qualified),
              'distance_resolved': sum(r['distance_resolved'] == '1' for r in qualified),
              'gain_resolved': sum(r['gain_resolved'] == '1' for r in qualified)}
    for key in ('predicted', 'reference_distance', 'actual_over_predicted',
                'step_relative_error', 'gain_over_predicted', 'anchor_agreement',
                'refinement_steps', 'refinement_ms'):
        v = values(qualified, key)
        result[key] = {'min': min(v), 'median': statistics.median(v), 'max': max(v)} if v else None
    # Threshold disagreements are descriptive: the point's Hessian defines
    # both the predicted and measured distance; references remain numerical.
    result['threshold_disagreements'] = {
        str(t): sum((float(r['predicted']) <= t) != (float(r['actual']) <= t)
                    for r in qualified) for t in (.003, .01, .03)
    }
    return result

summary = {}
for source in ('current', 'f12_x10'):
    summary[source] = summarize([r for r in rows if r['source'] == source])
for target in (.003, .01, .03, .1):
    summary[f'probe_{target}'] = summarize(
        [r for r in rows if r['source'] == 'probe' and float(r['target']) == target])
summary['excluded'] = [{k: r[k] for k in ('model', 'n', 'units', 'source',
                                         'interior', 'status', 'reference_status')}
                       for r in rows if r['reference_status'] != 'qualified']
(run / 'refinement-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['| Source | Qualified / attempted | Resolved distances | Actual / predicted range | Max vector error |',
         '|---|---:|---:|---:|---:|']
for key, value in summary.items():
    if key == 'excluded':
        continue
    ratio, error = value['actual_over_predicted'], value['step_relative_error']
    span = f"{ratio['min']:.6f}–{ratio['max']:.6f}" if ratio else 'unresolved'
    maxerror = f"{100*error['max']:.3g}%" if error else 'unresolved'
    lines.append(f"| {key} | {value['qualified']} / {value['attempted']} | "
                 f"{value['distance_resolved']} | {span} | {maxerror} |")
(run / 'refinement-summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
