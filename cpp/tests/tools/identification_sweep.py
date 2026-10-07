#!/usr/bin/env python3
"""Structural identification sweep over the C++ test corpora (board TASK-33.3).

Every fit route that runs the structural identification check reports it to a
test-build observer when MAGMAAN_IDENTIFICATION_SWEEP is set (see
cpp/tests/doctest_main.cpp). This script runs the optimized suite under
`ctest -V`, attributes each report to its test, separates golden (fixture
corpus) tests from unit tests, and summarizes:

  * counts by status and reason, golden and unit;
  * every golden report that is not Identified, with its test;
  * the calibration margins: the smallest certifying relative singular value
    among Identified reports and the largest null value among Unidentified
    ones (the gap rule's tolerances are 1e-10 and 1e-7);
  * the wall time per check.

From the repo root, after `nice -n 10 just jobs=2 test-opt` built the suite:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    nice -n 10 python3 cpp/tests/tools/identification_sweep.py --jobs 2

`--log FILE` parses an existing `ctest -V` log instead of running ctest.
Logs stay under ~/.cache/magmaan-logs.
"""
import argparse
import collections
import os
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--build', type=Path, default=None,
                    help='CTest build directory (default cpp/build/opt)')
parser.add_argument('--jobs', type=int, default=2)
parser.add_argument('--log', type=Path, default=None,
                    help='parse this ctest -V log instead of running ctest')
parser.add_argument('--regex', default=None, help='ctest -R filter')
args = parser.parse_args()

root = Path(__file__).resolve().parents[3]
build = args.build or root / 'cpp/build/opt'
logs = Path.home() / '.cache/magmaan-logs'
logs.mkdir(parents=True, exist_ok=True)

if args.log is None:
    log = logs / 'identification-sweep-ctest.log'
    env = dict(os.environ, MAGMAAN_IDENTIFICATION_SWEEP='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    cmd = ['ctest', '--test-dir', str(build), '-V', '-j', str(args.jobs)]
    if args.regex:
        cmd += ['-R', args.regex]
    with open(log, 'w') as out:
        status = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
    print(f'ctest exit {status}; log {log}')
else:
    log = args.log

# Golden tests: doctest's source-file filter on every test executable.
golden = set()
for exe in sorted((build / 'tests').glob('magmaan_test_*')):
    if not os.access(exe, os.X_OK) or exe.suffix:
        continue
    listing = subprocess.run([str(exe), '--list-test-cases', '--source-file=*/golden/*'],
                             capture_output=True, text=True).stdout
    for line in listing.splitlines():
        line = line.strip()
        if line and not line.startswith('[doctest]') and not line.startswith('='):
            golden.add(line)

start = re.compile(r'^\s*Start\s+(\d+): (.*)$')
report = re.compile(r'^(\d+): \[identification\] (.*)$')
names, rows = {}, []
for line in open(log, errors='replace'):
    line = line.rstrip('\n')
    m = start.match(line)
    if m:
        names[m.group(1)] = m.group(2)
        continue
    m = report.match(line)
    if m:
        fields = dict(kv.split('=', 1) for kv in m.group(2).split())
        fields['test'] = m.group(1)
        rows.append(fields)

def test_name(row):
    return names.get(row['test'], '#' + row['test'])

def is_golden(row):
    name = test_name(row)
    bare = name.split(': ', 1)[1] if ': ' in name else name
    return bare in golden

def floats(text):
    return [float(x) for x in text.split(',') if x]

summary = collections.Counter()
for row in rows:
    summary[('golden' if is_golden(row) else 'unit', row['status'], row['reason'])] += 1
print(f'{len(rows)} reports from {len({r["test"] for r in rows})} tests')
print('\nscope   status        reason                    count')
for (scope, status, reason), n in sorted(summary.items()):
    print(f'{scope:<7} {status:<13} {reason:<25} {n}')

print('\nGolden reports that are not Identified (test: status/reason map q/moments/rank):')
seen = set()
for row in rows:
    if not is_golden(row) or row['status'] == 'identified':
        continue
    key = (test_name(row), row['status'], row['reason'], row['map'], row['q'], row['rank'])
    if key in seen:
        continue
    seen.add(key)
    print(f"  {test_name(row)}: {row['status']}/{row['reason']} {row['map']} "
          f"q={row['q']} moments={row['moments']} rank={row['rank']}")

identified = [r for r in rows if r['status'] == 'identified' and r['reason'] == 'rank']
certifying = [floats(r['mins'])[-1] for r in identified if r['mins']]
if certifying:
    worst = min(range(len(certifying)), key=lambda i: certifying[i])
    print(f'\nIdentified: smallest certifying relative singular value {certifying[worst]:.3g} '
          f'({test_name(identified[worst])}, q={identified[worst]["q"]})')
    print(f'  points used: ' + ', '.join(f'{k}: {v}' for k, v in sorted(
        collections.Counter(r['points'] for r in identified).items())))
unidentified = [r for r in rows if r['status'] == 'unidentified']
null_values, gaps = [], []
for r in unidentified:
    nullity = int(r['q']) - int(r['rank'])
    smallest = floats(r['smallest'])
    if nullity > 0 and len(smallest) >= nullity:
        null_values.append(max(smallest[:nullity]))
        if len(smallest) > nullity:
            gaps.append(smallest[nullity])
if null_values:
    print(f'Unidentified: largest null relative singular value {max(null_values):.3g}; '
          f'smallest nonnull value above it {min(gaps) if gaps else float("nan"):.3g}')
ambiguous = [r for r in rows if r['reason'] == 'ambiguous_gap']
print(f'Ambiguous gaps: {len(ambiguous)}')
for r in ambiguous:
    print(f"  {test_name(r)}: mins={r['mins']} smallest={r['smallest']}")

seconds = sorted(float(r['seconds']) for r in rows if r['status'] != 'unchecked' or r['reason'] == 'ambiguous_gap')
if seconds:
    def q(p):
        return seconds[min(len(seconds) - 1, int(p * len(seconds)))]
    big = max(rows, key=lambda r: float(r['seconds']))
    print(f'\nSeconds per check: median {q(0.5):.2g}, 95% {q(0.95):.2g}, max {seconds[-1]:.2g} '
          f'(q={big["q"]}, moments={big["moments"]}, {test_name(big)})')
sys.exit(0)
