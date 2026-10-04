#!/usr/bin/env python3
"""Optional ASan/UBSan corpus gates; no CI dependency.

From the repo root, with BLAS/OpenMP pinned to one thread:
  nice -n 10 cmake --preset dev -S cpp
  nice -n 10 cmake --build cpp/build/dev --target magmaan -j2
  nice -n 10 python3 cpp/tests/tools/check_mplus_sanitizers.py \
    ~/.cache/magmaan-logs/mplus-input-sweep/manifest.txt

The manifest is produced by check_mplus_input_corpus.R. Each input runs in its
own process with a ten-second deadline. All diagnostics and example paths stay
in the ignored log directory. --preset opt is a nonsanitized diagnostic sweep.
"""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('manifest', type=Path)
parser.add_argument('--preset', choices=['dev', 'opt'], default='dev')
parser.add_argument('--timeout', type=float, default=10)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
logs = Path.home() / '.cache/magmaan-logs' / ('task-56-sweep-' + args.preset)
logs.mkdir(exist_ok=True)
build = root / 'cpp/build' / args.preset
cache = (build / 'CMakeCache.txt').read_text()
compiler = re.search(r'^CMAKE_CXX_COMPILER:[^=]*=(.*)$', cache, re.M)[1]
flags = ['-std=c++23', '-fno-exceptions', '-fno-rtti', '-DEIGEN_NO_EXCEPTIONS=1',
         '-DEIGEN_MAX_ALIGN_BYTES=64', '-DEIGEN_DONT_PARALLELIZE=1',
         '-DEIGEN_NO_AUTOMATIC_RESIZING=1', '-DEIGEN_RUNTIME_NO_MALLOC=1',
         '-DMAGMAAN_WITH_PORT', '-I' + str(root / 'cpp/include'),
         '-I' + str(root / 'cpp/src')]
# Resolve Eigen and dependency includes from this build's actual toolchain.
commands = json.loads((build / 'compile_commands.json').read_text())
command = next(c for c in commands if c['file'].endswith('/parse/mplus_input.cpp'))
tokens = shlex.split(command['command'])
for i, token in enumerate(tokens):
    if token == '-isystem':
        flags += ['-isystem', tokens[i+1]]
    elif token.startswith('-D'):
        flags.append(token)
if args.preset == 'dev':
    flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all']
else:
    flags += ['-O2', '-march=native']
parse_sources = ['mplus_input', 'mplus_parser', 'parser', 'lexer']
with (logs / 'build.log').open('w') as log:
    subprocess.run([compiler, *flags, str(root / 'cpp/tests/tools/mplus_input_sweep.cpp'),
        *[str(root / ('cpp/src/parse/' + name + '.cpp')) for name in parse_sources],
        '-o', str(logs / 'reader_parser')], stdout=log, stderr=log, check=True)
    subprocess.run([compiler, *flags, str(root / 'cpp/tests/tools/mplus_model_sweep.cpp'),
        str(build / 'libmagmaan.a'), str(build / 'libport.a'),
        str(build / 'libquadpack.a'), str(build / '_deps/nlopt-build/libnlopt.a'),
        '-lm', '-o', str(logs / 'lowering')], stdout=log, stderr=log, check=True)
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
# LeakSanitizer cannot run in the ptraced Codex sandbox. ASan memory-access and
# UBSan checks remain active; this is not a suppression of frontend defects.
tallies = {stage: collections.Counter() for stage in ['reader', 'parser', 'lowering']}
examples = {stage: {} for stage in tallies}
failures = collections.Counter()
files = args.manifest.read_text().splitlines()
with (logs / 'results.tsv').open('w') as report:
    writer = csv.writer(report, delimiter='\t')
    writer.writerow(['file', 'stage', 'status', 'detail'])
    for path in files:
        for driver, stages in [('reader_parser', ['reader', 'parser']), ('lowering', ['lowering'])]:
            try:
                run = subprocess.run([str(logs / driver), path], capture_output=True, text=True,
                                     env=env, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                failures['hang'] += 1
                writer.writerow([path, driver, 'hang', 'timeout'])
                continue
            if run.returncode or run.stderr:
                failures['crash_or_sanitizer'] += 1
                writer.writerow([path, driver, 'crash_or_sanitizer', run.stderr])
                continue
            fields = run.stdout.rstrip('\n').split('\t')
            if driver == 'reader_parser':
                results = [(fields[1], fields[3]), (fields[2], fields[4])]
            else:
                results = [(fields[0], fields[1])]
            for stage, (status, detail) in zip(stages, results):
                writer.writerow([path, stage, status, detail])
                ids = set(re.findall(r'\[([A-Z]{2}\d+)\]', detail))
                if status == 'accepted':
                    tallies[stage]['accepted'] += 1
                elif not ids:
                    failures['unclassified'] += 1
                else:
                    tallies[stage]['rejected'] += 1
                    if stage != 'lowering' and not all(part in detail for part in ["found '", 'Mplus', 'magmaan', 'instead']):
                        failures['incomplete_diagnostic'] += 1
                    for rule in sorted(ids):
                        tallies[stage][rule] += 1
                        examples[stage].setdefault(rule, {'file': path, 'detail': detail})
summary = {'files': len(files), 'stages': tallies, 'failures': failures, 'examples': examples}
(logs / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'examples'}, indent=2))
raise SystemExit(bool(failures))
