#!/usr/bin/env python3
"""Independent positive-tail reference for the HS four-block pEBA law.

Each of four repeated weights occurs six times: w * chi-square(6) is an
Erlang(3, rate=1/(2w)). The sum is the absorption time of 12 serial exponential
phases. Its survival is e_1' exp(Q*t) 1, with no oscillatory integration or
subtraction from a CDF. Requires mpmath; this is diagnostic, outside timing.
"""
import csv
import sys
from pathlib import Path
import mpmath as mp


def survival(folder, digits):
    mp.mp.dps = digits
    weights = [mp.mpf(r['weight']) for r in csv.DictReader((folder / 'weights.csv').open())]
    statistic = mp.mpf((folder / 'statistic.txt').read_text().strip())
    assert len(weights) == 24
    rates = []
    for i in range(0, 24, 6):
        assert len(set(weights[i:i + 6])) == 1
        rates.extend([1 / (2 * weights[i])] * 3)
    generator = mp.matrix(12)
    for i, rate in enumerate(rates):
        generator[i, i] = -rate
        if i < 11:
            generator[i, i + 1] = rate
    transition = mp.expm(generator * statistic)
    return sum(transition[0, i] for i in range(12))


folder = Path(sys.argv[1])
p50 = survival(folder, 50)
p70 = survival(folder, 70)
assert abs(p50 / p70 - 1) < mp.mpf('1e-40')
with (folder / 'independent.csv').open('w') as out:
    writer = csv.writer(out)
    writer.writerow(['method', 'digits', 'p_value'])
    writer.writerow(['Erlang-chain matrix exponential', 70, mp.nstr(p70, 60)])
print(mp.nstr(p70, 60))
