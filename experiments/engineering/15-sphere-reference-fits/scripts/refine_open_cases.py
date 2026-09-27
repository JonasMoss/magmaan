#!/usr/bin/env python3
"""Independent, high-precision local investigation; original markers stay fixed.

Simple two-factor covariance-only model, diagonal indicator residuals.
Internal Phi coordinates replace the regression by cov(X,Y); preserve var(X)'s
sign during refinement, and translate back to beta/psi for reporting.
No generic nonattainment or global-optimality claim is made.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def trace_product(a, b):
    return mp.fsum(a[i, j] * b[j, i] for i in range(a.rows) for j in range(a.cols))


class Model:
    def __init__(self, sample):
        self.S = sample
        self.logdetS = mp.log(mp.det(sample))
        self.E = []
        for row, col in [(1, 0), (2, 0), (4, 1), (5, 1)]:
            e = mp.matrix(6, 2)
            e[row, col] = 1
            self.E.append(e)
        self.P = [mp.matrix([[1, 0], [0, 0]]), mp.matrix([[0, 1], [1, 0]]), mp.matrix([[0, 0], [0, 1]])]

    def evaluate(self, x, derivatives=False):
        L = mp.matrix([[1, 0], [x[0], 0], [x[1], 0], [0, 1], [0, x[2]], [0, x[3]]])
        Phi = mp.matrix([[x[4], x[5]], [x[5], x[6]]])
        Sigma = L * Phi * L.T + mp.diag(list(x[7:13]))
        chol = mp.cholesky(Sigma)
        W = Sigma ** -1
        objective = (2 * mp.fsum(mp.log(chol[i, i]) for i in range(6)) - self.logdetS + trace_product(W, self.S) - 6) / 2
        if not derivatives:
            return objective, Sigma
        K = W * self.S * W
        G = (W - K) / 2
        D = [e * Phi * L.T + L * Phi * e.T for e in self.E]
        D += [L * p * L.T for p in self.P]
        for i in range(6):
            d = mp.matrix(6)
            d[i, i] = 1
            D.append(d)
        g = mp.matrix([trace_product(G, d) for d in D])
        H = mp.matrix(13)
        for j, d in enumerate(D):
            dG = (-W * d * W + W * d * K + K * d * W) / 2
            for i in range(j + 1):
                h = trace_product(dG, D[i])
                if i < 4 and j < 4:
                    second = self.E[i] * Phi * self.E[j].T + self.E[j] * Phi * self.E[i].T
                    h += trace_product(G, second)
                elif i < 4 and j < 7:
                    second = self.E[i] * self.P[j - 4] * L.T + L * self.P[j - 4] * self.E[i].T
                    h += trace_product(G, second)
                H[i, j] = H[j, i] = h
        return objective, Sigma, g, H

    def components(self, x):
        energy_x = 1 / self.S[0, 0] + x[0] ** 2 / self.S[1, 1] + x[1] ** 2 / self.S[2, 2]
        energy_y = 1 / self.S[3, 3] + x[2] ** 2 / self.S[4, 4] + x[3] ** 2 / self.S[5, 5]
        beta = x[5] / x[4]
        psi = x[6] - x[5] ** 2 / x[4]
        extent = max(abs(x[4] * energy_x), abs(x[6] * energy_y), abs(x[5]) * mp.sqrt(energy_x * energy_y),
                     abs(psi * energy_y), abs(beta) * mp.sqrt(energy_y / energy_x),
                     *(abs(x[7+i]) / self.S[i, i] for i in range(6)))
        return extent, beta, psi


def refine(model, start, maximum, identity, fixed=None):
    x = mp.matrix(start)
    sign = mp.sign(x[4])
    active = [i for i in range(13) if i != fixed]
    history = []
    initial_F, initial_Sigma = model.evaluate(x)
    previous_Sigma = initial_Sigma
    status = 'iteration_limit'
    for iteration in range(maximum + 1):
        f, Sigma, g, H = model.evaluate(x, True)
        ha = mp.matrix([[H[i,j] for j in active] for i in active])
        ga = mp.matrix([g[i] for i in active])
        scale = mp.diag([1 / mp.sqrt(max(abs(H[i, i]), mp.mpf('1e-60'))) for i in active])
        hs = scale * ha * scale
        gs = scale * ga
        grad = max(abs(v) for v in gs)
        extent, beta, psi = model.components(x)
        drift = max(abs(Sigma[i, j] - previous_Sigma[i, j]) / mp.sqrt(model.S[i, i] * model.S[j, j]) for i in range(6) for j in range(6))
        try:
            mp.cholesky(hs)
            positive = True
        except ValueError:
            positive = False
        history.append(dict(start_id=identity, iteration=iteration, objective=mp.nstr(f, 30), scaled_gradient=mp.nstr(grad, 12),
                            hessian_positive=positive, component_extent=mp.nstr(extent, 18), var_X=mp.nstr(x[4], 18),
                            beta=mp.nstr(beta, 18), disturbance_Y=mp.nstr(psi, 18), covariance_drift=mp.nstr(drift, 12)))
        if positive and grad < mp.mpf('1e-28'):
            status = 'finite_stationary_positive_curvature'
            break
        if iteration == maximum:
            break
        if positive:
            system = hs
        else:
            eigenvalues = mp.eigsy(hs, eigvals_only=True)
            ridge = max(mp.mpf('1e-6'), -eigenvalues[0] + mp.mpf('1e-6'))
            system = hs + ridge * mp.eye(len(active))
        step_active = scale * mp.lu_solve(system, -gs)
        step = mp.matrix(13,1)
        for i, value in zip(active, step_active):
            step[i] = value
        slope = mp.fdot(g, step)
        alpha = mp.mpf(1)
        accepted = False
        for _ in range(100):
            proposal = x + alpha * step
            if mp.sign(proposal[4]) == sign:
                try:
                    value, _ = model.evaluate(proposal)
                    if value <= f + mp.mpf('1e-4') * alpha * slope:
                        accepted = True
                        break
                except (ValueError, ZeroDivisionError):
                    pass
            alpha /= 2
        if not accepted:
            status = 'line_search_stalled'
            break
        previous_Sigma = Sigma
        x = proposal
    f, Sigma, g, H = model.evaluate(x, True)
    ha = mp.matrix([[H[i,j] for j in active] for i in active])
    ga = mp.matrix([g[i] for i in active])
    scale = mp.diag([1 / mp.sqrt(max(abs(H[i, i]), mp.mpf('1e-60'))) for i in active])
    eigenvalues = mp.eigsy(scale * ha * scale, eigvals_only=True)
    extent, beta, psi = model.components(x)
    result = dict(status=status, iterations=iteration, initial_objective=mp.nstr(initial_F, 30), objective=mp.nstr(f, 30),
                  objective_improvement=mp.nstr(initial_F-f, 20), scaled_gradient=mp.nstr(max(abs(v) for v in scale*ga), 12),
                  min_scaled_hessian_eigen=mp.nstr(eigenvalues[0], 16), component_extent=mp.nstr(extent, 18),
                  var_X=mp.nstr(x[4], 18), beta=mp.nstr(beta, 18), disturbance_Y=mp.nstr(psi, 18),
                  max_loading=mp.nstr(max(1, *(abs(v) for v in x[:4])), 18),
                  total_covariance_change=mp.nstr(max(abs(Sigma[i,j]-initial_Sigma[i,j])/mp.sqrt(model.S[i,i]*model.S[j,j]) for i in range(6) for j in range(6)), 16))
    return result, history, x, Sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, default=60)
    parser.add_argument('--max-iterations', type=int, default=60)
    parser.add_argument('--case', type=int)
    parser.add_argument('--run-id', default='open-case-refinement')
    args = parser.parse_args()
    mp.mp.dps = args.digits
    inputs = ROOT / 'results' / 'open-case-inputs'
    cases = read(inputs / 'cases.csv')
    moments = read(inputs / 'moments.csv')
    starts = read(inputs / 'starts.csv')
    out = ROOT / 'results' / args.run_id
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('invalid run id')
    if (out / 'summary.csv').exists():
        raise ValueError('choose a fresh run id')
    out.mkdir(parents=True, exist_ok=True)
    summary, histories, parameters, covariances = [], [], [], []
    begin = time.monotonic()
    for case in cases:
        if args.case and int(case['case_id']) != args.case:
            continue
        S = mp.matrix(6)
        for row in moments:
            if row['case_id'] == case['case_id']:
                S[int(row['row'])-1, int(row['column'])-1] = mp.mpf(row['value'])
        x = [mp.mpf(row['value']) for row in starts if row['start_id'] == case['start_id']]
        model = Model(S)
        initial_F = model.evaluate(mp.matrix(x))[0]
        assert abs(initial_F - mp.mpf(case['library_objective'])) < mp.mpf('1e-8')
        result, history, theta, Sigma = refine(model, x, args.max_iterations, case['start_id'])
        summary.append(dict(**case, **result))
        histories.extend(history)
        parameters.extend(dict(start_id=case['start_id'], parameter=j+1, value=mp.nstr(v, args.digits)) for j, v in enumerate(theta))
        covariances.extend(dict(start_id=case['start_id'], row=i+1, column=j+1, value=mp.nstr(Sigma[i,j], args.digits)) for i in range(6) for j in range(6))
        print(f"case {case['case_id']} {case['source']}: {result['status']}, {result['iterations']} steps, F={float(result['objective']):.12g}, extent={float(result['component_extent']):.5g}; {time.monotonic()-begin:.1f}s", flush=True)
        write(out / 'progress.csv', summary)
    write(out / 'summary.csv', summary)
    write(out / 'trace.csv', histories)
    write(out / 'parameters.csv', parameters)
    write(out / 'covariances.csv', covariances)
    (out / 'metadata.json').write_text(json.dumps(dict(digits=args.digits, max_iterations=args.max_iterations, mpmath=mp.__version__,
        identification='original markers x1/y1 fixed at one', objective='half Gaussian covariance ML discrepancy, unrestricted primitive variances',
        restriction='retain starting sign of var(X) to avoid crossing the singular SEM regression map',
        interpretation='local high-precision diagnostic only; no global-optimality or nonattainment proof'), indent=2)+'\n')


if __name__ == '__main__':
    main()
