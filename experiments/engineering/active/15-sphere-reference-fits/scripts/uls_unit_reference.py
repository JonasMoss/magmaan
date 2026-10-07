#!/usr/bin/env python3
"""Independent local ULS reference for this leaf's two-factor model.

Profile the six unrestricted indicator variances analytically: match the
sample diagonal exactly. Refine four loading ratios and three Phi entries
against the fifteen weighted off-diagonal discrepancies. These are the
original mixed-unit ULS discrepancies, not standardized ULS. Positive profile
curvature plus the six diagonal residual directions proves a finite strict
local minimum. No global claim or production audit exemption follows.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import time

import mpmath as mp


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(x):
    return mp.nstr(x, mp.mp.dps)


class ReferenceFailure(ValueError):
    def __init__(self, message, history):
        super().__init__(message)
        self.history = history


def components(x, markers=(1, 4)):
    free_rows = [i for i in range(6) if i not in markers]
    loading = mp.matrix(6, 2)
    loading[markers[0], 0] = loading[markers[1], 1] = 1
    directions = []
    for k, row in enumerate(free_rows):
        factor = 0 if row < 3 else 1
        loading[row, factor] = x[k]
        e = mp.matrix(6, 2)
        e[row, factor] = 1
        directions.append(e)
    phi = mp.matrix([[x[4], x[5]], [x[5], x[6]]])
    phi_directions = [mp.matrix([[1, 0], [0, 0]]), mp.matrix([[0, 1], [1, 0]]), mp.matrix([[0, 0], [0, 1]])]
    derivatives = [e * phi * loading.T + loading * phi * e.T for e in directions]
    derivatives += [loading * p * loading.T for p in phi_directions]
    second = {}
    for j in range(7):
        for i in range(j + 1):
            d = mp.matrix(6)
            if i < 4 and j < 4:
                d = directions[i] * phi * directions[j].T + directions[j] * phi * directions[i].T
            elif i < 4 <= j:
                d = directions[i] * phi_directions[j - 4] * loading.T + loading * phi_directions[j - 4] * directions[i].T
            second[i, j] = d
    return loading, phi, derivatives, second


class Profile:
    def __init__(self, sample):
        self.scales = [mp.mpf(v) for v in (".01", "100", "2", ".3", "10", ".1")]
        self.mixed = sample
        self.sample = mp.matrix([[sample[i, j] / self.scales[i] / self.scales[j] for j in range(6)] for i in range(6)])
        self.pairs = [(i, j) for i in range(6) for j in range(i)]

    def signed_moment_start(self, sign_x, sign_y):
        s = self.sample
        x = [sign_x * s[0, 2] / s[1, 2], sign_x * s[0, 2] / s[0, 1],
             sign_y * s[3, 5] / s[4, 5], sign_y * s[3, 5] / s[3, 4],
             sign_x * s[0, 1] * s[1, 2] / s[0, 2], mp.mpf(0),
             sign_y * s[3, 4] * s[4, 5] / s[3, 5]]
        lx, ly = [x[0], 1, x[1]], [x[2], 1, x[3]]
        weight = lambda i, j: (self.scales[i] * self.scales[j]) ** 2
        x[5] = mp.fsum(weight(i, j) * lx[i] * ly[j - 3] * s[i, j] for i in range(3) for j in range(3, 6)) / mp.fsum(weight(i, j) * (lx[i] * ly[j - 3]) ** 2 for i in range(3) for j in range(3, 6))
        return x

    def evaluate(self, x):
        loading, phi, derivatives, second = components(x)
        covariance = loading * phi * loading.T
        residual = {(i, j): covariance[i, j] - self.sample[i, j] for i, j in self.pairs}
        weight = {(i, j): (self.scales[i] * self.scales[j]) ** 2 for i, j in self.pairs}
        f = mp.fsum(weight[p] * residual[p] ** 2 for p in self.pairs) / 2
        g = mp.matrix([mp.fsum(weight[p] * residual[p] * d[p] for p in self.pairs) for d in derivatives])
        h = mp.matrix(7)
        for j in range(7):
            for i in range(j + 1):
                h[i, j] = h[j, i] = mp.fsum(weight[p] * (derivatives[i][p] * derivatives[j][p] + residual[p] * second[i, j][p]) for p in self.pairs)
        return f, g, h

    def refine(self, start, tolerance=None):
        x = mp.matrix(start)
        trace = []
        tol = mp.power(10, -mp.mp.dps + 15) if tolerance is None else mp.mpf(tolerance)
        for iteration in range(60):
            f, g, h = self.evaluate(x)
            score = max(abs(g[i]) / mp.sqrt(abs(h[i, i])) for i in range(7))
            trace.append(dict(iteration=iteration, objective=number(f), scaled_score=number(score),
                              phi_x=number(x[4]), phi_y=number(x[6])))
            if score <= tol:
                return x, trace
            scale = mp.diag([1 / mp.sqrt(abs(h[i, i])) for i in range(7)])
            least = mp.eigsy(scale * h * scale, eigvals_only=True)[0]
            shifted = h if least > 0 else h + (-least + mp.mpf("1e-8")) * mp.diag([abs(h[i, i]) for i in range(7)])
            step = mp.lu_solve(shifted, -g)
            if (g.T * step)[0] >= 0:
                raise ReferenceFailure("reference start has no descending Newton step", trace)
            alpha = mp.mpf(1)
            for _ in range(100):
                candidate = x + alpha * step
                if self.evaluate(candidate)[0] <= f:
                    x = candidate
                    break
                alpha /= 2
            else:
                raise ReferenceFailure("reference line search failed", trace)
        raise ReferenceFailure("reference did not reach its precision target", trace)

    def full_point(self, x):
        loading, phi, _, _ = components(x)
        diagonal_scale = mp.diag(self.scales)
        raw_loading = diagonal_scale * loading
        marker_scale = mp.diag([raw_loading[0, 0], raw_loading[3, 1]])
        marker_loading = raw_loading * marker_scale ** -1
        marker_phi = marker_scale * phi * marker_scale
        z = mp.matrix([marker_loading[1, 0], marker_loading[2, 0], marker_loading[4, 1], marker_loading[5, 1], marker_phi[0, 0], marker_phi[0, 1], marker_phi[1, 1]])
        loading, phi, derivatives, second = components(z, markers=(0, 3))
        covariance = loading * phi * loading.T
        residual_variances = [self.mixed[i, i] - covariance[i, i] for i in range(6)]
        for i in range(6):
            covariance[i, i] += residual_variances[i]
            e = mp.matrix(6)
            e[i, i] = 1
            derivatives.append(e)
        pairs = [(i, j) for j in range(6) for i in range(j, 6)]
        residual = mp.matrix([covariance[i, j] - self.mixed[i, j] for i, j in pairs])
        jacobian = mp.matrix([[d[i, j] for d in derivatives] for i, j in pairs])
        gradient = jacobian.T * residual
        hessian = jacobian.T * jacobian
        for j in range(7):
            for i in range(j + 1):
                correction = mp.fsum(residual[k] * second[i, j][p] for k, p in enumerate(pairs))
                hessian[i, j] += correction
                if i != j:
                    hessian[j, i] += correction
        gamma = mp.matrix([[self.mixed[i, k] * self.mixed[j, l] + self.mixed[i, l] * self.mixed[j, k]
                            for k, l in pairs] for i, j in pairs])
        # Original-marker regression coordinates [loadings, vx, beta, psi, theta].
        beta = phi[0, 1] / phi[0, 0]
        psi = phi[1, 1] - phi[0, 1] ** 2 / phi[0, 0]
        transform = mp.eye(13)
        transform[4, 4] = 1; transform[4, 5] = 0; transform[4, 6] = 0
        transform[5, 4] = beta; transform[5, 5] = phi[0, 0]; transform[5, 6] = 0
        transform[6, 4] = beta ** 2; transform[6, 5] = 2 * beta * phi[0, 0]; transform[6, 6] = 1
        n = 100
        g = n * transform.T * gradient
        h = n * transform.T * hessian * transform
        # Full nonlinear chart chain rule, including the negligible score terms.
        h[4, 5] += n * (gradient[5] + 2 * beta * gradient[6]); h[5, 4] = h[4, 5]
        h[5, 5] += n * 2 * phi[0, 0] * gradient[6]
        omega = n * transform.T * jacobian.T * gamma * jacobian * transform
        parameters = [("X", "=~", name, loading[i, 0]) for i, name in enumerate(("x1", "x2", "x3"))]
        parameters += [("Y", "=~", name, loading[i + 3, 1]) for i, name in enumerate(("y1", "y2", "y3"))]
        parameters += [("Y", "~", "X", beta), ("X", "~~", "X", phi[0, 0]), ("Y", "~~", "Y", psi)]
        parameters += [(name, "~~", name, residual_variances[i]) for i, name in enumerate(("x1", "x2", "x3", "y1", "y2", "y3"))]
        return covariance, parameters, g, h, omega


def equilibrated(matrix):
    diagonal = mp.diag([1 / mp.sqrt(matrix[i, i]) for i in range(matrix.rows)])
    values = mp.eigsy(diagonal * matrix * diagonal, eigvals_only=True)
    return values[0], values[values.rows - 1] / values[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, choices=(60, 90), default=60)
    args = parser.parse_args()
    mp.mp.dps = args.digits
    started = time.monotonic()
    summaries, parameters, traces, attempts = [], [], [], []
    covariances = read(args.run_dir / "covariances.csv")
    starts = read(args.run_dir / "reference_starts.csv")
    case_ids = sorted({int(row["case_id"]) for row in starts})
    for case_id in case_ids:
        sample = mp.matrix(6)
        for row in covariances:
            if int(row["case_id"]) == case_id:
                sample[int(row["row"]) - 1, int(row["col"]) - 1] = mp.mpf(row["value"])
        # The sample covariance is mathematically symmetric; retain the midpoint
        # of the two saved floating-point entries as the reference input.
        sample = (sample + sample.T) / 2
        start = [mp.mpf(row["value"]) for row in starts if int(row["case_id"]) == case_id]
        model = Profile(sample)
        candidates = [("nls_endpoint", start)]
        qualified = []
        for label, seed in candidates:
            try:
                point, history = model.refine(seed)
                objective, gradient, hessian = model.evaluate(point)
                minimum, _ = equilibrated(hessian)
                status = "finite_local_minimum" if minimum > 0 else "stationary_nonminimum"
                if minimum > 0:
                    qualified.append((objective, point, label))
            except ReferenceFailure as error:
                history = error.history
                status = str(error)
            traces.extend(dict(case_id=case_id, candidate=label, **row) for row in history)
            attempts.append(dict(case_id=case_id, candidate=label, status=status,
                objective=history[-1]["objective"], scaled_score=history[-1]["scaled_score"],
                phi_x=history[-1]["phi_x"], phi_y=history[-1]["phi_y"]))
            if label == "nls_endpoint" and not qualified:
                # Diagnostic alternatives only, after the sample-only fitting
                # grid is complete. Never feed fitted references into its starts.
                candidates.extend((f"signed_{sx}_{sy}", model.signed_moment_start(sx, sy))
                                  for sx, sy in ((1, -1), (-1, 1), (-1, -1)))
        if not qualified:
            summaries.append(dict(case_id=case_id, digits=mp.mp.dps, objective="nan",
                scaled_profile_score="nan", profile_min_eigenvalue="nan", profile_condition="nan",
                hessian_min_eigenvalue="nan", hessian_condition="nan", metric_min_eigenvalue="nan",
                metric_condition="nan", sandwich_distance="nan", sigma_min_eigenvalue="nan",
                finite_local_minimum=False, fixed_guard_pass=False, candidate="unresolved"))
            print(f"ULS reference {case_id}/{len(case_ids)}: unresolved after {len(candidates)} diagnostic starts", flush=True)
            continue
        _, point, selected = min(qualified, key=lambda entry: entry[0])
        objective, gradient, hessian = model.evaluate(point)
        profile_min, profile_condition = equilibrated(hessian)
        covariance, rows, full_gradient, full_hessian, omega = model.full_point(point)
        hessian_min, hessian_condition = equilibrated(full_hessian)
        metric_min, metric_condition = equilibrated(omega)
        distance = mp.sqrt((full_gradient.T * mp.lu_solve(omega, full_gradient))[0])
        summaries.append(dict(case_id=case_id, digits=mp.mp.dps, objective=number(objective),
            scaled_profile_score=number(max(abs(gradient[i]) / mp.sqrt(hessian[i, i]) for i in range(7))),
            profile_min_eigenvalue=number(profile_min), profile_condition=number(profile_condition),
            hessian_min_eigenvalue=number(hessian_min), hessian_condition=number(hessian_condition),
            metric_min_eigenvalue=number(metric_min), metric_condition=number(metric_condition),
            sandwich_distance=number(distance), sigma_min_eigenvalue=number(mp.eigsy(covariance, eigvals_only=True)[0]),
            finite_local_minimum=bool(profile_min > 0 and hessian_min > 0), fixed_guard_pass=bool(max(hessian_condition, metric_condition) <= mp.mpf("1e12")), candidate=selected))
        parameters.extend(dict(case_id=case_id, lhs=a, op=op, rhs=b, est=number(value)) for a, op, b, value in rows)
        print(f"ULS reference {case_id}/{len(case_ids)}: f={mp.nstr(objective, 10)}, metric condition={mp.nstr(metric_condition, 6)}", flush=True)
    write(args.run_dir / "reference_summary.csv", summaries)
    write(args.run_dir / "reference_parameters.csv", parameters)
    write(args.run_dir / "reference_trace.csv", traces)
    write(args.run_dir / "reference_attempts.csv", attempts)
    write(args.run_dir / "reference_metadata.csv", [dict(digits=mp.mp.dps, mpmath=mp.__version__,
        elapsed_s=time.monotonic() - started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=";".join(hashlib.sha256((args.run_dir / name).read_bytes()).hexdigest() for name in ("covariances.csv", "reference_starts.csv")),
        curvature="exact profile/full Hessian; original-marker regression chart; normal-theory sample Gamma",
        scope="finite strict local minima only; fixed production conditioning cap retained")])


if __name__ == "__main__":
    main()
