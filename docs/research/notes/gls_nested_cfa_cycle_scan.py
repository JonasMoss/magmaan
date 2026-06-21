"""
Population scan for a more SEM-like nested GLS profile example.

The data-generating covariance is a tau-equivalent one-factor model plus
unmodeled cyclic local dependence.  All off-diagonal covariances are nonzero.
The fitted pair is:

  A: one-factor congeneric model;
  B: tau-equivalent one-factor model.

By cyclic symmetry, the congeneric pseudo-true loadings remain equal, so the
nested pseudo-null is exact while both models are misspecified for delta > 0.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np

import gls_nested_profile_mc as mix
import gls_nested_profile_scan as scan

np.set_printoptions(precision=4, suppress=True, linewidth=160)


def load_cfa():
    path = Path(__file__).with_name("relative-and-nested-fit-cfa-check.py")
    spec = importlib.util.spec_from_file_location("rel_nested_cfa", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def cycle_residual_cov(p: int, delta: float) -> np.ndarray:
    out = np.zeros((p, p))
    for i in range(p):
        j = (i + 1) % p
        out[i, j] = out[j, i] = delta
    return out


def srmr(Sigma0: np.ndarray, Sigmastar: np.ndarray) -> float:
    p = Sigma0.shape[0]
    vals = []
    for j in range(p):
        for i in range(j, p):
            vals.append(
                (Sigma0[i, j] - Sigmastar[i, j])
                / np.sqrt(Sigma0[i, i] * Sigma0[j, j])
            )
    return float(np.sqrt(np.mean(np.square(vals))))


def correlation_range(Sigma: np.ndarray) -> tuple[float, float]:
    d = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(d, d)
    vals = []
    for j in range(Sigma.shape[0]):
        for i in range(j + 1, Sigma.shape[0]):
            vals.append(R[i, j])
    arr = np.array(vals, dtype=float)
    return float(arr.min()), float(arr.max())


def analyze_one(cfa, p: int, loading: float, delta: float) -> dict[str, float | int]:
    Dp = cfa.dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    residual_var = 1.0 - loading * loading
    loadings = loading * np.ones(p)
    Sigma0 = (
        np.outer(loadings, loadings)
        + np.diag(residual_var * np.ones(p))
        + cycle_residual_cov(p, delta)
    )
    minev = float(np.linalg.eigvalsh(Sigma0).min())
    if minev <= 0.0:
        raise ValueError(f"non-PD Sigma0 for delta={delta}")

    u0 = cfa.vech(Sigma0, p)
    Gamma = cfa.gamma_nt(Sigma0, Dplus)
    theta0_A = np.r_[loadings, residual_var * np.ones(p)]
    theta0_B = np.r_[loading, residual_var * np.ones(p)]

    theta_A = cfa.fit(cfa.sigma_A, theta0_A, u0, p, Dplus)[1]
    theta_B = cfa.fit(cfa.sigma_B, theta0_B, u0, p, Dplus)[1]
    fA = cfa.profile_value(cfa.sigma_A, theta0_A, u0, p, Dplus)
    fB = cfa.profile_value(cfa.sigma_B, theta0_B, u0, p, Dplus)
    score_diff = np.linalg.norm(
        cfa.profile_score(cfa.sigma_B, theta0_B, u0, p, Dplus)
        - cfa.profile_score(cfa.sigma_A, theta0_A, u0, p, Dplus)
    )

    Q_profile = cfa.score_hessian(cfa.sigma_B, theta0_B, u0, p, Dplus) - cfa.score_hessian(
        cfa.sigma_A, theta0_A, u0, p, Dplus
    )
    lambdas, imag = cfa.spectrum(Q_profile, Gamma)
    pos = mix.positive_lambdas(lambdas)
    df_diff = len(theta0_A) - len(theta0_B)
    q_profile = mix.weighted_chisq_quantile(pos, 0.95)
    q_chi = mix.weighted_chisq_quantile(np.ones(df_diff), 0.95)
    profile_tail_at_chi = scan.weighted_upper(pos, q_chi)
    chi_tail_at_profile = scan.weighted_upper(np.ones(df_diff), q_profile)

    m = p * (p + 1) // 2
    df_A = m - len(theta0_A)
    df_B = m - len(theta0_B)
    Sigma_A = cfa.sigma_A(theta_A, p)
    Sigma_B = cfa.sigma_B(theta_B, p)
    cor_min, cor_max = correlation_range(Sigma0)

    return {
        "p": p,
        "loading": loading,
        "delta": delta,
        "min_eigen": minev,
        "cor_min": cor_min,
        "cor_max": cor_max,
        "df_A": df_A,
        "df_B": df_B,
        "df_diff": df_diff,
        "f_A": fA,
        "f_B": fB,
        "f_diff_B_minus_A": fB - fA,
        "score_diff_norm": score_diff,
        "loading_sd_A": float(np.std(theta_A[:p])),
        "residual_sd_A": float(np.std(theta_A[p:])),
        "rmsea_A": float(np.sqrt((2.0 * fA) / df_A)),
        "rmsea_B": float(np.sqrt((2.0 * fB) / df_B)),
        "srmr_A": srmr(Sigma0, Sigma_A),
        "srmr_B": srmr(Sigma0, Sigma_B),
        "lambda_count": int(pos.size),
        "lambda_max": float(pos.max()),
        "lambda_trace": float(pos.sum()),
        "q95_profile": q_profile,
        "q95_chi": q_chi,
        "profile_tail_at_chi_q95": profile_tail_at_chi,
        "chi_tail_at_profile_q95": chi_tail_at_profile,
        "imag_max": float(imag),
    }


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(p: int, loading: float, deltas: tuple[float, ...], out: Path) -> None:
    cfa = load_cfa()
    rows = [analyze_one(cfa, p, loading, delta) for delta in deltas]
    write_csv(out, rows)
    print(f"wrote {out}")
    print("delta  corrange       rmsea_A rmsea_B  SRMR   tail@chi  q_prof  lmax trace")
    for r in rows:
        print(
            f"{r['delta']:5.3f}  [{r['cor_min']:.3f},{r['cor_max']:.3f}]"
            f"    {r['rmsea_A']:.3f}   {r['rmsea_B']:.3f}"
            f"  {r['srmr_A']:.3f}    {r['profile_tail_at_chi_q95']:.3f}"
            f"   {r['q95_profile']:.3f}  {r['lambda_max']:.3f} {r['lambda_trace']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--loading", type=float, default=0.7)
    parser.add_argument("--deltas", type=str, default="0.02,0.04,0.06,0.08,0.10,0.12")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().with_name("gls_nested_cfa_cycle_scan.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    deltas = tuple(float(x) for x in args.deltas.split(",") if x)
    run(args.p, args.loading, deltas, args.out)
