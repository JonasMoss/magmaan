"""
Population scan for larger nested-GLS profile/chi-square cutoff gaps.

This is deliberately population-only: for each pseudo-null diagonal
misspecification design it computes the profile-Hessian spectrum and asks what
the ordinary chi-square 0.95 cutoff would do under that profile mixture.
Finite-sample Monte Carlo should be reserved for the largest gaps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess

import numpy as np

import gls_nested_profile_mc as smoke


def pattern(name: str, p: int) -> np.ndarray:
    if name == "alternating":
        v = np.array([(-1) ** k for k in range(p)], dtype=float)
    elif name == "linear":
        v = np.linspace(-1.0, 1.0, p)
    elif name == "half":
        v = np.r_[np.ones(p // 2), -np.ones(p - p // 2)]
    elif name == "spike_hi":
        v = np.r_[p - 1.0, -np.ones(p - 1)]
    elif name == "spike_lo":
        v = np.r_[-(p - 1.0), np.ones(p - 1)]
    else:
        raise ValueError(name)
    v = v - v.mean()
    return v / np.max(np.abs(v))


def weighted_upper(lam: np.ndarray, x: float) -> float:
    pos = smoke.positive_lambdas(lam)
    args = [str(smoke.mixture_cli()), "upper", f"{x:.17g}"]
    args.extend(f"{z:.17g}" for z in pos)
    return float(subprocess.check_output(args, text=True).strip())


def scan_one(p: int, eps: float, pat: str) -> dict[str, float | int | str]:
    base = smoke.load_base()
    Dp = base.dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    Delta_A, Delta_B = base.jacobians(p)
    df_diff = Delta_A.shape[1] - Delta_B.shape[1]

    d = pattern(pat, p)
    Sig0 = np.eye(p) + eps * np.diag(d)
    if np.linalg.eigvalsh(Sig0).min() <= 0.0:
        raise ValueError("non-PD Sig0")

    u0 = base.vech(Sig0, p)
    Gamma = base.gamma_nt(Sig0, Dplus)
    Q_profile = base.score_hessian(Delta_B, u0, p, Dplus) - base.score_hessian(
        Delta_A, u0, p, Dplus
    )
    lam_profile, imag = base.spectrum(Q_profile, Gamma)
    full_lam = smoke.positive_lambdas(lam_profile)
    chi_lam = np.ones(df_diff)
    q_profile = smoke.weighted_chisq_quantile(full_lam, 0.95)
    q_chi = smoke.weighted_chisq_quantile(chi_lam, 0.95)
    profile_tail_at_chi = weighted_upper(full_lam, q_chi)
    chi_tail_at_profile = weighted_upper(chi_lam, q_profile)
    trace = float(full_lam.sum())
    trace_sq = float(np.sum(full_lam * full_lam))
    cv = float(np.sqrt(max(trace_sq / len(full_lam) - (trace / len(full_lam)) ** 2, 0.0)))
    return {
        "p": p,
        "df": df_diff,
        "pattern": pat,
        "eps": eps,
        "min_var": float(np.diag(Sig0).min()),
        "max_var": float(np.diag(Sig0).max()),
        "q95_profile": q_profile,
        "q95_chi": q_chi,
        "q95_gap_chi_minus_profile": q_chi - q_profile,
        "profile_tail_at_chi_q95": profile_tail_at_chi,
        "chi_tail_at_profile_q95": chi_tail_at_profile,
        "tail_gap_at_chi_q95": profile_tail_at_chi - 0.05,
        "trace": trace,
        "trace_over_df": trace / df_diff,
        "lambda_min": float(full_lam.min()),
        "lambda_max": float(full_lam.max()),
        "lambda_sd": cv,
        "imag_max": float(imag),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(max_p: int, eps_values: tuple[float, ...], top: int) -> None:
    pats = ("alternating", "linear", "half", "spike_hi", "spike_lo")
    rows: list[dict[str, float | int | str]] = []
    for p in range(4, max_p + 1):
        for pat in pats:
            for eps in eps_values:
                try:
                    rows.append(scan_one(p, eps, pat))
                except Exception as exc:
                    print(f"skip p={p} pattern={pat} eps={eps}: {exc}")

    rows.sort(key=lambda r: abs(float(r["tail_gap_at_chi_q95"])), reverse=True)
    out = Path(__file__).resolve().parent / "gls_nested_profile_scan.csv"
    write_csv(out, rows)

    print(f"wrote {out}")
    print("largest absolute deviations of profile tail at ordinary chi-square q95")
    print("p  df  pattern       eps   q_prof  q_chi   prof_tail@chi  gap")
    for r in rows[:top]:
        print(
            f"{r['p']:2d} {r['df']:3d}  {r['pattern']:<11s}"
            f" {float(r['eps']):4.2f}"
            f"  {float(r['q95_profile']):7.3f}"
            f" {float(r['q95_chi']):7.3f}"
            f"      {float(r['profile_tail_at_chi_q95']):8.4f}"
            f" {float(r['tail_gap_at_chi_q95']):8.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-p", type=int, default=6)
    parser.add_argument("--eps", type=str, default="0.05,0.15,0.30,0.50,0.70,0.85")
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eps_values = tuple(float(x) for x in args.eps.split(",") if x)
    run(args.max_p, eps_values, args.top)
