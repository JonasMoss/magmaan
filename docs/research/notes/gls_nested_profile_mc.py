"""
Finite-sample smoke check for the nested GLS profile-Hessian spectrum.

This uses the linear covariance example from relative-and-nested-fit-check.py:

  B: compound symmetry, one variance and one covariance
  A: common variance with free covariances

The size scenario keeps the nested restriction true at the pseudo-true level
(zero covariances) while making both models misspecified through unequal
variances.  The population profile Hessian Q_B - Q_A is then the regular nested
reference law.  Under estimated normal-theory GLS weights, the eigenvalues are
no longer all one, so the smoke compares:

  * empirical finite-sample 2N{f_B(S) - f_A(S)}
  * full profile-Hessian mixture
  * top-df profile-Hessian mixture
  * ordinary chi-square_df
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def load_base():
    path = Path(__file__).with_name("relative-and-nested-fit-check.py")
    spec = importlib.util.spec_from_file_location("rel_nested", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def qtiles(x: np.ndarray) -> tuple[float, float, float, float]:
    qs = np.quantile(x, [0.50, 0.90, 0.95, 0.99])
    return tuple(float(q) for q in qs)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def mixture_cli() -> Path:
    root = repo_root()
    src = Path(__file__).with_name("weighted_chisq_quantile_cli.cpp")
    out = root / "build" / "fast" / "weighted_chisq_quantile_cli"
    lib = root / "build" / "fast" / "libmagmaan.a"
    quadpack = root / "build" / "fast" / "libquadpack.a"
    eigen = root / "build" / "fast" / "_deps" / "eigen3-src"
    if (
        out.exists()
        and out.stat().st_mtime >= src.stat().st_mtime
        and out.stat().st_mtime >= lib.stat().st_mtime
    ):
        return out
    if not lib.exists() or not quadpack.exists() or not eigen.exists():
        raise RuntimeError("build/fast libraries not found; run `just fast` first")
    cxx = os.environ.get("CXX", "c++")
    cmd = [
        cxx,
        "-std=c++23",
        "-O2",
        "-DNDEBUG",
        "-fno-exceptions",
        "-fno-rtti",
        "-DEIGEN_NO_EXCEPTIONS",
        f"-I{root / 'include'}",
        f"-I{eigen}",
        str(src),
        str(lib),
        str(quadpack),
        "-lm",
        "-o",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def positive_lambdas(lam: np.ndarray) -> np.ndarray:
    return np.array([x for x in lam if x > 1e-8], dtype=float)


def weighted_chisq_quantile(lam: np.ndarray, prob: float) -> float:
    pos = positive_lambdas(lam)
    if pos.size == 0:
        return 0.0
    args = [str(mixture_cli()), "quantile", f"{prob:.17g}"]
    args.extend(f"{x:.17g}" for x in pos)
    out = subprocess.check_output(args, text=True).strip().split()
    return float(out[0])


def weighted_chisq_qtiles(lam: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(weighted_chisq_quantile(lam, p) for p in (0.50, 0.90, 0.95, 0.99))


def simulate_stats(base, Sig0, Delta_A, Delta_B, p, Dplus, N, M, seed):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sig0)
    out = np.empty(M)
    for i in range(M):
        Y = rng.standard_normal((N, p)) @ L.T
        S = (Y.T @ Y) / N
        u = base.vech(S, p)
        out[i] = 2.0 * N * (
            base.profile_value(Delta_B, u, p, Dplus)
            - base.profile_value(Delta_A, u, p, Dplus)
        )
    return out


def run(p=4, eps=0.15, Ns=(150, 300, 600, 1200, 3000), M=6000):
    base = load_base()
    Dp = base.dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    Delta_A, Delta_B = base.jacobians(p)
    df_diff = Delta_A.shape[1] - Delta_B.shape[1]

    a0, b0 = 1.0, 0.0
    CS = a0 * np.eye(p) + b0 * (np.ones((p, p)) - np.eye(p))
    d = np.array([(-1) ** k for k in range(p)], float)
    d -= d.mean()
    Sig0 = CS + eps * np.diag(d)
    u0 = base.vech(Sig0, p)
    Gamma = base.gamma_nt(Sig0, Dplus)
    W = np.linalg.inv(Gamma)

    f0_diff = base.profile_value(Delta_B, u0, p, Dplus) - base.profile_value(
        Delta_A, u0, p, Dplus
    )
    score_diff = base.profile_score(Delta_B, u0, p, Dplus) - base.profile_score(
        Delta_A, u0, p, Dplus
    )
    Q_profile = base.score_hessian(Delta_B, u0, p, Dplus) - base.score_hessian(
        Delta_A, u0, p, Dplus
    )
    Q_value = base.value_hessian_diff(Delta_B, Delta_A, u0, p, Dplus)
    Q_classic = base.fixed_weight_q(Delta_B, W) - base.fixed_weight_q(Delta_A, W)

    lam_profile, _ = base.spectrum(Q_profile, Gamma)
    lam_value, _ = base.spectrum(Q_value, Gamma)
    lam_classic, _ = base.spectrum(Q_classic, Gamma)

    full_lam = positive_lambdas(lam_profile)
    top_df_lam = np.sort(full_lam)[::-1][:df_diff]
    classic_lam = positive_lambdas(lam_classic)
    chi_lam = np.ones(df_diff)
    q_full = weighted_chisq_qtiles(full_lam)
    q_top = weighted_chisq_qtiles(top_df_lam)
    q_classic = weighted_chisq_qtiles(classic_lam)
    q_chi = weighted_chisq_qtiles(chi_lam)

    print(f"GLS nested finite-sample smoke: p={p}, eps={eps}, Ns={list(Ns)}, M={M}")
    print(f"df_diff={df_diff}, f0_diff={f0_diff:.3e}, ||score diff||={np.linalg.norm(score_diff):.3e}")
    print(f"profile/value Q max diff={np.abs(Q_profile - Q_value).max():.2e}")
    print(f"classic eig top: {lam_classic[:8]}")
    print(f"profile eig top: {lam_profile[:8]}")
    print(f"profile positive count={full_lam.size}, trace={full_lam.sum():.4f}")
    print()
    print("quantiles:          q50      q90      q95      q99")
    for label, vals in [
        ("profile full ", q_full),
        ("profile topdf", q_top),
        ("classic U    ", q_classic),
        ("chi-square   ", q_chi),
    ]:
        print(f"{label:14s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f}")
    print()
    print("finite-sample calibration by N")
    print("N        emp_q50  emp_q90  emp_q95  emp_q99  tail_prof95  tail_chi95")
    for idx, N in enumerate(Ns):
        emp = simulate_stats(
            base, Sig0, Delta_A, Delta_B, p, Dplus, N, M, seed=20260621 + idx
        )
        eq = qtiles(emp)
        tail_prof = np.mean(emp > q_full[2])
        tail_chi = np.mean(emp > q_chi[2])
        print(
            f"{N:5d}   {eq[0]:8.4f} {eq[1]:8.4f} {eq[2]:8.4f} {eq[3]:8.4f}"
            f"     {tail_prof:8.4f}    {tail_chi:8.4f}"
        )


if __name__ == "__main__":
    run()
