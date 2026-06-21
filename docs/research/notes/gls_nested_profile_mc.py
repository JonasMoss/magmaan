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
from pathlib import Path

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def load_base():
    path = Path(__file__).with_name("relative-and-nested-fit-check.py")
    spec = importlib.util.spec_from_file_location("rel_nested", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def q95(x: np.ndarray) -> float:
    return float(np.quantile(x, 0.95))


def qtiles(x: np.ndarray) -> tuple[float, float, float, float]:
    qs = np.quantile(x, [0.50, 0.90, 0.95, 0.99])
    return tuple(float(q) for q in qs)


def mixture_draws(lam: np.ndarray, ndraw: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = lam[lam > 1e-8]
    return rng.chisquare(1.0, size=(ndraw, pos.size)) @ pos


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


def run(p=4, eps=0.15, N=1200, M=6000, mix_draws_n=300000):
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

    full_lam = lam_profile[lam_profile > 1e-8]
    top_df_lam = np.sort(full_lam)[::-1][:df_diff]
    emp = simulate_stats(base, Sig0, Delta_A, Delta_B, p, Dplus, N, M, seed=20260621)
    mix_full = mixture_draws(full_lam, mix_draws_n, seed=1)
    mix_top = mixture_draws(top_df_lam, mix_draws_n, seed=2)
    mix_chi = np.random.default_rng(3).chisquare(df_diff, size=mix_draws_n)
    mix_classic = mixture_draws(lam_classic, mix_draws_n, seed=4)

    print(f"GLS nested finite-sample smoke: p={p}, eps={eps}, N={N}, M={M}")
    print(f"df_diff={df_diff}, f0_diff={f0_diff:.3e}, ||score diff||={np.linalg.norm(score_diff):.3e}")
    print(f"profile/value Q max diff={np.abs(Q_profile - Q_value).max():.2e}")
    print(f"classic eig top: {lam_classic[:8]}")
    print(f"profile eig top: {lam_profile[:8]}")
    print(f"profile positive count={full_lam.size}, trace={full_lam.sum():.4f}")
    print()
    print("quantiles:          q50      q90      q95      q99")
    for label, vals in [
        ("empirical stat", emp),
        ("profile full ", mix_full),
        ("profile topdf", mix_top),
        ("classic U    ", mix_classic),
        ("chi-square   ", mix_chi),
    ]:
        print(f"{label:14s} {qtiles(vals)[0]:8.4f} {qtiles(vals)[1]:8.4f} {qtiles(vals)[2]:8.4f} {qtiles(vals)[3]:8.4f}")
    print()
    for label, cutoff in [
        ("profile full q95", q95(mix_full)),
        ("profile topdf q95", q95(mix_top)),
        ("classic U q95", q95(mix_classic)),
        ("chi-square q95", q95(mix_chi)),
    ]:
        print(f"empirical tail at {label:17s}: {np.mean(emp > cutoff):.4f}  cutoff={cutoff:.4f}")


if __name__ == "__main__":
    run()
