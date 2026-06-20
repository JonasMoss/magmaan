"""
CFA check for relative-and-nested-fit.tex.

One-factor p=4 model with GLS / normal-theory weight W(u).  The goal is not a
finite-sample size calculation; the examples are generally off the nested null.
The goal is narrower and diagnostic:

  * compute the corrected profile-Hessian quadratic Q_B - Q_A by finite
    differences of the envelope/profile score;
  * verify that it matches finite differences of the profiled value;
  * print the fitted-residual covariance quadratic Mt' W Mt as a separate
    standard-error diagnostic, especially once curvature C != 0.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def vech_idx(p):
    return [(i, j) for j in range(p) for i in range(j, p)]


def dup(p):
    idx = vech_idx(p)
    pos = {(i, j): k for k, (i, j) in enumerate(idx)}
    D = np.zeros((p * p, len(idx)))
    for a in range(p):
        for b in range(p):
            i, j = (a, b) if a >= b else (b, a)
            D[a + b * p, pos[(i, j)]] = 1.0
    return D


def unvech(u, p):
    S = np.zeros((p, p))
    for k, (i, j) in enumerate(vech_idx(p)):
        S[i, j] = S[j, i] = u[k]
    return S


def vech(S, p):
    return np.array([S[i, j] for (i, j) in vech_idx(p)])


def gamma_nt(Sig, Dplus):
    return 2.0 * Dplus @ np.kron(Sig, Sig) @ Dplus.T


def weight(u, p, Dplus):
    return np.linalg.inv(gamma_nt(unvech(u, p), Dplus))


def weight_deriv(u, p, Dplus, h=1e-5):
    out = []
    for k in range(len(u)):
        up = u.copy()
        um = u.copy()
        up[k] += h
        um[k] -= h
        out.append((weight(up, p, Dplus) - weight(um, p, Dplus)) / (2.0 * h))
    return out


def sigma_A(theta, p):
    L = theta[:p]
    res = theta[p:]
    return np.outer(L, L) + np.diag(res)


def sigma_B(theta, p):
    lc = theta[0]
    res = theta[1:]
    L = lc * np.ones(p)
    return np.outer(L, L) + np.diag(res)


def eta(sig_fn, theta, p):
    return vech(sig_fn(theta, p), p)


def jac(sig_fn, theta, p, h=1e-6):
    m, q = p * (p + 1) // 2, len(theta)
    J = np.zeros((m, q))
    for k in range(q):
        tp = theta.copy()
        tm = theta.copy()
        tp[k] += h
        tm[k] -= h
        J[:, k] = (eta(sig_fn, tp, p) - eta(sig_fn, tm, p)) / (2.0 * h)
    return J


def gauss_newton(sig_fn, theta0, u, W, p, iters=200):
    theta = theta0.copy()
    for _ in range(iters):
        J = jac(sig_fn, theta, p)
        d = eta(sig_fn, theta, p) - u
        step = np.linalg.solve(J.T @ W @ J, J.T @ W @ d)
        theta = theta - step
        if np.linalg.norm(step) < 1e-12:
            break
    return theta


def grad(sig_fn, theta, u, W, p):
    J = jac(sig_fn, theta, p)
    return J.T @ W @ (eta(sig_fn, theta, p) - u)


def obs_bread(sig_fn, theta, u, W, p, h=1e-5):
    q = len(theta)
    H = np.zeros((q, q))
    for k in range(q):
        tp = theta.copy()
        tm = theta.copy()
        tp[k] += h
        tm[k] -= h
        H[:, k] = (grad(sig_fn, tp, u, W, p) - grad(sig_fn, tm, u, W, p)) / (2.0 * h)
    return 0.5 * (H + H.T)


def fit(sig_fn, theta0, u, p, Dplus):
    W = weight(u, p, Dplus)
    theta = gauss_newton(sig_fn, theta0, u, W, p)
    r = eta(sig_fn, theta, p) - u
    J = jac(sig_fn, theta, p)
    A = J.T @ W @ J
    H = obs_bread(sig_fn, theta, u, W, p)
    return W, theta, r, J, A, H


def profile_value(sig_fn, theta0, u, p, Dplus):
    W, theta, _, _, _, _ = fit(sig_fn, theta0, u, p, Dplus)
    r = eta(sig_fn, theta, p) - u
    return 0.5 * float(r @ W @ r)


def profile_score(sig_fn, theta0, u, p, Dplus):
    W, _, r, _, _, _ = fit(sig_fn, theta0, u, p, Dplus)
    b = np.array([float(r @ dW @ r) for dW in weight_deriv(u, p, Dplus)])
    return -W @ r + 0.5 * b


def score_hessian(sig_fn, theta0, u, p, Dplus, h=2e-5):
    m = len(u)
    H = np.zeros((m, m))
    for k in range(m):
        up = u.copy()
        um = u.copy()
        up[k] += h
        um[k] -= h
        H[:, k] = (
            profile_score(sig_fn, theta0, up, p, Dplus)
            - profile_score(sig_fn, theta0, um, p, Dplus)
        ) / (2.0 * h)
    return 0.5 * (H + H.T)


def value_hessian_diff(theta0_A, theta0_B, u, p, Dplus, h=2e-4):
    m = len(u)
    H = np.zeros((m, m))

    def fdiff(x):
        return (
            profile_value(sigma_B, theta0_B, x, p, Dplus)
            - profile_value(sigma_A, theta0_A, x, p, Dplus)
        )

    for i in range(m):
        for j in range(m):
            ei = np.zeros(m)
            ej = np.zeros(m)
            ei[i] = 1.0
            ej[j] = 1.0
            H[i, j] = (
                fdiff(u + h * ei + h * ej)
                - fdiff(u + h * ei - h * ej)
                - fdiff(u - h * ei + h * ej)
                + fdiff(u - h * ei - h * ej)
            ) / (4.0 * h * h)
    return 0.5 * (H + H.T)


def residual_q(J, W, bread, Wr):
    Mt = np.eye(W.shape[0]) - J @ np.linalg.inv(bread) @ J.T @ (W - Wr)
    return Mt.T @ W @ Mt


def fixed_profile_q(J, W, bread):
    return W - W @ J @ np.linalg.inv(bread) @ J.T @ W


def spectrum(Q, Gamma):
    ev = np.linalg.eigvals(Q @ Gamma)
    return np.sort(ev.real)[::-1], float(np.abs(ev.imag).max())


def run(p=4):
    Dp = dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    df_diff = 3

    lam0 = np.array([0.8, 0.7, 0.6, 0.5])
    res0 = 1.0 - lam0 ** 2
    theta0_A = np.r_[lam0, res0]
    theta0_B = np.r_[lam0.mean(), res0]

    def analyze(Sig0, label):
        u0 = vech(Sig0, p)
        Gamma = gamma_nt(Sig0, Dplus)
        W_A, _, rA, JA, AA, HA = fit(sigma_A, theta0_A, u0, p, Dplus)
        W_B, _, rB, JB, AB, HB = fit(sigma_B, theta0_B, u0, p, Dplus)
        assert np.abs(W_A - W_B).max() < 1e-12
        W = W_A
        dWs = weight_deriv(u0, p, Dplus)
        WrA = np.column_stack([dW @ rA for dW in dWs])
        WrB = np.column_stack([dW @ rB for dW in dWs])

        Q_profile = (
            score_hessian(sigma_B, theta0_B, u0, p, Dplus)
            - score_hessian(sigma_A, theta0_A, u0, p, Dplus)
        )
        Q_value = value_hessian_diff(theta0_A, theta0_B, u0, p, Dplus)
        q_err = np.abs(Q_profile - Q_value).max()
        if q_err > 5e-4:
            raise AssertionError(f"profile score/value Hessians disagree: {q_err}")

        Q_fixed_gn = fixed_profile_q(JB, W, AB) - fixed_profile_q(JA, W, AA)
        Q_fixed_obs = fixed_profile_q(JB, W, HB) - fixed_profile_q(JA, W, HA)
        Q_resid = residual_q(JB, W, HB, WrB) - residual_q(JA, W, HA, WrA)

        lam_profile, imag = spectrum(Q_profile, Gamma)
        lam_fixed_gn, _ = spectrum(Q_fixed_gn, Gamma)
        lam_fixed_obs, _ = spectrum(Q_fixed_obs, Gamma)
        lam_resid, _ = spectrum(Q_resid, Gamma)

        cnormA = np.linalg.norm(HA - AA) / np.linalg.norm(AA)
        cnormB = np.linalg.norm(HB - AB) / np.linalg.norm(AB)
        k = df_diff
        print(f"\n=== {label}  ||r*_A||_W={np.sqrt(max(rA@W@rA,0)):.4f}"
              f"  ||r*_B||_W={np.sqrt(max(rB@W@rB,0)):.4f} ===")
        print(f" curvature size ||C||/||A||:  A={cnormA:.4f}  B={cnormB:.4f}")
        print(f" score/value Hessian max diff: {q_err:.2e}")
        print(f" fixed-W GN profile   top-{k+1}: {lam_fixed_gn[:k+1]}")
        print(f" fixed-W obs profile  top-{k+1}: {lam_fixed_obs[:k+1]}")
        print(f" estimated-W profile  top-{k+1}: {lam_profile[:k+1]}  sum={lam_profile.sum():.4f}"
              f"  (imag {imag:.0e})")
        print(f" residual-vcov diag   top-{k+1}: {lam_resid[:k+1]}  sum={lam_resid.sum():.4f}")

    print(f"p={p}, df_diff={df_diff}: A congeneric vs B tau-equivalent, GLS weight.")
    base = sigma_A(theta0_A, p)
    E12 = np.zeros((p, p))
    E12[0, 1] = E12[1, 0] = 1.0

    analyze(base, "DGP = congeneric (A correct; B false)")
    for eps in (0.10, 0.20):
        analyze(base + eps * E12,
                f"DGP = congeneric + resid-cov(1,2) eps={eps} (both misspecified)")


if __name__ == "__main__":
    run()
