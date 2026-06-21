"""
Numerical check for relative-and-nested-fit.tex.

Setting: complete-data continuous, normal-theory (GLS) weight
W(u) = Gamma_NT(u)^{-1}, linear covariance structures so C = 0.

The nested test law in the corrected note is the Hessian of the profiled
discrepancy difference:

    2N { f_B(u_hat) - f_A(u_hat) } -> z' (Q_B - Q_A) z,
    Q_M = D_u s_M(u0),

where s_M is the envelope/profile score with respect to the sample moments.  The
residual quadratic Mt' W Mt is printed as a separate standard-error diagnostic;
the profile-value Hessian is the nested-test object.
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


def jacobians(p):
    idx = vech_idx(p)
    I = np.eye(p)
    J = np.ones((p, p))
    vI = vech(I, p)
    vJmI = vech(J - I, p)
    Delta_B = np.column_stack([vI, vJmI])
    cols = [vI]
    for (i, j) in idx:
        if i != j:
            E = np.zeros((p, p))
            E[i, j] = E[j, i] = 1.0
            cols.append(vech(E, p))
    Delta_A = np.column_stack(cols)
    return Delta_A, Delta_B


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


def fit_linear(Delta, u, W):
    A = Delta.T @ W @ Delta
    Ainv = np.linalg.inv(A)
    theta = Ainv @ Delta.T @ W @ u
    r = Delta @ theta - u
    return A, Ainv, theta, r


def profile_value(Delta, u, p, Dplus):
    W = weight(u, p, Dplus)
    _, _, _, r = fit_linear(Delta, u, W)
    return 0.5 * float(r @ W @ r)


def profile_score(Delta, u, p, Dplus):
    W = weight(u, p, Dplus)
    _, _, _, r = fit_linear(Delta, u, W)
    dWs = weight_deriv(u, p, Dplus)
    b = np.array([float(r @ dW @ r) for dW in dWs])
    return -W @ r + 0.5 * b


def score_hessian(Delta, u, p, Dplus, h=2e-5):
    m = len(u)
    H = np.zeros((m, m))
    for k in range(m):
        up = u.copy()
        um = u.copy()
        up[k] += h
        um[k] -= h
        H[:, k] = (
            profile_score(Delta, up, p, Dplus)
            - profile_score(Delta, um, p, Dplus)
        ) / (2.0 * h)
    return 0.5 * (H + H.T)


def value_hessian_diff(Delta_B, Delta_A, u, p, Dplus, h=2e-4):
    m = len(u)
    H = np.zeros((m, m))

    def fdiff(x):
        return profile_value(Delta_B, x, p, Dplus) - profile_value(Delta_A, x, p, Dplus)

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


def fixed_weight_q(Delta, W):
    A = Delta.T @ W @ Delta
    return W - W @ Delta @ np.linalg.inv(A) @ Delta.T @ W


def residual_q(Delta, u, p, Dplus):
    W = weight(u, p, Dplus)
    A, Ainv, _, r = fit_linear(Delta, u, W)
    Wr = np.column_stack([dW @ r for dW in weight_deriv(u, p, Dplus)])
    Mt = np.eye(len(u)) - Delta @ Ainv @ Delta.T @ (W - Wr)
    M = np.eye(len(u)) - Delta @ Ainv @ Delta.T @ W
    R = (Delta.T @ Wr).T @ Ainv @ (Delta.T @ Wr)
    err = np.abs(Mt.T @ W @ Mt - (M.T @ W @ M + R)).max()
    if err > 1e-8:
        raise AssertionError(f"residual identity failed: {err}")
    return Mt.T @ W @ Mt


def spectrum(Q, Gamma):
    ev = np.linalg.eigvals(Q @ Gamma)
    return np.sort(ev.real)[::-1], float(np.abs(ev.imag).max())


def chi2_q95(df):
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(0.95, df))
    except Exception:
        table = {
            1: 3.8415,
            2: 5.9915,
            3: 7.8147,
            4: 9.4877,
            5: 11.0705,
            6: 12.5916,
            7: 14.0671,
            8: 15.5073,
        }
        return table[df]


def plain_chi2_size(lams, df_diff, ndraw=300000, seed_offset=0):
    from numpy.random import default_rng

    rng = default_rng(12345 + seed_offset)
    pos = lams[lams > 1e-9]
    stat = rng.chisquare(1, size=(ndraw, len(pos))) @ pos
    return float(np.mean(stat > chi2_q95(df_diff)))


def run(p=4, a0=1.0, b0=0.0):
    Dp = dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    Delta_A, Delta_B = jacobians(p)
    df_diff = Delta_A.shape[1] - Delta_B.shape[1]
    m = p * (p + 1) // 2

    CS = a0 * np.eye(p) + b0 * (np.ones((p, p)) - np.eye(p))
    d = np.array([(-1) ** k for k in range(p)], float)
    d -= d.mean()
    D_benign = np.diag(d)
    D_malig = np.zeros((p, p))
    D_malig[0, 1] = D_malig[1, 0] = 1.0

    def analyze(Sig0, label, kind):
        u0 = vech(Sig0, p)
        Gamma = gamma_nt(Sig0, Dplus)
        W = np.linalg.inv(Gamma)
        _, _, _, rA = fit_linear(Delta_A, u0, W)
        _, _, _, rB = fit_linear(Delta_B, u0, W)

        Q_profile = (
            score_hessian(Delta_B, u0, p, Dplus)
            - score_hessian(Delta_A, u0, p, Dplus)
        )
        Q_value = value_hessian_diff(Delta_B, Delta_A, u0, p, Dplus)
        q_err = np.abs(Q_profile - Q_value).max()
        if q_err > 2e-4:
            raise AssertionError(f"profile score/value Hessians disagree: {q_err}")

        Q_classic = fixed_weight_q(Delta_B, W) - fixed_weight_q(Delta_A, W)
        Q_resid = residual_q(Delta_B, u0, p, Dplus) - residual_q(Delta_A, u0, p, Dplus)
        lam_profile, imag = spectrum(Q_profile, Gamma)
        lam_classic, _ = spectrum(Q_classic, Gamma)
        lam_resid, _ = spectrum(Q_resid, Gamma)
        f0_diff = profile_value(Delta_B, u0, p, Dplus) - profile_value(Delta_A, u0, p, Dplus)
        score_diff = profile_score(Delta_B, u0, p, Dplus) - profile_score(Delta_A, u0, p, Dplus)

        k = df_diff
        mA = float(np.sqrt(max(rA @ W @ rA, 0.0)))
        mB = float(np.sqrt(max(rB @ W @ rB, 0.0)))
        print(f"\n=== {label}   ||r*_A||_W={mA:.4f}  ||r*_B||_W={mB:.4f} ===")
        print(f" f0 diff: {f0_diff:.3e}   ||score diff||: {np.linalg.norm(score_diff):.3e}")
        print(f" score/value Hessian max diff: {q_err:.2e}")
        print(f" classical fixed-W top-{k+1}: {lam_classic[:k+1]}")
        print(f" profile-Hessian top-{k+1}: {lam_profile[:k+1]}   (max imag {imag:.1e})")
        print(f" residual-vcov   top-{k+1}: {lam_resid[:k+1]}")
        print(f" reference-law mean: chi2 df = {k}, profile sum(lam) = {lam_profile.sum():.4f}")
        if kind == "size":
            size = plain_chi2_size(lam_profile, k)
            tag = "anti-conservative" if size > 0.05 else "conservative"
            print(f" plain chi^2_{k} actual Type-I at nominal 0.05: {size:.4f}  ({tag})")
        elif kind == "power":
            print(" (power scenario: diff-test null is false; full calculation also needs noncentrality)")

    print(f"p={p}, df_diff={df_diff}, m={m}, CS=(var {a0}, cov {b0})")
    print("W = Gamma_NT(S)^-1, model linear so C = 0.")
    analyze(CS, "NULL  (correct model)", "null")
    print("\n--- SIZE: diff-test null true (covariances equal), base misspecified ---")
    for eps in (0.05, 0.15, 0.30):
        analyze(CS + eps * D_benign, f"SIZE: unequal variances, eps={eps}", "size")
    print("\n--- POWER: diff-test null false (one covariance differs; A fits) ---")
    for eps in (0.05, 0.15, 0.30):
        analyze(CS + eps * D_malig, f"POWER: one covariance, eps={eps}", "power")


if __name__ == "__main__":
    run(p=4)
