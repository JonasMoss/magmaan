"""
CFA (C != 0) numerical check for the relative-and-nested-fit note, Section 6.5.

One-factor model on p=4 indicators, GLS / normal-theory weight (regime-a, so the
matrix annihilator Mt is exact). Nested pair:
  A: congeneric (free loadings)            theta = [l0..l3, r0..r3]   (q=8, df=2)
  B: tau-equivalent (equal loadings)       theta = [lc, r0..r3]       (q=5, df=5)
df_diff = 3.

Headline: with curvature (C != 0) the clean linear-case identity
  Mt' W Mt = U + R            (Proposition: closed-form deformation, C=0)
must BREAK, because the cancellation used A = Delta' W Delta and the bread is now
H = A + C. We confirm the breakdown and split the deformation into its curvature
channel (W_r = 0, bread H) and its weight channel (C = 0, bread A).

No scipy: hand-rolled Gauss-Newton, chi^2 quantile from a small table.
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


# ----- one-factor model-implied moments -------------------------------------
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
        tp = theta.copy(); tp[k] += h
        tm = theta.copy(); tm[k] -= h
        J[:, k] = (eta(sig_fn, tp, p) - eta(sig_fn, tm, p)) / (2 * h)
    return J


def gauss_newton(sig_fn, theta0, u0, W, p, iters=200):
    theta = theta0.copy()
    for _ in range(iters):
        J = jac(sig_fn, theta, p)
        d = eta(sig_fn, theta, p) - u0
        step = np.linalg.solve(J.T @ W @ J, J.T @ W @ d)
        theta = theta - step
        if np.linalg.norm(step) < 1e-12:
            break
    return theta


def grad(sig_fn, theta, u0, W, p):
    J = jac(sig_fn, theta, p)
    return J.T @ W @ (eta(sig_fn, theta, p) - u0)


def obs_bread(sig_fn, theta, u0, W, p, h=1e-5):
    q = len(theta)
    H = np.zeros((q, q))
    for k in range(q):
        tp = theta.copy(); tp[k] += h
        tm = theta.copy(); tm[k] -= h
        H[:, k] = (grad(sig_fn, tp, u0, W, p) - grad(sig_fn, tm, u0, W, p)) / (2 * h)
    return 0.5 * (H + H.T)


def weight_deriv(u0, p, Dplus, h=1e-5):
    m = len(u0)
    out = []
    for l in range(m):
        up = u0.copy(); up[l] += h
        um = u0.copy(); um[l] -= h
        Wp = np.linalg.inv(gamma_nt(unvech(up, p), Dplus))
        Wm = np.linalg.inv(gamma_nt(unvech(um, p), Dplus))
        out.append((Wp - Wm) / (2 * h))
    return out


def Wr_matrix(dWs, rstar):
    return np.column_stack([dW @ rstar for dW in dWs])


def annih(Delta, W, bread, Wr):
    m = W.shape[0]
    return np.eye(m) - Delta @ np.linalg.inv(bread) @ Delta.T @ (W - Wr)


def spectrum(QB, QA, Gamma):
    ev = np.linalg.eigvals((QB - QA) @ Gamma)
    return np.sort(ev.real)[::-1], np.abs(ev.imag).max()


def chi2_q95(df):
    return {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.0705,
            6: 12.5916, 7: 14.0671, 8: 15.5073}[df]


def naive_size(lams, df, ndraw=400000):
    from numpy.random import default_rng
    rng = default_rng(7)
    pos = lams[np.abs(lams) > 1e-9]
    stat = rng.chisquare(1, size=(ndraw, len(pos))) @ pos
    return float(np.mean(stat > chi2_q95(df)))


def run(p=4):
    Dp = dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    m = p * (p + 1) // 2
    df_diff = 3

    lam0 = np.array([0.8, 0.7, 0.6, 0.5])           # congeneric, unequal loadings
    res0 = 1.0 - lam0 ** 2                           # unit indicator variances

    def analyze(Sig0, label):
        u0 = vech(Sig0, p)
        Gamma = gamma_nt(Sig0, Dplus)
        W = np.linalg.inv(Gamma)                     # GLS-at-S under normality
        # fits
        tA = gauss_newton(sigma_A, np.r_[lam0, res0], u0, W, p)
        tB = gauss_newton(sigma_B, np.r_[lam0.mean(), res0], u0, W, p)
        JA, JB = jac(sigma_A, tA, p), jac(sigma_B, tB, p)
        rA = eta(sigma_A, tA, p) - u0
        rB = eta(sigma_B, tB, p) - u0
        AA, AB = JA.T @ W @ JA, JB.T @ W @ JB        # Gauss-Newton bread
        HA = obs_bread(sigma_A, tA, u0, W, p)        # observed bread = A + C
        HB = obs_bread(sigma_B, tB, u0, W, p)
        cnormA = np.linalg.norm(HA - AA) / np.linalg.norm(AA)
        cnormB = np.linalg.norm(HB - AB) / np.linalg.norm(AB)
        dWs = weight_deriv(u0, p, Dplus)
        WrA, WrB = Wr_matrix(dWs, rA), Wr_matrix(dWs, rB)
        Z = np.zeros((m, m))

        def Q(Delta, bread, Wr):
            Mt = annih(Delta, W, bread, Wr)
            return Mt.T @ W @ Mt

        # classical: Gauss-Newton bread, no weight channel
        lam_cls, _ = spectrum(Q(JB, AB, Z), Q(JA, AA, Z), Gamma)
        # curvature channel only (observed bread, W_r = 0)
        lam_cur, _ = spectrum(Q(JB, HB, Z), Q(JA, HA, Z), Gamma)
        # weight channel only (Gauss-Newton bread, W_r on)
        lam_wt, _ = spectrum(Q(JB, AB, WrB), Q(JA, AA, WrA), Gamma)
        # full deformation
        lam_full, imag = spectrum(Q(JB, HB, WrB), Q(JA, HA, WrA), Gamma)

        # does the C=0 identity Mt'WMt = U + R survive?  (only when C = 0)
        def ident_break(Delta, A_gn, H, Wr):
            Mt = annih(Delta, W, H, Wr)
            Ai = np.linalg.inv(A_gn)
            Mgn = np.eye(m) - Delta @ Ai @ Delta.T @ W
            U = Mgn.T @ W @ Mgn
            R = (Delta.T @ Wr).T @ Ai @ (Delta.T @ Wr)
            return np.abs(Mt.T @ W @ Mt - (U + R)).max()
        ident_A = ident_break(JA, AA, HA, WrA)   # congeneric: C_A != 0 -> breaks
        ident_B = ident_break(JB, AB, HB, WrB)   # tau-equiv: C_B = 0 -> holds

        k = df_diff
        print(f"\n=== {label}  ||r*_A||_W={np.sqrt(max(rA@W@rA,0)):.4f}"
              f"  ||r*_B||_W={np.sqrt(max(rB@W@rB,0)):.4f} ===")
        print(f" curvature size ||C||/||A||:  A={cnormA:.4f}  B={cnormB:.4f}")
        print(f" classical (naive)     top-{k+1}: {lam_cls[:k+1]}")
        print(f" curvature-only        top-{k+1}: {lam_cur[:k+1]}   sum={lam_cur.sum():.4f}")
        print(f" weight-only           top-{k+1}: {lam_wt[:k+1]}   sum={lam_wt.sum():.4f}")
        print(f" full (curv + weight)  top-{k+1}: {lam_full[:k+1]}  sum={lam_full.sum():.4f}"
              f"  (imag {imag:.0e})")
        print(f" naive chi^2_{k} actual size: classical {naive_size(lam_cls,k):.4f}"
              f" -> full {naive_size(lam_full,k):.4f}")
        print(f" C=0 identity residual |Mt'WMt-(U+R)|:"
              f"  A(C!=0)={ident_A:.2e} ({'breaks' if ident_A > 1e-6 else 'holds'})"
              f"   B(C=0)={ident_B:.2e} ({'breaks' if ident_B > 1e-6 else 'holds'})")

    print(f"p={p}, df_diff={df_diff}: A congeneric vs B tau-equivalent, GLS weight.")
    base = sigma_A(np.r_[lam0, res0], p)            # exact congeneric covariance
    E12 = np.zeros((p, p)); E12[0, 1] = E12[1, 0] = 1.0

    # (i) congeneric DGP: A correct (r*_A=0); B's curvature is FOC-killed (tau-equiv),
    #     so this is a pure weight-channel case and the C=0 identity still holds.
    analyze(base, "DGP = congeneric (A correct; B curvature FOC-killed)")

    # (ii) congeneric + omitted residual covariance: BOTH models misspecified, the
    #      congeneric model's free-loading curvature is exercised, so C != 0.
    for eps in (0.10, 0.20):
        analyze(base + eps * E12,
                f"DGP = congeneric + resid-cov(1,2) eps={eps} (both misspecified, C!=0)")


if __name__ == "__main__":
    run()
