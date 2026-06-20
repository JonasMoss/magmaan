"""
Numerical check for the relative-and-nested-fit note, Section 6.

Setting: complete-data continuous, normal-theory (GLS) weight W = Gamma_NT(S)^{-1},
linear covariance structure so C = 0. Under a normal DGP, W = Gamma^{-1} exactly,
so the CLASSICAL nested reference law is exactly chi^2_{df_diff} (eigenvalues all
1). Any deviation in the DEFORMED spectrum is therefore purely the estimated-weight
channel W_r (curvature is provably absent here).

Models on p continuous variables (compound-symmetry nested pair):
  B: compound symmetry      Sigma = a*I + b*(J-I)                 (q=2)
  A: common variance, free covariances  Sigma = a*I + sum c_ij E_ij   (q=1+p(p-1)/2)
B is nested in A (all covariances equal). df_diff = p(p-1)/2 - 1.

The deformed nested law is eig[(Mt_B' W Mt_B - Mt_A' W Mt_A) Gamma] with
  Mt = I - Delta (Delta' W Delta)^{-1} Delta' (W - W_r),   W_r col l = (dW/du_l) r*.
No magmaan, no src; pure numpy.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def vech_idx(p):
    return [(i, j) for j in range(p) for i in range(j, p)]  # column-major lower tri


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
    idx = vech_idx(p)
    S = np.zeros((p, p))
    for k, (i, j) in enumerate(idx):
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
        if i != j:  # off-diagonal pair -> its own free covariance
            E = np.zeros((p, p))
            E[i, j] = E[j, i] = 1.0
            cols.append(vech(E, p))
    Delta_A = np.column_stack(cols)
    return Delta_A, Delta_B


def fit_pieces(Delta, W, u0):
    A = Delta.T @ W @ Delta
    Ainv = np.linalg.inv(A)
    P = Delta @ Ainv @ Delta.T @ W
    M = np.eye(len(u0)) - P
    theta = Ainv @ Delta.T @ W @ u0
    rstar = Delta @ theta - u0
    return A, Ainv, M, rstar


def weight_deriv(u0, p, Dplus, h=1e-5):
    """central-difference dW/du_l, W(u)=gamma_nt(unvech(u))^{-1}."""
    m = len(u0)
    dWs = []
    for l in range(m):
        up = u0.copy(); up[l] += h
        um = u0.copy(); um[l] -= h
        Wp = np.linalg.inv(gamma_nt(unvech(up, p), Dplus))
        Wm = np.linalg.inv(gamma_nt(unvech(um, p), Dplus))
        dWs.append((Wp - Wm) / (2 * h))
    return dWs  # list of m matrices, each m x m


def Wr_matrix(dWs, rstar):
    return np.column_stack([dW @ rstar for dW in dWs])


def spectrum(MB, MA, W, Gamma):
    Q = MB.T @ W @ MB - MA.T @ W @ MA
    ev = np.linalg.eigvals(Q @ Gamma)
    return np.sort(ev.real)[::-1], np.abs(ev.imag).max()


def naive_size(lams, df_diff, ndraw=400000, seed_offset=0):
    """actual size of the naive chi^2_{df_diff} test when truth is sum lam_j Z_j^2."""
    from numpy.random import default_rng
    rng = default_rng(12345 + seed_offset)
    pos = lams[np.abs(lams) > 1e-9]
    Z2 = rng.chisquare(1, size=(ndraw, len(pos)))
    stat = Z2 @ pos
    # naive critical value: 0.95 quantile of chi^2_{df_diff}
    crit = rng_chi2_q95(df_diff)
    return np.mean(stat > crit)


def rng_chi2_q95(df):
    # 0.95 quantile of chi^2_df via a fine grid + survival of a large MC, or use
    # scipy if present; fall back to a hardcoded small table.
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(0.95, df))
    except Exception:
        table = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.0705,
                 6: 12.5916, 7: 14.0671, 8: 15.5073}
        return table[df]


def run(p=4, a0=1.0, b0=0.3):
    Dp = dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    Delta_A, Delta_B = jacobians(p)
    df_diff = Delta_A.shape[1] - Delta_B.shape[1]
    m = p * (p + 1) // 2

    CS = a0 * np.eye(p) + b0 * (np.ones((p, p)) - np.eye(p))

    # perturbation directions
    d = np.array([(-1) ** k for k in range(p)], float)
    d -= d.mean()
    D_benign = np.diag(d)                       # unequal variances (orthogonal to A-vs-B)
    D_malig = np.zeros((p, p)); D_malig[0, 1] = D_malig[1, 0] = 1.0  # one covariance

    def analyze(Sig0, label, kind):
        # kind: "null" (correct), "size" (diff-test null TRUE, base misspecified),
        #       "power" (diff-test null FALSE).
        u0 = vech(Sig0, p)
        Gamma = gamma_nt(Sig0, Dplus)
        W = np.linalg.inv(Gamma)                # GLS-at-S under normality: W = Gamma^{-1}
        _, _, M_A, rA = fit_pieces(Delta_A, W, u0)
        _, _, M_B, rB = fit_pieces(Delta_B, W, u0)
        lam_cls, _ = spectrum(M_B, M_A, W, Gamma)
        dWs = weight_deriv(u0, p, Dplus)
        WrA, WrB = Wr_matrix(dWs, rA), Wr_matrix(dWs, rB)
        Mt_A = np.eye(m) - Delta_A @ np.linalg.inv(Delta_A.T @ W @ Delta_A) @ Delta_A.T @ (W - WrA)
        Mt_B = np.eye(m) - Delta_B @ np.linalg.inv(Delta_B.T @ W @ Delta_B) @ Delta_B.T @ (W - WrB)
        lam_def, imag = spectrum(Mt_B, Mt_A, W, Gamma)
        mA = float(np.sqrt(max(rA @ W @ rA, 0.0)))  # weighted misfit of model A
        mB = float(np.sqrt(max(rB @ W @ rB, 0.0)))  # weighted misfit of model B
        k = df_diff
        print(f"\n=== {label}   ||r*_A||_W={mA:.4f}  ||r*_B||_W={mB:.4f} ===")
        print(f" classical top-{k+1}: {lam_cls[:k+1]}")
        print(f" deformed  top-{k+1}: {lam_def[:k+1]}   (max imag {imag:.1e})")
        print(f" reference-law mean: naive df = {k},  deformed sum(lam) = {lam_def.sum():.4f}")
        if kind == "size":
            size = naive_size(lam_def, k)
            tag = "ANTI-CONSERVATIVE" if size > 0.05 else "conservative"
            print(f" naive chi^2_{k} actual Type-I at nominal 0.05: {size:.4f}  ({tag})")
        elif kind == "power":
            print(" (power scenario: diff-test null is FALSE, A fits / B does not;")
            print("  the spectrum deformation also shifts power, full calc needs noncentrality)")
        return lam_def

    print(f"p={p}, df_diff={df_diff}, m={m}, CS=(var {a0}, cov {b0})")
    print("W = Gamma_NT(S)^-1 (efficient under normality), so classical = chi^2_df exactly.")
    analyze(CS, "NULL  (correct model)", "null")
    print("\n--- SIZE: diff-test null TRUE (covariances equal), base misspecified ---")
    for eps in (0.05, 0.15, 0.30):
        analyze(CS + eps * D_benign, f"SIZE: unequal variances, eps={eps}", "size")
    print("\n--- POWER: diff-test null FALSE (one covariance differs; A fits) ---")
    for eps in (0.05, 0.15, 0.30):
        analyze(CS + eps * D_malig, f"POWER: one covariance, eps={eps}", "power")


if __name__ == "__main__":
    run(p=4)
