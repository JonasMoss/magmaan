"""Numerical check for higher_order_discrepancy_misspec.tex.

Validates the central claim of Theorem 1 / Corollary 1 for ULS (fixed weight, so
the weight-influence term beta_W vanishes identically):

    E[F_hat] = F0 + tr(Utilde @ Gamma) / N + o(1/N),

where Utilde = V - V dsigma B^{-1} dsigma' V uses the discrepancy HESSIAN
B = A - K (A = dsigma' V dsigma the Gauss-Newton part, K the residual-curvature
coupling), NOT the naive first-order projector U which uses A.

Decisive test: N * (E[F_hat] - F0) -> tr(Utilde Gamma), and is CLOSER to that
than to the naive tr(U Gamma). Curvature is present under normality, so the
normal case (closed-form Gamma_NT) is the clean test of the new object Utilde.
A nonnormal variant (Monte-Carlo Gamma) confirms the general-Gamma form.

ULS only; a 1-factor model is fit to a 2-factor population (unabsorbable misfit).
The sample covariance is the unbiased div-(N-1) form: the div-N form carries its
own O(1/N) moment bias that contaminates exactly the order under test.
Run: python3 higher_order_discrepancy_check.py
"""

import numpy as np

rng = np.random.default_rng(20260620)
p = 6

# ---- vech / duplication machinery (lower-tri, column-major) -----------------
vech_pairs = [(i, j) for j in range(p) for i in range(j, p)]
ps = len(vech_pairs)  # = 10
idx = {(i, j): k for k, (i, j) in enumerate(vech_pairs)}
idx.update({(j, i): k for (i, j), k in list(idx.items())})


def vech(M):
    return np.array([M[i, j] for (i, j) in vech_pairs])


def unvech(v):
    M = np.zeros((p, p))
    for k, (i, j) in enumerate(vech_pairs):
        M[i, j] = M[j, i] = v[k]
    return M


D = np.zeros((p * p, ps))
for a in range(p):
    for b in range(p):
        D[a + p * b, idx[(a, b)]] = 1.0
Dp = np.linalg.solve(D.T @ D, D.T)  # Moore-Penrose left inverse (D'D)^{-1}D'
V = 0.5 * (D.T @ D)  # ULS weight in vech metric: F = (s-sig)'V(s-sig)


# ---- 1-factor model: theta = [lambda(4), psi(4)], factor var fixed 1 -------
def sigma(theta):
    lam, psi = theta[:p], theta[p:]
    return vech(np.outer(lam, lam) + np.diag(psi))


def jac(theta):
    lam = theta[:p]
    J = np.zeros((ps, 2 * p))
    for k in range(p):  # d Sigma / d lambda_k
        M = np.zeros((p, p))
        M[k, :] += lam
        M[:, k] += lam
        J[:, k] = vech(M)
    for k in range(p):  # d Sigma / d psi_k
        M = np.zeros((p, p))
        M[k, k] = 1.0
        J[:, p + k] = vech(M)
    return J


def hess_tensor():
    # d^2 Sigma / d theta_i d theta_j, as ps-vectors; only lambda-lambda nonzero
    H = np.zeros((ps, 2 * p, 2 * p))
    for k in range(p):
        for l in range(p):
            M = np.zeros((p, p))
            M[k, l] += 1.0
            M[l, k] += 1.0
            H[:, k, l] = vech(M)
    return H


Hten = hess_tensor()


def uls_fit(s, theta0, iters=60, tol=1e-10):
    theta = theta0.copy()
    for _ in range(iters):
        res = s - sigma(theta)
        J = jac(theta)
        grad = -2.0 * J.T @ (V @ res)
        GN = 2.0 * J.T @ V @ J + 1e-10 * np.eye(2 * p)
        step = np.linalg.solve(GN, grad)
        theta = theta - step
        if np.max(np.abs(step)) < tol:
            break
    return theta


def Fval(s, theta):
    r = s - sigma(theta)
    return float(r @ (V @ r))


# ---- true (population) covariance: 2-factor truth, fit 1-factor -------------
# Strong, unabsorbable misspecification so the residual e (hence the curvature
# coupling K) is large enough that tr((Utilde-U)Gamma) clears the O(1/N^2) tail.
Lam2 = np.zeros((p, 2))
Lam2[:3, 0] = [0.80, 0.75, 0.70]   # F1 loads items 0,1,2
Lam2[3:, 1] = [0.80, 0.75, 0.70]   # F2 loads items 3,4,5
Phi = np.array([[1.0, 0.55], [0.55, 1.0]])
common = Lam2 @ Phi @ Lam2.T
psi_true = 1.0 - np.diag(common)   # unit variances
Sigma0 = common + np.diag(psi_true)
s0 = vech(Sigma0)

# pseudo-true ULS fit at the population (start from a 1-factor warm guess)
theta0 = np.concatenate([np.full(p, 0.6), np.full(p, 0.6)])
theta_star = uls_fit(s0, theta0)
e = s0 - sigma(theta_star)
F0 = Fval(s0, theta_star)

# curvature objects at theta_star
J = jac(theta_star)
A = J.T @ V @ J
Ve = V @ e
K = np.einsum("a,aij->ij", Ve, Hten)
B = A - K
U = V - V @ J @ np.linalg.solve(A, J.T @ V)
Utilde = V - V @ J @ np.linalg.solve(B, J.T @ V)

print("=== setup ===")
print(f"df = {ps - 2*p},  F0 = {F0:.6f},  ||e|| = {np.linalg.norm(e):.4f}")
print(f"FOC residual ||J' V e|| = {np.linalg.norm(J.T @ Ve):.2e} (should be ~0)")
print(f"||K|| = {np.linalg.norm(K):.4f}  (curvature-residual coupling)")


def report(Gamma, label, Ns, M):
    trU = float(np.trace(U @ Gamma))
    trUt = float(np.trace(Utilde @ Gamma))
    omega2 = float(4.0 * e @ V @ Gamma @ V @ e)  # CLT variance of sqrt(N)(F-F0)
    print(f"\n=== {label} ===")
    print(f"tr(U Gamma)      (naive, first-order) = {trU:.5f}")
    print(f"tr(Utilde Gamma) (curvature-corrected)= {trUt:.5f}")
    print(f"gap = tr((Utilde-U)Gamma) = {trUt - trU:+.5f},  CLT sd(F)~ {np.sqrt(omega2):.3f}/sqrt(N)")
    L = np.linalg.cholesky(Sigma0)
    for N in Ns:
        Fs = np.empty(M)
        for m in range(M):
            X = draw(N, L)
            S = np.cov(X, rowvar=False, bias=False)
            Fs[m] = Fval(vech(S), uls_fit(vech(S), theta_star))
        emp = Fs.mean() - F0
        se = Fs.std(ddof=1) / np.sqrt(M)
        # N*(E F - F0) should converge to tr(Utilde Gamma)
        print(
            f"N={N:5d} M={M}: N*bias = {N*emp:7.4f} "
            f"(MC se {N*se:.4f})  | pred naive {trU:7.4f}  corrected {trUt:7.4f}"
            f"  -> closer to {'CORRECTED' if abs(N*emp-trUt)<abs(N*emp-trU) else 'naive'}"
        )


# normal data: closed-form Gamma_NT = 2 D^+ (Sigma0 x Sigma0) D^+'
Gamma_NT = 2.0 * Dp @ np.kron(Sigma0, Sigma0) @ Dp.T


def draw_normal(N, L):
    return (L @ rng.standard_normal((p, N))).T


def draw_t(N, L):
    # multivariate-t-ish: scale normal by chi to add kurtosis, df=6
    nu = 6.0
    Z = rng.standard_normal((p, N))
    g = rng.chisquare(nu, size=N) / nu
    Y = Z / np.sqrt(g)  # heavy-tailed, Cov = nu/(nu-2) * I
    Y *= np.sqrt((nu - 2.0) / nu)  # rescale to unit variance
    return (L @ Y).T


if __name__ == "__main__":
    # As N grows, N*(E[F]-F0) must converge to tr(Utilde Gamma) (curvature-
    # corrected), NOT to the naive first-order tr(U Gamma).
    draw = draw_normal
    report(Gamma_NT, "NORMAL data (Gamma_NT closed form)", Ns=[200, 800], M=80000)

    # nonnormal: estimate Gamma by Monte Carlo, then verify the same formula
    # holds with the general (non-normal) Gamma in place of Gamma_NT.
    draw = draw_t
    Lc = np.linalg.cholesky(Sigma0)
    MG, NG = 120000, 2000
    SS = np.empty((MG, ps))
    for m in range(MG):
        SS[m] = vech(np.cov(draw_t(NG, Lc), rowvar=False, bias=False))
    Gamma_mc = NG * np.cov(SS, rowvar=False, bias=False)
    report(Gamma_mc, "NONNORMAL data (t6, Gamma Monte-Carlo)", Ns=[200, 800], M=80000)
