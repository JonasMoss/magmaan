"""Numerical checks for higher_order_discrepancy_misspec.tex.

Central object: the curvature-corrected residual-weight Utilde, the O(1/n) bias
coefficient of the fitted discrepancy under fixed misspecification,

    E[F_hat] = F0 + tr(Utilde @ Gamma) / N + o(1/N).

Two complementary checks.

(1) DETERMINISTIC (exact, no Monte Carlo). Utilde is exactly half the Hessian in
    the moment argument of the *profiled* discrepancy s -> min_theta F(s,theta),
    evaluated at the population s0:  Utilde = (1/2) d^2/ds^2 [profiled F]. We
    finite-difference that Hessian and compare to the closed forms:
      ULS:  Utilde = V - V dsig B^{-1} dsig' V          (single metric V = 0.5 D'D)
      ML:   Utilde = V0 - W* dsig B^{-1} dsig' W*        (TWO metrics: noise in the
            data metric V0 = 0.5 D'(S0^-1 (x) S0^-1)D, projection in the model
            metric W* = 0.5 D'(Sig*^-1 (x) Sig*^-1)D)
    with B = (1/2) d^2/dtheta^2 F the discrepancy HESSIAN (not Gauss-Newton A).
    This validates the formula for the slow/Heywood-prone ML fit without MC.

(2) MONTE CARLO (ULS, where the weight is fixed so beta_W = 0). Confirms the
    Hessian-to-bias step empirically: N*(E[F]-F0) -> tr(Utilde Gamma), beating
    the naive tr(U Gamma) (Gauss-Newton A), under normal and t6 data. The sample
    covariance is the unbiased div-(N-1) form; div-N carries its own O(1/N)
    moment bias that contaminates this order.

1-factor model fit to a 2-factor population (unabsorbable misfit).
Run: python3 higher_order_discrepancy_check.py
"""

import numpy as np

rng = np.random.default_rng(20260620)
p = 6

# ---- vech / duplication machinery (lower-tri, column-major) -----------------
vech_pairs = [(i, j) for j in range(p) for i in range(j, p)]
ps = len(vech_pairs)
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
Dp = np.linalg.solve(D.T @ D, D.T)
V = 0.5 * (D.T @ D)  # ULS weight: F_uls = (s-sig)'V(s-sig)


def Wmat(Sig):  # ML/GLS weight at a covariance: 0.5 D'(Sig^-1 (x) Sig^-1) D
    Pi = np.linalg.inv(Sig)
    return 0.5 * D.T @ np.kron(Pi, Pi) @ D


# ---- 1-factor model: theta = [lambda(p), psi(p)], factor variance fixed 1 ---
def sigma(theta):
    lam, psi = theta[:p], theta[p:]
    return vech(np.outer(lam, lam) + np.diag(psi))


def jac(theta):
    lam = theta[:p]
    J = np.zeros((ps, 2 * p))
    for k in range(p):
        M = np.zeros((p, p))
        M[k, :] += lam
        M[:, k] += lam
        J[:, k] = vech(M)
    for k in range(p):
        M = np.zeros((p, p))
        M[k, k] = 1.0
        J[:, p + k] = vech(M)
    return J


def hess_tensor():  # d^2 sigma / dtheta_i dtheta_j; only lambda-lambda nonzero
    H = np.zeros((ps, 2 * p, 2 * p))
    for k in range(p):
        for l in range(p):
            M = np.zeros((p, p))
            M[k, l] += 1.0
            M[l, k] += 1.0
            H[:, k, l] = vech(M)
    return H


Hten = hess_tensor()


def uls_fit(s, th, iters=80, tol=1e-12):  # Gauss-Newton
    for _ in range(iters):
        J = jac(th)
        step = np.linalg.solve(J.T @ V @ J + 1e-12 * np.eye(2 * p), J.T @ V @ (s - sigma(th)))
        th = th + step
        if np.max(np.abs(step)) < tol:
            break
    return th


def uls_F(s, th):
    r = s - sigma(th)
    return float(r @ (V @ r))


def ml_fit(s, th, iters=200, tol=1e-13):  # Fisher scoring
    for _ in range(iters):
        W = Wmat(unvech(sigma(th)))
        J = jac(th)
        step = np.linalg.solve(J.T @ W @ J, J.T @ W @ (s - sigma(th)))
        th = th + step
        if np.max(np.abs(step)) < tol:
            break
    return th


def ml_F(s, th):
    Sig, Sm = unvech(sigma(th)), unvech(s)
    _, ldS = np.linalg.slogdet(Sm)
    _, ldSig = np.linalg.slogdet(Sig)
    return float(np.trace(Sm @ np.linalg.inv(Sig)) - (ldS - ldSig) - p)


# ---- true (population) covariance: 2-factor truth, fit 1-factor -------------
Lam2 = np.zeros((p, 2))
Lam2[:3, 0] = [0.80, 0.75, 0.70]
Lam2[3:, 1] = [0.80, 0.75, 0.70]
Phi = np.array([[1.0, 0.55], [0.55, 1.0]])
common = Lam2 @ Phi @ Lam2.T
psi_true = 1.0 - np.diag(common)
Sigma0 = common + np.diag(psi_true)
s0 = vech(Sigma0)
V0 = Wmat(Sigma0)
theta0 = np.concatenate([np.full(p, 0.6), np.full(p, 0.6)])
df = ps - 2 * p


def fd_param_hessian(obj, th, h=1e-4):  # 0.5 * d^2 obj / dtheta^2 = discrepancy Hessian B
    t = th.size
    H = np.zeros((t, t))
    for i in range(t):
        for j in range(i, t):
            ei = np.zeros(t); ei[i] = h
            ej = np.zeros(t); ej[j] = h
            H[i, j] = H[j, i] = (
                obj(th + ei + ej) - obj(th + ei - ej) - obj(th - ei + ej) + obj(th - ei - ej)
            ) / (4 * h * h)
    return 0.5 * H


def profiled_hessian(fit, Fval, th_star, h=2e-4):  # 0.5 * d^2/ds^2 [profiled F] at s0
    Hs = np.zeros((ps, ps))
    def PF(s):
        return Fval(s, fit(s, th_star))
    for i in range(ps):
        for j in range(i, ps):
            ei = np.zeros(ps); ei[i] = h
            ej = np.zeros(ps); ej[j] = h
            Hs[i, j] = Hs[j, i] = (
                PF(s0 + ei + ej) - PF(s0 + ei - ej) - PF(s0 - ei + ej) + PF(s0 - ei - ej)
            ) / (4 * h * h)
    return 0.5 * Hs


# ---- ULS curvature objects --------------------------------------------------
th_uls = uls_fit(s0, theta0)
e_uls = s0 - sigma(th_uls)
F0_uls = uls_F(s0, th_uls)
Ju = jac(th_uls)
A_uls = Ju.T @ V @ Ju
K_uls = np.einsum("a,aij->ij", V @ e_uls, Hten)
B_uls = A_uls - K_uls
U_uls = V - V @ Ju @ np.linalg.solve(A_uls, Ju.T @ V)            # naive (Gauss-Newton A)
Utilde_uls = V - V @ Ju @ np.linalg.solve(B_uls, Ju.T @ V)       # corrected (Hessian B)

# ---- ML curvature objects (two metrics) -------------------------------------
th_ml = ml_fit(s0, theta0)
e_ml = s0 - sigma(th_ml)
F0_ml = ml_F(s0, th_ml)
Jm = jac(th_ml)
W_star = Wmat(unvech(sigma(th_ml)))
B_ml = fd_param_hessian(lambda th: ml_F(s0, th), th_ml)
Utilde_ml = V0 - W_star @ Jm @ np.linalg.solve(B_ml, Jm.T @ W_star)

Gamma_NT = 2.0 * Dp @ np.kron(Sigma0, Sigma0) @ Dp.T


def draw_normal(N, L):
    return (L @ rng.standard_normal((p, N))).T


def draw_t(N, L):  # symmetric heavy tails, t6, rescaled to unit marginal variance
    nu = 6.0
    Y = rng.standard_normal((p, N)) / np.sqrt(rng.chisquare(nu, size=N) / nu)
    return (L @ (Y * np.sqrt((nu - 2.0) / nu))).T


def mc_bias(Gamma, label, fit, Fval, F0, U, Utilde, draw, Ns, M):
    trU, trUt = float(np.trace(U @ Gamma)), float(np.trace(Utilde @ Gamma))
    print(f"\n[MC] {label}:  naive tr(U G) = {trU:.4f}   corrected tr(Utilde G) = {trUt:.4f}")
    L = np.linalg.cholesky(Sigma0)
    for N in Ns:
        Fs = np.empty(M)
        for m in range(M):
            S = np.cov(draw(N, L), rowvar=False, bias=False)
            Fs[m] = Fval(vech(S), fit(vech(S), th_uls))
        emp, se = Fs.mean() - F0, Fs.std(ddof=1) / np.sqrt(M)
        tag = "CORRECTED" if abs(N * emp - trUt) < abs(N * emp - trU) else "naive"
        print(f"   N={N:5d} M={M}: N*bias = {N*emp:7.4f} (se {N*se:.4f})  -> {tag}")


if __name__ == "__main__":
    print(f"=== setup: df={df}  F0(ULS)={F0_uls:.4f}  F0(ML)={F0_ml:.4f} "
          f"||e_uls||={np.linalg.norm(e_uls):.3f}  ||e_ml||={np.linalg.norm(e_ml):.3f}")
    print(f"    metric split ||W_star - V0|| = {np.linalg.norm(W_star - V0):.3f}")

    # (1) deterministic: Utilde == 0.5 * Hessian_s[profiled F]; naive uses A not B
    U_ml_fisher = V0 - W_star @ Jm @ np.linalg.solve(Jm.T @ W_star @ Jm, Jm.T @ W_star)
    print("\n[DET] Utilde = (1/2) Hessian_s[profiled discrepancy]  (exact, no MC):")
    for name, fit, Fval, th_star, Utilde, Unaive in [
        ("ULS", uls_fit, uls_F, th_uls, Utilde_uls, U_uls),
        ("ML ", ml_fit, ml_F, th_ml, Utilde_ml, U_ml_fisher),
    ]:
        Hp = profiled_hessian(fit, Fval, th_star)
        rel = np.linalg.norm(Hp - Utilde) / np.linalg.norm(Utilde)
        print(f"   {name}: ||Hprof/2 - Utilde||/||Utilde|| = {rel:.2e}   "
              f"corrected tr(Utilde G_NT) = {np.trace(Utilde @ Gamma_NT):7.4f}   "
              f"naive(A) = {np.trace(Unaive @ Gamma_NT):7.4f}")

    # (2) Monte-Carlo confirmation of the bias chain for ULS (beta_W = 0)
    Gmc = None
    Lc = np.linalg.cholesky(Sigma0)
    MG, NG = 120000, 2000
    SS = np.empty((MG, ps))
    for m in range(MG):
        SS[m] = vech(np.cov(draw_t(NG, Lc), rowvar=False, bias=False))
    Gmc = NG * np.cov(SS, rowvar=False, bias=False)
    mc_bias(Gamma_NT, "ULS normal", uls_fit, uls_F, F0_uls, U_uls, Utilde_uls, draw_normal, [200, 800], 80000)
    mc_bias(Gmc, "ULS t6", uls_fit, uls_F, F0_uls, U_uls, Utilde_uls, draw_t, [200, 800], 80000)
