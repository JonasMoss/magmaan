"""Numerical checks for profile_hessian_rank_accounting.tex.

Companion to higher_order_discrepancy_check.py. That note characterizes the
*mean* of the fixed-misspecification profile statistic 2N{v(x_hat)-v(x0)} ->
Z'QΓZ via the trace tr(QΓ) (the curvature-corrected df). This note characterizes
the *inertia* of the same QΓ: how many nonzero, positive, and negative
eigenvalues the limiting quadratic form has. In the PSD/local regime this is the
weighted-χ² term count; under fixed misspecification the same eigenvalues are
mean-level contributions to tr(QΓ), not an ordinary χ² mixture.

Q is magmaan's WeightedProfileRMSEAResult::profile_hessian, the curvature-
corrected residual-weight matrix Utilde of higher_order_discrepancy_misspec.tex:

    single metric (ULS / fixed weight):  Q = V  - V  J B^{-1} J' V
    two metric    (ML / estimated wt) :  Q = V0 - W* J B^{-1} J' W*

with B = A - K the discrepancy HESSIAN (A = J'W_proj J the Gauss-Newton part,
K = (W_proj e)'·sigma_ddot the residual-curvature coupling).

THE RANK-ACCOUNTING IDENTITY (proved in the note, checked here). Write the
profile Hessian as Q = Vout - Wproj J B^{-1} J' Wproj, set
    Gtil = Vout^{-1/2} Wproj J,     Atil = J' Wproj Vout^{-1} Wproj J,
so Q = Vout^{1/2}(I - Gtil B^{-1} Gtil') Vout^{1/2}. Then col(Gtil) is invariant
and the inner symmetric matrix splits as

    spec(I - Gtil B^{-1} Gtil') = { 1 (x (m-p)) }  U  { 1 - nu_j : j=1..p },
    nu_j = eig(B^{-1} Atil).

For the full-rank Γ used here, Sylvester congruence by Vout^{1/2} and then by
Γ^{1/2} gives the same inertia for Q, Γ^{1/2}QΓ^{1/2}, and the inner matrix
(equivalently, the nonzero eigenvalues of QΓ have the same signs), so

    rank(QΓ)        = (m - p) + #{ nu_j != 1 } = (m - p) + rank(K-or-metric gap)
    n_positive(QΓ)  = (m - p) + #{ nu_j <  1 }
    n_negative(QΓ)  =           #{ nu_j >  1 }.

If Γ is singular these counts apply only on range(Γ). The count is Γ-free only
conditional on a fixed Q; Q itself can change with the population moments,
pseudo-true point, and weight-estimation channel.

nu_j = 1 for all j  <=>  B = Atil  <=>  K = 0 AND Wproj = Vout (in their action
on col(J))  <=>  the classical df = m - p is recovered with all weights 1.
- ULS (Vout = Wproj = V): the metric gap is absent, Atil = A, so the only way to
  leave classical df is curvature, nu_j = eig(B^{-1}A), one channel.
- ML/GLS (Vout = V0 != W* = Wproj): a SECOND channel, Atil != A, opens even
  before curvature; both vanish together only under correct specification.

Run: python3 profile_hessian_rank_check.py   (numpy only).
"""

import numpy as np

rng = np.random.default_rng(20260622)
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


def uls_fit(s, th, iters=200, tol=1e-13):  # Gauss-Newton
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


def ml_fit(s, th, iters=400, tol=1e-14):  # Fisher scoring
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


# ---- linear-algebra helpers -------------------------------------------------
def sym(M):
    return 0.5 * (M + M.T)


def msqrt(M):  # symmetric PSD square root
    w, Qm = np.linalg.eigh(sym(M))
    w = np.clip(w, 0.0, None)
    return Qm @ np.diag(np.sqrt(w)) @ Qm.T


def msqrt_inv(M):
    w, Qm = np.linalg.eigh(sym(M))
    return Qm @ np.diag(1.0 / np.sqrt(w)) @ Qm.T


def inertia(M, tol=1e-7):  # (n_pos, n_zero, n_neg) of a symmetric matrix
    w = np.linalg.eigvalsh(sym(M))
    scale = max(1.0, np.max(np.abs(w)))
    return (int(np.sum(w > tol * scale)),
            int(np.sum(np.abs(w) <= tol * scale)),
            int(np.sum(w < -tol * scale)))


def gen_eigvals(Q, Gam):  # eigenvalues of Q @ Gamma, reported real-symmetrized
    # eig(QΓ) = eig(Γ^{1/2} Q Γ^{1/2}); symmetric so real.
    Gh = msqrt(Gam)
    return np.linalg.eigvalsh(sym(Gh @ Q @ Gh))


def matrank(M, tol=1e-7):
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * max(1.0, s[0])))


# ---- population: 2-factor truth, fit 1-factor (unabsorbable misfit) ----------
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
df = ps - 2 * p  # classical degrees of freedom, m - p
Gamma_NT = 2.0 * Dp @ np.kron(Sigma0, Sigma0) @ Dp.T


def profile_objects(fit, Fval, Wproj_fn, Vout, analytic_K=False, spop=None):
    """Return the profile-Hessian ingredients at the pseudo-true fit.

    Wproj_fn(theta_hat) -> projection metric (W* for ML, V for ULS); Vout is the
    outer/data metric (V0 for ML, V for ULS). analytic_K uses the exact ULS
    curvature K = (V e)'·sigma_ddot (no finite differences); otherwise B is the
    finite-difference discrepancy Hessian. spop overrides the population moments
    (defaults to s0)."""
    s_pop = s0 if spop is None else spop
    th = fit(s_pop, theta0)
    e = s_pop - sigma(th)
    J = jac(th)
    Wproj = Wproj_fn(th)
    A = J.T @ Wproj @ J
    if analytic_K:
        K = np.einsum("a,aij->ij", Wproj @ e, Hten)        # exact, fixed weight
        B = A - K
    else:
        B = fd_param_hessian(lambda t: Fval(s_pop, t), th)  # discrepancy Hessian
        K = A - B
    Q = Vout - Wproj @ J @ np.linalg.solve(B, J.T @ Wproj)  # = Utilde
    Atil = J.T @ Wproj @ np.linalg.solve(Vout, Wproj @ J)   # metric-gap A-tilde
    return dict(th=th, e=e, J=J, Wproj=Wproj, Vout=Vout, A=A, B=B, K=K, Q=sym(Q),
                Atil=Atil, F0=Fval(s_pop, th))


def check_case(name, o, Gam):
    print(f"\n================ {name} ================")
    J, B, Atil, Q, Vout = o["J"], o["B"], o["Atil"], o["Q"], o["Vout"]
    m = ps
    pfree = J.shape[1]
    print(f"  m={m}  p={pfree}  classical df = m-p = {m - pfree}   "
          f"||e||={np.linalg.norm(o['e']):.4f}  ||K||={np.linalg.norm(o['K']):.4f}  "
          f"||Wproj-Vout||={np.linalg.norm(o['Wproj'] - Vout):.4f}")

    # (1) spectral split of the inner matrix I - Gtil B^{-1} Gtil'
    Vout_ih = msqrt_inv(Vout)
    Gtil = Vout_ih @ o["Wproj"] @ J                  # m x p
    inner = np.eye(m) - Gtil @ np.linalg.solve(B, Gtil.T)
    nu = np.linalg.eigvals(np.linalg.solve(B, Atil)).real  # eig(B^{-1} Atil)
    predicted = np.concatenate([np.ones(m - pfree), 1.0 - np.sort(nu)])
    got = np.sort(np.linalg.eigvalsh(sym(inner)))
    err_split = np.max(np.abs(np.sort(predicted) - got))
    print(f"  (1) spec(I - Gtil B^-1 Gtil') == {{1}}^(m-p) U {{1-nu_j}} : "
          f"max|Δ| = {err_split:.2e}   {'OK' if err_split < 1e-6 else 'FAIL'}")

    # (2) Q = Vout^{1/2} inner Vout^{1/2}  (the profile Hessian factorization)
    Qrec = msqrt(Vout) @ inner @ msqrt(Vout)
    err_fac = np.max(np.abs(Qrec - Q))
    print(f"  (2) Q == Vout^1/2 (I - Gtil B^-1 Gtil') Vout^1/2 : "
          f"max|Δ| = {err_fac:.2e}   {'OK' if err_fac < 1e-6 else 'FAIL'}")

    # (3) full-rank Γ: inertia(QΓ) == inertia(Q) == inertia(inner)
    iQG, iQ, iIn = inertia(Gam_cong(Q, Gam)), inertia(Q), inertia(inner)
    print(f"  (3) inertia  QΓ={iQG}  Q={iQ}  inner={iIn} : "
          f"{'OK' if iQG == iQ == iIn else 'FAIL'}   (n_pos, n_zero, n_neg)")

    # (4) the rank-accounting counts from nu_j
    n_gt = int(np.sum(nu > 1 + 1e-7))
    n_lt = int(np.sum(nu < 1 - 1e-7))
    n_eq = int(np.sum(np.abs(nu - 1) <= 1e-7))
    pred_pos = (m - pfree) + n_lt
    pred_neg = n_gt
    pred_rank = (m - pfree) + (pfree - n_eq)
    lam = gen_eigvals(Q, Gam)
    obs_pos, obs_zero, obs_neg = inertia_from_vals(lam)
    obs_rank = obs_pos + obs_neg
    print(f"  (4) nu_j: #<1={n_lt} #>1={n_gt} #=1={n_eq}")
    print(f"      n_positive(QΓ): predicted (m-p)+#{{nu<1}} = {pred_pos:2d}   "
          f"observed = {obs_pos:2d}   {'OK' if pred_pos == obs_pos else 'FAIL'}")
    print(f"      n_negative(QΓ): predicted    #{{nu>1}}     = {pred_neg:2d}   "
          f"observed = {obs_neg:2d}   {'OK' if pred_neg == obs_neg else 'FAIL'}")
    print(f"      rank(QΓ)      : predicted (m-p)+rank(gap) = {pred_rank:2d}   "
          f"observed = {obs_rank:2d}   {'OK' if pred_rank == obs_rank else 'FAIL'}")

    # (5) signed trace (note's df) vs positive-only trace (magmaan bias_trace)
    tr_signed = float(np.sum(lam))
    tr_pos = float(np.sum(lam[lam > 0]))
    print(f"  (5) tr(QΓ) signed = {tr_signed:.4f}   Σ positive λ = {tr_pos:.4f}   "
          f"gap from negatives = {tr_pos - tr_signed:.4f}")
    return lam


def Gam_cong(Q, Gam):  # Γ^{1/2} Q Γ^{1/2}, same inertia as QΓ
    Gh = msqrt(Gam)
    return sym(Gh @ Q @ Gh)


def inertia_from_vals(w, tol=1e-7):
    scale = max(1.0, np.max(np.abs(w)))
    return (int(np.sum(w > tol * scale)),
            int(np.sum(np.abs(w) <= tol * scale)),
            int(np.sum(w < -tol * scale)))


def residual_shrink_scan(fit, Fval, Wproj_fn, Vout_fn, label, analytic_K=False, floor=1e-3):
    """As the population is pulled onto the model (alpha: 1 -> 0), e -> 0,
    K -> 0, the metric gap -> 0. The ALGEBRAIC rank stays (m-p)+rank(gap) for any
    e != 0, but the (m-p) extra eigenvalues emerge CONTINUOUSLY as O(||e||); only
    at e=0 exactly do they vanish and the count collapses to classical df = m-p.
    We report the effective rank at a physical floor and the largest extra |lambda|
    to expose that O(||e||) scaling rather than a brittle integer rank."""
    print(f"\n---- residual-shrink scan ({label}): effective rank -> df={df} "
          f"as e->0; extras are O(||e||) (floor={floor:g}) ----")
    th_full = fit(s0, theta0)
    Sig_fit = unvech(sigma(th_full))         # best-fit model covariance
    print("    alpha   ||e||    ||K||  eff_pos eff_neg  max|extra|  Σ+λ     tr_signed")
    for alpha in [1.0, 0.6, 0.3, 0.1, 0.03, 0.0]:
        Sig_a = Sig_fit + alpha * (Sigma0 - Sig_fit)
        s_a = vech(Sig_a)
        Gam_a = 2.0 * Dp @ np.kron(Sig_a, Sig_a) @ Dp.T
        th = fit(s_a, theta0)
        e = s_a - sigma(th)
        J = jac(th)
        Wproj = Wproj_fn(th)
        Vout = Vout_fn(Sig_a)
        A = J.T @ Wproj @ J
        if analytic_K:
            B = A - np.einsum("a,aij->ij", Wproj @ e, Hten)
        else:
            B = fd_param_hessian(lambda t: Fval(s_a, t), th)
        K = A - B
        Q = sym(Vout - Wproj @ J @ np.linalg.solve(B, J.T @ Wproj))
        lam = gen_eigvals(Q, Gam_a)
        eff_pos = int(np.sum(lam > floor))
        eff_neg = int(np.sum(lam < -floor))
        extra = np.sort(np.abs(lam))[:-(df)] if df > 0 else np.abs(lam)  # drop df largest
        max_extra = float(np.max(extra)) if extra.size else 0.0
        print(f"   {alpha:5.2f}  {np.linalg.norm(e):7.4f} {np.linalg.norm(K):7.4f}  "
              f"{eff_pos:5d}  {eff_neg:5d}   {max_extra:9.2e}  "
              f"{np.sum(lam[lam > 0]):7.3f}  {np.sum(lam):8.3f}")


def negative_weight_law_demo(o, Gam, label, seed=20260622):
    """Exhibit a negative mixture weight and show WHERE it lives.

    Under fixed misspecification the statistic T = N(F_hat - F0) is NOT a chi^2
    mixture: it is Gaussian (companion note, sqrt(N)(F_hat-F0) -> N(0,omega^2)),
    so its SD grows like sqrt(N). The signed spectrum (negative weight included)
    is the O(1) MEAN, E[T] -> tr(Q Gamma) = sum lambda_j, swamped by the sqrt(N)
    spread. A negative lambda_j is therefore a negative BIAS contribution (it
    lowers the generalized df), never a realized mixture weight."""
    print(f"\n================ negative weight: where it lives ({label}) ================")
    lam = np.sort(gen_eigvals(o["Q"], Gam))
    S1, S2 = float(lam.sum()), float((lam ** 2).sum())
    print(f"  QΓ spectrum min weight = {lam.min():.3f} (the negative one); "
          f"Σλ = {S1:.3f}, Σλ² = {S2:.3f}")
    print(f"  IF T ~ Σλⱼχ²₁ (mixture): mean = {S1:.2f}, SD = {np.sqrt(2*S2):.2f} "
          f"(CONSTANT in N)")
    rng2 = np.random.default_rng(seed)
    th0 = uls_fit(s0, theta0)
    F0 = uls_F(s0, th0)
    L = np.linalg.cholesky(Sigma0)
    print("  Monte Carlo of the actual T = N(F_hat - F0) under fixed misspecification:")
    print("     N        mean(T)   SD(T)    SD/√N   (SD/√N const ⇒ Gaussian, not mixture)")
    for N, M in [(250, 1500), (1000, 1500), (4000, 1500), (16000, 1200)]:
        Ts = np.empty(M)
        for r in range(M):
            X = (L @ rng2.standard_normal((p, N))).T
            s = vech(np.cov(X, rowvar=False, bias=False))
            Ts[r] = N * (uls_F(s, uls_fit(s, th0)) - F0)
        sd = Ts.std(ddof=1)
        print(f"   {N:6d}    {Ts.mean():7.2f}  {sd:7.2f}  {sd/np.sqrt(N):7.3f}")


if __name__ == "__main__":
    print(f"=== profile-Hessian rank accounting | p(vars)={p}  m={ps}  "
          f"classical df = m - 2p = {df} ===")

    # ULS: single metric, only the observed-bread (curvature) channel.
    # analytic_K -> the ULS validation is fully analytic, no finite differences.
    o_uls = profile_objects(uls_fit, uls_F, lambda th: V, V, analytic_K=True)
    lam_uls = check_case("ULS  (single metric V; fixed weight; analytic K)", o_uls, Gamma_NT)

    # ML: two metrics V0 (data) and W* (model) => estimated-weight channel too.
    o_ml = profile_objects(ml_fit, ml_F, lambda th: Wmat(unvech(sigma(th))), V0)
    lam_ml = check_case("ML   (two metrics V0, W*; estimated weight)", o_ml, Gamma_NT)

    # (6) envelope Q is the right profile Hessian, NOT the naive sandwich M.
    print("\n================ envelope vs naive sandwich (ULS) ================")
    J, B, Q = o_uls["J"], o_uls["B"], o_uls["Q"]
    L = np.eye(ps) - J @ np.linalg.solve(B, J.T @ V)       # I - J B^-1 J' V
    M_naive = sym(L.T @ V @ L)                              # (I-..)'V(I-..)
    print(f"  ||M_naive - Q|| = {np.linalg.norm(M_naive - Q):.4f}  (nonzero off H0: "
          "linearizing r(theta_hat) and squaring drops an O(1/N) cross term)")
    print(f"  rank(M_naive)={matrank(M_naive)}  rank(Q)={matrank(Q)}  "
          f"(Q is half the Hessian of the profiled discrepancy; see "
          "higher_order_discrepancy_check.py DET block)")

    # residual-shrink: classical df is the e->0 / K->0 limit, both estimators.
    residual_shrink_scan(uls_fit, uls_F, lambda th: V, lambda Sig: V, "ULS",
                         analytic_K=True)
    residual_shrink_scan(ml_fit, ml_F,
                         lambda th: Wmat(unvech(sigma(th))),
                         lambda Sig: Wmat(Sig), "ML")

    # the negative weight is a mean-level (bias) object, not a sampling-law weight.
    negative_weight_law_demo(o_uls, Gamma_NT, "ULS")
