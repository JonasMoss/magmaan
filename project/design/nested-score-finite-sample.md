# Finite-sample calibration of the observed nested ML score

TASK-93 is research-tier. No library or ordinary-policy change follows from its
local pilot. The target is a restriction at the larger working model's
pseudo-true parameter, including when that working model is wrong. Rows are
iid within two fixed-allocation groups; dependent/clustered rows and boundary
nulls require other contracts.

## Statistic and mechanism

Write the total likelihood score at the restricted estimate as
$S=\sum_i s_i(\hat\theta_0)$, its observed sensitivity as $H$, the expected
normal-theory information as $I$, the null tangent as $K$, and complementary
restriction directions as $D$. All information matrices here are totals.
The policy uses

$$
G=D-K(K^\top H K)^{-1}K^\top H D,\qquad
M=G^\top I G,\qquad Q=S^\top G M^{-1}G^\top S.
$$

The reference spectrum is that of
$M^{-1/2}G^\top(\sum_i s_i s_i^\top)G M^{-1/2}$, with raw likelihood rows.
Observed sensitivity is essential for nuisance adjustment under
misspecification; expected information is only a positive definite metric.
The baseline reports SB and pEBA4 approximations to this weighted chi-square
law. Bootstrap consistency below refers to the statistic's actual limiting
law, rather than asserting pEBA4 is an exact mixture tail.

TASK-73's replay found normal statistic variances 7.09/8.05 against reference
variances 12.01/12.28 (correct/mild). Under skewness these were 20.07/22.53
against 55.78/64.87. Meat trace ratios near one undermine uniform meat
inflation as an explanation. Freezing population projection and metric raises
statistic variance. These interventions implicate joint geometry estimation;
they neither derive a Bartlett coefficient nor prove a unique tail mechanism.

## Candidate 1: exactly recentered objective, with nuisance refitting

Let $\bar s=S/N$. Resample cases within groups and minimize on the null tangent
an adjusted criterion

$$
L_b^c(\theta)=\sum_i\ell(Z_{ib};\theta)
             -N\bar s^\top(\theta-\hat\theta_0),
$$

where $s=\nabla\ell$; reverse both signs for a log-likelihood convention.
The conditional expected gradient at $\hat\theta_0$ is zero in every larger-
model coordinate. The restriction therefore holds at a local bootstrap
pseudo-true point, without making the fitted covariance model the generator.
Recompute $H_b,I_b,G_b$ at the adjusted restricted optimum and form $Q_b^c$
from the adjusted score. Covariance uses **original likelihood rows**, centered
within sampling groups, rather than treating the artificial deterministic
recentering term as random variation. The empirical upper tail gives a test.

With an interior isolated pseudo-true solution, nonsingular null Hessian,
smooth derivatives and sufficient moments, Taylor expansion gives the same
observed-Hessian nuisance projection of the empirical score process in sample
and bootstrap. The adjusted empirical criterion retains the empirical law and
misspecified Hessian; its linear subtraction changes no Hessian. Conditional
CLT and continuous mapping then yield the same limiting quadratic. $B$ must
increase for Monte Carlo error to vanish. This is a first-order argument, not
an Edgeworth/refinement theorem for this SEM.

Cost: $B$ adjusted restricted optimizations and geometry evaluations, plus the
original two fits. A prepared larger-model structure suffices; its optimum is
not mathematically needed. Failure: singular projected information, boundary
fits or alternate minima invalidate the local expansion. Keep the failed draw,
make that bootstrap arm unavailable, and never silently replace it.

[Hall and Horowitz (1996)](https://www.jstor.org/stable/2171849) motivate
null recentering for GMM bootstrap critical values. Their refinement result
is not a theorem for this misspecified-SEM construction. More generally,
[Lee's misspecification-robust GMM bootstrap](https://arxiv.org/abs/1806.01450)
shows that model misspecification changes the required asymptotic argument;
recentring alone is not evidence of higher-order validity. The derivation
above is specific to likelihood-gradient M-estimation, not a recentered
covariance-residual SEM or an estimated-weight GMM shortcut.

## Candidate 2: post-fit recentering, recomputed geometry (pilot arm)

Existing bindings fit an ordinary restricted bootstrap model. At its estimate,
recompute observed sensitivity and expected metric, then use

$$
\widetilde S_b=S_b(\hat\theta_{0b})-S(\hat\theta_0),\qquad
\widetilde Q_b=\widetilde S_b^\top G_b M_b^{-1}G_b^\top\widetilde S_b.
$$

This is not an exactly imposed conditional bootstrap null. In affine common
coordinates $K^\top S(\hat\theta_0)=0$ up to optimization error, so subtracting
that vector does not change the nuisance estimating equation. Expanding the
restricted bootstrap solution around $\hat\theta_0$ gives

$$
\widetilde S_b=\{1-HK(K^\top H K)^{-1}K^\top\}
              \{S_b(\hat\theta_0)-S(\hat\theta_0)\}+o_{P^*}(\sqrt N).
$$

Since $G^\top H K=0$, its projected limit equals Candidate 1's projected
limit. Moreover $G_b-G=o_{P^*}(1)$ and $M_b/N-M/N=o_{P^*}(1)$ under the same
regularity conditions. This proves first-order validity under the nested
pseudo-null; it does not prove that post-fit recentering reproduces Candidate
1's higher-order bias. Refitted geometry can alter small-N tail behavior,
which is the pilot's question.

Cost: currently two ordinary fits per resample because the public component
route takes a fitted larger model, hence $2B+2$ fits per test. A future prepared
structure route could reduce it to $B+2$, but is not part of this task. Keep
common-coordinate/tangent checks and the original convergence verdict; no
ridge, relaxed tolerances, replacement resamples or deleting failed draws.

The cheaper control resamples original score rows, subtracts their full-sample
sum, and holds $G,M$ fixed. Within-group resampling implicitly centers by group
and preserves fixed allocation. It has the same first-order limit when the
pseudo-null score mean vanishes in each group (as in these identical-group
cells). It costs zero refits, $B$ row sums/quadratics, but cannot reproduce
random nuisance geometry compression. For heterogeneous group pseudo-score
means, apply the sampling covariance contract explicitly; raw pooled-row
resampling is not interchangeable with fixed group allocation.

## Candidate 3: bootstrap-assisted moment adjustment (deferred)

A uniform meat shrinkage is not justified by TASK-73. A joint higher-order
expansion would need derivatives of $G(H)$, $M(H,I)$ and their cross-cumulants
with $S$, through the order affecting the rejection tail. No such analytic
coefficient has been derived here; a classical correctly specified Bartlett
factor must not be reused under misspecification.

A derivable development alternative is an affine reference correction. Let
$(\mu_0,v_0)$ be the fitted mixture's first two moments, and $(\mu_b,v_b)$ the
recentered, geometry-refitting bootstrap moments. Transform the reference
$X_0$ to $a+cX_0$, where $c=\sqrt{v_b/v_0}$ and
$a=\mu_b-c\mu_0$. With consistent bootstrap moments, $a\to0,c\to1$,
so this preserves an exact mixture baseline asymptotically. It matches only two
moments, can imply negative reference support, and leaves skewness/tail bias;
with a pEBA approximation it preserves that approximation's limitation.
It costs Candidate 2's $2B$ fits and no additional fits; finite fourth moments
and uniform integrability are needed for moment consistency. It is deferred:
$B=199$ is poor for estimating skewed quadratic variance. Prefer an empirical
bootstrap tail before adding noisy moment fitting.

## Pilot and next study

The independent [research/54 report](../../experiments/research/active/54-nested-score-small-n/report.qmd)
contains fresh development size estimates, Wilson intervals, failures and cost.
Its four cells copy only decisions/04's frozen design constants: six loading
restrictions, N=100 per group, normal/skewed, correct/mild larger model.
Forty draws per cell and B=199 are requested within 1100 seconds, two workers,
nice and one math thread per worker. This is pricing and feasibility evidence;
40 draws cannot distinguish 2% from 5% size. No analytic correction or exact
adjusted-objective implementation is claimed to have been run.

Before any later simbox run, freeze a registration with these proposed cells:
original six-restriction family plus two factors with six indicators each
(ten restrictions); N=100,300,1000 per group; normal/skewed; correct/mild/strong
shared misspecification; null and matched loading-departure power. This is
72 cells, 2000 null and 1000 power draws/cell, 108000 draws. Check each
population's pseudo-null and local observed curvature before registration;
strong/larger-df constants and pseudo-target power departures must be committed.

Arms: current observed SB/pEBA4; fixed-geometry recentered score bootstrap;
post-fit recentered nuisance-refitting bootstrap; exact adjusted-objective
bootstrap only after independent case-gradient and nuisance-stationarity gates.
B=999; null-tail critical values for size-adjusted power; Wilson and paired
size-error intervals per cell, all failures against all draws. Do not pool
families. Fresh reserved draw base 1226100101; seed = base + 10000*cell_id +
replicate. Resampling streams in production must use distinct per-draw/subdraw
streams rather than pilot's overlapping index seeds. Reserve 1426100101 for
summary resampling; neither is used by this pilot. These are a proposal, not
approved production or evidence. Simbox remains unavailable until the user
says otherwise. Price from observed pilot cost in the report; cap, then budget
20% contingency. Investigate any loss versus baseline before adoption.
