# std.lv transport under equality constraints

Draft 2026-10-11 by the planner. Gap noted by TASK-131, TASK-137 and TASK-143:
`prepare_std_lv_transport()` (cpp/src/estimate/start_pipeline.cpp) refuses every
model with an active linear equality or a nonlinear constraint. Three features
depend on the transport: the automatic start fallback (TASK-131 now builds the
fallback in normalized units instead), and the std.lv-chart Newton audits for
barrier near-pole endpoints (TASK-137) and ML/GLS runaways (TASK-143). In the
2026-10-09 barrier rerun, 29 of the default arm's 31 remaining certified
near-pole fits were in the equality-constrained family, where the audit cannot
run.

## Why most equalities survive a rescaling

The transport is a per-latent rescaling. With factor scale $s_j$ in block $b$,
the parameters change as

$$
\lambda_{ij}\mapsto \lambda_{ij}s_j,\qquad
\psi_{jk}\mapsto \frac{\psi_{jk}}{s_js_k},\qquad
\beta_{jk}\mapsto \beta_{jk}\frac{s_k}{s_j},\qquad
\alpha_j\mapsto \frac{\alpha_j}{s_j},\qquad
\theta,\ \nu\ \text{unchanged}.
$$

Write $u=\log s$. Each parameter is multiplied by $e^{\ell(u)}$ for a linear
form $\ell$: $u_j$ for a loading, $-u_j-u_k$ for a latent covariance,
$u_k-u_j$ for a regression, $-u_j$ for a latent mean, $0$ for residual
variances and intercepts. A linear equality between parameters $p$ and $q$ is
preserved exactly when $\ell_p(u)=\ell_q(u)$. So the rescalings that keep the
constraint set form the linear subspace

$$
\mathcal U=\{u:\ \ell_p(u)=\ell_q(u)\ \text{for every equated pair}\},
$$

and the marker chart's gauge freedom restricted by the constraints is exactly
$\mathcal U$. The std.lv chart fixes that gauge differently (unit latent
variances instead of unit markers), which is what keeps parameter magnitudes
finite near a marker pole.

## Proposal

**Phase 1: simple ties (covers the common cases).** If every equality group
only ties scales of the same kind, merge the corresponding $u_j$ with a
union-find over (block, factor): equal loadings on one factor (tau-equivalence)
need no tie; equal loadings or latent means of the same factor across groups
(metric and scalar invariance) tie that factor's scale across groups; equal
loadings across different factors tie those factors. Equalities among
residual variances or intercepts constrain nothing. The target chart fixes the
latent variance of one representative factor per tied class (the first group's,
as lavaan does for invariance models) and leaves the others free, and keeps the
equalities as linear rows. Within a tied class of k factors the k unit markers
fix the common gauge and also k-1 restrictions on relative scales, so the chart
adds k-1 rows tying the released markers to the representative's marker; the
constrained tangent dimension must equal the source's (found by the TASK-160
lane). Groups that mix kinds with different forms (for
example a loading equal to a covariance) or that tie regressions, whose form
$u_k-u_j$ is a difference, are refused with a named reason, as now.

**Phase 2: general linear forms.** Compute $\mathcal U$ from all equality
groups and choose the chart's scale as the projection of the std.lv target
$u^\star_j=\tfrac12\log\psi_{jj}$ onto $\mathcal U$; the gauge is then fixed by
unit variances for a basis of $\mathcal U$'s free directions. This covers equal
regression paths in chains and equal latent covariances. Nonlinear constraints
stay refused.

## Evidence plan

Phase 1 is concrete: implement the classification and chart, then replay the
decisions/02 constrained family (equal loadings, equal means) and the TASK-143
equality-constrained runaways, reporting how many become auditable and whether
any certified fit flips. Phase 2 waits for phase 1's coverage numbers; the
equal-path chain (`eqchain_b20`) is its main target.

## Decisions (user, 2026-10-11)

Build phase 1, then phase 2. The audit chart fixes one unit-variance factor
per tied scale class. The improved transport also serves the automatic start
fallback for constrained models where it applies, as a separate step with its
own no-loss evidence, because it changes default starts.

## Questions as discussed

1. Is "one unit-variance representative per tied scale class" the right chart
   for the audit, or should the audit use a chart that is closer to the
   sphere/global-gauge coordinates for these models?
2. Should phase 2 be built, or should constrained near-pole cases be routed to
   the sphere fit instead once its constraint geometry is complete?
3. Should the transport also serve the start fallback for constrained models,
   replacing TASK-131's normalized-unit construction where it applies?
