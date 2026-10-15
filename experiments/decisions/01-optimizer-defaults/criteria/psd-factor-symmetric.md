# Lane psd-factor-symmetric: criteria (draft 2026-10-15, written before any run)

**Status.** Draft, awaiting the author's approval. The production run then
waits for simbox and a separate go from the author; the smoke run may start
once the criteria are approved.

**Decision.** Whether the default covariance factor of every PSD route
(complete-data ML, FIML, ML2S, the least-squares family and ordinal/mixed
DWLS) becomes the symmetric square root `PsdFactor::Symmetric` instead of the
lower triangular `PsdFactor::Triangular`, and, failing that, whether the
default becomes a triangular fit with a symmetric refit when it does not
converge.

**Why.** The symmetric factor writes a covariance block as $C = B^2$ with $B$
symmetric: $B$ is the matrix square root of $C$. It has the triangular
factor's $q(q+1)/2$ coordinates and no rotational freedom, and it keeps the
square factor's guarantee. In the eigenbasis of $B$ with eigenvalues
$\beta_i$, and with $G$ the gradient of an objective in $C$, first-order
stationarity of the lift gives $G_{ij}(\beta_i + \beta_j) = 0$. For pairs with
$\beta_i = -\beta_j \neq 0$ the Hessian block in $(B_{ii}, B_{ij})$ is
$\begin{pmatrix} \cdot & 2G_{ij} \\ 2G_{ij} & 0 \end{pmatrix}$, so second order
forces $G_{ij} = 0$; on the null block second order gives
$\langle G, E^2 \rangle \ge 0$. Every second-order critical point therefore has
$G \succeq 0$ and $GC = 0$, a KKT point of the PSD problem. At a positive
semidefinite root a local minimum of the lift is a local minimum of the PSD
problem, with links included, because the PSD square root is continuous. The
triangular factor lacks both properties at singular blocks of two or more
variables. Unlike the triangular and square factors, the symmetric root is not
equivariant under separate changes of units, so this lane tests units.

**Disclosure.** Development evidence, never gating:

- The psd-factor production run (seed base 2026101501, criteria
  `psd-factor.md`) failed its adoption rule. The square factor won F2 and F4
  in every estimator (143 and 14 wins, 1 loss) and lost F1 (57 losses, 17
  wins) and F3 with ULS (8 losses, 6 wins), almost always by using SLSQP's
  5,000 evaluations without converging.
- Diagnostic `diagnose-2026-10-15` on the 67 losses: four times the budget
  rescued 8 square fits, restarts from the returned estimate 49, consistent
  with rotational drift.
- Diagnostic `diagnose-symmetric-2026-10-15` on all 247 disagreement draws:
  symmetric succeeded in 232 and reached the best known objective in 231;
  triangular 178 and 176, square 182 and 176. Symmetric against triangular:
  62 wins and 8 losses, 7 of the losses in F1. These draws were selected on a
  failure, and they ran on a native opt build where production used the
  portable package build. Ten of them rerun on simbox reproduced production
  exactly, and the triangular code is the same in both builds, so the 117
  triangular convergence flags that differed came from the build: triangular
  outcomes on F2 depend on rounding. Both points favour the arm that
  was not selected, so these counts overstate symmetric's advantage.

## Configurations

All arms use the same route and differ only in the factor: transported FABIN3
start, SLSQP, diagonal preconditioning where the route has it, the library's
controls and tolerances. All arms go through the magmaanlab PSD wrappers.

| Arm | Factor | Role |
|---|---|---|
| `triangular` | library default, no options | baseline |
| `symmetric` | `psd_factor = "symmetric"` | candidate |
| `rescue` | `triangular`; if it does not converge, `symmetric` from the same start | candidate, composed from the two fits on the same draw without refitting; its time is the sum when the refit runs |
| `square` | `psd_factor = "square"` | reported only |

The report also gives, per family, the number of draws where `triangular`
differs from `magmaan(covariance = "psd")` by more than $10^{-4}$ in $T$.

## Problems

Test families F1 to F4 and the other estimators are those of `psd-factor.md`,
unchanged (F1 the test populations of `R/families.R`; F2 near-collinear
factors; F3 singular residual block; F4 collapsed-column starts; FIML with 15%
MCAR, ULS and ordinal DWLS on F2 and F3). One new test family:

- **F5. Separate units.** Every ML draw of F1, F2 and F3 whose model is unit
  invariant, refitted after the `mixed` transform of `R/fits.R` (observed
  variables multiplied alternately by 100 and 0.01). Paired outcomes are judged
  within the transformed fits. Consistency per arm: the number of draws whose
  success differs between native and mixed units.

Controls, reported only: Ernst and the weak-marker three-factor CFA, as in
`psd-factor.md`.

**Draws.** Seed base 2026101601, never used before, cell-stable seeds, 100
replications per cell. Smoke: two replications per cell at seed base
2026101602; it does not count.

**Build.** One build for every arm: magmaanlab and magmaan from a named main
commit, installed on simbox with `R CMD INSTALL` from the vendored sources, the
build users get. Metadata records the commit, the library path and the package
fingerprint.

## Judge

As in `psd-factor.md`: success is the library's `fit$converged`; the best known
objective is the lowest successful objective among `triangular`, `symmetric`
and `square` on the draw, reached within $10^{-6}(1 + \lvert f \rvert)$; a
candidate wins a draw when it succeeds and the baseline does not, or when both
succeed and only it reaches the best known objective, and loses in the mirror
case.

## Rules

They apply per test family and estimator, never pooled.

1. **Adopt** `Symmetric` as the default of every PSD route if, in every test
   family and estimator, `symmetric` has no more losses than wins against
   `triangular` in successes and in best known, its total seconds are at most
   1.25 times those of `triangular`, and in F5 its consistency count is no
   larger than `triangular`'s. Ties favour `symmetric`, because of its
   guarantee.
2. **Otherwise adopt the rescue** as the default if its total seconds are at
   most 1.25 times those of `triangular` in every test family and estimator.
   It returns the triangular fit whenever that converges, so it cannot lose
   to `triangular` on success or best known.
3. **Otherwise** nothing changes. The report lists every loss and attributes
   it to a mechanism (start, link feasibility, budget, units). Then the author
   decides.
4. **Regression test.** Whatever the outcome, the draw with the largest
   objective gap between `symmetric` and `triangular` in F2 to F5 becomes a C++
   regression test (follow-up card).

## Outputs

Per-run summary CSVs and `metadata.csv` under
`results/psd-factor-symmetric/<run-id>/`; raw per-fit rows stay ignored. The
report adds this lane to the default register. If rule 1 or 2 adopts: switch
the C++ default, update the capability area file and the roadmap, and give the
covariance-honest paper's §2.2 the symmetric square-root explanation.
