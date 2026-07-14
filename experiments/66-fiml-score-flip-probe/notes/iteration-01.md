# Iteration 01: direct-FIML score flips

## Construction

The sign action is on casewise observed-data likelihood scores evaluated at the
restricted FIML estimate. For case `i`, components missing in that row are
integrated out before its score is formed. The tested contribution is then
partialled against the nuisance-score tangent using the expected information.

Complete-data score flipping used one per-case Fisher-information matrix per
group. The FIML extension replaces that compression with one matrix per
`(group, observed-pattern)` stratum. For an observed set `o`, the conditional
normal-theory information contribution is

```text
I_ab(o) = 0.5 tr(Sigma_o^-1 dSigma_a Sigma_o^-1 dSigma_b)
          + dmu_a' Sigma_o^-1 dmu_b.
```

The all-observed special case therefore reduces exactly to the existing
complete-data path. Unit tests gate that reduction for all three flip
statistics, their p-values, and the score-mixture eigenvalues.

The three randomization arms separate distinct questions:

1. `basic` flips the tested score coordinates before nuisance removal;
2. `effective` flips contributions already projected off the nuisance tangent;
3. `standardized` uses the effective contributions but recomputes their
   quadratic metric from the flip-specific pattern/group sign sums.

The third step is an operation on the effective scores. It is not an
imputation, a label permutation, or a refit for every sign vector.

## Probe outcome

The probe used the published FIML--FMG two-group, six-indicator population and
the configural-to-metric comparison (`df=5`). It crossed `n1=50,100,200`, a
70% group-size ratio, 0/15/30% MCAR, normal/severe polynomial-logistic data,
and null/loading-power truths. There were 100 replications and 199 random signs
per cell.

- Basic flip null rejection averaged 0.009 across cells.
- Nuisance-effective flip averaged 0.064; 0.038 normal and 0.090 severe PL.
- Standardized flip averaged 0.062 and changed only three null decisions, all
  from reject to accept.
- Score pEBA4 averaged 0.038 normal and 0.060 PL. The direct-sandwich score was
  near nominal but somewhat less powerful in matched-null comparisons.
- Nested-LR pEBA4 averaged 0.035 normal and 0.185 PL. Its good normal result did
  not survive the distributional stress.
- Twenty-seven fits failed. A further 29 successful-fit replications lost the
  nested FMG battery to ill-conditioned/indefinite Satorra-2000 matrices. The
  score-flip path had no failures conditional on a successful fit.

The key comparison is paired, not pooled. Pattern-specific covariance movement
increased with missingness, but effective-versus-standardized null decision
disagreement remained 0%, 0.34%, and 0.17% at 0%, 15%, and 30% MCAR. With five
restrictions, standardization is a measurable correction without a material
decision benefit.

## Timing

The 3,600-replication probe completed in about 42 wall seconds on four local
workers. At normal-null `n1=100`, the median pattern-standardization cost was
about 0.8 ms complete, 4.9 ms at 15% MCAR, and 7.1 ms at 30% MCAR. Computing
the signed effective-score sums was below 2 ms. Thus pattern proliferation, not
the number of cases alone, makes the standardized arm the dominant flip cost.

Polynomial-logistic calibration was performed once per group and truth task,
outside replication timing. Its two tasks cost about 8.2 seconds in total.

## Working decision

For the next FIML run:

- treat nuisance-effective flipping as the primary randomization method;
- retain standardized flipping as a diagnostic until restriction rank or
  information heterogeneity is large enough to produce paired gains;
- retain score pEBA4, direct sandwich, and exact-mixture p-values;
- keep nested FMG methods as comparators and report their numerical
  availability explicitly;
- spend replication budget on a reduced null grid before adding more methods.

The immediate unresolved question is whether the severe-PL liberalism of the
effective flip persists with 500--1,000 successful null replications per cell.
Only then is it worth varying the number of restrictions or adding MAR.
