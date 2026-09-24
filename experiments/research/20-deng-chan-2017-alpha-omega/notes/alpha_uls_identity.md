# Provenance: alpha is the ULS plug-in of the tau-equivalent model

Supporting note for the alpha-omega corrective manuscript. Records the exact
statement, a short proof, the estimator subtlety, and the citation status of the
identity used as appendix validation item 4 and measured in
`results/validation_identity.csv`.

## Statement

Let `S` be the `p x p` sample covariance matrix of the items, with off-diagonal
mean `s_off = (1 / (p(p-1))) * sum_{i != j} s_ij` and total `1' S 1`. Cronbach's
alpha is

```
alpha = (p / (p-1)) * (sum_{i != j} s_ij) / (1' S 1).
```

Fit a one-factor model with equal loadings (essential tau-equivalence) to `S`
by unweighted least squares (ULS), and read off coefficient omega from the
fitted parameters. Then, at the interior (admissible) optimum,

```
omega_hat(ULS tau-equivalent fit) = alpha   exactly.
```

The same holds for the parallel model (equal loadings and equal unique
variances). So both parallel and tau-equivalent ULS fits return alpha; the
congeneric ULS fit does not (it returns the larger omega).

This is an algebraic identity at the sample level (and at the population level
with `Sigma` for `S`), not an asymptotic equivalence.

## Proof

Tau-equivalent structure: `Sigma(c, theta) = c * 1 1' + diag(theta_1, ..., theta_p)`,
`c = lambda^2 psi`. ULS minimizes `0.5 * sum_{i,j} (s_ij - sigma_ij)^2`.

1. The diagonal is free: each `theta_i` touches only `Sigma_ii = c + theta_i`, so
   for any `c` the diagonal residuals vanish (`theta_i = s_ii - c`). The
   criterion collapses onto the off-diagonals.
2. Every model off-diagonal equals `c`, so minimizing `sum_{i != j} (s_ij - c)^2`
   gives `c_hat = s_off`.

Then omega's numerator is `(p lambda)^2 psi = p^2 c_hat = p^2 s_off`, and the
implied total variance reproduces the observed total,

```
1' Sigma_hat 1 = sum_i s_ii + p(p-1) c_hat = sum_i s_ii + sum_{i != j} s_ij = 1' S 1,
```

so `omega_hat = p^2 s_off / (1' S 1) = (p/(p-1)) (sum_{i != j} s_ij) / (1' S 1) = alpha`.

Parallel model: pool the uniquenesses, `Sigma(c, theta) = c 1 1' + theta I`. The
diagonal can no longer be matched entry by entry, but the ULS stationarity
condition `partial_theta: sum_i (s_ii - c - theta) = 0` makes the diagonal
residuals sum to zero, which drops the cross term out of `partial_c` and again
gives `c_hat = s_off`. The diagonal is now average-matched
(`Sigma_ii = mean(s_ii)`) rather than exactly matched, but `sum_i theta_hat_i`
is the same as in the tau-equivalent fit, so `1' Sigma_hat 1 = 1' S 1` once more
and `omega_hat = alpha`.

The invariant: omega sees `Sigma_hat` only through `p^2 c_hat` (numerator) and
`1' Sigma_hat 1` (denominator). Both reduce to functions of `s_off` and the
diagonal sum, which ULS preserves under either constraint.

## It is ULS, not least squares in general

The reduction in step 2 produces the *unweighted* off-diagonal mean. A weighted
criterion `sum_{i != j} w_ij (s_ij - c)^2` gives the weighted mean
`c_hat = sum w_ij s_ij / sum w_ij`, which equals `s_off` only when the
off-diagonal weights are constant. Hence:

- ML, GLS, WLS, and DWLS do **not** return alpha. Their weight matrices couple
  the diagonal and off-diagonal residuals (ML does not even match the diagonal),
  so they return different, generally larger omega-type quantities.
- The common loose claim that "alpha is the ML estimate under tau-equivalence"
  is wrong at the estimator level. The equal-weighting of ULS is exactly what
  reproduces alpha's average-off-diagonal-covariance formula.

## Caveats

- **Interior solution only.** The argument uses the unconstrained ULS optimum.
  If `s_off > s_ii` for some item, the nonnegativity-constrained `theta_i >= 0`
  binds, diagonal matching breaks, and constrained-ULS omega departs from alpha.
- **Model-relative.** "Reliability" presupposes a true-score/error split. Alpha
  is the ULS plug-in of *tau-equivalent* reliability; under a congeneric truth
  the actual reliability is omega and alpha underestimates it (`alpha <= omega`).
  Alpha is not "the ULS estimator of reliability" in any model-free sense.

## Citation status

Web audit, 2026-06-03. The two layers have very different provenance.

Canonical (population-level facts about the model, silent on the estimator):

- Guttman, L. (1945). A basis for analyzing test-retest reliability.
  *Psychometrika*, 10(4), 255-282. (alpha is his `lambda_3`.)
- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests.
  *Psychometrika*, 16(3), 297-334.
- Novick, M. R., & Lewis, C. (1967). Coefficient alpha and the reliability of
  composite measurements. *Psychometrika*, 32(1), 1-13. (alpha = reliability iff
  essential tau-equivalence, `alpha <= reliability` otherwise.)
- Lord, F. M., & Novick, M. R. (1968). *Statistical Theories of Mental Test
  Scores.* Addison-Wesley.

Factor/SEM reliability framing and the estimation hierarchy (parallel subset of
tau-equivalent subset of congeneric):

- Joreskog, K. G. (1971). Statistical analysis of sets of congeneric tests.
  *Psychometrika*, 36(2), 109-133.
- McDonald, R. P. (1999). *Test Theory: A Unified Treatment.* Erlbaum. (omega.)
- Graham, J. M. (2006). Congeneric and (essentially) tau-equivalent estimates of
  score reliability. *Educational and Psychological Measurement*, 66(6), 930-944.
- Flora, D. B. (2020). Your coefficient alpha is probably wrong, but which
  coefficient omega is right? *AMPPS*, 3(4), 484-501.

Alpha as a constrained model-based estimator, and the least-squares /
minimum-trace fitting thread (closest the mainstream comes to an estimator-level
treatment):

- Bentler, P. M. (2009). Alpha, dimension-free, and model-based internal
  consistency reliability. *Psychometrika*, 74(1), 137-143.
- ten Berge, J. M. F., & Socan, G. (2004). The greatest lower bound to the
  reliability of a test and the hypothesis of unidimensionality.
  *Psychometrika*, 69(4), 613-625.
- ten Berge, J. M. F., Snijders, T. A. B., & Zegers, F. E. (1981). Computational
  aspects of the greatest lower bound to the reliability and constrained minimum
  trace factor analysis. *Psychometrika*, 46(2), 201-213.

Finding: no single paper states the "alpha = ULS plug-in of the tau-equivalent
model" identity as a named theorem, and treatments that gesture at it usually
say "ML" loosely, which is incorrect at the estimator level (see above). For the
manuscript, the defensible move is to cite Joreskog (1971) for the constrained
tau-equivalent fit and Bentler (2009) for the model-based framing, then state the
ULS step as a short lemma with the two-line proof here. The exp-20
`validation_identity.csv` is the numerical witness
(`omega_hat(ULS tau-equiv) - alpha` to machine precision).
