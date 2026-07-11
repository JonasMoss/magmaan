# Paper Sketch

## Working Question

Can a fully recomputed studentized Wald permutation test provide valid and
useful measurement-invariance inference when the groups are unequal in size or
otherwise non-exchangeable? The comparison is against the ordinary likelihood-
ratio difference test and the multivariate Wald test used in current applied
invariance work.

## Study 1: Reference-Law Study

Study 1 evaluates the reference distribution under a two-group, two-factor CFA.
It follows the design spine of Brace and Savalei (2017): 8 or 16 indicators,
balanced and 1:3 group allocations, and the ordered configural-to-metric-to-
scalar-to-strict ladder. The primary outcome is Type-I error under a fully
invariant measurement model with unequal factor means and variances, followed
by power against separately calibrated configural, metric, scalar, and strict
departures.

Continuous data are generated under normality and moderate/severe Pearson
independent-generator nonnormality. The latter conditions use the `(2, 7)` and
`(3, 21)` skewness/excess-kurtosis settings used in the repository's FIML-FMG
work, avoiding the weak symmetric stress supplied by a multivariate-t
generator. The factor-count question is covered by the two-factor `p=8/16`
design: it varies both the number of indicators per factor and the difference-
test dimension. A one-factor model is reserved for the direct replication in
Study 2; arbitrary 3- or 5-factor expansions are not part of the main
factorial unless the two-factor results show a factor-count interaction.

The categorical-data companion generates ordinal-probit responses but fits the
integer category scores with robust continuous ML. This reflects a common
applied practice and targets the Pearson covariance of the scores, rather than
the latent-response/polychoric estimand. Rhemtulla, Brosseau-Liard, and Savalei
(2012) show why category count and threshold asymmetry must be explicit: robust
continuous ML is most defensible with five or more approximately symmetric
categories, and can be badly biased for sparse or asymmetric categories.
Maydeu-Olivares (2017) supplies the continuous-ML robust-inference framework.

## Study 2: Unbalanced-Data Comparison

Study 2 is a direct comparison slice based on Chen and Chao (2026), rather than
another large factorial. It uses their two-group one-factor, six-indicator CFA,
with group A fixed at 200 and group B in `{400, 600, 800, 1000, 1400, 2000,
3000}` (ratios 1:2 through 1:15), plus matched balanced samples with the same
total N. Metric and scalar alternatives retain the affected-item proportion
(1/3 or 2/3) and the small, large, mixed, and non-uniform departure patterns
from their study. Strict invariance is added as our extension, using the same
affected-item proportions and multiplicative residual-variance departures.
Configural invariance is included as the fitted common-pattern baseline. A
group-specific cross-loading is used only as an auxiliary pattern-misspecification
alternative for its global-fit outcome; it is not relabeled as a conventional
configural invariance difference test.

This slice compares their multivariate Wald test, the ordinary likelihood-ratio
difference test, sandwich Wald, and the studentized permutation Wald. It gives
the paper a direct answer to the current SEM-journal result that multivariate
Wald tests can have better power than likelihood-ratio tests under severe group
imbalance. The permutation method is evaluated as a finite-sample calibration
of that same studentized-Wald idea, not as an unrelated resampling benchmark.

Configural fit is reported in both studies as the common-pattern baseline and
its global-fit outcome. The ordinary, sandwich, and permutation Wald comparisons
apply to the metric, scalar, and strict equality restrictions.

## References

- Brace, J. C., & Savalei, V. (2017). Type I error and power of scaled
  chi-square difference tests for measurement invariance. *Psychological
  Methods*.
- Chen, P.-Y., & Chao, T.-Y. (2026). A comparison of methods for testing
  measurement invariance with unbalanced data. *Structural Equation Modeling*,
  33, 404-411. https://doi.org/10.1080/10705511.2025.2591420
- Maydeu-Olivares, A. (2017). Maximum likelihood estimation of structural
  equation models for continuous data: Standard errors and goodness of fit.
  *Structural Equation Modeling*, 24, 383-394.
  https://doi.org/10.1080/10705511.2016.1269606
- Rhemtulla, M., Brosseau-Liard, P. E., & Savalei, V. (2012). When can
  categorical variables be treated as continuous? A comparison of robust
  continuous and categorical SEM estimation methods under suboptimal
  conditions. *Psychological Methods*, 17, 354-373.
  https://doi.org/10.1037/a0029315
