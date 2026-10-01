# Matrix and numerator conventions

Retained technical appendix. The merged report owns current evidence/activity.

## What EQS and the papers actually specify

There are two different sample-size conventions, and they should not be
silently unified.

- Savalei and Falk's Equation (4) defines the two-stage statistic as
  $d(n-1)F_{ML}(\widetilde\theta)/\operatorname{tr}(\widehat\Omega_1U_2)$.
- Their direct-FIML Equation (6), and EQS Manual Equations (12.6)--(12.7),
  scale the raw likelihood ratio
  $T_{ML}=-2\{\ell(\widehat\theta)-\ell(\widehat\beta)\}$ directly. There is
  no extra $n-1$ multiplier to substitute in this route.

The public EQS 6 manual predates the study's two-stage implementation: it says
that this missing-data two-stage method was not then available, whereas the
article used EQS 6.1 with `MISSING=TS`. Consequently, the public manual cannot
independently confirm the 6.1 two-stage numerator. Equation (4) and the supplied
6.1 syntax are the relevant evidence; the merged report separately compares the two numerator choices. For direct FIML, by contrast, the public manual documents the named
Yuan--Bentler route in detail.

For direct FIML, the [EQS 6 program manual](https://www.mvsoft.com/wp-content/uploads/2021/04/EQS_6_Prog_Manual_422pp.pdf)
calls the output a Yuan--Bentler correction and writes

$$
T_{YB}=\frac{T_{ML}}{c},\qquad
c=\frac{\operatorname{tr}(U_\beta V_\beta)}{d},
$$

with $V_\beta=A_\beta^{-1}B_\beta A_\beta^{-1}$ explicitly described as the
asymptotic covariance of the **saturated** estimator. Its Equation (12.11)
uses the simpler projected residual matrix

$$
U_\beta=A_\beta-A_\beta\Delta
  (\Delta^\top A_\beta\Delta)^{-1}\Delta^\top A_\beta,
$$

not the full structured-parameter observed Hessian. The manual says observed
information is the default finite-sample estimate of $A_\beta$, replaceable by
Fisher expected information on request. It does not say clearly whether the
$A_\beta$ inside $U_\beta$ and the breads inside saturated $V_\beta$ must be
evaluated at the same finite-sample point.

That omission is where one diagnostic expected-information hybrid lives:

$$
U_\beta=U\{A^E_\beta(\widehat\beta_0)\},\qquad
V_\beta=
  A^E_\beta(\widehat\beta)^{-1}B_\beta(\widehat\beta)
  A^E_\beta(\widehat\beta)^{-1}.
$$

Savalei and Falk's Appendix B requests `SE=EXACT`, which they explain as an
analytic observed-information calculation, and the article says observed
information was used throughout. The expected-information fingerprint in the merged report is not licensed
by that written method. Without EQS 6.1 output/source we cannot tell whether
a different path was selected, hard-coded, or mimicked by another finite-sample
difference.

## Named MLR check

Magmaan's named Yuan--Bentler/Mplus MLR statistic is not a forty-ninth setting
of the crossed direct-FIML grid. It computes a scalar difference of traces,

$$
c_{\mathrm{MLR}}
=\frac{\frac{1}{2}\mathrm{tr}(H_1^{-1}J_1)
      -\frac{1}{2}\mathrm{tr}(H_0^{-1}J_0)}{d},
$$

using its separately fitted saturated $H_1$ blocks and the full observed
structured-parameter Hessian under $H_0$. In incomplete data this is not
identical to the superficially similar mixed moment-space row in the 48-choice
grid.



## Generator conventions

Kurtosis in the runner is excess kurtosis. The selected increasing Fleishman
branch preserves the target covariance through a positive-definite latent-normal
correlation matrix. A decreasing root can be a global reflection of the same
joint law; nonmonotone alternatives may instead generate a different joint law
or fail to match negative correlations. Original `dgp_roots.csv` files retain
these distinctions. Treat a feasible alternate branch as a separate sensitivity
condition, not a numerical root interchangeable with the original run.

## Complete option enumeration

The original result CSVs enumerate the full crossed grid. `Invalid` counts samples with a nonpositive scaling factor, for which the
scaled chi-square is undefined. Rejection rates and mean scales use the
remaining usable samples. `Gap` is the rejection rate minus the corresponding
published rate, in percentage points.

The column notation separates decisions that the earlier labels accidentally
ran together:

- **Point** is where a matrix is evaluated: $H_1$ is the unrestricted saturated
  FIML estimate and $H_0$ is the fitted structured SEM. These labels do not
  change the null hypothesis being tested; every row tests the same SEM.
- **Type** applies to curvature matrices only: expected (E) means expected
  Hessian and observed (O) means the sample Hessian. “Expected” does not mean
  imputed or unobserved data. Under incomplete-data FIML, E and O need not agree
  in a finite sample.
- **B scores** is always an empirical score outer product, so it has an
  evaluation point but no expected/observed choice.
- **U: θ block** distinguishes the projected block $Delta^\top A\Delta$ from
  the full observed structured-parameter Hessian
  $H_{\theta\theta}^{O}(H_0)$, which additionally contains the second-derivative
  terms of the moment map.

### Direct FIML: all 48 choices

Each row independently chooses the $A$ used to construct $U$, the inner
$\theta$-curvature used inside $U$, the $A$ used as the sandwich bread, and the
point used for the empirical score meat $B$. The six $U$ constructions are
crossed with four sandwich breads and two score points.

### Two-stage FIML: all 32 choices

The same split notation makes the two-stage table parallel. The four
incomplete-data Stage-1 breads and two score points are crossed with the four
complete-data Stage-2 information matrices. The source-form analogue is $H_1$
observed Stage-1 bread, $H_1$ score meat, and $H_0$ observed Stage-2
information.
