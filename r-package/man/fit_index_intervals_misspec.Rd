\name{fit_index_intervals_misspec}
\alias{fit_index_intervals_misspec}
\title{Candidate continuous misspecification-robust fit-index intervals}
\usage{fit_index_intervals_misspec(fit, level = 0.90)}
\arguments{
\item{fit}{A fitted magmaanlab model retaining raw observations.}
\item{level}{Confidence level strictly between zero and one, default 0.90.}
}
\description{Lab-only Wald intervals for continuous complete-data ML and FIML,
including independent groups, random x and affine equality constraints.
Points agree with \code{policy_fit_measures()}. The variance candidate has not
been adopted or coverage-calibrated.}
\details{
The casewise likelihood contrast gives the first-order user discrepancy
variance V1. The second-order V2 is \eqn{2\mathrm{tr}(M^2)/N^2}, where
\eqn{M=(J_s^{-1}-\Delta J_0^{-1}\Delta^\top)K_s} uses saturated observed
information, constraint-reduced model observed information, the implied-moment
Jacobian and the saturated score outer product. Covariance-only ML includes
its profiled group-mean nuisance parameters.

\code{wald_f} maps a Wald interval for the corrected discrepancy to RMSEA.
\code{wald_sqrt} uses its square-root coordinate and requires a positive
corrected discrepancy. CFI and TLI use the joint user/baseline first-order delta
method plus V2 for the user discrepancy; their intervals coincide for both
variants. CFI endpoints are clipped to [0,1]; TLI endpoints are unbounded.
RMSEA lower bounds are clipped at zero. Ordinal, ML2S, other estimators,
clustered, penalized and failed fits return typed unavailable intervals.
}
\value{A data frame with index, variant, estimate, lower, upper, V1, V2 and
reason. V1/V2 are always on the user discrepancy scale, before mapping to an
index. The details attribute includes casewise and fit discrepancies, their
corrected point, Takeuchi and quadratic traces, sample size and level.
Unavailable intervals have NA bounds and explicit reasons.}
\seealso{\code{\link{policy_fit_measures}}}
