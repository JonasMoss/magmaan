\name{association_ml_ij}
\alias{association_ml_ij}
\title{Exact sampling sandwich for ordinal association ML}
\usage{association_ml_ij(fit, ordinal_stats)}
\arguments{
\item{fit}{An all-ordinal association-ML fit in DELTA coordinates.}
\item{ordinal_stats}{The complete-row Stage-1 statistics used for fitting.}
}
\description{Lab-only evaluation-point inference for the Gaussian discrepancy on
estimated polychoric correlations. Uses empirical-Jacobian first-stage rows,
observed sensitivity including correlation-map curvature, and fixed-allocation
group sampling. Ordinary inference remains unavailable.}
\value{A list with \code{available}; unavailable results include \code{reason}
and optionally \code{detail}. Available results contain the objective
\code{value}, active \code{score}, observed \code{H}, meat \code{B},
coordinate map \code{K}, group target derivatives \code{D},
\code{vcov_active}, full \code{vcov}, and group lists
\code{influence_active} and \code{influence}. Influence rows are unscaled:
summing their cross-products and dividing by total sample size squared
reconstructs the covariance. Full rows include saturated thresholds and their
cross-covariance with association parameters.}
\details{Requires complete all-ordinal rows, saturated thresholds, fixed response
scales, linear association constraints, an identified model, positive observed
sensitivity and interior primitive covariance matrices. Missing rows, mixed
indicators, nonlinear constraints, covariance boundaries and penalties are
unsupported. Sensitivity is computed by central differences of the analytic
score; no global, nested or modification-index reference law is supplied.}
