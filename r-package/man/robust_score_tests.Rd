\name{modification_indices_robust}
\alias{modification_indices_robust}
\alias{score_tests_robust}
\title{Robust modification indices and equality-release score tests}
\description{Compute one-parameter modification indices or one-at-a-time
equality-release tests, with a direction-specific sandwich correction.}
\usage{
modification_indices_robust(fit, data = NULL, weight = NULL,
  bread = "expected", moments = "structured", cov = "empirical",
  candidates = "all", include_loadings = TRUE, include_covariances = TRUE,
  information = "expected", estimated_weight = FALSE)
score_tests_robust(fit, data = NULL, weight = NULL,
  bread = "expected", moments = "structured", cov = "empirical",
  estimated_weight = FALSE)
}
\arguments{
\item{fit}{A fitted magmaanlab model.}
\item{data}{Fitting observations as a data frame, matrix, or grouped raw-data
blocks. FIML uses retained observations when this is omitted; an explicit
FIML raw-data object may also carry a missingness mask.}
\item{weight}{Explicit fitting weight for continuous WLS. Unsupported for FIML.}
\item{bread,information}{Sandwich sensitivity and score-information choices.
For FIML, omitted arguments select \code{"observed"}; explicitly requesting
\code{"expected"} is unsupported. Other estimator defaults remain expected.}
\item{moments,cov}{Continuous moment and covariance conventions. FIML accepts
only the default \code{moments="structured"}, \code{cov="empirical"} combination,
which selects observed-pattern casewise score covariance.}
\item{candidates}{\code{"fixed"} for existing fixed parameter rows or
\code{"all"} to include absent cross-loadings and covariances.}
\item{include_loadings,include_covariances}{Whether to enumerate absent loading
and covariance candidates.}
\item{estimated_weight}{Include supported estimated-weight influence terms.
FIML has no second-stage weight and rejects this option.}
}
\details{FIML uses the analytic observed information for both its statistic and
bread, and the observed-pattern casewise score meat. It needs no saturated-model
EM computation for these tests. Explicit data rebuild their missingness pack;
omitted data reuse the retained pack when available. Alternative FIML moment or
covariance recipes and supplied second-stage weights are rejected.

Identification-only releases are excluded. Ordinal and mixed-ordinal methods
use their retained categorical moment covariance and weight conventions; the
continuous \code{bread}, \code{moments}, and \code{cov} choices do not apply there.}
\value{A data frame with candidate names and groups, ordinary \code{mi}, EPCs,
\code{mi.scaled}, \code{scaling.factor}, and the robust \code{pvalue}.}
