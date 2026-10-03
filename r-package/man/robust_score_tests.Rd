\name{modification_indices_robust}
\alias{modification_indices_robust}
\alias{score_tests_robust}
\title{Robust modification indices and equality-release score tests}
\description{Compute one-parameter modification indices or one-at-a-time
equality-release tests, with a direction-specific sandwich correction.}
\usage{
modification_indices_robust(fit, data = NULL, weight = NULL,
  bread = "observed", moments = "structured", cov = "empirical",
  candidates = "all", include_loadings = TRUE, include_covariances = TRUE,
  information = "expected", estimated_weight = FALSE)
score_tests_robust(fit, data = NULL, weight = NULL,
  bread = "observed", moments = "structured", cov = "empirical",
  estimated_weight = FALSE)
}
\arguments{
\item{fit}{A fitted magmaanlab model.}
\item{data}{Fitting observations as a data frame, matrix, or grouped raw-data
blocks. FIML uses retained observations when this is omitted; an explicit
FIML raw-data object may also carry a missingness mask. Two-stage (ML2S) fits
use their retained Stage-1 data and reject \code{data}.}
\item{weight}{Fitting weight for continuous WLS-computed fits. Fits record
their weight in \code{fit$W}, so this is needed only for fits without that
record; otherwise it must equal the recorded weight or a positive multiple of
it. Unsupported for FIML and ML2S.}
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
\item{estimated_weight}{Include supported estimated-weight influence terms,
following the weight recipe the fit records (NT, ADF, DWLS, or DLS with its
mixing weight; ULS has none). Supplied weights have no recipe and ordinal NT
and DLS weights have no derived influence, so both fail with an
\code{UnsupportedInference} error. FIML has no second-stage weight and rejects
this option.}
}
\details{FIML uses the analytic observed information for both its statistic and
bread, and the observed-pattern casewise score meat. It needs no saturated-model
EM computation for these tests. Explicit data rebuild their missingness pack;
omitted data reuse the retained pack when available. Alternative FIML moment or
covariance recipes and supplied second-stage weights are rejected.

Continuous ULS/GLS/WLS support empirical covariance from fitting observations,
or model-implied normal-theory covariance from structured or unstructured
moments. WLS-computed fits (WLS, DWLS, DLS, supplied W) use the recorded
fitting weight for either source; the correction reduces to the ordinary
statistic only when that weight is the inverse of the chosen covariance. Browne-unbiased covariance is not implemented
for continuous LS. Estimated-weight mode requires raw observations and empirical
covariance; other covariance choices are rejected.

Two-stage (ML2S) fits report \code{mi} as the naive Stage-2 statistic on the
Stage-1 EM moments (attribute \code{mi_type = "naive_stage2"}) and
\code{mi.scaled} with the Stage-1 moment covariance as meat. They use the
recorded Stage-2 weight, expected information and the default
\code{bread}, \code{moments} and \code{cov}; a DWLS, ADF or DLS Stage-2 weight
adds its data influence with \code{estimated_weight = TRUE}. The naive NT row
uses the structured information; lavaan's two-stage \code{modindices()} uses
the unstructured one and differs.

Identification-only releases are excluded. Ordinal and mixed-ordinal methods
use their retained categorical moment covariance and weight conventions; the
continuous \code{bread}, \code{moments}, and \code{cov} choices do not apply there.}
\value{A data frame with candidate names and groups, ordinary \code{mi}, EPCs,
\code{mi.scaled}, \code{scaling.factor}, and the robust \code{pvalue}.}

\details{Generic modification_indices() and score_tests() compose empirical robust inference. To request the historical unscaled comparator explicitly, use cov = "model_implied", estimated_weight = FALSE, and the desired bread/information. Observed-sensitivity LS and ML2S score laws that are not implemented fail explicitly; selecting expected bread requests a comparator.}
