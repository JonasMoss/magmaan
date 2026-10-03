\name{scores}
\alias{scores}
\alias{prepare_inference}
\alias{score_components}
\alias{score_components_from_matrices}
\alias{project_scores}
\alias{score_quadratic}
\alias{score_spectrum}
\alias{score_sandwich}
\alias{resample_scores}
\alias{quadratic_reference}
\alias{calibrate_quadratic}
\alias{inference_information}
\alias{parameter_covariance}
\alias{wald_test}
\title{Reusable score ingredients and explicit inference}
\description{Prepare an immutable native inference snapshot, obtain scores and
geometry, and choose projection, calibration or resampling separately. These
methods-developer interfaces support complete-data ML, direct FIML and the
unregularized fixed-normal-theory ML2S global score construction.}
\usage{
prepare_inference(fit, data = NULL)
scores(object, data = NULL, space = c("parameter", "saturated"))
score_components(object, data = NULL, H1 = NULL,
  sensitivity = c("observed", "expected", "observed-h1",
                  "observed-shrink-light", "observed-shrink-sqrt"),
  metric = c("expected", "observed", "observed-h1"))
score_components_from_matrices(score, rows, sensitivity, metric = sensitivity,
  nuisance = matrix(0, length(score), 0L), directions = diag(length(score)),
  influence_rows = FALSE)
project_scores(components, retain_rows = FALSE, center = FALSE)
score_quadratic(score, metric, meat = NULL)
score_spectrum(projected)
score_sandwich(projected)
resample_scores(projected, n_flips = 999L, seed = 1,
  multiplier = c("rademacher", "mammen", "two-point", "gaussian",
                 "centered-exponential"), two_point_skewness = 1)
quadratic_reference(statistic, df, eigenvalues = NULL)
calibrate_quadratic(object, methods = "peba4")
inference_information(context, type = c("observed", "expected"))
parameter_covariance(context, information, meat = NULL)
wald_test(object, R, vcov, q = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Fitting raw data for complete ML, optionally a prepared dataset.
FIML and ML2S use the raw observations stored on the fit.}
\item{object,context}{A fit or prepared inference snapshot for score construction;
a projected score or quadratic reference for calibration; a fit, snapshot or
estimate vector for Wald inference.}
\item{space}{Parameter or saturated-moment likelihood coordinates. ML2S
influence contributions are available through \code{score_components}.}
\item{H1}{Optional unrestricted model specification or fit for affine nested
ML/FIML scores. No unrestricted fit is required. Null selects global GOF.}
\item{sensitivity,metric}{Nuisance sensitivity and quadratic metric choices.
For supplied-matrix \code{score_quadratic}, metric is a positive-definite matrix.}
\item{rows,nuisance,directions,influence_rows}{Supplied casewise rows, nuisance
and tested direction matrices, and whether the rows are influence contributions
rather than likelihood scores. Likelihood rows must sum to the observed score.}
\item{components,projected}{Reusable objects produced by the preceding stage.}
\item{retain_rows}{Retain projected rows for multiplier resampling.}
\item{center}{Globally center covariance rows. Does not change the observed
score. This is not within-pattern centering.}
\item{score}{An observed score or contrast vector in total-likelihood units.}
\item{meat}{Optional symmetric score covariance in corresponding total units.}
\item{statistic,df,eigenvalues}{A precomputed quadratic statistic, nominal
positive integer degrees of freedom, and optional spectrum including zero weights.}
\item{methods}{One or more of std, sb, peba2, peba4, all. Only all requests
exact-mixture integration.}
\item{n_flips,seed,multiplier,two_point_skewness}{Explicit multiplier simulation
settings. Reuses the projected metric factorization and retained rows.}
\item{type}{Expected Fisher or realized observed information.}
\item{information}{Previously computed total parameter information.}
\item{R,q,vcov}{Wald restriction matrix, target vector (zero by default), and
previously computed parameter covariance.}
}
\details{The preparation snapshot owns its native model/data/parameter state.
It is process-local; reprepare after serialization or changes to the model or
sample. New geometry never draws multipliers or computes p-values. Projection
retains a Cholesky factor; spectrum construction is separate and its result can
be reused for many calibrations. SB-only calibration of projected scores uses a
trace without a spectrum. PSD meat is allowed for mixture inference; the
separate sandwich operation requires full rank.

Likelihood scores have the sign of the log-likelihood derivative and summed
normalization. ML2S instead exposes the observed Stage-2 score separately from
its propagated Stage-1 influence rows: those rows need not sum to the observed
score. Global and nested targets use different coordinate spaces. Geometry
support restrictions follow the corresponding legacy score routines.

The snapshot can also be passed to \code{fmg_tests}, \code{fmg_nested}, and
\code{robust_nested_lrt}. Their native fit context is reused; these legacy
procedures still compute their own method-specific geometry. Save an existing
LR/GOF spectrum with \code{quadratic_reference} to recalibrate without repeating
that computation. Information and covariance preparation for Wald currently
support ML/FIML; ML2S uses its existing two-stage covariance interface.}
\value{Locked environments hold prepared snapshots, components, projections
and reference laws. Scores and multiplier summaries are lists, information and
covariance are matrices, and calibration returns a data frame. Wald returns the
existing statistic, df and p-value list.}
