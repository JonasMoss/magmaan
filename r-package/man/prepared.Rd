\name{prepare_model}
\alias{prepare_model}
\alias{prepare_data}
\alias{prepare_weight}
\alias{estimate}
\title{Reusable native model, data and weight preparation}
\description{Separate structural model preparation, dataset preparation,
estimation weights and estimation. Inference is requested separately.}
\usage{
prepare_model(model, ..., prototype = NULL)
prepare_data(model, data, kind = NULL, missing = c("error", "listwise"))
prepare_weight(data, method = c("DWLS", "WLS", "ULS", "GLS", "DLS"),
               W = NULL, full = TRUE, dls_a = 0.5)
estimate(model, data, estimator = NULL, weight = NULL,
         optimizer = NULL, control = NULL, bounds = NULL,
         covariance = NULL, psd = FALSE, barrier = NULL, dls_a = 0.5,
         options = NULL)
}
\arguments{
\item{model}{Model syntax or a model specification for \code{prepare_model};
a prepared model for the other functions.}
\item{...}{Options passed to \code{model_spec} when model is syntax.}
\item{prototype}{Data frame or named matrix supplying categorical levels and
order. No empirical moments or starts are retained from it.}
\item{data}{A data frame, named matrix or sample-moment list for
\code{prepare_data}; a prepared dataset for the other functions.}
\item{kind}{\code{"moments"}, \code{"raw"} (FIML), \code{"ordinal"} or
\code{"mixed"}. Default follows the model schema.}
\item{missing}{Reject missing observations or delete incomplete rows for moment
data. Raw FIML data retain missingness.}
\item{method}{Fixed weight: ULS, GLS/NT, DWLS, WLS/ADF or DLS for complete
continuous and all-ordinal data. Lower-case weight names are also accepted.
Mixed data retain DWLS/WLS.}
\item{W}{Optional weight matrix or per-group list in model moment
order. Otherwise empirical weights are constructed from raw observations.}
\item{full}{For categorical data, retain full Gamma for post-fit inference.
FALSE requests only the DWLS diagonal (or no Gamma for ULS) when W is omitted.
GLS, DLS, WLS and supplied ordinal weights retain full Gamma.
DWLS does not invert full Gamma.}
\item{estimator}{Continuous or all-ordinal ML, ULS, GLS, DWLS, WLS or DLS;
FIML for raw data; DWLS or WLS for mixed data. Default follows
the weight, or otherwise the data kind.}
\item{weight}{Optional prepared weight from this dataset, or a shared weight
name: \code{"uls"}, \code{"nt"}, \code{"dwls"}, \code{"adf"}, or \code{"dls"}.
Required weights are prepared internally if omitted.}
\item{dls_a}{Fixed DLS mixing scalar in [0, 1]. Zero selects NT and one ADF.
When a prepared weight is supplied its stored scalar governs.}
\item{optimizer, control}{Existing native optimizer name and control list.
A keyed data-frame \code{control$start} supplies fit-local start hints.}
\item{options}{Versioned fitting choices (preset, starts, optimizer and
convergence) for unrestricted continuous ML or FIML, as in \code{fit_model}.}
\item{bounds}{Explicit lower/upper bound list in free-parameter order. Not
supported by the prepared FIML path.}
\item{covariance}{Model covariance policy: \code{"unrestricted"},
\code{"psd"} or \code{"barrier"}. NULL follows \code{psd}.
Mixed models support unrestricted and PSD fitting; barriers are deferred.}
\item{psd}{Compatible spelling for \code{covariance = "psd"}. Explicit
contradictory requests are rejected.}
\item{barrier}{Options for barrier fitting: a list with \code{target}
(\code{"joint"} or \code{"determinacy"}) and \code{weight} (default 0.25).
The weight is finite and non-negative; zero disables the penalty.}
}
\value{Preparation returns immutable, process-local handles. A prepared model
exposes its specification; a weight exposes \code{W} for continuous data and
\code{stats} for categorical inference. Estimation returns the existing
\code{magmaan_fit} list with convergence and admissibility diagnostics.}
\details{Rebuild handles after serialization and in each worker. A new dataset
requires new data and weight preparation, but reuses the model. Category levels,
group order, observable order and mean-structure convention must remain
compatible. Group labels must be declared explicitly. Covariances supplied as
sample moments use divisor N.

The covariance policy changes final model fitting without repairing supplied
moments. A barrier minimizes \code{f - weight / N * P}, with \code{fmin}
retaining the unpenalized discrepancy and \code{penalty$penalized_fmin}
recording the fitted objective. Positive-weight barriers replace additional
bounds. Their general sampling/inference contract is not yet validated.
Ordinal ML fits associations with Stage-1 thresholds held fixed; it does not
provide Gaussian likelihood or generic post-fit sampling inference.

FIML retains its missingness pack but does not compute saturated H1 during
estimation unless versioned fitting options require it. Categorical inference
needs full Gamma; obtain it separately with
\code{prepare_weight(data, "DWLS", full = TRUE)$stats} and pass it to an
existing explicit inference function, without refitting.

Versioned fitting options use the same configured engines as \code{fit_model()}
for continuous ML/FIML and all-ordinal DWLS with unrestricted covariance,
including \code{options = list(preset = "lavaan-0.7.2")}.

The interface covers the estimators listed above.
ML2S raises a typed \code{magmaan_unsupported_estimator} error.
Two-level, FC-SEM, SAM and frontier methods continue using their existing
entry points, which remain supported without deprecation warnings. Categorical
weight preparation currently recomputes stage-one moments in the existing core
builder; fitting itself reuses the prepared objects.}
