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
prepare_weight(data, method = c("DWLS", "WLS", "ULS"), W = NULL, full = TRUE)
estimate(model, data, estimator = NULL, weight = NULL,
         optimizer = NULL, control = NULL, bounds = NULL)
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
\item{method}{Weight method: DWLS, WLS or ULS.}
\item{W}{Optional continuous weight matrix or per-group list in model moment
order. Otherwise empirical weights are constructed from raw observations.}
\item{full}{For categorical data, retain full Gamma for post-fit inference.
FALSE requests only the DWLS diagonal (or no Gamma for ULS). WLS needs full
Gamma regardless. DWLS does not invert full Gamma.}
\item{estimator}{Continuous ML, ULS, GLS, WLS or DWLS; FIML for raw data;
ULS, DWLS or WLS for ordinal data; DWLS or WLS for mixed data. Default follows
the weight, or otherwise the data kind.}
\item{weight}{Optional prepared weight from this dataset. WLS/DWLS prepare a
weight internally if omitted.}
\item{optimizer, control}{Existing native optimizer name and control list.}
\item{bounds}{Explicit lower/upper bound list in free-parameter order. Not
supported by the prepared FIML path.}
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

FIML retains its missingness pack but does not compute saturated H1 during
estimation. Categorical inference needs full Gamma; obtain it separately with
\code{prepare_weight(data, "DWLS", full = TRUE)$stats} and pass it to an
existing explicit inference function, without refitting.

The initial interface covers the estimators listed above. Specialized ML2S,
two-level, FC-SEM, SAM and frontier methods continue using their existing
entry points, which remain supported without deprecation warnings. Categorical
weight preparation currently recomputes stage-one moments in the existing core
builder; fitting itself reuses the prepared objects.}
