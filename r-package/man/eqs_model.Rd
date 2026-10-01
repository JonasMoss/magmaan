\name{eqs_model}
\alias{eqs_model}
\title{Construct a model from EQS equations}
\description{Translate explicit EQS covariance-model sections into the shared
magmaan model. Estimation and inference use magmaan, not the EQS application.}
\usage{eqs_model(syntax, observed_names = NULL)}
\arguments{
\item{syntax}{One string containing \code{/EQUATIONS}, \code{/VARIANCES} and
optionally \code{/COVARIANCES}. Unambiguous abbreviations with at least three
letters, \code{!} comments, ranges and an optional \code{/END} are accepted.}
\item{observed_names}{Optional names in EQS data-column order. \code{V1} maps
to the first name, \code{V2} to the second, etc. Without names, the data should
have columns named \code{V1}, \code{V2}, etc.}
}
\value{A \code{magmaan_model_spec} for \code{fit_model()}, \code{prepare_model()}
or \code{magmaan::magmaan()}. \code{eqs_source} retains the original input;
\code{syntax} is its explicit lavaan projection for rebuilding and refitting.}
\details{This initial subset supports single-group continuous covariance models.
Every equation has one distinct error variable with a fixed unit coefficient:
\code{E} for observed variables and \code{D} for factors. Every factor must
have an observed indicator. Error numbering need not match equation numbering.
A coefficient followed by \code{*} is free; a preceding number supplies its
start. A coefficient without \code{*} is fixed (implicit coefficient one).
Missing independent variances are free and missing covariances are zero.
Identification follows explicit EQS specifications; no marker loading or
single-indicator residual fix is added. Free parameters without supplied starts
use magmaan's start policy, not EQS's internal start algorithm.

Job files, \code{/MODEL}, \code{/LABELS}, constraints, means (\code{V999}),
multiple groups, shared errors, nonunit error paths and error/predictor or
\code{E}/\code{D} covariances are rejected. Observed indicators participating
in structural regressions are also outside this subset; separate observed
predictors and purely observed path models are supported.
Supply data and estimator separately.
Scientific exponents must have a sign, e.g. \code{1e-3}, to distinguish
\code{1E1}, which is a unit coefficient on error \code{E1}.}
\examples{
model <- eqs_model("
/EQUATIONS
V1 = *F1 + E1;
V2 = .7*F1 + E2;
V3 = .8*F1 + E3;
/VARIANCES
F1 = 1;
E1-E3 = .5*;
", observed_names = c("x1", "x2", "x3"))
}
