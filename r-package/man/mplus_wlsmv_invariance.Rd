\name{mplus_wlsmv_invariance}
\alias{mplus_wlsmv_invariance}
\title{Compose ordinal invariance fits}
\description{Constructs and fits the requested ordinal invariance sequence using the Mplus-style WLSMV conventions.}
\usage{
mplus_wlsmv_invariance(model, data, ordered, group, steps = c("configural", "metric", "scalar"),
    estimator = c("WLSMV", "DWLS", "WLS"), missing = c("listwise", "pairwise", "error"),
    group_labels = NULL, optimizer = "nlopt-lbfgs", control = NULL, bounds = NULL,
    pd_gamma = c("overlap", "nominal"), validate = TRUE)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{ordered}{Names of ordinal observed variables.}
\item{group}{Grouping-variable name or group identifiers, as appropriate for the data.}
\item{steps}{Invariance steps to construct: configural, metric or scalar.}
\item{estimator}{Estimation objective. Supported objectives depend on the data kind and fitting route.}
\item{missing}{Missing-data handling: listwise deletion, rejection, or pairwise moments where supported. FIML retains missing observations.}
\item{group_labels}{Explicit group labels in block order.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{bounds}{Lower and upper parameter-bound vectors or a named bounds preset; NULL leaves the route default.}
\item{pd_gamma}{Pairwise fourth-moment covariance convention: overlap or nominal.}
\item{validate}{Validate the constructed invariance models.}
}
\value{An invariance result with the constructed models, fits and comparisons.}
