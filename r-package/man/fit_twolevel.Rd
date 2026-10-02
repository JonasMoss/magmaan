\name{fit_twolevel}
\alias{fit_twolevel}
\title{Fit a two-level structural equation model}
\description{Fits the native two-level model using cluster identifiers and optional group identifiers.}
\usage{
fit_twolevel(model, data, cluster, group = NULL, optimizer = "nlopt-lbfgs", control = NULL,
    bounds = NULL, missing = c("listwise", "error"))
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{cluster}{Cluster-variable name or cluster identifiers for two-level fitting.}
\item{group}{Grouping-variable name or group identifiers, as appropriate for the data.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{bounds}{Lower and upper parameter-bound vectors or a named bounds preset; NULL leaves the route default.}
\item{missing}{Missing-data handling: listwise deletion, rejection, or pairwise moments where supported. FIML retains missing observations.}
}
\value{A two-level fitted-model result.}
