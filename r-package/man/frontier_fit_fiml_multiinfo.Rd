\name{frontier_fit_fiml_multiinfo}
\alias{frontier_fit_fiml_multiinfo}
\title{Experimental fitting route: frontier_fit_fiml_multiinfo}
\description{Fits the named objective through an explicit experimental covariance-domain route. PSD variants constrain covariance blocks; multiinfo variants apply the selected barrier target. Inference remains a separate operation.}
\usage{
frontier_fit_fiml_multiinfo(model, data, eta = 1.25, weight = NULL, optimizer = "port", control = NULL,
    bounds = NULL, target = c("joint", "determinacy"))
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{eta}{Compatibility barrier parameter; used when weight is omitted.}
\item{weight}{Weight selector or explicit fitting weight as required by the route.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{bounds}{Lower and upper parameter-bound vectors or a named bounds preset; NULL leaves the route default.}
\item{target}{Barrier target: joint covariance or factor-score determinacy.}
}
\value{A fitted-model list with parameter estimates and route diagnostics.}
