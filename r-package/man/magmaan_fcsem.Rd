\name{magmaan_fcsem}
\alias{magmaan_fcsem}
\title{Factor-component SEM: magmaan_fcsem}
\description{Constructs, fits or summarizes the native factor-component SEM representation. These objects use their own fitting route.}
\usage{
magmaan_fcsem(model, data, ..., missing = c("listwise", "error"), control = NULL)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
\item{missing}{Missing-data handling: listwise deletion, rejection, or pairwise moments where supported. FIML retains missing observations.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
}
\value{A factor-component model specification, fitted model or named post-fit summary, according to the operation.}
