\name{bounds_loading}
\alias{bounds_loading}
\title{Parameter bounds: bounds_loading}
\description{Constructs parameter bounds in model free-parameter order. Sample-based bounds use the supplied data summaries.}
\usage{
bounds_loading(model, data)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{A list of lower and upper parameter-bound vectors.}
