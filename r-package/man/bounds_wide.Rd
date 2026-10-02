\name{bounds_wide}
\alias{bounds_wide}
\title{Parameter bounds: bounds_wide}
\description{Constructs parameter bounds in model free-parameter order. Sample-based bounds use the supplied data summaries.}
\usage{
bounds_wide(model, data)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{A list of lower and upper parameter-bound vectors.}
