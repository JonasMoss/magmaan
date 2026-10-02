\name{bounds_standard}
\alias{bounds_standard}
\title{Parameter bounds: bounds_standard}
\description{Constructs parameter bounds in model free-parameter order. Sample-based bounds use the supplied data summaries.}
\usage{
bounds_standard(model, data)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{A list of lower and upper parameter-bound vectors.}
