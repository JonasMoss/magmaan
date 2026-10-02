\name{factor_scores}
\alias{factor_scores}
\title{Compute factor scores}
\description{Computes factor scores from raw observations. The default is regression for continuous fits and EBM for ordinal fits; explicit choices include regression, bartlett, EBM, ML and EAP.}
\usage{
factor_scores(fit, data, method = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{method}{Method selector; see the available choices in the usage signature and details.}
}
\value{Factor-score estimates returned by the native scoring operation.}
