\name{lav_residuals}
\alias{lav_residuals}
\title{Standardized residual diagnostics}
\description{Returns standardized residual matrices, residual SE and z statistics, SRMR and block summaries. Estimated-weight inference for continuous least squares requires raw observations.}
\usage{
lav_residuals(fit, estimated_weight = TRUE, data = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{estimated_weight}{Whether to include estimated-weight influence in continuous least-squares residual inference; requires raw data.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{A list of residual matrices and block-level diagnostic summaries.}
