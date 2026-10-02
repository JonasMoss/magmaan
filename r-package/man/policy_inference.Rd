\name{policy_inference}
\alias{policy_inference}
\title{Compose ordinary-user inference}
\description{Requests the C++ ordinary-user inference policy for the fitted estimator and data regime. Unsupported regimes remain explicit.}
\usage{
policy_inference(fit, data = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{An inference bundle with availability status, covariance and test summaries where supported.}
