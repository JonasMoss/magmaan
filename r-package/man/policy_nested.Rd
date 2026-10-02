\name{policy_nested}
\alias{policy_nested}
\title{Compose ordinary-user nested inference}
\description{Requests the C++ ordinary-user nested policy for a restricted and unrestricted fit.}
\usage{
policy_nested(fit_H1, fit_H0, data = NULL)
}
\arguments{
\item{fit_H1}{Unrestricted (alternative) fitted model.}
\item{fit_H0}{Restricted (null) fitted model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{A nested inference bundle with availability status and supported test summaries.}
