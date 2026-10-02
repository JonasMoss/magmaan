\name{composite_weights}
\alias{composite_weights}
\title{Composite score weights}
\description{Computes composite weights and uncertainty from a fit and an explicitly supplied parameter covariance.}
\usage{
composite_weights(fit, vcov)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{vcov}{Explicit covariance matrix of free parameters, in theta order.}
}
\value{Native composite-weight results.}
