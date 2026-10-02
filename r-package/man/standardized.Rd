\name{standardized}
\alias{standardized}
\title{Standardized parameter estimates}
\description{Standardizes parameter estimates and propagates the supplied covariance.}
\usage{
standardized(fit, vcov, type = c("all", "lv"))
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{vcov}{Explicit covariance matrix of free parameters, in theta order.}
\item{type}{Standardization: all variables or latent variables only.}
}
\value{A table of standardized parameter estimates and uncertainty.}
