\name{fcsem_standardized_rows}
\alias{fcsem_standardized_rows}
\title{Factor-component SEM: fcsem_standardized_rows}
\description{Constructs, fits or summarizes the native factor-component SEM representation. These objects use their own fitting route.}
\usage{
fcsem_standardized_rows(fit, vcov = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{vcov}{Explicit covariance matrix of free parameters, in theta order.}
}
\value{A factor-component model specification, fitted model or named post-fit summary, according to the operation.}
