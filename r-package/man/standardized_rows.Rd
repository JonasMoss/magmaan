\name{standardized_rows}
\alias{standardized_rows}
\title{Row-level standardized estimates}
\usage{standardized_rows(fit, vcov)}
\arguments{
\item{fit}{A lab fit.}
\item{vcov}{Explicit covariance in the fit's free-parameter coordinates.}}
\value{A list of \code{std.lv}, \code{std.lv.se}, \code{std.all} and
\code{std.all.se} vectors in partable row order.}
\description{Includes fixed markers, group-specific transformations and ordinal
latent-response residual variances. Standard errors use the delta method.
Defined parameters have missing values because their expressions have no
specified standardized scale.}
