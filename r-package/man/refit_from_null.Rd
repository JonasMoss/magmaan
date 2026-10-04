\name{refit_from_null}
\alias{refit_from_null}
\title{Refit a larger model from a restricted estimate}
\usage{refit_from_null(fit_H1, fit_H0)}
\arguments{
\item{fit_H1}{The larger fitted model.}
\item{fit_H0}{The restricted fitted model, using the same data and estimator.}
}
\value{A new estimate-only \code{magmaan_fit}, with its own convergence diagnostics and native verdict.}
\description{Embeds the restricted estimate using the native, moment-verified nesting map, then replays the larger model's estimator and fitting options with that explicit start. Neither input is modified.}
\details{Supports single-level ML, FIML and all-ordinal DWLS, ULS and WLS fits from \code{fit_model()} or \code{estimate()}. FIML requires retained observations; complete-data and ordinal routes reuse retained sufficient statistics. This explicit lab operation returns the new endpoint even when its verdict fails; callers must inspect convergence diagnostics.}
