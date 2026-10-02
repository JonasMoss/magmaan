\name{frontier_reidentify}
\alias{frontier_reidentify}
\title{Reidentify a sphere fit}
\description{Projects a sphere-coordinate fitted model into the requested model identification.}
\usage{
frontier_reidentify(fit, model, ..., pole_tol = 1e-06)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
\item{pole_tol}{Tolerance for detecting a sphere-coordinate pole.}
}
\value{A fitted model in the requested identification with projection diagnostics.}
