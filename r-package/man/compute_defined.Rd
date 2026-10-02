\name{compute_defined}
\alias{compute_defined}
\title{Evaluate defined model parameters}
\description{Evaluates := expressions and their delta-method uncertainty using an explicit parameter covariance.}
\usage{
compute_defined(model, fit, vcov)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{fit}{A fitted magmaan model.}
\item{vcov}{Explicit covariance matrix of free parameters, in theta order.}
}
\value{A table of defined parameters and uncertainty summaries.}
