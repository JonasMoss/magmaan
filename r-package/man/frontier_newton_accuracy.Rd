\name{frontier_newton_accuracy}
\alias{frontier_newton_accuracy}
\title{Local Newton accuracy diagnostic}
\description{Computes the local Newton distance for complete-data ML in units of information-based standard errors. It is a quadratic approximation, not a global-optimality certificate. PSD fits can restrict the step to the active covariance face.}
\usage{
frontier_newton_accuracy(fit, budget = 0.01, psd = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{budget}{Positive allowed Newton distance in information-based standard-error units.}
\item{psd}{Whether to constrain model covariance blocks to be positive semidefinite.}
}
\value{A diagnostic list with status, distance, predicted gain and whether the requested budget passed.}
