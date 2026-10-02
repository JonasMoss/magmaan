\name{score_tests}
\alias{score_tests}
\title{Post-fit parameter diagnostics: score_tests}
\description{Computes explicit release or restriction diagnostics. LR-type variants use the supplied observations for robust spectral inference. Unsupported candidate regimes are reported explicitly.}
\usage{
score_tests(fit, data = NULL, ...)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
}
\value{A table of candidate parameter diagnostics.}
