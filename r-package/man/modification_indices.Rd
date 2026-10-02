\name{modification_indices}
\alias{modification_indices}
\title{Post-fit parameter diagnostics: modification_indices}
\description{Computes explicit release or restriction diagnostics. LR-type variants use the supplied observations for robust spectral inference. Unsupported candidate regimes are reported explicitly.}
\usage{
modification_indices(fit, data = NULL, ..., candidates = "all")
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
\item{candidates}{Candidate parameter rows to release or test: all or the named subset supported by the operation.}
}
\value{A table of candidate parameter diagnostics.}
