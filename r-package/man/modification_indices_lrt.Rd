\name{modification_indices_lrt}
\alias{modification_indices_lrt}
\title{Post-fit parameter diagnostics: modification_indices_lrt}
\description{Computes explicit release or restriction diagnostics. LR-type variants use the supplied observations for robust spectral inference. Unsupported candidate regimes are reported explicitly.}
\usage{
modification_indices_lrt(fit, data, candidates = c("all", "loadings", "covariances"), robust = TRUE,
    weight = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{candidates}{Candidate parameter rows to release or test: all or the named subset supported by the operation.}
\item{robust}{Whether to compute robust LR-type diagnostics.}
\item{weight}{Weight selector or explicit fitting weight as required by the route.}
}
\value{A table of candidate parameter diagnostics.}
