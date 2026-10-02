\name{score_tests_lrt}
\alias{score_tests_lrt}
\title{Post-fit parameter diagnostics: score_tests_lrt}
\description{Computes explicit release or restriction diagnostics. LR-type variants use the supplied observations for robust spectral inference. Unsupported candidate regimes are reported explicitly.}
\usage{
score_tests_lrt(fit, data, candidates = c("loadings", "intercepts", "all"), method = c("restriction_map",
    "lavaan_sb2001", "lavaan_sb2010"), gamma = c("empirical", "NT"))
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{candidates}{Candidate parameter rows to release or test: all or the named subset supported by the operation.}
\item{method}{Method selector; see the available choices in the usage signature and details.}
\item{gamma}{Moment covariance convention used for the nested score diagnostic.}
}
\value{A table of candidate parameter diagnostics.}
