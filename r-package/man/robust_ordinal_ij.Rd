\name{robust_ordinal_ij}
\alias{robust_ordinal_ij}
\alias{infer_ordinal_robust_ij}
\title{All-ordinal estimated-weight IJ covariance}
\usage{
magmaan_core$robust_ordinal_ij(fit, ordinal_stats, weight = "", first_stage = "opg")
}
\arguments{
\item{fit}{An all-ordinal fitted lab model.}
\item{ordinal_stats}{The ordinal statistics used for fitting.}
\item{weight}{Computational weight; empty reuses the fit's weight.}
\item{first_stage}{\code{"opg"} (default) or the complete-data empirical
score-Jacobian comparator \code{"exact"}.}
}
\description{Composes observed-bread covariance with the estimated-weight
influence. The exact comparator changes sampling rows and their contribution
to weight movement; fitting continues to use OPG NACOV weights.}
\value{A list containing vcov, se, df and chisq_standard. The exact comparator
also returns per-group sampling_moment_influence matrices and sampling_gamma
(the cross-product of the rows divided by group sample size).}
