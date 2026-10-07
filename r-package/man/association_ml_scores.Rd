\name{association_ml_modification_indices}
\alias{association_ml_modification_indices}
\alias{association_ml_score_tests}
\title{Lab association-ML modification indices and linear releases}
\usage{
association_ml_modification_indices(fit, ordinal_stats, absent = TRUE)
association_ml_score_tests(fit, ordinal_stats)
}
\arguments{
\item{fit}{An all-ordinal association-ML lab fit with complete raw rows.}
\item{ordinal_stats}{The same Stage-1 statistics and integer rows used to fit.}
\item{absent}{Enumerate absent loadings, covariances and structural regressions among
equation variables in addition to fixed rows.}
}
\description{One-dimensional score tests using observed sensitivity for nuisance
projection and exact empirical Stage-1 influence for the score variance.}
\details{The robust statistic is N times the squared efficient score divided by
its empirical meat, referenced to chi-square with one degree of freedom. EPC
uses the observed Schur sensitivity. The unscaled \code{mi} uses that same
sensitivity. Equality rows index the original affine constraint system; EPC
is the coefficient along its normalized release direction, with
\code{epc_full} giving the full parameter displacement. Fixed-row EPC is in
the candidate parameter's units. No joint multi-direction release is exposed.
Threshold, mean and response-scale releases, singular information, covariance
boundaries, penalties, missing data and mixed indicators remain unavailable.
Ordinary association inference remains guarded pending policy calibration.}
\value{A list with \code{available}, \code{metric = "observed"} and \code{rows}.
Each row retains availability or a typed reason. Available rows contain
\code{statistic}, \code{df}, \code{p_value}, \code{epc}, \code{epc_full},
\code{mi} and diagnostic score, sensitivity, meat and coordinate matrices.}
