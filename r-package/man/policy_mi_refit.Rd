\name{policy_mi_refit}
\alias{policy_mi_refit}
\title{Refit one modification-index candidate from the embedded null}
\usage{policy_mi_refit(fit, row, candidate_row)}
\arguments{
\item{fit}{The fitted (restricted) magmaanlab model.}
\item{row}{One row of \code{\link{policy_modification_indices}} output.}
\item{candidate_row}{The matching element of that output's \code{candidate_row} attribute.}
}
\description{Frees one candidate (or releases one equality) and refits from the fitted estimate embedded in the augmented model, replaying the fit's options. The template is an evaluation point for the native embedding map; it is never optimized.}
\value{The fitted augmented model, for \code{\link{policy_nested}}.}
