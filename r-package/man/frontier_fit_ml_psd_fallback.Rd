\name{frontier_fit_ml_psd_fallback}
\alias{frontier_fit_ml_psd_fallback}
\title{Ordinary ML with Explicit PSD Recovery}
\description{
Fits ordinary complete-data normal-theory ML first, accepting it only when
both the common fit verdict and covariance admissibility pass. Otherwise
tries PSD-constrained ML once. The selection policy is implemented in C++.
}
\usage{
frontier_fit_ml_psd_fallback(
  model, data, ordinary_optimizer = "nlopt-lbfgs",
  psd_optimizer = "nlopt-slsqp", ordinary_control = NULL,
  psd_control = NULL, start_eigen_floor = 1e-6,
  feasibility_tol = 1e-6, missing = c("listwise", "error"),
  preconditioning = c("diagonal", "none")
)
}
\arguments{
  \item{model}{A model specification or model syntax string.}
  \item{data}{Sample statistics or a raw data frame.}
  \item{ordinary_optimizer}{Ordinary ML optimizer; defaults to NLopt L-BFGS.}
  \item{psd_optimizer}{PSD optimizer; defaults to NLopt SLSQP.}
  \item{ordinary_control, psd_control}{Separate optimizer control lists.
  Select the initial start policy with \code{ordinary_control$start};
  \code{psd_control$start} is rejected because the fallback determines its start.}
  \item{start_eigen_floor}{Positive floor used to initialize PSD factors,
  not a lower bound on the fitted covariance eigenvalues.}
  \item{feasibility_tol}{PSD covariance-link feasibility tolerance.}
  \item{missing}{Raw-data missing-value policy. This is not a FIML route.}
  \item{preconditioning}{PSD diagonal information scaling, or none.}
}
\details{
A warm start transfers ordinary parameter estimates only, not optimizer
memory. The ordinary return must have finite parameters and objective, a
positive-definite implied covariance, and satisfied model equalities to be
used as a start. It need not pass accuracy or primitive-covariance
admissibility. The PSD initializer projects covariance-factor starts;
link equalities may initially be violated and are enforced by the solver.
After an ordinary error or unusable return, PSD uses the original start.

Both stages must pass their common numerical verdict and covariance
admissibility before their result can be selected. This does not certify
global optimality or add boundary-aware inference. Ordinary ML defaults
are unchanged. Parsing, data preparation, and initial start-construction
errors occur before the fitting policy and raise R errors as usual.
}
\value{
A list with \code{fit} (the accepted ordinary or PSD fit, otherwise NULL),
\code{converged}, \code{fallback_used}, \code{fallback_reason}, and
\code{warm_start_used}. The reason is \code{"none"},
\code{"ordinary-error"}, \code{"ordinary-rejected"}, or
\code{"ordinary-inadmissible"}; a failed accuracy verdict takes precedence
over inadmissibility when both fail.

\code{ordinary} and \code{psd} retain each attempt as a list with
\code{fit} and \code{error}, one of which is NULL. An error records
\code{kind}, \code{detail}, \code{iterations}, and \code{f_value} from the
core error value. The entire \code{psd} entry is NULL when skipped.
Rejected returned estimates remain accessible even when \code{fit} is NULL.
}
\examples{
s <- list(S = list(matrix(c(1, .7, .7, .7, 1, .3, .7, .3, 1), 3)),
          nobs = 300L)
result <- frontier_fit_ml_psd_fallback("f =~ x1 + x2 + x3", s)
result$fallback_reason
result$warm_start_used
}
