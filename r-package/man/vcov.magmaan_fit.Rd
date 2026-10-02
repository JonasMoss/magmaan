\name{vcov.magmaan_fit}
\alias{vcov.magmaan_fit}
\title{Parameter covariance under an explicit statistical contract}
\usage{
\method{vcov}{magmaan_fit}(object, regime = NULL, data = NULL, ...)
}
\arguments{
\item{object}{A fitted \code{magmaanlab} model.}
\item{regime}{Covariance formula; see Details. \code{NULL} preserves the
estimator-specific historical default.}
\item{data}{Raw observations for empirical sandwiches or delta-method
covariances. Defaults to the observations retained by the fit. FIML always
uses its retained observations and rejects a replacement.}
\item{...}{Reserved for generic compatibility.}
}
\description{Selects existing C++ covariance primitives without changing the
fitted model or the ordinary-user inference policy.}
\details{
\describe{
\item{\code{information_expected}, \code{information_observed}}{Inverse
expected information or observed Hessian information, respectively. Available
for complete-data ML and FIML; neither includes empirical sandwich meat.}
\item{\code{sandwich_expected}, \code{sandwich_observed}}{Empirical-meat
sandwiches with expected/Gauss--Newton or observed-Hessian bread. Available for
continuous iterative fits and categorical fits where the corresponding core
primitive supports the estimator, and for FIML. Categorical covariances use
the retained categorical statistics. These are fixed-weight covariances.}
\item{\code{sandwich_ij}}{The estimated-weight (infinitesimal-jackknife)
sandwich for all-ordinal DWLS, WLS and ULS fits: observed bread, Stage-1
threshold and polychoric influence, and the influence of the estimated weight.
It is the ordinary DWLS policy covariance.}
\item{\code{delta_nt}, \code{delta_empirical}}{Non-iterative CFA delta-method
covariances using normal-theory or empirical moment covariance.}
\item{\code{stored}}{The covariance already computed by a SAM fit.}
}
Unsupported combinations raise an error. Defaults are \code{sandwich_expected}
for continuous/categorical iterative fits, \code{information_observed} for
FIML, \code{delta_nt} for non-iterative CFA, and \code{stored} for SAM.
Shared barrier compositions and pairwise MCAR fits reject this interface
pending their own sampling/inference contracts.

Legacy \code{model} selects the historical default. Legacy \code{robust}
selects \code{sandwich_observed}, except for non-iterative CFA
(\code{delta_empirical}) and SAM (\code{stored}). These aliases intentionally
retain their old estimator-dependent meanings; prefer the explicit names.
}
\value{A covariance matrix in the fit's free-parameter order.}
