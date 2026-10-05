\name{lab_inference_primitives}
\alias{infer_build_u_factor}
\alias{infer_build_u_factor_parts}
\alias{infer_build_u_factor_pairwise}
\alias{infer_build_u_factor_fit}
\alias{infer_continuous_ls_robust}
\alias{infer_continuous_ls_profile_lrt}
\alias{infer_ml_profile_lrt}
\alias{measures_reliability_cov}
\alias{measures_reliability_omega_multidim}
\title{Explicit geometry and sampling laws for lab primitives}
\description{Methods-development primitives exposed through \code{magmaan_core}.
Their ingredients must describe the component's null and sampling law.}
\usage{
infer_build_u_factor(fit, bread, moments = "structured")
infer_build_u_factor_parts(partable, sample_stats, theta, bread,
  moments = "structured")
infer_build_u_factor_pairwise(fit, X, bread, mask = NULL)
infer_build_u_factor_fit(fit, bread, moments = "structured")
infer_continuous_ls_robust(fit, raw_data = NULL, weight = NULL,
  bread = "observed", gamma = NULL, fixed_weight = FALSE)
infer_continuous_ls_profile_lrt(fit_H1, fit_H0, X_per_group = NULL,
  weight = NULL, eig_tol = 1e-10, gamma = NULL)
infer_ml_profile_lrt(fit_H1, fit_H0, X_per_group = NULL,
  eig_tol = 1e-10, gamma = NULL)
measures_reliability_cov(S, gamma = NULL, n = 0L, raw_data = NULL)
measures_reliability_omega_multidim(S, block, target = "total",
  weights = NULL, gamma = NULL, n = 0L, raw_data = NULL)
}
\arguments{
\item{fit}{Fitted model.}
\item{fit_H1,fit_H0}{Less and more restricted fits for the profile LRT.}
\item{X_per_group}{Raw observations in fitted group order, used when gamma
is NULL. Omit when supplying Gamma.}
\item{eig_tol}{Relative tolerance for profile-law eigenvalues.}
\item{bread}{Required geometry for U factors: expected for complete-data global
goodness-of-fit under the correct-model null; observed requests the explicit
Hessian convention. U factors alone do not certify pseudo-true covariance or
nested sampling laws.}
\item{moments}{For U factors, \code{"structured"} or \code{"unstructured"} selects the
fitted/sample normal-theory weight, not centering. These correct-null GOF
conventions retain their defaults and reject \code{"likelihood"}. Lab ML
SE/score meats instead default to \code{"auto"}: exact uncentered likelihood
projections for empirical covariance, structured for model-implied comparators.
Exact likelihood projections include fitted-mean shifts and group score means;
centered Zc and caller Gamma use the same affine transformation.}
\item{partable,sample_stats,theta}{Model, observed moments and fitted parameters.}
\item{X,mask}{Raw observations and optional observation mask.
Pass \code{mask} by name in the pairwise builder.}
\item{raw_data}{Raw fitting observations; a numeric matrix for reliability.}
\item{weight}{Fitting weight, optionally retained on the fit.}
\item{gamma}{For LS robust covariance, NULL retains empirical covariance from
raw_data; \code{"empirical"} and explicit \code{"normal"} sources remain supported.
A supplied symmetric PSD matrix (one group) or list of per-group matrices
replaces the meat and requires \code{fixed_weight = TRUE}. ML/LS profile LRTs
also accept these matrices; NULL builds empirical Gamma from X_per_group.
Blocks use means first when fitted, then lower-triangle covariances by columns
including diagonals, in fitted observed-variable order. Supply unscaled
per-group NACOV (N times sampling covariance); group weights are applied by
C++. Dimensions, finiteness, symmetry and PSD are validated. For reliability,
Gamma is an explicit asymptotic covariance of vech(S); NULL derives empirical
Gamma from raw_data and errors if those observations are absent.}
\item{fixed_weight}{Explicitly request the caller-fixed-weight LS law.
For data-estimated nt/adf/dwls/dls recipes, use
\code{infer_casewise_influence_ij_fit()} for parameter covariance.
This primitive errors unless fixed_weight = TRUE requests its comparator.
Grouped sample-centered moment meats describe fixed group allocation;
random composition with nonzero group estimating-equation means requires
additional allocation influence.}
\item{S}{Observed covariance matrix in the same metric as Gamma.}
\item{n}{Sample size; inferred from reliability raw_data if zero.}
\item{block}{Integer item-to-factor allocation.}
\item{target}{Total or hierarchical omega.}
\item{weights}{Optional item weights.}
}
\details{Scalar \code{frontier_profile_lrt_*} and interval primitives default
to the misspecification-scaled reference (\code{MisspecScaled}) when
\code{reference = NULL, robust = FALSE}. Supply raw data where required.
The explicit \code{reference = "ordinary"} and
\code{reference = "robust_scaled"} comparators remain available.
Explicit \code{robust = TRUE} with NULL reference retains RobustScaled.
Named Satorra, lavaan bundle and FMG/semTests replication APIs retain their
documented compatibility conventions. SAM and case-change SE/type options
standardize diagnostics and do not certify a generic inferential covariance.}
