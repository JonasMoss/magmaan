\name{lab_inference_primitives}
\alias{infer_build_u_factor}
\alias{infer_build_u_factor_parts}
\alias{infer_build_u_factor_pairwise}
\alias{infer_build_u_factor_fit}
\alias{infer_continuous_ls_robust}
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
infer_continuous_ls_robust(fit, raw_data, weight = NULL,
  bread = "observed", gamma = "empirical", fixed_weight = FALSE)
measures_reliability_cov(S, gamma = NULL, n = 0L, raw_data = NULL)
measures_reliability_omega_multidim(S, block, target = "total",
  weights = NULL, gamma = NULL, n = 0L, raw_data = NULL)
}
\arguments{
\item{fit}{Fitted model.}
\item{bread}{Required geometry for U factors: observed for covariance or
nested restrictions at a pseudo-true target; expected is admissible for
complete-data global goodness-of-fit under the correct-model null.}
\item{moments}{Existing structured or saturated moment evaluation.
The joint bread/meat derivation remains pending; this argument is unchanged.}
\item{partable,sample_stats,theta}{Model, observed moments and fitted parameters.}
\item{X,mask}{Raw observations and optional observation mask.
Pass \code{mask} by name in the pairwise builder.}
\item{raw_data}{Raw fitting observations; a numeric matrix for reliability.}
\item{weight}{Fitting weight, optionally retained on the fit.}
\item{gamma}{Empirical or normal moment covariance for LS; an explicit
asymptotic covariance of vech(S) for reliability. Reliability NULL derives
empirical Gamma from raw_data and errors if those observations are absent.}
\item{fixed_weight}{Explicitly request the caller-fixed-weight LS law.
For data-estimated nt/adf/dwls/dls recipes, use
\code{infer_casewise_influence_ij_fit()} for parameter covariance.
This primitive errors unless fixed_weight = TRUE requests its comparator.}
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
