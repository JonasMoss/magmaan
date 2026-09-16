\name{prepare_inference_data}
\alias{prepare_inference_data}
\alias{prepare_hypothesis}
\alias{inference_quadratic}
\alias{inference_covariance}
\alias{inference_reuse}
\title{Shared continuous NTML inference geometry}
\description{Own sample contributions and fitted NTML geometry across score,
likelihood-ratio, goodness-of-fit and covariance calculations. These are
methods-development interfaces for interior, random-X complete-data ML fits
with affine equality constraints.}
\usage{
prepare_inference_data(fit, data = NULL,
  storage = c("auto", "casewise", "tiled"))
prepare_hypothesis(null, alternative)
inference_quadratic(object, test = c("score", "lr"))
inference_covariance(context, robust = FALSE)
inference_reuse(context)
}
\arguments{
\item{fit}{A continuous ML fit defining the observed-variable/group layout.}
\item{data}{Fitting observations or a prepared dataset. Defaults to retained fitting data.}
\item{storage}{Cache centered casewise moment rows, use tiled projection, or select
casewise storage when the contribution matrix occupies at most 64 MiB.}
\item{null, alternative}{Inference snapshots created with
\code{prepare_inference(fit, shared_data)} using the same inference dataset.
Exact parameter nesting requires matching ambient parameter slots and numeric
models. The null adds affine restrictions.}
\item{object}{An inference snapshot (global test), or prepared hypothesis (nested test).}
\item{test}{Expected-information score or ML likelihood-ratio quadratic.}
\item{context}{A prepared inference snapshot.}
\item{robust}{Use the empirical sandwich covariance rather than inverse expected information.}
}
\details{
The core reuses the existing NTML U-factor construction and casewise covariance
reducers. It retains structured expected-information geometry and each test's
reduction separately. Global LR uses the centered empirical GOF spectrum;
global score uses likelihood contributions at the fitted null point. Restricted
means require additional linear and constant terms, so these spectra are not
silently equated. Nested LR uses the exact H1-anchored Satorra--2000 reduction;
nested score evaluates at H0.

No empirical Gamma or full U matrix is required. Casewise storage reuses one
contribution matrix across fits. Tiled storage reduces global score and GOF
together when the sample size is at least df, retaining reduced matrices;
otherwise the smaller row-space spectrum is used. Tiled storage trades repeated
raw-data passes for bounded contribution memory when different hypotheses need
different projections. Explicit Browne-unbiased GOF requests use the existing
casewise correction and may materialize contributions.

\code{score_spectrum} and \code{calibrate_quadratic} accept these quadratics.
Spectrum calculations are lazy and retained; SB alone uses the trace.
\code{inference_covariance} returns a snapshot-tagged matrix usable with
\code{wald_test}. Existing GOF, expected-information, expected sandwich-SE and
exact empirical streaming nested-LR wrappers also reuse compatible prepared
snapshots. Other bread/Gamma/nesting conventions retain their existing paths.

Shared NTML geometry does not yet cover FIML, ML2S, ordinal or fixed-X fits.
Their existing score interfaces remain available. Native handles are local to
the current R process: prepare again after serialization. Locked R environments
preserve ownership; edited extracted fit lists invalidate the native caches.
}
\value{Prepared data/hypothesis and quadratic calls return locked environments
owning native objects. Covariance returns a matrix; reuse returns construction
counts, including data contributions, fit geometry, U, information and global
spectra. Counts are diagnostics, not a timing guarantee.}
\examples{
\dontrun{
d <- prepare_inference_data(fit1, data)
g0 <- prepare_inference(fit0, d)
g1 <- prepare_inference(fit1, d)
h <- prepare_hypothesis(g0, g1)
qs <- inference_quadratic(h, "score")
ql <- inference_quadratic(h, "lr")
calibrate_quadratic(qs, c("sb", "peba4"))
calibrate_quadratic(ql, c("sb", "peba4"))
V <- inference_covariance(g0, robust = TRUE)
inference_reuse(g0)
}}
