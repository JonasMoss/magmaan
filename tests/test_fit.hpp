#pragma once

// Test-only thin wrappers around the estimate::fit* composers. They compute
// the conventional ("simple") start values and forward, so tests keep a
// compact call shape:
//
//     magmaan::test::fit(pt, rep, samp)            — normal-theory ML
//     magmaan::test::fit_gmm(pt, rep, samp, W)     — ULS (empty W) / WLS
//     magmaan::test::fit_gls(pt, rep, samp)        — GLS
//
// `fit_bounded` / `fit_gmm` / `fit_gls` auto-derive variance box bounds from
// the partable when the caller passes an empty `Bounds` — mirroring the old
// `fit_bounded<D,O>` shim. Tests that deliberately exercise specific start
// values should call the real estimate::fit* with an explicit x0 instead.

#include <utility>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/optim/problem.hpp"

// MAGMAAN_TEST_SPHERE_ROUTE: route `fit`, `fit_bounded`, `fit_gmm` and
// `fit_gls` through the frontier sphere chart
// (estimate/frontier/sphere.hpp). Every golden compiled with the define then
// checks that the sphere route reproduces the ordinary user-chart estimate.
#ifdef MAGMAAN_TEST_SPHERE_ROUTE
#include <cstdio>

#include "magmaan/estimate/frontier/sphere.hpp"
#endif

namespace magmaan::test {

namespace detail {

// Variance box bounds for an LS fit: an empty `b` auto-derives them from the
// partable (the old `fit_bounded` shim's behavior).
inline fit_expected<estimate::Bounds>
auto_bounds(const spec::LatentStructure& pt, estimate::Bounds b) {
  if (!b.empty()) return b;
  auto d = estimate::bounds_from_partable(pt);
  if (!d.has_value()) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "test::auto_bounds: " + d.error().detail, 0, 0.0});
  }
  return *d;
}

#ifdef MAGMAAN_TEST_SPHERE_ROUTE
// Tally of sphere-route fits: how many ran and how many carried at least one
// gauge unit (the rest ran entirely in the user chart). Printed at exit.
struct SphereRouteTally {
  long fits = 0;
  long with_units = 0;
  long units = 0;
  ~SphereRouteTally() {
    std::fprintf(stderr,
                 "[sphere route] fits: %ld, with gauge units: %ld, units: %ld\n",
                 fits, with_units, units);
  }
};
inline SphereRouteTally sphere_route_tally;

inline fit_expected<estimate::Estimates>
sphere_result(fit_expected<estimate::frontier::SphereFit> r) {
  if (!r.has_value()) return std::unexpected(r.error());
  ++sphere_route_tally.fits;
  const auto n = static_cast<long>(r->report.plan.units.size());
  if (n > 0) ++sphere_route_tally.with_units;
  sphere_route_tally.units += n;
  if (!r->user_chart) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "sphere route: the estimate lies outside the user's chart", 0, 0.0});
  }
  return std::move(r->estimates);
}
#endif

}  // namespace detail

// Normal-theory ML. `backend` selects the optimizer (default NLopt L-BFGS; the
// NLopt SLSQP and PORT (nlminb) cross-check backends are also accepted).
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit(const Pt& pt, const Rep& rep, const Samp& samp,
    estimate::Bounds bounds = {},
    estimate::Backend backend = estimate::Backend::NloptLbfgs,
    optim::OptimOptions opts = {}) {
  auto x0 = estimate::simple_start_values(pt, rep, samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
#ifdef MAGMAAN_TEST_SPHERE_ROUTE
  return detail::sphere_result(estimate::frontier::fit_ml_sphere(
      pt, rep, samp, *x0, std::move(bounds), backend, opts));
#else
  return estimate::fit_ml(pt, rep, samp, *x0, std::move(bounds), backend, opts);
#endif
}

// Normal-theory ML with box bounds; an empty `bounds` auto-derives
// the variance bounds from the partable.
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit_bounded(const Pt& pt, const Rep& rep, const Samp& samp,
            estimate::Bounds bounds, optim::OptimOptions opts = {}) {
  auto b = detail::auto_bounds(pt, std::move(bounds));
  if (!b.has_value()) return std::unexpected(b.error());
  return fit(pt, rep, samp, std::move(*b), estimate::Backend::NloptLbfgs, opts);
}

// Moment-quadratic least squares: an empty `weight` ⇒ ULS, a caller-supplied
// weight ⇒ WLS / DWLS. An empty `bounds` auto-derives the variance bounds.
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit_gmm(const Pt& pt, const Rep& rep, const Samp& samp,
        estimate::gmm::Weight weight = {}, estimate::Bounds bounds = {},
        estimate::Backend backend = estimate::Backend::NloptLbfgs,
        optim::OptimOptions opts = {}) {
  auto x0 = estimate::simple_start_values(pt, rep, samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  auto b = detail::auto_bounds(pt, std::move(bounds));
  if (!b.has_value()) return std::unexpected(b.error());
#ifdef MAGMAAN_TEST_SPHERE_ROUTE
  return detail::sphere_result(estimate::frontier::fit_gmm_sphere(
      pt, rep, samp, *x0, std::move(weight), std::move(*b), backend, opts));
#else
  return estimate::fit_gmm(pt, rep, samp, *x0, std::move(weight),
                           std::move(*b), backend, opts);
#endif
}

// Generalized least squares (normal-theory weight built from S).
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit_gls(const Pt& pt, const Rep& rep, const Samp& samp,
        estimate::Bounds bounds = {},
        estimate::Backend backend = estimate::Backend::NloptLbfgs,
        optim::OptimOptions opts = {}) {
  auto x0 = estimate::simple_start_values(pt, rep, samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  auto b = detail::auto_bounds(pt, std::move(bounds));
  if (!b.has_value()) return std::unexpected(b.error());
#ifdef MAGMAAN_TEST_SPHERE_ROUTE
  return detail::sphere_result(estimate::frontier::fit_gls_sphere(
      pt, rep, samp, *x0, std::move(*b), backend, opts));
#else
  return estimate::fit_gls(pt, rep, samp, *x0, std::move(*b), backend, opts);
#endif
}

// Full-information ML over raw continuous data.
template <class Pt, class Rep, class Raw>
fit_expected<estimate::Estimates>
fit_fiml(const Pt& pt, const Rep& rep, const Raw& raw,
         optim::OptimOptions opts = {},
         estimate::Backend backend = estimate::Backend::NloptLbfgs) {
  auto samp = estimate::fiml::fiml_start_sample_stats(raw);
  if (!samp.has_value()) return std::unexpected(samp.error());
  auto x0 = estimate::simple_start_values(pt, rep, *samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  return estimate::fit_fiml(pt, rep, raw, *x0, estimate::fiml::FIML{},
                            backend, opts);
}

// Ordinal DWLS / WLS. `parameterization` selects Delta (default) or Theta.
template <class Pt, class Rep, class Stats>
fit_expected<estimate::Estimates>
fit_ordinal_bounded(const Pt& pt, const Rep& rep, const Stats& stats,
                    estimate::Bounds bounds, estimate::OrdinalWeightKind weights,
                    estimate::Backend backend = estimate::Backend::NloptLbfgs,
                    optim::OptimOptions opts = {},
                    estimate::OrdinalParameterization parameterization =
                        estimate::OrdinalParameterization::Delta) {
  auto x0 = estimate::ordinal_start_values(pt, rep, stats, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  return estimate::fit_ordinal_bounded(pt, rep, stats, std::move(bounds),
                                       weights, *x0, backend, opts,
                                       parameterization);
}

template <class Pt, class Rep, class Stats>
fit_expected<estimate::Estimates>
fit_mixed_ordinal_bounded(const Pt& pt, const Rep& rep, const Stats& stats,
                          estimate::Bounds bounds,
                          estimate::OrdinalWeightKind weights,
                          estimate::Backend backend = estimate::Backend::NloptLbfgs,
                          optim::OptimOptions opts = {},
                          estimate::OrdinalParameterization parameterization =
                              estimate::OrdinalParameterization::Delta) {
  auto x0 = estimate::mixed_ordinal_start_values(pt, rep, stats, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  return estimate::fit_mixed_ordinal_bounded(pt, rep, stats, std::move(bounds),
                                             weights, *x0, backend, opts,
                                             parameterization);
}

}  // namespace magmaan::test
