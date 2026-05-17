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
#include "magmaan/optim/lbfgs_optimizer.hpp"

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

}  // namespace detail

// Normal-theory ML. `backend` selects the optimizer (default L-BFGS; the NLopt
// SLSQP and trust-region cross-check backends are also accepted).
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit(const Pt& pt, const Rep& rep, const Samp& samp,
    estimate::Bounds bounds = {},
    estimate::Backend backend = estimate::Backend::Lbfgs,
    optim::LbfgsOptions opts = {}) {
  auto x0 = estimate::simple_start_values(pt, rep, samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  return estimate::fit_ml(pt, rep, samp, *x0, std::move(bounds), backend, opts);
}

// Normal-theory ML with box bounds (LBFGS-B); an empty `bounds` auto-derives
// the variance bounds from the partable.
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit_bounded(const Pt& pt, const Rep& rep, const Samp& samp,
            estimate::Bounds bounds, optim::LbfgsOptions opts = {}) {
  auto b = detail::auto_bounds(pt, std::move(bounds));
  if (!b.has_value()) return std::unexpected(b.error());
  return fit(pt, rep, samp, std::move(*b), estimate::Backend::Lbfgs, opts);
}

// Moment-quadratic least squares: an empty `weight` ⇒ ULS, a caller-supplied
// weight ⇒ WLS / DWLS. An empty `bounds` auto-derives the variance bounds.
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit_gmm(const Pt& pt, const Rep& rep, const Samp& samp,
        estimate::gmm::Weight weight = {}, estimate::Bounds bounds = {},
        estimate::Backend backend = estimate::Backend::Lbfgs,
        optim::LbfgsOptions opts = {}) {
  auto x0 = estimate::simple_start_values(pt, rep, samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  auto b = detail::auto_bounds(pt, std::move(bounds));
  if (!b.has_value()) return std::unexpected(b.error());
  return estimate::fit_gmm(pt, rep, samp, *x0, std::move(weight),
                           std::move(*b), backend, opts);
}

// Generalized least squares (normal-theory weight built from S).
template <class Pt, class Rep, class Samp>
fit_expected<estimate::Estimates>
fit_gls(const Pt& pt, const Rep& rep, const Samp& samp,
        estimate::Bounds bounds = {},
        estimate::Backend backend = estimate::Backend::Lbfgs,
        optim::LbfgsOptions opts = {}) {
  auto x0 = estimate::simple_start_values(pt, rep, samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  auto b = detail::auto_bounds(pt, std::move(bounds));
  if (!b.has_value()) return std::unexpected(b.error());
  return estimate::fit_gls(pt, rep, samp, *x0, std::move(*b), backend, opts);
}

// Full-information ML over raw continuous data.
template <class Pt, class Rep, class Raw>
fit_expected<estimate::Estimates>
fit_fiml(const Pt& pt, const Rep& rep, const Raw& raw,
         optim::LbfgsOptions opts = {}) {
  auto samp = estimate::fiml::fiml_start_sample_stats(raw);
  if (!samp.has_value()) return std::unexpected(samp.error());
  auto x0 = estimate::simple_start_values(pt, rep, *samp, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  return estimate::fit_fiml(pt, rep, raw, *x0, {}, opts);
}

// Ordinal DWLS / WLS. `parameterization` selects Delta (default) or Theta.
template <class Pt, class Rep, class Stats>
fit_expected<estimate::Estimates>
fit_ordinal_bounded(const Pt& pt, const Rep& rep, const Stats& stats,
                    estimate::Bounds bounds, estimate::OrdinalWeightKind weights,
                    estimate::Backend backend = estimate::Backend::Lbfgs,
                    optim::LbfgsOptions opts = {},
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
                          estimate::Backend backend = estimate::Backend::Lbfgs,
                          optim::LbfgsOptions opts = {}) {
  auto x0 = estimate::mixed_ordinal_start_values(pt, rep, stats, {});
  if (!x0.has_value()) return std::unexpected(x0.error());
  return estimate::fit_mixed_ordinal_bounded(pt, rep, stats, std::move(bounds),
                                             weights, *x0, backend, opts);
}

}  // namespace magmaan::test
