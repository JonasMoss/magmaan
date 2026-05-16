#include "magmaan/data/h_score.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "magmaan/error.hpp"

namespace magmaan::data {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

PostError make_err(std::string detail) {
  return PostError{PostError::Kind::NumericIssue, std::move(detail)};
}

double ml_phi(double t) noexcept {
  if (t == 0.0) return 0.0;
  return t * std::log(t);
}

double smootherstep(double x) noexcept {
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}

double smootherstep_integral(double x) noexcept {
  return x - x * x * x * x * (2.5 + x * (-3.0 + x));
}

double smooth_cap_h(double t, double a, double b) noexcept {
  if (t <= a) return t;
  const double d = b - a;
  if (t >= b) return a + 0.5 * d;
  const double x = (t - a) / d;
  return a + d * smootherstep_integral(x);
}

double smooth_cap_dh(double t, double a, double b) noexcept {
  if (t < a) return 1.0;
  if (t > b) return 0.0;
  const double x = (t - a) / (b - a);
  return 1.0 - smootherstep(std::clamp(x, 0.0, 1.0));
}

double exp_cap_h(double t, double k, double lambda) noexcept {
  if (t <= k) return t;
  return k + lambda * (1.0 - std::exp(-(t - k) / lambda));
}

double exp_cap_dh(double t, double k, double lambda) noexcept {
  if (t < k) return 1.0;
  return std::exp(-(t - k) / lambda);
}

template <class F>
double phi_from_h(double t, double split, F&& h) noexcept {
  if (t <= split) return ml_phi(t);
  static constexpr double nodes[32] = {
      -0.9972638618494816, -0.9856115115452684, -0.9647622555875064,
      -0.9349060759377397, -0.8963211557660521, -0.8493676137325700,
      -0.7944837959679424, -0.7321821187402897, -0.6630442669302152,
      -0.5877157572407623, -0.5068999089322294, -0.4213512761306353,
      -0.3318686022821277, -0.2392873622521371, -0.1444719615827965,
      -0.0483076656877383,  0.0483076656877383,  0.1444719615827965,
       0.2392873622521371,  0.3318686022821277,  0.4213512761306353,
       0.5068999089322294,  0.5877157572407623,  0.6630442669302152,
       0.7321821187402897,  0.7944837959679424,  0.8493676137325700,
       0.8963211557660521,  0.9349060759377397,  0.9647622555875064,
       0.9856115115452684,  0.9972638618494816};
  static constexpr double weights[32] = {
      0.0070186100094701, 0.0162743947309057, 0.0253920653092621,
      0.0342738629130214, 0.0428358980222267, 0.0509980592623762,
      0.0586840934785355, 0.0658222227763618, 0.0723457941088485,
      0.0781938957870703, 0.0833119242269467, 0.0876520930044038,
      0.0911738786957639, 0.0938443990808046, 0.0956387200792749,
      0.0965400885147278, 0.0965400885147278, 0.0956387200792749,
      0.0938443990808046, 0.0911738786957639, 0.0876520930044038,
      0.0833119242269467, 0.0781938957870703, 0.0723457941088485,
      0.0658222227763618, 0.0586840934785355, 0.0509980592623762,
      0.0428358980222267, 0.0342738629130214, 0.0253920653092621,
      0.0162743947309057, 0.0070186100094701};

  const double mid = 0.5 * (t + split);
  const double half = 0.5 * (t - split);
  double integral = 0.0;
  for (int i = 0; i < 32; ++i) {
    const double u = mid + half * nodes[i];
    integral += weights[i] * h(u) / (u * u);
  }
  return t * (std::log(split) + half * integral);
}

post_expected<void>
validate(double t, const PolychoricHScoreOptions& options) {
  if (!std::isfinite(t) || t < 0.0) {
    return std::unexpected(make_err(
        "eval_polychoric_h_score: t must be finite and nonnegative"));
  }
  switch (options.kind) {
    case PolychoricHScoreKind::ML:
      return {};
    case PolychoricHScoreKind::WmaHardCap:
      if (!((std::isfinite(options.k) && options.k >= 1.0) ||
            options.k == std::numeric_limits<double>::infinity())) {
        return std::unexpected(make_err(
            "eval_polychoric_h_score: WMA hard cap k must be at least 1"));
      }
      return {};
    case PolychoricHScoreKind::SmoothCap:
      if (!(std::isfinite(options.a) && std::isfinite(options.b) &&
            options.a >= 1.0 && options.a < options.b)) {
        return std::unexpected(make_err(
            "eval_polychoric_h_score: smooth cap requires 1 <= a < b"));
      }
      return {};
    case PolychoricHScoreKind::ExpCap:
      if (!(std::isfinite(options.k) && options.k >= 1.0 &&
            std::isfinite(options.lambda) && options.lambda > 0.0)) {
        return std::unexpected(make_err(
            "eval_polychoric_h_score: exp cap requires k >= 1 and lambda > 0"));
      }
      return {};
  }
  return std::unexpected(make_err("eval_polychoric_h_score: unknown h-score kind"));
}

}  // namespace

post_expected<PolychoricHScoreEval>
eval_polychoric_h_score(double t, const PolychoricHScoreOptions& options) {
  auto ok = validate(t, options);
  if (!ok.has_value()) return std::unexpected(ok.error());

  switch (options.kind) {
    case PolychoricHScoreKind::ML:
      return PolychoricHScoreEval{.h = t, .dh = 1.0, .phi = ml_phi(t)};

    case PolychoricHScoreKind::WmaHardCap: {
      if (std::isinf(options.k)) {
        return PolychoricHScoreEval{.h = t, .dh = 1.0, .phi = ml_phi(t)};
      }
      const double k = options.k;
      if (t < k) {
        return PolychoricHScoreEval{.h = t, .dh = 1.0, .phi = ml_phi(t)};
      }
      const double phi = t * (std::log(k) + 1.0) - k;
      return PolychoricHScoreEval{.h = k, .dh = 0.0, .phi = phi};
    }

    case PolychoricHScoreKind::SmoothCap: {
      const double h = smooth_cap_h(t, options.a, options.b);
      const double dh = smooth_cap_dh(t, options.a, options.b);
      const double phi = phi_from_h(t, options.a, [&](double u) {
        return smooth_cap_h(u, options.a, options.b);
      });
      return PolychoricHScoreEval{.h = h, .dh = dh, .phi = phi};
    }

    case PolychoricHScoreKind::ExpCap: {
      const double h = exp_cap_h(t, options.k, options.lambda);
      const double dh = exp_cap_dh(t, options.k, options.lambda);
      const double phi = phi_from_h(t, options.k, [&](double u) {
        return exp_cap_h(u, options.k, options.lambda);
      });
      return PolychoricHScoreEval{.h = h, .dh = dh, .phi = phi};
    }
  }

  return PolychoricHScoreEval{.h = kNaN, .dh = kNaN, .phi = kNaN};
}

}  // namespace magmaan::data
