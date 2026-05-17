#include "magmaan/estimate/bounds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "magmaan/error.hpp"
#include "magmaan/parse/op.hpp"

namespace magmaan::estimate {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

using spec::VarRole;

// One bound-able parameter family in the lavaan `optim.bounds` algorithm:
// whether a lower/upper bound is written, and the range-enlargement factors.
struct TypeRule {
  bool   lower = false;
  bool   upper = false;
  double lower_factor = 1.0;
  double upper_factor = 1.0;
};

// A lavaan `optim.bounds` preset (a "profile").
struct Preset {
  TypeRule ov_var;        // (residual) observed variances
  TypeRule lv_var;        // latent variances
  TypeRule loadings;      // factor loadings
  TypeRule covariances;   // residual / latent covariances
  double   rel = 0.0;             // min.reliability.marker
  double   min_var_lv_endo = 0.0;
  double   min_var_lv_exo = 0.0;
};

// lavaan `bounds = "standard"`.
Preset standard_preset() {
  Preset p;
  p.ov_var      = {true, true, 1.0,   1.0};
  p.lv_var      = {true, true, 1.0,   1.0};
  p.loadings    = {true, true, 1.0,   1.0};
  p.covariances = {true, true, 0.999, 0.999};
  p.rel = 0.1;
  p.min_var_lv_endo = 0.005;
  return p;
}

// lavaan `bounds = "wide"` (= "default").
Preset wide_preset() {
  Preset p;
  p.ov_var      = {true, true, 1.05, 1.20};
  p.lv_var      = {true, true, 1.0,  1.30};
  p.loadings    = {true, true, 1.10, 1.10};
  p.covariances = {true, true, 1.0,  1.0};
  p.rel = 0.1;
  p.min_var_lv_endo = 0.005;
  return p;
}

// Loadings-only slice of the standard profile.
Preset loading_preset() {
  Preset p;
  p.loadings = {true, true, 1.0, 1.0};
  p.rel = 0.1;
  p.min_var_lv_endo = 0.005;
  return p;
}

// Range-enlargement for variances and loadings: lavaan widens with the
// absolute range delta, and additionally bumps an ov-variance upper by 0.5%
// when its factor is exactly 1.0.
void enlarge_abs(double& lo, double& hi, double range, const TypeRule& r,
                 bool is_ov_var) {
  if (!std::isfinite(range)) return;
  if (r.lower && std::isfinite(r.lower_factor) && r.lower_factor != 1.0) {
    lo -= std::abs(range * r.lower_factor - range);
  }
  if (r.upper) {
    if (std::isfinite(r.upper_factor) && r.upper_factor != 1.0) {
      hi += std::abs(range * r.upper_factor - range);
    } else if (is_ov_var && r.upper_factor == 1.0) {
      hi += std::abs(range * 1.005 - range);
    }
  }
}

// Range-enlargement for covariances: lavaan uses the *signed* range delta, so
// a factor below 1.0 (e.g. the standard profile's 0.999) shrinks the box
// slightly to keep the implied matrix strictly positive-definite.
void enlarge_cov(double& lo, double& hi, double range, const TypeRule& r) {
  if (!std::isfinite(range)) return;
  if (r.lower && std::isfinite(r.lower_factor) && r.lower_factor != 1.0) {
    lo -= range * r.lower_factor - range;
  }
  if (r.upper && std::isfinite(r.upper_factor) && r.upper_factor != 1.0) {
    hi += range * r.upper_factor - range;
  }
}

// The shared lavaan `lav_partable_add_bounds` engine, parameterised by a
// preset. Mirrors lavaan/R/lav_partable_bounds.R.
post_expected<Bounds>
compute_auto_bounds(const spec::LatentStructure& pt,
                    const data::SampleStats& samp, const Preset& preset) {
  const std::int32_t n_free = pt.n_free();
  Bounds out;
  out.lower = Eigen::VectorXd::Constant(n_free, -kInf);
  out.upper = Eigen::VectorXd::Constant(n_free,  kInf);
  if (n_free == 0) return out;

  const std::int32_t n_groups = pt.n_groups();
  if (static_cast<std::int32_t>(samp.S.size()) < n_groups) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "auto bounds: sample covariance has fewer blocks than the model"));
  }
  const auto n_vars = static_cast<std::size_t>(pt.n_vars);

  auto role = [&](std::int32_t v) -> VarRole {
    return pt.var_role[static_cast<std::size_t>(v)];
  };
  auto is_latent = [&](std::int32_t v) -> bool {
    return v >= 0 && static_cast<std::size_t>(v) < n_vars &&
           role(v) == VarRole::Latent;
  };

  // Observed-variable sample variance in a block (NaN if not resolvable).
  auto ov_var = [&](std::int32_t v, std::int32_t blk) -> double {
    if (v < 0 || static_cast<std::size_t>(v) >= n_vars) return kNaN;
    if (blk < 0 || blk >= static_cast<std::int32_t>(samp.S.size())) return kNaN;
    const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
    const Eigen::MatrixXd& S = samp.S[static_cast<std::size_t>(blk)];
    if (pos < 0 || pos >= S.rows() || pos >= S.cols()) return kNaN;
    return S(pos, pos);
  };

  // Per-block largest observed variance — the fallback latent-variance upper
  // for a factor with no marker indicator.
  std::vector<double> max_ov_var(static_cast<std::size_t>(n_groups), kNaN);
  for (std::int32_t g = 0; g < n_groups; ++g) {
    const Eigen::MatrixXd& S = samp.S[static_cast<std::size_t>(g)];
    if (S.rows() > 0) {
      max_ov_var[static_cast<std::size_t>(g)] = S.diagonal().maxCoeff();
    }
  }

  // Structural facts (block-independent): each factor's marker indicator, the
  // fixed value of a std.lv-identified latent variance, and which latents are
  // endogenous (left-hand side of a regression).
  std::vector<std::int32_t> marker_ov(n_vars, -1);
  std::vector<double>       std_lv_value(n_vars, kNaN);
  std::vector<char>         is_endo(n_vars, 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const parse::Op op = pt.op[i];
    const std::int32_t lhs = pt.lhs_var[i];
    const std::int32_t rhs = pt.rhs_var[i];
    if (op == parse::Op::Measurement) {
      if (is_latent(lhs) && rhs >= 0 && pt.free[i] == 0 &&
          std::isfinite(pt.fixed_value[i]) && pt.fixed_value[i] == 1.0 &&
          marker_ov[static_cast<std::size_t>(lhs)] < 0) {
        marker_ov[static_cast<std::size_t>(lhs)] = rhs;
      }
    } else if (op == parse::Op::Covariance && lhs >= 0 && lhs == rhs) {
      if (is_latent(lhs) && pt.free[i] == 0 &&
          std::isfinite(pt.fixed_value[i])) {
        std_lv_value[static_cast<std::size_t>(lhs)] = pt.fixed_value[i];
      }
    } else if (op == parse::Op::Regression && is_latent(lhs)) {
      is_endo[static_cast<std::size_t>(lhs)] = 1;
    }
  }
  std::vector<char> is_marker(n_vars, 0);
  for (std::size_t c = 0; c < n_vars; ++c) {
    if (marker_ov[c] >= 0 && static_cast<std::size_t>(marker_ov[c]) < n_vars) {
      is_marker[static_cast<std::size_t>(marker_ov[c])] = 1;
    }
  }

  // Latent-variance lower/upper for a factor in a block. `lb` is the
  // marker-based lower (used for loading bounds); `lb2` is the lower used for
  // the variance bound itself (endogenous factors get `min.var.lv.endo`).
  struct LvVar { double lb, lb2, ub; };
  auto lv_bounds = [&](std::int32_t c, std::int32_t blk) -> LvVar {
    LvVar r{kNaN, kNaN, kNaN};
    if (c >= 0 && static_cast<std::size_t>(c) < n_vars &&
        std::isfinite(std_lv_value[static_cast<std::size_t>(c)])) {
      r.lb = r.lb2 = r.ub = std_lv_value[static_cast<std::size_t>(c)];
      return r;
    }
    const std::int32_t mk =
        (c >= 0 && static_cast<std::size_t>(c) < n_vars)
            ? marker_ov[static_cast<std::size_t>(c)]
            : -1;
    const double mv = (mk >= 0) ? ov_var(mk, blk) : kNaN;
    if (std::isfinite(mv)) {
      r.ub = mv;
      r.lb = std::max(preset.rel * mv, preset.min_var_lv_exo);
    } else {
      r.lb = preset.min_var_lv_exo;
      r.ub = max_ov_var[static_cast<std::size_t>(blk)];
    }
    if (r.lb < 0.0) r.lb = 0.0;
    r.lb2 = (c >= 0 && static_cast<std::size_t>(c) < n_vars &&
             is_endo[static_cast<std::size_t>(c)])
                ? preset.min_var_lv_endo
                : r.lb;
    return r;
  };

  auto put_lower = [&](std::int32_t fr, double lo) {
    if (!std::isfinite(lo)) return;
    const Eigen::Index k = fr - 1;
    out.lower(k) = std::max(out.lower(k), lo);
  };
  auto put_upper = [&](std::int32_t fr, double hi) {
    if (std::isnan(hi)) return;
    const Eigen::Index k = fr - 1;
    out.upper(k) = std::min(out.upper(k), hi);
  };

  // Pass A — variances and loadings. Records each free variance's (enlarged)
  // upper, keyed (block, var id), for the covariance pass.
  std::map<std::pair<std::int32_t, std::int32_t>, double> var_upper;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const std::int32_t fr = pt.free[i];
    if (fr <= 0) continue;
    if (fr > n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "auto bounds: free[" + std::to_string(i) + "] exceeds n_free"));
    }
    if (pt.group[i] <= 0) continue;
    const std::int32_t blk = pt.group[i] - 1;
    const parse::Op op = pt.op[i];
    const std::int32_t lhs = pt.lhs_var[i];
    const std::int32_t rhs = pt.rhs_var[i];

    if (op == parse::Op::Covariance && lhs >= 0 && lhs == rhs) {
      if (is_latent(lhs)) {
        const LvVar lv = lv_bounds(lhs, blk);
        if (!std::isfinite(lv.ub)) continue;
        double lo = lv.lb2;
        double hi = lv.ub;
        const double range = hi - std::max(lo, 0.0);
        enlarge_abs(lo, hi, range, preset.lv_var, /*is_ov_var=*/false);
        var_upper[{blk, lhs}] = hi;
        if (preset.lv_var.lower) put_lower(fr, lo);
        if (preset.lv_var.upper) put_upper(fr, hi);
      } else {
        const double ovv = ov_var(lhs, blk);
        if (!std::isfinite(ovv)) continue;
        double lo = 0.0;
        double hi = (preset.rel > 0.0 &&
                     static_cast<std::size_t>(lhs) < n_vars &&
                     is_marker[static_cast<std::size_t>(lhs)])
                        ? (1.0 - preset.rel) * ovv
                        : ovv;
        const double range = hi - std::max(lo, 0.0);
        enlarge_abs(lo, hi, range, preset.ov_var, /*is_ov_var=*/true);
        var_upper[{blk, lhs}] = hi;
        if (preset.ov_var.lower) put_lower(fr, lo);
        if (preset.ov_var.upper) put_upper(fr, hi);
      }
    } else if (op == parse::Op::Measurement) {
      if (!preset.loadings.lower && !preset.loadings.upper) continue;
      const double var_all = ov_var(rhs, blk);
      if (!std::isfinite(var_all)) continue;
      const LvVar lv = lv_bounds(lhs, blk);
      const double tmp = std::max(lv.lb, 0.0);
      if (!(tmp > 0.0)) continue;             // unbounded loading (lavaan: ±Inf)
      const double raw = std::sqrt(var_all / tmp);
      double lo = -raw;
      double hi = +raw;
      enlarge_abs(lo, hi, hi - lo, preset.loadings, /*is_ov_var=*/false);
      if (preset.loadings.lower) put_lower(fr, lo);
      if (preset.loadings.upper) put_upper(fr, hi);
    }
  }

  // Pass B — covariances, derived from the free-variance uppers of Pass A.
  if (preset.covariances.lower || preset.covariances.upper) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      const std::int32_t fr = pt.free[i];
      if (fr <= 0 || pt.group[i] <= 0) continue;
      if (pt.op[i] != parse::Op::Covariance) continue;
      const std::int32_t lhs = pt.lhs_var[i];
      const std::int32_t rhs = pt.rhs_var[i];
      if (lhs < 0 || rhs < 0 || lhs == rhs) continue;
      const std::int32_t blk = pt.group[i] - 1;
      const auto lit = var_upper.find({blk, lhs});
      const auto rit = var_upper.find({blk, rhs});
      if (lit == var_upper.end() || rit == var_upper.end()) continue;
      if (!(lit->second >= 0.0) || !(rit->second >= 0.0)) continue;
      const double ucov = std::sqrt(lit->second) * std::sqrt(rit->second);
      if (!std::isfinite(ucov)) continue;
      double lo = -ucov;
      double hi = +ucov;
      enlarge_cov(lo, hi, hi - lo, preset.covariances);
      if (preset.covariances.lower) put_lower(fr, lo);
      if (preset.covariances.upper) put_upper(fr, hi);
    }
  }

  return out;
}

}  // namespace

post_expected<Bounds>
variance_bounds(const spec::LatentStructure& pt) {
  const auto n_free = pt.n_free();
  Bounds b;
  b.lower = Eigen::VectorXd::Constant(n_free, -kInf);
  b.upper = Eigen::VectorXd::Constant(n_free,  kInf);
  if (n_free == 0) return b;

  // A variance row in lavaan-speak: `lhs ~~ rhs` with `lhs == rhs`. Could be a
  // latent variance (Ψ-diag, e.g. `f ~~ f`) or a residual variance (Θ-diag,
  // e.g. `x1 ~~ x1`). Either way: free → lower bound at zero (Heywood barrier).
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Covariance) continue;
    if (pt.lhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i]) continue;
    const auto fr = pt.free[i];
    if (fr <= 0) continue;                  // fixed param: no bound needed
    if (fr > n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "variance_bounds: free[" + std::to_string(i) + "] = " +
              std::to_string(fr) + " exceeds n_free = " +
              std::to_string(n_free)));
    }
    // `free` is 1-based; theta index is `fr - 1`. Setting the same coordinate
    // twice (e.g. shared-label invariance across groups) is idempotent.
    b.lower(static_cast<Eigen::Index>(fr - 1)) = 0.0;
  }
  return b;
}

post_expected<Bounds>
standard_bounds(const spec::LatentStructure& pt,
                const data::SampleStats& samp) {
  return compute_auto_bounds(pt, samp, standard_preset());
}

post_expected<Bounds>
wide_bounds(const spec::LatentStructure& pt, const data::SampleStats& samp) {
  return compute_auto_bounds(pt, samp, wide_preset());
}

post_expected<Bounds>
loading_bounds(const spec::LatentStructure& pt,
               const data::SampleStats& samp) {
  return compute_auto_bounds(pt, samp, loading_preset());
}

post_expected<Bounds>
intersect_bounds(const Bounds& a, const Bounds& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (a.lower.size() != b.lower.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "intersect_bounds: operand sizes differ"));
  }
  Bounds out;
  out.lower = a.lower.cwiseMax(b.lower);
  out.upper = a.upper.cwiseMin(b.upper);
  return out;
}

post_expected<StartProjection>
project_start_into_bounds(const Eigen::VectorXd& x0, const Bounds& bounds) {
  StartProjection out;
  out.x0 = x0;
  if (bounds.empty()) return out;
  if (bounds.lower.size() != x0.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "project_start_into_bounds: x0 size does not match the bounds"));
  }
  for (Eigen::Index k = 0; k < x0.size(); ++k) {
    const double clamped =
        std::min(std::max(x0(k), bounds.lower(k)), bounds.upper(k));
    if (clamped != x0(k)) {
      out.x0(k) = clamped;
      out.clamped.push_back(static_cast<std::int32_t>(k));
    }
  }
  return out;
}

post_expected<ActiveBoundDiagnostics>
active_bounds(const Eigen::VectorXd& theta, const Bounds& bounds, double tol) {
  ActiveBoundDiagnostics out;
  if (bounds.empty()) return out;
  if (bounds.lower.size() != theta.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "active_bounds: theta size does not match the bounds"));
  }
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    if (std::isfinite(bounds.lower(k)) &&
        std::abs(theta(k) - bounds.lower(k)) <= tol) {
      out.at_lower.push_back(static_cast<std::int32_t>(k));
    }
    if (std::isfinite(bounds.upper(k)) &&
        std::abs(theta(k) - bounds.upper(k)) <= tol) {
      out.at_upper.push_back(static_cast<std::int32_t>(k));
    }
  }
  return out;
}

}  // namespace magmaan::estimate
