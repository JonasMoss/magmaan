#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/optim/concepts.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using optim::LbfgsOutput;
using optim::LsResidualFn;
using optim::LsJacobianFn;

struct SnllsDiagnostics {
  std::int32_t n_nonlinear = 0;
  std::int32_t n_linear = 0;
  std::int32_t profile_evaluations = 0;
  std::int32_t profile_cache_hits = 0;
  std::int32_t gradient_evaluations = 0;
  std::int32_t jacobian_evaluations = 0;
  bool admissible = true;
  double min_variance = std::numeric_limits<double>::infinity();
};

struct SnllsEstimates : Estimates {
  SnllsDiagnostics snlls;
};

namespace detail_snlls {

enum class BlockKind : std::uint8_t { None, Nonlinear, Linear };

inline FitError fit_err(FitError::Kind kind, std::string detail,
                        int iterations = 0, double f = 0.0) {
  return FitError{kind, std::move(detail), iterations, f};
}

inline BlockKind block_kind_for(model::MatId mat) noexcept {
  switch (mat) {
    case model::MatId::Lambda:
    case model::MatId::Beta:
      return BlockKind::Nonlinear;
    case model::MatId::Theta:
    case model::MatId::Psi:
    case model::MatId::Nu:
    case model::MatId::Alpha:
      return BlockKind::Linear;
  }
  return BlockKind::None;
}

inline std::string_view block_kind_name(BlockKind k) noexcept {
  switch (k) {
    case BlockKind::None: return "none";
    case BlockKind::Nonlinear: return "nonlinear";
    case BlockKind::Linear: return "linear";
  }
  return "?";
}

struct Classification {
  EqConstraints constraints;
  std::vector<Eigen::Index> beta_cols;
  std::vector<Eigen::Index> alpha_cols;
  Eigen::VectorXd beta0;
  Eigen::MatrixXd K_beta;
  Eigen::MatrixXd K_alpha;
  std::vector<BlockKind> full_kind;
};

inline fit_expected<Classification>
classify(const spec::LatentStructure& pt,
         const model::ModelEvaluator& ev,
         const Eigen::VectorXd& theta_start) {
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS compatibility: constraint: " + con_or.error().detail));
  }
  EqConstraints con = std::move(*con_or);
  const auto locs = ev.param_locations();
  if (static_cast<std::int32_t>(locs.size()) != pt.n_free()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS compatibility: parameter-location count does not match n_free"));
  }

  Classification out;
  out.constraints = std::move(con);
  out.full_kind.assign(locs.size(), BlockKind::None);
  for (std::size_t k = 0; k < locs.size(); ++k) {
    out.full_kind[k] = block_kind_for(locs[k].mat);
    if (out.full_kind[k] == BlockKind::None) {
      return std::unexpected(fit_err(
          FitError::Kind::NumericIssue,
          "SNLLS compatibility: free parameter " + std::to_string(k + 1) +
              " has unsupported matrix location"));
    }
  }

  constexpr double tol = 1e-10;
  for (Eigen::Index c = 0; c < out.constraints.Kmat.cols(); ++c) {
    BlockKind kind = BlockKind::None;
    for (Eigen::Index r = 0; r < out.constraints.Kmat.rows(); ++r) {
      if (std::abs(out.constraints.Kmat(r, c)) <= tol) continue;
      const BlockKind rk = out.full_kind[static_cast<std::size_t>(r)];
      if (kind == BlockKind::None) {
        kind = rk;
      } else if (kind != rk) {
        return std::unexpected(fit_err(
            FitError::Kind::NumericIssue,
            "SNLLS compatibility: equality/linear constraint basis column " +
                std::to_string(c + 1) + " mixes " +
                std::string(block_kind_name(kind)) + " and " +
                std::string(block_kind_name(rk)) + " parameters"));
      }
    }
    if (kind == BlockKind::None) continue;
    if (kind == BlockKind::Nonlinear) out.beta_cols.push_back(c);
    if (kind == BlockKind::Linear) out.alpha_cols.push_back(c);
  }

  if (out.alpha_cols.empty()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS compatibility: no conditionally linear free parameters to profile"));
  }

  const Eigen::VectorXd q0 = out.constraints.contract(theta_start);
  out.beta0.resize(static_cast<Eigen::Index>(out.beta_cols.size()));
  for (std::size_t i = 0; i < out.beta_cols.size(); ++i) {
    out.beta0(static_cast<Eigen::Index>(i)) = q0(out.beta_cols[i]);
  }

  out.K_beta.resize(out.constraints.npar,
                    static_cast<Eigen::Index>(out.beta_cols.size()));
  for (std::size_t j = 0; j < out.beta_cols.size(); ++j) {
    out.K_beta.col(static_cast<Eigen::Index>(j)) =
        out.constraints.Kmat.col(out.beta_cols[j]);
  }
  out.K_alpha.resize(out.constraints.npar,
                     static_cast<Eigen::Index>(out.alpha_cols.size()));
  for (std::size_t j = 0; j < out.alpha_cols.size(); ++j) {
    out.K_alpha.col(static_cast<Eigen::Index>(j)) =
        out.constraints.Kmat.col(out.alpha_cols[j]);
  }
  return out;
}

inline Eigen::VectorXd expand_beta(const Classification& cls,
                                   const Eigen::VectorXd& beta) {
  if (cls.K_beta.cols() == 0) return cls.constraints.theta0;
  return cls.constraints.theta0 + cls.K_beta * beta;
}

struct Profile {
  Eigen::VectorXd theta;
  Eigen::VectorXd residual;
  Eigen::MatrixXd H;
  double fmin = 0.0;
};

struct Counters {
  std::int32_t profile_evaluations = 0;
  std::int32_t profile_cache_hits = 0;
  std::int32_t gradient_evaluations = 0;
  std::int32_t jacobian_evaluations = 0;
};

struct ProfileCache {
  bool valid = false;
  Eigen::VectorXd beta;
  Profile profile;
};

inline bool same_beta(const Eigen::VectorXd& a,
                      const Eigen::VectorXd& b) noexcept {
  if (a.size() != b.size()) return false;
  if (a.size() == 0) return true;
  return (a.array() == b.array()).all();
}

template <LsDiscrepancy D>
fit_expected<Profile>
profile_at(const model::ModelEvaluator& ev,
           const SampleStats& samp,
           const D& discrepancy,
           const Classification& cls,
           const Eigen::VectorXd& beta) {
  Profile out;
  const Eigen::VectorXd theta_base = expand_beta(cls, beta);

  auto eval = ev.evaluate(theta_base, true, true);
  if (!eval.has_value()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS profile evaluate: " + eval.error().detail));
  }
  auto r0_or = discrepancy.residuals(samp, eval->moments);
  if (!r0_or.has_value()) return std::unexpected(r0_or.error());
  auto Jr_or = discrepancy.residual_jacobian(
      samp, eval->moments, eval->J_sigma, eval->J_mu);
  if (!Jr_or.has_value()) return std::unexpected(Jr_or.error());
  if (Jr_or->rows() != r0_or->size() ||
      Jr_or->cols() != cls.K_alpha.rows()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS profiling: residual Jacobian shape does not match residuals/free parameters"));
  }
  const Eigen::VectorXd& r0 = *r0_or;

  const Eigen::Index n_alpha = cls.K_alpha.cols();
  out.H = (*Jr_or) * cls.K_alpha;

  Eigen::VectorXd alpha_hat(n_alpha);
  if (n_alpha == 0) {
    alpha_hat.resize(0);
    out.residual = r0;
  } else {
    alpha_hat = out.H.colPivHouseholderQr().solve(-r0);
    out.residual = r0 + out.H * alpha_hat;
  }
  out.theta = theta_base + cls.K_alpha * alpha_hat;
  out.fmin = 0.5 * out.residual.squaredNorm();
  if (!std::isfinite(out.fmin) || !out.theta.allFinite() ||
      !out.residual.allFinite()) {
    return std::unexpected(fit_err(
        FitError::Kind::NonFiniteObjective,
        "SNLLS profiling produced non-finite values"));
  }
  return out;
}

template <LsDiscrepancy D>
fit_expected<Profile>
cached_profile_at(ProfileCache& cache,
                  Counters& counters,
                  const model::ModelEvaluator& ev,
                  const SampleStats& samp,
                  const D& discrepancy,
                  const Classification& cls,
                  const Eigen::VectorXd& beta) {
  if (cache.valid && same_beta(cache.beta, beta)) {
    ++counters.profile_cache_hits;
    return cache.profile;
  }
  auto prof = profile_at(ev, samp, discrepancy, cls, beta);
  if (!prof.has_value()) return std::unexpected(prof.error());
  ++counters.profile_evaluations;
  cache.valid = true;
  cache.beta = beta;
  cache.profile = *prof;
  return cache.profile;
}

template <LsDiscrepancy D>
fit_expected<Eigen::MatrixXd>
profile_jacobian_at(const model::ModelEvaluator& ev,
                    const SampleStats& samp,
                    const D& discrepancy,
                    const Classification& cls,
                    const Profile& prof) {
  auto eval = ev.evaluate(prof.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS jacobian evaluate: " + eval.error().detail));
  }
  auto J_or = discrepancy.residual_jacobian(
      samp, eval->moments, eval->J_sigma, eval->J_mu);
  if (!J_or.has_value()) return std::unexpected(J_or.error());
  if (J_or->cols() != cls.K_beta.rows()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS jacobian: full Jacobian column count does not match n_free"));
  }
  Eigen::MatrixXd Jb = *J_or * cls.K_beta;
  if (prof.H.cols() > 0) {
    const Eigen::MatrixXd coeff = prof.H.colPivHouseholderQr().solve(Jb);
    Jb.noalias() -= prof.H * coeff;
  }
  return Jb;
}

template <LsDiscrepancy D>
fit_expected<Eigen::VectorXd>
profile_gradient_at(const model::ModelEvaluator& ev,
                    const SampleStats& samp,
                    const D& discrepancy,
                    const Classification& cls,
                    const Profile& prof) {
  auto eval = ev.evaluate(prof.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS gradient evaluate: " + eval.error().detail));
  }
  auto J_or = discrepancy.residual_jacobian(
      samp, eval->moments, eval->J_sigma, eval->J_mu);
  if (!J_or.has_value()) return std::unexpected(J_or.error());
  if (J_or->cols() != cls.K_beta.rows()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS gradient: full Jacobian column count does not match n_free"));
  }
  const Eigen::MatrixXd Jb = (*J_or) * cls.K_beta;
  return Jb.transpose() * prof.residual;
}

template <class O>
inline constexpr bool has_optimizer_name = requires {
  { std::remove_cvref_t<O>::name } -> std::convertible_to<std::string_view>;
};

template <class O>
constexpr bool use_native_ls_optimizer_v() {
  if constexpr (has_optimizer_name<O>) {
    return std::string_view(std::remove_cvref_t<O>::name) != "lbfgsb";
  } else {
    return true;
  }
}

inline Bounds beta_bounds_from(const Classification& cls,
                               const Bounds& full_bounds) {
  Bounds b;
  const Eigen::Index nb = cls.K_beta.cols();
  b.lower = Eigen::VectorXd::Constant(
      nb, -std::numeric_limits<double>::infinity());
  b.upper = Eigen::VectorXd::Constant(
      nb, std::numeric_limits<double>::infinity());
  if (full_bounds.lower.size() != cls.constraints.npar ||
      full_bounds.upper.size() != cls.constraints.npar) {
    return b;
  }

  constexpr double tol = 1e-12;
  for (Eigen::Index j = 0; j < nb; ++j) {
    Eigen::Index support = -1;
    double scale = 0.0;
    bool simple = true;
    for (Eigen::Index r = 0; r < cls.K_beta.rows(); ++r) {
      const double v = cls.K_beta(r, j);
      if (std::abs(v) <= tol) continue;
      if (support >= 0) { simple = false; break; }
      support = r;
      scale = v;
    }
    if (!simple || support < 0 || std::abs(scale) <= tol) continue;
    const double lo = full_bounds.lower(support);
    const double hi = full_bounds.upper(support);
    const double offset = cls.constraints.theta0(support);
    if (scale > 0.0) {
      b.lower(j) = (lo - offset) / scale;
      b.upper(j) = (hi - offset) / scale;
    } else {
      b.lower(j) = (hi - offset) / scale;
      b.upper(j) = (lo - offset) / scale;
    }
  }
  return b;
}

inline SnllsDiagnostics diagnostics_for(const spec::LatentStructure& pt,
                                        const model::MatrixRep& rep,
                                        const Classification& cls,
                                        const Eigen::VectorXd& theta,
                                        const Counters& counters = {}) {
  SnllsDiagnostics d;
  d.n_nonlinear = static_cast<std::int32_t>(cls.beta_cols.size());
  d.n_linear = static_cast<std::int32_t>(cls.alpha_cols.size());
  d.profile_evaluations = counters.profile_evaluations;
  d.profile_cache_hits = counters.profile_cache_hits;
  d.gradient_evaluations = counters.gradient_evaluations;
  d.jacobian_evaluations = counters.jacobian_evaluations;
  d.admissible = true;

  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    if (i >= rep.cell_for_row.size()) continue;
    const auto& c = rep.cell_for_row[i];
    if (!c.used) continue;
    if ((c.mat == model::MatId::Theta || c.mat == model::MatId::Psi) &&
        c.row == c.col) {
      const double v = theta(pt.free[i] - 1);
      d.min_variance = std::min(d.min_variance, v);
      if (v < -1e-10) d.admissible = false;
    }
  }
  return d;
}

}  // namespace detail_snlls

template <LsDiscrepancy D, BoundedOptimizer O>
fit_expected<SnllsEstimates>
fit_snlls_bounded(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const SampleStats& samp,
                  Bounds bounds,
                  D discrepancy = {},
                  O optimizer = {},
                  spec::Starts starts = {}) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(e.error());
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(detail_snlls::fit_err(
        FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const model::ModelEvaluator ev = std::move(*ev_or);

  auto x0_or = simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(detail_snlls::fit_err(
          FitError::Kind::NumericIssue,
          "fit_snlls_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }

  auto cls_or = detail_snlls::classify(pt, ev, *x0_or);
  if (!cls_or.has_value()) return std::unexpected(cls_or.error());
  const detail_snlls::Classification cls = std::move(*cls_or);

  detail_snlls::Counters counters;
  auto r_start = detail_snlls::profile_at(ev, samp, discrepancy, cls, cls.beta0);
  ++counters.profile_evaluations;
  if (!r_start.has_value()) return std::unexpected(r_start.error());
  const Eigen::Index n_resid = r_start->residual.size();

  if (cls.beta0.size() == 0) {
    SnllsEstimates est;
    est.theta = std::move(r_start->theta);
    est.fmin = r_start->fmin;
    est.iterations = 0;
    est.snlls = detail_snlls::diagnostics_for(pt, rep, cls, est.theta,
                                              counters);
    return est;
  }

  Bounds beta_bounds = detail_snlls::beta_bounds_from(cls, bounds);
  auto profile_cache = std::make_shared<detail_snlls::ProfileCache>();
  auto counter_state = std::make_shared<detail_snlls::Counters>(counters);
  profile_cache->valid = true;
  profile_cache->beta = cls.beta0;
  profile_cache->profile = *r_start;

  auto residual_fn = [&, profile_cache, counter_state](const Eigen::VectorXd& beta)
      -> fit_expected<Eigen::VectorXd> {
    auto prof = detail_snlls::cached_profile_at(
        *profile_cache, *counter_state, ev, samp, discrepancy, cls, beta);
    if (!prof.has_value()) return std::unexpected(prof.error());
    return prof->residual;
  };
  auto jacobian_fn = [&, profile_cache, counter_state](const Eigen::VectorXd& beta)
      -> fit_expected<Eigen::MatrixXd> {
    auto prof = detail_snlls::cached_profile_at(
        *profile_cache, *counter_state, ev, samp, discrepancy, cls, beta);
    if (!prof.has_value()) return std::unexpected(prof.error());
    ++counter_state->jacobian_evaluations;
    return detail_snlls::profile_jacobian_at(ev, samp, discrepancy, cls, *prof);
  };
  auto objective_fn = [&, profile_cache, counter_state](
                         const Eigen::VectorXd& beta,
                         Eigen::VectorXd& grad) -> double {
    auto prof = detail_snlls::cached_profile_at(
        *profile_cache, *counter_state, ev, samp, discrepancy, cls, beta);
    if (!prof.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    ++counter_state->gradient_evaluations;
    auto g = detail_snlls::profile_gradient_at(
        ev, samp, discrepancy, cls, *prof);
    if (!g.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    grad = *g;
    return prof->fmin;
  };

  fit_expected<LbfgsOutput> out_or =
      [&]() -> fit_expected<LbfgsOutput> {
        if constexpr (LsBoundedOptimizer<O> &&
                      detail_snlls::use_native_ls_optimizer_v<O>()) {
          return optimizer.minimize_ls(LsResidualFn(residual_fn),
                                       LsJacobianFn(jacobian_fn),
                                       n_resid,
                                       cls.beta0,
                                       beta_bounds.lower,
                                       beta_bounds.upper);
        } else {
          return optimizer.minimize(objective_fn,
                                    cls.beta0,
                                    beta_bounds.lower,
                                    beta_bounds.upper);
        }
      }();
  if (!out_or.has_value()) return std::unexpected(out_or.error());

  auto final_prof = detail_snlls::cached_profile_at(
      *profile_cache, *counter_state, ev, samp, discrepancy, cls, out_or->theta_hat);
  if (!final_prof.has_value()) return std::unexpected(final_prof.error());

  SnllsEstimates est;
  est.theta = std::move(final_prof->theta);
  est.fmin = final_prof->fmin;
  est.iterations = out_or->iterations;
  est.snlls = detail_snlls::diagnostics_for(pt, rep, cls, est.theta,
                                            *counter_state);
  return est;
}

}  // namespace magmaan::estimate
