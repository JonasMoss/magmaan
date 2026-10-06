#include "magmaan/measures/standardized.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/op.hpp"

#include "detail_vech.hpp"

namespace magmaan::measures::standardize {

using data::SampleStats;
using estimate::Estimates;

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

post_expected<StandardizedRows>
standardized_rows(const spec::LatentStructure& pt,
                  const model::MatrixRep& rep,
                  const Estimates& est,
                  const Eigen::MatrixXd& vcov,
                  bool ordinal_delta) {
  const Eigen::Index n = est.theta.size();
  if (vcov.rows() != n || vcov.cols() != n)
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardized_rows: covariance shape does not match theta"));
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or) return std::unexpected(make_err(PostError::Kind::NumericIssue,
      "standardized_rows: " + ev_or.error().detail));
  const auto& ev = *ev_or;
  const double missing = std::numeric_limits<double>::quiet_NaN();
  auto values = [&](const Eigen::VectorXd& theta, bool all)
      -> post_expected<Eigen::VectorXd> {
    auto assembled = ev.assembled(theta);
    auto moments = ev.sigma(theta);
    if (!assembled || !moments)
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "standardized_rows: model evaluation failed"));
    auto raw = [&](std::size_t i) {
      return pt.free[i] > 0 ? theta(pt.free[i] - 1) : pt.fixed_value[i];
    };
    Eigen::VectorXd out = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(pt.size()), missing);
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.group[i] <= 0 || pt.lhs_var[i] < 0) continue;
      const auto b = static_cast<std::size_t>(pt.group[i] - 1);
      const auto& am = assembled->blocks[b];
      auto is_ordered = [&](std::int32_t v) {
        for (std::size_t j = 0; j < pt.size(); ++j)
          if (pt.group[j] == pt.group[i] && pt.lhs_var[j] == v &&
              pt.op[j] == parse::Op::Threshold) return true;
        return false;
      };
      auto variance = [&](std::int32_t v) {
        if (pt.is_user_latent[static_cast<std::size_t>(v)]) return am.Mid(pt.lv_ext_pos[static_cast<std::size_t>(v)], pt.lv_ext_pos[static_cast<std::size_t>(v)]);
        if (ordinal_delta && is_ordered(v)) {
          double delta = 1.0;
          for (std::size_t j = 0; j < pt.size(); ++j)
            if (pt.group[j] == pt.group[i] && pt.lhs_var[j] == v &&
                pt.op[j] == parse::Op::ResponseScale) delta = raw(j);
          return 1.0 / (delta * delta);
        }
        return moments->sigma[b](pt.ov_pos[static_cast<std::size_t>(v)], pt.ov_pos[static_cast<std::size_t>(v)]);
      };
      auto residual = [&](std::int32_t v) {
        if (pt.is_user_latent[static_cast<std::size_t>(v)]) return am.Psi(pt.lv_ext_pos[static_cast<std::size_t>(v)], pt.lv_ext_pos[static_cast<std::size_t>(v)]);
        const auto o = pt.ov_pos[static_cast<std::size_t>(v)];
        if (ordinal_delta && is_ordered(v))
          return variance(v) - (am.Lambda.row(o) * am.Mid *
                                 am.Lambda.row(o).transpose())(0, 0);
        // Reduced LISREL represents structural observed residuals in Psi.
        const auto l = pt.lv_ext_pos[static_cast<std::size_t>(v)];
        return l >= 0 ? am.Psi(l, l) : am.Theta(o, o);
      };
      auto scale = [&](std::int32_t v) {
        if (!all && !pt.is_user_latent[static_cast<std::size_t>(v)]) return 1.0;
        const double value = variance(v);
        return value > 0 ? std::sqrt(value) : missing;
      };
      const auto l = pt.lhs_var[i], r = pt.rhs_var[i];
      double value = raw(i);
      switch (pt.op[i]) {
        case parse::Op::Measurement:
          value *= scale(l) / scale(r); break;
        case parse::Op::Regression:
          value *= scale(r) / scale(l); break;
        case parse::Op::Covariance:
          if (l == r) value = residual(l) / (scale(l) * scale(l));
          else {
            // lavaan cov.std scales residual covariances by residual SDs.
            auto cov_scale = [&](std::int32_t v) {
              if (!all && !pt.is_user_latent[static_cast<std::size_t>(v)]) return 1.0;
              return std::sqrt(std::abs(residual(v)));
            };
            value /= cov_scale(l) * cov_scale(r);
          }
          break;
        case parse::Op::Intercept:
        case parse::Op::Threshold:
          value /= scale(l); break;
        case parse::Op::ResponseScale:
          value = all ? 1.0 : (ordinal_delta ? value : 1.0 / std::sqrt(variance(l)));
          break;
        default: value = missing; break;
      }
      out(static_cast<Eigen::Index>(i)) = value;
    }
    return out;
  };
  auto lv = values(est.theta, false), all = values(est.theta, true);
  if (!lv) return std::unexpected(lv.error());
  if (!all) return std::unexpected(all.error());
  Eigen::MatrixXd jl(static_cast<Eigen::Index>(pt.size()), n), ja(static_cast<Eigen::Index>(pt.size()), n);
  for (Eigen::Index k = 0; k < n; ++k) {
    // Central differences with cube-root epsilon balance rounding and truncation.
    const double h = std::cbrt(std::numeric_limits<double>::epsilon()) *
                     (1.0 + std::abs(est.theta(k)));
    Eigen::VectorXd plus = est.theta, minus = est.theta;
    plus(k) += h; minus(k) -= h;
    auto lp = values(plus, false), lm = values(minus, false);
    auto ap = values(plus, true), a_m = values(minus, true);
    if (!lp || !lm || !ap || !a_m)
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "standardized_rows: perturbed model evaluation failed"));
    jl.col(k) = (*lp - *lm) / (2 * h);
    ja.col(k) = (*ap - *a_m) / (2 * h);
  }
  auto errors = [&](const Eigen::MatrixXd& j) {
    Eigen::VectorXd se(static_cast<Eigen::Index>(pt.size()));
    for (Eigen::Index i = 0; i < se.size(); ++i) {
      const double v = (j.row(i) * vcov * j.row(i).transpose())(0, 0);
      se(i) = std::isfinite(v) ? std::sqrt(std::max(0.0, v)) : missing;
    }
    return se;
  };
  return StandardizedRows{*lv, errors(jl), *all, errors(ja)};
}

post_expected<StandardizedSolution>
standardize_lv(const spec::LatentStructure& pt,
               const model::MatrixRep&   rep,
               const Estimates&          est,
               const Eigen::MatrixXd&    vcov) {
  const Eigen::Index n_free = est.theta.size();
  if (vcov.rows() != n_free || vcov.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_lv: vcov shape doesn't match Estimates.theta size"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_lv: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  const auto locs = ev.param_locations();

  // Build the Jacobian J of θ → θ_std and the values theta_std. Each row k
  // of J is the gradient of g_k(θ).
  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n_free, n_free);
  Eigen::VectorXd theta_std = est.theta;

  auto latent_std_value =
      [&](const Eigen::VectorXd& theta,
          const model::ParamLocation& L) -> post_expected<double> {
    auto a_or = ev.assembled(theta);
    if (!a_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "standardize_lv: ev.assembled failed while standardizing latent "
          "parameter: " + a_or.error().detail));
    }
    const auto& bmat = a_or->blocks[static_cast<std::size_t>(L.block)];
    auto latent_var = [&](Eigen::Index j) -> post_expected<double> {
      const double v = bmat.Mid(j, j);
      if (v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "standardize_lv: non-positive latent variance; cannot standardize"));
      }
      return v;
    };
    switch (L.mat) {
      case model::MatId::Lambda: {
        auto v_or = latent_var(L.col);
        if (!v_or.has_value()) return std::unexpected(v_or.error());
        return bmat.Lambda(L.row, L.col) * std::sqrt(*v_or);
      }
      case model::MatId::Psi: {
        auto vr_or = latent_var(L.row);
        auto vc_or = latent_var(L.col);
        if (!vr_or.has_value()) return std::unexpected(vr_or.error());
        if (!vc_or.has_value()) return std::unexpected(vc_or.error());
        return bmat.Psi(L.row, L.col) / std::sqrt((*vr_or) * (*vc_or));
      }
      case model::MatId::Beta: {
        auto vy_or = latent_var(L.row);
        auto vx_or = latent_var(L.col);
        if (!vy_or.has_value()) return std::unexpected(vy_or.error());
        if (!vx_or.has_value()) return std::unexpected(vx_or.error());
        return bmat.Beta(L.row, L.col) * std::sqrt(*vx_or) / std::sqrt(*vy_or);
      }
      case model::MatId::Alpha: {
        auto v_or = latent_var(L.row);
        if (!v_or.has_value()) return std::unexpected(v_or.error());
        return bmat.Alpha(L.row) / std::sqrt(*v_or);
      }
      case model::MatId::Theta:
      case model::MatId::Nu:
        return theta(static_cast<Eigen::Index>(&L - locs.data()));
    }
    return theta(static_cast<Eigen::Index>(&L - locs.data()));
  };

  auto fill_fd_row = [&](Eigen::Index ki, const model::ParamLocation& L)
      -> post_expected<void> {
    auto val_or = latent_std_value(est.theta, L);
    if (!val_or.has_value()) return std::unexpected(val_or.error());
    theta_std(ki) = *val_or;
    J.row(ki).setZero();
    for (Eigen::Index j = 0; j < n_free; ++j) {
      const double base = std::abs(est.theta(j)) + 1.0;
      const double h = std::sqrt(std::numeric_limits<double>::epsilon()) * base;
      Eigen::VectorXd tp = est.theta;
      Eigen::VectorXd tm = est.theta;
      tp(j) += h;
      tm(j) -= h;
      auto vp = latent_std_value(tp, L);
      auto vm = latent_std_value(tm, L);
      if (!vp.has_value()) return std::unexpected(vp.error());
      if (!vm.has_value()) return std::unexpected(vm.error());
      J(ki, j) = (*vp - *vm) / (2.0 * h);
    }
    return {};
  };

  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    if (L.row < 0 || L.block < 0) continue;
    const Eigen::Index ki = static_cast<Eigen::Index>(k);
    switch (L.mat) {
      case model::MatId::Lambda:
      case model::MatId::Psi:
      case model::MatId::Beta:
      case model::MatId::Alpha: {
        auto row_or = fill_fd_row(ki, L);
        if (!row_or.has_value()) return std::unexpected(row_or.error());
        break;
      }
      case model::MatId::Theta:
      case model::MatId::Nu:
        // Identity transform — J already has 1 on the diagonal from
        // the initial Identity, value already correct in theta_std.
        break;
    }
  }

  StandardizedSolution out;
  out.theta = std::move(theta_std);
  // SE_std = √diag(J · vcov · Jᵀ). Compute J · vcov · Jᵀ explicitly for
  // small v0 sizes; for production-scale code we'd row-wise solve.
  const Eigen::MatrixXd JV  = J * vcov;
  out.se.resize(n_free);
  for (Eigen::Index k = 0; k < n_free; ++k) {
    const double var = JV.row(k).dot(J.row(k));
    out.se(k) = (var > 0.0) ? std::sqrt(var) : 0.0;
  }
  return out;
}

namespace {
using detail::vech_index;
using detail::vech_len;
}  // namespace

post_expected<StandardizedSolution>
standardize_all(const spec::LatentStructure& pt,
                const model::MatrixRep&   rep,
                const Estimates&          est,
                const Eigen::MatrixXd&    vcov,
                bool ordinal_delta_unit) {
  const Eigen::Index n_free = est.theta.size();
  if (vcov.rows() != n_free || vcov.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: vcov shape doesn't match Estimates.theta size"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;

  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ev.sigma failed: " + sm_or.error().detail));
  }
  auto J_or = ev.dsigma_dtheta(est.theta);
  if (!J_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ev.dsigma_dtheta failed: " + J_or.error().detail));
  }
  auto am_or = ev.assembled(est.theta);
  if (!am_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardize_all: ev.assembled failed: " + am_or.error().detail));
  }
  const auto& sm = *sm_or;
  const auto& Js = *J_or;
  const auto& am = *am_or;
  const auto  locs = ev.param_locations();
  const std::size_t n_blocks = am.blocks.size();

  // Ordinal (delta) indicators have unit-variance latent responses, so their
  // standardized loadings must be divided by the latent SD only, not by the
  // assembled σ_rr (which carries a fixed unit residual). Mark those indicator
  // rows per block from the threshold rows of the partable.
  std::vector<std::vector<char>> ordinal_ov(n_blocks);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    ordinal_ov[b].assign(
        static_cast<std::size_t>(am.blocks[b].Lambda.rows()), 0);
  }
  if (ordinal_delta_unit) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] != parse::Op::Threshold) continue;
      const std::int32_t g = pt.group[i];
      const std::int32_t vid = pt.lhs_var[i];
      if (g <= 0 || vid < 0) continue;
      const std::size_t b = static_cast<std::size_t>(g - 1);
      if (b >= n_blocks) continue;
      const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(vid)];
      if (ov >= 0 &&
          ov < static_cast<std::int32_t>(ordinal_ov[b].size())) {
        ordinal_ov[b][static_cast<std::size_t>(ov)] = 1;
      }
    }
  }

  // Per-block vech offset into the Jacobian Js (rows = sum_b vech_len(p_b)).
  std::vector<Eigen::Index> vech_off(n_blocks, 0);
  Eigen::Index running = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    vech_off[b] = running;
    running += vech_len(static_cast<Eigen::Index>(am.blocks[b].Lambda.rows()));
  }

  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n_free, n_free);
  Eigen::VectorXd theta_std = est.theta;

  // Helper: add `−scalar · ∂σ_rr/∂θ_*` to J row `k_out`. ∂σ_rr/∂θ comes
  // from row `vech_off[b] + vech_index(p, r, r)` of Js.
  auto sub_dsigma_rr = [&](Eigen::Index k_out, std::size_t b, Eigen::Index r,
                            double scalar) {
    const Eigen::Index p = static_cast<Eigen::Index>(am.blocks[b].Lambda.rows());
    const Eigen::Index row = vech_off[b] + vech_index(p, r, r);
    J.row(k_out) -= scalar * Js.row(row);
  };

  auto latent_all_value =
      [&](const Eigen::VectorXd& theta,
          const model::ParamLocation& L) -> post_expected<double> {
    auto a_or = ev.assembled(theta);
    if (!a_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "standardize_all: ev.assembled failed while standardizing latent "
          "parameter: " +
              a_or.error().detail));
    }
    auto sm_theta_or = ev.sigma(theta);
    if (!sm_theta_or.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "standardize_all: ev.sigma failed while standardizing latent "
          "parameter: " + sm_theta_or.error().detail));
    }
    const auto& bmat = a_or->blocks[static_cast<std::size_t>(L.block)];
    const auto& sigma = sm_theta_or->sigma[static_cast<std::size_t>(L.block)];
    auto latent_var = [&](Eigen::Index j) -> post_expected<double> {
      const double v = bmat.Mid(j, j);
      if (v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "standardize_all: non-positive latent variance; cannot standardize"));
      }
      return v;
    };
    switch (L.mat) {
      case model::MatId::Lambda: {
        auto v_or = latent_var(L.col);
        if (!v_or.has_value()) return std::unexpected(v_or.error());
        if (ordinal_delta_unit &&
            ordinal_ov[static_cast<std::size_t>(L.block)]
                      [static_cast<std::size_t>(L.row)]) {
          // DELTA scales specify the response SD directly (1/delta).
          double delta = 1.0;
          for (std::size_t i = 0; i < pt.size(); ++i) {
            if (pt.op[i] != parse::Op::ResponseScale || pt.group[i] != L.block + 1 || pt.lhs_var[i] < 0) continue;
            if (pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])] == L.row)
              delta = pt.free[i] > 0 ? theta(pt.free[i] - 1) : pt.fixed_value[i];
          }
          return delta * bmat.Lambda(L.row, L.col) * std::sqrt(*v_or);
        }
        const double sigma_rr = sigma(L.row, L.row);
        if (sigma_rr <= 0.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "standardize_all: non-positive σ_rr for Lambda"));
        }
        return bmat.Lambda(L.row, L.col) * std::sqrt(*v_or) /
               std::sqrt(sigma_rr);
      }
      case model::MatId::Psi: {
        auto vr_or = latent_var(L.row);
        auto vc_or = latent_var(L.col);
        if (!vr_or.has_value()) return std::unexpected(vr_or.error());
        if (!vc_or.has_value()) return std::unexpected(vc_or.error());
        return bmat.Psi(L.row, L.col) / std::sqrt((*vr_or) * (*vc_or));
      }
      case model::MatId::Beta: {
        auto vy_or = latent_var(L.row);
        auto vx_or = latent_var(L.col);
        if (!vy_or.has_value()) return std::unexpected(vy_or.error());
        if (!vx_or.has_value()) return std::unexpected(vx_or.error());
        // Beta is m × m: regressions among latents (~ lv on lv). Both endpoints
        // are latent, so a structural path standardizes by the predictor and
        // outcome latent SDs alike, b·SD(pred)/SD(out) — including under the
        // ordinal delta parameterization, where the unit-latent-response scaling
        // applies to measurement loadings (handled in the Lambda case), not to
        // latent→latent regressions. Mid is the m × m latent covariance, so
        // there are no observed rows in Beta to special-case here.
        return bmat.Beta(L.row, L.col) * std::sqrt(*vx_or) / std::sqrt(*vy_or);
      }
      case model::MatId::Alpha: {
        auto v_or = latent_var(L.row);
        if (!v_or.has_value()) return std::unexpected(v_or.error());
        return bmat.Alpha(L.row) / std::sqrt(*v_or);
      }
      case model::MatId::Theta:
      case model::MatId::Nu:
        return theta(static_cast<Eigen::Index>(&L - locs.data()));
    }
    return theta(static_cast<Eigen::Index>(&L - locs.data()));
  };

  auto fill_fd_row = [&](Eigen::Index ki, const model::ParamLocation& L)
      -> post_expected<void> {
    auto val_or = latent_all_value(est.theta, L);
    if (!val_or.has_value()) return std::unexpected(val_or.error());
    theta_std(ki) = *val_or;
    J.row(ki).setZero();
    for (Eigen::Index j = 0; j < n_free; ++j) {
      const double base = std::abs(est.theta(j)) + 1.0;
      const double h = std::sqrt(std::numeric_limits<double>::epsilon()) * base;
      Eigen::VectorXd tp = est.theta;
      Eigen::VectorXd tm = est.theta;
      tp(j) += h;
      tm(j) -= h;
      auto vp = latent_all_value(tp, L);
      auto vm = latent_all_value(tm, L);
      if (!vp.has_value()) return std::unexpected(vp.error());
      if (!vm.has_value()) return std::unexpected(vm.error());
      J(ki, j) = (*vp - *vm) / (2.0 * h);
    }
    return {};
  };

  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto& L = locs[k];
    if (L.row < 0 || L.block < 0) continue;
    const auto  b = static_cast<std::size_t>(L.block);
    const Eigen::Index ki = static_cast<Eigen::Index>(k);
    switch (L.mat) {
      case model::MatId::Lambda:
      case model::MatId::Psi:
      case model::MatId::Alpha:
      case model::MatId::Beta: {
        auto row_or = fill_fd_row(ki, L);
        if (!row_or.has_value()) return std::unexpected(row_or.error());
        break;
      }
      case model::MatId::Theta:
        if (L.row == L.col) {
          const double sigma_rr = sm.sigma[b](L.row, L.row);
          if (sigma_rr <= 0.0) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "standardize_all: non-positive σ_rr for Θ_rr"));
          }
          const double th = est.theta(ki);
          theta_std(ki) = th / sigma_rr;
          J.row(ki).setZero();
          J(ki, ki) = 1.0 / sigma_rr;
          sub_dsigma_rr(ki, b, L.row, th / (sigma_rr * sigma_rr));
        }
        // Θ off-diagonals: identity (preserved from Identity init).
        break;
      case model::MatId::Nu: {
        const double sigma_rr = sm.sigma[b](L.row, L.row);
        if (sigma_rr <= 0.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "standardize_all: non-positive σ_rr for ν_r"));
        }
        const double ss = std::sqrt(sigma_rr);
        const double nu = est.theta(ki);
        theta_std(ki) = nu / ss;
        J.row(ki).setZero();
        J(ki, ki) = 1.0 / ss;
        sub_dsigma_rr(ki, b, L.row, nu / (2.0 * sigma_rr * ss));
        break;
      }
    }
  }

  StandardizedSolution out;
  out.theta = std::move(theta_std);
  const Eigen::MatrixXd JV = J * vcov;
  out.se.resize(n_free);
  for (Eigen::Index k = 0; k < n_free; ++k) {
    const double var = JV.row(k).dot(J.row(k));
    out.se(k) = (var > 0.0) ? std::sqrt(var) : 0.0;
  }
  return out;
}

namespace {

bool is_composite_var(const spec::LatentStructure& pt,
                      std::int32_t var) noexcept {
  for (const auto& c : pt.composite_blocks)
    if (c.composite_var == var) return true;
  return false;
}

bool is_user_lv(const spec::LatentStructure& pt, std::int32_t var) noexcept {
  return var >= 0 && var < pt.n_vars &&
         pt.is_user_latent[static_cast<std::size_t>(var)] != 0;
}

bool is_latent_like(const spec::LatentStructure& pt,
                    std::int32_t var) noexcept {
  return is_user_lv(pt, var) || is_composite_var(pt, var);
}

std::int32_t ov_pos(const spec::LatentStructure& pt,
                    std::int32_t var) noexcept {
  if (var < 0 || var >= pt.n_vars) return -1;
  return pt.ov_pos[static_cast<std::size_t>(var)];
}

std::int32_t lv_pos(const spec::LatentStructure& pt,
                    std::int32_t var) noexcept {
  if (var < 0 || var >= pt.n_vars) return -1;
  return pt.lv_ext_pos[static_cast<std::size_t>(var)];
}

post_expected<double>
positive_sd(double variance, std::string detail) {
  if (!std::isfinite(variance) || variance <= 0.0) {
    return std::unexpected(
        make_err(PostError::Kind::NumericIssue, std::move(detail)));
  }
  return std::sqrt(variance);
}

post_expected<double>
row_value_fcsem(const spec::LatentStructure& pt, std::size_t row,
                Eigen::Ref<const Eigen::VectorXd> theta) {
  const std::int32_t free = pt.free[row];
  if (free > 0) {
    const Eigen::Index k = static_cast<Eigen::Index>(free - 1);
    if (k < 0 || k >= theta.size()) {
      return std::unexpected(make_err(
          PostError::Kind::NumericIssue,
          "native FC-SEM standardization row has an out-of-range free index"));
    }
    return theta(k);
  }
  const double v = pt.fixed_value[row];
  if (!std::isfinite(v)) {
    return std::unexpected(make_err(
        PostError::Kind::NumericIssue,
        "native FC-SEM standardization row has no finite fixed value"));
  }
  return v;
}

enum class FcSemStdKind : std::uint8_t { Lv, All };

struct FcSemScales {
  std::vector<Eigen::MatrixXd> sigma;
  std::vector<Eigen::MatrixXd> constructs;
};

post_expected<FcSemScales>
fcsem_scales(const model::FcSemEvaluator& ev, const SampleStats& samp,
             Eigen::Ref<const Eigen::VectorXd> theta) {
  auto sm_or = ev.sigma(samp, theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "native FC-SEM standardization: sigma failed: " +
            sm_or.error().detail));
  }
  auto vc_or = ev.construct_covariance(samp, theta);
  if (!vc_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "native FC-SEM standardization: construct covariance failed: " +
            vc_or.error().detail));
  }
  return FcSemScales{std::move(sm_or->sigma), std::move(*vc_or)};
}

post_expected<double>
construct_sd_fcsem(const spec::LatentStructure& pt, const FcSemScales& sc,
                   std::size_t block, std::int32_t var) {
  if (block >= sc.constructs.size()) {
    return std::unexpected(make_err(
        PostError::Kind::NumericIssue,
        "native FC-SEM standardization block is outside construct covariance"));
  }
  const std::int32_t pos = lv_pos(pt, var);
  if (pos < 0) {
    return std::unexpected(make_err(
        PostError::Kind::NumericIssue,
        "native FC-SEM standardization expected a construct endpoint"));
  }
  const Eigen::Index i = static_cast<Eigen::Index>(pos);
  const auto& V = sc.constructs[block];
  if (i >= V.rows() || i >= V.cols()) {
    return std::unexpected(make_err(
        PostError::Kind::NumericIssue,
        "native FC-SEM construct endpoint is outside covariance dimensions"));
  }
  return positive_sd(V(i, i),
                     "native FC-SEM construct variance is not positive");
}

post_expected<double>
observed_sd_fcsem(const spec::LatentStructure& pt, const FcSemScales& sc,
                  std::size_t block, std::int32_t var) {
  if (block >= sc.sigma.size()) {
    return std::unexpected(make_err(
        PostError::Kind::NumericIssue,
        "native FC-SEM standardization block is outside implied covariance"));
  }
  const std::int32_t pos = ov_pos(pt, var);
  if (pos < 0) {
    return std::unexpected(make_err(
        PostError::Kind::NumericIssue,
        "native FC-SEM standardization expected an observed endpoint"));
  }
  const Eigen::Index i = static_cast<Eigen::Index>(pos);
  const auto& S = sc.sigma[block];
  if (i >= S.rows() || i >= S.cols()) {
    return std::unexpected(make_err(
        PostError::Kind::NumericIssue,
        "native FC-SEM observed endpoint is outside covariance dimensions"));
  }
  return positive_sd(S(i, i),
                     "native FC-SEM observed variance is not positive");
}

post_expected<double>
fcsem_endpoint_scale(const spec::LatentStructure& pt, const FcSemScales& sc,
                     std::size_t block, std::int32_t var,
                     FcSemStdKind kind) {
  if (kind == FcSemStdKind::Lv) {
    if (!is_latent_like(pt, var)) return 1.0;
    return construct_sd_fcsem(pt, sc, block, var);
  }
  if (is_latent_like(pt, var)) {
    return construct_sd_fcsem(pt, sc, block, var);
  }
  return observed_sd_fcsem(pt, sc, block, var);
}

post_expected<double>
fcsem_standardized_row_value(const spec::LatentStructure& pt,
                             const model::FcSemEvaluator& ev,
                             const SampleStats& samp,
                             Eigen::Ref<const Eigen::VectorXd> theta,
                             std::size_t row, FcSemStdKind kind) {
  auto raw_or = row_value_fcsem(pt, row, theta);
  if (!raw_or.has_value()) return std::unexpected(raw_or.error());

  const std::int32_t group = pt.group[row];
  if (group <= 0) {
    return *raw_or;
  }
  const std::size_t block = static_cast<std::size_t>(group - 1);
  auto sc_or = fcsem_scales(ev, samp, theta);
  if (!sc_or.has_value()) return std::unexpected(sc_or.error());
  const auto& sc = *sc_or;

  const std::int32_t lhs = pt.lhs_var[row];
  const std::int32_t rhs = pt.rhs_var[row];

  switch (pt.op[row]) {
    case parse::Op::Composite: {
      auto lhs_sd = construct_sd_fcsem(pt, sc, block, lhs);
      if (!lhs_sd.has_value()) return std::unexpected(lhs_sd.error());
      double value = *raw_or / *lhs_sd;
      if (kind == FcSemStdKind::All) {
        auto rhs_sd = observed_sd_fcsem(pt, sc, block, rhs);
        if (!rhs_sd.has_value()) return std::unexpected(rhs_sd.error());
        value *= *rhs_sd;
      }
      return value;
    }
    case parse::Op::Measurement: {
      if (is_latent_like(pt, rhs)) {
        auto pred = fcsem_endpoint_scale(pt, sc, block, lhs, kind);
        auto out = fcsem_endpoint_scale(pt, sc, block, rhs, kind);
        if (!pred.has_value()) return std::unexpected(pred.error());
        if (!out.has_value()) return std::unexpected(out.error());
        return (*raw_or) * (*pred) / (*out);
      }
      auto lhs_sd = construct_sd_fcsem(pt, sc, block, lhs);
      if (!lhs_sd.has_value()) return std::unexpected(lhs_sd.error());
      double value = (*raw_or) * (*lhs_sd);
      if (kind == FcSemStdKind::All) {
        auto rhs_sd = observed_sd_fcsem(pt, sc, block, rhs);
        if (!rhs_sd.has_value()) return std::unexpected(rhs_sd.error());
        value /= *rhs_sd;
      }
      return value;
    }
    case parse::Op::Regression: {
      auto pred = fcsem_endpoint_scale(pt, sc, block, rhs, kind);
      auto out = fcsem_endpoint_scale(pt, sc, block, lhs, kind);
      if (!pred.has_value()) return std::unexpected(pred.error());
      if (!out.has_value()) return std::unexpected(out.error());
      return (*raw_or) * (*pred) / (*out);
    }
    case parse::Op::Covariance: {
      auto lscale = fcsem_endpoint_scale(pt, sc, block, lhs, kind);
      auto rscale = fcsem_endpoint_scale(pt, sc, block, rhs, kind);
      if (!lscale.has_value()) return std::unexpected(lscale.error());
      if (!rscale.has_value()) return std::unexpected(rscale.error());
      return (*raw_or) / ((*lscale) * (*rscale));
    }
    case parse::Op::Intercept:
    case parse::Op::Threshold:
    case parse::Op::ResponseScale:
    case parse::Op::AuxiliaryParam:
    case parse::Op::DefineParam:
    case parse::Op::EqConstraint:
    case parse::Op::LtConstraint:
    case parse::Op::GtConstraint:
      return *raw_or;
  }
  return *raw_or;
}

post_expected<StandardizedSolution>
standardize_fcsem_impl(const spec::LatentStructure& pt,
                       const SampleStats& samp,
                       const Estimates& est,
                       const Eigen::MatrixXd& vcov,
                       FcSemStdKind kind,
                       const char* label) {
  const Eigen::Index n_free = est.theta.size();
  if (vcov.rows() != n_free || vcov.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(label) +
            ": vcov shape doesn't match Estimates.theta size"));
  }
  if (pt.composite_mode != spec::CompositeMode::FcSem) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(label) + ": LatentStructure is not a native FC-SEM model"));
  }

  auto ev_or = model::FcSemEvaluator::build(pt);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(label) + ": FcSemEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<Eigen::Index>(ev.n_free()) != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(label) +
            ": evaluator free-parameter count doesn't match Estimates.theta"));
  }

  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n_free, n_free);
  Eigen::VectorXd theta_std = est.theta;

  auto fill_fd_row = [&](Eigen::Index ki,
                         std::size_t row) -> post_expected<void> {
    auto val_or =
        fcsem_standardized_row_value(pt, ev, samp, est.theta, row, kind);
    if (!val_or.has_value()) return std::unexpected(val_or.error());
    theta_std(ki) = *val_or;
    J.row(ki).setZero();
    for (Eigen::Index j = 0; j < n_free; ++j) {
      const double h = 1e-6 * std::max(1.0, std::abs(est.theta(j)));
      Eigen::VectorXd tp = est.theta;
      Eigen::VectorXd tm = est.theta;
      tp(j) += h;
      tm(j) -= h;
      auto vp = fcsem_standardized_row_value(pt, ev, samp, tp, row, kind);
      auto vm = fcsem_standardized_row_value(pt, ev, samp, tm, row, kind);
      if (!vp.has_value()) return std::unexpected(vp.error());
      if (!vm.has_value()) return std::unexpected(vm.error());
      J(ki, j) = (*vp - *vm) / (2.0 * h);
    }
    return {};
  };

  for (std::size_t row = 0; row < pt.size(); ++row) {
    if (pt.free[row] <= 0) continue;
    if (pt.group[row] <= 0 || pt.is_constraint_row(row)) continue;
    const Eigen::Index ki = static_cast<Eigen::Index>(pt.free[row] - 1);
    if (ki < 0 || ki >= n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(label) +
              ": free parameter index is outside Estimates.theta"));
    }
    auto row_or = fill_fd_row(ki, row);
    if (!row_or.has_value()) return std::unexpected(row_or.error());
  }

  StandardizedSolution out;
  out.theta = std::move(theta_std);
  const Eigen::MatrixXd JV = J * vcov;
  out.se.resize(n_free);
  for (Eigen::Index k = 0; k < n_free; ++k) {
    const double var = JV.row(k).dot(J.row(k));
    out.se(k) = (var > 0.0) ? std::sqrt(var) : 0.0;
  }
  return out;
}

}  // namespace

post_expected<StandardizedSolution>
standardize_lv_fcsem(const spec::LatentStructure& pt,
                     const SampleStats& samp,
                     const Estimates& est,
                     const Eigen::MatrixXd& vcov) {
  return standardize_fcsem_impl(pt, samp, est, vcov, FcSemStdKind::Lv,
                                "standardize_lv_fcsem");
}

post_expected<StandardizedSolution>
standardize_all_fcsem(const spec::LatentStructure& pt,
                      const SampleStats& samp,
                      const Estimates& est,
                      const Eigen::MatrixXd& vcov) {
  return standardize_fcsem_impl(pt, samp, est, vcov, FcSemStdKind::All,
                                "standardize_all_fcsem");
}

namespace {

bool fcsem_reported_op(parse::Op op) noexcept {
  return op == parse::Op::Composite || op == parse::Op::Measurement ||
         op == parse::Op::Regression;
}

post_expected<double>
fcsem_standardized_row_se(const spec::LatentStructure& pt,
                          const model::FcSemEvaluator& ev,
                          const SampleStats& samp,
                          const Estimates& est,
                          const Eigen::MatrixXd& vcov,
                          std::size_t row,
                          FcSemStdKind kind) {
  const Eigen::Index n_free = est.theta.size();
  Eigen::RowVectorXd grad = Eigen::RowVectorXd::Zero(n_free);
  for (Eigen::Index j = 0; j < n_free; ++j) {
    const double h = 1e-6 * std::max(1.0, std::abs(est.theta(j)));
    Eigen::VectorXd tp = est.theta;
    Eigen::VectorXd tm = est.theta;
    tp(j) += h;
    tm(j) -= h;
    auto vp = fcsem_standardized_row_value(pt, ev, samp, tp, row, kind);
    auto vm = fcsem_standardized_row_value(pt, ev, samp, tm, row, kind);
    if (!vp.has_value()) return std::unexpected(vp.error());
    if (!vm.has_value()) return std::unexpected(vm.error());
    grad(j) = (*vp - *vm) / (2.0 * h);
  }
  const double var = (grad * vcov * grad.transpose())(0, 0);
  return (var > 0.0) ? std::sqrt(var) : 0.0;
}

}  // namespace

post_expected<std::vector<FcSemStandardizedRow>>
standardized_rows_fcsem(const spec::LatentStructure& pt,
                        const spec::LatentNames& names,
                        const SampleStats& samp,
                        const Estimates& est,
                        const Eigen::MatrixXd& vcov) {
  const Eigen::Index n_free = est.theta.size();
  if (vcov.rows() != n_free || vcov.cols() != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardized_rows_fcsem: vcov shape doesn't match Estimates.theta "
        "size"));
  }
  if (pt.composite_mode != spec::CompositeMode::FcSem) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardized_rows_fcsem: LatentStructure is not a native FC-SEM "
        "model"));
  }
  if (names.row_lhs.size() != pt.size() || names.row_rhs.size() != pt.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardized_rows_fcsem: LatentNames row columns do not match "
        "LatentStructure"));
  }

  auto ev_or = model::FcSemEvaluator::build(pt);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardized_rows_fcsem: FcSemEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<Eigen::Index>(ev.n_free()) != n_free) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "standardized_rows_fcsem: evaluator free-parameter count doesn't "
        "match Estimates.theta"));
  }

  std::vector<FcSemStandardizedRow> out;
  for (std::size_t row = 0; row < pt.size(); ++row) {
    if (pt.group[row] <= 0 || pt.is_constraint_row(row)) continue;
    if (!fcsem_reported_op(pt.op[row])) continue;

    auto est_or = row_value_fcsem(pt, row, est.theta);
    if (!est_or.has_value()) return std::unexpected(est_or.error());
    auto slv_or =
        fcsem_standardized_row_value(pt, ev, samp, est.theta, row,
                                     FcSemStdKind::Lv);
    if (!slv_or.has_value()) return std::unexpected(slv_or.error());
    auto sall_or =
        fcsem_standardized_row_value(pt, ev, samp, est.theta, row,
                                     FcSemStdKind::All);
    if (!sall_or.has_value()) return std::unexpected(sall_or.error());
    auto slv_se_or = fcsem_standardized_row_se(
        pt, ev, samp, est, vcov, row, FcSemStdKind::Lv);
    if (!slv_se_or.has_value()) return std::unexpected(slv_se_or.error());
    auto sall_se_or = fcsem_standardized_row_se(
        pt, ev, samp, est, vcov, row, FcSemStdKind::All);
    if (!sall_se_or.has_value()) return std::unexpected(sall_se_or.error());

    FcSemStandardizedRow r;
    r.row = row;
    r.lhs = names.row_lhs[row];
    r.op = pt.op[row];
    r.rhs = names.row_rhs[row];
    r.group = pt.group[row];
    r.free = pt.free[row];
    r.est = *est_or;
    if (pt.free[row] > 0) {
      const Eigen::Index k = static_cast<Eigen::Index>(pt.free[row] - 1);
      if (k < 0 || k >= n_free) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "standardized_rows_fcsem: free parameter index is outside "
            "Estimates.theta"));
      }
      const double var = vcov(k, k);
      r.se = (var > 0.0) ? std::sqrt(var) : 0.0;
    }
    r.std_lv = *slv_or;
    r.std_lv_se = *slv_se_or;
    r.std_all = *sall_or;
    r.std_all_se = *sall_se_or;
    out.push_back(std::move(r));
  }
  return out;
}

}  // namespace magmaan::measures::standardize
