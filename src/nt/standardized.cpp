#include "magmaan/fit/standardized.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/model_evaluator.hpp"

#include "detail_vech.hpp"

namespace magmaan::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

post_expected<StandardizedSolution>
standardize_lv(const partable::LatentStructure& pt,
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
standardize_all(const partable::LatentStructure& pt,
                const model::MatrixRep&   rep,
                const Estimates&          est,
                const Eigen::MatrixXd&    vcov) {
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

}  // namespace magmaan::fit
