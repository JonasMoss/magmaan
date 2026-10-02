#include "ordinal_internal.hpp"

namespace magmaan::estimate {

using namespace detail_ordinal;

namespace detail_ordinal {
Eigen::Index ordinal_sandwich_block_rows(const data::OrdinalStats& stats,
                                         std::size_t b) {
  const Eigen::Index p = stats.R[b].rows();
  return static_cast<Eigen::Index>(stats.thresholds[b].size()) +
         p * (p - 1) / 2;
}

Eigen::Index ordinal_sandwich_block_rows(const data::MixedOrdinalStats& stats,
                                         std::size_t b) {
  return stats.moments[b].size();
}

post_expected<std::vector<Eigen::MatrixXd>>
ordinal_sandwich_weights(const data::OrdinalStats& stats,
                         OrdinalWeightKind kind) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(stats.NACOV.size());
  if (kind == OrdinalWeightKind::ULS) {
    for (const auto& G : stats.NACOV) {
      out.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
    }
    return out;
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    if (Ws[b].size() == 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust score tests: weight matrix unavailable in block " +
              std::to_string(b)));
    }
    out.push_back(Ws[b]);
  }
  return out;
}

post_expected<std::vector<Eigen::MatrixXd>>
ordinal_sandwich_weights(const data::MixedOrdinalStats& stats,
                         OrdinalWeightKind kind) {
  if (kind == OrdinalWeightKind::ULS) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal robust score tests support DWLS/WLS weights only"));
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<Eigen::MatrixXd> out;
  out.reserve(Ws.size());
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    if (Ws[b].size() == 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed ordinal robust score tests: weight matrix unavailable in "
          "block " + std::to_string(b)));
    }
    out.push_back(Ws[b]);
  }
  return out;
}



data::SampleStats sample_stats_for_starts(const data::MixedOrdinalStats& stats) {
  data::SampleStats samp;
  samp.S = stats.R;
  samp.mean = stats.mean;
  samp.n_obs = stats.n_obs;
  return samp;
}


data::SampleStats sample_stats_for_starts(const data::MixedOrdinalMoments& moments) {
  data::SampleStats samp;
  samp.S = moments.R;
  samp.mean = moments.mean;
  samp.n_obs = moments.n_obs;
  return samp;
}


void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::OrdinalStats& stats) {
  for (std::size_t b = 0; b < stats.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = stats.thresholds[b](k);
    }
  }
}


void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::MixedOrdinalStats& stats) {
  for (std::size_t b = 0; b < stats.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = stats.thresholds[b](k);
    }
  }
}


void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::MixedOrdinalMoments& moments) {
  for (std::size_t b = 0; b < moments.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < moments.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = moments.thresholds[b](k);
    }
  }
}


OrdinalRobustResult ordinal_result_from_weighted(const WeightedRobustResult& r) {
  OrdinalRobustResult out;
  out.vcov = r.vcov;
  out.se = r.se;
  out.eigvals = r.eigvals;
  out.chisq_standard = r.chisq_standard;
  out.df = r.df;
  out.satorra_bentler = r.satorra_bentler;
  out.mean_var_adjusted = r.mean_var_adjusted;
  out.scaled_shifted = r.scaled_shifted;
  return out;
}


// Per-block missing-pattern flags for the estimated-weight ordinal IJ, plus the
// int_data validation `robust_ordinal_ij` requires (0-based category codes;
// missing entries only with pairwise-overlap Gamma). ULS carries no estimated
// weight, so it returns all-false without touching int_data.
post_expected<std::vector<bool>>
ordinal_ij_block_missing(const data::OrdinalStats& stats,
                         OrdinalWeightKind weights) {
  std::vector<bool> block_has_missing(stats.R.size(), false);
  if (weights == OrdinalWeightKind::ULS) return block_has_missing;
  if (stats.int_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: integer data unavailable; recompute ordinal "
        "stats with int_data to include the estimated-weight influence"));
  }
  const bool allow_pairwise_missing = stats.pairwise_gamma == "overlap";
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::MatrixXi& Xcat = stats.int_data[b];
    if (stats.n_levels[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal_ij: n_levels length mismatch in block " +
              std::to_string(b)));
    }
    if (Xcat.rows() != stats.n_obs[b] || Xcat.cols() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal_ij: int_data shape mismatch in block " +
              std::to_string(b)));
    }
    for (Eigen::Index r = 0; r < Xcat.rows(); ++r) {
      for (Eigen::Index j = 0; j < Xcat.cols(); ++j) {
        const int c = Xcat(r, j);
        if (c < 0) {
          if (!allow_pairwise_missing) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "robust_ordinal_ij: missing ordinal int_data requires "
                "pairwise overlap Gamma in block " + std::to_string(b)));
          }
          block_has_missing[b] = true;
          continue;
        }
        const int max_level = stats.n_levels[b][static_cast<std::size_t>(j)];
        if (c >= max_level) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_ordinal_ij: int_data must contain 0-based category "
              "codes in block " + std::to_string(b)));
        }
      }
    }
  }
  return block_has_missing;
}


// The estimated-weight channels are derivatives of diag(NACOV)^-1 (DWLS) and
// NACOV^-1 (WLS). Ordinal NT, DLS and caller-supplied weights occupy the same
// slots under the computational DWLS/WLS labels, but their data influence
// differs: NT depends on the polychoric correlations, DLS inverts a mixture,
// and a supplied weight has no known influence. Applying the NACOV channel to
// them would describe the wrong estimator, so refuse. The reference inverse is
// the stats builder's own routine, so a default weight compares exactly; the
// tolerance only absorbs rounding from other builders of the same weight.
post_expected<void>
require_nacov_weight(const std::vector<Eigen::MatrixXd>& Ws,
                     const std::vector<Eigen::MatrixXd>& NACOV,
                     OrdinalWeightKind weights, const char* who) {
  if (weights == OrdinalWeightKind::ULS) return {};
  constexpr double kRelativeTolerance = 1e-6;
  if (Ws.size() != NACOV.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(who) + ": weight and NACOV block counts differ"));
  }
  for (std::size_t b = 0; b < NACOV.size(); ++b) {
    const Eigen::MatrixXd& G = NACOV[b];
    Eigen::MatrixXd expected;
    if (weights == OrdinalWeightKind::DWLS) {
      expected = Eigen::MatrixXd::Zero(G.rows(), G.cols());
      for (Eigen::Index k = 0; k < G.rows(); ++k) {
        if (!(G(k, k) > 0.0) || !std::isfinite(G(k, k))) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              std::string(who) + ": NACOV diagonal is not positive in block " +
                  std::to_string(b)));
        }
        expected(k, k) = 1.0 / G(k, k);
      }
    } else {
      ::magmaan::detail::SymInverseResult inverse =
          ::magmaan::detail::symmetric_inverse_pd_gated(G);
      if (!inverse.ok) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            std::string(who) + ": NACOV is not positive definite in block " +
                std::to_string(b)));
      }
      expected = std::move(inverse.inverse);
    }
    if (Ws[b].rows() != expected.rows() || Ws[b].cols() != expected.cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": weight dimension mismatch in block " +
              std::to_string(b)));
    }
    const double scale =
        std::max(expected.norm(), std::numeric_limits<double>::min());
    const double gap = (Ws[b] - expected).norm() / scale;
    if (!(gap <= kRelativeTolerance)) {
      return std::unexpected(make_post_err(PostError::Kind::UnsupportedInference,
          std::string(who) + ": estimated-weight inference needs the " +
              (weights == OrdinalWeightKind::DWLS ? "DWLS weight diag(NACOV)^-1"
                                                  : "WLS weight NACOV^-1") +
              ", but block " + std::to_string(b) +
              " holds a different weight (normal-theory, DLS or supplied; "
              "relative difference " + std::to_string(gap) +
              "). Its data influence is not implemented; use fixed-weight "
              "inference"));
    }
  }
  return {};
}


// Per-case IJ blocks (Δ_b, W_b, moment_influence, IF(Ŵ) correction) for an
// all-ordinal DWLS/WLS fit, evaluated at `theta`/`moments` (the fitted point for
// the SE path, the freed-candidate null for the score-test sandwich). Shared by
// `robust_ordinal_ij` and `ordinal_param_space_sandwich_ij` so the SE and MI
// paths build identical meat. The DWLS/WLS weight-influence channels are
// recomputed per call (theta-independent but inexpensive enough for the v1
// frontier sweep).
post_expected<std::vector<WeightedMomentIJBlock>>
build_ordinal_ij_blocks(const data::OrdinalStats& stats,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const Eigen::VectorXd& theta,
                        const std::vector<Eigen::MatrixXd>& Ws,
                        const Eigen::MatrixXd& Delta_full,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization,
                        const std::vector<bool>& block_has_missing) {
  if (stats.moment_influence.size() != stats.R.size() ||
      stats.NACOV.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: per-case influence functions unavailable; recompute "
        "ordinal stats (moment_influence is required for the IJ)"));
  }
  if (auto e = require_nacov_weight(Ws, stats.NACOV, weights,
                                    "ordinal estimated-weight inference");
      !e.has_value()) {
    return std::unexpected(e.error());
  }
  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    const Eigen::MatrixXd& G = stats.moment_influence[b];   // n_b × mb
    if (G.cols() != mb || G.rows() != stats.n_obs[b]) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal_ij: moment_influence shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd d_b = ordinal_block_residual(
        stats, layout, moments, theta, parameterization, b);
    Eigen::MatrixXd correction;
    if (weights == OrdinalWeightKind::DWLS) {
      // The IF of the estimated weight Ŵ=diag(Γ̂)⁻¹ enters as corr_{i,k} =
      // d_k·IF_{i,k}(Γ̂)/Γ̂_kk², with IF(Γ̂) = [data-direct sandwich influence at
      // fixed κ] + [κ-movement Σ_l(∂Γ̂_kk/∂κ_l)g_{i,l}]. Both need the integer
      // data; pairwise-overlap MCAR blocks use the observed-support helpers.
      Eigen::MatrixXd IFG;  // n_b × mb: data-direct IF of Γ̂_kk (V̂ + Â variation)
      Eigen::MatrixXd GD;   // n_b × mb: κ-movement IF of Γ̂_kk (FD of Γ̂ over κ)
      auto inf_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_diag_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_diag_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!inf_or.has_value()) return std::unexpected(inf_or.error());
      IFG = std::move(*inf_or);
      auto D_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_diag_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_diag_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!D_or.has_value()) return std::unexpected(D_or.error());
      GD.noalias() = G * D_or->transpose();
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index k = 0; k < mb; ++k) {
        const double gkk = stats.NACOV[b](k, k);
        if (!(gkk > 0.0)) continue;
        const Eigen::VectorXd if_k = (IFG.col(k) + GD.col(k)).eval();
        correction.col(k) = (d_b(k) / (gkk * gkk)) * if_k;
      }
    } else if (weights == OrdinalWeightKind::WLS) {
      // Full-WLS analogue: IF(Ŵ_i) = -W IF_i(Γ̂) W, so the row correction added
      // to g_i W is d' W IF_i(Γ̂) W. `IF_i(Γ̂)` combines the data-direct
      // sandwich channel with the κ-movement channel DΓ/Dκ · IF_i(κ).
      auto inf_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!inf_or.has_value()) return std::unexpected(inf_or.error());
      auto D_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!D_or.has_value()) return std::unexpected(D_or.error());
      if (inf_or->rows() != G.rows() || inf_or->cols() != mb * mb ||
          D_or->rows() != mb * mb || D_or->cols() != mb) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "robust_ordinal_ij: full Gamma influence shape mismatch in block " +
                std::to_string(b)));
      }
      const Eigen::RowVectorXd lhs = d_b.transpose() * Ws[b];
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index i = 0; i < G.rows(); ++i) {
        Eigen::VectorXd if_vec = inf_or->row(i).transpose();
        if_vec.noalias() += (*D_or) * G.row(i).transpose();
        Eigen::Map<const Eigen::MatrixXd> IFGamma(if_vec.data(), mb, mb);
        correction.row(i) = lhs * IFGamma * Ws[b];
      }
    }
    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .moment_influence = G,
        .weight_correction = std::move(correction),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }
  return ij_blocks;
}


// Estimated-weight ("complete-sandwich") parameter-space sandwich {A1, B1} for
// an all-ordinal DWLS/WLS fit: the IJ counterpart of
// `ordinal_param_space_sandwich`, carrying the IF(Ŵ) meat term. Used by the
// estimated-weight robust modification-index / score-test path.
post_expected<robust::ParamSpaceSandwich>
ordinal_param_space_sandwich_ij(const data::OrdinalStats& stats,
                                const ThresholdLayout& layout,
                                const model::ImpliedMoments& moments,
                                const Eigen::VectorXd& theta,
                                const std::vector<Eigen::MatrixXd>& Ws,
                                const Eigen::MatrixXd& Delta_full,
                                OrdinalWeightKind weights,
                                OrdinalParameterization parameterization,
                                const std::vector<bool>& block_has_missing) {
  auto blocks = build_ordinal_ij_blocks(stats, layout, moments, theta, Ws,
                                        Delta_full, weights, parameterization,
                                        block_has_missing);
  if (!blocks.has_value()) return std::unexpected(blocks.error());
  if (Delta_full.rows() != [&] {
        Eigen::Index s = 0;
        for (const auto& blk : *blocks) s += blk.jacobian.rows();
        return s;
      }()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal estimated-weight score tests: moment Jacobian row count does "
        "not match the block layout"));
  }
  return weighted_param_space_sandwich_ij(*blocks);
}

}  // namespace detail_ordinal

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalStats& stats,
               const Estimates& est,
               OrdinalWeightKind weights,
               OrdinalParameterization parameterization,
               robust::Information bread,
               const std::vector<std::int8_t>* row_user) {
  if (est.association) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal association ML requires its own Stage-1 sampling/inference contract"));
  }
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "OrdinalStats NACOV block count does not match MatrixRep"));
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr, row_user);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: fitted theta length does not match ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization, eval->J_mu);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: constraint reparameterization has incompatible shape"));
  }

  // ULS: the model weight is the identity (the ULSMV sandwich uses NACOV
  // directly). Build it once so `Ws[b]` references a live matrix through the
  // robust_weighted_moments() call below.
  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);
  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    if (stats.NACOV[b].rows() != mb || stats.NACOV[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "OrdinalStats NACOV dimension mismatch in block " + std::to_string(b)));
    }
    blocks.push_back(WeightedMomentBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .gamma = stats.NACOV[b],
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  std::optional<Eigen::MatrixXd> bread_override;
  if (bread == robust::Information::Observed) {
    auto ob = ordinal_observed_bread_analytic(
        pt, rep, stats, est, *layout_or, Ws, K, parameterization);
    if (!ob.has_value()) return std::unexpected(ob.error());
    bread_override = std::move(*ob);
  }

  // 2·est.fmin = F (est.fmin = ½F): robust_weighted_moments forms N·F.
  auto out = robust_weighted_moments(blocks, K, 2.0 * est.fmin, bread_override);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<OrdinalRobustResult>
robust_ordinal_ij(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalStats& stats,
                  const Estimates& est,
                  OrdinalWeightKind weights,
                  OrdinalParameterization parameterization,
                  const std::vector<std::int8_t>* row_user) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: per-case influence functions unavailable; recompute "
        "ordinal stats (moment_influence is required for the IJ)"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(missing_or.error());
  const std::vector<bool> block_has_missing = std::move(*missing_or);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr, row_user); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: fitted theta length does not match ordinal delta "
        "partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization, eval->J_mu);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: constraint reparameterization has incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  auto ij_blocks = build_ordinal_ij_blocks(
      stats, *layout_or, eval->moments, est.theta, Ws, Delta_full, weights,
      parameterization, block_has_missing);
  if (!ij_blocks.has_value()) return std::unexpected(ij_blocks.error());

  auto out = robust_weighted_moment_ij(*ij_blocks, K, 2.0 * est.fmin, A);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<WeightedMomentRBMParts>
ordinal_rbm_parts(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalStats& stats,
                  const Estimates& est,
                  OrdinalWeightKind weights,
                  OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: per-case influence functions unavailable; "
        "recompute ordinal stats (moment_influence is required for RBM)"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(missing_or.error());
  const std::vector<bool> block_has_missing = std::move(*missing_or);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: fitted theta length does not match ordinal delta "
        "partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization, eval->J_mu);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: constraint reparameterization has incompatible "
        "shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  auto ij_blocks = build_ordinal_ij_blocks(
      stats, *layout_or, eval->moments, est.theta, Ws, Delta_full, weights,
      parameterization, block_has_missing);
  if (!ij_blocks.has_value()) return std::unexpected(ij_blocks.error());

  return weighted_moment_rbm_parts(*ij_blocks, K, A);
}

post_expected<CasewiseInfluenceIJ>
ordinal_casewise_influence_ij(spec::LatentStructure pt,
                              const model::MatrixRep& rep,
                              const data::OrdinalStats& stats,
                              const Estimates& est,
                              OrdinalWeightKind weights,
                              OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: per-case influence functions "
        "unavailable; recompute ordinal stats (moment_influence is required for "
        "the IJ)"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(missing_or.error());
  const std::vector<bool> block_has_missing = std::move(*missing_or);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: fitted theta length does not match "
        "ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization, eval->J_mu);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: constraint reparameterization has "
        "incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  auto ij_blocks = build_ordinal_ij_blocks(
      stats, *layout_or, eval->moments, est.theta, Ws, Delta_full, weights,
      parameterization, block_has_missing);
  if (!ij_blocks.has_value()) return std::unexpected(ij_blocks.error());

  return casewise_influence_from_ij_blocks(*ij_blocks, K, A);
}

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalMoments& moments,
               data::OrdinalGammaCache& gamma_cache,
               const Estimates& est,
               data::OrdinalWeightPlan plan) {
  if (plan.purpose == data::OrdinalWorkspacePurpose::FitOnly) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: cache-aware inference requires a fit-plus-inference "
        "or inference-only plan"));
  }
  if (plan.materialization != data::OrdinalGammaMaterialization::Full) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: cache-aware inference currently requires full Gamma"));
  }
  OrdinalWeightKind weight_kind;
  if (plan.estimator == data::OrdinalEstimatorKind::DWLS) {
    weight_kind = OrdinalWeightKind::DWLS;
  } else if (plan.estimator == data::OrdinalEstimatorKind::WLS) {
    weight_kind = OrdinalWeightKind::WLS;
  } else {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: cache-aware inference currently supports DWLS/WLS"));
  }

  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (gamma_cache.blocks.size() != moments.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: OrdinalGammaCache block count mismatch"));
  }
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const Eigen::Index p = moments.R[b].rows();
    const Eigen::Index mdim =
        moments.thresholds[b].size() + p * (p - 1) / 2;
    auto& block = gamma_cache.blocks[b];
    if (!block.has_full || block.gamma.rows() != mdim ||
        block.gamma.cols() != mdim || !matrix_all_finite(block.gamma)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal: full Gamma missing or malformed in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < mdim; ++k) {
      if (!(block.gamma(k, k) > 0.0)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "robust_ordinal: full Gamma diagonal is not positive in block " +
                std::to_string(b)));
      }
    }
  }

  post_expected<void> weights_ok =
      weight_kind == OrdinalWeightKind::DWLS
          ? data::ordinal_gamma_cache_ensure_dwls_weights(gamma_cache)
          : data::ordinal_gamma_cache_ensure_wls_weights(gamma_cache);
  if (!weights_ok.has_value()) return std::unexpected(weights_ok.error());

  data::OrdinalStats stats = stats_adapter(moments);
  stats.NACOV.reserve(gamma_cache.blocks.size());
  if (weight_kind == OrdinalWeightKind::DWLS) {
    stats.W_dwls.reserve(gamma_cache.blocks.size());
  } else {
    stats.W_wls.reserve(gamma_cache.blocks.size());
  }
  for (const auto& block : gamma_cache.blocks) {
    stats.NACOV.push_back(block.gamma);
    if (weight_kind == OrdinalWeightKind::DWLS) {
      stats.W_dwls.push_back(block.w_dwls);
    } else {
      stats.W_wls.push_back(block.w_wls);
    }
  }

  return robust_ordinal(std::move(pt), rep, stats, est, weight_kind,
                        to_estimate_parameterization(plan.parameterization));
}

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalStats& stats,
                     const Estimates& est,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization,
                     robust::Information bread,
                     const std::vector<std::int8_t>* row_user) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr, row_user);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: fitted theta length does not match mixed delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu, est.theta,
                            parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: constraint reparameterization has incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);
  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    blocks.push_back(WeightedMomentBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .gamma = stats.NACOV[b],
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  std::optional<Eigen::MatrixXd> bread_override;
  if (bread == robust::Information::Observed) {
    auto ob = mixed_observed_bread_analytic(
        pt, rep, stats, est, *layout_or, Ws, K, parameterization);
    if (!ob.has_value()) return std::unexpected(ob.error());
    bread_override = std::move(*ob);
  }

  // 2·est.fmin = F (est.fmin = ½F): robust_weighted_moments forms N·F.
  auto out = robust_weighted_moments(blocks, K, 2.0 * est.fmin, bread_override);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<OrdinalRobustResult>
robust_mixed_ordinal_ij(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const data::MixedOrdinalStats& stats,
                        const Estimates& est,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: per-case influence functions unavailable; "
        "recompute mixed ordinal stats (moment_influence is required for the IJ)"));
  }
  const bool has_diag_gamma_if =
      stats.gamma_diag_influence.size() == stats.R.size();
  const bool has_full_gamma_if =
      stats.gamma_full_influence.size() == stats.R.size();
  if (weights == OrdinalWeightKind::DWLS && !has_diag_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: DWLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_diag_influence or raw_data"));
  }
  if (weights == OrdinalWeightKind::WLS && !has_full_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: WLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_full_influence or raw_data"));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: fitted theta length does not match mixed "
        "delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu, est.theta,
                            parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: constraint reparameterization has "
        "incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = mixed_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const Eigen::MatrixXd& G = stats.moment_influence[b];
    if (G.rows() != stats.n_obs[b] || G.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_mixed_ordinal_ij: moment_influence shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd model_m = mixed_model_moments(
        stats, *layout_or, eval->moments, est.theta, b, parameterization);
    const Eigen::VectorXd d_b = model_m - stats.moments[b];
    Eigen::MatrixXd correction;
    if (weights == OrdinalWeightKind::DWLS) {
      Eigen::MatrixXd if_gamma;
      if (has_diag_gamma_if) {
        if_gamma = stats.gamma_diag_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: precomputed Gamma diagonal influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb ||
            D_or->rows() != mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: mixed Gamma diagonal influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index k = 0; k < mb; ++k) {
        const double gkk = stats.NACOV[b](k, k);
        if (!(gkk > 0.0)) continue;
        correction.col(k) = (d_b(k) / (gkk * gkk)) * if_gamma.col(k);
      }
    } else if (weights == OrdinalWeightKind::WLS) {
      Eigen::MatrixXd if_gamma;
      if (has_full_gamma_if) {
        if_gamma = stats.gamma_full_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb * mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: precomputed Gamma full influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb * mb ||
            D_or->rows() != mb * mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: mixed full Gamma influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      const Eigen::RowVectorXd lhs = d_b.transpose() * Ws[b];
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index i = 0; i < G.rows(); ++i) {
        const Eigen::VectorXd if_vec = if_gamma.row(i).transpose();
        Eigen::Map<const Eigen::MatrixXd> IFGamma(if_vec.data(), mb, mb);
        correction.row(i) = lhs * IFGamma * Ws[b];
      }
    }
    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .moment_influence = G,
        .weight_correction = std::move(correction),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  auto out = robust_weighted_moment_ij(ij_blocks, K, 2.0 * est.fmin, A);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<WeightedMomentRBMParts>
mixed_ordinal_rbm_parts(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const data::MixedOrdinalStats& stats,
                        const Estimates& est,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: per-case influence functions unavailable; "
        "recompute mixed ordinal stats (moment_influence is required for RBM)"));
  }
  const bool has_diag_gamma_if =
      stats.gamma_diag_influence.size() == stats.R.size();
  const bool has_full_gamma_if =
      stats.gamma_full_influence.size() == stats.R.size();
  if (weights == OrdinalWeightKind::DWLS && !has_diag_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: DWLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_diag_influence or raw_data"));
  }
  if (weights == OrdinalWeightKind::WLS && !has_full_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: WLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_full_influence or raw_data"));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: fitted theta length does not match mixed "
        "delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu, est.theta,
                            parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: constraint reparameterization has "
        "incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = mixed_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const Eigen::MatrixXd& G = stats.moment_influence[b];
    if (G.rows() != stats.n_obs[b] || G.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_rbm_parts: moment_influence shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd model_m = mixed_model_moments(
        stats, *layout_or, eval->moments, est.theta, b, parameterization);
    const Eigen::VectorXd d_b = model_m - stats.moments[b];
    Eigen::MatrixXd correction;
    if (weights == OrdinalWeightKind::DWLS) {
      Eigen::MatrixXd if_gamma;
      if (has_diag_gamma_if) {
        if_gamma = stats.gamma_diag_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: precomputed Gamma diagonal influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb ||
            D_or->rows() != mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: mixed Gamma diagonal influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index k = 0; k < mb; ++k) {
        const double gkk = stats.NACOV[b](k, k);
        if (!(gkk > 0.0)) continue;
        correction.col(k) = (d_b(k) / (gkk * gkk)) * if_gamma.col(k);
      }
    } else if (weights == OrdinalWeightKind::WLS) {
      Eigen::MatrixXd if_gamma;
      if (has_full_gamma_if) {
        if_gamma = stats.gamma_full_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb * mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: precomputed Gamma full influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb * mb ||
            D_or->rows() != mb * mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: mixed full Gamma influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      const Eigen::RowVectorXd lhs = d_b.transpose() * Ws[b];
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index i = 0; i < G.rows(); ++i) {
        const Eigen::VectorXd if_vec = if_gamma.row(i).transpose();
        Eigen::Map<const Eigen::MatrixXd> IFGamma(if_vec.data(), mb, mb);
        correction.row(i) = lhs * IFGamma * Ws[b];
      }
    }
    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .moment_influence = G,
        .weight_correction = std::move(correction),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  return weighted_moment_rbm_parts(ij_blocks, K, A);
}

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalMoments& moments,
                     data::OrdinalGammaCache& gamma_cache,
                     const Estimates& est,
                     data::OrdinalWeightPlan plan) {
  if (plan.purpose == data::OrdinalWorkspacePurpose::FitOnly) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: cache-aware inference requires a "
        "fit-plus-inference or inference-only plan"));
  }
  if (plan.materialization != data::OrdinalGammaMaterialization::Full) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: cache-aware inference currently requires full "
        "Gamma"));
  }
  OrdinalWeightKind weight_kind;
  if (plan.estimator == data::OrdinalEstimatorKind::DWLS) {
    weight_kind = OrdinalWeightKind::DWLS;
  } else if (plan.estimator == data::OrdinalEstimatorKind::WLS) {
    weight_kind = OrdinalWeightKind::WLS;
  } else {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: cache-aware inference currently supports "
        "DWLS/WLS"));
  }

  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (gamma_cache.blocks.size() != moments.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: OrdinalGammaCache block count mismatch"));
  }
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const Eigen::Index mdim = moments.moments[b].size();
    auto& block = gamma_cache.blocks[b];
    if (!block.has_full || block.gamma.rows() != mdim ||
        block.gamma.cols() != mdim || !matrix_all_finite(block.gamma)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_mixed_ordinal: full Gamma missing or malformed in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < mdim; ++k) {
      if (!(block.gamma(k, k) > 0.0)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "robust_mixed_ordinal: full Gamma diagonal is not positive in "
            "block " +
                std::to_string(b)));
      }
    }
  }

  post_expected<void> weights_ok =
      weight_kind == OrdinalWeightKind::DWLS
          ? data::ordinal_gamma_cache_ensure_dwls_weights(gamma_cache)
          : data::ordinal_gamma_cache_ensure_wls_weights(gamma_cache);
  if (!weights_ok.has_value()) return std::unexpected(weights_ok.error());

  data::MixedOrdinalStats stats = stats_adapter(moments);
  stats.NACOV.reserve(gamma_cache.blocks.size());
  if (weight_kind == OrdinalWeightKind::DWLS) {
    stats.W_dwls.reserve(gamma_cache.blocks.size());
  } else {
    stats.W_wls.reserve(gamma_cache.blocks.size());
  }
  for (const auto& block : gamma_cache.blocks) {
    stats.NACOV.push_back(block.gamma);
    if (weight_kind == OrdinalWeightKind::DWLS) {
      stats.W_dwls.push_back(block.w_dwls);
    } else {
      stats.W_wls.push_back(block.w_wls);
    }
  }

  return robust_mixed_ordinal(
      std::move(pt), rep, stats, est, weight_kind,
      to_estimate_parameterization(plan.parameterization));
}


}  // namespace magmaan::estimate
