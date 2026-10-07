#include "ordinal_internal.hpp"
#include "../inference/detail_mi_candidates.hpp"

namespace magmaan::estimate {

using namespace detail_ordinal;

namespace {

template<class Stats>
post_expected<EqConstraints> ordinal_score_constraints(const spec::LatentStructure& pt,
                                                       const Eigen::VectorXd& theta) {
  if constexpr (std::is_same_v<Stats, data::OrdinalStats>) return build_eq_tangent(pt, theta);
  else return build_eq_constraints(pt);
}

post_expected<Eigen::MatrixXd> ordinal_null_space(const Eigen::MatrixXd& A,
                                                  Eigen::Index n_cols) {
  if (A.rows() == 0) return Eigen::MatrixXd::Identity(n_cols, n_cols);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  svd.setThreshold(1e-9);
  return Eigen::MatrixXd(svd.matrixV().rightCols(n_cols - svd.rank()));
}

post_expected<Eigen::VectorXd>
ordinal_release_direction(const EqConstraints& con, Eigen::Index release_row) {
  Eigen::MatrixXd A_rel(con.A_eq.rows() - 1, con.A_eq.cols());
  Eigen::Index out = 0;
  for (Eigen::Index r = 0; r < con.A_eq.rows(); ++r) {
    if (r == release_row) continue;
    A_rel.row(out++) = con.A_eq.row(r);
  }
  auto K_rel = ordinal_null_space(A_rel, con.npar);
  if (!K_rel.has_value()) return std::unexpected(K_rel.error());
  const Eigen::MatrixXd M = K_rel->transpose() * con.K();
  auto z = ordinal_null_space(M.transpose(), K_rel->cols());
  if (!z.has_value()) return std::unexpected(z.error());
  if (z->cols() != 1) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score equality release is not one-dimensional"));
  }
  Eigen::VectorXd d = (*K_rel) * z->col(0);
  const double norm = d.norm();
  if (!(norm > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score equality-release direction is degenerate"));
  }
  d /= norm;
  return d;
}

bool ordinal_fixed_candidate(const spec::LatentStructure& pt,
                             const model::MatrixRep& rep,
                             std::size_t row) {
  if (row >= pt.size()) return false;
  if (pt.is_constraint_row(row)) return false;
  if (pt.free[row] != 0) return false;
  if (row < pt.exo.size() && pt.exo[row] != 0) return false;
  if (row >= pt.fixed_value.size() || !std::isfinite(pt.fixed_value[row])) {
    return false;
  }
  return pt.op[row] == parse::Op::Threshold ||
         (row < rep.cell_for_row.size() && rep.cell_for_row[row].used);
}

using inference::detail_mi::ModificationCandidateRow;
using inference::detail_mi::enumerate_modification_candidates;
using inference::detail_mi::append_modification_candidates;

struct OrdinalModificationIndexModel {
  spec::LatentStructure pt;
  model::MatrixRep rep;
  std::size_t original_rows = 0;
};

post_expected<OrdinalModificationIndexModel>
prepare_ordinal_modification_index_model(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const inference::ModificationIndexOptions& options) {
  if (options.candidates == inference::ScoreCandidateSet::FixedRowsOnly) {
    const std::size_t original_rows = pt.size();
    return OrdinalModificationIndexModel{std::move(pt), rep, original_rows};
  }

  const std::size_t original_rows = pt.size();
  const std::vector<ModificationCandidateRow> absent =
      enumerate_modification_candidates(pt, options);
  pt = append_modification_candidates(std::move(pt), absent);
  auto mr = model::build_matrix_rep(pt);
  if (!mr.has_value()) return std::unexpected(model_to_post(mr.error()));
  return OrdinalModificationIndexModel{std::move(pt), std::move(*mr),
                                       original_rows};
}

void ordinal_add_free_group(spec::LatentStructure& pt, std::int32_t old_n) {
  if (static_cast<std::int32_t>(pt.eq_groups.size()) == old_n) {
    pt.eq_groups.push_back(old_n);
  } else if (!pt.eq_groups.empty()) {
    pt.eq_groups.clear();
  }
}

// Association-ML estimates minimise another discrepancy than the LS moment
// quadratic these score tests differentiate, and their score-test contract is
// not derived; refuse rather than report LS statistics at ML estimates.
post_expected<void> require_ls_ordinal_estimates(const Estimates& est) {
  if (!est.association) return {};
  return std::unexpected(make_post_err(PostError::Kind::UnsupportedInference,
      "ordinal association ML has no modification-index or release-test "
      "contract"));
}

template <class Stats, class ResidualFn, class JacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_modification_indices_impl(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const Stats& stats,
                                  const Estimates& est,
                                  OrdinalWeightKind weights,
                                  const inference::ModificationIndexOptions& options,
                                  ResidualFn residual_fn,
                                  JacobianFn jacobian_fn,
                                  PrepareFn prepare_fn) {
  if (auto ok = require_ls_ordinal_estimates(est); !ok) {
    return std::unexpected(ok.error());
  }
  auto work = prepare_ordinal_modification_index_model(std::move(pt), rep,
                                                       options);
  if (!work.has_value()) return std::unexpected(work.error());

  if (auto v = validate_stats(stats, work->rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(work->pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != work->pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal modification indices: fitted theta length does not match delta partable"));
  }
  auto con0 = ordinal_score_constraints<Stats>(work->pt, est.theta);
  if (!con0.has_value()) return std::unexpected(con0.error());
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);

  inference::ScoreTestTable table;
  for (std::size_t row = 0; row < work->pt.size(); ++row) {
    if (!ordinal_fixed_candidate(work->pt, work->rep, row)) continue;
    spec::LatentStructure aug = work->pt;
    const double fixed_value = aug.fixed_value[row];
    const std::int32_t old_n = aug.n_free();
    aug.free[row] = old_n + 1;
    aug.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
    ordinal_add_free_group(aug, old_n);

    Eigen::VectorXd theta(est.theta.size() + 1);
    if (est.theta.size() > 0) theta.head(est.theta.size()) = est.theta;
    theta(est.theta.size()) = fixed_value;

    auto layout = make_threshold_layout(aug, work->rep, stats);
    if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
    auto factors = weight_factors(stats, weights);
    if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
    auto ev = model::ModelEvaluator::build(aug, work->rep);
    if (!ev.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal modification indices: ModelEvaluator::build failed: " +
              ev.error().detail));
    }
    auto eval = ev->evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal modification indices: fitted evaluation failed: " +
              eval.error().detail));
    }
    auto r = residual_fn(stats, *layout, eval->moments, *factors, theta);
    if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
    auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                         eval->J_mu, *factors, theta);
    if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));

    // The fitter minimizes F/2, where F = r'r and r includes sqrt(W)
    // and block fractions. The score and GN metric of N*F/2 are -N*J'r
    // and N*J'J for both ordinal families, as for equality releases.
    const Eigen::VectorXd score = -n_total * (J->transpose() * *r);
    Eigen::MatrixXd info = n_total * (J->transpose() * *J);
    info = 0.5 * (info + info.transpose());

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(score.size());
    direction(score.size() - 1) = 1.0;
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::FixedParam;
    cand.row = row;
    cand.op = work->pt.op[row];
    cand.lhs_var = work->pt.lhs_var[row];
    cand.rhs_var = work->pt.rhs_var[row];
    cand.group = work->pt.group[row];
    Eigen::MatrixXd K_aug = Eigen::MatrixXd::Zero(score.size(), con0->K().cols());
    if (con0->K().rows() > 0) K_aug.topRows(con0->K().rows()) = con0->K();
    auto res = inference::score_for_direction(cand, score, info, K_aug, direction);
    if (res.has_value()) table.rows.push_back(*res);
    else if (cand.op == parse::Op::Regression) {
      inference::ScoreTestResult unavailable;
      unavailable.candidate = cand;
      unavailable.score = unavailable.information = unavailable.v_eff = unavailable.scaling_factor =
            unavailable.mi = unavailable.mi_scaled = unavailable.p_value = unavailable.epc =
          unavailable.epc_lv = unavailable.epc_all = std::numeric_limits<double>::quiet_NaN();
      unavailable.failure = res.error();
      table.rows.push_back(std::move(unavailable));
    }
  }
  return table;
}

template <class Stats, class ResidualFn, class JacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_score_tests_impl(spec::LatentStructure pt,
                         const model::MatrixRep& rep,
                         const Stats& stats,
                         const Estimates& est,
                         OrdinalWeightKind weights,
                         ResidualFn residual_fn,
                         JacobianFn jacobian_fn,
                         PrepareFn prepare_fn) {
  if (auto ok = require_ls_ordinal_estimates(est); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: fitted theta length does not match delta partable"));
  }
  auto con = ordinal_score_constraints<Stats>(pt, est.theta);
  if (!con.has_value()) return std::unexpected(con.error());
  inference::ScoreTestTable table;
  if (!con->active()) return table;
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);

  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: ModelEvaluator::build failed: " + ev.error().detail));
  }
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: fitted evaluation failed: " + eval.error().detail));
  }
  auto r = residual_fn(stats, *layout, eval->moments, *factors, est.theta);
  if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
  auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                       eval->J_mu, *factors, est.theta);
  if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));
  const Eigen::VectorXd score = -n_total * (J->transpose() * *r);
  Eigen::MatrixXd info = n_total * (J->transpose() * *J);
  info = 0.5 * (info + info.transpose());

  for (Eigen::Index row = 0; row < con->A_eq.rows(); ++row) {
    auto d = ordinal_release_direction(*con, row);
    if (!d.has_value()) return std::unexpected(d.error());
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::EqualityRelease;
    cand.row = static_cast<std::size_t>(row);
    cand.op = parse::Op::EqConstraint;
    auto res = inference::score_for_direction(cand, score, info, con->K(), *d);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}

// ── Robust (generalized / SB-scaled) ordinal score-test machinery ────────────
// The robust twins of the two sweeps above additionally build the moment-metric
// parameter-space sandwich {A1, B1} at each (augmented) null point — the same
// {Δ_b, W_b, Γ̂_b = NACOV_b, n_b} blocks `robust_ordinal` assembles — and hand
// the per-direction scaling to `inference::frontier::score_for_direction_robust`.
// The sandwich uses the unwhitened estimation weight; c carries no
// additional criterion factor, so `mi_scaled` uses the same N*F/2 convention
// as ordinary MI/releases and reduces to it under WLS (W = Γ̂⁻¹).

template <class Stats>
post_expected<void> validate_ordinal_nacov(const Stats& stats) {
  if (stats.NACOV.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: stats.NACOV is not populated"));
  }
  for (std::size_t b = 0; b < stats.NACOV.size(); ++b) {
    const Eigen::Index mb = ordinal_sandwich_block_rows(stats, b);
    if (stats.NACOV[b].rows() != mb || stats.NACOV[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust score tests: NACOV dimension mismatch in block " +
              std::to_string(b)));
    }
  }
  return {};
}

template <class Stats>
post_expected<Eigen::MatrixXd> ordinal_score_sensitivity(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const Stats& stats, const Eigen::VectorXd& theta,
    OrdinalWeightKind weights, OrdinalParameterization parameterization,
    robust::Information bread) {
  if (bread == robust::Information::Expected) return Eigen::MatrixXd{};
  auto parts = [&]() {
    if constexpr (std::is_same_v<Stats, data::OrdinalStats>) {
      return frontier::ordinal_ls_newton_parts_prepared(pt, rep, stats, theta,
                                                     weights, parameterization);
    } else {
      return frontier::mixed_ordinal_ls_newton_parts_prepared(pt, rep, stats, theta,
                                                           weights, parameterization);
    }
  }();
  if (!parts) return std::unexpected(fit_to_post(parts.error()));
  return std::move(parts->hessian);
}

template <class Stats, class ResidualFn, class JacobianFn,
          class MomentJacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_modification_indices_robust_impl(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const Stats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options,
    OrdinalParameterization parameterization,
    bool estimated_weight,
    robust::Information bread,
    ResidualFn residual_fn,
    JacobianFn jacobian_fn,
    MomentJacobianFn moment_jacobian_fn,
    PrepareFn prepare_fn) {
  if (estimated_weight || bread == robust::Information::Observed) {
    if (auto ok = require_linear_sensitivity(pt); !ok) return std::unexpected(ok.error());
  }
  if (auto ok = require_ls_ordinal_estimates(est); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto v = validate_ordinal_nacov(stats); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto work = prepare_ordinal_modification_index_model(std::move(pt), rep,
                                                       options);
  if (!work.has_value()) return std::unexpected(work.error());

  if (auto v = validate_stats(stats, work->rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(work->pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != work->pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust modification indices: fitted theta length does not "
        "match delta partable"));
  }
  auto con0 = ordinal_score_constraints<Stats>(work->pt, est.theta);
  if (!con0.has_value()) return std::unexpected(con0.error());
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);
  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());

  // Estimated-weight (complete-sandwich) meat: precompute the per-block missing
  // pattern once for all-ordinal stats; mixed blocks retain their own recipe.
  std::vector<bool> block_has_missing;
  if constexpr (std::is_same_v<Stats, data::OrdinalStats>) {
    if (estimated_weight) {
      auto missing_or = ordinal_ij_block_missing(stats, weights);
      if (!missing_or.has_value()) return std::unexpected(missing_or.error());
      block_has_missing = std::move(*missing_or);
    }
  }

  inference::ScoreTestTable table;
  for (std::size_t row = 0; row < work->pt.size(); ++row) {
    if (!ordinal_fixed_candidate(work->pt, work->rep, row)) continue;
    spec::LatentStructure aug = work->pt;
    const double fixed_value = aug.fixed_value[row];
    const std::int32_t old_n = aug.n_free();
    aug.free[row] = old_n + 1;
    aug.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
    ordinal_add_free_group(aug, old_n);

    Eigen::VectorXd theta(est.theta.size() + 1);
    if (est.theta.size() > 0) theta.head(est.theta.size()) = est.theta;
    theta(est.theta.size()) = fixed_value;

    auto layout = make_threshold_layout(aug, work->rep, stats);
    if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
    auto factors = weight_factors(stats, weights);
    if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
    auto ev = model::ModelEvaluator::build(aug, work->rep);
    if (!ev.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust modification indices: ModelEvaluator::build failed: " +
              ev.error().detail));
    }
    auto eval = ev->evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust modification indices: fitted evaluation failed: " +
              eval.error().detail));
    }
    auto r = residual_fn(stats, *layout, eval->moments, *factors, theta);
    if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
    auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                         eval->J_mu, *factors, theta);
    if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));

    const Eigen::MatrixXd Delta_full = moment_jacobian_fn(
        stats, *layout, eval->moments, eval->J_sigma, eval->J_mu, theta);
    auto sw = [&]() -> post_expected<robust::ParamSpaceSandwich> {
      if constexpr (std::is_same_v<Stats, data::OrdinalStats>) {
        if (estimated_weight)
          return ordinal_param_space_sandwich_ij(
              stats, *layout, eval->moments, theta, *Ws, Delta_full, weights,
              parameterization, block_has_missing);
      } else if (estimated_weight || bread == robust::Information::Observed) {
        if (!estimated_weight && stats.raw_data.empty() && stats.sampling_moment_influence.empty())
          return ordinal_param_space_sandwich(stats, *Ws, Delta_full);
        auto blocks = build_mixed_ordinal_ij_blocks(stats, *layout, eval->moments,
            theta, *Ws, Delta_full, weights, parameterization, estimated_weight, true);
        if (!blocks) return std::unexpected(blocks.error());
        return weighted_param_space_sandwich_ij(*blocks);
      }
      return ordinal_param_space_sandwich(stats, *Ws, Delta_full);
    }();
    if (!sw.has_value()) return std::unexpected(sw.error());

    // The fitter minimizes F/2, where F = r'r and r includes sqrt(W)
    // and block fractions. The score and GN metric of N*F/2 are -N*J'r
    // and N*J'J for both ordinal families, as for equality releases.
    const Eigen::VectorXd score = -n_total * (J->transpose() * *r);
    Eigen::MatrixXd info = n_total * (J->transpose() * *J);
    info = 0.5 * (info + info.transpose());

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(score.size());
    direction(score.size() - 1) = 1.0;
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::FixedParam;
    cand.row = row;
    cand.op = work->pt.op[row];
    cand.lhs_var = work->pt.lhs_var[row];
    cand.rhs_var = work->pt.rhs_var[row];
    cand.group = work->pt.group[row];
    Eigen::MatrixXd K_aug = Eigen::MatrixXd::Zero(score.size(), con0->K().cols());
    if (con0->K().rows() > 0) K_aug.topRows(con0->K().rows()) = con0->K();
    auto sensitivity = ordinal_score_sensitivity(aug, work->rep, stats, theta,
                                                  weights, parameterization, bread);
    if (!sensitivity) return std::unexpected(sensitivity.error());
    auto res = inference::frontier::score_for_direction_robust(
        cand, score, info, sw->A1, sw->B1, K_aug, direction,
        sensitivity->size() ? &*sensitivity : nullptr);
    if (res.has_value()) table.rows.push_back(*res);
    else if (cand.op == parse::Op::Regression) {
      inference::ScoreTestResult unavailable;
      unavailable.candidate = cand;
      unavailable.score = unavailable.information = unavailable.v_eff = unavailable.scaling_factor =
            unavailable.mi = unavailable.mi_scaled = unavailable.p_value = unavailable.epc =
          unavailable.epc_lv = unavailable.epc_all = std::numeric_limits<double>::quiet_NaN();
      unavailable.failure = res.error();
      table.rows.push_back(std::move(unavailable));
    }
  }
  return table;
}

template <class Stats, class ResidualFn, class JacobianFn,
          class MomentJacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_score_tests_robust_impl(spec::LatentStructure pt,
                                const model::MatrixRep& rep,
                                const Stats& stats,
                                const Estimates& est,
                                OrdinalWeightKind weights,
                                OrdinalParameterization parameterization,
                                bool estimated_weight,
                                robust::Information bread,
                                ResidualFn residual_fn,
                                JacobianFn jacobian_fn,
                                MomentJacobianFn moment_jacobian_fn,
                                PrepareFn prepare_fn) {
  if (estimated_weight || bread == robust::Information::Observed) {
    if (auto ok = require_linear_sensitivity(pt); !ok) return std::unexpected(ok.error());
  }
  if (auto ok = require_ls_ordinal_estimates(est); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto v = validate_ordinal_nacov(stats); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: fitted theta length does not match delta "
        "partable"));
  }
  auto con = ordinal_score_constraints<Stats>(pt, est.theta);
  if (!con.has_value()) return std::unexpected(con.error());
  inference::ScoreTestTable table;
  if (!con->active()) return table;
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);
  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());

  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: ModelEvaluator::build failed: " +
            ev.error().detail));
  }
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: fitted evaluation failed: " +
            eval.error().detail));
  }
  auto r = residual_fn(stats, *layout, eval->moments, *factors, est.theta);
  if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
  auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                       eval->J_mu, *factors, est.theta);
  if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));
  const Eigen::VectorXd score = -n_total * (J->transpose() * *r);
  Eigen::MatrixXd info = n_total * (J->transpose() * *J);
  info = 0.5 * (info + info.transpose());

  const Eigen::MatrixXd Delta_full = moment_jacobian_fn(
      stats, *layout, eval->moments, eval->J_sigma, eval->J_mu, est.theta);
  auto sw = [&]() -> post_expected<robust::ParamSpaceSandwich> {
    if constexpr (std::is_same_v<Stats, data::OrdinalStats>) {
      if (estimated_weight) {
        auto missing_or = ordinal_ij_block_missing(stats, weights);
        if (!missing_or.has_value())
          return std::unexpected(missing_or.error());
        return ordinal_param_space_sandwich_ij(
            stats, *layout, eval->moments, est.theta, *Ws, Delta_full, weights,
            parameterization, *missing_or);
      }
    } else if (estimated_weight || bread == robust::Information::Observed) {
      if (!estimated_weight && stats.raw_data.empty() && stats.sampling_moment_influence.empty())
        return ordinal_param_space_sandwich(stats, *Ws, Delta_full);
      auto blocks = build_mixed_ordinal_ij_blocks(stats, *layout, eval->moments,
          est.theta, *Ws, Delta_full, weights, parameterization, estimated_weight, true);
      if (!blocks) return std::unexpected(blocks.error());
      return weighted_param_space_sandwich_ij(*blocks);
    }
    return ordinal_param_space_sandwich(stats, *Ws, Delta_full);
  }();
  if (!sw.has_value()) return std::unexpected(sw.error());

  auto sensitivity = ordinal_score_sensitivity(pt, rep, stats, est.theta,
                                                weights, parameterization, bread);
  if (!sensitivity) return std::unexpected(sensitivity.error());

  for (Eigen::Index row = 0; row < con->A_eq.rows(); ++row) {
    auto d = ordinal_release_direction(*con, row);
    if (!d.has_value()) return std::unexpected(d.error());
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::EqualityRelease;
    cand.row = static_cast<std::size_t>(row);
    cand.op = parse::Op::EqConstraint;
    auto res = inference::frontier::score_for_direction_robust(
        cand, score, info, sw->A1, sw->B1, con->K(), *d,
        sensitivity->size() ? &*sensitivity : nullptr);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}


}  // namespace
post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights,
                             OrdinalParameterization parameterization,
                             const std::vector<std::int8_t>* row_user) {
  inference::ModificationIndexOptions options;
  return modification_indices_ordinal(std::move(pt), rep, stats, est, weights,
                                      options, parameterization, row_user);
}

post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights,
                             const inference::ModificationIndexOptions& options,
                             OrdinalParameterization parameterization,
                             const std::vector<std::int8_t>* row_user) {
  auto residual_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             parameterization);
  };
  auto jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd& J_mu,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors,
                            theta, parameterization, J_mu);
  };
  auto prepare_fn = [row_user, parameterization](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_partable(p, s, parameterization, nullptr, row_user);
  };
  return ordinal_modification_indices_impl(std::move(pt), rep, stats, est,
                                           weights, options, residual_fn,
                                           jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
score_tests_ordinal(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    const Estimates& est,
                    OrdinalWeightKind weights,
                    OrdinalParameterization parameterization,
                    const std::vector<std::int8_t>* row_user) {
  auto residual_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             parameterization);
  };
  auto jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd& J_mu,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors,
                            theta, parameterization, J_mu);
  };
  auto prepare_fn = [row_user, parameterization](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_partable(p, s, parameterization, nullptr, row_user);
  };
  return ordinal_score_tests_impl(std::move(pt), rep, stats, est, weights,
                                  residual_fn, jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::MixedOrdinalStats& stats,
                                   const Estimates& est,
                                   OrdinalWeightKind weights,
                                   OrdinalParameterization parameterization,
                                   const std::vector<std::int8_t>* row_user) {
  inference::ModificationIndexOptions options;
  return modification_indices_mixed_ordinal(std::move(pt), rep, stats, est,
                                            weights, options, parameterization,
                                            row_user);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options,
    OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* row_user) {
  auto residual_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta,
                                   parameterization);
  };
  auto jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd& J_mu,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors,
                                  theta, parameterization);
  };
  auto prepare_fn = [row_user](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr, row_user);
  };
  return ordinal_modification_indices_impl(std::move(pt), rep, stats, est,
                                           weights, options, residual_fn,
                                           jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
score_tests_mixed_ordinal(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          const Estimates& est,
                          OrdinalWeightKind weights,
                          OrdinalParameterization parameterization,
                          const std::vector<std::int8_t>* row_user) {
  auto residual_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta,
                                   parameterization);
  };
  auto jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd& J_mu,
                                        const WhitenFactors& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors,
                                  theta, parameterization);
  };
  auto prepare_fn = [row_user](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr, row_user);
  };
  return ordinal_score_tests_impl(std::move(pt), rep, stats, est, weights,
                                  residual_fn, jacobian_fn, prepare_fn);
}
namespace frontier {
namespace {

// The (residual, jacobian, moment-jacobian, prepare) lambdas for the robust
// sweeps, binding the parameterization exactly like the non-robust wrappers
// above. The moment-jacobian lambda produces the UNWHITENED Δ the sandwich
// blocks contract against. Returned as a struct of lambdas so the MI and
// score-test wrappers share one definition per stats type.
auto ordinal_robust_handles(OrdinalParameterization parameterization) {
  auto residual_fn = [parameterization](const data::OrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const WhitenFactors& factors,
                                     const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             parameterization);
  };
  auto jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const Eigen::MatrixXd& J_sigma,
                                     const Eigen::MatrixXd& J_mu,
                                     const WhitenFactors& factors,
                                     const Eigen::VectorXd& theta) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors, theta,
                            parameterization, J_mu);
  };
  auto moment_jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                            const ThresholdLayout& layout,
                                            const model::ImpliedMoments& moments,
                                            const Eigen::MatrixXd& J_sigma,
                                            const Eigen::MatrixXd& J_mu,
                                            const Eigen::VectorXd& theta) {
    return ordinal_moment_jacobian(s, layout, moments, J_sigma, theta,
                                   parameterization, J_mu);
  };
  auto prepare_fn = [parameterization](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_partable(p, s, parameterization, nullptr);
  };
  struct Handles {
    decltype(residual_fn) residual;
    decltype(jacobian_fn) jacobian;
    decltype(moment_jacobian_fn) moment_jacobian;
    decltype(prepare_fn) prepare;
  };
  return Handles{std::move(residual_fn), std::move(jacobian_fn),
                 std::move(moment_jacobian_fn), std::move(prepare_fn)};
}

auto mixed_ordinal_robust_handles(OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* row_user) {
  auto residual_fn = [parameterization](const data::MixedOrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const WhitenFactors& factors,
                                     const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta,
                                   parameterization);
  };
  auto jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const Eigen::MatrixXd& J_sigma,
                                     const Eigen::MatrixXd& J_mu,
                                     const WhitenFactors& factors,
                                     const Eigen::VectorXd& theta) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors,
                                  theta, parameterization);
  };
  auto moment_jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                            const ThresholdLayout& layout,
                                            const model::ImpliedMoments& moments,
                                            const Eigen::MatrixXd& J_sigma,
                                            const Eigen::MatrixXd& J_mu,
                                            const Eigen::VectorXd& theta) {
    return mixed_moment_jacobian(s, layout, moments, J_sigma, J_mu, theta,
                                 parameterization);
  };
  auto prepare_fn = [row_user](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr, row_user);
  };
  struct Handles {
    decltype(residual_fn) residual;
    decltype(jacobian_fn) jacobian;
    decltype(moment_jacobian_fn) moment_jacobian;
    decltype(prepare_fn) prepare;
  };
  return Handles{std::move(residual_fn), std::move(jacobian_fn),
                 std::move(moment_jacobian_fn), std::move(prepare_fn)};
}

}  // namespace

post_expected<inference::ScoreTestTable>
modification_indices_ordinal_robust(spec::LatentStructure pt,
                                    const model::MatrixRep& rep,
                                    const data::OrdinalStats& stats,
                                    const Estimates& est,
                                    OrdinalWeightKind weights,
                                    const inference::ModificationIndexOptions&
                                        options,
                                    OrdinalParameterization parameterization,
                                    bool estimated_weight,
                                    robust::Information bread) {
  auto h = ordinal_robust_handles(parameterization);
  return ordinal_modification_indices_robust_impl(
      std::move(pt), rep, stats, est, weights, options, parameterization,
      estimated_weight, bread, h.residual, h.jacobian, h.moment_jacobian, h.prepare);
}

post_expected<inference::ScoreTestTable>
score_tests_ordinal_robust(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::OrdinalStats& stats,
                           const Estimates& est,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization,
                           bool estimated_weight,
                           robust::Information bread) {
  auto h = ordinal_robust_handles(parameterization);
  return ordinal_score_tests_robust_impl(
      std::move(pt), rep, stats, est, weights, parameterization,
      estimated_weight, bread, h.residual, h.jacobian, h.moment_jacobian, h.prepare);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal_robust(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options,
    OrdinalParameterization parameterization,
    bool estimated_weight, robust::Information bread,
    const std::vector<std::int8_t>* row_user) {
  auto h = mixed_ordinal_robust_handles(parameterization, row_user);
  return ordinal_modification_indices_robust_impl(
      std::move(pt), rep, stats, est, weights, options, parameterization,
      estimated_weight, bread, h.residual, h.jacobian,
      h.moment_jacobian, h.prepare);
}

post_expected<inference::ScoreTestTable>
score_tests_mixed_ordinal_robust(spec::LatentStructure pt,
                                 const model::MatrixRep& rep,
                                 const data::MixedOrdinalStats& stats,
                                 const Estimates& est,
                                 OrdinalWeightKind weights,
                                 OrdinalParameterization parameterization,
                                 bool estimated_weight, robust::Information bread,
    const std::vector<std::int8_t>* row_user) {
  auto h = mixed_ordinal_robust_handles(parameterization, row_user);
  return ordinal_score_tests_robust_impl(
      std::move(pt), rep, stats, est, weights, parameterization,
      estimated_weight, bread, h.residual, h.jacobian,
      h.moment_jacobian, h.prepare);
}


namespace {
AssociationMlScore association_score_direction(
    const inference::ScoreCandidate& candidate,
    const post_expected<AssociationMlIJ>& ij,
    const Eigen::MatrixXd& nuisance_full, const Eigen::VectorXd& direction,
    double N) {
  AssociationMlScore out{candidate, std::unexpected(make_post_err(
      PostError::Kind::InfoMatrixSingular, "association score: degenerate release")),
      {}, {}, {}, {}, {}, {}, {}};
  if (!ij) { out.result = std::unexpected(ij.error()); return out; }
  const auto& H = ij->sensitivity;
  const auto& K = ij->coordinates;
  // Pull both the original nuisance tangent and release direction into the
  // augmented active chart. QR avoids assuming either chart is orthonormal.
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> chart(K);
  Eigen::MatrixXd L = chart.solve(nuisance_full);
  Eigen::VectorXd d = chart.solve(direction);
  if ((K*L-nuisance_full).norm() > 1e-8 || (K*d-direction).norm() > 1e-8)
    return out;
  Eigen::VectorXd v = d;
  if (L.cols()) {
    Eigen::MatrixXd Hnn = L.transpose()*H*L;
    Eigen::LDLT<Eigen::MatrixXd> solve(Hnn);
    if (solve.info()!=Eigen::Success || !solve.isPositive()) return out;
    v -= L*solve.solve(L.transpose()*H*d);
  }
  const double score = v.dot(ij->score);
  const double h = v.dot(H*v), b = v.dot(ij->meat*v);
  out.efficient_direction = v;
  out.score = ij->score; out.H = H; out.B = ij->meat;
  out.K = K; out.nuisance = L;
  if (!(h > 0) || !(b > 0) || !std::isfinite(h+b+score)) return out;
  inference::ScoreTestResult r;
  r.candidate = candidate; r.score = score; r.information = h;
  r.v_eff = b; r.mi = N*score*score/h; r.scaling_factor = b/h;
  r.mi_scaled = N*score*score/b;
  r.p_value = std::erfc(std::sqrt(r.mi_scaled/2));
  r.epc = -score/h;
  out.epc_full = K*v*r.epc;
  out.result = r;
  return out;
}
}

post_expected<AssociationMlScoreTable>
association_ml_modification_indices(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::OrdinalStats& stats,
    const Estimates& est, const inference::ModificationIndexOptions& options,
    const std::vector<std::int8_t>* row_user, bool penalized) {
  if (!est.association) return std::unexpected(make_post_err(
      PostError::Kind::UnsupportedInference, "association ML estimates required"));
  auto base = association_ml_ij(pt,rep,stats,est,row_user,penalized);
  if (!base) return std::unexpected(base.error());
  auto work = prepare_ordinal_modification_index_model(pt,rep,options);
  if (!work) return std::unexpected(work.error());
  auto count = total_n_obs(stats);
  if (!count) return std::unexpected(fit_to_post(count.error()));
  AssociationMlScoreTable table;
  for (std::size_t row = 0; row < work->pt.size(); ++row) {
    const bool scale = work->pt.op[row] == parse::Op::ResponseScale && work->pt.free[row]==0;
    if (!scale && !ordinal_fixed_candidate(work->pt,work->rep,row)) continue;
    inference::ScoreCandidate cand;
    cand.row=row; cand.op=work->pt.op[row]; cand.lhs_var=work->pt.lhs_var[row];
    cand.rhs_var=work->pt.rhs_var[row]; cand.group=work->pt.group[row];
    const bool variance = cand.op==parse::Op::Covariance && cand.lhs_var==cand.rhs_var &&
        inference::detail_mi::var_is_indicator(work->pt,cand.lhs_var);
    if (scale || variance || cand.op==parse::Op::Threshold || cand.op==parse::Op::Intercept) {
      table.rows.push_back(AssociationMlScore{cand,std::unexpected(make_post_err(
          PostError::Kind::UnsupportedInference,
          "association score: threshold, mean and response-scale releases change the estimator")),
          {},{},{},{},{},{},{}});
      continue;
    }
    auto aug = work->pt;
    const auto n = aug.n_free();
    const double value = aug.fixed_value[row];
    aug.free[row]=n+1; aug.fixed_value[row]=std::numeric_limits<double>::quiet_NaN();
    // Preparation can remove parameters without compacting merge IDs. The
    // new direction must not collide with a surviving threshold's raw ID.
    if (aug.eq_groups.size()==static_cast<std::size_t>(n)) {
      const auto next=aug.eq_groups.empty() ? 0
          : *std::max_element(aug.eq_groups.begin(),aug.eq_groups.end())+1;
      aug.eq_groups.push_back(next);
    } else if (!aug.eq_groups.empty()) aug.eq_groups.clear();
    // General affine constraints are row-major over the old parameter vector.
    if (!aug.lin_constraint_d.empty()) {
      std::vector<double> R;
      for (std::size_t j=0;j<aug.lin_constraint_d.size();++j) {
        for (int k=0;k<n;++k) R.push_back(aug.lin_constraint_R[j*static_cast<std::size_t>(n)+static_cast<std::size_t>(k)]);
        R.push_back(0);
      }
      aug.lin_constraint_R=std::move(R);
    }
    Estimates point=est; point.theta.conservativeResize(n+1); point.theta(n)=value;
    auto mr = model::build_matrix_rep(aug);
    if (!mr) return std::unexpected(model_to_post(mr.error()));
    Eigen::MatrixXd nuisance=Eigen::MatrixXd::Zero(n+1,base->coordinates.cols());
    nuisance.topRows(n)=base->coordinates;
    Eigen::VectorXd direction=Eigen::VectorXd::Zero(n+1); direction(n)=1;
    auto ij=association_ml_ij(aug,*mr,stats,point,row_user,penalized);
    table.rows.push_back(association_score_direction(cand,ij,nuisance,direction,static_cast<double>(*count)));
  }
  return table;
}

post_expected<AssociationMlScoreTable>
association_ml_score_tests(spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::OrdinalStats& stats, const Estimates& est,
    const std::vector<std::int8_t>* row_user, bool penalized) {
  if (!est.association) return std::unexpected(make_post_err(
      PostError::Kind::UnsupportedInference, "association ML estimates required"));
  auto base=association_ml_ij(pt,rep,stats,est,row_user,penalized);
  if (!base) return std::unexpected(base.error());
  auto con=build_eq_constraints(pt);
  if (!con) return std::unexpected(con.error());
  auto count=total_n_obs(stats);
  if (!count) return std::unexpected(fit_to_post(count.error()));
  AssociationMlScoreTable table;
  for (Eigen::Index row=0;row<con->A_eq.rows();++row) {
    auto direction=ordinal_release_direction(*con,row);
    if (!direction) return std::unexpected(direction.error());
    auto aug=pt;
    aug.eq_groups.clear(); aug.lin_constraint_R.clear(); aug.lin_constraint_d.clear();
    for (Eigen::Index j=0;j<con->A_eq.rows();++j) {
      if (j==row) continue;
      for (Eigen::Index k=0;k<con->A_eq.cols();++k) aug.lin_constraint_R.push_back(con->A_eq(j,k));
      aug.lin_constraint_d.push_back(con->b_eq(j));
    }
    inference::ScoreCandidate cand;
    cand.kind=inference::ScoreCandidateKind::EqualityRelease;
    cand.row=static_cast<std::size_t>(row); cand.op=parse::Op::EqConstraint;
    auto ij=association_ml_ij(aug,rep,stats,est,row_user,penalized);
    table.rows.push_back(association_score_direction(cand,ij,base->coordinates,*direction,static_cast<double>(*count)));
  }
  return table;
}

}  // namespace frontier

}  // namespace magmaan::estimate
