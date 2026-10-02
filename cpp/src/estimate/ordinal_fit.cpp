#include "ordinal_internal.hpp"

namespace magmaan::estimate {

using namespace detail_ordinal;

namespace frontier {
namespace {

struct OrdinalObjectiveState {
  data::OrdinalStats stats;
  ThresholdLayout layout;
  model::ModelEvaluator ev;
  WhitenFactors factors;
  OrdinalParameterization parameterization = OrdinalParameterization::Delta;
};

struct MixedOrdinalObjectiveState {
  data::MixedOrdinalStats stats;
  ThresholdLayout layout;
  model::ModelEvaluator ev;
  WhitenFactors factors;
  OrdinalParameterization parameterization = OrdinalParameterization::Delta;
};


}  // namespace
fit_expected<OrdinalLsObjective>
ordinal_ls_objective(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     const Estimates& at,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization,
                     const std::vector<std::int8_t>* row_user) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr, row_user);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  if (at.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ordinal_ls_objective: theta length does not match ordinal delta "
        "partable"));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ordinal_ls_objective: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval0 = ev_or->evaluate(at.theta, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ordinal_ls_objective: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, *layout_or, eval0->moments, *factors_or,
                              at.theta, parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  auto state = std::make_shared<OrdinalObjectiveState>(OrdinalObjectiveState{
      stats, std::move(*layout_or), std::move(*ev_or), std::move(*factors_or),
      parameterization});

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = at.theta.size();
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = state->ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ordinal_ls_objective: evaluate failed: " + eval.error().detail));
    }
    return ordinal_residuals(state->stats, state->layout, eval->moments,
                             state->factors, x, state->parameterization);
  };
  prob.J = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ordinal_ls_objective: evaluate failed: " + eval.error().detail));
    }
    return ordinal_jacobian(state->stats, state->layout, eval->moments,
                            eval->J_sigma, state->factors, x,
                            state->parameterization, eval->J_mu);
  };
  prob.eval =
      [state](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ordinal_ls_objective: evaluate failed: " + eval.error().detail));
    }
    auto r = ordinal_residuals(state->stats, state->layout, eval->moments,
                               state->factors, x, state->parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = ordinal_jacobian(state->stats, state->layout, eval->moments,
                              eval->J_sigma, state->factors, x,
                              state->parameterization, eval->J_mu);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  return OrdinalLsObjective{std::move(pt), std::move(prob)};
}

fit_expected<OrdinalLsObjective>
mixed_ordinal_ls_objective(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           const Estimates& at,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  if (at.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "mixed_ordinal_ls_objective: theta length does not match mixed delta "
        "partable"));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "mixed_ordinal_ls_objective: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval0 = ev_or->evaluate(at.theta, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "mixed_ordinal_ls_objective: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = mixed_ordinal_residuals(stats, *layout_or, eval0->moments,
                                    *factors_or, at.theta, parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  auto state =
      std::make_shared<MixedOrdinalObjectiveState>(MixedOrdinalObjectiveState{
          stats, std::move(*layout_or), std::move(*ev_or),
          std::move(*factors_or), parameterization});

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = at.theta.size();
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = state->ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "mixed_ordinal_ls_objective: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_residuals(state->stats, state->layout, eval->moments,
                                   state->factors, x,
                                   state->parameterization);
  };
  prob.J = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "mixed_ordinal_ls_objective: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_jacobian(state->stats, state->layout, eval->moments,
                                  eval->J_sigma, eval->J_mu, state->factors,
                                  x, state->parameterization);
  };
  prob.eval =
      [state](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "mixed_ordinal_ls_objective: evaluate failed: " +
              eval.error().detail));
    }
    auto r = mixed_ordinal_residuals(state->stats, state->layout,
                                     eval->moments, state->factors, x,
                                     state->parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = mixed_ordinal_jacobian(state->stats, state->layout, eval->moments,
                                    eval->J_sigma, eval->J_mu,
                                    state->factors, x,
                                    state->parameterization);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  return OrdinalLsObjective{std::move(pt), std::move(prob)};
}

namespace {


FitError newton_parts_error(const std::string& who, const std::string& detail) {
  return make_err(FitError::Kind::NumericIssue, who + ": " + detail);
}

}  // namespace

fit_expected<OrdinalNewtonParts>
ordinal_ls_newton_parts(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const data::OrdinalStats& stats,
                        const Eigen::VectorXd& theta,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization) {
  const std::string who = "ordinal_ls_newton_parts";
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(p.error());
  }
  return ordinal_ls_newton_parts_prepared(pt, rep, stats, theta, weights, parameterization);
}

fit_expected<OrdinalNewtonParts>
ordinal_ls_newton_parts_prepared(const spec::LatentStructure& pt,
                                 const model::MatrixRep& rep,
                                 const data::OrdinalStats& stats,
                                 const Eigen::VectorXd& theta,
                                 OrdinalWeightKind weights,
                                 OrdinalParameterization parameterization) {
  const std::string who = "ordinal_ls_newton_parts";
  if (theta.size() != pt.n_free() || !theta.allFinite()) {
    return std::unexpected(newton_parts_error(who, "invalid theta"));
  }
  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(layout.error());
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(factors.error());
  auto raw = ordinal_newton_parts_prepared(pt, rep, stats, *layout,
      dense_weights_from_factors(*factors), theta, parameterization);
  if (!raw.has_value()) return std::unexpected(newton_parts_error(who, raw.error().detail));
  return OrdinalNewtonParts{pt, std::move(raw->hessian), std::move(raw->metric)};
}

fit_expected<OrdinalNewtonParts>
mixed_ordinal_ls_newton_parts(spec::LatentStructure pt,
                              const model::MatrixRep& rep,
                              const data::MixedOrdinalStats& stats,
                              const Eigen::VectorXd& theta,
                              OrdinalWeightKind weights,
                              OrdinalParameterization parameterization) {
  const std::string who = "mixed_ordinal_ls_newton_parts";
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(p.error());
  }
  return mixed_ordinal_ls_newton_parts_prepared(pt, rep, stats, theta, weights, parameterization);
}

fit_expected<OrdinalNewtonParts>
mixed_ordinal_ls_newton_parts_prepared(const spec::LatentStructure& pt,
                                       const model::MatrixRep& rep,
                                       const data::MixedOrdinalStats& stats,
                                       const Eigen::VectorXd& theta,
                                       OrdinalWeightKind weights,
                                       OrdinalParameterization parameterization) {
  const std::string who = "mixed_ordinal_ls_newton_parts";
  if (theta.size() != pt.n_free() || !theta.allFinite()) {
    return std::unexpected(newton_parts_error(who, "invalid theta"));
  }
  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(layout.error());
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(factors.error());
  auto raw = mixed_newton_parts_prepared(pt, rep, stats, *layout,
      dense_weights_from_factors(*factors), theta, parameterization);
  if (!raw.has_value()) return std::unexpected(newton_parts_error(who, raw.error().detail));
  return OrdinalNewtonParts{pt, std::move(raw->hessian), std::move(raw->metric)};
}

post_expected<std::vector<Eigen::MatrixXd>>
ordinal_stage2_weight_blocks(const data::OrdinalStats& stats,
                             OrdinalStage2Weight kind,
                             OrdinalStage2DlsOptions dls) {
  if (kind == OrdinalStage2Weight::Dls &&
      !(dls.a >= 0.0 && dls.a <= 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_stage2_weight: DLS mixing scalar a must lie in [0, 1]"));
  }
  const std::size_t nb = stats.R.size();
  if (stats.thresholds.size() != nb || stats.NACOV.size() != nb) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_stage2_weight: incomplete OrdinalStats blocks"));
  }

  std::vector<Eigen::MatrixXd> out;
  out.reserve(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mdim = stats.thresholds[b].size() + p * (p - 1) / 2;
    if (stats.R[b].cols() != p || stats.NACOV[b].rows() != mdim ||
        stats.NACOV[b].cols() != mdim || !matrix_all_finite(stats.R[b]) ||
        !matrix_all_finite(stats.NACOV[b])) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_stage2_weight: malformed block " + std::to_string(b)));
    }

    switch (kind) {
      case OrdinalStage2Weight::Uls:
        out.push_back(Eigen::MatrixXd::Identity(mdim, mdim));
        break;
      case OrdinalStage2Weight::Dwls: {
        if (stats.W_dwls.size() == nb && stats.W_dwls[b].rows() == mdim &&
            stats.W_dwls[b].cols() == mdim && matrix_all_finite(stats.W_dwls[b])) {
          out.push_back(stats.W_dwls[b]);
        } else {
          auto W_or = dwls_weight_from_gamma(
              stats.NACOV[b],
              "ordinal_stage2_weight: observed Gamma block " +
                  std::to_string(b));
          if (!W_or.has_value()) return std::unexpected(W_or.error());
          out.push_back(std::move(*W_or));
        }
        break;
      }
      case OrdinalStage2Weight::Wls: {
        if (stats.W_wls.size() == nb && stats.W_wls[b].rows() == mdim &&
            stats.W_wls[b].cols() == mdim && matrix_all_finite(stats.W_wls[b])) {
          out.push_back(stats.W_wls[b]);
        } else {
          auto W_or = sym_inverse_pd_post(
              stats.NACOV[b],
              "ordinal_stage2_weight: observed Gamma block " +
                  std::to_string(b));
          if (!W_or.has_value()) return std::unexpected(W_or.error());
          out.push_back(std::move(*W_or));
        }
        break;
      }
      case OrdinalStage2Weight::Nt: {
        auto Gnt_or = ordinal_nt_gamma_block(stats, b);
        if (!Gnt_or.has_value()) return std::unexpected(Gnt_or.error());
        auto W_or = sym_inverse_pd_post(
            *Gnt_or, "ordinal_stage2_weight: NT Gamma block " +
                         std::to_string(b));
        if (!W_or.has_value()) return std::unexpected(W_or.error());
        out.push_back(std::move(*W_or));
        break;
      }
      case OrdinalStage2Weight::Dls: {
        auto Gnt_or = ordinal_nt_gamma_block(stats, b);
        if (!Gnt_or.has_value()) return std::unexpected(Gnt_or.error());
        Eigen::MatrixXd Gmix =
            (1.0 - dls.a) * (*Gnt_or) + dls.a * stats.NACOV[b];
        Gmix = 0.5 * (Gmix + Gmix.transpose()).eval();
        auto W_or = sym_inverse_pd_post(
            Gmix, "ordinal_stage2_weight: DLS Gamma block " +
                      std::to_string(b));
        if (!W_or.has_value()) return std::unexpected(W_or.error());
        out.push_back(std::move(*W_or));
        break;
      }
    }
  }
  return out;
}

post_expected<data::OrdinalStats>
ordinal_stats_with_stage2_weight(const data::OrdinalStats& stats,
                                 OrdinalStage2Weight kind,
                                 OrdinalStage2DlsOptions dls) {
  auto W_or = ordinal_stage2_weight_blocks(stats, kind, dls);
  if (!W_or.has_value()) return std::unexpected(W_or.error());
  data::OrdinalStats out = stats;
  out.W_wls = std::move(*W_or);
  return out;
}


}  // namespace frontier
namespace detail_ordinal {

namespace {

struct TransformedOrdinalState {
  data::OrdinalStats stats;
  ThresholdLayout layout;
  WhitenFactors factors;
  OrdinalParameterization parameterization;
  TransformedOrdinalEvaluationFn evaluate;
};

struct TransformedMixedOrdinalState {
  data::MixedOrdinalStats stats;
  ThresholdLayout layout;
  WhitenFactors factors;
  OrdinalParameterization parameterization;
  TransformedOrdinalEvaluationFn evaluate;
};

FitError transformed_error(FitError error, const char* callback) {
  error.detail = std::string("ordinal_ls_problem_transformed: ") + callback +
                 ": " + error.detail;
  return error;
}

}  // namespace

fit_expected<optim::GmmProblem>
ordinal_ls_problem_transformed(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Eigen::VectorXd& x0,
    Eigen::Index n_param,
    OrdinalWeightKind weights,
    OrdinalParameterization parameterization,
    TransformedOrdinalEvaluationFn evaluate,
    optim::ExpandFn expand) {
  if (!evaluate || !expand || n_param < 0 || x0.size() != n_param) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        "ordinal_ls_problem_transformed: invalid transformed-coordinate contract"));
  }
  // This builder consumes moments and a fitting weight, not sampling Gamma.
  // Prepared fit-only ULS/DWLS intentionally do not materialize full NACOV.
  // Inference entry points retain the stricter validate_stats contract.
  if (auto v = validate_moments(data::ordinal_moments_from_stats(stats), rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (weights != OrdinalWeightKind::ULS) {
    const auto& Ws = weights == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
    if (Ws.size() != stats.R.size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal_ls_problem_transformed: weight block count mismatch"));
    }
    for (std::size_t b = 0; b < Ws.size(); ++b) {
      const auto p = stats.R[b].rows();
      const auto mdim = stats.thresholds[b].size() + p * (p - 1) / 2;
      if (Ws[b].rows() != mdim || Ws[b].cols() != mdim || !matrix_all_finite(Ws[b])) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "ordinal_ls_problem_transformed: weight dimension or finiteness mismatch"));
      }
    }
  }
  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(layout.error());
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(factors.error());

  auto start_eval = evaluate(x0, false);
  if (!start_eval.has_value()) {
    return std::unexpected(transformed_error(start_eval.error(), "start"));
  }
  auto r0 = ordinal_residuals(stats, *layout, start_eval->evaluation.moments,
                              *factors, start_eval->theta,
                              parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  auto state = std::make_shared<TransformedOrdinalState>(
      TransformedOrdinalState{stats, std::move(*layout), std::move(*factors),
                              parameterization, std::move(evaluate)});

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = n_param;
  prob.expand = std::move(expand);
  prob.r = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto transformed = state->evaluate(x, false);
    if (!transformed.has_value()) {
      return std::unexpected(transformed_error(transformed.error(), "residual"));
    }
    return ordinal_residuals(
        state->stats, state->layout, transformed->evaluation.moments,
        state->factors, transformed->theta, state->parameterization);
  };
  prob.J = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto transformed = state->evaluate(x, true);
    if (!transformed.has_value()) {
      return std::unexpected(transformed_error(transformed.error(), "jacobian"));
    }
    return ordinal_jacobian(
        state->stats, state->layout, transformed->evaluation.moments,
        transformed->evaluation.J_sigma, state->factors, transformed->theta,
        state->parameterization, transformed->evaluation.J_mu,
        &transformed->J_theta);
  };
  prob.eval =
      [state](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    auto transformed = state->evaluate(x, true);
    if (!transformed.has_value()) {
      return std::unexpected(transformed_error(transformed.error(), "evaluate"));
    }
    auto r = ordinal_residuals(
        state->stats, state->layout, transformed->evaluation.moments,
        state->factors, transformed->theta, state->parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = ordinal_jacobian(
        state->stats, state->layout, transformed->evaluation.moments,
        transformed->evaluation.J_sigma, state->factors, transformed->theta,
        state->parameterization, transformed->evaluation.J_mu,
        &transformed->J_theta);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };
  return prob;
}

fit_expected<optim::GmmProblem>
mixed_ordinal_ls_problem_transformed(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Eigen::VectorXd& x0,
    Eigen::Index n_param,
    OrdinalWeightKind weights,
    OrdinalParameterization parameterization,
    TransformedOrdinalEvaluationFn evaluate,
    optim::ExpandFn expand) {
  constexpr const char* who = "mixed_ordinal_ls_problem_transformed";
  if (!evaluate || !expand || n_param < 0 || x0.size() != n_param) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        std::string(who) + ": invalid transformed-coordinate contract"));
  }
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(layout.error());
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(factors.error());

  auto transformed_error = [](FitError error, const char* callback) {
    error.detail = std::string("mixed_ordinal_ls_problem_transformed: ") +
                   callback + ": " + error.detail;
    return error;
  };
  auto start_eval = evaluate(x0, false);
  if (!start_eval.has_value()) {
    return std::unexpected(transformed_error(start_eval.error(), "start"));
  }
  auto r0 = mixed_ordinal_residuals(
      stats, *layout, start_eval->evaluation.moments, *factors,
      start_eval->theta, parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  auto state = std::make_shared<TransformedMixedOrdinalState>(
      TransformedMixedOrdinalState{stats, std::move(*layout),
                                   std::move(*factors), parameterization,
                                   std::move(evaluate)});

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = n_param;
  prob.expand = std::move(expand);
  prob.r = [state, transformed_error](const Eigen::VectorXd& x)
      -> fit_expected<Eigen::VectorXd> {
    auto transformed = state->evaluate(x, false);
    if (!transformed.has_value()) {
      return std::unexpected(
          transformed_error(transformed.error(), "residual"));
    }
    return mixed_ordinal_residuals(
        state->stats, state->layout, transformed->evaluation.moments,
        state->factors, transformed->theta, state->parameterization);
  };
  prob.J = [state, transformed_error](const Eigen::VectorXd& x)
      -> fit_expected<Eigen::MatrixXd> {
    auto transformed = state->evaluate(x, true);
    if (!transformed.has_value()) {
      return std::unexpected(
          transformed_error(transformed.error(), "jacobian"));
    }
    return mixed_ordinal_jacobian(
        state->stats, state->layout, transformed->evaluation.moments,
        transformed->evaluation.J_sigma, transformed->evaluation.J_mu,
        state->factors, transformed->theta, state->parameterization,
        &transformed->J_theta);
  };
  prob.eval = [state, transformed_error](const Eigen::VectorXd& x)
      -> fit_expected<optim::LsEvaluation> {
    auto transformed = state->evaluate(x, true);
    if (!transformed.has_value()) {
      return std::unexpected(
          transformed_error(transformed.error(), "evaluate"));
    }
    auto r = mixed_ordinal_residuals(
        state->stats, state->layout, transformed->evaluation.moments,
        state->factors, transformed->theta, state->parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = mixed_ordinal_jacobian(
        state->stats, state->layout, transformed->evaluation.moments,
        transformed->evaluation.J_sigma, transformed->evaluation.J_mu,
        state->factors, transformed->theta, state->parameterization,
        &transformed->J_theta);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };
  return prob;
}

}  // namespace detail_ordinal

namespace {

// Dispatch a bounded least-squares problem (residual + Jacobian closures) to
// the chosen backend. The problem is driven directly — any equality
// reparameterization has already been folded into the closures by the caller.
fit_expected<optim::OptimResult>
run_ordinal_ls(const optim::GmmProblem& prob, const Eigen::VectorXd& x0,
               const Bounds& bounds, Backend backend, OptimOptions opts) {
  if (backend == Backend::Ceres) {
#ifdef MAGMAAN_WITH_CERES
    optim::CeresOptions copts;
    copts.max_iter = opts.max_iter;
    copts.ftol     = opts.ceres.function_tolerance.value_or(opts.ftol);
    copts.gtol     = opts.ceres.gradient_tolerance.value_or(opts.gtol);
    copts.ptol     = opts.ceres.parameter_tolerance.value_or(copts.ptol);
    return optim::ceres_lm(prob, x0, bounds, copts);
#else
    (void)opts;
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: Ceres backend requested but MAGMAAN_WITH_CERES "
        "is off"));
#endif
  }
  if (backend == Backend::PortNls) {
#ifdef MAGMAAN_WITH_PORT
    // NL2SOL sees the multi-residual structure directly, matching the Ceres
    // LM dispatch above; no scalarisation.
    return optim::port_nls(prob, x0, bounds, opts);
#else
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: PortNls backend requested but MAGMAAN_WITH_PORT "
        "is off"));
#endif
  }
  if (backend == Backend::Ipopt) {
#ifdef MAGMAAN_WITH_IPOPT
    return optim::ipopt(optim::scalarize(prob), x0, bounds, opts);
#else
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: IPOPT backend requested but MAGMAAN_WITH_IPOPT "
        "is off"));
#endif
  }
  const optim::ScalarProblem scalar = optim::scalarize(prob);
  if (backend == Backend::NloptSlsqp) {
    return optim::nlopt_slsqp(scalar, x0, bounds, opts);
  }
  if (backend == Backend::NloptLbfgsSlsqpFallback) {
    return optim::nlopt_lbfgs_slsqp_fallback(scalar, x0, bounds, opts);
  }
  return optim::nlopt_lbfgs(scalar, x0, bounds, opts);
}

fit_expected<optim::OptimResult>
run_ordinal_scalar_constrained(const optim::ScalarProblem& prob,
                               const optim::ConstraintFn& h,
                               const optim::ConstraintJacFn& J_h,
                               Eigen::Index n_constraint,
                               const Eigen::VectorXd& x0,
                               const Bounds& bounds,
                               Backend backend,
                               OptimOptions opts,
                               const char* who) {
  optim::ConstrainedScalarProblem cprob;
  cprob.objective = prob;
  cprob.h = h;
  cprob.J_h = J_h;
  cprob.n_constraint = n_constraint;
  cprob.constraint_lower = Eigen::VectorXd::Zero(n_constraint);
  cprob.constraint_upper = Eigen::VectorXd::Zero(n_constraint);

  switch (backend) {
    case Backend::NloptSlsqp:
    case Backend::NloptLbfgsSlsqpFallback:
      return optim::nlopt_slsqp_constrained(cprob, x0, bounds, opts);
    case Backend::Ipopt:
#ifdef MAGMAAN_WITH_IPOPT
      return optim::ipopt_constrained(cprob, x0, bounds, opts);
#else
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) +
              ": nonlinear equality constraints require optimizer "
              "\"nlopt-slsqp\" or an IPOPT-enabled build with optimizer "
              "\"ipopt\""));
#endif
    default:
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) +
              ": nonlinear equality constraints require optimizer "
              "\"nlopt-slsqp\" or \"ipopt\""));
  }
  return std::unexpected(make_err(FitError::Kind::NumericIssue,
      "unknown constrained optimizer backend"));
}

fit_expected<void>
validate_extra_ordinal_constraints(
    const frontier::ExtraNonlinearEqConstraints& extra,
    const Eigen::VectorXd& theta0,
    Eigen::Index npar,
    const char* who) {
  if (!extra.active()) return {};
  if (!extra.h || !extra.jacobian) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) +
            ": active extra nonlinear constraints require both h and jacobian"));
  }
  const Eigen::VectorXd h0 = extra.h(theta0);
  if (h0.size() != extra.n_constraint) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": extra h(theta) length (" +
            std::to_string(h0.size()) + ") != n_constraint (" +
            std::to_string(extra.n_constraint) + ")"));
  }
  const Eigen::MatrixXd J0 = extra.jacobian(theta0);
  if (J0.rows() != extra.n_constraint || J0.cols() != npar) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": extra jacobian shape (" +
            std::to_string(J0.rows()) + " x " + std::to_string(J0.cols()) +
            ") != (" + std::to_string(extra.n_constraint) + " x " +
            std::to_string(npar) + ")"));
  }
  return {};
}

// Solve a data-only ordinal LS problem `prob` (residual size `prob.n_resid`,
// `prob.expand` the identity in full θ) under linear equality constraints.
//
// Equality constraints are eliminated by the affine reparameterization
// θ = θ₀ + K·α (same as the ML / GMM path in `fit.cpp`), *not* by a quadratic
// penalty: a large penalty makes the constrained directions O(μ)-stiff, so the
// optimizer stalls on function-value stagnation at an FP-path-dependent iterate
// well short of the true optimum. The reduced α-problem carries no such
// conditioning and converges on the gradient stop.
fit_expected<Estimates>
solve_ordinal_ls(const optim::GmmProblem& prob, const Eigen::VectorXd& x0,
                 const Bounds& bounds, const EqConstraints& con,
                 Backend backend, OptimOptions opts, const char* who) {
  if (!con.active()) {
    auto out = run_ordinal_ls(prob, x0, bounds, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    return Estimates{prob.expand(out->x), out->fmin, out->iterations,
                     out->f_evals, out->g_evals, out->status,
                     out->grad_inf_norm, std::move(out->audit)};
  }

  const optim::GmmProblem prob_a = optim::reparameterize(prob, con);
  if (con.n_alpha == 0) {  // every parameter pinned by the linear system
    auto r = prob_a.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    return Estimates{prob_a.expand(Eigen::VectorXd(0)), 0.5 * r->squaredNorm(), 0};
  }

  Eigen::VectorXd alpha0 = con.contract(x0);
  Bounds abounds;
  const bool pure_merge = !con.group.empty();
  if (!bounds.empty() && pure_merge) {
    abounds = optim::fold_alpha_bounds(con, bounds);
    alpha0  = alpha0.cwiseMax(abounds.lower).cwiseMin(abounds.upper);
  }
  auto out = run_ordinal_ls(prob_a, alpha0, abounds, backend, opts);
  if (!out.has_value()) return std::unexpected(out.error());
  Eigen::VectorXd theta_hat = prob_a.expand(out->x);
  if (!bounds.empty() && !pure_merge) {
    // General-linear α was optimized unbounded — verify θ̂ honors the box.
    constexpr double tol = 1e-6;
    for (Eigen::Index k = 0; k < theta_hat.size(); ++k) {
      if (theta_hat(k) < bounds.lower(k) - tol ||
          theta_hat(k) > bounds.upper(k) + tol) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            std::string(who) +
                ": general-linear equality drove parameter " +
                std::to_string(k) + " past its bound"));
      }
    }
  }
  return Estimates{std::move(theta_hat), out->fmin, out->iterations,
                   out->f_evals, out->g_evals, out->status,
                   out->grad_inf_norm, std::move(out->audit)};
}

// What the fit-time Newton check needs from an ordinal fit: the moments, the
// threshold layout and the whitening factors the fit used.
struct OrdinalNewtonContext {
  const model::MatrixRep* rep = nullptr;
  const data::OrdinalStats* stats = nullptr;       // all-ordinal
  const data::MixedOrdinalStats* mixed = nullptr;  // mixed continuous/ordinal
  const ThresholdLayout* layout = nullptr;
  const WhitenFactors* factors = nullptr;
  OrdinalParameterization parameterization = OrdinalParameterization::Delta;
};

// Newton check of an ordinal least-squares fit: the exact Hessian of its
// objective, with the Gauss-Newton sandwich metric. `value` and `gradient` are
// the per-observation objective and gradient at est.theta.
void attach_ordinal_newton_accuracy(Estimates& est,
                                    const spec::LatentStructure& pt,
                                    const OrdinalNewtonContext& c,
                                    double value,
                                    const Eigen::VectorXd& gradient) {
  if (c.rep == nullptr || c.layout == nullptr || c.factors == nullptr ||
      (c.stats == nullptr && c.mixed == nullptr)) return;
  frontier::NewtonDerivatives d;
  d.theta = est.theta;
  d.objective_kind = c.stats != nullptr ? NewtonObjectiveKind::OrdinalLeastSquares
                                        : NewtonObjectiveKind::MixedOrdinalLeastSquares;
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  d.metric_kind = NewtonMetricKind::Sandwich;
  auto n_or = c.stats != nullptr ? total_n_obs(*c.stats) : total_n_obs(*c.mixed);
  if (!n_or.has_value()) return;
  const double n = static_cast<double>(*n_or);
  d.n_obs = n;
  d.native_to_total = n;
  d.objective = value;
  d.gradient = n * gradient;
  if (std::isfinite(value) && d.gradient.allFinite()) {
    const auto Ws = dense_weights_from_factors(*c.factors);
    auto raw = c.stats != nullptr
        ? ordinal_newton_parts_prepared(pt, *c.rep, *c.stats, *c.layout, Ws,
                                        est.theta, c.parameterization)
        : mixed_newton_parts_prepared(pt, *c.rep, *c.mixed, *c.layout, Ws,
                                      est.theta, c.parameterization);
    if (raw.has_value() && raw->hessian.allFinite() && raw->metric.allFinite()) {
      d.hessian = std::move(raw->hessian);
      d.metric = std::move(raw->metric);
      d.status = NewtonAccuracyStatus::Available;
    } else {
      d.detail = raw.has_value() ? "non-finite ordinal Hessian" : raw.error().detail;
    }
  }
  est.diagnostics.newton_accuracy = frontier::audit_newton_derivatives(
      pt, *c.rep, std::move(d), StationarityDomain::Ambient).diagnostics;
}

void attach_ordinal_geometric_diagnostics(
    Estimates& est,
    const spec::LatentStructure& pt,
    const model::ModelEvaluator& ev,
    const EqConstraints& con,
    const Bounds& bounds,
    const optim::GmmProblem& full_theta_problem,
    const OrdinalNewtonContext& newton = {}) {
  const NonlinearEqConstraints nl = build_nl_constraints(pt);
  est.diagnostics = finalize_fit_diagnostics(
      est.theta, pt, ev, con, nl, bounds);
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(est.theta.size());
  const optim::ScalarProblem scalar = optim::scalarize(full_theta_problem);
  const double value = scalar.f(est.theta, gradient);
  if (!std::isfinite(value)) {
    gradient.setConstant(std::numeric_limits<double>::quiet_NaN());
  }
  audit_full_model_fit(est.diagnostics, est.theta, gradient, est.fmin, value,
                       pt, ev, con, nl, bounds);
  attach_ordinal_newton_accuracy(est, pt, newton, value, gradient);
}

// Threshold profiling must be audited against the original moment residuals,
// including threshold coordinates eliminated during optimization.
void attach_reconstructed_ordinal_diagnostics(
    Estimates& est, const spec::LatentStructure& pt,
    const model::ModelEvaluator& ev, const data::OrdinalStats& stats,
    const ThresholdLayout& layout, const WhitenFactors& factors,
    const Bounds& bounds, OrdinalParameterization parameterization,
    const model::MatrixRep* rep = nullptr) {
  auto con = build_eq_constraints(pt);
  if (!con.has_value()) return;
  const auto nl = build_nl_constraints(pt);
  est.diagnostics = finalize_fit_diagnostics(
      est.theta, pt, ev, *con, nl, bounds);
  Eigen::VectorXd gradient = Eigen::VectorXd::Constant(
      est.theta.size(), std::numeric_limits<double>::quiet_NaN());
  double value = std::numeric_limits<double>::infinity();
  auto eval = ev.evaluate(est.theta, true, true);
  if (eval.has_value()) {
    auto r = ordinal_residuals(stats, layout, eval->moments, factors,
                              est.theta, parameterization);
    auto J = ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                             factors, est.theta, parameterization, eval->J_mu);
    if (r.has_value() && J.has_value()) {
      value = 0.5 * r->squaredNorm();
      gradient = J->transpose() * *r;
    }
  }
  audit_full_model_fit(est.diagnostics, est.theta, gradient, est.fmin, value,
                       pt, ev, *con, nl, bounds);
  OrdinalNewtonContext newton;
  newton.rep = rep;
  newton.stats = &stats;
  newton.layout = &layout;
  newton.factors = &factors;
  newton.parameterization = parameterization;
  attach_ordinal_newton_accuracy(est, pt, newton, value, gradient);
}

fit_expected<Estimates>
solve_ordinal_ls_extra(const optim::GmmProblem& prob,
                       const Eigen::VectorXd& x0,
                       const Bounds& bounds,
                       const EqConstraints& con,
                       const NonlinearEqConstraints& nl,
                       const frontier::ExtraNonlinearEqConstraints& extra,
                       Backend backend,
                       OptimOptions opts,
                       const char* who) {
  if (auto ok = validate_extra_ordinal_constraints(extra, x0, prob.n_param, who);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  const Eigen::Index n_constraint =
      static_cast<Eigen::Index>(nl.m()) + extra.n_constraint;
  if (n_constraint == 0) {
    return solve_ordinal_ls(prob, x0, bounds, con, backend, opts, who);
  }

  auto h_theta = [&nl, &extra, n_constraint](const Eigen::VectorXd& theta) {
    Eigen::VectorXd out(n_constraint);
    Eigen::Index off = 0;
    if (nl.active()) {
      out.segment(off, nl.m()) = nl.h(theta);
      off += nl.m();
    }
    if (extra.active()) {
      out.segment(off, extra.n_constraint) = extra.h(theta);
    }
    return out;
  };
  auto J_theta = [&nl, &extra, n_constraint](const Eigen::VectorXd& theta) {
    Eigen::MatrixXd out(n_constraint, theta.size());
    Eigen::Index off = 0;
    if (nl.active()) {
      out.block(off, 0, nl.m(), theta.size()) = nl.jacobian(theta);
      off += nl.m();
    }
    if (extra.active()) {
      out.block(off, 0, extra.n_constraint, theta.size()) =
          extra.jacobian(theta);
    }
    return out;
  };

  if (!con.active()) {
    const optim::ScalarProblem sprob = optim::scalarize(prob);
    auto r = run_ordinal_scalar_constrained(
        sprob, h_theta, J_theta, n_constraint, x0, bounds, backend, opts, who);
    if (!r.has_value()) return std::unexpected(r.error());
    return Estimates{prob.expand(r->x), r->fmin, r->iterations,
                     r->f_evals, r->g_evals, r->status, r->grad_inf_norm,
                     std::move(r->audit)};
  }

  const optim::GmmProblem prob_a = optim::reparameterize(prob, con);
  if (con.n_alpha == 0) {
    Eigen::VectorXd theta = con.expand(Eigen::VectorXd(0));
    const Eigen::VectorXd h0 = h_theta(theta);
    constexpr double tol = 1e-8;
    if (h0.lpNorm<Eigen::Infinity>() > tol) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) +
              ": all parameters are fixed by linear constraints and nonlinear "
              "constraints are not satisfied"));
    }
    auto r = prob.r(theta);
    if (!r.has_value()) return std::unexpected(r.error());
    return Estimates{std::move(theta), 0.5 * r->squaredNorm(), 0};
  }

  const optim::ScalarProblem sprob_a = optim::scalarize(prob_a);
  auto h_a = [&h_theta, &con](const Eigen::VectorXd& a) {
    return h_theta(con.expand(a));
  };
  auto J_a = [&J_theta, &con](const Eigen::VectorXd& a) {
    return Eigen::MatrixXd(J_theta(con.expand(a)) * con.K());
  };

  Eigen::VectorXd alpha0 = con.contract(x0);
  Bounds abounds;
  const bool pure_merge = !con.group.empty();
  if (!bounds.empty() && pure_merge) {
    abounds = optim::fold_alpha_bounds(con, bounds);
    alpha0 = alpha0.cwiseMax(abounds.lower).cwiseMin(abounds.upper);
  }

  auto r = run_ordinal_scalar_constrained(
      sprob_a, h_a, J_a, n_constraint, alpha0, abounds, backend, opts, who);
  if (!r.has_value()) return std::unexpected(r.error());

  Eigen::VectorXd theta_hat = prob_a.expand(r->x);
  if (!bounds.empty() && !pure_merge) {
    constexpr double tol = 1e-6;
    for (Eigen::Index k = 0; k < theta_hat.size(); ++k) {
      if (theta_hat(k) < bounds.lower(k) - tol ||
          theta_hat(k) > bounds.upper(k) + tol) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            std::string(who) +
                ": general-linear equality drove parameter " +
                std::to_string(k) + " past its bound"));
      }
    }
  }
  return Estimates{std::move(theta_hat), r->fmin, r->iterations,
                   r->f_evals, r->g_evals, r->status, r->grad_inf_norm,
                   std::move(r->audit)};
}

Eigen::VectorXd fd_scalar_gradient_ordinal(
    const std::function<double(const Eigen::VectorXd&)>& value,
    const Eigen::VectorXd& theta) {
  Eigen::VectorXd grad(theta.size());
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    const double h = 1e-6 * std::max(1.0, std::abs(theta(k)));
    Eigen::VectorXd xp = theta;
    Eigen::VectorXd xm = theta;
    xp(k) += h;
    xm(k) -= h;
    const double fp = value(xp);
    const double fm = value(xm);
    grad(k) = (fp - fm) / (2.0 * h);
  }
  return grad;
}

fit_expected<Eigen::VectorXd>
scalar_gradient_ordinal(const frontier::ScalarFunctional& functional,
                        const Eigen::VectorXd& theta,
                        Eigen::Index npar,
                        const char* who) {
  if (!functional.value) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar functional needs a value callback"));
  }
  Eigen::VectorXd grad =
      functional.gradient ? functional.gradient(theta)
                          : fd_scalar_gradient_ordinal(functional.value, theta);
  if (grad.size() != npar) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar functional gradient length (" +
            std::to_string(grad.size()) + ") != n_free (" +
            std::to_string(npar) + ")"));
  }
  if (!grad.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar functional gradient is not finite"));
  }
  return grad;
}

fit_expected<double>
scalar_profile_scale_from_reduced_ordinal(const Eigen::MatrixXd& A,
                                          const Eigen::MatrixXd& B,
                                          const Eigen::VectorXd& r,
                                          const char* who) {
  if (r.size() == 0 || A.rows() != A.cols() || B.rows() != B.cols() ||
      A.rows() != B.rows() || A.rows() != r.size() || !A.allFinite() ||
      !B.allFinite() || !r.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": non-finite robust profile scaling inputs"));
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (A + A.transpose()).eval());
  if (ldlt.info() != Eigen::Success) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": robust profile bread factorization failed"));
  }
  const Eigen::VectorXd x = ldlt.solve(r);
  if (ldlt.info() != Eigen::Success || !x.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": robust profile bread solve failed"));
  }
  const Eigen::MatrixXd Bsym = 0.5 * (B + B.transpose()).eval();
  const double bread_q = x.dot(A * x);
  const double meat_q = x.dot(Bsym * x);
  const double tol = 1e-12 * std::max(1.0, std::abs(bread_q));
  if (!(bread_q > tol) || !std::isfinite(meat_q) || !(meat_q > 0.0)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": robust profile sandwich quadratic form is not "
            "positive"));
  }
  return meat_q / bread_q;
}

double standard_normal_quantile_ordinal(double p) noexcept {
  if (!(p > 0.0 && p < 1.0) || !std::isfinite(p)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 =  2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 =  1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 =  2.506628277459239e+00;

  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 =  1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 =  6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;

  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 =  4.374664141464968e+00;
  constexpr double c6 =  2.938163982698783e+00;

  constexpr double d1 =  7.784695709041462e-03;
  constexpr double d2 =  3.224671290700398e-01;
  constexpr double d3 =  2.445134137142996e+00;
  constexpr double d4 =  3.754408661907416e+00;

  constexpr double plow = 0.02425;
  constexpr double phigh = 1.0 - plow;

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

fit_expected<double>
profile_ci_cutoff_ordinal(
    const frontier::ScalarProfileCiOptions& options,
    const char* who) {
  if (!(options.confidence_level > 0.0 && options.confidence_level < 1.0) ||
      !std::isfinite(options.confidence_level)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": confidence_level must be in (0, 1)"));
  }
  if (std::isfinite(options.cutoff)) {
    if (!(options.cutoff > 0.0)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": cutoff must be positive"));
    }
    return options.cutoff;
  }
  const double z = standard_normal_quantile_ordinal(
      0.5 * (1.0 + options.confidence_level));
  if (!std::isfinite(z)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": failed to compute chi-square cutoff"));
  }
  return z * z;
}

fit_expected<void>
validate_ci_options_ordinal(
    const frontier::ScalarProfileCiOptions& options,
    double estimate,
    const char* who) {
  if (!(options.target_tol > 0.0) || !std::isfinite(options.target_tol)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": target_tol must be positive and finite"));
  }
  if (!(options.statistic_tol > 0.0) ||
      !std::isfinite(options.statistic_tol)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": statistic_tol must be positive and finite"));
  }
  if (options.max_iter <= 0 || options.max_expand <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": max_iter and max_expand must be positive"));
  }
  if (std::isfinite(options.initial_step) && !(options.initial_step > 0.0)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": initial_step must be positive"));
  }
  if (std::isfinite(options.lower_bound) &&
      !(options.lower_bound < estimate)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": lower_bound must be below the estimate"));
  }
  if (std::isfinite(options.upper_bound) &&
      !(options.upper_bound > estimate)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": upper_bound must be above the estimate"));
  }
  return {};
}

struct OrdinalProfileRootPoint {
  frontier::ScalarProfileLrtResult profile;
  double signed_stat = 0.0;
};

struct OrdinalProfileRootResult {
  frontier::ScalarProfileLrtResult profile;
  double root = 0.0;
  int evals = 0;
  bool at_bound = false;
};

using OrdinalProfileEvaluator =
    std::function<fit_expected<frontier::ScalarProfileLrtResult>(double)>;

fit_expected<OrdinalProfileRootPoint>
eval_ordinal_profile_point(const OrdinalProfileEvaluator& eval,
                           double target,
                           double cutoff,
                           const frontier::ScalarProfileCiOptions& options,
                           const char* who) {
  auto r = eval(target);
  if (!r.has_value()) return std::unexpected(r.error());
  double stat = r->T;
  double endpoint_cutoff = cutoff;
  switch (options.reference) {
    case frontier::ScalarProfileReference::Ordinary:
      stat = r->T;
      break;
    case frontier::ScalarProfileReference::RobustScaled:
      stat = r->T_scaled;
      break;
    case frontier::ScalarProfileReference::MisspecScaled:
      stat = r->T_misspec_scaled;
      break;
    case frontier::ScalarProfileReference::MisspecMixture:
      stat = r->T;
      if (!std::isfinite(options.cutoff)) {
        if (r->misspec_eigvals.size() == 0 ||
            !r->misspec_eigvals.allFinite()) {
          return std::unexpected(make_err(FitError::Kind::NumericIssue,
              std::string(who) + ": misspec mixture reference requires finite "
                  "weighted chi-square eigenvalues"));
        }
        endpoint_cutoff = robust::weighted_chisq_quantile(
            r->misspec_eigvals, options.confidence_level);
      }
      r->misspec_mixture_cutoff = endpoint_cutoff;
      break;
  }
  if (!std::isfinite(stat) || !std::isfinite(endpoint_cutoff)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": profile statistic is not finite at target " +
            std::to_string(target)));
  }
  OrdinalProfileRootPoint out;
  out.profile = std::move(*r);
  out.signed_stat = stat - endpoint_cutoff;
  return out;
}

fit_expected<OrdinalProfileRootResult>
solve_ordinal_profile_root_side(
    const OrdinalProfileEvaluator& eval,
    double estimate,
    const frontier::ScalarProfileCiOptions& options,
    double cutoff,
    bool lower_side,
    const char* who) {
  int evals = 0;
  const double sign = lower_side ? -1.0 : 1.0;
  const double finite_bound =
      lower_side ? options.lower_bound : options.upper_bound;
  const bool has_bound = std::isfinite(finite_bound);
  double outside_x = finite_bound;
  OrdinalProfileRootPoint outside;

  if (has_bound) {
    auto p = eval_ordinal_profile_point(eval, outside_x, cutoff,
                                        options, who);
    ++evals;
    if (!p.has_value()) return std::unexpected(p.error());
    if (p->signed_stat < 0.0) {
      return OrdinalProfileRootResult{
          std::move(p->profile), outside_x, evals, true};
    }
    outside = std::move(*p);
  } else {
    double step = std::isfinite(options.initial_step)
                    ? options.initial_step
                    : 0.1 * std::max(1.0, std::abs(estimate));
    bool bracketed = false;
    for (int k = 0; k < options.max_expand; ++k) {
      outside_x = estimate + sign * step;
      auto p = eval_ordinal_profile_point(eval, outside_x, cutoff,
                                          options, who);
      ++evals;
      if (!p.has_value()) return std::unexpected(p.error());
      if (p->signed_stat >= 0.0) {
        outside = std::move(*p);
        bracketed = true;
        break;
      }
      step *= 2.0;
    }
    if (!bracketed) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + (lower_side
              ? ": failed to bracket lower profile-CI root"
              : ": failed to bracket upper profile-CI root")));
    }
  }

  double inside_x = estimate;
  OrdinalProfileRootPoint best = std::move(outside);
  double best_x = outside_x;
  for (int k = 0; k < options.max_iter; ++k) {
    const double mid = 0.5 * (inside_x + outside_x);
    auto p = eval_ordinal_profile_point(eval, mid, cutoff, options, who);
    ++evals;
    if (!p.has_value()) return std::unexpected(p.error());
    const double signed_stat = p->signed_stat;
    if (std::abs(p->signed_stat) < std::abs(best.signed_stat)) {
      best = std::move(*p);
      best_x = mid;
    }
    if (signed_stat >= 0.0) {
      outside_x = mid;
    } else {
      inside_x = mid;
    }

    const double width = std::abs(outside_x - inside_x);
    const double scale = std::max(1.0, std::abs(mid));
    if (std::abs(best.signed_stat) <= options.statistic_tol ||
        width <= options.target_tol * scale) {
      return OrdinalProfileRootResult{
          std::move(best.profile), best_x, evals, false};
    }
  }

  return OrdinalProfileRootResult{
      std::move(best.profile), best_x, evals, false};
}

fit_expected<frontier::ScalarProfileCiResult>
ordinal_profile_ci_from_evaluator(
    double estimate,
    const frontier::ScalarProfileCiOptions& options,
    const OrdinalProfileEvaluator& eval,
    const char* who) {
  if (!std::isfinite(estimate)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": estimate is not finite"));
  }
  if (auto ok = validate_ci_options_ordinal(options, estimate, who);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  auto cutoff = profile_ci_cutoff_ordinal(options, who);
  if (!cutoff.has_value()) return std::unexpected(cutoff.error());

  auto lower = solve_ordinal_profile_root_side(eval, estimate, options,
                                               *cutoff, true, who);
  if (!lower.has_value()) return std::unexpected(lower.error());
  auto upper = solve_ordinal_profile_root_side(eval, estimate, options,
                                               *cutoff, false, who);
  if (!upper.has_value()) return std::unexpected(upper.error());

  frontier::ScalarProfileCiResult out;
  out.lower_profile = std::move(lower->profile);
  out.upper_profile = std::move(upper->profile);
  out.estimate = estimate;
  out.lower = lower->root;
  out.upper = upper->root;
  out.confidence_level = options.confidence_level;
  out.cutoff = *cutoff;
  out.lower_cutoff =
      options.reference == frontier::ScalarProfileReference::MisspecMixture
          ? lower->profile.misspec_mixture_cutoff
          : *cutoff;
  out.upper_cutoff =
      options.reference == frontier::ScalarProfileReference::MisspecMixture
          ? upper->profile.misspec_mixture_cutoff
          : *cutoff;
  out.lower_evals = lower->evals;
  out.upper_evals = upper->evals;
  out.lower_at_bound = lower->at_bound;
  out.upper_at_bound = upper->at_bound;
  return out;
}

fit_expected<Eigen::MatrixXd>
ordinal_polychoric_model_correlation(
    const ThresholdLayout& layout,
    const model::Evaluation& eval,
    OrdinalParameterization parameterization,
    const char* who) {
  if (eval.moments.sigma.empty()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": fitted model has no implied covariance block"));
  }
  constexpr std::size_t b = 0;
  const Eigen::MatrixXd& Sig = eval.moments.sigma[b];
  const Eigen::Index p = Sig.rows();

  const bool theta_param = parameterization == OrdinalParameterization::Theta;
  const bool block_released =
      b < layout.scale_free.size() &&
      std::any_of(layout.scale_free[b].begin(), layout.scale_free[b].end(),
                  [](char c) { return c != 0; });
  if (!theta_param && block_released) {
    std::vector<std::int32_t> scale_free(static_cast<std::size_t>(p), 0);
    for (Eigen::Index i = 0; i < p; ++i) {
      if (static_cast<std::size_t>(i) < layout.scale_free[b].size()) {
        scale_free[static_cast<std::size_t>(i)] =
            layout.scale_free[b][static_cast<std::size_t>(i)];
      }
    }
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        R(i, j) = mixed_assoc_moment(
            Sig, scale_free, i, j, OrdinalParameterization::Theta);
        R(j, i) = R(i, j);
      }
    }
    return R;
  }

  auto R_or = ordinal_catml_correlation_matrix(
      Sig, parameterization,
      std::string(who) + " implied latent-response covariance");
  if (!R_or.has_value()) return std::unexpected(post_to_fit(R_or.error()));
  return std::move(*R_or);
}

fit_expected<double>
ordinal_polychoric_omega_value(
    const ThresholdLayout& layout,
    const model::ModelEvaluator& ev,
    const Eigen::VectorXd& theta,
    const measures::frontier::reliability::OmegaSpec& omega_spec,
    measures::frontier::reliability::OmegaTarget omega_target,
    OrdinalParameterization parameterization,
    const char* who) {
  namespace rel = measures::frontier::reliability;
  auto eval = ev.evaluate(theta, false, false);
  if (!eval.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": fitted evaluation failed: " +
            eval.error().detail));
  }
  auto R_or = ordinal_polychoric_model_correlation(
      layout, *eval, parameterization, who);
  if (!R_or.has_value()) return std::unexpected(R_or.error());
  auto omega_or = rel::omega_multidim(omega_target, *R_or, omega_spec);
  if (!omega_or.has_value()) return std::unexpected(post_to_fit(omega_or.error()));
  return *omega_or;
}

struct OrdinalPolychoricOmegaFunctional {
  frontier::ScalarFunctional functional;
  double unrestricted_value = 0.0;
};

struct OrdinalPolychoricOmegaState {
  ThresholdLayout layout;
  model::ModelEvaluator ev;
  measures::frontier::reliability::OmegaSpec omega_spec;
  measures::frontier::reliability::OmegaTarget omega_target =
      measures::frontier::reliability::OmegaTarget::Total;
  OrdinalParameterization parameterization = OrdinalParameterization::Delta;
  const char* who = "";
};

fit_expected<OrdinalPolychoricOmegaFunctional>
make_ordinal_polychoric_omega_functional(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Estimates& unrestricted,
    const measures::frontier::reliability::OmegaSpec& omega_spec,
    measures::frontier::reliability::OmegaTarget omega_target,
    OrdinalParameterization parameterization,
    const char* who) {
  if (stats.R.size() != 1) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": only single-group ordinal fits are supported"));
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  if (unrestricted.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": unrestricted theta size does not match "
            "prepared ordinal partable"));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }

  auto state = std::make_shared<OrdinalPolychoricOmegaState>(
      OrdinalPolychoricOmegaState{
          std::move(*layout_or), std::move(*ev_or), omega_spec, omega_target,
          parameterization, who});
  auto value = ordinal_polychoric_omega_value(
      state->layout, state->ev, unrestricted.theta, state->omega_spec,
      state->omega_target, state->parameterization, who);
  if (!value.has_value()) return std::unexpected(value.error());

  frontier::ScalarFunctional functional;
  functional.value = [state](const Eigen::VectorXd& theta) {
    auto v = ordinal_polychoric_omega_value(
        state->layout, state->ev, theta, state->omega_spec,
        state->omega_target, state->parameterization, state->who);
    return v.has_value() ? *v : std::numeric_limits<double>::quiet_NaN();
  };
  return OrdinalPolychoricOmegaFunctional{std::move(functional), *value};
}

fit_expected<double>
ordinal_scalar_profile_scaling_factor(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Estimates& constrained,
    const frontier::ScalarFunctional& functional,
    OrdinalWeightKind weights,
    OrdinalParameterization parameterization,
    const char* who) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": robust profile scaling requires ordinal stats "
            "with NACOV and per-case moment influence"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(post_to_fit(missing_or.error()));
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  if (constrained.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": constrained theta size does not match prepared "
            "ordinal partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(constrained.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constrained evaluation failed: " +
            eval.error().detail));
  }
  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              constrained.theta, parameterization, eval->J_mu);
  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(post_to_fit(Ws.error()));
  auto sw = ordinal_param_space_sandwich_ij(
      stats, *layout_or, eval->moments, constrained.theta, *Ws, Delta_full,
      weights, parameterization, *missing_or);
  if (!sw.has_value()) return std::unexpected(post_to_fit(sw.error()));

  auto con_or = build_eq_constraints(pt, true);
  if (!con_or.has_value()) return std::unexpected(post_to_fit(con_or.error()));
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != sw->A1.rows()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint reparameterization has incompatible "
            "shape for robust scaling"));
  }

  auto grad = scalar_gradient_ordinal(functional, constrained.theta,
                                      constrained.theta.size(), who);
  if (!grad.has_value()) return std::unexpected(grad.error());
  if (grad->size() != constrained.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar gradient length mismatch for robust "
            "profile scaling"));
  }

  const Eigen::MatrixXd A =
      (K.transpose() * sw->A1 * K).eval();
  const Eigen::MatrixXd B =
      (K.transpose() * sw->B1 * K).eval();
  const Eigen::VectorXd r = K.transpose() * *grad;
  return scalar_profile_scale_from_reduced_ordinal(A, B, r, who);
}

fit_expected<double>
ordinal_scalar_profile_misspec_scaling_factor(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Estimates& constrained,
    const frontier::ScalarFunctional& functional,
    OrdinalWeightKind weights,
    OrdinalParameterization parameterization,
    const char* who) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": misspec profile scaling requires ordinal stats "
            "with NACOV and per-case moment influence"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(post_to_fit(missing_or.error()));
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  if (constrained.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": constrained theta size does not match prepared "
            "ordinal partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(constrained.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constrained evaluation failed: " +
            eval.error().detail));
  }
  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              constrained.theta, parameterization, eval->J_mu);
  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(post_to_fit(Ws.error()));
  auto sw = ordinal_param_space_sandwich_ij(
      stats, *layout_or, eval->moments, constrained.theta, *Ws, Delta_full,
      weights, parameterization, *missing_or);
  if (!sw.has_value()) return std::unexpected(post_to_fit(sw.error()));

  auto con_or = build_eq_constraints(pt, true);
  if (!con_or.has_value()) return std::unexpected(post_to_fit(con_or.error()));
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != sw->B1.rows()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint reparameterization has incompatible "
            "shape for misspec profile scaling"));
  }

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, constrained, *layout_or, *Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(post_to_fit(ob.error()));
  auto grad = scalar_gradient_ordinal(functional, constrained.theta,
                                      constrained.theta.size(), who);
  if (!grad.has_value()) return std::unexpected(grad.error());
  const Eigen::MatrixXd B = (K.transpose() * sw->B1 * K).eval();
  const Eigen::VectorXd r = K.transpose() * *grad;
  return scalar_profile_scale_from_reduced_ordinal(*ob, B, r, who);
}

void fill_misspec_profile_ordinal(frontier::ScalarProfileLrtResult& out,
                                  double scale) {
  out.misspec_scaling_factor = scale;
  out.T_misspec_scaled = out.T / scale;
  out.p_value_misspec_scaled = chi2_pvalue(out.T_misspec_scaled, 1);
  out.misspec_eigvals = Eigen::VectorXd::Constant(1, scale);
  out.p_value_misspec_mixture =
      robust::weighted_chisq_upper(out.misspec_eigvals, out.T);
}

bool reference_needs_sandwich_ordinal(
    frontier::ScalarProfileReference reference) noexcept {
  return reference == frontier::ScalarProfileReference::RobustScaled ||
         reference == frontier::ScalarProfileReference::MisspecScaled ||
         reference == frontier::ScalarProfileReference::MisspecMixture;
}

bool reference_wants_robust_scaled_ordinal(
    frontier::ScalarProfileReference reference) noexcept {
  return reference == frontier::ScalarProfileReference::RobustScaled;
}

bool reference_wants_misspec_ordinal(
    frontier::ScalarProfileReference reference) noexcept {
  return reference == frontier::ScalarProfileReference::MisspecScaled ||
         reference == frontier::ScalarProfileReference::MisspecMixture;
}

fit_expected<double>
mixed_ordinal_scalar_profile_scaling_factor(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& constrained,
    const frontier::ScalarFunctional& functional,
    OrdinalWeightKind weights,
    OrdinalParameterization parameterization,
    bool observed_bread,
    const char* who) {
  auto parts = mixed_ordinal_rbm_parts(
      spec::LatentStructure(pt), rep, stats, constrained, weights,
      parameterization);
  if (!parts.has_value()) return std::unexpected(post_to_fit(parts.error()));
  const Eigen::MatrixXd& K = parts->K;
  if (K.rows() != constrained.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": mixed-ordinal profile sandwich K has "
            "incompatible shape"));
  }

  Eigen::MatrixXd A;
  if (observed_bread) {
    A = parts->information;
  } else {
    if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
      return std::unexpected(v.error());
    }
    if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
        !p.has_value()) {
      return std::unexpected(p.error());
    }
    if (constrained.theta.size() != pt.n_free()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          std::string(who) + ": constrained theta size does not match mixed "
              "delta partable"));
    }
    auto layout_or = make_threshold_layout(pt, rep, stats);
    if (!layout_or.has_value()) return std::unexpected(layout_or.error());
    auto ev_or = model::ModelEvaluator::build(pt, rep);
    if (!ev_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          std::string(who) + ": ModelEvaluator::build failed: " +
              ev_or.error().detail));
    }
    auto eval = ev_or->evaluate(constrained.theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": constrained evaluation failed: " +
              eval.error().detail));
    }
    const Eigen::MatrixXd Delta_full =
        mixed_moment_jacobian(stats, *layout_or, eval->moments,
                              eval->J_sigma, eval->J_mu,
                              constrained.theta, parameterization);
    if (Delta_full.cols() != K.rows()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": mixed-ordinal moment Jacobian/K shape "
              "mismatch"));
    }
    std::vector<Eigen::MatrixXd> uls_identity;
    const std::vector<Eigen::MatrixXd>* Ws = nullptr;
    if (weights == OrdinalWeightKind::ULS) {
      uls_identity.reserve(stats.NACOV.size());
      for (const auto& G : stats.NACOV) {
        uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
      }
      Ws = &uls_identity;
    } else {
      auto Ws_or = ordinal_sandwich_weights(stats, weights);
      if (!Ws_or.has_value()) return std::unexpected(post_to_fit(Ws_or.error()));
      uls_identity = std::move(*Ws_or);
      Ws = &uls_identity;
    }
    auto sw = ordinal_param_space_sandwich(stats, *Ws, Delta_full);
    if (!sw.has_value()) return std::unexpected(post_to_fit(sw.error()));
    const double N = static_cast<double>(parts->n_obs);
    A = N * (K.transpose() * sw->A1 * K).eval();
    A = 0.5 * (A + A.transpose()).eval();
  }

  Eigen::MatrixXd B = 0.5 * (parts->meat + parts->meat.transpose()).eval();
  auto grad = scalar_gradient_ordinal(functional, constrained.theta,
                                      constrained.theta.size(), who);
  if (!grad.has_value()) return std::unexpected(grad.error());
  const Eigen::VectorXd r = K.transpose() * *grad;
  return scalar_profile_scale_from_reduced_ordinal(A, B, r, who);
}

}  // namespace

namespace frontier {

fit_expected<Estimates>
fit_ordinal_constrained(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const data::OrdinalStats& stats,
                        const Estimates& start,
                        ExtraNonlinearEqConstraints extra,
                        Bounds bounds,
                        OrdinalWeightKind weights,
                        Backend backend,
                        OptimOptions opts,
                        OrdinalParameterization parameterization) {
  constexpr const char* who = "fit_ordinal_constrained";
  auto obj = ordinal_ls_objective(
      std::move(pt), rep, stats, start, weights, parameterization);
  if (!obj.has_value()) return std::unexpected(obj.error());

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(obj->pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != start.theta.size() ||
      bounds.upper.size() != start.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": bounds size mismatch"));
  }

  auto con_or = build_eq_constraints(obj->pt, true);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint: " + con_or.error().detail));
  }
  const NonlinearEqConstraints nl = build_nl_constraints(obj->pt);
  return solve_ordinal_ls_extra(
      obj->problem, start.theta, bounds, *con_or, nl, extra, backend, opts, who);
}

fit_expected<ScalarProfileLrtResult>
profile_lrt_scalar_ordinal(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::OrdinalStats& stats,
                           const Estimates& unrestricted,
                           ScalarFunctional functional,
                           double target,
                           Bounds bounds,
                           OrdinalWeightKind weights,
                           Backend backend,
                           OptimOptions opts,
                           OrdinalParameterization parameterization,
                           double constraint_tol,
                           bool robust_scaled,
                           ScalarProfileReference reference) {
  constexpr const char* who = "profile_lrt_scalar_ordinal";
  if (!functional.value) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar functional needs a value callback"));
  }
  if (!std::isfinite(target)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": target must be finite"));
  }
  if (!(constraint_tol > 0.0) || !std::isfinite(constraint_tol)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint_tol must be positive and finite"));
  }

  const double g_un = functional.value(unrestricted.theta);
  if (!std::isfinite(g_un)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar functional is not finite at theta_hat"));
  }
  auto grad0 = scalar_gradient_ordinal(
      functional, unrestricted.theta, unrestricted.theta.size(), who);
  if (!grad0.has_value()) return std::unexpected(grad0.error());

  ExtraNonlinearEqConstraints extra;
  extra.n_constraint = 1;
  extra.h = [functional, target](const Eigen::VectorXd& theta) {
    Eigen::VectorXd h(1);
    h(0) = functional.value(theta) - target;
    return h;
  };
  extra.jacobian = [functional](const Eigen::VectorXd& theta) {
    Eigen::MatrixXd J(1, theta.size());
    auto grad = scalar_gradient_ordinal(functional, theta, theta.size(), who);
    if (!grad.has_value()) {
      J.setConstant(std::numeric_limits<double>::quiet_NaN());
    } else {
      J.row(0) = grad->transpose();
    }
    return J;
  };

  auto constrained = fit_ordinal_constrained(
      spec::LatentStructure(pt), rep, stats, unrestricted, std::move(extra),
      std::move(bounds), weights, backend, opts, parameterization);
  if (!constrained.has_value()) return std::unexpected(constrained.error());

  const double g_con = functional.value(constrained->theta);
  const double resid = g_con - target;
  if (!std::isfinite(g_con) || std::abs(resid) > constraint_tol) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constrained fit did not satisfy g(theta)=target; "
            "residual=" + std::to_string(resid)));
  }
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(n_or.error());
  const double n = static_cast<double>(*n_or);
  const double diff = constrained->fmin - unrestricted.fmin;
  const double raw_T = 2.0 * n * diff;
  const double stat_tol =
      1e-10 * std::max({1.0, std::abs(constrained->fmin),
                        std::abs(unrestricted.fmin)}) * 2.0 * n;
  if (raw_T < -stat_tol) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constrained fit improved the unrestricted "
            "objective by " + std::to_string(-raw_T)));
  }
  const double T = std::max(0.0, raw_T);

  ScalarProfileLrtResult out;
  out.constrained = std::move(*constrained);
  out.unrestricted_value = g_un;
  out.constrained_value = g_con;
  out.target = target;
  out.constraint_residual = resid;
  out.fmin_unrestricted = unrestricted.fmin;
  out.fmin_constrained = out.constrained.fmin;
  out.T = T;
  out.p_value = chi2_pvalue(T, 1);
  out.n_obs = n;
  out.df = 1;
  if (robust_scaled && reference_wants_robust_scaled_ordinal(reference)) {
    auto scale = ordinal_scalar_profile_scaling_factor(
        spec::LatentStructure(pt), rep, stats, out.constrained, functional, weights,
        parameterization, who);
    if (!scale.has_value()) return std::unexpected(scale.error());
    out.scaling_factor = *scale;
    out.T_scaled = out.T / *scale;
    out.p_value_scaled = chi2_pvalue(out.T_scaled, 1);
  }
  if (robust_scaled && reference_wants_misspec_ordinal(reference)) {
    auto misspec_scale = ordinal_scalar_profile_misspec_scaling_factor(
        std::move(pt), rep, stats, out.constrained, functional, weights,
        parameterization, who);
    if (!misspec_scale.has_value()) {
      return std::unexpected(misspec_scale.error());
    }
    fill_misspec_profile_ordinal(out, *misspec_scale);
  }
  return out;
}

fit_expected<ScalarProfileLrtResult>
profile_lrt_parameter_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Estimates& unrestricted,
    Eigen::Index parameter,
    double target,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization,
    double constraint_tol,
    bool robust_scaled,
    ScalarProfileReference reference) {
  constexpr const char* who = "profile_lrt_parameter_ordinal";
  if (parameter < 0 || parameter >= unrestricted.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": parameter index out of range"));
  }
  ScalarFunctional functional;
  functional.value = [parameter](const Eigen::VectorXd& theta) {
    return theta(parameter);
  };
  functional.gradient = [parameter](const Eigen::VectorXd& theta) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(theta.size());
    grad(parameter) = 1.0;
    return grad;
  };
  return profile_lrt_scalar_ordinal(
      std::move(pt), rep, stats, unrestricted, std::move(functional), target,
      std::move(bounds), weights, backend, opts, parameterization,
      constraint_tol, robust_scaled, reference);
}

fit_expected<ScalarProfileCiResult>
profile_lrt_ci_parameter_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Estimates& unrestricted,
    Eigen::Index parameter,
    ScalarProfileCiOptions ci_options,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization,
    double constraint_tol,
    bool robust_scaled) {
  constexpr const char* who = "profile_lrt_ci_parameter_ordinal";
  if (parameter < 0 || parameter >= unrestricted.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": parameter index out of range"));
  }
  const double estimate = unrestricted.theta(parameter);
  const bool use_sandwich_reference =
      reference_needs_sandwich_ordinal(ci_options.reference);
  OrdinalProfileEvaluator eval =
      [pt, &rep, &stats, unrestricted, parameter, bounds, weights, backend, opts,
       parameterization, constraint_tol, robust_scaled,
       use_sandwich_reference,
       reference = ci_options.reference](double target) mutable {
        spec::LatentStructure pt_eval = pt;
        Bounds bounds_eval = bounds;
        return profile_lrt_parameter_ordinal(
            std::move(pt_eval), rep, stats, unrestricted, parameter, target,
            std::move(bounds_eval), weights, backend, opts, parameterization,
            constraint_tol, robust_scaled || use_sandwich_reference,
            reference);
      };
  return ordinal_profile_ci_from_evaluator(estimate, ci_options, eval, who);
}

fit_expected<ScalarProfileLrtResult>
profile_lrt_ordinal_polychoric_omega(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Estimates& unrestricted,
    const measures::frontier::reliability::OmegaSpec& omega_spec,
    measures::frontier::reliability::OmegaTarget omega_target,
    double target,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization,
    double constraint_tol,
    bool robust_scaled,
    ScalarProfileReference reference) {
  constexpr const char* who = "profile_lrt_ordinal_polychoric_omega";
  auto fn = make_ordinal_polychoric_omega_functional(
      pt, rep, stats, unrestricted, omega_spec, omega_target,
      parameterization, who);
  if (!fn.has_value()) return std::unexpected(fn.error());
  return profile_lrt_scalar_ordinal(
      std::move(pt), rep, stats, unrestricted, std::move(fn->functional),
      target, std::move(bounds), weights, backend, opts, parameterization,
      constraint_tol, robust_scaled, reference);
}

fit_expected<ScalarProfileCiResult>
profile_lrt_ci_ordinal_polychoric_omega(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Estimates& unrestricted,
    const measures::frontier::reliability::OmegaSpec& omega_spec,
    measures::frontier::reliability::OmegaTarget omega_target,
    ScalarProfileCiOptions ci_options,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization,
    double constraint_tol,
    bool robust_scaled) {
  constexpr const char* who = "profile_lrt_ci_ordinal_polychoric_omega";
  auto fn = make_ordinal_polychoric_omega_functional(
      pt, rep, stats, unrestricted, omega_spec, omega_target,
      parameterization, who);
  if (!fn.has_value()) return std::unexpected(fn.error());

  const double estimate = fn->unrestricted_value;
  OrdinalProfileEvaluator eval =
      [pt, &rep, &stats, unrestricted, functional = fn->functional, bounds,
       weights, backend, opts, parameterization, constraint_tol, robust_scaled,
       reference = ci_options.reference](
          double target_value) mutable {
        spec::LatentStructure pt_eval = pt;
        frontier::ScalarFunctional fn_eval = functional;
        Bounds bounds_eval = bounds;
        return profile_lrt_scalar_ordinal(
            std::move(pt_eval), rep, stats, unrestricted, std::move(fn_eval),
            target_value, std::move(bounds_eval), weights, backend, opts,
            parameterization, constraint_tol, robust_scaled ||
                reference_needs_sandwich_ordinal(reference),
            reference);
      };
  return ordinal_profile_ci_from_evaluator(estimate, ci_options, eval, who);
}

fit_expected<Estimates>
fit_mixed_ordinal_constrained(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& start,
    ExtraNonlinearEqConstraints extra,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization) {
  constexpr const char* who = "fit_mixed_ordinal_constrained";
  auto obj = mixed_ordinal_ls_objective(
      std::move(pt), rep, stats, start, weights, parameterization);
  if (!obj.has_value()) return std::unexpected(obj.error());

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(obj->pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(who) + ": bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != start.theta.size() ||
      bounds.upper.size() != start.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": bounds size mismatch"));
  }

  auto con_or = build_eq_constraints(obj->pt, true);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint: " + con_or.error().detail));
  }
  const NonlinearEqConstraints nl = build_nl_constraints(obj->pt);
  return solve_ordinal_ls_extra(
      obj->problem, start.theta, bounds, *con_or, nl, extra, backend, opts,
      who);
}

fit_expected<ScalarProfileLrtResult>
profile_lrt_scalar_mixed_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& unrestricted,
    ScalarFunctional functional,
    double target,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization,
    double constraint_tol,
    bool robust_scaled,
    ScalarProfileReference reference) {
  constexpr const char* who = "profile_lrt_scalar_mixed_ordinal";
  if (!functional.value) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar functional needs a value callback"));
  }
  if (!std::isfinite(target)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": target must be finite"));
  }
  if (!(constraint_tol > 0.0) || !std::isfinite(constraint_tol)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint_tol must be positive and finite"));
  }
  const double g_un = functional.value(unrestricted.theta);
  if (!std::isfinite(g_un)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": scalar functional is not finite at theta_hat"));
  }
  auto grad0 = scalar_gradient_ordinal(
      functional, unrestricted.theta, unrestricted.theta.size(), who);
  if (!grad0.has_value()) return std::unexpected(grad0.error());

  ExtraNonlinearEqConstraints extra;
  extra.n_constraint = 1;
  extra.h = [functional, target](const Eigen::VectorXd& theta) {
    Eigen::VectorXd h(1);
    h(0) = functional.value(theta) - target;
    return h;
  };
  extra.jacobian = [functional](const Eigen::VectorXd& theta) {
    Eigen::MatrixXd J(1, theta.size());
    auto grad = scalar_gradient_ordinal(functional, theta, theta.size(), who);
    if (!grad.has_value()) {
      J.setConstant(std::numeric_limits<double>::quiet_NaN());
    } else {
      J.row(0) = grad->transpose();
    }
    return J;
  };

  auto constrained = fit_mixed_ordinal_constrained(
      spec::LatentStructure(pt), rep, stats, unrestricted, std::move(extra),
      std::move(bounds), weights, backend, opts, parameterization);
  if (!constrained.has_value()) return std::unexpected(constrained.error());

  const double g_con = functional.value(constrained->theta);
  const double resid = g_con - target;
  if (!std::isfinite(g_con) || std::abs(resid) > constraint_tol) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constrained fit did not satisfy g(theta)=target; "
            "residual=" + std::to_string(resid)));
  }
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(n_or.error());
  const double n = static_cast<double>(*n_or);
  const double diff = constrained->fmin - unrestricted.fmin;
  const double raw_T = 2.0 * n * diff;
  const double stat_tol =
      1e-10 * std::max({1.0, std::abs(constrained->fmin),
                        std::abs(unrestricted.fmin)}) * 2.0 * n;
  if (raw_T < -stat_tol) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constrained fit improved the unrestricted "
            "objective by " + std::to_string(-raw_T)));
  }

  ScalarProfileLrtResult out;
  out.constrained = std::move(*constrained);
  out.unrestricted_value = g_un;
  out.constrained_value = g_con;
  out.target = target;
  out.constraint_residual = resid;
  out.fmin_unrestricted = unrestricted.fmin;
  out.fmin_constrained = out.constrained.fmin;
  out.T = std::max(0.0, raw_T);
  out.p_value = chi2_pvalue(out.T, 1);
  out.n_obs = n;
  out.df = 1;
  if (robust_scaled && reference_wants_robust_scaled_ordinal(reference)) {
    auto scale = mixed_ordinal_scalar_profile_scaling_factor(
        spec::LatentStructure(pt), rep, stats, out.constrained, functional,
        weights, parameterization, /*observed_bread=*/false, who);
    if (!scale.has_value()) return std::unexpected(scale.error());
    out.scaling_factor = *scale;
    out.T_scaled = out.T / *scale;
    out.p_value_scaled = chi2_pvalue(out.T_scaled, 1);
  }
  if (robust_scaled && reference_wants_misspec_ordinal(reference)) {
    auto scale = mixed_ordinal_scalar_profile_scaling_factor(
        std::move(pt), rep, stats, out.constrained, functional, weights,
        parameterization, /*observed_bread=*/true, who);
    if (!scale.has_value()) return std::unexpected(scale.error());
    fill_misspec_profile_ordinal(out, *scale);
  }
  return out;
}

fit_expected<ScalarProfileLrtResult>
profile_lrt_parameter_mixed_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& unrestricted,
    Eigen::Index parameter,
    double target,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization,
    double constraint_tol,
    bool robust_scaled,
    ScalarProfileReference reference) {
  constexpr const char* who = "profile_lrt_parameter_mixed_ordinal";
  if (parameter < 0 || parameter >= unrestricted.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": parameter index out of range"));
  }
  ScalarFunctional functional;
  functional.value = [parameter](const Eigen::VectorXd& theta) {
    return theta(parameter);
  };
  functional.gradient = [parameter](const Eigen::VectorXd& theta) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(theta.size());
    grad(parameter) = 1.0;
    return grad;
  };
  return profile_lrt_scalar_mixed_ordinal(
      std::move(pt), rep, stats, unrestricted, std::move(functional), target,
      std::move(bounds), weights, backend, opts, parameterization,
      constraint_tol, robust_scaled, reference);
}

fit_expected<ScalarProfileCiResult>
profile_lrt_ci_parameter_mixed_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& unrestricted,
    Eigen::Index parameter,
    ScalarProfileCiOptions ci_options,
    Bounds bounds,
    OrdinalWeightKind weights,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization,
    double constraint_tol,
    bool robust_scaled) {
  constexpr const char* who = "profile_lrt_ci_parameter_mixed_ordinal";
  if (parameter < 0 || parameter >= unrestricted.theta.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(who) + ": parameter index out of range"));
  }
  const double estimate = unrestricted.theta(parameter);
  OrdinalProfileEvaluator eval =
      [pt, &rep, &stats, unrestricted, parameter, bounds, weights, backend,
       opts, parameterization, constraint_tol, robust_scaled,
       reference = ci_options.reference](double target) mutable {
        spec::LatentStructure pt_eval = pt;
        Bounds bounds_eval = bounds;
        return profile_lrt_parameter_mixed_ordinal(
            std::move(pt_eval), rep, stats, unrestricted, parameter, target,
            std::move(bounds_eval), weights, backend, opts, parameterization,
            constraint_tol,
            robust_scaled || reference_needs_sandwich_ordinal(reference),
            reference);
      };
  return ordinal_profile_ci_from_evaluator(estimate, ci_options, eval, who);
}

}  // namespace frontier

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    const Eigen::VectorXd& x0,
                    Backend backend,
                    OptimOptions opts,
                    OrdinalParameterization parameterization,
                    const std::vector<std::int8_t>* row_user) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr, row_user);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: bounds size mismatch"));
  }

  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  const EqConstraints& con = *con_or;

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                              parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = x0.size();
  prob.expand  = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return ordinal_residuals(stats, layout, eval->moments, factors, x,
                             parameterization);
  };
  prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(x, true, true);  // J_mu: released-intercept gradient
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                            factors, x, parameterization, eval->J_mu);
  };
  // `optim::scalarize` falls back to calling `r` then `J` separately when
  // `eval` is unset, which evaluates the model and re-whitens the moments twice
  // per gradient. One evaluation serves both.
  prob.eval = [&](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto r = ordinal_residuals(stats, layout, eval->moments, factors, x,
                               parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                              factors, x, parameterization, eval->J_mu);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto est = solve_ordinal_ls(prob, x0, bounds, con, backend, opts,
                              "fit_ordinal_bounded");
  if (!est.has_value()) return est;
  attach_ordinal_geometric_diagnostics(
      *est, pt, ev, con, bounds, prob, OrdinalNewtonContext{&rep, &stats, nullptr, &layout, &factors, parameterization});
  return est;
}

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalMoments& moments,
                    data::OrdinalGammaCache* gamma_cache,
                    Bounds bounds,
                    data::OrdinalWeightPlan plan,
                    const Eigen::VectorXd& x0,
                    Backend backend,
                    OptimOptions opts) {
  const OrdinalParameterization parameterization =
      to_estimate_parameterization(plan.parameterization);
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  data::OrdinalStats stats = stats_adapter(moments);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }
  if (parameterization == OrdinalParameterization::Theta) {
    if (bounds.empty()) {
      auto b_or = bounds_from_partable(pt);
      if (!b_or.has_value()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_bounded: bounds_from_partable failed: " +
                b_or.error().detail));
      }
      bounds = std::move(*b_or);
    }
    if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: bounds size mismatch"));
    }
    auto factors_or = full_weight_factors(moments, gamma_cache, plan);
    if (!factors_or.has_value()) return std::unexpected(factors_or.error());
    const auto& factors = *factors_or;

    auto con_or = build_eq_constraints(pt);
    if (!con_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "constraint: " + con_or.error().detail));
    }
    const EqConstraints& con = *con_or;

    auto eval0 = ev.evaluate(x0, false, false);
    if (!eval0.has_value()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fit_ordinal_bounded: start evaluation failed: " +
              eval0.error().detail));
    }
    auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                                parameterization);
    if (!r0.has_value()) return std::unexpected(r0.error());

    optim::GmmProblem prob;
    prob.n_resid = r0->size();
    prob.n_param = x0.size();
    prob.expand = [](const Eigen::VectorXd& x) { return x; };
    prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
      auto eval = ev.evaluate(x, false, false);
      if (!eval.has_value()) {
        return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
            "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
      }
      return ordinal_residuals(stats, layout, eval->moments, factors, x,
                               parameterization);
    };
    prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
      auto eval = ev.evaluate(x, true, true);  // J_mu: released-intercept gradient
      if (!eval.has_value()) {
        return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
            "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
      }
      return ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                              factors, x, parameterization, eval->J_mu);
    };
    prob.eval =
        [&](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
      auto eval = ev.evaluate(x, true, true);
      if (!eval.has_value()) {
        return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
            "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
      }
      auto r = ordinal_residuals(stats, layout, eval->moments, factors, x,
                                 parameterization);
      if (!r.has_value()) return std::unexpected(r.error());
      auto J = ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                                factors, x, parameterization, eval->J_mu);
      if (!J.has_value()) return std::unexpected(J.error());
      return optim::LsEvaluation{std::move(*r), std::move(*J)};
    };

    auto est = solve_ordinal_ls(prob, x0, bounds, con, backend, opts,
                                 "fit_ordinal_bounded");
    if (!est.has_value()) return est;
    attach_ordinal_geometric_diagnostics(*est, pt, ev, con, bounds, prob,
                                         OrdinalNewtonContext{&rep, &stats, nullptr, &layout, &factors, parameterization});
    return est;
  }
  auto profile_or = make_threshold_design(pt, layout, stats, x0);
  if (!profile_or.has_value()) return std::unexpected(profile_or.error());
  ThresholdDesign profile = std::move(*profile_or);

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: bounds size mismatch"));
  }
  Bounds active_bounds = profile_bounds(profile, bounds);

  auto weights_or =
      profiled_weight_workspace(moments, layout, profile, gamma_cache, plan);
  if (!weights_or.has_value()) return std::unexpected(weights_or.error());
  const auto& profiled_weights = *weights_or;

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  if (con_or->active()) {
    auto constraint_check = ensure_no_unprofiled_equality_constraints(pt, profile);
    if (!constraint_check.has_value()) {
      return std::unexpected(constraint_check.error());
    }
  }

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = profiled_ordinal_residuals(moments, eval0->moments,
                                       profiled_weights);
  if (!r0.has_value()) return std::unexpected(r0.error());

  Eigen::VectorXd active_x0 = profile_contract(profile, x0);
  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = active_x0.size();
  prob.expand  = [profile](const Eigen::VectorXd& x) {
    return profile_expand(profile, x);
  };
  prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    const Eigen::VectorXd theta = profile_expand(profile, x);
    auto eval = ev.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return profiled_ordinal_residuals(moments, eval->moments,
                                      profiled_weights);
  };
  prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    const Eigen::VectorXd theta = profile_expand(profile, x);
    auto eval = ev.evaluate(theta, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto J_full = profiled_ordinal_jacobian(moments, eval->moments,
                                            eval->J_sigma, profiled_weights);
    if (!J_full.has_value()) return std::unexpected(J_full.error());
    return profile_jacobian(profile, *J_full);
  };
  // Without this the scalarized gradient evaluates the model twice and whitens
  // the moments twice; this is the default weighted-ordinal fit path.
  prob.eval = [&](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    const Eigen::VectorXd theta = profile_expand(profile, x);
    auto eval = ev.evaluate(theta, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto r = profiled_ordinal_residuals(moments, eval->moments,
                                        profiled_weights);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J_full = profiled_ordinal_jacobian(moments, eval->moments,
                                            eval->J_sigma, profiled_weights);
    if (!J_full.has_value()) return std::unexpected(J_full.error());
    return optim::LsEvaluation{std::move(*r),
                               profile_jacobian(profile, *J_full)};
  };

  auto est = solve_ordinal_ls(prob, active_x0, active_bounds, EqConstraints{},
                              backend, opts,
                              "fit_ordinal_bounded");
  if (!est.has_value()) return std::unexpected(est.error());
  auto eval_hat = ev.evaluate(est->theta, false, false);
  if (!eval_hat.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "fit_ordinal_bounded: final reconstruction evaluate failed: " +
            eval_hat.error().detail));
  }
  auto theta_or = reconstruct_profiled_thresholds(
      stats, layout, eval_hat->moments, profiled_weights, est->theta);
  if (!theta_or.has_value()) return std::unexpected(theta_or.error());
  est->theta = std::move(*theta_or);
  auto factors = full_weight_factors(moments, gamma_cache, plan);
  if (factors.has_value()) {
    attach_reconstructed_ordinal_diagnostics(
        *est, pt, ev, stats, layout, *factors, bounds, parameterization, &rep);
  }
  return est;
}

namespace {

// Deliberately narrow eligibility: other charts and threshold constraints
// retain the general GP implementation. Covariance parameters stay nonlinear.
fit_expected<std::optional<Estimates>> fit_theta_free_thresholds(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::OrdinalMoments& moments, data::OrdinalGammaCache* cache,
    data::OrdinalWeightPlan plan, const Eigen::VectorXd& x0,
    Backend backend, OptimOptions opts) {
  if (auto valid = validate_moments(moments, rep); !valid.has_value())
    return std::unexpected(valid.error());
  auto stats = stats_adapter(moments);
  if (auto prep = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !prep.has_value()) return std::unexpected(prep.error());
  if (x0.size() != pt.n_free())
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "theta threshold profile: start dimension mismatch"));
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const auto& layout = *layout_or;
  auto con = build_eq_constraints(pt);
  if (!con.has_value()) return std::unexpected(make_err(FitError::Kind::NumericIssue, con.error().detail));
  if (con->active() || build_nl_constraints(pt).active()) return std::nullopt;
  auto bounds = bounds_from_partable(pt);
  if (!bounds.has_value()) return std::unexpected(make_err(FitError::Kind::NumericIssue, bounds.error().detail));
  std::vector<char> seen(static_cast<std::size_t>(pt.n_free()), 0);
  Eigen::Index nth_total = 0;
  for (const auto& block : layout.free) {
    for (const auto fr : block) {
      if (fr <= 0 || fr > pt.n_free()) return std::nullopt;
      const auto k = static_cast<std::size_t>(fr - 1);
      if (seen[k] || (!bounds->empty() &&
          (std::isfinite(bounds->lower(fr - 1)) ||
           std::isfinite(bounds->upper(fr - 1))))) return std::nullopt;
      seen[k] = 1;
      ++nth_total;
    }
  }
  if (nth_total == 0) return std::nullopt;
  auto fixed = fix_thresholds_for_snlls(pt, layout, stats, x0);
  if (!fixed.has_value()) return std::unexpected(fixed.error());
  auto ev = model::ModelEvaluator::build(fixed->pt, rep);
  if (!ev.has_value()) return std::unexpected(make_err(
      FitError::Kind::NumericIssue, ev.error().detail));
  auto factors = full_weight_factors(moments, cache, plan);
  if (!factors.has_value()) return std::unexpected(factors.error());
  auto N = total_n_obs(moments);
  if (!N.has_value()) return std::unexpected(N.error());
  std::vector<detail::ThetaThresholdProfile> weights;
  Eigen::Index nr = 0;
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    auto w = detail::theta_threshold_profile((*factors)[b],
        moments.thresholds[b].size(),
        plan.estimator != data::OrdinalEstimatorKind::WLS);
    if (!w.has_value()) return std::unexpected(w.error());
    w->factor.scale(std::sqrt(static_cast<double>(moments.n_obs[b]) /
                              static_cast<double>(*N)));
    nr += w->factor.rows();
    weights.push_back(std::move(*w));
  }
  auto evaluate = [&](const Eigen::VectorXd& x, bool jacobian)
      -> fit_expected<optim::LsEvaluation> {
    auto e = ev->evaluate(x, jacobian, false);
    if (!e.has_value()) return std::unexpected(make_err(
        FitError::Kind::NonPositiveDefiniteSigma, e.error().detail));
    optim::LsEvaluation out;
    out.residual.resize(nr);
    if (jacobian) out.jacobian.resize(nr, x.size());
    Eigen::Index off = 0, sigma_off = 0;
    for (std::size_t b = 0; b < moments.R.size(); ++b) {
      const auto& S = e->moments.sigma[b];
      if (!S.diagonal().allFinite() || (S.diagonal().array() <= 0).any())
        return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
            "theta threshold profile: nonpositive response variance"));
      const auto nc = weights[b].factor.rows();
      out.residual.segment(off, nc) = weights[b].factor.apply(
          (std_corr_lower(S) - corr_lower(moments.R[b])).eval());
      if (jacobian) out.jacobian.middleRows(off, nc) = weights[b].factor.apply(
          std_corr_jacobian(S, e->J_sigma, sigma_off));
      sigma_off += vech_len(S.rows());
      off += nc;
    }
    if (!out.residual.allFinite() || (jacobian && !out.jacobian.allFinite()))
      return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
          "theta threshold profile: nonfinite residual or Jacobian"));
    return out;
  };
  optim::GmmProblem prob;
  prob.n_resid = nr;
  prob.n_param = fixed->x0.size();
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.eval = [&](const Eigen::VectorXd& x) { return evaluate(x, true); };
  prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto e = evaluate(x, false);
    if (!e.has_value()) return std::unexpected(e.error());
    return std::move(e->residual);
  };
  prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto e = evaluate(x, true);
    if (!e.has_value()) return std::unexpected(e.error());
    return std::move(e->jacobian);
  };
  fit_expected<Estimates> est = Estimates{};
  if (prob.n_param == 0) {
    auto r = prob.r(fixed->x0);
    if (!r.has_value()) return std::unexpected(r.error());
    est->theta = fixed->x0;
    est->fmin = 0.5 * r->squaredNorm();
  } else {
    est = solve_ordinal_ls(prob, fixed->x0, Bounds{}, EqConstraints{},
        backend, opts, "theta threshold profile");
  }
  if (!est.has_value()) return std::unexpected(est.error());
  auto e = ev->evaluate(est->theta, false, false);
  if (!e.has_value()) return std::unexpected(make_err(
      FitError::Kind::NumericIssue, e.error().detail));
  auto full = expand_threshold_fixed_theta(*fixed, est->theta);
  if (!full.has_value()) return std::unexpected(full.error());
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const auto& S = e->moments.sigma[b];
    const Eigen::VectorXd z = moments.thresholds[b] -
        weights[b].threshold_from_corr *
        (std_corr_lower(S) - corr_lower(moments.R[b]));
    for (Eigen::Index k = 0; k < z.size(); ++k) {
      const auto ov = stats.threshold_ov[b][static_cast<std::size_t>(k)];
      const double mu = b < e->moments.mu.size() && e->moments.mu[b].size() > 0
          ? e->moments.mu[b](ov) : 0.0;
      (*full)(layout.free[b][static_cast<std::size_t>(k)] - 1) =
          mu + std::sqrt(S(ov, ov)) * z(k);
    }
  }
  est->theta = std::move(*full);
  est->n_nonlinear = static_cast<std::int32_t>(fixed->x0.size());
  est->n_linear = static_cast<std::int32_t>(nth_total);
  auto ev_full = model::ModelEvaluator::build(pt, rep);
  if (!ev_full.has_value()) return std::unexpected(make_err(
      FitError::Kind::NumericIssue, ev_full.error().detail));
  attach_reconstructed_ordinal_diagnostics(*est, pt, *ev_full, stats, layout,
      *factors, Bounds{}, OrdinalParameterization::Theta, &rep);
  return std::optional<Estimates>{std::move(*est)};
}

}  // namespace

fit_expected<Estimates>
fit_ordinal_snlls(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalMoments& moments,
                  data::OrdinalGammaCache* gamma_cache,
                  data::OrdinalWeightPlan plan,
                  const Eigen::VectorXd& x0,
                  Backend backend,
                  OptimOptions opts) {
  const OrdinalParameterization parameterization =
      to_estimate_parameterization(plan.parameterization);
  if (parameterization == OrdinalParameterization::Theta) {
    auto fast = fit_theta_free_thresholds(pt, rep, moments, gamma_cache, plan,
                                         x0, backend, opts);
    if (!fast.has_value()) return std::unexpected(fast.error());
    if (fast->has_value()) return std::move(**fast);
    return fit_ordinal_snlls_full_thresholds(
        std::move(pt), rep, moments, gamma_cache, plan, x0, backend, opts);
  }
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }

  data::OrdinalStats stats = stats_adapter(moments);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;
  if (auto valid = validate_ordinal_snlls_chart(layout, parameterization);
      !valid.has_value()) return std::unexpected(valid.error());

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  auto threshold_fixed_or = fix_thresholds_for_snlls(pt, layout, stats, x0);
  if (!threshold_fixed_or.has_value()) {
    return std::unexpected(threshold_fixed_or.error());
  }
  ThresholdFixedProfile threshold_fixed = std::move(*threshold_fixed_or);

  auto profiled_weights_or = profiled_weight_workspace(
      moments, layout, threshold_fixed.design, gamma_cache, plan);
  if (!profiled_weights_or.has_value()) {
    return std::unexpected(profiled_weights_or.error());
  }
  const ProfiledWeightWorkspace& profiled_weights = *profiled_weights_or;

  auto ev_reduced_or = model::ModelEvaluator::build(threshold_fixed.pt, rep);
  if (!ev_reduced_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: ModelEvaluator::build failed: " +
            ev_reduced_or.error().detail));
  }
  auto ev_full_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_full_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: full ModelEvaluator::build failed: " +
            ev_full_or.error().detail));
  }
  auto ev_reduced = std::move(*ev_reduced_or);
  auto ev_full = std::move(*ev_full_or);

  auto eval0 = ev_reduced.evaluate(threshold_fixed.x0, true, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = profiled_ordinal_residuals(moments, eval0->moments,
                                       profiled_weights);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem base;
  base.n_resid = r0->size();
  base.n_param = threshold_fixed.x0.size();
  base.expand = [](const Eigen::VectorXd& x) { return x; };
  base.r = [&](const Eigen::VectorXd& theta)
      -> fit_expected<Eigen::VectorXd> {
    auto eval = ev_reduced.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls: evaluate failed: " + eval.error().detail));
    }
    return profiled_ordinal_residuals(moments, eval->moments,
                                      profiled_weights);
  };
  base.J = [&](const Eigen::VectorXd& theta)
      -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev_reduced.evaluate(theta, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls: evaluate failed: " + eval.error().detail));
    }
    return profiled_ordinal_jacobian(moments, eval->moments, eval->J_sigma,
                                     profiled_weights);
  };
  base.eval = [&](const Eigen::VectorXd& theta)
      -> fit_expected<optim::LsEvaluation> {
    auto eval = ev_reduced.evaluate(theta, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls: evaluate failed: " + eval.error().detail));
    }
    auto r = profiled_ordinal_residuals(moments, eval->moments,
                                        profiled_weights);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = profiled_ordinal_jacobian(moments, eval->moments, eval->J_sigma,
                                       profiled_weights);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto gp_or = gmm::gp(base, threshold_fixed.pt, ev_reduced,
                      threshold_fixed.x0);
  if (!gp_or.has_value()) return std::unexpected(gp_or.error());
  const optim::GmmProblem& prob = gp_or->problem;

  Eigen::VectorXd theta_reduced;
  double fmin = 0.0;
  int iterations = 0;
  int f_evals = 0;
  int g_evals = 0;
  optim::OptimStatus status = optim::OptimStatus::Converged;
  double grad_inf_norm = -1.0;
  optim::TerminalAudit audit;

  if (prob.n_param == 0) {
    auto r = prob.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    theta_reduced = prob.expand(Eigen::VectorXd(0));
    fmin = 0.5 * r->squaredNorm();
  } else {
    auto out = run_ordinal_ls(prob, gp_or->beta0, Bounds{}, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    theta_reduced = prob.expand(out->x);
    fmin = out->fmin;
    iterations = out->iterations;
    f_evals = out->f_evals;
    g_evals = out->g_evals;
    status = out->status;
    grad_inf_norm = out->grad_inf_norm;
    audit = std::move(out->audit);
  }

  auto theta_full_or =
      expand_threshold_fixed_theta(threshold_fixed, theta_reduced);
  if (!theta_full_or.has_value()) return std::unexpected(theta_full_or.error());
  auto eval_hat = ev_full.evaluate(*theta_full_or, false, false);
  if (!eval_hat.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "fit_ordinal_snlls: final reconstruction evaluate failed: " +
            eval_hat.error().detail));
  }
  auto theta_final_or = reconstruct_profiled_thresholds(
      stats, layout, eval_hat->moments, profiled_weights, *theta_full_or);
  if (!theta_final_or.has_value()) {
    return std::unexpected(theta_final_or.error());
  }
  Estimates est{std::move(*theta_final_or), fmin, iterations, f_evals,
                g_evals, status, grad_inf_norm, std::move(audit), {},
                gp_or->n_nonlinear, gp_or->n_linear};
  auto factors = full_weight_factors(moments, gamma_cache, plan);
  if (factors.has_value()) {
    attach_reconstructed_ordinal_diagnostics(
        est, pt, ev_full, stats, layout, *factors, Bounds{}, parameterization, &rep);
  }
  return est;
}

fit_expected<Estimates>
fit_ordinal_snlls_full_thresholds(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const data::OrdinalMoments& moments,
                                  data::OrdinalGammaCache* gamma_cache,
                                  data::OrdinalWeightPlan plan,
                                  const Eigen::VectorXd& x0,
                                  Backend backend,
                                  OptimOptions opts) {
  const OrdinalParameterization parameterization =
      to_estimate_parameterization(plan.parameterization);
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }

  data::OrdinalStats stats = stats_adapter(moments);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;
  if (auto valid = validate_ordinal_snlls_chart(layout, parameterization);
      !valid.has_value()) return std::unexpected(valid.error());

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls_full_thresholds: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls_full_thresholds: x0 size (" +
            std::to_string(x0.size()) + ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  auto factors_or = full_weight_factors(moments, gamma_cache, plan);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto block_kinds_or =
      ordinal_gp_block_kinds(pt, layout, ev, parameterization);
  if (!block_kinds_or.has_value()) return std::unexpected(block_kinds_or.error());
  const auto& block_kinds = *block_kinds_or;

  auto eval0 = ev.evaluate(x0, true, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls_full_thresholds: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                              parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem base;
  base.n_resid = r0->size();
  base.n_param = x0.size();
  base.expand = [](const Eigen::VectorXd& x) { return x; };
  base.r = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return ordinal_residuals(stats, layout, eval->moments, factors, theta,
                             parameterization);
  };
  base.J = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(theta, true, true);  // J_mu: released-intercept gradient
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                            factors, theta, parameterization, eval->J_mu);
  };
  base.eval = [&](const Eigen::VectorXd& theta)
      -> fit_expected<optim::LsEvaluation> {
    auto eval = ev.evaluate(theta, true, true);  // J_mu: released-intercept gradient
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    auto r = ordinal_residuals(stats, layout, eval->moments, factors, theta,
                               parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                              factors, theta, parameterization, eval->J_mu);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto gp_or = gmm::gp(base, pt, ev, x0, block_kinds);
  if (!gp_or.has_value()) return std::unexpected(gp_or.error());
  const optim::GmmProblem& prob = gp_or->problem;

  Eigen::VectorXd theta_hat;
  double fmin = 0.0;
  int iterations = 0;
  int f_evals = 0;
  int g_evals = 0;
  optim::OptimStatus status = optim::OptimStatus::Converged;
  double grad_inf_norm = -1.0;
  optim::TerminalAudit audit;

  if (prob.n_param == 0) {
    auto r = prob.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    theta_hat = prob.expand(Eigen::VectorXd(0));
    fmin = 0.5 * r->squaredNorm();
  } else {
    auto out = run_ordinal_ls(prob, gp_or->beta0, Bounds{}, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    theta_hat = prob.expand(out->x);
    fmin = out->fmin;
    iterations = out->iterations;
    f_evals = out->f_evals;
    g_evals = out->g_evals;
    status = out->status;
    grad_inf_norm = out->grad_inf_norm;
    audit = std::move(out->audit);
  }

  Estimates est{std::move(theta_hat), fmin, iterations, f_evals, g_evals,
                status, grad_inf_norm, std::move(audit), {},
                gp_or->n_nonlinear, gp_or->n_linear};
  auto con = build_eq_constraints(pt);
  if (con.has_value()) {
    attach_ordinal_geometric_diagnostics(est, pt, ev, *con, Bounds{}, base,
                                         OrdinalNewtonContext{&rep, &stats, nullptr, &layout, &factors, parameterization});
  }
  return est;
}

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          const Eigen::VectorXd& x0,
                          Backend backend,
                          OptimOptions opts,
                          OrdinalParameterization parameterization,
                          const std::vector<std::int8_t>* row_user) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr, row_user);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_mixed_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_mixed_ordinal_bounded: bounds size mismatch"));
  }

  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  const EqConstraints& con = *con_or;

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = mixed_ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                                    parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = x0.size();
  prob.expand  = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return mixed_ordinal_residuals(stats, layout, eval->moments, factors, x,
                                   parameterization);
  };
  prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return mixed_ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                                  eval->J_mu, factors, x, parameterization);
  };
  prob.eval = [&](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto r = mixed_ordinal_residuals(stats, layout, eval->moments, factors, x,
                                     parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = mixed_ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                                    eval->J_mu, factors, x, parameterization);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto est = solve_ordinal_ls(prob, x0, bounds, con, backend, opts,
                              "fit_mixed_ordinal_bounded");
  if (!est.has_value()) return est;
  attach_ordinal_geometric_diagnostics(
      *est, pt, ev, con, bounds, prob, OrdinalNewtonContext{&rep, nullptr, &stats, &layout, &factors, parameterization});
  return est;
}

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalMoments& moments,
                          data::OrdinalGammaCache* gamma_cache,
                          Bounds bounds,
                          data::OrdinalWeightPlan plan,
                          const Eigen::VectorXd& x0,
                          Backend backend,
                          OptimOptions opts) {
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto stats_or = mixed_stats_from_moments_cache(
      moments, gamma_cache, plan, "fit_mixed_ordinal_bounded");
  if (!stats_or.has_value()) return std::unexpected(stats_or.error());
  return fit_mixed_ordinal_bounded(
      std::move(pt), rep, stats_or->first, std::move(bounds),
      stats_or->second, x0, backend, opts,
      to_estimate_parameterization(plan.parameterization));
}

fit_expected<Estimates>
fit_mixed_ordinal_snlls_full_thresholds(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    OrdinalWeightKind weights,
    const Eigen::VectorXd& x0,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization) {
  // Theta is supported through the same full-threshold moment stack: the
  // standardized covariance moments make the non-threshold covariance block
  // nonlinear, so ordinal_gp_block_kinds marks only thresholds as
  // Golub-Pereyra linear coordinates under theta (the implementation answer
  // to the response-scale separability boundary).
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_snlls_full_thresholds: ModelEvaluator::build "
        "failed: " +
            ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_snlls_full_thresholds: x0 size (" +
            std::to_string(x0.size()) + ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto block_kinds_or =
      ordinal_gp_block_kinds(pt, layout, ev, parameterization);
  if (!block_kinds_or.has_value()) {
    return std::unexpected(block_kinds_or.error());
  }
  const auto& block_kinds = *block_kinds_or;

  auto eval0 = ev.evaluate(x0, true, true);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_snlls_full_thresholds: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = mixed_ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                                    parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem base;
  base.n_resid = r0->size();
  base.n_param = x0.size();
  base.expand = [](const Eigen::VectorXd& x) { return x; };
  base.r = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(
          FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_residuals(stats, layout, eval->moments, factors, theta,
                                   parameterization);
  };
  base.J = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(
          FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                                  eval->J_mu, factors, theta, parameterization);
  };
  base.eval =
      [&](const Eigen::VectorXd& theta) -> fit_expected<optim::LsEvaluation> {
    auto eval = ev.evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(
          FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    auto r = mixed_ordinal_residuals(stats, layout, eval->moments, factors,
                                     theta, parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J =
        mixed_ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                               eval->J_mu, factors, theta, parameterization);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto gp_or = gmm::gp(base, pt, ev, x0, block_kinds);
  if (!gp_or.has_value()) return std::unexpected(gp_or.error());
  const optim::GmmProblem& prob = gp_or->problem;

  Eigen::VectorXd theta_hat;
  double fmin = 0.0;
  int iterations = 0;
  int f_evals = 0;
  int g_evals = 0;
  optim::OptimStatus status = optim::OptimStatus::Converged;
  double grad_inf_norm = -1.0;
  optim::TerminalAudit audit;

  if (prob.n_param == 0) {
    auto r = prob.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    theta_hat = prob.expand(Eigen::VectorXd(0));
    fmin = 0.5 * r->squaredNorm();
  } else {
    auto out = run_ordinal_ls(prob, gp_or->beta0, Bounds{}, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    theta_hat = prob.expand(out->x);
    fmin = out->fmin;
    iterations = out->iterations;
    f_evals = out->f_evals;
    g_evals = out->g_evals;
    status = out->status;
    grad_inf_norm = out->grad_inf_norm;
    audit = std::move(out->audit);
  }

  Estimates est{std::move(theta_hat), fmin, iterations, f_evals, g_evals,
                status, grad_inf_norm, std::move(audit), {},
                gp_or->n_nonlinear, gp_or->n_linear};
  auto con = build_eq_constraints(pt);
  if (con.has_value()) {
    attach_ordinal_geometric_diagnostics(est, pt, ev, *con, Bounds{}, base,
                                         OrdinalNewtonContext{&rep, nullptr, &stats, &layout, &factors, parameterization});
  }
  return est;
}

fit_expected<Estimates>
fit_mixed_ordinal_snlls_full_thresholds(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalMoments& moments,
    data::OrdinalGammaCache* gamma_cache,
    data::OrdinalWeightPlan plan,
    const Eigen::VectorXd& x0,
    Backend backend,
    OptimOptions opts) {
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto stats_or = mixed_stats_from_moments_cache(
      moments, gamma_cache, plan, "fit_mixed_ordinal_snlls_full_thresholds");
  if (!stats_or.has_value()) return std::unexpected(stats_or.error());
  return fit_mixed_ordinal_snlls_full_thresholds(
      std::move(pt), rep, stats_or->first, stats_or->second, x0, backend, opts,
      to_estimate_parameterization(plan.parameterization));
}


}  // namespace magmaan::estimate
