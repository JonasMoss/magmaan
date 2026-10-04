#include "ordinal_internal.hpp"

namespace magmaan::estimate {

using namespace detail_ordinal;

post_expected<robust::LRSatorra2000Result>
lr_test_satorra2000_ordinal(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const data::OrdinalStats& stats,
    const Estimates& est_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const Estimates& est_H0,
    OrdinalWeightKind weights,
    double T_H0,
    double T_H1,
    int df_H0,
    int df_H1,
    robust::SatorraAMethod a_method,
    OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* row_user_H1,
    const std::vector<std::int8_t>* row_user_H0) {
  (void)df_H0;
  (void)df_H1;
  if (auto v = validate_stats(stats, rep_H1, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto v = validate_stats(stats, rep_H0, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_ordinal_partable(pt_H1, stats, parameterization, nullptr, row_user_H1);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (auto p = prepare_ordinal_partable(pt_H0, stats, parameterization, nullptr, row_user_H0);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est_H1.theta.size() != pt_H1.n_free() ||
      est_H0.theta.size() != pt_H0.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_ordinal: fitted theta length does not match "
        "prepared ordinal partable"));
  }

  auto con1 = build_eq_tangent(pt_H1, est_H1.theta);
  if (!con1.has_value()) return std::unexpected(con1.error());
  auto con0 = build_eq_tangent(pt_H0, est_H0.theta);
  if (!con0.has_value()) return std::unexpected(con0.error());
  const int df_H1_ordinal =
      static_cast<int>(ordinal_moment_rows(stats) - con1->n_alpha);
  const int df_H0_ordinal =
      static_cast<int>(ordinal_moment_rows(stats) - con0->n_alpha);
  const int df_diff = df_H0_ordinal - df_H1_ordinal;
  if (df_diff < 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_ordinal: ordinal df_H0 - df_H1 is negative; "
        "H1 must be the less-restricted model"));
  }

  auto delta_alpha = [&](const spec::LatentStructure& pt,
                         const model::MatrixRep& rep,
                         const Estimates& est,
                         const EqConstraints& con,
                         const char* who) -> post_expected<Eigen::MatrixXd> {
    auto layout_or = make_threshold_layout(pt, rep, stats);
    if (!layout_or.has_value()) {
      return std::unexpected(fit_to_post(layout_or.error()));
    }
    auto ev_or = model::ModelEvaluator::build(pt, rep);
    if (!ev_or.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": ModelEvaluator::build failed: " +
          ev_or.error().detail));
    }
    auto eval = ev_or->evaluate(est.theta, true, true);  // J_mu for released μ
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": fitted evaluation failed: " +
          eval.error().detail));
    }
    const Eigen::MatrixXd Delta_full =
        ordinal_moment_jacobian(stats, *layout_or, eval->moments,
                                eval->J_sigma, est.theta, parameterization,
                                eval->J_mu);
    if (con.Kmat.rows() != Delta_full.cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": constraint reparameterization has "
          "incompatible shape"));
    }
    return Eigen::MatrixXd(Delta_full * con.Kmat);
  };

  auto D1_or = delta_alpha(pt_H1, rep_H1, est_H1, *con1,
                           "lr_test_satorra2000_ordinal H1");
  if (!D1_or.has_value()) return std::unexpected(D1_or.error());

  Eigen::MatrixXd A_alpha;
  if (a_method == robust::SatorraAMethod::Exact) {
    auto embedding = robust::embed_nested_null(pt_H1,rep_H1,pt_H0,rep_H0,
        est_H0.theta,*con1,*con0);
    if (!embedding) return std::unexpected(embedding.error());
    auto restr_or = &embedding->restriction;
    A_alpha = std::move(restr_or->A);
  } else {
    auto D0_or = delta_alpha(pt_H0, rep_H0, est_H0, *con0,
                             "lr_test_satorra2000_ordinal H0");
    if (!D0_or.has_value()) return std::unexpected(D0_or.error());
    auto A_or = robust::restriction_alpha_delta_from_jacobians(
        *D1_or, *D0_or, df_diff);
    if (!A_or.has_value()) return std::unexpected(A_or.error());
    A_alpha = std::move(*A_or);
  }

  const int df_diff_from_A = static_cast<int>(A_alpha.rows());
  if (df_diff_from_A != df_diff) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_ordinal: df_diff mismatch -- derived A has m = " +
        std::to_string(df_diff_from_A) + " but ordinal df_H0 - df_H1 = " +
        std::to_string(df_diff)));
  }

  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());
  auto sandwich = ordinal_param_space_sandwich(stats, *Ws, *D1_or);
  if (!sandwich.has_value()) return std::unexpected(sandwich.error());
  auto sd_or = robust::compute_satorra2000_from_sandwich(
      sandwich->A1, sandwich->B1, A_alpha);
  if (!sd_or.has_value()) return std::unexpected(sd_or.error());
  return robust::lr_test_satorra2000(T_H0 - T_H1, *sd_or);
}

post_expected<robust::LRSatorra2000Result>
lr_test_satorra2000_mixed_ordinal(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const data::MixedOrdinalStats& stats,
    const Estimates& est_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const Estimates& est_H0,
    OrdinalWeightKind weights,
    double T_H0,
    double T_H1,
    int df_H0,
    int df_H1,
    robust::SatorraAMethod a_method,
    OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* row_user_H1,
    const std::vector<std::int8_t>* row_user_H0) {
  (void)df_H0;
  (void)df_H1;
  if (weights == OrdinalWeightKind::ULS) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: mixed ordinal nested tests support "
        "DWLS/WLS weights only"));
  }
  if (auto v = validate_stats(stats, rep_H1, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto v = validate_stats(stats, rep_H0, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt_H1, stats, nullptr, row_user_H1);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt_H0, stats, nullptr, row_user_H0);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est_H1.theta.size() != pt_H1.n_free() ||
      est_H0.theta.size() != pt_H0.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: fitted theta length does not "
        "match prepared mixed ordinal partable"));
  }

  auto con1 = build_eq_constraints(pt_H1);
  if (!con1.has_value()) return std::unexpected(con1.error());
  auto con0 = build_eq_constraints(pt_H0);
  if (!con0.has_value()) return std::unexpected(con0.error());
  const int df_H1_mixed =
      static_cast<int>(mixed_moment_rows(stats) - con1->n_alpha);
  const int df_H0_mixed =
      static_cast<int>(mixed_moment_rows(stats) - con0->n_alpha);
  const int df_diff = df_H0_mixed - df_H1_mixed;
  if (df_diff < 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: mixed ordinal df_H0 - df_H1 is "
        "negative; H1 must be the less-restricted model"));
  }

  auto delta_alpha = [&](const spec::LatentStructure& pt,
                         const model::MatrixRep& rep,
                         const Estimates& est,
                         const EqConstraints& con,
                         const char* who) -> post_expected<Eigen::MatrixXd> {
    auto layout_or = make_threshold_layout(pt, rep, stats);
    if (!layout_or.has_value()) {
      return std::unexpected(fit_to_post(layout_or.error()));
    }
    auto ev_or = model::ModelEvaluator::build(pt, rep);
    if (!ev_or.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": ModelEvaluator::build failed: " +
          ev_or.error().detail));
    }
    auto eval = ev_or->evaluate(est.theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": fitted evaluation failed: " +
          eval.error().detail));
    }
    const Eigen::MatrixXd Delta_full =
        mixed_moment_jacobian(stats, *layout_or, eval->moments,
                              eval->J_sigma, eval->J_mu, est.theta,
                              parameterization);
    if (con.Kmat.rows() != Delta_full.cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": constraint reparameterization has "
          "incompatible shape"));
    }
    return Eigen::MatrixXd(Delta_full * con.Kmat);
  };

  auto D1_or = delta_alpha(pt_H1, rep_H1, est_H1, *con1,
                           "lr_test_satorra2000_mixed_ordinal H1");
  if (!D1_or.has_value()) return std::unexpected(D1_or.error());

  Eigen::MatrixXd A_alpha;
  if (a_method == robust::SatorraAMethod::Exact) {
    auto embedding = robust::embed_nested_null(pt_H1,rep_H1,pt_H0,rep_H0,
        est_H0.theta,*con1,*con0);
    if (!embedding) return std::unexpected(embedding.error());
    auto restr_or = &embedding->restriction;
    A_alpha = std::move(restr_or->A);
  } else {
    auto D0_or = delta_alpha(pt_H0, rep_H0, est_H0, *con0,
                             "lr_test_satorra2000_mixed_ordinal H0");
    if (!D0_or.has_value()) return std::unexpected(D0_or.error());
    auto A_or = robust::restriction_alpha_delta_from_jacobians(
        *D1_or, *D0_or, df_diff);
    if (!A_or.has_value()) return std::unexpected(A_or.error());
    A_alpha = std::move(*A_or);
  }

  const int df_diff_from_A = static_cast<int>(A_alpha.rows());
  if (df_diff_from_A != df_diff) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: df_diff mismatch -- derived A "
        "has m = " +
        std::to_string(df_diff_from_A) +
        " but mixed ordinal df_H0 - df_H1 = " + std::to_string(df_diff)));
  }

  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());
  auto sandwich = ordinal_param_space_sandwich(stats, *Ws, *D1_or);
  if (!sandwich.has_value()) return std::unexpected(sandwich.error());
  auto sd_or = robust::compute_satorra2000_from_sandwich(
      sandwich->A1, sandwich->B1, A_alpha);
  if (!sd_or.has_value()) return std::unexpected(sd_or.error());
  return robust::lr_test_satorra2000(T_H0 - T_H1, *sd_or);
}

}  // namespace magmaan::estimate
