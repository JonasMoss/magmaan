#include "ordinal_test_helpers.hpp"
#include "magmaan/estimate/constraints.hpp"

TEST_CASE("Ordinal robust reporting returns sandwich SEs and scaled-test eigenvalues") {
  std::mt19937 rng(20240514);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(800, 4);
  const double loading[4] = {0.95, 0.85, 0.75, 0.65};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.7) + (y > 0.0) + (y > 0.7);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2 + t3\n"
      "x2 | t1 + t2 + t3\n"
      "x3 | t1 + t2 + t3\n"
      "x4 | t1 + t2 + t3\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto fit = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  auto rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(rob.has_value());

  CHECK(rob->vcov.rows() == fit->theta.size());
  CHECK(rob->vcov.cols() == fit->theta.size());
  CHECK(rob->se.size() == fit->theta.size());
  CHECK(rob->se.allFinite());
  CHECK(rob->df == 2);
  CHECK(rob->eigvals.size() == rob->df);
  CHECK(rob->eigvals.minCoeff() > 0.0);
  CHECK(rob->chisq_standard == doctest::Approx(2.0 * 800.0 * fit->fmin));
  CHECK(rob->satorra_bentler.df == rob->df);
  CHECK(std::isfinite(rob->satorra_bentler.scale_c));
  CHECK(std::isfinite(rob->mean_var_adjusted.df_adj));
  CHECK(std::isfinite(rob->scaled_shifted.scale_a));
}

TEST_CASE("frontier ordinal observed omega uses DWLS IJ sandwich") {
  std::mt19937 rng(20260702);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(700, 4);
  const double loading[4] = {0.86, 0.78, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.45);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  using magmaan::estimate::OrdinalWeightKind;
  auto fit = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  namespace rel = magmaan::measures::frontier::reliability;
  rel::OmegaSpec omega_spec;
  omega_spec.block = Eigen::VectorXi::Zero(4);
  auto omega = magmaan::estimate::frontier::ordinal_observed_omega(
      *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
      OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(omega.has_value(),
      "ordinal observed omega failed: "
          << (omega.has_value() ? "" : omega.error().detail));
  CHECK(omega->value > 0.0);
  CHECK(omega->value < 1.0);
  CHECK(omega->gradient.size() == fit->theta.size());
  CHECK(omega->se > 0.0);
  CHECK(std::isfinite(omega->se));
  CHECK(omega->avar ==
        doctest::Approx(omega->se * omega->se *
                        static_cast<double>(stats->n_obs[0]))
            .epsilon(1e-10));
}

TEST_CASE("frontier ordinal profile_lrt_parameter reports the ordinary df-1 statistic") {
  std::mt19937 rng(20260703);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(720, 4);
  const double loading[4] = {0.84, 0.76, 0.68, 0.60};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.50) + (y > 0.50);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  using magmaan::estimate::OrdinalWeightKind;
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto fit = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      prepared, *stats, nullptr);
  REQUIRE(prep.has_value());
  auto ev = magmaan::model::ModelEvaluator::build(prepared, *mr);
  REQUIRE(ev.has_value());
  Eigen::Index k_loading = -1;
  const auto locs = ev->param_locations();
  for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(locs.size()); ++k) {
    if (locs[static_cast<std::size_t>(k)].mat ==
            magmaan::model::MatId::Lambda &&
        locs[static_cast<std::size_t>(k)].row == 1 &&
        locs[static_cast<std::size_t>(k)].col == 0) {
      k_loading = k;
      break;
    }
  }
  REQUIRE(k_loading >= 0);
  const double target = 0.95 * fit->theta(k_loading);

  auto lrt = magmaan::estimate::frontier::profile_lrt_parameter_ordinal(
      *pt, *mr, *stats, *fit, k_loading, target, {},
      OrdinalWeightKind::DWLS, magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(lrt.has_value(),
      "ordinal parameter profile LRT failed: "
          << (lrt.has_value() ? "" : lrt.error().detail));

  CHECK(lrt->unrestricted_value == doctest::Approx(fit->theta(k_loading)));
  CHECK(lrt->constrained_value == doctest::Approx(target).epsilon(1e-7));
  CHECK(std::abs(lrt->constraint_residual) < 1e-7);
  CHECK(lrt->fmin_unrestricted == doctest::Approx(fit->fmin));
  CHECK(lrt->T == doctest::Approx(
      2.0 * static_cast<double>(stats->n_obs[0]) *
      (lrt->fmin_constrained - fit->fmin)));
  CHECK(lrt->p_value == doctest::Approx(
      magmaan::inference::chi2_pvalue(lrt->T, 1)));
  CHECK(lrt->df == 1);

  auto lrt_robust = magmaan::estimate::frontier::profile_lrt_parameter_ordinal(
      *pt, *mr, *stats, *fit, k_loading, target, {},
      OrdinalWeightKind::DWLS, magmaan::estimate::Backend::NloptSlsqp, opts,
      magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true);
  REQUIRE_MESSAGE(lrt_robust.has_value(),
      "ordinal parameter robust profile LRT failed: "
          << (lrt_robust.has_value() ? "" : lrt_robust.error().detail));
  CHECK(lrt_robust->T == doctest::Approx(lrt->T).epsilon(1e-8));
  CHECK(lrt_robust->scaling_factor > 0.0);
  CHECK(std::isfinite(lrt_robust->scaling_factor));
  CHECK(lrt_robust->T_scaled ==
        doctest::Approx(lrt_robust->T / lrt_robust->scaling_factor));
  CHECK(lrt_robust->p_value_scaled == doctest::Approx(
      magmaan::inference::chi2_pvalue(lrt_robust->T_scaled, 1)));

  magmaan::estimate::frontier::ScalarProfileCiOptions ci_opts;
  ci_opts.target_tol = 1e-4;
  ci_opts.statistic_tol = 1e-4;
  ci_opts.initial_step = 0.08 * std::abs(fit->theta(k_loading));
  ci_opts.max_iter = 40;

  auto ci = magmaan::estimate::frontier::profile_lrt_ci_parameter_ordinal(
      *pt, *mr, *stats, *fit, k_loading, ci_opts, {},
      OrdinalWeightKind::DWLS, magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(ci.has_value(),
      "ordinal parameter profile CI failed: "
          << (ci.has_value() ? "" : ci.error().detail));

  CHECK(ci->lower < fit->theta(k_loading));
  CHECK(ci->upper > fit->theta(k_loading));
  CHECK_FALSE(ci->lower_at_bound);
  CHECK_FALSE(ci->upper_at_bound);
  CHECK(ci->lower_profile.T == doctest::Approx(ci->cutoff).epsilon(1e-3));
  CHECK(ci->upper_profile.T == doctest::Approx(ci->cutoff).epsilon(1e-3));
  CHECK(magmaan::inference::chi2_pvalue(ci->cutoff, 1) ==
        doctest::Approx(1.0 - ci->confidence_level).epsilon(1e-6));

  auto ci_robust_opts = ci_opts;
  ci_robust_opts.reference =
      magmaan::estimate::frontier::ScalarProfileReference::RobustScaled;
  auto ci_robust =
      magmaan::estimate::frontier::profile_lrt_ci_parameter_ordinal(
          *pt, *mr, *stats, *fit, k_loading, ci_robust_opts, {},
          OrdinalWeightKind::DWLS, magmaan::estimate::Backend::NloptSlsqp,
          opts, magmaan::estimate::OrdinalParameterization::Delta, 1e-6,
          true);
  REQUIRE_MESSAGE(ci_robust.has_value(),
      "ordinal parameter robust profile CI failed: "
          << (ci_robust.has_value() ? "" : ci_robust.error().detail));
  CHECK(ci_robust->lower < fit->theta(k_loading));
  CHECK(ci_robust->upper > fit->theta(k_loading));
  CHECK(ci_robust->lower_profile.scaling_factor > 0.0);
  CHECK(ci_robust->upper_profile.scaling_factor > 0.0);
  CHECK(ci_robust->lower_profile.T_scaled ==
        doctest::Approx(ci_robust->cutoff).epsilon(1e-3));
  CHECK(ci_robust->upper_profile.T_scaled ==
        doctest::Approx(ci_robust->cutoff).epsilon(1e-3));

  namespace rel = magmaan::measures::frontier::reliability;
  rel::OmegaSpec omega_spec;
  omega_spec.block = Eigen::VectorXi::Zero(X.cols());
  auto omega_sample =
      rel::omega_multidim(rel::OmegaTarget::Total, stats->R[0], omega_spec);
  REQUIRE(omega_sample.has_value());
  const double omega_target = 0.98 * *omega_sample;

  auto omega_lrt =
      magmaan::estimate::frontier::profile_lrt_ordinal_polychoric_omega(
          *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
          omega_target, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(omega_lrt.has_value(),
      "ordinal polychoric omega profile LRT failed: "
          << (omega_lrt.has_value() ? "" : omega_lrt.error().detail));

  CHECK(omega_lrt->unrestricted_value > 0.0);
  CHECK(omega_lrt->unrestricted_value < 1.0);
  CHECK(omega_lrt->constrained_value ==
        doctest::Approx(omega_target).epsilon(1e-5));
  CHECK(std::abs(omega_lrt->constraint_residual) < 1e-5);
  CHECK(omega_lrt->fmin_unrestricted == doctest::Approx(fit->fmin));
  CHECK(omega_lrt->T == doctest::Approx(
      2.0 * static_cast<double>(stats->n_obs[0]) *
      (omega_lrt->fmin_constrained - fit->fmin)));
  CHECK(omega_lrt->p_value == doctest::Approx(
      magmaan::inference::chi2_pvalue(omega_lrt->T, 1)));
  CHECK(omega_lrt->df == 1);

  auto omega_lrt_robust =
      magmaan::estimate::frontier::profile_lrt_ordinal_polychoric_omega(
          *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
          omega_target, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true);
  REQUIRE_MESSAGE(omega_lrt_robust.has_value(),
      "ordinal polychoric omega robust profile LRT failed: "
          << (omega_lrt_robust.has_value() ? ""
                                           : omega_lrt_robust.error().detail));
  CHECK(omega_lrt_robust->T == doctest::Approx(omega_lrt->T).epsilon(1e-8));
  CHECK(omega_lrt_robust->scaling_factor > 0.0);
  CHECK(omega_lrt_robust->T_scaled ==
        doctest::Approx(omega_lrt_robust->T /
                        omega_lrt_robust->scaling_factor));
  CHECK(omega_lrt_robust->p_value_scaled == doctest::Approx(
      magmaan::inference::chi2_pvalue(omega_lrt_robust->T_scaled, 1)));
  auto omega_lrt_misspec =
      magmaan::estimate::frontier::profile_lrt_ordinal_polychoric_omega(
          *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
          omega_target, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true,
          magmaan::estimate::frontier::ScalarProfileReference::MisspecMixture);
  REQUIRE_MESSAGE(omega_lrt_misspec.has_value(),
      "ordinal polychoric omega misspec profile LRT failed: "
          << (omega_lrt_misspec.has_value() ? ""
                                            : omega_lrt_misspec.error().detail));
  CHECK(omega_lrt_misspec->T == doctest::Approx(omega_lrt->T).epsilon(1e-8));
  CHECK(omega_lrt_misspec->misspec_scaling_factor > 0.0);
  CHECK(std::isfinite(omega_lrt_misspec->misspec_scaling_factor));
  REQUIRE(omega_lrt_misspec->misspec_eigvals.size() == 1);
  CHECK(omega_lrt_misspec->misspec_eigvals(0) == doctest::Approx(
      omega_lrt_misspec->misspec_scaling_factor));
  CHECK(omega_lrt_misspec->T_misspec_scaled == doctest::Approx(
      omega_lrt_misspec->T / omega_lrt_misspec->misspec_scaling_factor));
  CHECK(omega_lrt_misspec->p_value_misspec_scaled == doctest::Approx(
      magmaan::inference::chi2_pvalue(
          omega_lrt_misspec->T_misspec_scaled, 1)));
  CHECK(omega_lrt_misspec->p_value_misspec_mixture == doctest::Approx(
      magmaan::robust::weighted_chisq_upper(
          omega_lrt_misspec->misspec_eigvals, omega_lrt_misspec->T)));

  magmaan::estimate::frontier::ScalarProfileCiOptions omega_ci_opts;
  omega_ci_opts.target_tol = 1e-5;
  omega_ci_opts.statistic_tol = 1e-5;
  omega_ci_opts.initial_step = 0.02;
  omega_ci_opts.max_iter = 50;

  auto omega_ci =
      magmaan::estimate::frontier::profile_lrt_ci_ordinal_polychoric_omega(
          *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
          omega_ci_opts, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(omega_ci.has_value(),
      "ordinal polychoric omega profile CI failed: "
          << (omega_ci.has_value() ? "" : omega_ci.error().detail));

  CHECK(omega_ci->lower < omega_ci->estimate);
  CHECK(omega_ci->upper > omega_ci->estimate);
  CHECK_FALSE(omega_ci->lower_at_bound);
  CHECK_FALSE(omega_ci->upper_at_bound);
  CHECK(omega_ci->lower_profile.T ==
        doctest::Approx(omega_ci->cutoff).epsilon(1e-3));
  CHECK(omega_ci->upper_profile.T ==
        doctest::Approx(omega_ci->cutoff).epsilon(1e-3));

  auto omega_ci_robust_opts = omega_ci_opts;
  omega_ci_robust_opts.reference =
      magmaan::estimate::frontier::ScalarProfileReference::RobustScaled;
  auto omega_ci_robust =
      magmaan::estimate::frontier::profile_lrt_ci_ordinal_polychoric_omega(
          *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
          omega_ci_robust_opts, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true);
  REQUIRE_MESSAGE(omega_ci_robust.has_value(),
      "ordinal polychoric omega robust profile CI failed: "
          << (omega_ci_robust.has_value() ? ""
                                          : omega_ci_robust.error().detail));
  CHECK(omega_ci_robust->lower < omega_ci_robust->estimate);
  CHECK(omega_ci_robust->upper > omega_ci_robust->estimate);
  CHECK(omega_ci_robust->lower_profile.scaling_factor > 0.0);
  CHECK(omega_ci_robust->upper_profile.scaling_factor > 0.0);
  CHECK(omega_ci_robust->lower_profile.T_scaled ==
        doctest::Approx(omega_ci_robust->cutoff).epsilon(1e-3));
  CHECK(omega_ci_robust->upper_profile.T_scaled ==
        doctest::Approx(omega_ci_robust->cutoff).epsilon(1e-3));

  auto omega_ci_misspec_scaled_opts = omega_ci_opts;
  omega_ci_misspec_scaled_opts.reference =
      magmaan::estimate::frontier::ScalarProfileReference::MisspecScaled;
  auto omega_ci_misspec_scaled =
      magmaan::estimate::frontier::profile_lrt_ci_ordinal_polychoric_omega(
          *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
          omega_ci_misspec_scaled_opts, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true);
  REQUIRE_MESSAGE(omega_ci_misspec_scaled.has_value(),
      "ordinal polychoric omega misspec-scaled profile CI failed: "
          << (omega_ci_misspec_scaled.has_value() ? ""
              : omega_ci_misspec_scaled.error().detail));
  CHECK(omega_ci_misspec_scaled->lower_profile.misspec_scaling_factor > 0.0);
  CHECK(omega_ci_misspec_scaled->upper_profile.misspec_scaling_factor > 0.0);
  CHECK(omega_ci_misspec_scaled->lower_profile.T_misspec_scaled ==
        doctest::Approx(omega_ci_misspec_scaled->cutoff).epsilon(3e-3));
  CHECK(omega_ci_misspec_scaled->upper_profile.T_misspec_scaled ==
        doctest::Approx(omega_ci_misspec_scaled->cutoff).epsilon(3e-3));
  CHECK(omega_ci_misspec_scaled->lower_cutoff ==
        doctest::Approx(omega_ci_misspec_scaled->cutoff));
  CHECK(omega_ci_misspec_scaled->upper_cutoff ==
        doctest::Approx(omega_ci_misspec_scaled->cutoff));

  auto omega_ci_mixture_opts = omega_ci_opts;
  omega_ci_mixture_opts.reference =
      magmaan::estimate::frontier::ScalarProfileReference::MisspecMixture;
  auto omega_ci_mixture =
      magmaan::estimate::frontier::profile_lrt_ci_ordinal_polychoric_omega(
          *pt, *mr, *stats, *fit, omega_spec, rel::OmegaTarget::Total,
          omega_ci_mixture_opts, {}, OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true);
  REQUIRE_MESSAGE(omega_ci_mixture.has_value(),
      "ordinal polychoric omega misspec-mixture profile CI failed: "
          << (omega_ci_mixture.has_value() ? ""
              : omega_ci_mixture.error().detail));
  REQUIRE(omega_ci_mixture->lower_profile.misspec_eigvals.size() == 1);
  REQUIRE(omega_ci_mixture->upper_profile.misspec_eigvals.size() == 1);
  CHECK(omega_ci_mixture->lower_cutoff == doctest::Approx(
      magmaan::robust::weighted_chisq_quantile(
          omega_ci_mixture->lower_profile.misspec_eigvals,
          omega_ci_mixture->confidence_level)));
  CHECK(omega_ci_mixture->upper_cutoff == doctest::Approx(
      magmaan::robust::weighted_chisq_quantile(
          omega_ci_mixture->upper_profile.misspec_eigvals,
          omega_ci_mixture->confidence_level)));
  CHECK(omega_ci_mixture->lower_profile.misspec_mixture_cutoff ==
        doctest::Approx(omega_ci_mixture->lower_cutoff));
  CHECK(omega_ci_mixture->upper_profile.misspec_mixture_cutoff ==
        doctest::Approx(omega_ci_mixture->upper_cutoff));
  CHECK(omega_ci_mixture->lower_profile.T ==
        doctest::Approx(omega_ci_mixture->lower_cutoff).epsilon(3e-3));
  CHECK(omega_ci_mixture->upper_profile.T ==
        doctest::Approx(omega_ci_mixture->upper_cutoff).epsilon(3e-3));
}

TEST_CASE("Ordinal nonlinear profile scaling uses the full equality tangent") {
  using namespace magmaan;
  std::mt19937 rng(542073);
  std::normal_distribution<double> norm(0.0,1.0);
  Eigen::MatrixXd X(700,4);
  const double loading[] = {.85,.75,.7,.65};
  for (Eigen::Index i=0;i<X.rows();++i) {
    const double eta=norm(rng);
    for (Eigen::Index j=0;j<X.cols();++j) {
      const double y=loading[j]*eta+.8*norm(rng);
      X(i,j)=1+(y>-.5)+(y>.5);
    }
  }
  auto stats=data::ordinal_stats_from_integer_data({X}); REQUIRE(stats);
  auto fp=parse::Parser::parse("f =~ x1+a*x2+b*x3+x4\na == b^2\nx1 | t1+t2\nx2 | t1+t2\nx3 | t1+t2\nx4 | t1+t2\n"); REQUIRE(fp);
  spec::LatentNames names;
  auto pt=spec::build(*fp,{},nullptr,&names); REQUIRE(pt);
  auto rep=model::build_matrix_rep(*pt,&names); REQUIRE(rep);
  REQUIRE(estimate::prepare_ordinal_partable(*pt,*stats,estimate::OrdinalParameterization::Delta,nullptr,&names.row_user));
  auto start=estimate::ordinal_start_values(*pt,*rep,*stats,{},&names.row_user); REQUIRE(start);
  optim::OptimOptions options; options.max_iter=4000; options.ftol=1e-13; options.gtol=1e-8;
  auto fit=estimate::fit_ordinal_bounded(*pt,*rep,*stats,{},estimate::OrdinalWeightKind::ULS,*start,estimate::Backend::NloptSlsqp,options); REQUIRE(fit);
  Eigen::Index k=-1;
  for (std::size_t i=0;i<pt->size();++i) if(names.row_label[i]=="a" && pt->free[i]>0) k=pt->free[i]-1;
  REQUIRE(k>=0);
  auto profile=estimate::frontier::profile_lrt_parameter_ordinal(*pt,*rep,*stats,*fit,k,.98*fit->theta(k),{},
      estimate::OrdinalWeightKind::ULS,estimate::Backend::NloptSlsqp,options,
      estimate::OrdinalParameterization::Delta,1e-6,true);
  REQUIRE_MESSAGE(profile.has_value(), (profile ? "" : profile.error().detail));
  estimate::Estimates at; at.theta=profile->constrained.theta; at.fmin=profile->fmin_constrained;
  auto tangent=estimate::build_eq_tangent(*pt,at.theta); REQUIRE(tangent);
  auto objective=estimate::frontier::ordinal_ls_objective(*pt,*rep,*stats,at,estimate::OrdinalWeightKind::ULS); REQUIRE(objective);
  auto J=objective->problem.J(at.theta); REQUIRE(J);
  const Eigen::MatrixXd JK=*J*tangent->K();
  const Eigen::VectorXd g=tangent->K().row(k).transpose();
  const double denominator=g.dot((JK.transpose()*JK).ldlt().solve(g));
  auto robust=estimate::robust_ordinal(*pt,*rep,*stats,at,estimate::OrdinalWeightKind::ULS); REQUIRE(robust);
  const double expected=static_cast<double>(X.rows())*robust->vcov(k,k)/denominator;
  CHECK(profile->scaling_factor==doctest::Approx(expected).epsilon(1e-8));
  // The former affine-only basis treats a and b as independent and yields a
  // different scaling factor even though the fit itself obeys a=b^2.
  auto refused=estimate::frontier::profile_lrt_parameter_ordinal(*pt,*rep,*stats,*fit,k,.98*fit->theta(k),{},
      estimate::OrdinalWeightKind::ULS,estimate::Backend::NloptSlsqp,options,
      estimate::OrdinalParameterization::Delta,1e-6,true,estimate::frontier::ScalarProfileReference::MisspecScaled);
  REQUIRE_FALSE(refused); CHECK(refused.error().detail.find("Lagrangian curvature")!=std::string::npos);
}
