#include "ordinal_test_helpers.hpp"

TEST_CASE("ordinal_dwls_profile_rmsea assembles the extended (u, gamma) law") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260622, 600, {0.80, 0.74, 0.68, 0.62}, -0.5, 0.6);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  REQUIRE(stats->moment_influence.size() == 1);
  REQUIRE(stats->int_data.size() == 1);

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
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Expected);
  REQUIRE(rob.has_value());

  auto prof = magmaan::estimate::ordinal_dwls_profile_rmsea(*pt, *mr, *stats,
                                                            *fit);
  REQUIRE_MESSAGE(prof.has_value(),
      "profile RMSEA failed: " << (prof.has_value() ? "" : prof.error().detail));

  const Eigen::Index p = stats->R[0].rows();
  const Eigen::Index m = stats->thresholds[0].size() + p * (p - 1) / 2;
  REQUIRE(prof->profile_hessian.rows() == 2 * m);
  REQUIRE(prof->gamma.rows() == 2 * m);

  // Standard statistic and classical df agree with the standard DWLS path.
  CHECK(prof->df == rob->df);
  CHECK(prof->chisq_standard == doctest::Approx(rob->chisq_standard));
  CHECK(prof->spectrum_size > 0);
  CHECK(prof->bias_trace > 0.0);
  CHECK(prof->rmsea >= 0.0);

  // Gamma_x uu-block reproduces the polychoric NACOV (influence stack + scale).
  CHECK(prof->gamma.topLeftCorner(m, m).isApprox(stats->NACOV[0], 1e-9));
  // The estimated-weight channel populates the gamma-gamma and cross blocks.
  CHECK(prof->gamma.bottomRightCorner(m, m).cwiseAbs().maxCoeff() > 0.0);
  CHECK(prof->gamma.topRightCorner(m, m).cwiseAbs().maxCoeff() > 0.0);
}

TEST_CASE("ordinal_dwls_profile_lrt compares nested ordinal DWLS models") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260623, 700, {0.78, 0.72, 0.66, 0.60}, -0.4, 0.7);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());

  const char* thresholds =
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  const std::string syntax_h1 =
      std::string("f =~ x1 + x2 + x3 + x4\n") + thresholds;
  // Tau-equivalence as a strict restriction of the marker-identified H1: the
  // remaining loadings are fixed to the marker value of 1.
  const std::string syntax_h0 =
      std::string("f =~ x1 + 1*x2 + 1*x3 + 1*x4\n") + thresholds;

  auto build_fit = [&](const std::string& syntax) {
    auto fp = magmaan::parse::Parser::parse(syntax);
    REQUIRE(fp.has_value());
    auto pt = magmaan::spec::build(*fp);
    REQUIRE(pt.has_value());
    auto mr = magmaan::model::build_matrix_rep(*pt);
    REQUIRE(mr.has_value());
    auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
    REQUIRE(x0.has_value());
    magmaan::optim::OptimOptions opts;
    opts.max_iter = 1500;
    opts.ftol = 1e-12;
    opts.gtol = 1e-8;
    auto fit = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    REQUIRE_MESSAGE(fit.has_value(),
        "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));
    return std::make_tuple(std::move(*pt), std::move(*mr), std::move(*fit));
  };

  auto [pt1, mr1, fit1] = build_fit(syntax_h1);
  auto [pt0, mr0, fit0] = build_fit(syntax_h0);

  auto rob1 = magmaan::estimate::robust_ordinal(
      pt1, mr1, *stats, fit1, magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Expected);
  auto rob0 = magmaan::estimate::robust_ordinal(
      pt0, mr0, *stats, fit0, magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Expected);
  REQUIRE(rob1.has_value());
  REQUIRE(rob0.has_value());

  auto lrt = magmaan::estimate::ordinal_dwls_profile_lrt(
      pt1, mr1, *stats, fit1, pt0, mr0, fit0);
  REQUIRE_MESSAGE(lrt.has_value(),
      "profile LRT failed: " << (lrt.has_value() ? "" : lrt.error().detail));

  CHECK(lrt->df_diff == rob0->df - rob1->df);
  CHECK(lrt->df_diff > 0);
  CHECK(lrt->spectrum_size > 0);
  CHECK(lrt->T_diff >= -1e-6);
  CHECK(std::isfinite(lrt->p_mixture));
  CHECK(lrt->p_mixture >= 0.0);
  CHECK(lrt->p_mixture <= 1.0);
}

TEST_CASE("ordinal CRMR point estimate and SRMR denominator relation") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260624, 600, {0.80, 0.74, 0.68, 0.62}, -0.5, 0.6);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto fm = magmaan::estimate::fit_measures_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fm.has_value());
  // p=4: ncorr=6, vech_len=10. SRMR and CRMR share the numerator.
  CHECK(fm->crmr > 0.0);
  CHECK(fm->srmr ==
        doctest::Approx(fm->crmr * std::sqrt(6.0 / 10.0)).epsilon(1e-9));
}

TEST_CASE("ordinal_crmr_misspec_inference runs and the gamma channel is active") {
  // Tau-equivalence imposed on unequal-loading data => misspecified => nonzero
  // residual => the estimated-weight (gamma) channel is live.
  const Eigen::MatrixXd X =
      ordinal_test_block(20260625, 800, {0.82, 0.66, 0.52, 0.40}, -0.4, 0.7);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + 1*x2 + 1*x3 + 1*x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto fm = magmaan::estimate::fit_measures_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fm.has_value());

  auto est = magmaan::estimate::ordinal_crmr_misspec_inference(
      *pt, *mr, *stats, *fit);
  REQUIRE_MESSAGE(est.has_value(),
      "crmr inference failed: " << (est.has_value() ? "" : est.error().detail));

  CHECK(est->point == doctest::Approx(fm->crmr).epsilon(1e-9));
  CHECK(est->k == 6);
  CHECK(est->bias_trace > 0.0);
  CHECK(std::isfinite(est->bias_trace));
  CHECK(est->grad_var >= 0.0);
  CHECK(std::isfinite(est->grad_var));
  CHECK(est->spectrum_size > 0);
  CHECK(est->ci_lower <= est->ci_upper);
  CHECK(est->ci_lower >= 0.0);
  CHECK(std::isfinite(est->exact_fit_pvalue));
  CHECK(est->exact_fit_pvalue >= 0.0);
  CHECK(est->exact_fit_pvalue <= 1.0);

  // Fixed-weight comparator: drops the gamma channel, so the law differs under
  // misspecification (its bias trace is not identical to the estimated-weight
  // one). Both must be finite.
  auto fixed = magmaan::estimate::ordinal_crmr_misspec_inference(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE(fixed.has_value());
  CHECK(fixed->point == doctest::Approx(est->point).epsilon(1e-12));  // same G
  CHECK(std::isfinite(fixed->grad_var));
  CHECK(fixed->bias_trace > 0.0);
  CHECK(std::abs(fixed->bias_trace - est->bias_trace) > 1e-8);  // gamma active

  // SRMR variant: same statistic, different denominator/scale.
  auto srmr = magmaan::estimate::ordinal_crmr_misspec_inference(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/true, /*srmr_denominator=*/true);
  REQUIRE(srmr.has_value());
  CHECK(srmr->k == 10);
  CHECK(srmr->stat == doctest::Approx(est->stat).epsilon(1e-12));
  CHECK(srmr->point ==
        doctest::Approx(est->point * std::sqrt(6.0 / 10.0)).epsilon(1e-9));
}

TEST_CASE("ordinal_rmsea_misspec_inference matches profile point and runs the gamma channel") {
  // Tau-equivalence on unequal-loading data => misspecified => live gamma channel.
  const Eigen::MatrixXd X =
      ordinal_test_block(20260626, 800, {0.82, 0.66, 0.52, 0.40}, -0.4, 0.7);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + 1*x2 + 1*x3 + 1*x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto prof = magmaan::estimate::ordinal_dwls_profile_rmsea(*pt, *mr, *stats,
                                                            *fit);
  REQUIRE(prof.has_value());

  auto est = magmaan::estimate::ordinal_rmsea_misspec_inference(*pt, *mr, *stats,
                                                                *fit);
  REQUIRE_MESSAGE(est.has_value(),
      "rmsea inference failed: " << (est.has_value() ? "" : est.error().detail));

  // The bias-corrected point equals the profile RMSEA; stat/df/fmin pass through.
  CHECK(est->point == doctest::Approx(prof->rmsea).epsilon(1e-9));
  CHECK(est->stat == doctest::Approx(prof->chisq_standard).epsilon(1e-9));
  CHECK(est->df == prof->df);
  CHECK(est->fmin == doctest::Approx(prof->fmin).epsilon(1e-12));
  CHECK(est->bias_trace == doctest::Approx(prof->trace_signed).epsilon(1e-9));

  CHECK(est->ci_lower <= est->ci_upper);
  CHECK(est->ci_lower >= 0.0);
  CHECK(est->point >= est->ci_lower);
  CHECK(est->point <= est->ci_upper);
  CHECK(std::isfinite(est->grad_var));
  CHECK(est->grad_var >= 0.0);
  CHECK(est->spectrum_size > 0);
  CHECK(std::isfinite(est->exact_fit_pvalue));
  CHECK(est->exact_fit_pvalue >= 0.0);
  CHECK(est->exact_fit_pvalue <= 1.0);

  // Fixed-weight comparator: drops the gamma channel, so its bias differs from
  // the estimated-weight one under misspecification (the channel is active).
  auto fixed = magmaan::estimate::ordinal_rmsea_misspec_inference(
      *pt, *mr, *stats, *fit,
      magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE(fixed.has_value());
  CHECK(fixed->fmin == doctest::Approx(est->fmin).epsilon(1e-12));  // same F
  CHECK(std::isfinite(fixed->grad_var));
  CHECK(std::abs(fixed->bias_trace - est->bias_trace) > 1e-8);  // gamma active
}

TEST_CASE("ordinal_cfi_tli_misspec_inference: joint two-model law for CFI/TLI") {
  // Tau-equivalence on unequal loadings => misspecified user model => live
  // gamma channel and a detectable incremental-fit gap.
  const Eigen::MatrixXd X =
      ordinal_test_block(20260627, 800, {0.82, 0.66, 0.52, 0.40}, -0.4, 0.7);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + 1*x2 + 1*x3 + 1*x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto prof = magmaan::estimate::ordinal_dwls_profile_rmsea(*pt, *mr, *stats,
                                                            *fit);
  REQUIRE(prof.has_value());
  auto fm = magmaan::estimate::fit_measures_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fm.has_value());

  auto est = magmaan::estimate::ordinal_cfi_tli_misspec_inference(*pt, *mr,
                                                                  *stats, *fit);
  REQUIRE_MESSAGE(est.has_value(),
      "cfi/tli inference failed: " << (est.has_value() ? "" : est.error().detail));

  // User statistic passes through the profile path; baseline matches the
  // analytic independence chi-square and its nominal df = ncorr = 6.
  CHECK(est->stat_user == doctest::Approx(prof->chisq_standard).epsilon(1e-9));
  CHECK(est->df_user == prof->df);
  CHECK(est->gendf_user == doctest::Approx(prof->trace_signed).epsilon(1e-9));
  CHECK(est->stat_baseline == doctest::Approx(fm->baseline.chi2).epsilon(1e-6));
  CHECK(est->df_baseline == fm->baseline.df);
  CHECK(est->df_baseline == 6);

  // Noncentralities are the bias-corrected statistics; baseline misfits grossly.
  CHECK(est->delta_user ==
        doctest::Approx(est->stat_user - est->gendf_user).epsilon(1e-12));
  CHECK(est->delta_baseline ==
        doctest::Approx(est->stat_baseline - est->gendf_baseline).epsilon(1e-12));
  CHECK(est->delta_baseline > est->delta_user);

  // CFI is a proper [0,1] index with an ordered, finite interval.
  CHECK(est->cfi >= 0.0);
  CHECK(est->cfi <= 1.0);
  CHECK(est->cfi_ci_lower <= est->cfi_ci_upper);
  CHECK(est->cfi_ci_lower >= 0.0);
  CHECK(est->cfi_ci_upper <= 1.0);
  CHECK(std::isfinite(est->var_cfi));
  CHECK(est->var_cfi >= 0.0);
  CHECK(std::isfinite(est->var_user));
  CHECK(std::isfinite(est->var_baseline));
  CHECK(std::isfinite(est->cov_user_baseline));
  CHECK(est->var_user >= 0.0);
  CHECK(est->var_baseline >= 0.0);

  // TLI is the same ratio r rescaled by the generalized-df ratio c = Q̄_b/Q̄_u,
  // and its variance is c² times the CFI variance (Corollaries 1-2 of the note).
  const double r = est->delta_user / est->delta_baseline;
  const double c = est->gendf_baseline / est->gendf_user;
  CHECK(est->tli == doctest::Approx(1.0 - c * r).epsilon(1e-10));
  CHECK(est->var_tli == doctest::Approx(c * c * est->var_cfi).epsilon(1e-10));
  CHECK(est->tli_ci_lower <= est->tli_ci_upper);
  CHECK(std::isfinite(est->tli));

  // Baseline-dominated regime: to leading order Var(CFI) ≈ V_uu / δ_b², the
  // user-statistic variance scaled by the squared baseline noncentrality. The
  // full bivariate form must stay within the O(r) corrections of that leading
  // term (here r is small because the baseline dwarfs the user misfit).
  const double db = est->delta_baseline;
  const double leading = est->var_user / (db * db);
  CHECK(std::abs(est->var_cfi - leading) <= 0.5 * leading + 1e-12);

  // Fixed-weight comparator: drops the gamma channel, so the generalized df (and
  // hence the noncentralities and variances) differ under misspecification while
  // the raw statistics are unchanged.
  auto fixed = magmaan::estimate::ordinal_cfi_tli_misspec_inference(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE(fixed.has_value());
  CHECK(fixed->stat_user == doctest::Approx(est->stat_user).epsilon(1e-12));
  CHECK(fixed->stat_baseline ==
        doctest::Approx(est->stat_baseline).epsilon(1e-12));
  CHECK(std::abs(fixed->gendf_user - est->gendf_user) > 1e-8);  // gamma active
  CHECK(std::isfinite(fixed->cfi));
  CHECK(std::isfinite(fixed->var_cfi));
}

TEST_CASE("ordinal_fit_measures_misspec_inference bundles the per-index results") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260628, 800, {0.82, 0.66, 0.52, 0.40}, -0.4, 0.7);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + 1*x2 + 1*x3 + 1*x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  using magmaan::estimate::OrdinalParameterization;
  auto rm = magmaan::estimate::ordinal_rmsea_misspec_inference(*pt, *mr, *stats,
                                                               *fit);
  auto cr = magmaan::estimate::ordinal_crmr_misspec_inference(*pt, *mr, *stats,
                                                              *fit);
  auto sr = magmaan::estimate::ordinal_crmr_misspec_inference(
      *pt, *mr, *stats, *fit, OrdinalParameterization::Delta,
      /*estimated_weight=*/true, /*srmr_denominator=*/true);
  auto ct = magmaan::estimate::ordinal_cfi_tli_misspec_inference(*pt, *mr,
                                                                 *stats, *fit);
  REQUIRE(rm.has_value());
  REQUIRE(cr.has_value());
  REQUIRE(sr.has_value());
  REQUIRE(ct.has_value());

  auto all = magmaan::estimate::ordinal_fit_measures_misspec_inference(
      *pt, *mr, *stats, *fit);
  REQUIRE_MESSAGE(all.has_value(),
      "fit measures inference failed: "
          << (all.has_value() ? "" : all.error().detail));

  // The bundle equals the per-index entry points exactly (it delegates).
  CHECK(all->rmsea == doctest::Approx(rm->point).epsilon(1e-12));
  CHECK(all->rmsea_ci_lower == doctest::Approx(rm->ci_lower).epsilon(1e-12));
  CHECK(all->rmsea_ci_upper == doctest::Approx(rm->ci_upper).epsilon(1e-12));
  CHECK(all->rmsea_pvalue == doctest::Approx(rm->exact_fit_pvalue).epsilon(1e-12));
  CHECK(all->crmr == doctest::Approx(cr->point).epsilon(1e-12));
  CHECK(all->crmr_ci_lower == doctest::Approx(cr->ci_lower).epsilon(1e-12));
  CHECK(all->crmr_pvalue == doctest::Approx(cr->exact_fit_pvalue).epsilon(1e-12));
  CHECK(all->cfi == doctest::Approx(ct->cfi).epsilon(1e-12));
  CHECK(all->cfi_ci_lower == doctest::Approx(ct->cfi_ci_lower).epsilon(1e-12));
  CHECK(all->tli == doctest::Approx(ct->tli).epsilon(1e-12));
  CHECK(all->tli_ci_upper == doctest::Approx(ct->tli_ci_upper).epsilon(1e-12));
  CHECK(all->stat_user == doctest::Approx(ct->stat_user).epsilon(1e-12));
  CHECK(all->df_baseline == ct->df_baseline);

  // SRMR is the CRMR result rescaled to the vech denominator (p=4: 6 -> 10).
  const double scale = std::sqrt(6.0 / 10.0);
  CHECK(all->srmr == doctest::Approx(cr->point * scale).epsilon(1e-12));
  CHECK(all->srmr == doctest::Approx(sr->point).epsilon(1e-9));
  CHECK(all->srmr_ci_lower == doctest::Approx(sr->ci_lower).epsilon(1e-9));
  CHECK(all->srmr_ci_upper == doctest::Approx(sr->ci_upper).epsilon(1e-9));
  CHECK(all->conf_level == doctest::Approx(0.90));
  CHECK_FALSE(all->fixed_weight);
}

TEST_CASE("misspec fit-index inference is multi-group (duplicate-group reduction)") {
  // Tau-equivalence on unequal loadings => misspecified, so the indices are
  // non-trivial. Duplicating the data into a configural two-group fit must leave
  // every index UNCHANGED (the pooled statistic and df both double, so the
  // bias-corrected criteria and the CFI/TLI ratios are invariant). This pins the
  // n_b-weighted pooling and the per-block γ channel against the single-group
  // path that the earlier cases already validate.
  const char* syntax =
      "f =~ x1 + 1*x2 + 1*x3 + 1*x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  const Eigen::MatrixXd X =
      ordinal_test_block(20260629, 700, {0.82, 0.66, 0.52, 0.40}, -0.4, 0.7);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  opts.ftol = 1e-13;
  opts.gtol = 1e-9;
  auto fit_for = [&](const std::vector<Eigen::MatrixXd>& blocks, int n_groups) {
    auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true);
    REQUIRE(stats.has_value());
    auto fp = magmaan::parse::Parser::parse(syntax);
    REQUIRE(fp.has_value());
    magmaan::spec::BuildOptions bo;
    bo.n_groups = n_groups;
    auto pt = magmaan::spec::build(*fp, bo);
    REQUIRE(pt.has_value());
    auto mr = magmaan::model::build_matrix_rep(*pt);
    REQUIRE(mr.has_value());
    auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
    REQUIRE(x0.has_value());
    auto fit = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    REQUIRE_MESSAGE(fit.has_value(),
        "fit failed: " << (fit.has_value() ? "" : fit.error().detail));
    return std::make_tuple(std::move(*stats), std::move(*pt), std::move(*mr),
                           std::move(*fit));
  };

  auto [s1, pt1, mr1, f1] = fit_for({X}, 1);
  auto [s2, pt2, mr2, f2] = fit_for({X, X}, 2);  // two identical groups

  auto a1 = magmaan::estimate::ordinal_fit_measures_misspec_inference(
      pt1, mr1, s1, f1);
  auto a2 = magmaan::estimate::ordinal_fit_measures_misspec_inference(
      pt2, mr2, s2, f2);
  REQUIRE_MESSAGE(a1.has_value(),
      "single-group bundle failed: " << (a1.has_value() ? "" : a1.error().detail));
  REQUIRE_MESSAGE(a2.has_value(),
      "two-group bundle failed: " << (a2.has_value() ? "" : a2.error().detail));

  // The duplicated fit doubles the raw statistic and df, leaving every POINT
  // index invariant (they are functions of population discrepancy ratios). The
  // intervals are NOT invariant: twice the data tightens them, so they should
  // be narrower, not equal.
  CHECK(a2->stat_user == doctest::Approx(2.0 * a1->stat_user).epsilon(2e-3));
  CHECK(a2->stat_baseline == doctest::Approx(2.0 * a1->stat_baseline).epsilon(2e-3));
  CHECK(a2->df_user == 2 * a1->df_user);
  CHECK(a2->df_baseline == 2 * a1->df_baseline);
  CHECK(a2->rmsea == doctest::Approx(a1->rmsea).epsilon(2e-3));
  CHECK(a2->cfi == doctest::Approx(a1->cfi).epsilon(2e-3));
  CHECK(a2->tli == doctest::Approx(a1->tli).epsilon(5e-3));
  CHECK(a2->crmr == doctest::Approx(a1->crmr).epsilon(2e-3));
  CHECK(a2->srmr == doctest::Approx(a1->srmr).epsilon(2e-3));

  // Intervals stay ordered and tighten with the doubled sample.
  CHECK(a2->rmsea_ci_lower <= a2->rmsea_ci_upper);
  CHECK(a2->cfi_ci_lower <= a2->cfi_ci_upper);
  const double w1 = a1->rmsea_ci_upper - a1->rmsea_ci_lower;
  const double w2 = a2->rmsea_ci_upper - a2->rmsea_ci_lower;
  CHECK(w2 < w1);  // 2x data -> narrower RMSEA interval

  // Two genuinely different groups: everything runs, intervals are ordered, and
  // the estimated-weight γ channel is active (differs from the fixed-weight
  // comparator).
  const Eigen::MatrixXd Y =
      ordinal_test_block(20260630, 500, {0.78, 0.70, 0.55, 0.45}, -0.3, 0.8);
  auto [s3, pt3, mr3, f3] = fit_for({X, Y}, 2);
  auto est = magmaan::estimate::ordinal_fit_measures_misspec_inference(
      pt3, mr3, s3, f3);
  auto fix = magmaan::estimate::ordinal_fit_measures_misspec_inference(
      pt3, mr3, s3, f3, magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE_MESSAGE(est.has_value(),
      "two-group different bundle failed: "
          << (est.has_value() ? "" : est.error().detail));
  REQUIRE(fix.has_value());
  CHECK(est->df_baseline == 12);  // 2 groups * ncorr 6
  CHECK(est->rmsea >= 0.0);
  CHECK(est->rmsea_ci_lower <= est->rmsea_ci_upper);
  CHECK(est->cfi >= 0.0);
  CHECK(est->cfi <= 1.0);
  CHECK(est->cfi_ci_lower <= est->cfi_ci_upper);
  CHECK(est->crmr_ci_lower <= est->crmr_ci_upper);
  CHECK(std::isfinite(est->tli));
  // Raw statistics are weight-independent; the bias correction is not.
  CHECK(fix->stat_user == doctest::Approx(est->stat_user).epsilon(1e-9));
  CHECK(std::abs(fix->rmsea - est->rmsea) >= 0.0);  // both finite, may differ
  CHECK(std::isfinite(fix->cfi));
}

TEST_CASE("ordinal CRMR multi-group pooling: quadratic-form vs lavaan per-group-root") {
  // ordinal_crmr_misspec_inference reports CRMR on the pooled quadratic-form
  // scale, point = sqrt(Σ_b (n_b/N)‖r_b‖²/k), consistent with its statistic N·G
  // and its CI. fit_measures_ordinal / ordinal_crmr use lavaan's pooling, a
  // sample-size-weighted mean of per-group roots, Σ_b (n_b/N)·sqrt(‖r_b‖²/k).
  // The two agree at one group; for G>1, by Jensen (sqrt concave),
  //   lavaan crmr ≤ misspec point,  strict once per-group misfit differs.
  // This pins the single-group identity and documents the deliberate multi-group
  // divergence: the misspec point is NOT lavaan's multi-group CRMR.
  const char* syntax =
      "f =~ x1 + 1*x2 + 1*x3 + 1*x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  opts.ftol = 1e-13;
  opts.gtol = 1e-9;
  auto fit_for = [&](const std::vector<Eigen::MatrixXd>& blocks, int n_groups) {
    auto stats = magmaan::data::ordinal_stats_from_integer_data(blocks, true);
    REQUIRE(stats.has_value());
    auto fp = magmaan::parse::Parser::parse(syntax);
    REQUIRE(fp.has_value());
    magmaan::spec::BuildOptions bo;
    bo.n_groups = n_groups;
    auto pt = magmaan::spec::build(*fp, bo);
    REQUIRE(pt.has_value());
    auto mr = magmaan::model::build_matrix_rep(*pt);
    REQUIRE(mr.has_value());
    auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
    REQUIRE(x0.has_value());
    auto fit = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    REQUIRE_MESSAGE(fit.has_value(),
        "fit failed: " << (fit.has_value() ? "" : fit.error().detail));
    return std::make_tuple(std::move(*stats), std::move(*pt), std::move(*mr),
                           std::move(*fit));
  };
  auto crmr_point = [&](const auto& pt, const auto& mr, const auto& s,
                        const auto& f) {
    auto inf = magmaan::estimate::ordinal_crmr_misspec_inference(pt, mr, s, f);
    REQUIRE_MESSAGE(inf.has_value(),
        "crmr inference failed: " << (inf.has_value() ? "" : inf.error().detail));
    return inf->point;
  };
  auto fm_crmr = [&](const auto& pt, const auto& mr, const auto& s,
                     const auto& f) {
    auto fm = magmaan::estimate::fit_measures_ordinal(
        pt, mr, s, f, magmaan::estimate::OrdinalWeightKind::DWLS,
        magmaan::estimate::OrdinalParameterization::Delta);
    REQUIRE(fm.has_value());
    return fm->crmr;
  };

  // Two groups with deliberately different misfit so the poolings diverge.
  const Eigen::MatrixXd X =
      ordinal_test_block(20260629, 700, {0.82, 0.66, 0.52, 0.40}, -0.4, 0.7);
  const Eigen::MatrixXd Y =
      ordinal_test_block(20260630, 500, {0.78, 0.70, 0.55, 0.45}, -0.3, 0.8);

  // Single group: the misspec point equals the lavaan CRMR exactly.
  {
    auto [s, pt, mr, f] = fit_for({X}, 1);
    CHECK(crmr_point(pt, mr, s, f) ==
          doctest::Approx(fm_crmr(pt, mr, s, f)).epsilon(1e-9));
  }

  // Two genuinely different groups: the poolings diverge, with the misspec
  // (root-of-pooled-mean) point the larger of the two.
  {
    auto [s, pt, mr, f] = fit_for({X, Y}, 2);
    const double point = crmr_point(pt, mr, s, f);
    const double crmr = fm_crmr(pt, mr, s, f);
    CHECK(point >= crmr - 1e-12);       // Jensen: mean-of-roots ≤ root-of-mean
    CHECK(point > crmr + 1e-6);         // and the gap is material, not rounding
  }
}

TEST_CASE("mixed_ordinal_dwls_profile_rmsea assembles the extended (u, gamma) law") {
  std::mt19937 rng(20260630);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.35);
    X(i, 1) = 1.0 + (0.70 * eta + 0.72 * norm(rng) > 0.05);
    X(i, 2) = 0.76 * eta + 0.64 * norm(rng) + 0.20;
    X(i, 3) = 0.66 * eta + 0.75 * norm(rng) - 0.10;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());
  REQUIRE(stats->moment_influence.size() == 1);
  REQUIRE(stats->raw_data.size() == 1);

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(fit.has_value(),
      "mixed DWLS fit failed: "
          << (fit.has_value() ? "" : fit.error().detail));

  auto rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Expected);
  REQUIRE(rob.has_value());

  auto prof = magmaan::estimate::mixed_ordinal_dwls_profile_rmsea(
      *pt, *mr, *stats, *fit);
  REQUIRE_MESSAGE(prof.has_value(),
      "mixed profile RMSEA failed: "
          << (prof.has_value() ? "" : prof.error().detail));

  const Eigen::Index m = stats->moments[0].size();
  REQUIRE(prof->profile_hessian.rows() == 2 * m);
  REQUIRE(prof->gamma.rows() == 2 * m);
  CHECK(prof->df == rob->df);
  CHECK(prof->chisq_standard ==
        doctest::Approx(rob->chisq_standard).epsilon(1e-8));
  CHECK(prof->spectrum_size > 0);
  CHECK(prof->bias_trace > 0.0);
  CHECK(prof->rmsea >= 0.0);
  CHECK(prof->gamma.topLeftCorner(m, m).isApprox(stats->NACOV[0], 1e-9));
  CHECK(prof->gamma.bottomRightCorner(m, m).cwiseAbs().maxCoeff() > 0.0);
  CHECK(prof->gamma.topRightCorner(m, m).cwiseAbs().maxCoeff() > 0.0);

  auto rmsea = magmaan::estimate::mixed_ordinal_rmsea_misspec_inference(
      *pt, *mr, *stats, *fit);
  auto fixed = magmaan::estimate::mixed_ordinal_rmsea_misspec_inference(
      *pt, *mr, *stats, *fit,
      magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE_MESSAGE(rmsea.has_value(),
      "mixed RMSEA inference failed: "
          << (rmsea.has_value() ? "" : rmsea.error().detail));
  REQUIRE_MESSAGE(fixed.has_value(),
      "mixed fixed-weight RMSEA inference failed: "
          << (fixed.has_value() ? "" : fixed.error().detail));
  CHECK_FALSE(rmsea->fixed_weight);
  CHECK(fixed->fixed_weight);
  CHECK(rmsea->point == doctest::Approx(prof->rmsea).epsilon(1e-12));
  CHECK(rmsea->stat == doctest::Approx(prof->chisq_standard).epsilon(1e-12));
  CHECK(rmsea->df == prof->df);
  CHECK(rmsea->ci_lower <= rmsea->ci_upper);
  CHECK(std::isfinite(rmsea->exact_fit_pvalue));
  CHECK(rmsea->exact_fit_pvalue >= 0.0);
  CHECK(rmsea->exact_fit_pvalue <= 1.0);
  CHECK(fixed->stat == doctest::Approx(rmsea->stat).epsilon(1e-12));
  CHECK(fixed->df == rmsea->df);
  CHECK(std::abs(fixed->bias_trace - rmsea->bias_trace) > 1e-10);

  auto crmr = magmaan::estimate::mixed_ordinal_crmr_misspec_inference(
      *pt, *mr, *stats, *fit);
  auto srmr = magmaan::estimate::mixed_ordinal_crmr_misspec_inference(
      *pt, *mr, *stats, *fit,
      magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/true, /*srmr_denominator=*/true);
  auto crmr_fixed = magmaan::estimate::mixed_ordinal_crmr_misspec_inference(
      *pt, *mr, *stats, *fit,
      magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE_MESSAGE(crmr.has_value(),
      "mixed CRMR inference failed: "
          << (crmr.has_value() ? "" : crmr.error().detail));
  REQUIRE_MESSAGE(srmr.has_value(),
      "mixed SRMR inference failed: "
          << (srmr.has_value() ? "" : srmr.error().detail));
  REQUIRE_MESSAGE(crmr_fixed.has_value(),
      "mixed fixed-weight CRMR inference failed: "
          << (crmr_fixed.has_value() ? "" : crmr_fixed.error().detail));
  auto fm = magmaan::estimate::fit_measures_mixed_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::estimate::OrdinalParameterization::Delta);
  REQUIRE(fm.has_value());
  CHECK(crmr->k == 6);
  CHECK(srmr->k == 10);
  CHECK(srmr->srmr_denominator);
  CHECK(crmr_fixed->fixed_weight);
  CHECK(crmr->point ==
        doctest::Approx(srmr->point * std::sqrt(10.0 / 6.0)).epsilon(1e-9));
  CHECK(srmr->point == doctest::Approx(fm->srmr).epsilon(1e-9));
  CHECK(crmr->ci_lower <= crmr->ci_upper);
  CHECK(srmr->ci_lower <= srmr->ci_upper);
  CHECK(std::isfinite(crmr->exact_fit_pvalue));
  CHECK(crmr->exact_fit_pvalue >= 0.0);
  CHECK(crmr->exact_fit_pvalue <= 1.0);
  CHECK(crmr_fixed->stat == doctest::Approx(crmr->stat).epsilon(1e-12));
  CHECK(crmr_fixed->k == crmr->k);

  auto cfi = magmaan::estimate::mixed_ordinal_cfi_tli_misspec_inference(
      *pt, *mr, *stats, *fit);
  auto cfi_fixed = magmaan::estimate::mixed_ordinal_cfi_tli_misspec_inference(
      *pt, *mr, *stats, *fit,
      magmaan::estimate::OrdinalParameterization::Delta,
      /*estimated_weight=*/false);
  REQUIRE_MESSAGE(cfi.has_value(),
      "mixed CFI/TLI inference failed: "
          << (cfi.has_value() ? "" : cfi.error().detail));
  REQUIRE_MESSAGE(cfi_fixed.has_value(),
      "mixed fixed-weight CFI/TLI inference failed: "
          << (cfi_fixed.has_value() ? "" : cfi_fixed.error().detail));
  CHECK(cfi->stat_user == doctest::Approx(prof->chisq_standard).epsilon(1e-9));
  CHECK(cfi->df_user == prof->df);
  CHECK(cfi->stat_baseline ==
        doctest::Approx(fm->baseline.chi2).epsilon(1e-9));
  CHECK(cfi->df_baseline == fm->baseline.df);
  CHECK(cfi->df_baseline == 6);
  CHECK(cfi->delta_user ==
        doctest::Approx(cfi->stat_user - cfi->gendf_user).epsilon(1e-12));
  CHECK(cfi->delta_baseline ==
        doctest::Approx(cfi->stat_baseline - cfi->gendf_baseline).epsilon(1e-12));
  CHECK(cfi->delta_baseline > 0.0);
  CHECK(cfi->cfi >= 0.0);
  CHECK(cfi->cfi <= 1.0);
  CHECK(cfi->cfi_ci_lower <= cfi->cfi_ci_upper);
  CHECK(cfi->cfi_ci_lower >= 0.0);
  CHECK(cfi->cfi_ci_upper <= 1.0);
  CHECK(std::isfinite(cfi->var_cfi));
  CHECK(std::isfinite(cfi->var_user));
  CHECK(std::isfinite(cfi->var_baseline));
  CHECK(std::isfinite(cfi->cov_user_baseline));
  CHECK(cfi->var_cfi >= 0.0);
  CHECK(cfi->var_user >= 0.0);
  CHECK(cfi->var_baseline >= 0.0);
  if (cfi->gendf_user > 0.0) {
    const double r = cfi->delta_user / cfi->delta_baseline;
    const double cc = cfi->gendf_baseline / cfi->gendf_user;
    CHECK(cfi->tli == doctest::Approx(1.0 - cc * r).epsilon(1e-10));
    CHECK(cfi->var_tli == doctest::Approx(cc * cc * cfi->var_cfi).epsilon(1e-10));
    CHECK(std::isfinite(cfi->tli));
  }
  CHECK(cfi_fixed->stat_user == doctest::Approx(cfi->stat_user).epsilon(1e-12));
  CHECK(cfi_fixed->stat_baseline ==
        doctest::Approx(cfi->stat_baseline).epsilon(1e-12));
  CHECK(cfi_fixed->fixed_weight);

  auto all = magmaan::estimate::mixed_ordinal_fit_measures_misspec_inference(
      *pt, *mr, *stats, *fit);
  auto all_fixed =
      magmaan::estimate::mixed_ordinal_fit_measures_misspec_inference(
          *pt, *mr, *stats, *fit,
          magmaan::estimate::OrdinalParameterization::Delta,
          /*estimated_weight=*/false);
  REQUIRE_MESSAGE(all.has_value(),
      "mixed fit-measures inference failed: "
          << (all.has_value() ? "" : all.error().detail));
  REQUIRE_MESSAGE(all_fixed.has_value(),
      "mixed fixed-weight fit-measures inference failed: "
          << (all_fixed.has_value() ? "" : all_fixed.error().detail));
  CHECK(all->rmsea == doctest::Approx(rmsea->point).epsilon(1e-12));
  CHECK(all->rmsea_ci_lower == doctest::Approx(rmsea->ci_lower).epsilon(1e-12));
  CHECK(all->rmsea_ci_upper == doctest::Approx(rmsea->ci_upper).epsilon(1e-12));
  CHECK(all->rmsea_pvalue ==
        doctest::Approx(rmsea->exact_fit_pvalue).epsilon(1e-12));
  CHECK(all->crmr == doctest::Approx(crmr->point).epsilon(1e-12));
  CHECK(all->crmr_ci_lower == doctest::Approx(crmr->ci_lower).epsilon(1e-12));
  CHECK(all->crmr_ci_upper == doctest::Approx(crmr->ci_upper).epsilon(1e-12));
  CHECK(all->crmr_pvalue ==
        doctest::Approx(crmr->exact_fit_pvalue).epsilon(1e-12));
  CHECK(all->srmr == doctest::Approx(srmr->point).epsilon(1e-12));
  CHECK(all->srmr_ci_lower == doctest::Approx(srmr->ci_lower).epsilon(1e-12));
  CHECK(all->srmr_ci_upper == doctest::Approx(srmr->ci_upper).epsilon(1e-12));
  CHECK(all->cfi == doctest::Approx(cfi->cfi).epsilon(1e-12));
  CHECK(all->cfi_ci_lower == doctest::Approx(cfi->cfi_ci_lower).epsilon(1e-12));
  CHECK(all->cfi_ci_upper == doctest::Approx(cfi->cfi_ci_upper).epsilon(1e-12));
  CHECK(all->tli == doctest::Approx(cfi->tli).epsilon(1e-12));
  CHECK(all->tli_ci_lower == doctest::Approx(cfi->tli_ci_lower).epsilon(1e-12));
  CHECK(all->tli_ci_upper == doctest::Approx(cfi->tli_ci_upper).epsilon(1e-12));
  CHECK(all->stat_user == doctest::Approx(cfi->stat_user).epsilon(1e-12));
  CHECK(all->stat_baseline ==
        doctest::Approx(cfi->stat_baseline).epsilon(1e-12));
  CHECK(all->df_user == cfi->df_user);
  CHECK(all->df_baseline == cfi->df_baseline);
  CHECK(all->conf_level == doctest::Approx(0.90));
  CHECK_FALSE(all->fixed_weight);
  CHECK(all_fixed->fixed_weight);
  CHECK(all_fixed->stat_user == doctest::Approx(all->stat_user).epsilon(1e-12));
  CHECK(all_fixed->stat_baseline ==
        doctest::Approx(all->stat_baseline).epsilon(1e-12));

  auto no_raw = *stats;
  no_raw.raw_data.clear();
  auto missing = magmaan::estimate::mixed_ordinal_dwls_profile_rmsea(
      *pt, *mr, no_raw, *fit);
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error().detail.find("estimated-weight influence unavailable") !=
        std::string::npos);
}

TEST_CASE("mixed_ordinal_dwls_profile_lrt compares nested mixed DWLS models") {
  std::mt19937 rng(20260701);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(620, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.55) + (eta > 0.40);
    X(i, 1) = 1.0 + (0.68 * eta + 0.74 * norm(rng) > 0.10);
    X(i, 2) = 0.74 * eta + 0.66 * norm(rng) + 0.12;
    X(i, 3) = 0.64 * eta + 0.77 * norm(rng) - 0.08;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  const char* thresholds =
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n";
  const std::string syntax_h1 =
      std::string("f =~ x1 + x2 + x3 + x4\n") + thresholds;
  const std::string syntax_h0 =
      std::string("f =~ x1 + 1*x2 + 1*x3 + 1*x4\n") + thresholds;

  auto build_fit = [&](const std::string& syntax) {
    auto fp = magmaan::parse::Parser::parse(syntax);
    REQUIRE(fp.has_value());
    auto pt = magmaan::spec::build(*fp, opts);
    REQUIRE(pt.has_value());
    auto mr = magmaan::model::build_matrix_rep(*pt);
    REQUIRE(mr.has_value());
    auto fit = magmaan::test::fit_mixed_ordinal_bounded(
        *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
    REQUIRE_MESSAGE(fit.has_value(),
        "mixed DWLS fit failed: "
            << (fit.has_value() ? "" : fit.error().detail));
    return std::make_tuple(std::move(*pt), std::move(*mr), std::move(*fit));
  };

  auto [pt1, mr1, fit1] = build_fit(syntax_h1);
  auto [pt0, mr0, fit0] = build_fit(syntax_h0);

  auto rob1 = magmaan::estimate::robust_mixed_ordinal(
      pt1, mr1, *stats, fit1, magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Expected);
  auto rob0 = magmaan::estimate::robust_mixed_ordinal(
      pt0, mr0, *stats, fit0, magmaan::estimate::OrdinalWeightKind::DWLS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Expected);
  REQUIRE(rob1.has_value());
  REQUIRE(rob0.has_value());

  auto lrt = magmaan::estimate::mixed_ordinal_dwls_profile_lrt(
      pt1, mr1, *stats, fit1, pt0, mr0, fit0);
  REQUIRE_MESSAGE(lrt.has_value(),
      "mixed profile LRT failed: "
          << (lrt.has_value() ? "" : lrt.error().detail));

  CHECK(lrt->df_diff == rob0->df - rob1->df);
  CHECK(lrt->df_diff > 0);
  CHECK(lrt->spectrum_size > 0);
  CHECK(lrt->T_diff >= -1e-6);
  CHECK(std::isfinite(lrt->p_mixture));
  CHECK(lrt->p_mixture >= 0.0);
  CHECK(lrt->p_mixture <= 1.0);

  auto ev1 = magmaan::model::ModelEvaluator::build(pt1, mr1);
  REQUIRE(ev1.has_value());
  Eigen::Index k_loading = -1;
  const auto locs = ev1->param_locations();
  for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(locs.size()); ++k) {
    const auto& loc = locs[static_cast<std::size_t>(k)];
    if (loc.mat == magmaan::model::MatId::Lambda &&
        loc.row == 1 && loc.col == 0) {
      k_loading = k;
      break;
    }
  }
  REQUIRE(k_loading >= 0);

  magmaan::optim::OptimOptions prof_opts;
  prof_opts.max_iter = 500;
  prof_opts.ftol = 1e-10;
  prof_opts.gtol = 1e-7;
  const double target = 0.96 * fit1.theta(k_loading);
  auto param_lrt =
      magmaan::estimate::frontier::profile_lrt_parameter_mixed_ordinal(
          pt1, mr1, *stats, fit1, k_loading, target, {},
          magmaan::estimate::OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, prof_opts,
          magmaan::estimate::OrdinalParameterization::Delta);
  REQUIRE_MESSAGE(param_lrt.has_value(),
      "mixed ordinal parameter profile LRT failed: "
          << (param_lrt.has_value() ? "" : param_lrt.error().detail));
  CHECK(param_lrt->unrestricted_value ==
        doctest::Approx(fit1.theta(k_loading)));
  CHECK(param_lrt->constrained_value ==
        doctest::Approx(target).epsilon(1e-7));
  CHECK(std::abs(param_lrt->constraint_residual) < 1e-7);
  CHECK(param_lrt->T == doctest::Approx(
      2.0 * 620.0 * (param_lrt->fmin_constrained - fit1.fmin)));
  CHECK(param_lrt->p_value == doctest::Approx(
      magmaan::inference::chi2_pvalue(param_lrt->T, 1)));
  CHECK(param_lrt->df == 1);

  auto robust_param_lrt =
      magmaan::estimate::frontier::profile_lrt_parameter_mixed_ordinal(
          pt1, mr1, *stats, fit1, k_loading, target, {},
          magmaan::estimate::OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, prof_opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true,
          magmaan::estimate::frontier::ScalarProfileReference::RobustScaled);
  REQUIRE_MESSAGE(robust_param_lrt.has_value(),
      "mixed ordinal robust parameter profile LRT failed: "
          << (robust_param_lrt.has_value()
                  ? "" : robust_param_lrt.error().detail));
  CHECK(robust_param_lrt->T == doctest::Approx(param_lrt->T).epsilon(1e-10));
  CHECK(robust_param_lrt->scaling_factor > 0.0);
  CHECK(std::isfinite(robust_param_lrt->T_scaled));

  auto misspec_param_lrt =
      magmaan::estimate::frontier::profile_lrt_parameter_mixed_ordinal(
          pt1, mr1, *stats, fit1, k_loading, target, {},
          magmaan::estimate::OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, prof_opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true,
          magmaan::estimate::frontier::ScalarProfileReference::MisspecMixture);
  REQUIRE_MESSAGE(misspec_param_lrt.has_value(),
      "mixed ordinal misspec parameter profile LRT failed: "
          << (misspec_param_lrt.has_value()
                  ? "" : misspec_param_lrt.error().detail));
  CHECK(misspec_param_lrt->misspec_scaling_factor > 0.0);
  CHECK(misspec_param_lrt->misspec_eigvals.size() == 1);
  CHECK(std::isfinite(misspec_param_lrt->p_value_misspec_mixture));

  magmaan::estimate::frontier::ScalarProfileCiOptions ci_opts;
  ci_opts.cutoff = 0.25;
  ci_opts.initial_step = 0.03 * std::abs(fit1.theta(k_loading));
  ci_opts.target_tol = 1e-4;
  ci_opts.statistic_tol = 1e-4;
  ci_opts.max_iter = 35;
  auto param_ci =
      magmaan::estimate::frontier::profile_lrt_ci_parameter_mixed_ordinal(
          pt1, mr1, *stats, fit1, k_loading, ci_opts, {},
          magmaan::estimate::OrdinalWeightKind::DWLS,
          magmaan::estimate::Backend::NloptSlsqp, prof_opts,
          magmaan::estimate::OrdinalParameterization::Delta);
  REQUIRE_MESSAGE(param_ci.has_value(),
      "mixed ordinal parameter profile CI failed: "
          << (param_ci.has_value() ? "" : param_ci.error().detail));
  CHECK(param_ci->lower < fit1.theta(k_loading));
  CHECK(param_ci->upper > fit1.theta(k_loading));
  CHECK(param_ci->lower_profile.T ==
        doctest::Approx(ci_opts.cutoff).epsilon(1e-3));
  CHECK(param_ci->upper_profile.T ==
        doctest::Approx(ci_opts.cutoff).epsilon(1e-3));
}
