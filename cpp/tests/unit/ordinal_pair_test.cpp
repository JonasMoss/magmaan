#include "ordinal_test_helpers.hpp"

TEST_CASE("Polychoric h-score API evaluates predefined caps") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  auto ml = magmaan::data::eval_polychoric_h_score(1.7);
  REQUIRE(ml.has_value());
  CHECK(ml->h == doctest::Approx(1.7));
  CHECK(ml->dh == doctest::Approx(1.0));
  CHECK(ml->phi == doctest::Approx(1.7 * std::log(1.7)));

  auto hard_inf = magmaan::data::eval_polychoric_h_score(
      2.4, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap,
               .k = std::numeric_limits<double>::infinity()});
  REQUIRE(hard_inf.has_value());
  CHECK(hard_inf->h == doctest::Approx(2.4));
  CHECK(hard_inf->dh == doctest::Approx(1.0));
  CHECK(hard_inf->phi == doctest::Approx(2.4 * std::log(2.4)));

  auto hard = magmaan::data::eval_polychoric_h_score(
      2.4, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap, .k = 1.6});
  REQUIRE(hard.has_value());
  CHECK(hard->h == doctest::Approx(1.6));
  CHECK(hard->dh == doctest::Approx(0.0));
  CHECK(hard->phi == doctest::Approx(2.4 * (std::log(1.6) + 1.0) - 1.6));

  auto hard_kink = magmaan::data::eval_polychoric_h_score(
      1.6, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap, .k = 1.6});
  REQUIRE(hard_kink.has_value());
  CHECK(hard_kink->h == doctest::Approx(1.6));
  CHECK(hard_kink->dh == doctest::Approx(0.0));

  auto smooth_low = magmaan::data::eval_polychoric_h_score(
      1.2, PolychoricHScoreOptions{.kind = PolychoricHScoreKind::SmoothCap});
  REQUIRE(smooth_low.has_value());
  CHECK(smooth_low->h == doctest::Approx(1.2));
  CHECK(smooth_low->dh == doctest::Approx(1.0));
  CHECK(smooth_low->phi == doctest::Approx(1.2 * std::log(1.2)));

  auto smooth_high = magmaan::data::eval_polychoric_h_score(
      2.8, PolychoricHScoreOptions{.kind = PolychoricHScoreKind::SmoothCap});
  REQUIRE(smooth_high.has_value());
  CHECK(smooth_high->h == doctest::Approx(1.9));
  CHECK(smooth_high->dh == doctest::Approx(0.0));
  CHECK(std::isfinite(smooth_high->phi));

  auto exp = magmaan::data::eval_polychoric_h_score(
      2.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::ExpCap, .k = 1.6, .lambda = 0.2});
  REQUIRE(exp.has_value());
  CHECK(exp->h == doctest::Approx(1.6 + 0.2 * (1.0 - std::exp(-2.0))));
  CHECK(exp->dh == doctest::Approx(std::exp(-2.0)));
  CHECK(std::isfinite(exp->phi));
}

TEST_CASE("Polychoric h-score objective contribution matches score identity") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  const PolychoricHScoreOptions options[] = {
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::ML},
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::WmaHardCap,
                              .k = 1.6},
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::SmoothCap},
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::ExpCap,
                              .k = 1.6,
                              .lambda = 0.2},
  };

  for (const auto& option : options) {
    const double t = 2.0;
    const double step = 1e-5;
    auto mid = magmaan::data::eval_polychoric_h_score(t, option);
    auto plus = magmaan::data::eval_polychoric_h_score(t + step, option);
    auto minus = magmaan::data::eval_polychoric_h_score(t - step, option);
    REQUIRE(mid.has_value());
    REQUIRE(plus.has_value());
    REQUIRE(minus.has_value());
    const double dphi = (plus->phi - minus->phi) / (2.0 * step);
    CHECK(mid->h == doctest::Approx(t * dphi - mid->phi).epsilon(1e-8));
  }
}

TEST_CASE("Polychoric h-score API rejects invalid inputs") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  auto negative = magmaan::data::eval_polychoric_h_score(-0.1);
  REQUIRE_FALSE(negative.has_value());
  CHECK(negative.error().detail.find("nonnegative") != std::string::npos);

  auto bad_hard = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap, .k = 0.8});
  REQUIRE_FALSE(bad_hard.has_value());
  CHECK(bad_hard.error().detail.find("at least 1") != std::string::npos);

  auto bad_hard_inf = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap,
               .k = -std::numeric_limits<double>::infinity()});
  REQUIRE_FALSE(bad_hard_inf.has_value());
  CHECK(bad_hard_inf.error().detail.find("at least 1") != std::string::npos);

  auto bad_smooth = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::SmoothCap, .a = 2.0, .b = 1.8});
  REQUIRE_FALSE(bad_smooth.has_value());
  CHECK(bad_smooth.error().detail.find("1 <= a < b") != std::string::npos);

  auto bad_exp = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::ExpCap, .k = 1.6, .lambda = 0.0});
  REQUIRE_FALSE(bad_exp.has_value());
  CHECK(bad_exp.error().detail.find("lambda > 0") != std::string::npos);
}

TEST_CASE("Ordinal pair ML kernel: probabilities, rho fit, and scores") {
  const double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd thi(2);
  thi << -0.4, 0.7;
  Eigen::VectorXd thj(1);
  thj << 0.2;

  double total = 0.0;
  const double bi[4] = {-inf, thi(0), thi(1), inf};
  const double bj[3] = {-inf, thj(0), inf};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 2; ++b) {
      total += magmaan::data::ordinal_bvn_rect_prob(
          bi[a], bi[a + 1], bj[b], bj[b + 1], 0.35);
    }
  }
  CHECK(total == doctest::Approx(1.0).epsilon(1e-8));

  const double h = 1e-5;
  const double pp = magmaan::data::ordinal_bvn_rect_prob(
      thi(0), thi(1), -inf, thj(0), 0.3 + h);
  const double pm = magmaan::data::ordinal_bvn_rect_prob(
      thi(0), thi(1), -inf, thj(0), 0.3 - h);
  const double fd = (pp - pm) / (2.0 * h);
  const double analytic = magmaan::data::ordinal_bvn_rect_drho(
      thi(0), thi(1), -inf, thj(0), 0.3);
  CHECK(analytic == doctest::Approx(fd).epsilon(1e-4));

  Eigen::VectorXi xi(6);
  Eigen::VectorXi xj(6);
  xi << 0, 0, 1, 1, 2, 2;
  xj << 0, 1, 0, 1, 0, 1;
  auto table = magmaan::data::ordinal_pair_table(xi, xj, 3, 2);
  REQUIRE(table.has_value());
  CHECK((*table)(0, 0) == 1.0);
  CHECK((*table)(0, 1) == 1.0);
  CHECK((*table)(2, 0) == 1.0);
  CHECK((*table)(2, 1) == 1.0);

  auto fit = magmaan::data::fit_ordinal_pair_rho_ml(*table, thi, thj);
  REQUIRE(fit.has_value());
  CHECK(std::isfinite(fit->rho));
  CHECK(std::isfinite(fit->negloglik));

  auto scores = magmaan::data::ordinal_pair_scores(xi, xj, fit->rho, thi, thj);
  REQUIRE(scores.has_value());
  CHECK(scores->rho.size() == 6);
  CHECK(scores->threshold_i.rows() == 6);
  CHECK(scores->threshold_i.cols() == 2);
  CHECK(scores->threshold_j.rows() == 6);
  CHECK(scores->threshold_j.cols() == 1);
  CHECK(scores->rho.allFinite());
  CHECK(scores->threshold_i.allFinite());
  CHECK(scores->threshold_j.allFinite());
}

TEST_CASE("Ordinal bvn corner grids reproduce per-cell rectangle values") {
  Eigen::VectorXd thi(3);
  thi << -1.1, -0.2, 0.9;
  Eigen::VectorXd thj(2);
  thj << -0.5, 0.6;
  const double rho = 0.37;
  const double inf = std::numeric_limits<double>::infinity();

  Eigen::MatrixXd cdf;
  Eigen::MatrixXd pdf;
  magmaan::data::ordinal_bvn_corner_cdf(thi, thj, rho, cdf);
  magmaan::data::ordinal_bvn_corner_pdf(thi, thj, rho, pdf);
  REQUIRE(cdf.rows() == thi.size() + 2);
  REQUIRE(cdf.cols() == thj.size() + 2);

  for (Eigen::Index a = 0; a < thi.size() + 1; ++a) {
    const double lo_i = (a == 0) ? -inf : thi(a - 1);
    const double hi_i = (a == thi.size()) ? inf : thi(a);
    for (Eigen::Index b = 0; b < thj.size() + 1; ++b) {
      const double lo_j = (b == 0) ? -inf : thj(b - 1);
      const double hi_j = (b == thj.size()) ? inf : thj(b);
      const double p_cell = cdf(a + 1, b + 1) - cdf(a, b + 1) -
                            cdf(a + 1, b) + cdf(a, b);
      const double d_cell = pdf(a + 1, b + 1) - pdf(a, b + 1) -
                            pdf(a + 1, b) + pdf(a, b);
      CHECK(p_cell == doctest::Approx(magmaan::data::ordinal_bvn_rect_prob(
          lo_i, hi_i, lo_j, hi_j, rho)).epsilon(1e-14));
      CHECK(d_cell == doctest::Approx(magmaan::data::ordinal_bvn_rect_drho(
          lo_i, hi_i, lo_j, hi_j, rho)).epsilon(1e-14));
    }
  }
}

TEST_CASE("Ordinal bvn cdf matches closed forms and a brute-force reference") {
  const double inf = std::numeric_limits<double>::infinity();
  const auto cdf = [&](double h, double k, double rho) {
    return magmaan::data::ordinal_bvn_rect_prob(-inf, h, -inf, k, rho);
  };
  const auto phi = [](double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
  };

  // P(X<0, Y<0) = 1/4 + asin(rho) / (2 pi), exact for all rho.
  constexpr double pi = 3.14159265358979323846;
  for (double rho : {-0.999, -0.95, -0.926, -0.9, -0.5, -0.2, 0.0,
                     0.3, 0.75, 0.924, 0.99, 0.9999}) {
    CHECK(cdf(0.0, 0.0, rho) ==
          doctest::Approx(0.25 + std::asin(rho) / (2.0 * pi)).epsilon(1e-13));
  }

  // Independence factorizes; the degenerate corners are exact.
  CHECK(cdf(0.7, -1.3, 0.0) == doctest::Approx(phi(0.7) * phi(-1.3)).epsilon(1e-14));
  CHECK(cdf(0.7, -1.3, 1.0) == doctest::Approx(phi(-1.3)).epsilon(1e-14));
  CHECK(cdf(0.7, -0.3, -1.0) ==
        doctest::Approx(std::max(0.0, phi(0.7) + phi(-0.3) - 1.0)).epsilon(1e-14));

  // Brute-force Simpson reference on Phi((k - rho z)/sd) phi(z) dz over
  // [-9, h]; the integrand truncation error is below Phi(-9) ~ 1.1e-19.
  const auto reference = [&](double h, double k, double rho) {
    const double lo = -9.0;
    const double hi = std::min(9.0, h);
    if (hi <= lo) return 0.0;
    // Simpson panels; the reference error is dominated by the steep Phi
    // transition at |rho| ~ 1 (~3e-9 at rho = 0.999), hence the 1e-8 gate
    // below rather than the implementation's own ~5e-16 accuracy.
    const int n = 4000;
    const double step = (hi - lo) / (2.0 * n);
    const double sd = std::sqrt(1.0 - rho * rho);
    const auto f = [&](double z) {
      constexpr double inv_sqrt_2pi = 0.39894228040143267794;
      return inv_sqrt_2pi * std::exp(-0.5 * z * z) *
             phi((k - rho * z) / sd);
    };
    double sum = f(lo) + f(hi);
    for (int i = 1; i < 2 * n; ++i) {
      sum += f(lo + i * step) * ((i % 2 == 1) ? 4.0 : 2.0);
    }
    return sum * step / 3.0;
  };
  for (double rho : {-0.999, -0.93, -0.6, -0.1, 0.0, 0.4, 0.85, 0.924,
                     0.926, 0.999}) {
    for (double h : {-2.5, -0.8, 0.0, 1.2, 3.0}) {
      for (double k : {-3.0, -1.1, 0.3, 2.2}) {
        CHECK(cdf(h, k, rho) ==
              doctest::Approx(reference(h, k, rho)).epsilon(1e-8));
      }
    }
  }
}

TEST_CASE("Ordinal pair ML rho search lands on a stationary minimum") {
  Eigen::VectorXd thi(3);
  thi << -1.0, -0.1, 0.8;
  Eigen::VectorXd thj(2);
  thj << -0.4, 0.5;
  Eigen::MatrixXd counts(4, 3);
  counts << 24.0,  9.0,  2.0,
            14.0, 21.0,  6.0,
             5.0, 17.0, 12.0,
             1.0,  8.0, 19.0;

  auto fit = magmaan::data::fit_ordinal_pair_rho_ml(counts, thi, thj);
  REQUIRE(fit.has_value());
  CHECK(!fit->hit_lower);
  CHECK(!fit->hit_upper);

  const double h = 1e-5;
  auto nll = [&](double r) {
    auto v = magmaan::data::ordinal_pair_negloglik(counts, thi, thj, r);
    REQUIRE(v.has_value());
    return *v;
  };
  const double f0 = nll(fit->rho);
  const double fp = nll(fit->rho + h);
  const double fm = nll(fit->rho - h);
  // Interior minimum: neighbors are no lower and the FD slope vanishes at
  // the curvature scale.
  CHECK(fp >= f0 - 1e-10);
  CHECK(fm >= f0 - 1e-10);
  const double slope = (fp - fm) / (2.0 * h);
  const double curvature = (fp - 2.0 * f0 + fm) / (h * h);
  CHECK(std::abs(slope) <= 1e-4 * std::max(1.0, std::abs(curvature)));
}

TEST_CASE("Polyserial ML rho search lands on a stationary minimum") {
  Eigen::VectorXd th(2);
  th << -0.6, 0.7;
  const Eigen::Index n = 160;
  Eigen::VectorXd u(n);
  Eigen::VectorXi cat(n);
  // Deterministic latent draws with a positive association.
  for (Eigen::Index i = 0; i < n; ++i) {
    const double z = std::cos(0.7 * static_cast<double>(i) + 0.3) +
                     0.4 * std::sin(1.9 * static_cast<double>(i));
    u(i) = z;
    const double y = 0.6 * z + 0.5 * std::sin(3.7 * static_cast<double>(i));
    cat(i) = (y < th(0)) ? 0 : (y < th(1)) ? 1 : 2;
  }
  u.array() -= u.mean();
  u /= std::sqrt(u.squaredNorm() / static_cast<double>(n));

  auto fit = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
  REQUIRE(fit.has_value());
  CHECK(!fit->hit_lower);
  CHECK(!fit->hit_upper);
  CHECK(fit->rho > 0.0);

  const double h = 1e-5;
  auto nll = [&](double r) {
    auto v = magmaan::data::polyserial_pair_negloglik(cat, u, th, r);
    REQUIRE(v.has_value());
    return *v;
  };
  const double f0 = nll(fit->rho);
  const double fp = nll(fit->rho + h);
  const double fm = nll(fit->rho - h);
  CHECK(fp >= f0 - 1e-10);
  CHECK(fm >= f0 - 1e-10);
  const double slope = (fp - fm) / (2.0 * h);
  const double curvature = (fp - 2.0 * f0 + fm) / (h * h);
  CHECK(std::abs(slope) <= 1e-4 * std::max(1.0, std::abs(curvature)));
}

TEST_CASE("Ordinal pair ML kernel: independence and lavaan 2x2 adjustment") {
  Eigen::VectorXd th(1);
  th << 0.0;

  Eigen::MatrixXd balanced(2, 2);
  balanced << 10.0, 10.0,
              10.0, 10.0;
  auto independent = magmaan::data::fit_ordinal_pair_rho_ml(balanced, th, th);
  REQUIRE(independent.has_value());
  CHECK(independent->rho == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(independent->iterations == 0);

  Eigen::MatrixXd sparse(2, 2);
  sparse << 0.0, 4.0,
            5.0, 6.0;
  auto adjusted = magmaan::data::fit_ordinal_pair_rho_ml(sparse, th, th);
  REQUIRE(adjusted.has_value());
  CHECK(adjusted->adjusted_counts(0, 0) == doctest::Approx(0.5));
  CHECK(adjusted->adjusted_counts(1, 1) == doctest::Approx(6.5));
  CHECK(adjusted->adjusted_counts(1, 0) == doctest::Approx(4.5));
  CHECK(adjusted->adjusted_counts(0, 1) == doctest::Approx(3.5));
  CHECK(std::isfinite(adjusted->rho));
  CHECK(adjusted->rho > -1.0);
  CHECK(adjusted->rho < 1.0);
}

TEST_CASE("Ordinal pair h-weighted rho preserves ML limits and diagnostics") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd thi(2);
  thi << -0.4, 0.7;
  Eigen::VectorXd thj(1);
  thj << 0.2;
  Eigen::MatrixXd counts(3, 2);
  counts << 8.0, 3.0,
            4.0, 7.0,
            2.0, 9.0;

  auto ml = magmaan::data::fit_ordinal_pair_rho_ml(counts, thi, thj);
  auto h_ml = magmaan::data::fit_ordinal_pair_rho_h_weighted(counts, thi, thj);
  auto hard_inf = magmaan::data::fit_ordinal_pair_rho_h_weighted(
      counts, thi, thj,
      magmaan::data::OrdinalPairHWeightedOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = std::numeric_limits<double>::infinity()}});
  REQUIRE(ml.has_value());
  REQUIRE(h_ml.has_value());
  REQUIRE(hard_inf.has_value());
  CHECK(h_ml->rho == doctest::Approx(ml->rho));
  CHECK(hard_inf->rho == doctest::Approx(ml->rho));
  CHECK(h_ml->converged);
  CHECK(hard_inf->converged);
  CHECK(std::isfinite(h_ml->objective));
  CHECK(std::isfinite(h_ml->score));
  CHECK(h_ml->probabilities.rows() == counts.rows());
  CHECK(h_ml->probabilities.cols() == counts.cols());
  CHECK(h_ml->expected_counts.rows() == counts.rows());
  CHECK(h_ml->residual_counts.isApprox(
      h_ml->adjusted_counts - h_ml->expected_counts, 1e-12));
  CHECK(h_ml->pearson_residuals.allFinite());
  CHECK(h_ml->weights.isApprox(Eigen::MatrixXd::Ones(3, 2), 1e-12));
  CHECK(hard_inf->weights.isApprox(Eigen::MatrixXd::Ones(3, 2), 1e-12));
}

TEST_CASE("Ordinal pair h-weighted rho downweights contaminated cells") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd th(2);
  th << -0.55, 0.75;
  const Eigen::MatrixXd clean = ordinal_expected_counts(th, th, 0.55, 5000.0);
  Eigen::MatrixXd contaminated = clean;
  contaminated(0, 2) += 900.0;

  auto clean_ml = magmaan::data::fit_ordinal_pair_rho_ml(clean, th, th);
  auto contaminated_ml =
      magmaan::data::fit_ordinal_pair_rho_ml(contaminated, th, th);
  auto robust = magmaan::data::fit_ordinal_pair_rho_h_weighted(
      contaminated, th, th,
      magmaan::data::OrdinalPairHWeightedOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = 1.15}});
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(robust.has_value());
  CHECK(robust->converged);
  CHECK(contaminated_ml->rho < clean_ml->rho);
  CHECK(robust->rho > contaminated_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(contaminated_ml->rho - clean_ml->rho));
  CHECK(robust->weights(0, 2) < 1.0);
  CHECK(robust->pearson_residuals(0, 2) > 0.0);
}

TEST_CASE("Polyserial pair ML kernel: likelihood, rho fit, and scores") {
  Eigen::VectorXi cat(8);
  cat << 0, 0, 1, 1, 1, 2, 2, 2;
  Eigen::VectorXd u(8);
  u << -1.4, -0.8, -0.5, 0.0, 0.4, 0.7, 1.1, 1.5;
  Eigen::VectorXd th(2);
  th << -0.45, 0.65;

  auto nll0 = magmaan::data::polyserial_pair_negloglik(cat, u, th, 0.0);
  auto nll1 = magmaan::data::polyserial_pair_negloglik(cat, u, th, 0.45);
  REQUIRE(nll0.has_value());
  REQUIRE(nll1.has_value());
  CHECK(std::isfinite(*nll0));
  CHECK(std::isfinite(*nll1));
  CHECK(*nll1 < *nll0);

  auto fit = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
  REQUIRE(fit.has_value());
  CHECK(std::isfinite(fit->rho));
  CHECK(std::isfinite(fit->negloglik));
  CHECK(fit->rho > 0.0);
  CHECK(fit->rho > -1.0);
  CHECK(fit->rho < 1.0);

  auto scores = magmaan::data::polyserial_pair_scores(cat, u, fit->rho, th);
  REQUIRE(scores.has_value());
  CHECK(scores->rho.size() == cat.size());
  CHECK(scores->thresholds.rows() == cat.size());
  CHECK(scores->thresholds.cols() == th.size());
  CHECK(scores->rho.allFinite());
  CHECK(scores->thresholds.allFinite());
  CHECK(scores->score_contributions.rows() == cat.size());
  CHECK(scores->score_contributions.cols() == th.size() + 1);
  CHECK(scores->score_contributions.leftCols(th.size()).isApprox(
      scores->thresholds, 0.0));
  CHECK(scores->score_contributions.col(th.size()).isApprox(scores->rho, 0.0));
  CHECK(scores->score_gamma.rows() == th.size() + 1);
  CHECK(scores->score_gamma.cols() == th.size() + 1);
  CHECK(scores->score_gamma.isApprox(
      (scores->score_contributions.transpose() * scores->score_contributions) /
          static_cast<double>(cat.size()),
      1e-12));
}

TEST_CASE("Polyserial pair joint DPD estimates thresholds and downweights continuous-tail discordance") {
  std::mt19937 rng(20260516);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::VectorXd th(2);
  th << -0.45, 0.65;

  constexpr Eigen::Index n_clean = 420;
  constexpr Eigen::Index n_bad = 35;
  Eigen::VectorXi cat_clean(n_clean);
  Eigen::VectorXd u_clean(n_clean);
  const double rho_true = 0.62;
  const double sd = std::sqrt(1.0 - rho_true * rho_true);
  for (Eigen::Index r = 0; r < n_clean; ++r) {
    const double u = norm(rng);
    const double z = rho_true * u + sd * norm(rng);
    u_clean(r) = u;
    cat_clean(r) = (z > th(0)) + (z > th(1));
  }

  Eigen::VectorXi cat_cont(n_clean + n_bad);
  Eigen::VectorXd u_cont(n_clean + n_bad);
  Eigen::VectorXd x_cont(n_clean + n_bad);
  cat_cont.head(n_clean) = cat_clean;
  u_cont.head(n_clean) = u_clean;
  x_cont.head(n_clean) = u_clean;
  for (Eigen::Index r = 0; r < n_bad; ++r) {
    cat_cont(n_clean + r) = 0;
    u_cont(n_clean + r) = 5.5 + 0.01 * static_cast<double>(r);
    x_cont(n_clean + r) = u_cont(n_clean + r);
  }

  auto clean_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_clean, u_clean, th);
  auto cont_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_cont, u_cont, th);
  auto robust = magmaan::data::fit_polyserial_pair_joint_dpd(
      cat_cont, x_cont,
      magmaan::data::PolyserialPairJointDpdOptions{.alpha = 0.5});

  REQUIRE(clean_ml.has_value());
  REQUIRE(cont_ml.has_value());
  REQUIRE(robust.has_value());
  CHECK(cont_ml->rho < clean_ml->rho);
  CHECK(robust->rho > cont_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(cont_ml->rho - clean_ml->rho));
  CHECK(robust->thresholds.size() == th.size());
  CHECK(robust->sd > 0.0);
  CHECK(robust->weights.tail(n_bad).maxCoeff() <
        robust->weights.head(n_clean).mean());
  CHECK(robust->probabilities.size() == cat_cont.size());
  CHECK(robust->joint_densities.size() == cat_cont.size());
  CHECK(robust->weights.size() == cat_cont.size());
  CHECK(robust->weights.allFinite());
}

TEST_CASE("Polyserial pair ML kernel rejects malformed inputs") {
  Eigen::VectorXi cat(3);
  cat << 0, 1, 3;
  Eigen::VectorXd u(3);
  u << -0.5, 0.0, 0.5;
  Eigen::VectorXd th(2);
  th << -0.4, 0.6;

  auto bad_cat = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
  REQUIRE_FALSE(bad_cat.has_value());
  CHECK(bad_cat.error().detail.find("category outside") != std::string::npos);

  cat << 0, 1, 2;
  th << 0.6, -0.4;
  auto bad_th = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
  REQUIRE_FALSE(bad_th.has_value());
  CHECK(bad_th.error().detail.find("strictly increasing") != std::string::npos);
}

TEST_CASE("Polyserial fixed-marginal DPD preserves ML limit and downweights discordance") {
  std::mt19937 rng(20260517);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::VectorXd th(2);
  th << -0.45, 0.65;

  constexpr Eigen::Index n_clean = 420;
  constexpr Eigen::Index n_bad = 35;
  Eigen::VectorXi cat_clean(n_clean);
  Eigen::VectorXd u_clean(n_clean);
  const double rho_true = 0.62;
  const double sd = std::sqrt(1.0 - rho_true * rho_true);
  for (Eigen::Index r = 0; r < n_clean; ++r) {
    const double u = norm(rng);
    const double z = rho_true * u + sd * norm(rng);
    u_clean(r) = u;
    cat_clean(r) = (z > th(0)) + (z > th(1));
  }

  Eigen::VectorXi cat_cont(n_clean + n_bad);
  Eigen::VectorXd u_cont(n_clean + n_bad);
  cat_cont.head(n_clean) = cat_clean;
  u_cont.head(n_clean) = u_clean;
  for (Eigen::Index r = 0; r < n_bad; ++r) {
    cat_cont(n_clean + r) = 0;
    u_cont(n_clean + r) = 5.5 + 0.01 * static_cast<double>(r);
  }

  auto clean_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_clean, u_clean, th);
  auto cont_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_cont, u_cont, th);
  auto dpd0 = magmaan::data::fit_polyserial_pair_rho_dpd(
      cat_cont, u_cont, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.0});
  auto robust = magmaan::data::fit_polyserial_pair_rho_dpd(
      cat_cont, u_cont, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.5});

  REQUIRE(clean_ml.has_value());
  REQUIRE(cont_ml.has_value());
  REQUIRE(dpd0.has_value());
  REQUIRE(robust.has_value());
  CHECK(dpd0->rho == doctest::Approx(cont_ml->rho));
  CHECK(cont_ml->rho < clean_ml->rho);
  CHECK(robust->rho > cont_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(cont_ml->rho - clean_ml->rho));
  CHECK(robust->probabilities.size() == cat_cont.size());
  CHECK(robust->weights.tail(n_bad).maxCoeff() <
        robust->weights.head(n_clean).mean());

  auto ml_scores = magmaan::data::polyserial_pair_scores(
      cat_cont, u_cont, cont_ml->rho, th);
  auto dpd0_scores = magmaan::data::polyserial_pair_dpd_scores(
      cat_cont, u_cont, cont_ml->rho, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.0});
  auto robust_scores = magmaan::data::polyserial_pair_dpd_scores(
      cat_cont, u_cont, robust->rho, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.5});
  REQUIRE(ml_scores.has_value());
  REQUIRE(dpd0_scores.has_value());
  REQUIRE(robust_scores.has_value());
  CHECK(dpd0_scores->score_contributions.isApprox(
      ml_scores->score_contributions, 0.0));
  CHECK(dpd0_scores->bread.isApprox(ml_scores->score_gamma, 0.0));
  CHECK(robust_scores->score_contributions.rows() == cat_cont.size());
  CHECK(robust_scores->score_contributions.cols() == th.size() + 1);
  CHECK(robust_scores->score_gamma.isApprox(
      (robust_scores->score_contributions.transpose() *
       robust_scores->score_contributions) /
          static_cast<double>(cat_cont.size()),
      1e-12));
  CHECK(robust_scores->bread.rows() == th.size() + 1);
  CHECK(robust_scores->bread.cols() == th.size() + 1);
  CHECK(robust_scores->bread(th.size(), th.size()) > 0.0);
}

TEST_CASE("Continuous pair normal ML kernel returns complete-data pair diagnostics") {
  Eigen::VectorXd x(4);
  Eigen::VectorXd y(4);
  x << 1.0, 2.0, 3.0, 4.0;
  y << 2.0, 3.0, 5.0, 7.0;

  auto fit = magmaan::data::fit_continuous_pair_normal_ml(x, y);
  REQUIRE(fit.has_value());
  CHECK(fit->n_obs == 4);
  CHECK(fit->mean_i == doctest::Approx(2.5));
  CHECK(fit->mean_j == doctest::Approx(4.25));
  CHECK(fit->var_i == doctest::Approx(1.25));
  CHECK(fit->var_j == doctest::Approx(3.6875));
  CHECK(fit->cov == doctest::Approx(2.125));
  CHECK(fit->rho == doctest::Approx(2.125 / std::sqrt(1.25 * 3.6875)));
  CHECK(std::isfinite(fit->negloglik));
  CHECK(fit->score_contributions.rows() == x.size());
  CHECK(fit->score_contributions.cols() == 5);
  CHECK(fit->score_contributions.allFinite());
  CHECK(fit->score_contributions.colwise().sum().norm() < 1e-10);
  CHECK(fit->score_gamma.rows() == 5);
  CHECK(fit->score_gamma.cols() == 5);
  CHECK(fit->score_gamma.isApprox(
      (fit->score_contributions.transpose() * fit->score_contributions) /
          static_cast<double>(x.size()),
      1e-12));

  auto nll = magmaan::data::continuous_pair_normal_negloglik(
      x, y, fit->mean_i, fit->mean_j, fit->var_i, fit->var_j, fit->cov);
  REQUIRE(nll.has_value());
  CHECK(*nll == doctest::Approx(fit->negloglik));
  auto scores = magmaan::data::continuous_pair_normal_scores(
      x, y, fit->mean_i, fit->mean_j, fit->var_i, fit->var_j, fit->cov);
  REQUIRE(scores.has_value());
  CHECK(scores->score_contributions.isApprox(fit->score_contributions, 0.0));
  CHECK(scores->score_gamma.isApprox(fit->score_gamma, 0.0));

  y << 2.0, 4.0, 6.0, 8.0;
  auto singular = magmaan::data::fit_continuous_pair_normal_ml(x, y);
  REQUIRE_FALSE(singular.has_value());
  CHECK(singular.error().detail.find("positive definite") != std::string::npos);
}

TEST_CASE("Mixed pair labels match MixedOrdinalStats moment order") {
  std::vector<std::int32_t> ordered{0, 1, 0};
  std::vector<std::int32_t> threshold_ov{1, 1};
  std::vector<std::int32_t> threshold_level{1, 2};

  auto moments = magmaan::data::mixed_moment_labels(
      ordered, threshold_ov, threshold_level);
  REQUIRE(moments.has_value());
  REQUIRE(moments->size() == 9);

  CHECK((*moments)[0].kind == magmaan::data::MixedMomentKind::threshold);
  CHECK((*moments)[0].variable == 1);
  CHECK((*moments)[0].threshold_level == 1);
  CHECK((*moments)[1].kind == magmaan::data::MixedMomentKind::threshold);
  CHECK((*moments)[1].variable == 1);
  CHECK((*moments)[1].threshold_level == 2);
  CHECK((*moments)[2].kind == magmaan::data::MixedMomentKind::continuous_mean);
  CHECK((*moments)[2].variable == 0);
  CHECK((*moments)[3].kind == magmaan::data::MixedMomentKind::continuous_mean);
  CHECK((*moments)[3].variable == 2);
  CHECK((*moments)[4].kind == magmaan::data::MixedMomentKind::continuous_variance);
  CHECK((*moments)[4].variable == 0);
  CHECK((*moments)[5].kind == magmaan::data::MixedMomentKind::continuous_variance);
  CHECK((*moments)[5].variable == 2);

  auto pairs = magmaan::data::mixed_pair_labels(
      ordered, static_cast<std::int32_t>(threshold_ov.size()));
  REQUIRE(pairs.has_value());
  REQUIRE(pairs->size() == 3);
  CHECK((*pairs)[0].i == 1);
  CHECK((*pairs)[0].j == 0);
  CHECK((*pairs)[0].moment_index == 6);
  CHECK((*pairs)[0].kind == magmaan::data::MixedPairKind::continuous_ordinal);
  CHECK((*pairs)[1].i == 2);
  CHECK((*pairs)[1].j == 0);
  CHECK((*pairs)[1].moment_index == 7);
  CHECK((*pairs)[1].kind == magmaan::data::MixedPairKind::continuous_continuous);
  CHECK((*pairs)[2].i == 2);
  CHECK((*pairs)[2].j == 1);
  CHECK((*pairs)[2].moment_index == 8);
  CHECK((*pairs)[2].kind == magmaan::data::MixedPairKind::continuous_ordinal);

  CHECK((*moments)[6].kind == magmaan::data::MixedMomentKind::pair);
  CHECK((*moments)[6].variable_i == 1);
  CHECK((*moments)[6].variable_j == 0);
  CHECK((*moments)[6].pair_kind == magmaan::data::MixedPairKind::continuous_ordinal);
  CHECK((*moments)[7].kind == magmaan::data::MixedMomentKind::pair);
  CHECK((*moments)[7].variable_i == 2);
  CHECK((*moments)[7].variable_j == 0);
  CHECK((*moments)[7].pair_kind == magmaan::data::MixedPairKind::continuous_continuous);
  CHECK((*moments)[8].kind == magmaan::data::MixedMomentKind::pair);
  CHECK((*moments)[8].variable_i == 2);
  CHECK((*moments)[8].variable_j == 1);
  CHECK((*moments)[8].pair_kind == magmaan::data::MixedPairKind::continuous_ordinal);

  threshold_ov = {0};
  threshold_level = {1};
  auto bad = magmaan::data::mixed_moment_labels(
      ordered, threshold_ov, threshold_level);
  REQUIRE_FALSE(bad.has_value());
  CHECK(bad.error().detail.find("invalid ordered variable") != std::string::npos);
}

TEST_CASE("Ordinal pair observed table skips NaN by observed-pair semantics") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd xi(7);
  Eigen::VectorXd xj(7);
  xi << 1.0, 2.0, nan, 3.0, 1.0, 2.0, 3.0;
  xj << 1.0, 2.0, 1.0, nan, 2.0, 1.0, 2.0;

  auto table = magmaan::data::ordinal_pair_observed_table(xi, xj, 3, 2);
  REQUIRE(table.has_value());
  CHECK(table->n_obs == 5);
  CHECK(table->n_missing == 2);
  CHECK(table->counts.rows() == 3);
  CHECK(table->counts.cols() == 2);
  CHECK(table->counts(0, 0) == doctest::Approx(1.0));
  CHECK(table->counts(0, 1) == doctest::Approx(1.0));
  CHECK(table->counts(1, 0) == doctest::Approx(1.0));
  CHECK(table->counts(1, 1) == doctest::Approx(1.0));
  CHECK(table->counts(2, 0) == doctest::Approx(0.0));
  CHECK(table->counts(2, 1) == doctest::Approx(1.0));
}

TEST_CASE("Ordinal pair observed table rejects malformed observed categories") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd xi(3);
  Eigen::VectorXd xj(3);
  xi << 1.0, 2.5, nan;
  xj << 1.0, 2.0, 1.0;

  auto malformed = magmaan::data::ordinal_pair_observed_table(xi, xj, 2, 2);
  REQUIRE_FALSE(malformed.has_value());
  CHECK(malformed.error().detail.find("finite integers") != std::string::npos);

  xi << nan, nan, nan;
  xj << 1.0, 2.0, nan;
  auto empty = magmaan::data::ordinal_pair_observed_table(xi, xj, 2, 2);
  REQUIRE_FALSE(empty.has_value());
  CHECK(empty.error().detail.find("no observed pairs") != std::string::npos);
}

TEST_CASE("Ordinal pair observed ML wrappers reuse observed-pair table counts") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd xi(8);
  Eigen::VectorXd xj(8);
  xi << 1.0, 2.0, 3.0, nan, 1.0, 2.0, 3.0, 2.0;
  xj << 1.0, 2.0, 2.0, 1.0, nan, 1.0, 1.0, 2.0;
  Eigen::VectorXd thi(2);
  thi << -0.4, 0.7;
  Eigen::VectorXd thj(1);
  thj << 0.2;

  auto table = magmaan::data::ordinal_pair_observed_table(xi, xj, 3, 2);
  REQUIRE(table.has_value());
  auto direct_rho = magmaan::data::fit_ordinal_pair_rho_ml(
      table->counts, thi, thj);
  auto observed_rho = magmaan::data::fit_ordinal_pair_observed_rho_ml(
      xi, xj, 3, 2, thi, thj);
  REQUIRE(direct_rho.has_value());
  REQUIRE(observed_rho.has_value());
  CHECK(observed_rho->n_obs == table->n_obs);
  CHECK(observed_rho->n_missing == table->n_missing);
  CHECK(observed_rho->counts.isApprox(table->counts, 0.0));
  CHECK(observed_rho->fit.rho == doctest::Approx(direct_rho->rho));
  CHECK(observed_rho->fit.negloglik == doctest::Approx(direct_rho->negloglik));

  auto direct_joint = magmaan::data::fit_ordinal_pair_joint_ml(table->counts);
  auto observed_joint = magmaan::data::fit_ordinal_pair_observed_joint_ml(
      xi, xj, 3, 2);
  REQUIRE(direct_joint.has_value());
  REQUIRE(observed_joint.has_value());
  CHECK(observed_joint->n_obs == table->n_obs);
  CHECK(observed_joint->n_missing == table->n_missing);
  CHECK(observed_joint->counts.isApprox(table->counts, 0.0));
  CHECK(observed_joint->fit.thresholds_i.isApprox(direct_joint->thresholds_i, 1e-12));
  CHECK(observed_joint->fit.thresholds_j.isApprox(direct_joint->thresholds_j, 1e-12));
  CHECK(observed_joint->fit.rho == doctest::Approx(direct_joint->rho));
  CHECK(observed_joint->fit.negloglik == doctest::Approx(direct_joint->negloglik));
}

TEST_CASE("Observed ordinal stats degenerate to complete-data ordinal stats") {
  Eigen::MatrixXd X(20, 2);
  Eigen::Index r = 0;
  for (int k = 0; k < 4; ++k) X.row(r++) << 1, 1;
  for (int k = 0; k < 5; ++k) X.row(r++) << 1, 2;
  for (int k = 0; k < 5; ++k) X.row(r++) << 2, 1;
  for (int k = 0; k < 6; ++k) X.row(r++) << 2, 2;

  auto complete = magmaan::data::ordinal_stats_from_integer_data({X});
  auto observed = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap);
  REQUIRE(complete.has_value());
  REQUIRE(observed.has_value());
  REQUIRE(observed->R.size() == 1);
  CHECK(observed->R[0].isApprox(complete->R[0], 1e-12));
  CHECK(observed->thresholds[0].isApprox(complete->thresholds[0], 1e-12));
  CHECK(observed->NACOV[0].isApprox(complete->NACOV[0], 1e-10));
  REQUIRE(observed->moment_influence.size() == 1);
  CHECK(observed->moment_influence[0].isApprox(
      complete->moment_influence[0], 1e-10));
  REQUIRE(observed->int_data.size() == 1);
  CHECK((observed->int_data[0] - complete->int_data[0]).cwiseAbs().maxCoeff() == 0);
  CHECK(observed->W_dwls[0].isApprox(complete->W_dwls[0], 1e-10));
  CHECK(observed->pairwise_gamma == "overlap");
}
