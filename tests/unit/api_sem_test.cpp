#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/api/sem.hpp"

namespace {

magmaan::data::RawData continuous_raw() {
  Eigen::MatrixXd X(80, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double t = static_cast<double>(i);
    const double z = std::sin(0.19 * t) + 0.4 * std::cos(0.07 * t);
    X(i, 0) = z + 0.15 * std::sin(0.31 * t);
    X(i, 1) = 0.8 * z + 0.20 * std::cos(0.23 * t);
    X(i, 2) = 0.7 * z + 0.18 * std::sin(0.41 * t + 0.5);
    X(i, 3) = 0.9 * z + 0.16 * std::cos(0.37 * t + 0.2);
  }
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  return raw;
}

Eigen::MatrixXd ordinal_block(Eigen::Index n = 140) {
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double t = static_cast<double>(i);
    const double z = std::sin(0.17 * t) + 0.5 * std::cos(0.11 * t);
    const double vals[4] = {
        z + 0.25 * std::sin(0.31 * t),
        0.85 * z + 0.25 * std::cos(0.29 * t),
        0.75 * z + 0.20 * std::sin(0.43 * t + 0.6),
        0.95 * z + 0.22 * std::cos(0.37 * t + 0.3),
    };
    for (Eigen::Index j = 0; j < 4; ++j) {
      int level = 1;
      if (vals[j] > -0.65) {
        level = 2;
      }
      if (vals[j] > -0.10) {
        level = 3;
      }
      if (vals[j] > 0.45) {
        level = 4;
      }
      if (vals[j] > 0.95) {
        level = 5;
      }
      X(i, j) = static_cast<double>(level);
    }
  }
  return X;
}

std::string ordinal_syntax() {
  return "f =~ x1 + x2 + x3 + x4\n"
         "x1 | t1 + t2 + t3 + t4\n"
         "x2 | t1 + t2 + t3 + t4\n"
         "x3 | t1 + t2 + t3 + t4\n"
         "x4 | t1 + t2 + t3 + t4\n"
         "x1 ~*~ 1*x1\n"
         "x2 ~*~ 1*x2\n"
         "x3 ~*~ 1*x3\n"
         "x4 ~*~ 1*x4\n";
}

} // namespace

#define REQUIRE_OK(value)                                                     \
  do {                                                                        \
    INFO("api error: " << ((value).has_value() ? "" : (value).error().detail)); \
    REQUIRE((value).has_value());                                             \
  } while (false)

TEST_CASE("api sem header compiles standalone and parse errors keep stage") {
  auto bad = magmaan::api::model_from_lavaan("f =~");
  REQUIRE(!bad.has_value());
  CHECK(bad.error().stage == magmaan::api::ErrorStage::Parse);
}

TEST_CASE("api complete-data ML exposes staged post-fit calls") {
  const auto model = magmaan::api::model_from_lavaan(
      "f =~ x1 + a*x2 + x3 + x4\na_sq := a^2");
  REQUIRE_OK(model);

  const auto raw = continuous_raw();
  const auto stats = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(stats.has_value());
  const auto data = magmaan::api::data_from_sample_stats(*model, *stats);
  REQUIRE(data.has_value());

  const auto fit = magmaan::api::fit(*model, *data, magmaan::api::ml());
  REQUIRE_OK(fit);

  const auto se =
      magmaan::api::standard_errors(*fit, magmaan::api::expected_information());
  REQUIRE_OK(se);
  CHECK(se->se.size() == fit->estimates().theta.size());

  const auto z = magmaan::api::z_test(*fit, se->se);
  CHECK(z.z.size() == fit->estimates().theta.size());

  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, fit->estimates().theta.size());
  R(0, 0) = 1.0;
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  const auto wald = magmaan::api::wald_test(*fit, R, se->vcov, q);
  REQUIRE(wald.has_value());
  CHECK(wald->df == 1);

  const auto tst = magmaan::api::test(*fit, magmaan::api::standard_chi_square());
  REQUIRE(tst.has_value());
  CHECK(tst->df > 0);

  const auto fm = magmaan::api::fit_measures(*fit);
  REQUIRE(fm.has_value());
  CHECK(fm->complete_data_extras.has_value());

  const auto defs = magmaan::api::compute_defined(*fit, se->vcov);
  REQUIRE(defs.has_value());
  REQUIRE(defs->entries.size() == 1);
  CHECK(defs->entries[0].name == "a_sq");

  const auto std_lv = magmaan::api::standardize_lv(*fit, se->vcov);
  REQUIRE(std_lv.has_value());
  CHECK(std_lv->theta.size() == fit->estimates().theta.size());

  const auto rob = magmaan::api::robust_se(*fit, raw);
  REQUIRE_OK(rob);
  CHECK(rob->se.size() == fit->estimates().theta.size());

  const auto mi = magmaan::api::modification_indices(*fit);
  REQUIRE_OK(mi);
  const auto scores = magmaan::api::score_tests(*fit);
  REQUIRE_OK(scores);
}

TEST_CASE("api continuous LS fits dispatch estimator-aware chi-square") {
  const auto model = magmaan::api::model_from_lavaan("f =~ x1 + x2 + x3 + x4");
  REQUIRE(model.has_value());
  const auto raw = continuous_raw();
  const auto stats = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(stats.has_value());
  const auto data = magmaan::api::data_from_sample_stats(*model, *stats);
  REQUIRE(data.has_value());

  auto uls_fit = magmaan::api::fit(
      *model, *data, magmaan::api::uls().auto_variance_bounds());
  REQUIRE_OK(uls_fit);
  auto uls_t = magmaan::api::test(*uls_fit, magmaan::api::standard_chi_square());
  REQUIRE(uls_t.has_value());
  CHECK(std::isfinite(uls_t->statistic));

  auto gls_fit = magmaan::api::fit(
      *model, *data, magmaan::api::gls().auto_variance_bounds());
  REQUIRE_OK(gls_fit);
  auto gls_t = magmaan::api::test(*gls_fit, magmaan::api::standard_chi_square());
  REQUIRE(gls_t.has_value());
  CHECK(std::isfinite(gls_t->statistic));

  magmaan::estimate::gmm::Weight weight;
  weight.push_back(Eigen::MatrixXd::Identity(10, 10));
  auto wls_fit = magmaan::api::fit(
      *model, *data, magmaan::api::wls(std::move(weight)).auto_variance_bounds());
  REQUIRE_OK(wls_fit);
  auto wls_t = magmaan::api::test(*wls_fit, magmaan::api::standard_chi_square());
  REQUIRE(wls_t.has_value());
  CHECK(std::isfinite(wls_t->statistic));

  const std::vector<Eigen::MatrixXd> gamma{
      magmaan::data::empirical_gamma(raw.X[0]).value()};
  const auto ls_rob = magmaan::api::robust_continuous_ls(*uls_fit, gamma);
  REQUIRE(ls_rob.has_value());
  CHECK(ls_rob->df > 0);

  const auto mi = magmaan::api::modification_indices(*uls_fit);
  REQUIRE_OK(mi);
  const auto scores = magmaan::api::score_tests(*uls_fit);
  REQUIRE_OK(scores);
}

TEST_CASE("api FIML exposes likelihood test, fit measures, and MLR reporting") {
  magmaan::api::ModelOptions options;
  options.build.meanstructure = true;
  const auto model =
      magmaan::api::model_from_lavaan("f =~ x1 + x2 + x3 + x4", options);
  REQUIRE(model.has_value());
  const auto data = magmaan::api::data_from_raw(*model, continuous_raw());
  REQUIRE(data.has_value());

  const auto fit = magmaan::api::fit(*model, *data, magmaan::api::fiml());
  REQUIRE_OK(fit);

  const auto tst = magmaan::api::test(*fit, magmaan::api::standard_chi_square());
  REQUIRE(tst.has_value());
  CHECK(tst->df > 0);

  const auto fm = magmaan::api::fit_measures(*fit);
  REQUIRE(fm.has_value());
  CHECK(fm->fiml_extras.has_value());

  const auto mlr = magmaan::api::fiml_robust_mlr(*fit);
  REQUIRE_OK(mlr);
  CHECK(mlr->df == tst->df);

  const auto mi = magmaan::api::modification_indices(*fit);
  REQUIRE_OK(mi);
  const auto scores = magmaan::api::score_tests(*fit);
  REQUIRE_OK(scores);
}

TEST_CASE("api ordinal DWLS/WLS fits and robust ordinal reporting") {
  const auto model = magmaan::api::model_from_lavaan(ordinal_syntax());
  REQUIRE_OK(model);
  const auto stats =
      magmaan::data::ordinal_stats_from_integer_data({ordinal_block()});
  REQUIRE_OK(stats);
  const auto data = magmaan::api::data_from_ordinal(*model, *stats);
  REQUIRE(data.has_value());

  const auto dwls_fit =
      magmaan::api::fit(*model, *data, magmaan::api::ordinal_dwls());
  REQUIRE_OK(dwls_fit);
  const auto dwls_rob = magmaan::api::robust_ordinal(*dwls_fit);
  REQUIRE_OK(dwls_rob);
  CHECK(dwls_rob->df > 0);

  const auto wls_fit =
      magmaan::api::fit(*model, *data, magmaan::api::ordinal_wls());
  REQUIRE_OK(wls_fit);
  const auto wls_t =
      magmaan::api::test(*wls_fit, magmaan::api::standard_chi_square());
  REQUIRE(wls_t.has_value());
  CHECK(wls_t->df == dwls_rob->df);

  const auto mi = magmaan::api::modification_indices(*dwls_fit);
  REQUIRE_OK(mi);
  const auto scores = magmaan::api::score_tests(*dwls_fit);
  REQUIRE_OK(scores);
}

TEST_CASE("api Analysis preserves first error and post-fit calls require fit") {
  const auto model = magmaan::api::model_from_lavaan("f =~ x1 + x2 + x3 + x4");
  REQUIRE(model.has_value());
  const auto raw = continuous_raw();
  const auto stats = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(stats.has_value());
  const auto data = magmaan::api::data_from_sample_stats(*model, *stats);
  REQUIRE(data.has_value());

  auto no_fit = magmaan::api::analyze(*model, *data)
                    .standard_errors(magmaan::api::expected_information())
                    .fit(magmaan::api::ml())
                    .summary();
  REQUIRE(!no_fit.has_value());
  CHECK(no_fit.error().detail == "standard_errors() requires fit() first");

  auto ok = magmaan::api::analyze(*model, *data)
                .fit(magmaan::api::ml())
                .test(magmaan::api::standard_chi_square())
                .summary();
  REQUIRE_OK(ok);
  CHECK(ok->test.has_value());
}

#undef REQUIRE_OK
