#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/api/sem.hpp"
#include "../oracle.hpp"

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
  return "f =~ x1 + l2*x2 + l3*x3 + x4\n"
         "x1 | t1 + t2 + t3 + t4\n"
         "x2 | t1 + t2 + t3 + t4\n"
         "x3 | t1 + t2 + t3 + t4\n"
         "x4 | t1 + t2 + t3 + t4\n"
         "x1 ~*~ 1*x1\n"
         "x2 ~*~ 1*x2\n"
         "x3 ~*~ 1*x3\n"
         "x4 ~*~ 1*x4\n"
         "lprod := l2*l3\n";
}

using magmaan::test::load_json_fixture;

using magmaan::test::matrix_from_json;

std::vector<std::string> fixture_observed_order(const nlohmann::json& j) {
  if (j["sample_cov"][0].contains("names")) {
    return j["sample_cov"][0]["names"].get<std::vector<std::string>>();
  }
  return {"x1", "x2", "x3", "x4", "x5"};
}

std::vector<std::string>
model_observed_order(const magmaan::api::Model& model) {
  std::vector<std::string> out;
  out.reserve(model.structure().ov_order.size());
  for (const auto id : model.structure().ov_order) {
    REQUIRE(id >= 0);
    out.push_back(model.names().var_name[static_cast<std::size_t>(id)]);
  }
  return out;
}

Eigen::MatrixXd reorder_matrix(const Eigen::MatrixXd& raw,
                               const std::vector<std::string>& source,
                               const std::vector<std::string>& target) {
  REQUIRE(raw.rows() == static_cast<Eigen::Index>(source.size()));
  REQUIRE(raw.cols() == static_cast<Eigen::Index>(source.size()));
  Eigen::MatrixXd out(target.size(), target.size());
  for (std::size_t r = 0; r < target.size(); ++r) {
    const auto sr = std::find(source.begin(), source.end(), target[r]);
    REQUIRE(sr != source.end());
    for (std::size_t c = 0; c < target.size(); ++c) {
      const auto sc = std::find(source.begin(), source.end(), target[c]);
      REQUIRE(sc != source.end());
      out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
          raw(static_cast<Eigen::Index>(std::distance(source.begin(), sr)),
              static_cast<Eigen::Index>(std::distance(source.begin(), sc)));
    }
  }
  return out;
}

magmaan::data::SampleStats fcsem_fixture_stats(
    const magmaan::api::Model& model, const nlohmann::json& j) {
  magmaan::data::SampleStats stats;
  stats.S = {reorder_matrix(matrix_from_json(j["sample_cov"][0]["matrix"]),
                            fixture_observed_order(j),
                            model_observed_order(model))};
  stats.n_obs = {j["n_obs"].get<std::int64_t>()};
  return stats;
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

  // FABIN starts: NLopt L-BFGS stalls from simple_starts() on this synthetic
  // 80-sample ML objective. The R bridge uses FABIN as its default; this
  // post-fit-coverage test follows the same recipe.
  const auto fit = magmaan::api::fit(
      *model, *data,
      magmaan::api::ml().starts(magmaan::api::fabin_starts()));
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

  // fit_measures is exposed for continuous least-squares fits — CFI/TLI/
  // RMSEA/SRMR are estimator-agnostic functions of the LS χ² and Σ̂(θ̂).
  const auto uls_fm = magmaan::api::fit_measures(*uls_fit);
  REQUIRE_OK(uls_fm);
  CHECK(std::isfinite(uls_fm->indices.cfi));
  CHECK(std::isfinite(uls_fm->indices.rmsea));
  REQUIRE(uls_fm->complete_data_extras.has_value());
  CHECK(std::isfinite(uls_fm->complete_data_extras->srmr));
  CHECK(uls_fm->complete_data_extras->srmr >= 0.0);
  const auto gls_fm = magmaan::api::fit_measures(*gls_fit);
  REQUIRE_OK(gls_fm);
  CHECK(std::isfinite(gls_fm->indices.cfi));
  const auto wls_fm = magmaan::api::fit_measures(*wls_fit);
  REQUIRE_OK(wls_fm);
  CHECK(std::isfinite(wls_fm->indices.rmsea));

  // Non-robust (information-inverse) standard errors for least-squares fits.
  const auto uls_se = magmaan::api::standard_errors(
      *uls_fit, magmaan::api::expected_information());
  REQUIRE_OK(uls_se);
  CHECK(uls_se->se.size() == uls_fit->estimates().theta.size());
  const auto wls_se = magmaan::api::standard_errors(
      *wls_fit, magmaan::api::expected_information());
  REQUIRE_OK(wls_se);
  // The WLS weight here is the identity, so the WLS information matches the
  // ULS information ⇒ identical standard errors.
  REQUIRE(wls_se->se.size() == uls_se->se.size());
  for (Eigen::Index k = 0; k < uls_se->se.size(); ++k) {
    CHECK(std::isfinite(uls_se->se(k)));
    CHECK(wls_se->se(k) == doctest::Approx(uls_se->se(k)).epsilon(1e-5));
  }
  const auto gls_se = magmaan::api::standard_errors(
      *gls_fit, magmaan::api::expected_information());
  REQUIRE_OK(gls_se);
  CHECK(std::isfinite(gls_se->se(0)));

  const auto mi = magmaan::api::modification_indices(*uls_fit);
  REQUIRE_OK(mi);
  const auto scores = magmaan::api::score_tests(*uls_fit);
  REQUIRE_OK(scores);
}

TEST_CASE("api exposes Satorra-Bentler 2001/2010 nested tests") {
  const auto h1_model =
      magmaan::api::model_from_lavaan("f =~ x1 + x2 + x3 + x4");
  REQUIRE_OK(h1_model);
  const auto h0_model =
      magmaan::api::model_from_lavaan("f =~ x1 + a*x2 + a*x3 + x4");
  REQUIRE_OK(h0_model);

  const auto raw = continuous_raw();
  const auto stats = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(stats.has_value());
  const auto h1_data = magmaan::api::data_from_sample_stats(*h1_model, *stats);
  REQUIRE_OK(h1_data);
  const auto h0_data = magmaan::api::data_from_sample_stats(*h0_model, *stats);
  REQUIRE_OK(h0_data);

  // FABIN starts: NLopt L-BFGS stalls from simple_starts() on this synthetic
  // 80-sample ML objective. Matches the R bridge's default start path.
  const auto ml_fabin = magmaan::api::ml().starts(magmaan::api::fabin_starts());
  const auto h1_fit = magmaan::api::fit(*h1_model, *h1_data, ml_fabin);
  REQUIRE_OK(h1_fit);
  const auto h0_fit = magmaan::api::fit(*h0_model, *h0_data, ml_fabin);
  REQUIRE_OK(h0_fit);

  const auto standard = magmaan::api::lr_test(*h1_fit, *h0_fit);
  REQUIRE_OK(standard);
  CHECK(standard->df_diff == 1);

  const auto sb2000 = magmaan::api::lr_test_satorra2000(*h1_fit, *h0_fit, raw);
  REQUIRE_OK(sb2000);
  CHECK(sb2000->df_diff == standard->df_diff);
  CHECK(std::isfinite(sb2000->T_scaled));

  const auto sb2001 =
      magmaan::api::lr_test_satorra_bentler2001(*h1_fit, *h0_fit, raw);
  REQUIRE_OK(sb2001);
  CHECK(sb2001->df_diff == standard->df_diff);
  CHECK(std::isfinite(sb2001->T_scaled));

  const auto sb2010 =
      magmaan::api::lr_test_satorra_bentler2010(*h1_fit, *h0_fit, raw);
  REQUIRE_OK(sb2010);
  CHECK(sb2010->df_diff == standard->df_diff);
  CHECK(std::isfinite(sb2010->T_scaled));
  CHECK(std::isfinite(sb2010->c_hybrid));
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
  REQUIRE(fm->fiml_extras.has_value());
  // FIML now also reports SRMR (vs the saturated EM moments).
  CHECK(std::isfinite(fm->fiml_extras->srmr));
  CHECK(fm->fiml_extras->srmr >= 0.0);

  // Non-robust FIML standard errors — the inverse observed information.
  const auto fiml_se = magmaan::api::standard_errors(
      *fit, magmaan::api::expected_information());
  REQUIRE_OK(fiml_se);
  CHECK(fiml_se->se.size() == fit->estimates().theta.size());
  for (Eigen::Index k = 0; k < fiml_se->se.size(); ++k) {
    CHECK(fiml_se->se(k) > 0.0);
  }

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

  const auto ord_fm = magmaan::api::fit_measures(*dwls_fit);
  REQUIRE_OK(ord_fm);
  CHECK(std::isfinite(ord_fm->indices.cfi));
  CHECK(std::isfinite(ord_fm->indices.rmsea));
  REQUIRE(ord_fm->ordinal_srmr.has_value());
  CHECK(std::isfinite(*ord_fm->ordinal_srmr));
  CHECK(*ord_fm->ordinal_srmr >= 0.0);

  // Standardization and `:=` defined parameters are parameterization-agnostic
  // transforms of the fit, so they are exposed for ordinal fits (no guard).
  // The robust ordinal result carries the full free-parameter vcov.
  const Eigen::MatrixXd &vcov = dwls_rob->vcov;

  const auto std_all = magmaan::api::standardize_all(*dwls_fit, vcov);
  REQUIRE_OK(std_all);
  CHECK(std_all->theta.size() == dwls_fit->estimates().theta.size());

  const auto std_lv = magmaan::api::standardize_lv(*dwls_fit, vcov);
  REQUIRE_OK(std_lv);
  CHECK(std_lv->theta.size() == dwls_fit->estimates().theta.size());

  const auto defs = magmaan::api::compute_defined(*dwls_fit, vcov);
  REQUIRE_OK(defs);
  REQUIRE(defs->entries.size() == 1);
  CHECK(defs->entries[0].name == "lprod");
  CHECK(std::isfinite(defs->entries[0].value));
  CHECK(defs->entries[0].se > 0.0);

  // Factor scores stay guarded for ordinal fits: lavaan scores ordinal
  // indicators by latent-response integration, a distinct estimator the
  // continuous predictor here does not implement (see speculative.md).
  magmaan::data::RawData ordinal_raw;
  ordinal_raw.X.push_back(ordinal_block());
  const auto fs = magmaan::api::factor_scores(
      *dwls_fit, ordinal_raw, magmaan::measures::FactorScoreMethod::Regression);
  REQUIRE_FALSE(fs.has_value());
}

TEST_CASE("api frontier exposes native FC-SEM fit and post-fit calls") {
  const auto j =
      load_json_fixture("composite/0002_composite_factor_hs.fit.json");
  REQUIRE_FALSE(j.is_discarded());
  auto model = magmaan::api::frontier::model_from_lavaan_fcsem(
      j["input"].get<std::string>());
  REQUIRE_OK(model);
  CHECK(model->structure().composite_mode ==
        magmaan::spec::CompositeMode::FcSem);

  const auto stats = fcsem_fixture_stats(*model, j);
  const auto data = magmaan::api::data_from_sample_stats(*model, stats);
  REQUIRE_OK(data);

  const auto core_fit = magmaan::api::fit(*model, *data, magmaan::api::ml());
  REQUIRE_FALSE(core_fit.has_value());
  CHECK(core_fit.error().stage ==
        magmaan::api::ErrorStage::UnsupportedCombination);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 4000;
  magmaan::api::OptimizerSpec optimizer;
  optimizer.options = opts;
  const auto fit = magmaan::api::frontier::fit_ml_fcsem(
      *model, *data, optimizer);
  REQUIRE_OK(fit);

  const auto tst =
      magmaan::api::test(*fit, magmaan::api::standard_chi_square());
  REQUIRE_OK(tst);
  CHECK(tst->statistic == doctest::Approx(j["chi2"].get<double>())
                              .epsilon(2e-5));
  CHECK(tst->df == j["df"].get<int>());

  const auto se = magmaan::api::frontier::standard_errors_fcsem(*fit);
  REQUIRE_OK(se);
  CHECK(se->se.size() == fit->estimates().theta.size());

  const auto fm = magmaan::api::frontier::fit_measures_fcsem(*fit);
  REQUIRE_OK(fm);
  REQUIRE(fm->complete_data_extras.has_value());
  CHECK(fm->indices.cfi == doctest::Approx(j["cfi"].get<double>())
                               .epsilon(2e-5));
  CHECK(fm->complete_data_extras->logl ==
        doctest::Approx(j["logl"].get<double>()).epsilon(2e-5));

  const auto rows =
      magmaan::api::frontier::standardized_rows_fcsem(*fit, se->vcov);
  REQUIRE_OK(rows);
  CHECK(rows->size() == j["weights"].size() + j["rows"].size());
}

TEST_CASE("api composite doors are explicit") {
  const std::string composite_syntax = "C <~ x1 + x2 + x3\n y ~ C";
  auto closed = magmaan::api::model_from_lavaan(composite_syntax);
  REQUIRE_FALSE(closed.has_value());
  CHECK(closed.error().stage == magmaan::api::ErrorStage::Model);
  const auto* err =
      std::get_if<magmaan::PartableError>(&closed.error().underlying);
  REQUIRE(err != nullptr);
  CHECK(err->kind == magmaan::PartableError::Kind::CompositeModeRequired);

  magmaan::api::ModelOptions historical;
  historical.build.composite_mode =
      magmaan::spec::CompositeMode::HenselerOgasawara;
  auto opened = magmaan::api::model_from_lavaan(composite_syntax, historical);
  REQUIRE_OK(opened);
  CHECK(opened->structure().composite_mode ==
        magmaan::spec::CompositeMode::HenselerOgasawara);

  auto no_composite =
      magmaan::api::frontier::model_from_lavaan_fcsem("f =~ x1 + x2 + x3");
  REQUIRE_FALSE(no_composite.has_value());
  CHECK(no_composite.error().stage ==
        magmaan::api::ErrorStage::UnsupportedCombination);
}

TEST_CASE("api second-order CFA fits ordinal data and matches the correlated model") {
  // Nine ordinal (5-level) items, three first-order factors, one general
  // factor over them — the higher-order + ordinal combination. As with
  // continuous data, the second-order CFA is a just-identified
  // reparameterization of the correlated three-factor model, so the two
  // reach the same chi-square through the DWLS/polychoric path.
  Eigen::MatrixXd X(500, 9);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double t = static_cast<double>(i);
    const double g = std::sin(0.13 * t) + 0.5 * std::cos(0.071 * t);
    const double visual  = g        + 0.45 * std::sin(0.29 * t + 1.0);
    const double textual = 0.85 * g + 0.45 * std::cos(0.31 * t + 2.0);
    const double speed   = 0.70 * g + 0.45 * std::sin(0.23 * t + 3.0);
    const double factor[9] = {visual,  0.9 * visual,  0.8 * visual,
                              textual, 0.9 * textual, 0.8 * textual,
                              speed,   0.9 * speed,   0.8 * speed};
    const double freq[9]  = {0.41, 0.43, 0.47, 0.53, 0.59,
                             0.61, 0.67, 0.71, 0.73};
    const double phase[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    for (Eigen::Index j = 0; j < 9; ++j) {
      const double v = factor[j] + 0.35 * std::sin(freq[j] * t + phase[j]);
      int level = 1;
      if (v > -0.65) level = 2;
      if (v > -0.10) level = 3;
      if (v >  0.45) level = 4;
      if (v >  0.95) level = 5;
      X(i, j) = static_cast<double>(level);
    }
  }
  const auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE_OK(stats);

  // Ordinal models carry explicit threshold (`|`) and scaling (`~*~`) rows —
  // four thresholds for each 5-level item.
  std::string ordinal_rows;
  for (int j = 1; j <= 9; ++j) {
    const std::string x = "x" + std::to_string(j);
    ordinal_rows += x + " | t1 + t2 + t3 + t4\n";
    ordinal_rows += x + " ~*~ 1*" + x + "\n";
  }
  const std::string first_order =
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9\n" + ordinal_rows;
  const std::string second_order =
      first_order + "general =~ visual + textual + speed\n";

  struct Outcome { double chisq; int df; };
  auto fit_ord = [&](const std::string& syntax) -> Outcome {
    const auto model = magmaan::api::model_from_lavaan(syntax);
    REQUIRE_OK(model);
    const auto data = magmaan::api::data_from_ordinal(*model, *stats);
    REQUIRE_OK(data);
    const auto fit =
        magmaan::api::fit(*model, *data, magmaan::api::ordinal_dwls());
    REQUIRE_OK(fit);
    const auto rob = magmaan::api::robust_ordinal(*fit);
    REQUIRE_OK(rob);
    return {rob->chisq_standard, rob->df};
  };

  const auto m1 = fit_ord(first_order);
  const auto m2 = fit_ord(second_order);
  MESSAGE("ordinal first-order:  df=" << m1.df << " chisq=" << m1.chisq);
  MESSAGE("ordinal second-order: df=" << m2.df << " chisq=" << m2.chisq);

  CHECK(m1.df > 0);
  CHECK(m2.df == m1.df);
  CHECK(m2.chisq == doctest::Approx(m1.chisq).epsilon(1e-2));
}

TEST_CASE("api Analysis preserves first error and post-fit calls require fit") {
  const auto model = magmaan::api::model_from_lavaan("f =~ x1 + x2 + x3 + x4");
  REQUIRE(model.has_value());
  const auto raw = continuous_raw();
  const auto stats = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(stats.has_value());
  const auto data = magmaan::api::data_from_sample_stats(*model, *stats);
  REQUIRE(data.has_value());

  // FABIN starts: NLopt L-BFGS stalls from simple_starts() on this synthetic
  // 80-sample ML objective. This test exercises Analysis chain ergonomics,
  // not optimizer numerics, so we follow the R bridge's default start recipe.
  const auto ml_fabin = magmaan::api::ml().starts(magmaan::api::fabin_starts());

  auto no_fit = magmaan::api::analyze(*model, *data)
                    .standard_errors(magmaan::api::expected_information())
                    .fit(ml_fabin)
                    .summary();
  REQUIRE(!no_fit.has_value());
  CHECK(no_fit.error().detail == "standard_errors() requires fit() first");

  auto ok = magmaan::api::analyze(*model, *data)
                .fit(ml_fabin)
                .test(magmaan::api::standard_chi_square())
                .summary();
  REQUIRE_OK(ok);
  CHECK(ok->test.has_value());
}

TEST_CASE("api second-order CFA fits and equals the correlated first-order model") {
  // Nine indicators driven by three first-order factors, themselves driven by
  // one general factor. A second-order CFA with three first-order factors is a
  // just-identified reparameterization of the three-factor correlated model,
  // so the two must reach the same chi-square — a strong end-to-end check that
  // the higher-order `=~` rows are classified and lowered correctly.
  Eigen::MatrixXd X(300, 9);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double t = static_cast<double>(i);
    const double g = std::sin(0.13 * t) + 0.5 * std::cos(0.071 * t);
    const double visual  = g        + 0.5 * std::sin(0.29 * t + 1.0);
    const double textual = 0.85 * g + 0.5 * std::cos(0.31 * t + 2.0);
    const double speed   = 0.70 * g + 0.5 * std::sin(0.23 * t + 3.0);
    X(i, 0) = visual        + 0.30 * std::sin(0.41 * t + 0.1);
    X(i, 1) = 0.9 * visual  + 0.30 * std::cos(0.43 * t + 0.2);
    X(i, 2) = 0.8 * visual  + 0.30 * std::sin(0.47 * t + 0.3);
    X(i, 3) = textual       + 0.30 * std::cos(0.53 * t + 0.4);
    X(i, 4) = 0.9 * textual + 0.30 * std::sin(0.59 * t + 0.5);
    X(i, 5) = 0.8 * textual + 0.30 * std::cos(0.61 * t + 0.6);
    X(i, 6) = speed         + 0.30 * std::sin(0.67 * t + 0.7);
    X(i, 7) = 0.9 * speed   + 0.30 * std::cos(0.71 * t + 0.8);
    X(i, 8) = 0.8 * speed   + 0.30 * std::sin(0.73 * t + 0.9);
  }
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  const auto stats = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(stats.has_value());

  const std::string first_order =
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9";
  const std::string second_order =
      first_order + "\ngeneral =~ visual + textual + speed";

  struct Outcome { magmaan::api::TestResult test; std::ptrdiff_t npar = 0; };
  auto fit_chi = [&](const std::string& syntax) -> Outcome {
    const auto model = magmaan::api::model_from_lavaan(syntax);
    REQUIRE_OK(model);
    // 9-column data resolves only if the first-order factors are kept latent;
    // a misclassification as observed would demand 12 data columns here.
    const auto data = magmaan::api::data_from_sample_stats(*model, *stats);
    REQUIRE_OK(data);
    const auto fit = magmaan::api::fit(*model, *data, magmaan::api::ml());
    REQUIRE_OK(fit);
    const auto tst =
        magmaan::api::test(*fit, magmaan::api::standard_chi_square());
    REQUIRE_OK(tst);
    return {*tst, fit->estimates().theta.size()};
  };

  const auto m1 = fit_chi(first_order);
  const auto m2 = fit_chi(second_order);

  MESSAGE("first-order: npar=" << m1.npar << " df=" << m1.test.df
          << " chi2=" << m1.test.statistic);
  MESSAGE("second-order: npar=" << m2.npar << " df=" << m2.test.df
          << " chi2=" << m2.test.statistic);

  // A second-order CFA over three first-order factors is a just-identified
  // reparameterization of the correlated three-factor model: same parameter
  // count, same degrees of freedom, same chi-square.
  CHECK(m1.npar == 21);
  CHECK(m2.npar == m1.npar);
  CHECK(m2.test.df == m1.test.df);
  CHECK(m2.test.statistic == doctest::Approx(m1.test.statistic).epsilon(1e-3));
}

#undef REQUIRE_OK
