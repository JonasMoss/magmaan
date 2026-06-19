#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "magmaan/data/h_score.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/pairwise_mixed.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/frontier/pairwise.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/frontier/fmg.hpp"
#include "magmaan/spec/build.hpp"

namespace {

Eigen::MatrixXd ordinal_expected_counts(const Eigen::VectorXd& th_i,
                                        const Eigen::VectorXd& th_j,
                                        double rho,
                                        double total) {
  const double inf = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd out(th_i.size() + 1, th_j.size() + 1);
  for (Eigen::Index a = 0; a < out.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == out.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < out.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == out.cols()) ? inf : th_j(b);
      out(a, b) = total * magmaan::data::ordinal_bvn_rect_prob(
          lo_i, hi_i, lo_j, hi_j, rho);
    }
  }
  return out;
}

double h_score_pair_objective(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho,
    const magmaan::data::PolychoricHScoreOptions& options) {
  const double inf = std::numeric_limits<double>::infinity();
  const double total = counts.sum();
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? inf : th_j(b);
      const double p = std::max(
          1.4901161193847656e-8,
          magmaan::data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double t = counts(a, b) / (total * p);
      auto h = magmaan::data::eval_polychoric_h_score(t, options);
      if (!h.has_value()) return std::numeric_limits<double>::quiet_NaN();
      out += p * h->phi;
    }
  }
  return out;
}

double dpd_pair_objective(const Eigen::MatrixXd& counts,
                          const Eigen::VectorXd& th_i,
                          const Eigen::VectorXd& th_j,
                          double rho,
                          double alpha) {
  const double inf = std::numeric_limits<double>::infinity();
  const double total = counts.sum();
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? inf : th_j(b);
      const double p = std::max(
          1.4901161193847656e-8,
          magmaan::data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double fhat = counts(a, b) / total;
      const double pa = std::pow(p, alpha);
      out += p * pa - ((1.0 + alpha) / alpha) * fhat * pa;
    }
  }
  return out;
}

Eigen::MatrixXd ordinal_pair_score_rows_from_counts(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho) {
  std::int64_t n = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      n += static_cast<std::int64_t>(std::llround(counts(a, b)));
    }
  }
  Eigen::VectorXi xi(n);
  Eigen::VectorXi xj(n);
  Eigen::Index row = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const auto reps = static_cast<Eigen::Index>(std::llround(counts(a, b)));
      for (Eigen::Index k = 0; k < reps; ++k) {
        xi(row) = static_cast<int>(a);
        xj(row) = static_cast<int>(b);
        ++row;
      }
    }
  }

  auto scores = magmaan::data::ordinal_pair_scores(xi, xj, rho, th_i, th_j);
  REQUIRE(scores.has_value());
  Eigen::MatrixXd out(n, th_i.size() + th_j.size() + 1);
  out.leftCols(th_i.size()) = scores->threshold_i;
  out.middleCols(th_i.size(), th_j.size()) = scores->threshold_j;
  out.col(out.cols() - 1) = scores->rho;
  return out;
}

Eigen::MatrixXd ordinal_data_from_pair_counts(const Eigen::MatrixXd& counts) {
  Eigen::Index n = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      n += static_cast<Eigen::Index>(std::llround(counts(a, b)));
    }
  }
  Eigen::MatrixXd out(n, 2);
  Eigen::Index row = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const auto reps = static_cast<Eigen::Index>(std::llround(counts(a, b)));
      for (Eigen::Index k = 0; k < reps; ++k) {
        out(row, 0) = static_cast<double>(a + 1);
        out(row, 1) = static_cast<double>(b + 1);
        ++row;
      }
    }
  }
  return out;
}

double std_normal_cdf(double x) noexcept {
  if (x == std::numeric_limits<double>::infinity()) return 1.0;
  if (x == -std::numeric_limits<double>::infinity()) return 0.0;
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double std_normal_pdf(double x) noexcept {
  if (!std::isfinite(x)) return 0.0;
  constexpr double inv_sqrt_2pi = 0.39894228040143267794;
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

struct GammaDiagInfluenceProbe {
  Eigen::MatrixXd SC;
  Eigen::MatrixXd B;
  Eigen::MatrixXd Gamma;
  std::vector<Eigen::MatrixXd> b_case;
};

GammaDiagInfluenceProbe gamma_diag_influence_probe_2var(
    const Eigen::MatrixXi& Xcat,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::MatrixXd& R) {
  REQUIRE(Xcat.cols() == 2);
  REQUIRE(levels.size() == 2);
  const Eigen::Index n = Xcat.rows();
  Eigen::Index nth = 0;
  std::array<Eigen::Index, 2> th_start{};
  std::array<Eigen::Index, 2> th_len{};
  std::array<Eigen::VectorXd, 2> th_by_var;
  for (Eigen::Index j = 0; j < 2; ++j) {
    th_start[static_cast<std::size_t>(j)] = nth;
    th_len[static_cast<std::size_t>(j)] =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    th_by_var[static_cast<std::size_t>(j)] =
        thresholds.segment(nth, th_len[static_cast<std::size_t>(j)]);
    nth += th_len[static_cast<std::size_t>(j)];
  }
  const Eigen::Index mdim = nth + 1;

  Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
  constexpr double prob_floor = 1.4901161193847656e-8;
  const double inf = std::numeric_limits<double>::infinity();
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < 2; ++j) {
      const int c = Xcat(r, j);
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -inf : thj(c - 1);
      const double hi = (c == thj.size()) ? inf : thj(c);
      const double pr = std::max(prob_floor, std_normal_cdf(hi) - std_normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC_TH(r, base + c) += std_normal_pdf(thj(c)) / pr;
      if (c > 0) SC_TH(r, base + c - 1) -= std_normal_pdf(thj(c - 1)) / pr;
    }
  }

  auto ps = magmaan::data::ordinal_pair_scores(
      Xcat.col(1), Xcat.col(0), R(1, 0), th_by_var[1], th_by_var[0]);
  REQUIRE(ps.has_value());

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(nth) = SC_TH;
  SC.col(nth) = ps->rho;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(mdim, mdim);
  const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
  for (Eigen::Index j = 0; j < 2; ++j) {
    const Eigen::Index s = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index l = th_len[static_cast<std::size_t>(j)];
    B.block(s, s, l, l) = INNER_TH.block(s, s, l, l);
  }
  B.block(nth, th_start[1], 1, th_len[1]) =
      ps->rho.transpose() * ps->threshold_i;
  B.block(nth, th_start[0], 1, th_len[0]) =
      ps->rho.transpose() * ps->threshold_j;
  B(nth, nth) = ps->rho.squaredNorm();

  const Eigen::MatrixXd B_inv = B.inverse();
  GammaDiagInfluenceProbe out;
  out.SC = SC;
  out.B = B;
  out.Gamma = static_cast<double>(n) * B_inv * INNER * B_inv.transpose();
  out.Gamma = 0.5 * (out.Gamma + out.Gamma.transpose()).eval();
  out.b_case.reserve(static_cast<std::size_t>(n));

  const Eigen::MatrixXd pair_a21_i = ps->rho.asDiagonal() * ps->threshold_i;
  const Eigen::MatrixXd pair_a21_j = ps->rho.asDiagonal() * ps->threshold_j;
  for (Eigen::Index r = 0; r < n; ++r) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index j = 0; j < 2; ++j) {
      const Eigen::Index s = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index l = th_len[static_cast<std::size_t>(j)];
      const Eigen::VectorXd sv = SC_TH.row(r).segment(s, l).transpose();
      bi.block(s, s, l, l) = sv * sv.transpose();
    }
    bi(nth, nth) = ps->rho(r) * ps->rho(r);
    bi.block(nth, th_start[1], 1, th_len[1]) = pair_a21_i.row(r);
    bi.block(nth, th_start[0], 1, th_len[0]) = pair_a21_j.row(r);
    out.b_case.push_back(std::move(bi));
  }
  return out;
}

Eigen::VectorXd finite_diff_gamma_diag_case_influence(
    const GammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps = 1e-6) {
  const Eigen::Index n = probe.SC.rows();
  const Eigen::MatrixXd A = probe.B / static_cast<double>(n);
  const Eigen::MatrixXd V =
      (probe.SC.transpose() * probe.SC) / static_cast<double>(n);
  const Eigen::MatrixXd A_inv = A.inverse();
  const Eigen::MatrixXd Gamma = A_inv * V * A_inv.transpose();
  const Eigen::VectorXd s_i = probe.SC.row(row).transpose();
  const Eigen::MatrixXd A_eps =
      A + eps * (probe.b_case[static_cast<std::size_t>(row)] - A);
  const Eigen::MatrixXd V_eps = V + eps * (s_i * s_i.transpose() - V);
  const Eigen::MatrixXd A_eps_inv = A_eps.inverse();
  const Eigen::MatrixXd Gamma_eps =
      A_eps_inv * V_eps * A_eps_inv.transpose();
  return (Gamma_eps.diagonal() - Gamma.diagonal()) / eps;
}

double symmetric_condition_number(const Eigen::MatrixXd& x) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (x + x.transpose()));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite()) {
    return std::numeric_limits<double>::infinity();
  }
  const double max_eval = es.eigenvalues().cwiseAbs().maxCoeff();
  const double min_eval = es.eigenvalues().cwiseAbs().minCoeff();
  if (!(min_eval > 0.0) || !std::isfinite(max_eval)) {
    return std::numeric_limits<double>::infinity();
  }
  return max_eval / min_eval;
}

}  // namespace

TEST_CASE("Ordinal workspace adapters split moments from Gamma cache") {
  magmaan::data::OrdinalStats stats;
  Eigen::MatrixXd R(2, 2);
  R << 1.0, 0.35,
       0.35, 1.0;
  Eigen::VectorXd thresholds(2);
  thresholds << -0.25, 0.40;
  Eigen::MatrixXd gamma(3, 3);
  gamma << 4.0, 0.2, 0.1,
           0.2, 3.0, 0.3,
           0.1, 0.3, 2.0;
  stats.R.push_back(R);
  stats.thresholds.push_back(thresholds);
  stats.threshold_ov.push_back({0, 1});
  stats.threshold_level.push_back({1, 1});
  stats.NACOV.push_back(gamma);
  stats.W_dwls.push_back(gamma.diagonal().cwiseInverse().asDiagonal());
  stats.W_wls.push_back(gamma.inverse());
  stats.n_obs.push_back(80);
  stats.n_levels.push_back({2, 2});
  stats.ov_names.push_back({"y1", "y2"});

  auto moments = magmaan::data::ordinal_moments_from_stats(stats);
  CHECK(moments.R[0].isApprox(stats.R[0], 0.0));
  CHECK(moments.thresholds[0].isApprox(stats.thresholds[0], 0.0));
  CHECK(moments.threshold_ov[0] == stats.threshold_ov[0]);
  CHECK(moments.n_obs[0] == 80);
  CHECK(moments.ov_names[0][1] == "y2");

  auto cache = magmaan::data::ordinal_gamma_cache_from_stats(stats);
  REQUIRE(cache.block_count() == 1);
  CHECK(cache.blocks[0].has_full);
  CHECK(cache.blocks[0].has_diagonal);
  CHECK(cache.blocks[0].has_dwls_weight);
  CHECK(cache.blocks[0].has_wls_weight);
  CHECK(cache.blocks[0].gamma.isApprox(gamma, 0.0));
  CHECK(cache.blocks[0].diagonal.isApprox(gamma.diagonal(), 0.0));
}

TEST_CASE("OrdinalGammaCache can materialize DWLS without full Gamma") {
  Eigen::VectorXd diagonal(3);
  diagonal << 2.0, 4.0, 5.0;
  auto cache = magmaan::data::ordinal_gamma_cache_from_diagonal({diagonal});

  REQUIRE(cache.block_count() == 1);
  CHECK(cache.blocks[0].has_diagonal);
  CHECK_FALSE(cache.blocks[0].has_full);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);

  auto dwls = magmaan::data::ordinal_gamma_cache_ensure_dwls_weights(cache);
  REQUIRE(dwls.has_value());
  CHECK(cache.blocks[0].has_dwls_weight);
  CHECK(cache.blocks[0].w_dwls.diagonal().isApprox(
      diagonal.cwiseInverse(), 0.0));

  auto wls = magmaan::data::ordinal_gamma_cache_ensure_wls_weights(cache);
  CHECK_FALSE(wls.has_value());
}

TEST_CASE("OrdinalWeightPlan encodes fit-only Gamma cost rules") {
  using magmaan::data::OrdinalEstimatorKind;
  using magmaan::data::OrdinalGammaMaterialization;
  using magmaan::data::OrdinalWorkspacePurpose;

  auto uls = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitOnly, OrdinalEstimatorKind::ULS);
  CHECK(uls.materialization == OrdinalGammaMaterialization::None);

  auto dwls = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitOnly, OrdinalEstimatorKind::DWLS);
  CHECK(dwls.materialization == OrdinalGammaMaterialization::Diagonal);

  auto wls = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitOnly, OrdinalEstimatorKind::WLS);
  CHECK(wls.materialization == OrdinalGammaMaterialization::Full);

  auto inference = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitPlusInference, OrdinalEstimatorKind::DWLS);
  CHECK(inference.materialization == OrdinalGammaMaterialization::Full);
}

TEST_CASE("Ordinal raw workspace builder honors fit-only materialization") {
  std::mt19937 rng(20260609);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(420, 4);
  const double loading[4] = {0.88, 0.80, 0.72, 0.64};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.45);
    }
  }

  auto legacy = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(legacy.has_value());

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS);
  auto uls = magmaan::data::ordinal_workspace_from_integer_data({X}, uls_plan);
  REQUIRE(uls.has_value());
  REQUIRE(uls->moments.R.size() == 1);
  CHECK(uls->moments.R[0].isApprox(legacy->R[0], 0.0));
  CHECK(uls->moments.thresholds[0].isApprox(legacy->thresholds[0], 0.0));
  CHECK(uls->moments.threshold_ov[0] == legacy->threshold_ov[0]);
  CHECK(uls->moments.threshold_level[0] == legacy->threshold_level[0]);
  CHECK(uls->moments.n_levels[0] == legacy->n_levels[0]);
  CHECK(uls->gamma_cache.block_count() == 0);

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto dwls =
      magmaan::data::ordinal_workspace_from_integer_data({X}, dwls_plan);
  REQUIRE(dwls.has_value());
  REQUIRE(dwls->gamma_cache.block_count() == 1);
  CHECK(dwls->moments.R[0].isApprox(legacy->R[0], 0.0));
  CHECK(dwls->moments.thresholds[0].isApprox(legacy->thresholds[0], 0.0));
  CHECK(dwls->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls->gamma_cache.blocks[0].has_full);
  CHECK_FALSE(dwls->gamma_cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(dwls->gamma_cache.blocks[0].has_wls_weight);
  CHECK(dwls->gamma_cache.blocks[0].diagonal.isApprox(
      legacy->NACOV[0].diagonal(), 1e-8));

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto wls = magmaan::data::ordinal_workspace_from_integer_data({X}, wls_plan);
  REQUIRE(wls.has_value());
  REQUIRE(wls->gamma_cache.block_count() == 1);
  CHECK(wls->gamma_cache.blocks[0].has_full);
  CHECK(wls->gamma_cache.blocks[0].has_wls_weight);
}

TEST_CASE("Cached ordinal DWLS fit consumes lazy raw workspace diagonal") {
  std::mt19937 rng(20260610);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(500, 3);
  const double loading[3] = {0.90, 0.78, 0.68};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.45) + (y > 0.55);
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, plan);
  REQUIRE(workspace.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto x0 = magmaan::estimate::ordinal_start_values(
      *pt, *mr, workspace->moments, {});
  REQUIRE(x0.has_value());
  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0);
  auto cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan, *x0);
  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy DWLS failed: " << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(cached.has_value(),
      "cached DWLS failed: " << (cached.has_value() ? "" : cached.error().detail));

  CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 1e-6);
  REQUIRE(workspace->gamma_cache.block_count() == 1);
  CHECK(workspace->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(workspace->gamma_cache.blocks[0].has_full);
}

TEST_CASE("Cached ordinal DWLS fit consumes diagonal Gamma only") {
  std::mt19937 rng(20260525);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(500, 3);
  const double loading[3] = {0.90, 0.78, 0.68};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.45) + (y > 0.55);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);
  auto cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());
  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  Eigen::VectorXd x0_profile = *x0;
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      x0_profile(pt_prepared.free[row] - 1) += 1.25;
    }
  }
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0);
  auto cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &cache, {}, plan, x0_profile);
  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy DWLS failed: " << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(cached.has_value(),
      "cached DWLS failed: " << (cached.has_value() ? "" : cached.error().detail));

  CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 1e-6);
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      CHECK(cached->theta(pt_prepared.free[row] - 1) ==
            doctest::Approx((*x0)(pt_prepared.free[row] - 1)));
    }
  }
  REQUIRE(cache.block_count() == 1);
  CHECK(cache.blocks[0].has_diagonal);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(cache.blocks[0].has_full);
  CHECK_FALSE(cache.blocks[0].has_wls_weight);
}

TEST_CASE("Cached ordinal DWLS fit-plus-inference reuses Gamma for robust reporting") {
  std::mt19937 rng(20260527);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(560, 4);
  const double loading[4] = {0.88, 0.80, 0.72, 0.64};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.50) + (y > 0.45);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  magmaan::data::OrdinalGammaCache fit_cache;
  fit_cache.blocks.resize(1);
  fit_cache.blocks[0].gamma = stats->NACOV[0];
  fit_cache.blocks[0].has_full = true;

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

  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());
  auto fit_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitPlusInference,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto cached_fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &fit_cache, {}, fit_plan, *x0);
  REQUIRE_MESSAGE(cached_fit.has_value(),
      "cached DWLS fit-plus-inference failed: "
          << (cached_fit.has_value() ? "" : cached_fit.error().detail));
  CHECK(fit_cache.blocks[0].has_full);
  CHECK(fit_cache.blocks[0].has_diagonal);
  CHECK(fit_cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(fit_cache.blocks[0].has_wls_weight);

  auto materialized_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *cached_fit,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(materialized_rob.has_value(),
      "materialized robust ordinal failed: "
          << (materialized_rob.has_value() ? "" : materialized_rob.error().detail));

  magmaan::data::OrdinalGammaCache inference_cache;
  inference_cache.blocks.resize(1);
  inference_cache.blocks[0].gamma = stats->NACOV[0];
  inference_cache.blocks[0].has_full = true;
  auto infer_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::InferenceOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto cached_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, moments, inference_cache, *cached_fit, infer_plan);
  REQUIRE_MESSAGE(cached_rob.has_value(),
      "cached robust ordinal failed: "
          << (cached_rob.has_value() ? "" : cached_rob.error().detail));

  CHECK(inference_cache.blocks[0].has_full);
  CHECK(inference_cache.blocks[0].has_diagonal);
  CHECK(inference_cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(inference_cache.blocks[0].has_wls_weight);
  CHECK(cached_rob->chisq_standard ==
        doctest::Approx(materialized_rob->chisq_standard).epsilon(1e-12));
  CHECK(cached_rob->df == materialized_rob->df);
  CHECK((cached_rob->se - materialized_rob->se).cwiseAbs().maxCoeff() < 1e-10);
  CHECK((cached_rob->vcov - materialized_rob->vcov).cwiseAbs().maxCoeff() <
        1e-10);
  CHECK((cached_rob->eigvals - materialized_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);

  auto materialized_wls_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *cached_fit,
      magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(materialized_wls_rob.has_value(),
      "materialized WLS robust ordinal failed: "
          << (materialized_wls_rob.has_value()
                  ? ""
                  : materialized_wls_rob.error().detail));

  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::InferenceOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto cached_wls_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, moments, wls_cache, *cached_fit, wls_plan);
  REQUIRE_MESSAGE(cached_wls_rob.has_value(),
      "cached WLS robust ordinal failed: "
          << (cached_wls_rob.has_value() ? "" : cached_wls_rob.error().detail));
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(wls_cache.blocks[0].has_dwls_weight);
  CHECK((cached_wls_rob->se - materialized_wls_rob->se)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_wls_rob->vcov - materialized_wls_rob->vcov)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_wls_rob->eigvals - materialized_wls_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
}

// Anchors the ordinal FMG path (Paper 2 gate): the estimator-agnostic FMG
// eigenvalue-tail transform fed the `robust_ordinal` (chisq_standard, df,
// eigvals) triple reproduces the same Satorra-Bentler scaling `robust_ordinal`
// already reports, and the tail transforms (pEBA/pOLS) yield proper p-values on
// the polychoric UGamma spectrum. This is exactly the composition the R
// `fmg_tests_ordinal()` wrapper performs (infer_ordinal_robust -> infer_fmg_test).
TEST_CASE("Ordinal FMG transforms consume the robust_ordinal UGamma spectrum") {
  std::mt19937 rng(20260613);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(600, 4);
  const double loading[4] = {0.86, 0.78, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.50);
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
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(rob.has_value(),
      "robust_ordinal failed: " << (rob.has_value() ? "" : rob.error().detail));
  REQUIRE(rob->df > 0);
  REQUIRE(rob->eigvals.size() == rob->df);
  // The projected UGamma spectrum is positive-definite, so FMG's default
  // negative-eigenvalue truncation is a no-op here (keeps the SB comparison exact).
  CHECK(rob->eigvals.minCoeff() > 0.0);

  // FMG SatorraBentler must reproduce robust_ordinal's stored SB scaling, since
  // both apply the same robust::satorra_bentler() to the same (T, df, eigvals).
  const auto fmg_sb = magmaan::robust::frontier::fmg_test(
      rob->chisq_standard, rob->df, rob->eigvals,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::SatorraBentler,
          .truncate_negative = false});
  const double sb_p_direct = magmaan::inference::chi2_pvalue(
      rob->satorra_bentler.chi2_scaled, rob->satorra_bentler.df);
  CHECK(fmg_sb.p_value == doctest::Approx(sb_p_direct).epsilon(1e-12));

  // The eigenvalue-tail transforms (the actual FMG winners) yield proper
  // p-values on the ordinal spectrum.
  for (const auto method : {magmaan::robust::frontier::FmgMethod::Peba,
                            magmaan::robust::frontier::FmgMethod::Pols,
                            magmaan::robust::frontier::FmgMethod::PenalizedAll}) {
    const auto r = magmaan::robust::frontier::fmg_test(
        rob->chisq_standard, rob->df, rob->eigvals,
        magmaan::robust::frontier::FmgOptions{.method = method, .param = 4.0});
    CHECK(std::isfinite(r.p_value));
    CHECK(r.p_value >= 0.0);
    CHECK(r.p_value <= 1.0);
  }
}

TEST_CASE("Cached ordinal WLS fit uses Schur threshold profiling") {
  std::mt19937 rng(20260526);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  const double loading[4] = {0.88, 0.76, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.35);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);
  magmaan::data::OrdinalGammaCache cache;
  cache.blocks.resize(1);
  cache.blocks[0].gamma = stats->NACOV[0];
  cache.blocks[0].has_full = true;

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

  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());
  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  Eigen::VectorXd x0_profile = *x0;
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      x0_profile(pt_prepared.free[row] - 1) -= 0.90;
    }
  }

  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0);
  auto cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &cache, {}, plan, x0_profile);
  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy WLS failed: " << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(cached.has_value(),
      "cached WLS failed: " << (cached.has_value() ? "" : cached.error().detail));

  CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-7));
  CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 2e-5);
  CHECK(cache.blocks[0].has_full);
  CHECK(cache.blocks[0].has_wls_weight);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Cached ordinal profiling keeps fixed threshold rows in the objective") {
  std::mt19937 rng(20260529);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(540, 4);
  const double loading[4] = {0.86, 0.78, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.60) + (y > 0.40);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  REQUIRE(std::abs(stats->thresholds[0](0)) > 0.05);
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | 0*t1 + t2\n"
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
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto uls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_cached.has_value(),
      "cached ULS failed: "
          << (uls_cached.has_value() ? "" : uls_cached.error().detail));
  CHECK(std::isfinite(uls_cached->fmin));

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto dwls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &dwls_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_legacy.has_value(),
      "legacy DWLS failed: "
          << (dwls_legacy.has_value() ? "" : dwls_legacy.error().detail));
  REQUIRE_MESSAGE(dwls_cached.has_value(),
      "cached DWLS failed: "
          << (dwls_cached.has_value() ? "" : dwls_cached.error().detail));
  CHECK(dwls_cached->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-8));
  CHECK((dwls_cached->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        2e-5);
  CHECK(dwls_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls_cache.blocks[0].has_full);
  CHECK_FALSE(dwls_cache.blocks[0].has_dwls_weight);

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_legacy.has_value(),
      "legacy WLS failed: "
          << (wls_legacy.has_value() ? "" : wls_legacy.error().detail));
  REQUIRE_MESSAGE(wls_cached.has_value(),
      "cached WLS failed: "
          << (wls_cached.has_value() ? "" : wls_cached.error().detail));
  CHECK(wls_cached->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-7));
  CHECK((wls_cached->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(wls_cache.blocks[0].has_dwls_weight);

  magmaan::data::OrdinalGammaCache snlls_wls_cache;
  snlls_wls_cache.blocks.resize(1);
  snlls_wls_cache.blocks[0].gamma = stats->NACOV[0];
  snlls_wls_cache.blocks[0].has_full = true;
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_wls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "SNLLS WLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_cached->fmin).epsilon(1e-8));
  CHECK((wls_snlls->theta - wls_cached->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(snlls_wls_cache.blocks[0].has_full);
  CHECK(snlls_wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(snlls_wls_cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Cached ordinal profiling handles shared threshold labels") {
  std::mt19937 rng(20260530);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(560, 4);
  const double loading[4] = {0.84, 0.76, 0.68, 0.60};
  const double shift[4] = {-0.15, 0.10, 0.00, 0.05};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = shift[j] + loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.45);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  REQUIRE(std::abs(stats->thresholds[0](0) - stats->thresholds[0](2)) >
          0.01);
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + t2\n"
      "x2 | a*t1 + t2\n"
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
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  std::vector<Eigen::Index> shared_threshold_free;
  std::int32_t shared_group = -1;
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] != magmaan::parse::Op::Threshold ||
        pt_prepared.free[row] <= 0) {
      continue;
    }
    const std::int32_t group =
        pt_prepared.eq_groups[static_cast<std::size_t>(pt_prepared.free[row] - 1)];
    int count = 0;
    for (std::size_t other = 0; other < pt_prepared.size(); ++other) {
      if (pt_prepared.op[other] == magmaan::parse::Op::Threshold &&
          pt_prepared.free[other] > 0 &&
          pt_prepared.eq_groups[static_cast<std::size_t>(
              pt_prepared.free[other] - 1)] == group) {
        ++count;
      }
    }
    if (count == 2) {
      shared_group = group;
      break;
    }
  }
  REQUIRE(shared_group >= 0);
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0 &&
        pt_prepared.eq_groups[static_cast<std::size_t>(
            pt_prepared.free[row] - 1)] == shared_group) {
      shared_threshold_free.push_back(pt_prepared.free[row] - 1);
    }
  }
  REQUIRE(shared_threshold_free.size() == 2);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto uls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_cached.has_value(),
      "cached ULS failed: "
          << (uls_cached.has_value() ? "" : uls_cached.error().detail));
  CHECK(std::isfinite(uls_cached->fmin));
  CHECK(uls_cached->theta(shared_threshold_free[0]) ==
        doctest::Approx(uls_cached->theta(shared_threshold_free[1])).epsilon(1e-12));

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto dwls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &dwls_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_legacy.has_value(),
      "legacy DWLS failed: "
          << (dwls_legacy.has_value() ? "" : dwls_legacy.error().detail));
  REQUIRE_MESSAGE(dwls_cached.has_value(),
      "cached DWLS failed: "
          << (dwls_cached.has_value() ? "" : dwls_cached.error().detail));
  CHECK(dwls_cached->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-8));
  CHECK((dwls_cached->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        2e-5);
  CHECK(dwls_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls_cache.blocks[0].has_full);
  CHECK_FALSE(dwls_cache.blocks[0].has_dwls_weight);

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_legacy.has_value(),
      "legacy WLS failed: "
          << (wls_legacy.has_value() ? "" : wls_legacy.error().detail));
  REQUIRE_MESSAGE(wls_cached.has_value(),
      "cached WLS failed: "
          << (wls_cached.has_value() ? "" : wls_cached.error().detail));
  CHECK(wls_cached->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-7));
  CHECK((wls_cached->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(wls_cache.blocks[0].has_dwls_weight);

  magmaan::data::OrdinalGammaCache snlls_wls_cache;
  snlls_wls_cache.blocks.resize(1);
  snlls_wls_cache.blocks[0].gamma = stats->NACOV[0];
  snlls_wls_cache.blocks[0].has_full = true;
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_wls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "SNLLS WLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_cached->fmin).epsilon(1e-8));
  CHECK((wls_snlls->theta - wls_cached->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(snlls_wls_cache.blocks[0].has_full);
  CHECK(snlls_wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(snlls_wls_cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Cached ordinal ULS fit does not require Gamma") {
  Eigen::MatrixXd X(240, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 10; ++rep) {
    for (int x1 = 1; x1 <= 3; ++x1) {
      for (int x2 = 1; x2 <= 3; ++x2) {
        for (int x3 = 1; x3 <= 3; ++x3) {
          if (r >= X.rows()) break;
          X(r, 0) = x1;
          X(r, 1) = x2;
          X(r, 2) = x3;
          ++r;
        }
      }
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  magmaan::data::OrdinalGammaCache cache;
  cache.blocks.resize(1);
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS);
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &cache, {}, plan, *x0);
  REQUIRE_MESSAGE(fit.has_value(),
      "cached ULS failed: " << (fit.has_value() ? "" : fit.error().detail));
  CHECK(std::isfinite(fit->fmin));
  CHECK_FALSE(cache.blocks[0].has_diagonal);
  CHECK_FALSE(cache.blocks[0].has_full);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(cache.blocks[0].has_wls_weight);
}

TEST_CASE("Ordinal SNLLS profiles thresholds and linear covariance block") {
  std::mt19937 rng(20260528);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(620, 4);
  const double loading[4] = {0.90, 0.82, 0.74, 0.66};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.45) + (y > 0.50);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

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
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS);
  auto uls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto uls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, nullptr, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto uls_snlls_full_thresholds =
      magmaan::estimate::fit_ordinal_snlls_full_thresholds(
          *pt, *mr, moments, nullptr, uls_plan, *x0,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_bounded.has_value(),
      "bounded ULS failed: "
          << (uls_bounded.has_value() ? "" : uls_bounded.error().detail));
  REQUIRE_MESSAGE(uls_snlls.has_value(),
      "SNLLS ULS failed: "
          << (uls_snlls.has_value() ? "" : uls_snlls.error().detail));
  REQUIRE_MESSAGE(uls_snlls_full_thresholds.has_value(),
      "full-threshold SNLLS ULS failed: "
          << (uls_snlls_full_thresholds.has_value()
                  ? ""
                  : uls_snlls_full_thresholds.error().detail));
  CHECK(uls_snlls->fmin ==
        doctest::Approx(uls_bounded->fmin).epsilon(1e-8));
  CHECK((uls_snlls->theta - uls_bounded->theta).cwiseAbs().maxCoeff() < 2e-5);
  CHECK(uls_snlls_full_thresholds->fmin ==
        doctest::Approx(uls_bounded->fmin).epsilon(1e-8));
  CHECK((uls_snlls_full_thresholds->theta - uls_bounded->theta)
            .cwiseAbs()
            .maxCoeff() < 2e-5);

  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      CHECK(uls_snlls->theta(pt_prepared.free[row] - 1) ==
            doctest::Approx((*x0)(pt_prepared.free[row] - 1)));
    }
  }

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto bounded_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto snlls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &bounded_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_cache, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto full_threshold_dwls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_snlls_full_thresholds =
      magmaan::estimate::fit_ordinal_snlls_full_thresholds(
          *pt, *mr, moments, &full_threshold_dwls_cache, dwls_plan, *x0,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_bounded.has_value(),
      "bounded DWLS failed: "
          << (dwls_bounded.has_value() ? "" : dwls_bounded.error().detail));
  REQUIRE_MESSAGE(dwls_snlls.has_value(),
      "SNLLS DWLS failed: "
          << (dwls_snlls.has_value() ? "" : dwls_snlls.error().detail));
  REQUIRE_MESSAGE(dwls_snlls_full_thresholds.has_value(),
      "full-threshold SNLLS DWLS failed: "
          << (dwls_snlls_full_thresholds.has_value()
                  ? ""
                  : dwls_snlls_full_thresholds.error().detail));
  CHECK(dwls_snlls->fmin ==
        doctest::Approx(dwls_bounded->fmin).epsilon(1e-8));
  CHECK((dwls_snlls->theta - dwls_bounded->theta).cwiseAbs().maxCoeff() <
        2e-5);
  CHECK(dwls_snlls_full_thresholds->fmin ==
        doctest::Approx(dwls_bounded->fmin).epsilon(1e-8));
  CHECK((dwls_snlls_full_thresholds->theta - dwls_bounded->theta)
            .cwiseAbs()
            .maxCoeff() < 2e-5);
  CHECK(snlls_cache.blocks[0].has_diagonal);
  CHECK_FALSE(snlls_cache.blocks[0].has_full);
  CHECK_FALSE(snlls_cache.blocks[0].has_dwls_weight);

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  magmaan::data::OrdinalGammaCache bounded_wls_cache;
  bounded_wls_cache.blocks.resize(1);
  bounded_wls_cache.blocks[0].gamma = stats->NACOV[0];
  bounded_wls_cache.blocks[0].has_full = true;
  magmaan::data::OrdinalGammaCache snlls_wls_cache;
  snlls_wls_cache.blocks.resize(1);
  snlls_wls_cache.blocks[0].gamma = stats->NACOV[0];
  snlls_wls_cache.blocks[0].has_full = true;
  magmaan::data::OrdinalGammaCache full_threshold_wls_cache;
  full_threshold_wls_cache.blocks.resize(1);
  full_threshold_wls_cache.blocks[0].gamma = stats->NACOV[0];
  full_threshold_wls_cache.blocks[0].has_full = true;
  auto wls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &bounded_wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_wls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_snlls_full_thresholds =
      magmaan::estimate::fit_ordinal_snlls_full_thresholds(
          *pt, *mr, moments, &full_threshold_wls_cache, wls_plan, *x0,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_bounded.has_value(),
      "bounded WLS failed: "
          << (wls_bounded.has_value() ? "" : wls_bounded.error().detail));
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "SNLLS WLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  REQUIRE_MESSAGE(wls_snlls_full_thresholds.has_value(),
      "full-threshold SNLLS WLS failed: "
          << (wls_snlls_full_thresholds.has_value()
                  ? ""
                  : wls_snlls_full_thresholds.error().detail));
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_bounded->fmin).epsilon(1e-8));
  CHECK((wls_snlls->theta - wls_bounded->theta).cwiseAbs().maxCoeff() <
        3e-5);
  CHECK(wls_snlls_full_thresholds->fmin ==
        doctest::Approx(wls_bounded->fmin).epsilon(1e-8));
  CHECK((wls_snlls_full_thresholds->theta - wls_bounded->theta)
            .cwiseAbs()
            .maxCoeff() < 3e-5);
  CHECK(snlls_wls_cache.blocks[0].has_full);
  CHECK(snlls_wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(snlls_wls_cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Ordinal full-threshold SNLLS accepts linear threshold constraints") {
  std::mt19937 rng(20260602);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  const double loading[4] = {0.86, 0.79, 0.72, 0.68};
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
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + b*t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n"
      "a + b == 0\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto cache =
      magmaan::data::ordinal_gamma_cache_from_diagonal({stats->NACOV[0].diagonal()});
  auto snlls = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
      *pt, *mr, moments, &cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);

  REQUIRE_MESSAGE(bounded.has_value(),
      "bounded DWLS failed: "
          << (bounded.has_value() ? "" : bounded.error().detail));
  REQUIRE_MESSAGE(snlls.has_value(),
      "full-threshold SNLLS DWLS failed: "
          << (snlls.has_value() ? "" : snlls.error().detail));
  CHECK(snlls->fmin == doctest::Approx(bounded->fmin).epsilon(1e-8));
  CHECK((snlls->theta - bounded->theta).cwiseAbs().maxCoeff() < 3e-5);

  // The threshold-profiled path absorbs threshold-only linear constraints
  // into the threshold design (general H) and must agree with the
  // full-threshold SNLLS and bounded fits.
  auto profiled = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(profiled.has_value(),
      "threshold-profiled SNLLS DWLS failed: "
          << (profiled.has_value() ? "" : profiled.error().detail));
  CHECK(profiled->fmin == doctest::Approx(bounded->fmin).epsilon(1e-8));
  CHECK((profiled->theta - bounded->theta).cwiseAbs().maxCoeff() < 3e-5);
  // The constraint itself holds at the profiled solution.
  double a_hat = 0.0;
  double b_hat = 0.0;
  bool found_a = false;
  bool found_b = false;
  for (std::size_t row = 0; row < pt->size(); ++row) {
    if (pt->op[row] != magmaan::parse::Op::Threshold) continue;
    if (pt->free[row] <= 0) continue;
    // x1's two thresholds carry labels a and b; x1 rows come first.
    if (!found_a) {
      a_hat = profiled->theta(pt->free[row] - 1);
      found_a = true;
    } else if (!found_b) {
      b_hat = profiled->theta(pt->free[row] - 1);
      found_b = true;
    }
  }
  REQUIRE(found_a);
  REQUIRE(found_b);
  CHECK(std::abs(a_hat + b_hat) < 1e-10);
}

namespace {

// Synthetic one-factor 3-category block for multi-group profiling tests.
Eigen::MatrixXd ordinal_test_block(std::uint32_t seed,
                                   Eigen::Index n,
                                   const std::array<double, 4>& loading,
                                   double cut1,
                                   double cut2) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < 4; ++j) {
      const double lj = loading[static_cast<std::size_t>(j)];
      const double eps = std::sqrt(1.0 - lj * lj) * norm(rng);
      const double y = lj * eta + eps;
      X(i, j) = 1.0 + (y > cut1) + (y > cut2);
    }
  }
  return X;
}

constexpr const char* k2GroupOrdinalCfa =
    "f =~ x1 + x2 + x3 + x4\n"
    "x1 | t1 + t2\n"
    "x2 | t1 + t2\n"
    "x3 | t1 + t2\n"
    "x4 | t1 + t2\n"
    "x1 ~*~ 1*x1\n"
    "x2 ~*~ 1*x2\n"
    "x3 ~*~ 1*x3\n"
    "x4 ~*~ 1*x4\n";

}  // namespace

// Regression for the joint cross-block profiling refactor: the per-block
// profiled solve used to build each block's normal matrix over the global
// gamma coordinate (singular for any second group), so the profiled bounded
// and SNLLS paths could not fit multi-group ordinal models at all. The two
// groups deliberately have different sample sizes so a missing n_b/N weight
// in the joint normal equations would break parity with the legacy
// unprofiled fit.
TEST_CASE("Joint threshold profiling handles two-group ordinal fits") {
  const Eigen::MatrixXd X1 =
      ordinal_test_block(20260610, 700, {0.85, 0.78, 0.71, 0.66}, -0.45, 0.55);
  const Eigen::MatrixXd X2 =
      ordinal_test_block(20260611, 450, {0.80, 0.74, 0.69, 0.62}, -0.30, 0.70);

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X1, X2});
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions build_opts;
  build_opts.n_groups = 2;
  auto fp = magmaan::parse::Parser::parse(k2GroupOrdinalCfa);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  const std::array<std::pair<magmaan::data::OrdinalEstimatorKind,
                             magmaan::estimate::OrdinalWeightKind>,
                   3>
      kinds = {{{magmaan::data::OrdinalEstimatorKind::ULS,
                 magmaan::estimate::OrdinalWeightKind::ULS},
                {magmaan::data::OrdinalEstimatorKind::DWLS,
                 magmaan::estimate::OrdinalWeightKind::DWLS},
                {magmaan::data::OrdinalEstimatorKind::WLS,
                 magmaan::estimate::OrdinalWeightKind::WLS}}};
  for (const auto& [estimator, weight_kind] : kinds) {
    CAPTURE(static_cast<int>(estimator));
    auto plan = magmaan::data::ordinal_weight_plan(
        magmaan::data::OrdinalWorkspacePurpose::FitOnly, estimator);
    auto workspace =
        magmaan::data::ordinal_workspace_from_integer_data({X1, X2}, plan);
    REQUIRE(workspace.has_value());
    auto x0 = magmaan::estimate::ordinal_start_values(
        *pt, *mr, workspace->moments, {});
    REQUIRE(x0.has_value());

    auto legacy = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, weight_kind, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto cached = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto snlls = magmaan::estimate::fit_ordinal_snlls(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);

    REQUIRE_MESSAGE(legacy.has_value(),
        "legacy failed: " << (legacy.has_value() ? "" : legacy.error().detail));
    REQUIRE_MESSAGE(cached.has_value(),
        "profiled bounded failed: "
            << (cached.has_value() ? "" : cached.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
        "profiled SNLLS failed: "
            << (snlls.has_value() ? "" : snlls.error().detail));

    CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK(snlls->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
    CHECK((snlls->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
  }
}

// Cross-group threshold invariance is a structured threshold design: shared
// labels replicate across groups and merge into joint gamma coordinates. The
// legacy unprofiled fit enforces the same equalities through the constrained
// solver and is the parity oracle.
TEST_CASE("Joint threshold profiling enforces cross-group threshold invariance") {
  const Eigen::MatrixXd X1 =
      ordinal_test_block(20260612, 650, {0.84, 0.77, 0.70, 0.64}, -0.40, 0.60);
  const Eigen::MatrixXd X2 =
      ordinal_test_block(20260613, 480, {0.81, 0.73, 0.68, 0.61}, -0.35, 0.65);

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X1, X2});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t11*t1 + t12*t2\n"
      "x2 | t21*t1 + t22*t2\n"
      "x3 | t31*t1 + t32*t2\n"
      "x4 | t41*t1 + t42*t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  magmaan::spec::BuildOptions build_opts;
  build_opts.n_groups = 2;
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 400;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  for (const auto estimator : {magmaan::data::OrdinalEstimatorKind::DWLS,
                               magmaan::data::OrdinalEstimatorKind::WLS}) {
    CAPTURE(static_cast<int>(estimator));
    const auto weight_kind =
        estimator == magmaan::data::OrdinalEstimatorKind::DWLS
            ? magmaan::estimate::OrdinalWeightKind::DWLS
            : magmaan::estimate::OrdinalWeightKind::WLS;
    auto plan = magmaan::data::ordinal_weight_plan(
        magmaan::data::OrdinalWorkspacePurpose::FitOnly, estimator);
    auto workspace =
        magmaan::data::ordinal_workspace_from_integer_data({X1, X2}, plan);
    REQUIRE(workspace.has_value());
    auto x0 = magmaan::estimate::ordinal_start_values(
        *pt, *mr, workspace->moments, {});
    REQUIRE(x0.has_value());

    auto legacy = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, weight_kind, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto cached = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto snlls = magmaan::estimate::fit_ordinal_snlls(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto full = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);

    REQUIRE_MESSAGE(legacy.has_value(),
        "legacy failed: " << (legacy.has_value() ? "" : legacy.error().detail));
    REQUIRE_MESSAGE(cached.has_value(),
        "profiled bounded failed: "
            << (cached.has_value() ? "" : cached.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
        "profiled SNLLS failed: "
            << (snlls.has_value() ? "" : snlls.error().detail));
    REQUIRE_MESSAGE(full.has_value(),
        "full-threshold SNLLS failed: "
            << (full.has_value() ? "" : full.error().detail));

    CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK(snlls->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK(full->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
    CHECK((snlls->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
    CHECK((full->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);

    // The invariance must be materialized (some equality group spans two
    // distinct threshold free parameters) and hold exactly in the profiled
    // solution.
    auto pt_prepared = *pt;
    auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
        pt_prepared, workspace->moments);
    REQUIRE(prep.has_value());
    REQUIRE(pt_prepared.eq_groups.size() ==
            static_cast<std::size_t>(pt_prepared.n_free()));
    std::map<std::int32_t, std::vector<std::int32_t>> threshold_groups;
    for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
      if (pt_prepared.op[row] != magmaan::parse::Op::Threshold) continue;
      const std::int32_t fr = pt_prepared.free[row];
      if (fr <= 0) continue;
      threshold_groups[pt_prepared.eq_groups[static_cast<std::size_t>(fr - 1)]]
          .push_back(fr);
    }
    bool found_cross_group_merge = false;
    for (auto& [group, members] : threshold_groups) {
      std::sort(members.begin(), members.end());
      members.erase(std::unique(members.begin(), members.end()),
                    members.end());
      if (members.size() < 2) continue;
      found_cross_group_merge = true;
      for (std::size_t k = 1; k < members.size(); ++k) {
        CHECK(snlls->theta(members[k] - 1) ==
              doctest::Approx(snlls->theta(members[0] - 1)).epsilon(1e-10));
        CHECK(cached->theta(members[k] - 1) ==
              doctest::Approx(cached->theta(members[0] - 1)).epsilon(1e-10));
      }
    }
    REQUIRE(found_cross_group_merge);
  }
}

// WLS couples the threshold and correlation residual blocks (W_tr != 0), so
// the profiled threshold rows stay active and depend on the correlation
// residual. General-H profiling must reproduce the bounded and full-threshold
// fits in exactly this regime, and contradictory threshold constraints must
// fail loudly.
TEST_CASE("Profiled WLS threshold maps couple thresholds and correlations") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260614, 560, {0.86, 0.79, 0.72, 0.68}, -0.55, 0.45);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + b*t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n"
      "a + b == 0\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 400;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, plan);
  REQUIRE(workspace.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(
      *pt, *mr, workspace->moments, {});
  REQUIRE(x0.has_value());

  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto full = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);

  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy WLS failed: "
          << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(snlls.has_value(),
      "profiled SNLLS WLS failed: "
          << (snlls.has_value() ? "" : snlls.error().detail));
  REQUIRE_MESSAGE(full.has_value(),
      "full-threshold SNLLS WLS failed: "
          << (full.has_value() ? "" : full.error().detail));
  CHECK(snlls->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK(full->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK((snlls->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
  CHECK((full->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);

  // Contradictory threshold constraints must produce a clear error, not a
  // silent fit.
  const char* bad_syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + b*t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n"
      "a + b == 0\n"
      "a + b == 0.5\n";
  auto bad_fp = magmaan::parse::Parser::parse(bad_syntax);
  REQUIRE(bad_fp.has_value());
  auto bad_pt = magmaan::spec::build(*bad_fp);
  REQUIRE(bad_pt.has_value());
  auto bad_mr = magmaan::model::build_matrix_rep(*bad_pt);
  REQUIRE(bad_mr.has_value());
  auto bad_x0 = magmaan::estimate::ordinal_start_values(
      *bad_pt, *bad_mr, workspace->moments, {});
  REQUIRE(bad_x0.has_value());
  auto bad = magmaan::estimate::fit_ordinal_snlls(
      *bad_pt, *bad_mr, workspace->moments, &workspace->gamma_cache, plan,
      *bad_x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_FALSE(bad.has_value());
  CHECK(bad.error().detail.find("infeasible") != std::string::npos);
}

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
  CHECK(observed->W_dwls[0].isApprox(complete->W_dwls[0], 1e-10));
  CHECK(observed->pairwise_gamma == "overlap");
}

TEST_CASE("ordinal_gamma_diag_data_influence matches case-weight finite differences") {
  Eigen::MatrixXd counts(3, 3);
  counts << 11, 7, 5,
             6, 14, 9,
             4, 10, 12;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  Eigen::MatrixXi Xcat = (X.cast<int>().array() - 1).matrix();

  auto IFG = magmaan::data::ordinal_gamma_diag_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(IFG.has_value());
  auto probe = gamma_diag_influence_probe_2var(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  CHECK((probe.Gamma - stats->NACOV[0]).cwiseAbs().maxCoeff() < 1e-8);

  double max_abs = 0.0;
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const Eigen::VectorXd fd = finite_diff_gamma_diag_case_influence(probe, i);
    max_abs = std::max(
        max_abs, (IFG->row(i).transpose() - fd).cwiseAbs().maxCoeff());
  }
  const double scale = 1.0 + IFG->cwiseAbs().maxCoeff();
  CHECK(max_abs < 5e-5 * scale);
}

TEST_CASE("robust_ordinal_ij rejects DWLS stats without complete integer data") {
  Eigen::MatrixXd counts(3, 3);
  counts << 11, 7, 5,
             6, 14, 9,
             4, 10, 12;
  auto stats = magmaan::data::ordinal_stats_from_integer_data(
      {ordinal_data_from_pair_counts(counts)});
  REQUIRE(stats.has_value());
  stats->int_data.clear();

  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::estimate::Estimates est;
  auto r = magmaan::estimate::robust_ordinal_ij(
      *pt, *mr, *stats, est, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().detail.find("int_data") != std::string::npos);
}

TEST_CASE("robust_ordinal_ij ULS matches observed-bread fixed-weight sandwich") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260619, 320, {0.82, 0.76, 0.70, 0.64}, -0.45, 0.55);
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
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1000;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::ULS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "ULS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto fixed = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Observed);
  auto ij = magmaan::estimate::robust_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE_MESSAGE(fixed.has_value(),
      "fixed robust failed: "
          << (fixed.has_value() ? "" : fixed.error().detail));
  REQUIRE_MESSAGE(ij.has_value(),
      "IJ robust failed: " << (ij.has_value() ? "" : ij.error().detail));
  CHECK(ij->df == fixed->df);
  CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
  CHECK(ij->vcov.isApprox(fixed->vcov, 1e-8));
  CHECK(ij->se.isApprox(fixed->se, 1e-8));
}

TEST_CASE("Observed ordinal stats expose overlap counts and nominal gamma variant") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(10, 3);
  X << 1,   1,   1,
       2,   1,   nan,
       1,   2,   1,
       2,   2,   2,
       1,   nan, 2,
       2,   1,   1,
       nan, 2,   2,
       1,   1,   2,
       2,   2,   1,
       nan, nan, 2;

  auto overlap = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap);
  auto nominal = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Nominal);
  REQUIRE(overlap.has_value());
  REQUIRE(nominal.has_value());
  CHECK(overlap->R[0].isApprox(nominal->R[0], 1e-12));
  CHECK(overlap->thresholds[0].isApprox(nominal->thresholds[0], 1e-12));
  REQUIRE(overlap->moment_n_obs.size() == 1);
  REQUIRE(overlap->moment_overlap_n_obs.size() == 1);
  const auto& nobs = overlap->moment_n_obs[0];
  const auto& ovlp = overlap->moment_overlap_n_obs[0];
  REQUIRE(nobs.size() == 6);
  CHECK(nobs[0] == 8);   // y1 threshold
  CHECK(nobs[1] == 8);   // y2 threshold
  CHECK(nobs[2] == 9);   // y3 threshold
  CHECK(nobs[3] == 7);   // y2,y1 polychoric
  CHECK(nobs[4] == 7);   // y3,y1 polychoric
  CHECK(nobs[5] == 7);   // y3,y2 polychoric
  CHECK(ovlp(0, 0) == 8);
  CHECK(ovlp(0, 1) == 7);
  CHECK(ovlp(3, 3) == 7);
  CHECK(ovlp(3, 5) == 6);

  const Eigen::MatrixXd& Gp = overlap->NACOV[0];
  const Eigen::MatrixXd& Gn = nominal->NACOV[0];
  CHECK(Gp(0, 0) / Gn(0, 0) == doctest::Approx(10.0 / 8.0).epsilon(1e-10));
  CHECK(Gp(3, 3) / Gn(3, 3) == doctest::Approx(10.0 / 7.0).epsilon(1e-10));
  CHECK((Gp - Gn).cwiseAbs().maxCoeff() > 1e-3);
}

TEST_CASE("Ordinal stage-2 weights reuse observed Gamma and expose DLS endpoints") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(24, 2);
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    X(r, 0) = (r % 2) + 1;
    X(r, 1) = ((r / 2) % 2) + 1;
  }
  X(3, 0) = nan;
  X(7, 1) = nan;
  X(11, 0) = nan;
  X(19, 1) = nan;

  auto stats_or = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap,
      /*full_wls_weight=*/true);
  REQUIRE(stats_or.has_value());
  const auto& stats = *stats_or;
  REQUIRE(stats.W_wls.size() == 1);
  REQUIRE(stats.W_wls[0].size() > 0);

  namespace mf = magmaan::estimate::frontier;
  auto uls = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Uls);
  auto dwls = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Dwls);
  auto wls = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Wls);
  auto nt = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Nt);
  auto dls0 = mf::ordinal_stage2_weight_blocks(
      stats, mf::OrdinalStage2Weight::Dls, {0.0});
  auto dls1 = mf::ordinal_stage2_weight_blocks(
      stats, mf::OrdinalStage2Weight::Dls, {1.0});

  REQUIRE(uls.has_value());
  REQUIRE(dwls.has_value());
  REQUIRE(wls.has_value());
  REQUIRE(nt.has_value());
  REQUIRE(dls0.has_value());
  REQUIRE(dls1.has_value());

  const Eigen::Index mdim = stats.NACOV[0].rows();
  CHECK((*uls)[0].isApprox(Eigen::MatrixXd::Identity(mdim, mdim), 1e-12));
  CHECK((*dwls)[0].isApprox(stats.W_dwls[0], 1e-12));
  CHECK((*wls)[0].isApprox(stats.W_wls[0], 1e-9));
  CHECK((*dls0)[0].isApprox((*nt)[0], 1e-9));
  CHECK((*dls1)[0].isApprox((*wls)[0], 1e-9));

  auto adapted = mf::ordinal_stats_with_stage2_weight(
      stats, mf::OrdinalStage2Weight::Dls, {0.35});
  REQUIRE(adapted.has_value());
  REQUIRE(adapted->W_wls.size() == 1);
  CHECK(adapted->NACOV[0].isApprox(stats.NACOV[0], 1e-12));
  CHECK(adapted->W_dwls[0].isApprox(stats.W_dwls[0], 1e-12));
  CHECK(adapted->W_wls[0].rows() == mdim);
  Eigen::LLT<Eigen::MatrixXd> llt(adapted->W_wls[0]);
  CHECK(llt.info() == Eigen::Success);
}

TEST_CASE("Observed ordinal stats count four-variable moment support overlaps") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(12, 4);
  X << 1,   1,   1,   1,
       1,   1,   2,   2,
       1,   2,   1,   2,
       1,   2,   2,   1,
       2,   1,   1,   2,
       2,   1,   2,   1,
       2,   2,   1,   1,
       2,   2,   2,   2,
       nan, 1,   1,   1,
       1,   nan, 1,   1,
       1,   1,   nan, 1,
       1,   1,   1,   nan;

  auto overlap = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap);
  auto nominal = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Nominal);
  REQUIRE(overlap.has_value());
  REQUIRE(nominal.has_value());
  REQUIRE(overlap->moment_n_obs.size() == 1);
  REQUIRE(overlap->moment_overlap_n_obs.size() == 1);
  const auto& nobs = overlap->moment_n_obs[0];
  const auto& ovlp = overlap->moment_overlap_n_obs[0];
  REQUIRE(nobs.size() == 10);
  CHECK(nobs[4] == 10);   // y2,y1 polychoric
  CHECK(nobs[9] == 10);   // y4,y3 polychoric
  CHECK(ovlp(4, 4) == 10);
  CHECK(ovlp(9, 9) == 10);
  CHECK(ovlp(4, 9) == 8); // union support y1,y2,y3,y4

  const Eigen::MatrixXd& Gp = overlap->NACOV[0];
  const Eigen::MatrixXd& Gn = nominal->NACOV[0];
  CHECK(Gp(4, 4) / Gn(4, 4) == doctest::Approx(12.0 / 10.0).epsilon(1e-10));
}

TEST_CASE("Ordinal pair joint ML estimates pair-local thresholds and rho") {
  Eigen::VectorXd thi(2);
  thi << -0.55, 0.85;
  Eigen::VectorXd thj(2);
  thj << -0.25, 0.65;
  const double rho = 0.42;
  const Eigen::MatrixXd counts = ordinal_expected_counts(thi, thj, rho, 50000.0);

  auto joint = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  REQUIRE(joint.has_value());
  CHECK(joint->thresholds_i.size() == 2);
  CHECK(joint->thresholds_j.size() == 2);
  CHECK(joint->thresholds_i(0) == doctest::Approx(thi(0)).epsilon(5e-4));
  CHECK(joint->thresholds_i(1) == doctest::Approx(thi(1)).epsilon(5e-4));
  CHECK(joint->thresholds_j(0) == doctest::Approx(thj(0)).epsilon(5e-4));
  CHECK(joint->thresholds_j(1) == doctest::Approx(thj(1)).epsilon(5e-4));
  CHECK(joint->rho == doctest::Approx(rho).epsilon(5e-4));
  CHECK(joint->thresholds_i(0) < joint->thresholds_i(1));
  CHECK(joint->thresholds_j(0) < joint->thresholds_j(1));
  CHECK(std::isfinite(joint->negloglik));
  CHECK(joint->adjusted_counts.isApprox(counts, 0.0));
}

TEST_CASE("Ordinal pair joint h-weighted estimator preserves ML limits") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd thi(2);
  thi << -0.55, 0.85;
  Eigen::VectorXd thj(2);
  thj << -0.25, 0.65;
  const Eigen::MatrixXd counts = ordinal_expected_counts(thi, thj, 0.42, 50000.0);

  auto ml = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  auto h_ml = magmaan::data::fit_ordinal_pair_joint_h_weighted(counts);
  auto hard_inf = magmaan::data::fit_ordinal_pair_joint_h_weighted(
      counts, magmaan::data::OrdinalPairJointHWeightedOptions{
                  .h_score = PolychoricHScoreOptions{
                      .kind = PolychoricHScoreKind::WmaHardCap,
                      .k = std::numeric_limits<double>::infinity()}});
  REQUIRE(ml.has_value());
  REQUIRE(h_ml.has_value());
  REQUIRE(hard_inf.has_value());
  CHECK(h_ml->thresholds_i.isApprox(ml->thresholds_i, 1e-12));
  CHECK(h_ml->thresholds_j.isApprox(ml->thresholds_j, 1e-12));
  CHECK(h_ml->rho == doctest::Approx(ml->rho));
  CHECK(hard_inf->thresholds_i.isApprox(ml->thresholds_i, 1e-12));
  CHECK(hard_inf->thresholds_j.isApprox(ml->thresholds_j, 1e-12));
  CHECK(hard_inf->rho == doctest::Approx(ml->rho));
  CHECK(h_ml->converged);
  CHECK(hard_inf->converged);
  CHECK(std::isfinite(h_ml->objective));
  CHECK(h_ml->expected_counts.rows() == counts.rows());
  CHECK(h_ml->expected_counts.cols() == counts.cols());
  CHECK(h_ml->residual_counts.isApprox(
      h_ml->adjusted_counts - h_ml->expected_counts, 1e-12));
  CHECK(h_ml->pearson_residuals.allFinite());
  CHECK(h_ml->weights.isApprox(Eigen::MatrixXd::Ones(3, 3), 1e-12));
}

TEST_CASE("Ordinal pair joint h-weighted estimator downweights contaminated cells") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd th(2);
  th << -0.55, 0.75;
  const Eigen::MatrixXd clean = ordinal_expected_counts(th, th, 0.55, 5000.0);
  Eigen::MatrixXd contaminated = clean;
  contaminated(0, 2) += 900.0;

  auto clean_ml = magmaan::data::fit_ordinal_pair_joint_ml(clean);
  auto contaminated_ml =
      magmaan::data::fit_ordinal_pair_joint_ml(contaminated);
  auto robust = magmaan::data::fit_ordinal_pair_joint_h_weighted(
      contaminated, magmaan::data::OrdinalPairJointHWeightedOptions{
                        .h_score = PolychoricHScoreOptions{
                            .kind = PolychoricHScoreKind::WmaHardCap,
                            .k = 1.15}});
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(robust.has_value());
  CHECK(robust->thresholds_i(0) < robust->thresholds_i(1));
  CHECK(robust->thresholds_j(0) < robust->thresholds_j(1));
  CHECK(robust->rho > contaminated_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(contaminated_ml->rho - clean_ml->rho));
  CHECK(robust->weights(0, 2) < 1.0);
  CHECK(robust->pearson_residuals(0, 2) > 0.0);
  CHECK(robust->expected_counts.sum() ==
        doctest::Approx(robust->adjusted_counts.sum()).epsilon(0.03));

  const PolychoricHScoreOptions h_options{
      .kind = PolychoricHScoreKind::WmaHardCap, .k = 1.15};
  const double objective = h_score_pair_objective(
      contaminated, robust->thresholds_i, robust->thresholds_j, robust->rho,
      h_options);
  CHECK(robust->objective == doctest::Approx(objective).epsilon(1e-12));
  CHECK(objective < h_score_pair_objective(
      contaminated, contaminated_ml->thresholds_i, contaminated_ml->thresholds_j,
      contaminated_ml->rho, h_options));

  for (Eigen::Index k = 0; k < robust->thresholds_i.size(); ++k) {
    Eigen::VectorXd plus = robust->thresholds_i;
    Eigen::VectorXd minus = robust->thresholds_i;
    plus(k) += 0.02;
    minus(k) -= 0.02;
    CHECK(objective <= h_score_pair_objective(
        contaminated, plus, robust->thresholds_j, robust->rho, h_options));
    CHECK(objective <= h_score_pair_objective(
        contaminated, minus, robust->thresholds_j, robust->rho, h_options));
  }
  for (Eigen::Index k = 0; k < robust->thresholds_j.size(); ++k) {
    Eigen::VectorXd plus = robust->thresholds_j;
    Eigen::VectorXd minus = robust->thresholds_j;
    plus(k) += 0.02;
    minus(k) -= 0.02;
    CHECK(objective <= h_score_pair_objective(
        contaminated, robust->thresholds_i, plus, robust->rho, h_options));
    CHECK(objective <= h_score_pair_objective(
        contaminated, robust->thresholds_i, minus, robust->rho, h_options));
  }
  CHECK(objective <= h_score_pair_objective(
      contaminated, robust->thresholds_i, robust->thresholds_j,
      robust->rho + 0.02, h_options));
  CHECK(objective <= h_score_pair_objective(
      contaminated, robust->thresholds_i, robust->thresholds_j,
      robust->rho - 0.02, h_options));
}

TEST_CASE("Ordinal pair joint DPD estimator preserves ML limit") {
  Eigen::VectorXd thi(2);
  thi << -0.55, 0.85;
  Eigen::VectorXd thj(2);
  thj << -0.25, 0.65;
  const Eigen::MatrixXd counts = ordinal_expected_counts(thi, thj, 0.42, 50000.0);

  auto ml = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  auto dpd0 = magmaan::data::fit_ordinal_pair_joint_dpd(
      counts, magmaan::data::OrdinalPairJointDpdOptions{.alpha = 0.0});
  REQUIRE(ml.has_value());
  REQUIRE(dpd0.has_value());
  CHECK(dpd0->thresholds_i.isApprox(ml->thresholds_i, 1e-12));
  CHECK(dpd0->thresholds_j.isApprox(ml->thresholds_j, 1e-12));
  CHECK(dpd0->rho == doctest::Approx(ml->rho));
  CHECK(dpd0->converged);
  CHECK(dpd0->weights.isApprox(Eigen::MatrixXd::Ones(3, 3), 1e-12));
  CHECK(dpd0->objective ==
        doctest::Approx(ml->negloglik / counts.sum()).epsilon(1e-12));

  auto bad = magmaan::data::fit_ordinal_pair_joint_dpd(
      counts, magmaan::data::OrdinalPairJointDpdOptions{.alpha = -0.1});
  REQUIRE_FALSE(bad.has_value());
  CHECK(bad.error().detail.find("invalid options") != std::string::npos);
}

TEST_CASE("Ordinal pair joint DPD estimator tempers low-probability cells") {
  Eigen::VectorXd th(2);
  th << -0.55, 0.75;
  const Eigen::MatrixXd clean = ordinal_expected_counts(th, th, 0.55, 5000.0);
  Eigen::MatrixXd contaminated = clean;
  contaminated(0, 2) += 900.0;

  auto clean_ml = magmaan::data::fit_ordinal_pair_joint_ml(clean);
  auto contaminated_ml =
      magmaan::data::fit_ordinal_pair_joint_ml(contaminated);
  auto dpd = magmaan::data::fit_ordinal_pair_joint_dpd(
      contaminated, magmaan::data::OrdinalPairJointDpdOptions{.alpha = 0.35});
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(dpd.has_value());
  CHECK(dpd->converged);
  CHECK(dpd->thresholds_i(0) < dpd->thresholds_i(1));
  CHECK(dpd->thresholds_j(0) < dpd->thresholds_j(1));
  CHECK(contaminated_ml->rho != doctest::Approx(clean_ml->rho));
  CHECK(dpd->rho != doctest::Approx(contaminated_ml->rho));
  CHECK(dpd->weights(0, 2) < dpd->weights(1, 1));
  CHECK(dpd->pearson_residuals(0, 2) > 0.0);

  const double objective = dpd_pair_objective(
      contaminated, dpd->thresholds_i, dpd->thresholds_j, dpd->rho, 0.35);
  CHECK(dpd->objective == doctest::Approx(objective).epsilon(1e-12));
  CHECK(objective < dpd_pair_objective(
      contaminated, contaminated_ml->thresholds_i, contaminated_ml->thresholds_j,
      contaminated_ml->rho, 0.35));
}

TEST_CASE("Ordinal pair h-weighted influence gives casewise sandwich rows") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::MatrixXd counts(3, 3);
  counts << 20.0, 10.0, 5.0,
             8.0, 25.0, 12.0,
             4.0, 14.0, 22.0;

  auto fit = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  REQUIRE(fit.has_value());
  auto influence = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, fit->thresholds_i, fit->thresholds_j, fit->rho);
  REQUIRE(influence.has_value());
  const Eigen::MatrixXd ordinary_scores = ordinal_pair_score_rows_from_counts(
      counts, fit->thresholds_i, fit->thresholds_j, fit->rho);
  Eigen::MatrixXd centered_scores = ordinary_scores;
  centered_scores.rowwise() -= centered_scores.colwise().mean();
  CHECK(influence->n_obs == counts.sum());
  CHECK(influence->estimating_functions.isApprox(centered_scores, 1e-12));
  CHECK(influence->score_gamma.isApprox(
      (centered_scores.transpose() * centered_scores) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->bread.rows() == ordinary_scores.cols());
  CHECK(influence->bread.cols() == ordinary_scores.cols());
  CHECK(influence->influence.rows() == ordinary_scores.rows());
  CHECK(influence->influence.cols() == ordinary_scores.cols());
  CHECK(influence->gamma.isApprox(
      (influence->influence.transpose() * influence->influence) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->gamma.isApprox(influence->gamma.transpose(), 1e-12));
  CHECK(influence->gamma.allFinite());

  auto hard_inf = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, fit->thresholds_i, fit->thresholds_j, fit->rho,
      magmaan::data::OrdinalPairHWeightedInfluenceOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = std::numeric_limits<double>::infinity()}});
  REQUIRE(hard_inf.has_value());
  CHECK(hard_inf->estimating_functions.isApprox(
      influence->estimating_functions, 1e-12));
  CHECK(hard_inf->gamma.isApprox(influence->gamma, 1e-12));
}

TEST_CASE("Ordinal pair h-weighted influence reports hard-cap kink convention") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd th(1);
  th << 0.0;
  Eigen::MatrixXd counts(2, 2);
  counts << 16.0, 8.0,
             8.0, 8.0;
  auto influence = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, th, th, 0.0,
      magmaan::data::OrdinalPairHWeightedInfluenceOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = 1.6}});
  REQUIRE(influence.has_value());
  CHECK(influence->ratios(0, 0) == doctest::Approx(1.6));
  CHECK(influence->h_values(0, 0) == doctest::Approx(1.6));
  CHECK(influence->dh_values(0, 0) == doctest::Approx(0.0));
  CHECK(influence->weights(0, 0) == doctest::Approx(1.0));
  CHECK(influence->ratios(0, 1) == doctest::Approx(0.8));
  CHECK(influence->dh_values(0, 1) == doctest::Approx(1.0));
}

TEST_CASE("Ordinal pair h-weighted influence downweights inflated cells and scales Gamma") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::MatrixXd counts(3, 3);
  counts << 594.0, 365.0, 858.0,
            453.0, 715.0, 372.0,
            311.0, 614.0, 718.0;
  auto robust = magmaan::data::fit_ordinal_pair_joint_h_weighted(
      counts, magmaan::data::OrdinalPairJointHWeightedOptions{
                  .h_score = PolychoricHScoreOptions{
                      .kind = PolychoricHScoreKind::WmaHardCap,
                      .k = 1.15}});
  REQUIRE(robust.has_value());
  auto influence = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, robust->thresholds_i, robust->thresholds_j, robust->rho,
      magmaan::data::OrdinalPairHWeightedInfluenceOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = 1.15}});
  REQUIRE(influence.has_value());
  CHECK(influence->weights(0, 2) < 1.0);
  CHECK(influence->estimating_functions.rows() == counts.sum());
  CHECK(influence->estimating_functions.cols() == 5);
  CHECK(influence->score_gamma.isApprox(
      (influence->estimating_functions.transpose() *
       influence->estimating_functions) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->gamma.isApprox(
      (influence->influence.transpose() * influence->influence) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->gamma.allFinite());
  CHECK(influence->gamma.diagonal().minCoeff() > 0.0);
}

TEST_CASE("Ordinal pair joint ML rejects empty marginal categories") {
  Eigen::MatrixXd counts(3, 2);
  counts << 3.0, 2.0,
            0.0, 0.0,
            4.0, 5.0;

  auto joint = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  REQUIRE_FALSE(joint.has_value());
  CHECK(joint.error().detail.find("marginal categories") != std::string::npos);
}

TEST_CASE("Pairwise ordinal stats wrap OrdinalStats and expose diagnostics") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto base = magmaan::data::ordinal_stats_from_integer_data({X});
  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(base.has_value());
  REQUIRE(pairwise.has_value());
  REQUIRE(pairwise->stats.R.size() == 1);
  CHECK(pairwise->stats.R[0].isApprox(base->R[0], 0.0));
  CHECK(pairwise->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(pairwise->stats.NACOV[0].isApprox(base->NACOV[0], 0.0));
  CHECK(pairwise->stats.W_dwls[0].isApprox(base->W_dwls[0], 0.0));
  CHECK(pairwise->stats.W_wls[0].isApprox(base->W_wls[0], 0.0));
  CHECK(pairwise->stats.n_obs == base->n_obs);
  CHECK(pairwise->stats.n_levels == base->n_levels);
  CHECK(pairwise->stats.threshold_ov == base->threshold_ov);
  CHECK(pairwise->stats.threshold_level == base->threshold_level);

  REQUIRE(pairwise->block_diagnostics.size() == 1);
  const auto& bd = pairwise->block_diagnostics[0];
  REQUIRE(bd.pair_diagnostics.size() == 3);
  CHECK(bd.moment_influence.rows() == 24);
  CHECK(bd.moment_influence.cols() == pairwise->stats.NACOV[0].rows());
  CHECK(bd.moment_influence.allFinite());
  CHECK(bd.gamma.isApprox(pairwise->stats.NACOV[0], 0.0));
  const Eigen::MatrixXd gamma_from_if =
      (bd.moment_influence.transpose() * bd.moment_influence) / 24.0;
  CHECK(gamma_from_if.isApprox(pairwise->stats.NACOV[0], 1e-10));
  CHECK(bd.pair_diagnostics[0].label.i == 1);
  CHECK(bd.pair_diagnostics[0].label.j == 0);
  CHECK(bd.pair_diagnostics[1].label.i == 2);
  CHECK(bd.pair_diagnostics[1].label.j == 0);
  CHECK(bd.pair_diagnostics[2].label.i == 2);
  CHECK(bd.pair_diagnostics[2].label.j == 1);
  for (const auto& pd : bd.pair_diagnostics) {
    CHECK(pd.label.block == 0);
    CHECK(pd.label.n_levels_i == 3);
    CHECK(pd.label.n_levels_j == 3);
    CHECK(std::isfinite(pd.rho));
    CHECK(std::isfinite(pd.negloglik));
    CHECK(pd.n_obs == 24);
    CHECK(pd.n_missing == 0);
    CHECK_FALSE(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(0.0));
    CHECK_FALSE(pd.shrinkage_applied);
    CHECK(pd.shrinkage_intensity == doctest::Approx(0.0));
    CHECK(pd.counts.rows() == 3);
    CHECK(pd.counts.cols() == 3);
    CHECK(pd.adjusted_counts.rows() == 3);
    CHECK(pd.adjusted_counts.cols() == 3);
    CHECK(pd.expected_counts.rows() == 3);
    CHECK(pd.expected_counts.cols() == 3);
    CHECK(pd.residual_counts.rows() == 3);
    CHECK(pd.residual_counts.cols() == 3);
    CHECK(pd.expected_counts.allFinite());
    CHECK(pd.residual_counts.allFinite());
    CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
    CHECK(pd.residual_counts.sum() == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(pd.residual_counts.isApprox(pd.adjusted_counts - pd.expected_counts, 1e-12));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(pairwise->stats.R[0]);
  REQUIRE(es.info() == Eigen::Success);
  CHECK(bd.min_eigen_r == doctest::Approx(es.eigenvalues().minCoeff()));
}

TEST_CASE("Pairwise ordinal stats diagnostics report lavaan 2x2 adjustment") {
  Eigen::MatrixXd X(15, 2);
  Eigen::Index r = 0;
  for (int k = 0; k < 4; ++k) X.row(r++) << 1, 2;
  for (int k = 0; k < 5; ++k) X.row(r++) << 2, 1;
  for (int k = 0; k < 6; ++k) X.row(r++) << 2, 2;

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());
  REQUIRE(pairwise->block_diagnostics.size() == 1);
  REQUIRE(pairwise->block_diagnostics[0].pair_diagnostics.size() == 1);
  CHECK(pairwise->block_diagnostics[0].gamma.isApprox(pairwise->stats.NACOV[0], 0.0));
  const auto& pd = pairwise->block_diagnostics[0].pair_diagnostics[0];
  CHECK(pd.counts(0, 0) == doctest::Approx(0.0));
  CHECK(pd.counts(0, 1) == doctest::Approx(5.0));
  CHECK(pd.counts(1, 0) == doctest::Approx(4.0));
  CHECK(pd.counts(1, 1) == doctest::Approx(6.0));
  CHECK(pd.adjusted_counts(0, 0) == doctest::Approx(0.5));
  CHECK(pd.adjusted_counts(0, 1) == doctest::Approx(4.5));
  CHECK(pd.adjusted_counts(1, 0) == doctest::Approx(3.5));
  CHECK(pd.adjusted_counts(1, 1) == doctest::Approx(6.5));
  CHECK(pd.expected_counts.rows() == 2);
  CHECK(pd.expected_counts.cols() == 2);
  CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
  CHECK(pd.residual_counts.isApprox(pd.adjusted_counts - pd.expected_counts, 1e-12));
  CHECK(pd.n_obs == 15);
  CHECK(pd.n_missing == 0);
  CHECK_FALSE(pd.ridge_applied);
  CHECK_FALSE(pd.shrinkage_applied);
  CHECK(std::isfinite(pd.rho));
  CHECK(std::isfinite(pairwise->block_diagnostics[0].min_eigen_r));
}

TEST_CASE("Pairwise ordinal h-weighted stats preserve ML limit") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto base = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  auto robust =
      magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data({X});
  REQUIRE(base.has_value());
  REQUIRE(robust.has_value());
  CHECK(robust->stats.thresholds[0].isApprox(base->stats.thresholds[0], 0.0));
  CHECK(robust->stats.R[0].isApprox(base->stats.R[0], 1e-7));
  CHECK(robust->stats.NACOV[0].rows() == base->stats.NACOV[0].rows());
  CHECK(robust->stats.NACOV[0].cols() == base->stats.NACOV[0].cols());
  CHECK(robust->stats.NACOV[0].allFinite());
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(robust->block_diagnostics[0].gamma.isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  for (const auto& pd : robust->block_diagnostics[0].pair_diagnostics) {
    CHECK_FALSE(pd.h_weighted);
    CHECK(pd.converged);
    CHECK(pd.weights.rows() == pd.counts.rows());
    CHECK(pd.weights.cols() == pd.counts.cols());
  }

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto fit = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, robust->stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(std::isfinite(fit->fmin));
}

TEST_CASE("Pairwise ordinal h-weighted stats jointly estimate shared thresholds and Gamma") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::MatrixXd counts(3, 3);
  counts << 594.0, 365.0, 858.0,
            453.0, 715.0, 372.0,
            311.0, 614.0, 718.0;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto base = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(base.has_value());

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions options;
  options.rho.h_score = PolychoricHScoreOptions{
      .kind = PolychoricHScoreKind::WmaHardCap,
      .k = 1.15};
  auto robust = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, options);
  REQUIRE(robust.has_value());

  CHECK_FALSE(robust->stats.thresholds[0].isApprox(
      base->stats.thresholds[0], 1e-6));
  CHECK(std::abs(robust->stats.R[0](1, 0) - base->stats.R[0](1, 0)) > 1e-4);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK_FALSE(robust->stats.NACOV[0].isApprox(base->stats.NACOV[0], 1e-6));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(robust->block_diagnostics[0].pair_diagnostics.size() == 1);
  const auto& pd = robust->block_diagnostics[0].pair_diagnostics[0];
  CHECK(pd.h_weighted);
  CHECK(pd.weights.rows() == counts.cols());
  CHECK(pd.weights.cols() == counts.rows());
  CHECK(pd.weights(2, 0) < 1.0);
  CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
  CHECK(pd.expected_counts.allFinite());
}

TEST_CASE("Pairwise ordinal DPD stats jointly estimate shared thresholds") {
  Eigen::MatrixXd counts(3, 3);
  counts << 594.0, 365.0, 858.0,
            453.0, 715.0, 372.0,
            311.0, 614.0, 718.0;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto base = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(base.has_value());

  magmaan::data::PairwiseOrdinalDpdStatsOptions options;
  options.alpha = 0.35;
  auto robust = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      {X}, options);
  REQUIRE(robust.has_value());
  options.alpha = 0.0;
  auto ml_limit = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      {X}, options);
  REQUIRE(ml_limit.has_value());
  CHECK(ml_limit->stats.thresholds[0].isApprox(base->stats.thresholds[0], 0.0));
  CHECK(ml_limit->stats.R[0].isApprox(base->stats.R[0], 0.0));
  CHECK_FALSE(robust->stats.thresholds[0].isApprox(
      base->stats.thresholds[0], 1e-6));
  CHECK(std::abs(robust->stats.R[0](1, 0) - base->stats.R[0](1, 0)) > 1e-4);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(robust->block_diagnostics[0].pair_diagnostics.size() == 1);
  const auto& pd = robust->block_diagnostics[0].pair_diagnostics[0];
  CHECK_FALSE(pd.h_weighted);
  CHECK(pd.weights.minCoeff() > 0.0);
  CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
}

TEST_CASE("Pairwise ordinal Huber residual stats preserve ML limit and downweight contamination") {
  Eigen::MatrixXd clean_counts(3, 3);
  clean_counts << 250.0, 120.0, 35.0,
                  110.0, 300.0, 120.0,
                  25.0, 130.0, 260.0;
  Eigen::MatrixXd contaminated_counts = clean_counts;
  contaminated_counts(0, 2) += 900.0;
  const Eigen::MatrixXd clean = ordinal_data_from_pair_counts(clean_counts);
  const Eigen::MatrixXd contaminated =
      ordinal_data_from_pair_counts(contaminated_counts);

  auto clean_ml = magmaan::data::pairwise_ordinal_stats_from_integer_data({clean});
  auto contaminated_ml =
      magmaan::data::pairwise_ordinal_stats_from_integer_data({contaminated});
  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions huber_ml_options;
  huber_ml_options.clip.kind = magmaan::data::HuberResidualClipKind::None;
  auto huber_ml = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      {contaminated}, huber_ml_options);
  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions huber_options;
  huber_options.clip.kind = magmaan::data::HuberResidualClipKind::PseudoHuber;
  huber_options.clip.k = 1.25;
  auto huber = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      {contaminated}, huber_options);
  magmaan::data::PairwiseOrdinalHWeightedStatsOptions h_options;
  h_options.rho.h_score = magmaan::data::PolychoricHScoreOptions{
      .kind = magmaan::data::PolychoricHScoreKind::WmaHardCap,
      .k = 1.15};
  auto h_weighted =
      magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
          {contaminated}, h_options);
  magmaan::data::PairwiseOrdinalDpdStatsOptions dpd_options;
  dpd_options.alpha = 0.35;
  auto dpd = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      {contaminated}, dpd_options);
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(huber_ml.has_value());
  REQUIRE(huber.has_value());
  REQUIRE(h_weighted.has_value());
  REQUIRE(dpd.has_value());

  CHECK(huber_ml->stats.thresholds[0].isApprox(
      contaminated_ml->stats.thresholds[0], 0.0));
  CHECK(huber_ml->stats.R[0].isApprox(contaminated_ml->stats.R[0], 0.0));
  CHECK(huber_ml->stats.NACOV[0].isApprox(contaminated_ml->stats.NACOV[0], 0.0));

  const double contaminated_r = contaminated_ml->stats.R[0](1, 0);
  const double huber_r = huber->stats.R[0](1, 0);
  CHECK(std::isfinite(huber_r));
  CHECK(std::abs(huber_r - contaminated_r) > 1e-4);
  CHECK(std::abs(h_weighted->stats.R[0](1, 0) - contaminated_r) > 1e-4);
  CHECK(std::abs(dpd->stats.R[0](1, 0) - contaminated_r) > 1e-4);
  CHECK(huber->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(h_weighted->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(dpd->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(huber->block_diagnostics[0].pair_diagnostics.size() == 1);
  const auto& pd = huber->block_diagnostics[0].pair_diagnostics[0];
  CHECK(pd.weights.minCoeff() < 1.0);
  CHECK(pd.pearson_residuals.cwiseAbs().maxCoeff() > 1.25);
  CHECK(huber->stats.NACOV[0].isApprox(
      (huber->block_diagnostics[0].moment_influence.transpose() *
       huber->block_diagnostics[0].moment_influence) /
          static_cast<double>(huber->stats.n_obs[0]),
      1e-12));
}

TEST_CASE("Pairwise ordinal Huber residual repair keeps correlation influence stable") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions raw_options;
  raw_options.clip = magmaan::data::HuberResidualClipOptions{
      .kind = magmaan::data::HuberResidualClipKind::HardHuber,
      .k = 1.20};
  auto raw = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      {X}, raw_options);
  REQUIRE(raw.has_value());
  const auto& raw_bd = raw->block_diagnostics[0];
  REQUIRE(raw_bd.min_eigen_r < 0.95);

  auto ridge_options = raw_options;
  ridge_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Ridge;
  ridge_options.correlation_repair.min_eigenvalue = 0.95;
  auto ridged =
      magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
          {X}, ridge_options);
  REQUIRE(ridged.has_value());
  const auto& ridged_bd = ridged->block_diagnostics[0];
  CHECK(ridged_bd.raw_min_eigen_r == doctest::Approx(raw_bd.raw_min_eigen_r));
  CHECK(ridged_bd.min_eigen_r == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(ridged_bd.r_repair_applied);
  REQUIRE(ridged_bd.r_ridge > 0.0);

  const Eigen::Index nth = ridged->stats.thresholds[0].size();
  const Eigen::Index ncorr = ridged->stats.R[0].cols() *
      (ridged->stats.R[0].cols() - 1) / 2;
  const double corr_scale = 1.0 / (1.0 + ridged_bd.r_ridge);
  CHECK(ridged_bd.moment_influence.leftCols(nth).isApprox(
      raw_bd.moment_influence.leftCols(nth), 1e-12));
  CHECK(ridged_bd.moment_influence.rightCols(ncorr).isApprox(
      corr_scale * raw_bd.moment_influence.rightCols(ncorr), 1e-12));
  CHECK(ridged->stats.NACOV[0].isApprox(
      (ridged_bd.moment_influence.transpose() *
       ridged_bd.moment_influence) /
          static_cast<double>(ridged->stats.n_obs[0]),
      1e-12));
  CHECK(ridged->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  for (const auto& pd : ridged_bd.pair_diagnostics) {
    CHECK(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(ridged_bd.r_ridge));
  }
}

TEST_CASE("Pairwise ordinal h-weighted stats repair low-eigen robust R on request") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions raw_options;
  raw_options.rho.h_score = magmaan::data::PolychoricHScoreOptions{
      .kind = magmaan::data::PolychoricHScoreKind::WmaHardCap,
      .k = 1.30};
  auto raw = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, raw_options);
  REQUIRE(raw.has_value());
  const auto& raw_bd = raw->block_diagnostics[0];
  CHECK(raw_bd.raw_min_eigen_r == doctest::Approx(raw_bd.min_eigen_r));
  CHECK_FALSE(raw_bd.r_repair_applied);
  CHECK(raw_bd.r_ridge == doctest::Approx(0.0));
  CHECK(raw_bd.r_shrinkage_intensity == doctest::Approx(0.0));
  REQUIRE(raw_bd.min_eigen_r < 0.95);

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions err_options;
  err_options.rho.h_score = raw_options.rho.h_score;
  err_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Error;
  err_options.correlation_repair.min_eigenvalue = 0.95;
  auto err = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, err_options);
  REQUIRE_FALSE(err.has_value());
  CHECK(err.error().detail.find("minimum eigenvalue") != std::string::npos);

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions shrink_options;
  shrink_options.rho.h_score = raw_options.rho.h_score;
  shrink_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Shrinkage;
  shrink_options.correlation_repair.min_eigenvalue = 0.95;
  auto shrunk = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, shrink_options);
  REQUIRE(shrunk.has_value());
  const auto& shrunk_bd = shrunk->block_diagnostics[0];
  CHECK(shrunk_bd.raw_min_eigen_r == doctest::Approx(raw_bd.raw_min_eigen_r));
  CHECK(shrunk_bd.min_eigen_r == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(shrunk_bd.r_repair_applied);
  CHECK(shrunk_bd.r_ridge == doctest::Approx(0.0));
  REQUIRE(shrunk_bd.r_shrinkage_intensity > 0.0);
  CHECK(shrunk_bd.r_shrinkage_intensity < 1.0);
  CHECK(shrunk->stats.R[0].diagonal().isOnes(1e-12));
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> shrunk_es(shrunk->stats.R[0]);
  REQUIRE(shrunk_es.info() == Eigen::Success);
  CHECK(shrunk_es.eigenvalues().minCoeff() == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(shrunk->stats.NACOV[0].isApprox(
      (shrunk_bd.moment_influence.transpose() *
       shrunk_bd.moment_influence) /
          static_cast<double>(shrunk->stats.n_obs[0]),
      1e-12));
  for (const auto& pd : shrunk_bd.pair_diagnostics) {
    CHECK_FALSE(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(0.0));
    CHECK(pd.shrinkage_applied);
    CHECK(pd.shrinkage_intensity ==
          doctest::Approx(shrunk_bd.r_shrinkage_intensity));
  }

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions ridge_options;
  ridge_options.rho.h_score = raw_options.rho.h_score;
  ridge_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Ridge;
  ridge_options.correlation_repair.min_eigenvalue = 0.95;
  auto ridged = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, ridge_options);
  REQUIRE(ridged.has_value());
  const auto& ridged_bd = ridged->block_diagnostics[0];
  CHECK(ridged_bd.raw_min_eigen_r == doctest::Approx(raw_bd.raw_min_eigen_r));
  CHECK(ridged_bd.min_eigen_r == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(ridged_bd.r_repair_applied);
  CHECK(ridged_bd.r_ridge > 0.0);
  CHECK(ridged_bd.r_shrinkage_intensity == doctest::Approx(0.0));
  CHECK(ridged->stats.R[0].isApprox(shrunk->stats.R[0], 1e-12));
  for (const auto& pd : ridged_bd.pair_diagnostics) {
    CHECK(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(ridged_bd.r_ridge));
    CHECK_FALSE(pd.shrinkage_applied);
    CHECK(pd.shrinkage_intensity == doctest::Approx(0.0));
  }
}

TEST_CASE("Pairwise ordinal composite objective uses pair diagnostics and explicit scaling") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());

  auto obj = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, pairwise->stats.R);
  REQUIRE(obj.has_value());
  REQUIRE(obj->blocks.size() == 1);
  REQUIRE(obj->blocks[0].pairs.size() == 3);

  double expected_nll = 0.0;
  for (const auto& pd : pairwise->block_diagnostics[0].pair_diagnostics) {
    expected_nll += pd.negloglik;
  }
  CHECK(obj->negloglik == doctest::Approx(expected_nll));
  CHECK(obj->weighted_negloglik == doctest::Approx(expected_nll));
  CHECK(obj->scaling_denominator == doctest::Approx(72.0));
  CHECK(obj->objective == doctest::Approx(expected_nll / 72.0));
  CHECK_FALSE(obj->reports_chisq);
  CHECK(obj->df == -1);
  CHECK_FALSE(obj->blocks[0].reports_chisq);
  CHECK(obj->blocks[0].df == -1);

  for (const auto& pair : obj->blocks[0].pairs) {
    CHECK(pair.n_obs == 24);
    CHECK(pair.n_missing == 0);
    CHECK(pair.scaling_weight == doctest::Approx(24.0));
    CHECK(pair.expected_counts.sum() == doctest::Approx(pair.counts.sum()));
    CHECK(pair.residual_counts.isApprox(pair.counts - pair.expected_counts, 1e-12));
  }

  auto sum_obj = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, pairwise->stats.R,
      magmaan::estimate::frontier::PairwiseOrdinalCompositeOptions{
          .weighting = magmaan::estimate::frontier::PairwiseCompositeWeighting::ObservedPairCount,
          .scaling = magmaan::estimate::frontier::PairwiseCompositeScaling::SumNegLogLik});
  REQUIRE(sum_obj.has_value());
  CHECK(sum_obj->objective == doctest::Approx(expected_nll));
}

TEST_CASE("Pairwise ordinal composite objective validates implied pair mapping") {
  Eigen::MatrixXd X(15, 2);
  Eigen::Index r = 0;
  for (int k = 0; k < 4; ++k) X.row(r++) << 1, 1;
  for (int k = 0; k < 3; ++k) X.row(r++) << 1, 2;
  for (int k = 0; k < 2; ++k) X.row(r++) << 2, 1;
  for (int k = 0; k < 6; ++k) X.row(r++) << 2, 2;

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());

  auto implied_r = pairwise->stats.R;
  implied_r[0](1, 0) = implied_r[0](0, 1) = 0.25;
  auto obj = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, implied_r);
  REQUIRE(obj.has_value());
  REQUIRE(obj->blocks[0].pairs.size() == 1);
  CHECK(obj->blocks[0].pairs[0].rho == doctest::Approx(0.25));

  implied_r[0](1, 0) = implied_r[0](0, 1) = 1.0;
  auto bad_rho = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, implied_r);
  REQUIRE_FALSE(bad_rho.has_value());
  CHECK(bad_rho.error().detail.find("inside (-1, 1)") != std::string::npos);

  auto bad_thresholds = pairwise->stats.thresholds;
  bad_thresholds[0](0) = std::numeric_limits<double>::quiet_NaN();
  auto bad_th = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, bad_thresholds, pairwise->stats.R);
  REQUIRE_FALSE(bad_th.has_value());
  CHECK(bad_th.error().detail.find("non-finite") != std::string::npos);
}

TEST_CASE("Pairwise ordinal joint composite objective fits complete pair-local margins") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());
  auto joint = magmaan::estimate::frontier::pairwise_ordinal_joint_composite_objective(
      *pairwise);
  REQUIRE(joint.has_value());
  REQUIRE(joint->blocks.size() == 1);
  REQUIRE(joint->blocks[0].pairs.size() == 3);
  CHECK_FALSE(joint->reports_chisq);
  CHECK(joint->df == -1);

  double expected_nll = 0.0;
  for (std::size_t k = 0; k < joint->blocks[0].pairs.size(); ++k) {
    const auto& pd = pairwise->block_diagnostics[0].pair_diagnostics[k];
    const auto direct = magmaan::data::fit_ordinal_pair_joint_ml(pd.counts);
    REQUIRE(direct.has_value());
    const auto& pair = joint->blocks[0].pairs[k];
    expected_nll += direct->negloglik;
    CHECK(pair.label.i == pd.label.i);
    CHECK(pair.label.j == pd.label.j);
    CHECK(pair.thresholds_i.isApprox(direct->thresholds_i, 1e-12));
    CHECK(pair.thresholds_j.isApprox(direct->thresholds_j, 1e-12));
    CHECK(pair.rho == doctest::Approx(direct->rho));
    CHECK(pair.negloglik == doctest::Approx(direct->negloglik));
    CHECK(pair.counts.isApprox(pd.counts, 0.0));
    CHECK(pair.adjusted_counts.isApprox(direct->adjusted_counts, 0.0));
    CHECK(pair.expected_counts.sum() == doctest::Approx(pair.adjusted_counts.sum()));
    CHECK(pair.residual_counts.isApprox(pair.adjusted_counts - pair.expected_counts, 1e-12));
    CHECK(pair.scaling_weight == doctest::Approx(24.0));
    const Eigen::Index n_score_cols =
        pair.thresholds_i.size() + pair.thresholds_j.size() + 1;
    CHECK(pair.score_contributions.rows() == pair.n_obs);
    CHECK(pair.score_contributions.cols() == n_score_cols);
    CHECK(pair.score_contributions.allFinite());
    CHECK(pair.score_gamma.rows() == n_score_cols);
    CHECK(pair.score_gamma.cols() == n_score_cols);
    CHECK(pair.score_gamma.isApprox(
        (pair.score_contributions.transpose() * pair.score_contributions) /
            static_cast<double>(pair.n_obs),
        1e-12));
  }
  CHECK(joint->negloglik == doctest::Approx(expected_nll));
  CHECK(joint->weighted_negloglik == doctest::Approx(expected_nll));
  CHECK(joint->scaling_denominator == doctest::Approx(72.0));
  CHECK(joint->objective == doctest::Approx(expected_nll / 72.0));
}

TEST_CASE("Pairwise ordinal joint composite objective preserves lavaan 2x2 adjustment choice") {
  Eigen::MatrixXd X(15, 2);
  Eigen::Index r = 0;
  for (int k = 0; k < 4; ++k) X.row(r++) << 1, 2;
  for (int k = 0; k < 5; ++k) X.row(r++) << 2, 1;
  for (int k = 0; k < 6; ++k) X.row(r++) << 2, 2;

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());
  auto adjusted = magmaan::estimate::frontier::pairwise_ordinal_joint_composite_objective(
      *pairwise);
  auto raw = magmaan::estimate::frontier::pairwise_ordinal_joint_composite_objective(
      *pairwise,
      magmaan::estimate::frontier::PairwiseOrdinalCompositeOptions{
          .lavaan_adjust_2x2 = false});
  REQUIRE(adjusted.has_value());
  REQUIRE(raw.has_value());
  REQUIRE(adjusted->blocks[0].pairs.size() == 1);
  REQUIRE(raw->blocks[0].pairs.size() == 1);

  const auto& adj_pair = adjusted->blocks[0].pairs[0];
  const auto& raw_pair = raw->blocks[0].pairs[0];
  CHECK(adj_pair.counts.isApprox(raw_pair.counts, 0.0));
  CHECK(adj_pair.adjusted_counts(0, 0) == doctest::Approx(0.5));
  CHECK(raw_pair.adjusted_counts(0, 0) == doctest::Approx(0.0));
  CHECK(adj_pair.negloglik != doctest::Approx(raw_pair.negloglik));
}

TEST_CASE("Pairwise ordinal observed joint composite objective preserves pairwise missingness") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(8, 3);
  X << 1.0, 1.0, 1.0,
       2.0, 2.0, 1.0,
       3.0, 1.0, 2.0,
       nan, 2.0, 2.0,
       1.0, nan, 1.0,
       2.0, 1.0, nan,
       3.0, 2.0, 1.0,
       nan, nan, 2.0;
  std::vector<std::vector<std::int32_t>> levels{{3, 2, 2}};

  auto observed =
      magmaan::estimate::frontier::pairwise_ordinal_observed_joint_composite_objective(
          {X}, levels);
  REQUIRE(observed.has_value());
  REQUIRE(observed->blocks.size() == 1);
  REQUIRE(observed->blocks[0].pairs.size() == 3);
  CHECK(observed->blocks[0].n_obs == 8);
  CHECK_FALSE(observed->reports_chisq);
  CHECK(observed->df == -1);

  double expected_nll = 0.0;
  std::int64_t expected_pair_n = 0;
  for (const auto& pair : observed->blocks[0].pairs) {
    auto direct = magmaan::data::fit_ordinal_pair_observed_joint_ml(
        X.col(pair.label.i), X.col(pair.label.j),
        pair.label.n_levels_i, pair.label.n_levels_j);
    REQUIRE(direct.has_value());
    expected_nll += direct->fit.negloglik;
    expected_pair_n += direct->n_obs;
    CHECK(pair.thresholds_i.isApprox(direct->fit.thresholds_i, 1e-12));
    CHECK(pair.thresholds_j.isApprox(direct->fit.thresholds_j, 1e-12));
    CHECK(pair.rho == doctest::Approx(direct->fit.rho));
    CHECK(pair.negloglik == doctest::Approx(direct->fit.negloglik));
    CHECK(pair.n_obs == direct->n_obs);
    CHECK(pair.n_missing == direct->n_missing);
    CHECK(pair.n_obs == 5);
    CHECK(pair.n_missing == 3);
    CHECK(pair.counts.isApprox(direct->counts, 0.0));
    CHECK(pair.adjusted_counts.isApprox(direct->fit.adjusted_counts, 0.0));
    CHECK(pair.expected_counts.sum() ==
          doctest::Approx(pair.adjusted_counts.sum()).epsilon(0.03));
    CHECK(pair.residual_counts.isApprox(pair.adjusted_counts - pair.expected_counts, 1e-12));
    const Eigen::Index n_score_cols =
        pair.thresholds_i.size() + pair.thresholds_j.size() + 1;
    CHECK(pair.score_contributions.rows() == pair.n_obs);
    CHECK(pair.score_contributions.cols() == n_score_cols);
    CHECK(pair.score_contributions.allFinite());
    CHECK(pair.score_gamma.rows() == n_score_cols);
    CHECK(pair.score_gamma.cols() == n_score_cols);
    CHECK(pair.score_gamma.isApprox(
        (pair.score_contributions.transpose() * pair.score_contributions) /
            static_cast<double>(pair.n_obs),
        1e-12));
  }
  CHECK(observed->negloglik == doctest::Approx(expected_nll));
  CHECK(observed->weighted_negloglik == doctest::Approx(expected_nll));
  CHECK(observed->scaling_denominator == doctest::Approx(static_cast<double>(expected_pair_n)));
  CHECK(observed->objective == doctest::Approx(expected_nll / static_cast<double>(expected_pair_n)));
}

TEST_CASE("Pairwise ordinal observed joint composite objective rejects invalid observed pairs") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::vector<std::int32_t>> levels{{2, 2}};

  Eigen::MatrixXd all_missing(3, 2);
  all_missing << nan, 1.0,
                 nan, 2.0,
                 nan, nan;
  auto missing =
      magmaan::estimate::frontier::pairwise_ordinal_observed_joint_composite_objective(
          {all_missing}, levels);
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error().detail.find("no observed pairs") != std::string::npos);

  Eigen::MatrixXd empty_margin(3, 2);
  empty_margin << 1.0, 1.0,
                  1.0, 2.0,
                  nan, 1.0;
  auto empty =
      magmaan::estimate::frontier::pairwise_ordinal_observed_joint_composite_objective(
          {empty_margin}, levels);
  REQUIRE_FALSE(empty.has_value());
  CHECK(empty.error().detail.find("marginal categories") != std::string::npos);
}

TEST_CASE("Pairwise ordinal composite fit and Godambe handle observed-pair missingness") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::mt19937 rng(20260618);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(90, 3);
  const double loading[3] = {0.85, 0.75, 0.65};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.35) + (y > 0.55);
    }
  }
  X(3, 0) = nan;
  X(14, 1) = nan;
  X(29, 2) = nan;
  X(51, 0) = nan;

  std::vector<std::vector<std::int32_t>> levels{{3, 3, 3}};
  auto data = magmaan::estimate::frontier::pairwise_ordinal_observed_data(
      {X}, levels);
  REQUIRE(data.has_value());
  REQUIRE(data->saturated.blocks.size() == 1);
  REQUIRE(data->saturated.blocks[0].pairs.size() == 3);
  bool saw_missing = false;
  for (const auto& pair : data->saturated.blocks[0].pairs) {
    saw_missing = saw_missing || pair.n_missing > 0;
  }
  CHECK(saw_missing);

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 80;
  auto fit = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      *pt, *mr, *data, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "pairwise composite fit failed: "
          << (fit.has_value() ? "" : fit.error().detail));
  CHECK(std::isfinite(fit->objective.negloglik));
  CHECK(fit->objective.negloglik <= data->saturated.negloglik + 100.0);

  auto god = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      *pt, *mr, *data, fit->estimates, 2e-5);
  REQUIRE_MESSAGE(god.has_value(),
      "pairwise composite Godambe failed: "
          << (god.has_value() ? "" : god.error().detail));
  CHECK(god->vcov.rows() == fit->estimates.theta.size());
  CHECK(god->vcov.cols() == fit->estimates.theta.size());
  CHECK(god->se.size() == fit->estimates.theta.size());
  CHECK(god->casewise_scores.rows() == X.rows());
  CHECK(god->casewise_scores.cols() == fit->estimates.theta.size());
  CHECK(god->vcov.allFinite());
  CHECK(god->se.allFinite());
}

TEST_CASE("Pairwise ordinal composite nested LR reports Satorra spectrum") {
  std::mt19937 rng(20260619);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(110, 3);
  const double loading[3] = {0.82, 0.70, 0.70};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.40) + (y > 0.50);
    }
  }

  auto data = magmaan::estimate::frontier::pairwise_ordinal_observed_data(
      {X}, {{3, 3, 3}});
  REQUIRE(data.has_value());
  const char* h1_syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  const char* h0_syntax =
      "f =~ x1 + a*x2 + a*x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp1 = magmaan::parse::Parser::parse(h1_syntax);
  auto fp0 = magmaan::parse::Parser::parse(h0_syntax);
  REQUIRE(fp1.has_value());
  REQUIRE(fp0.has_value());
  auto pt1 = magmaan::spec::build(*fp1);
  auto pt0 = magmaan::spec::build(*fp0);
  REQUIRE(pt1.has_value());
  REQUIRE(pt0.has_value());
  auto mr1 = magmaan::model::build_matrix_rep(*pt1);
  auto mr0 = magmaan::model::build_matrix_rep(*pt0);
  REQUIRE(mr1.has_value());
  REQUIRE(mr0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 80;
  auto fit1 = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      *pt1, *mr1, *data, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto fit0 = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      *pt0, *mr0, *data, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit1.has_value(),
      "H1 pairwise composite fit failed: "
          << (fit1.has_value() ? "" : fit1.error().detail));
  REQUIRE_MESSAGE(fit0.has_value(),
      "H0 pairwise composite fit failed: "
          << (fit0.has_value() ? "" : fit0.error().detail));

  auto lr = magmaan::estimate::frontier::lr_test_pairwise_ordinal_composite(
      *pt1, *mr1, *data, *fit1, *pt0, *mr0, *fit0,
      magmaan::robust::SatorraAMethod::Exact, 2e-5);
  REQUIRE_MESSAGE(lr.has_value(),
      "pairwise composite LR failed: "
          << (lr.has_value() ? "" : lr.error().detail));
  CHECK(lr->df_diff == 1);
  CHECK(lr->T_diff == doctest::Approx(
      2.0 * (fit0->objective.negloglik - fit1->objective.negloglik)));
  CHECK(lr->eigenvalues.size() == 1);
  CHECK(std::isfinite(lr->p_scaled));
  CHECK(std::isfinite(lr->p_adjusted));
  CHECK(std::isfinite(lr->p_mixture));

  auto god0 = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      *pt0, *mr0, *data, fit0->estimates, 2e-5);
  REQUIRE_MESSAGE(god0.has_value(),
      "constrained pairwise composite Godambe failed: "
          << (god0.has_value() ? "" : god0.error().detail));
  CHECK(god0->vcov.rows() == fit0->estimates.theta.size());
  CHECK(god0->vcov.cols() == fit0->estimates.theta.size());
  CHECK(god0->se.allFinite());
}

TEST_CASE("Ordinal stats: thresholds, polychoric R, and weights have expected shapes") {
  Eigen::MatrixXd X(320, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 5; ++rep) {
    for (int x1 = 1; x1 <= 4; ++x1) {
      for (int x2 = 1; x2 <= 4; ++x2) {
        for (int x3 = 1; x3 <= 4; ++x3) {
          X(r, 0) = x1;
          X(r, 1) = x2;
          X(r, 2) = x3;
          ++r;
        }
      }
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  REQUIRE(stats->R.size() == 1);
  CHECK(stats->R[0].rows() == 3);
  CHECK(stats->R[0].cols() == 3);
  CHECK(stats->R[0].diagonal().isOnes(1e-12));
  CHECK(std::abs(stats->R[0](1, 0)) < 1e-6);
  CHECK(std::abs(stats->R[0](2, 0)) < 1e-6);
  CHECK(std::abs(stats->R[0](2, 1)) < 1e-6);
  CHECK(stats->thresholds[0].size() == 9);
  CHECK(stats->thresholds[0](0) == doctest::Approx(-0.67448975).epsilon(1e-7));
  CHECK(stats->thresholds[0](1) == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(stats->thresholds[0](2) == doctest::Approx(0.67448975).epsilon(1e-7));
  CHECK(stats->threshold_ov[0].size() == 9);
  CHECK(stats->threshold_ov[0] == std::vector<std::int32_t>({0, 0, 0, 1, 1, 1, 2, 2, 2}));
  CHECK(stats->threshold_level[0] == std::vector<std::int32_t>({1, 2, 3, 1, 2, 3, 1, 2, 3}));
  CHECK(stats->NACOV[0].rows() == 12);
  CHECK(stats->NACOV[0].cols() == 12);
  CHECK(stats->W_dwls[0].rows() == 12);
  CHECK(stats->W_dwls[0].cols() == 12);
  CHECK(stats->W_wls[0].rows() == 12);
  CHECK(stats->W_wls[0].cols() == 12);
  CHECK((stats->W_wls[0] * stats->NACOV[0])
            .isApprox(Eigen::MatrixXd::Identity(12, 12), 1e-8));
  for (Eigen::Index k = 0; k < 12; ++k) {
    CHECK(stats->W_dwls[0](k, k) == doctest::Approx(1.0 / stats->NACOV[0](k, k)));
  }
  CHECK(stats->n_obs[0] == 320);
}

TEST_CASE("Ordinal stats: metadata and moments use documented order") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  CHECK(stats->n_obs[0] == 24);
  CHECK(stats->n_levels[0] == std::vector<std::int32_t>({3, 3, 3}));
  CHECK(stats->threshold_ov[0] == std::vector<std::int32_t>({0, 0, 1, 1, 2, 2}));
  CHECK(stats->threshold_level[0] == std::vector<std::int32_t>({1, 2, 1, 2, 1, 2}));

  Eigen::VectorXd moments(9);
  moments.head(6) = stats->thresholds[0];
  moments(6) = stats->R[0](1, 0);
  moments(7) = stats->R[0](2, 0);
  moments(8) = stats->R[0](2, 1);
  CHECK(moments.allFinite());
  CHECK(stats->NACOV[0].rows() == moments.size());
  CHECK(stats->W_dwls[0].rows() == moments.size());
  CHECK(stats->W_wls[0].rows() == moments.size());
}

TEST_CASE("Ordinal stats: empty marginal categories are explicit errors") {
  Eigen::MatrixXd X(4, 2);
  X << 1, 1,
       1, 2,
       3, 2,
       3, 3;
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  CHECK_FALSE(stats.has_value());
}

TEST_CASE("Ordinal stats: near-empty categories stay finite") {
  Eigen::MatrixXd X(200, 3);
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    X(r, 0) = r == 0 ? 1 : (r + 1 == X.rows() ? 5 : 2 + (r % 3));
    X(r, 1) = 1 + ((2 * r + r / 7) % 5);
    X(r, 2) = 1 + ((3 * r + r / 11) % 5);
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  CHECK(stats->thresholds[0].allFinite());
  CHECK(stats->R[0].allFinite());
  CHECK(stats->NACOV[0].allFinite());
  CHECK(stats->W_dwls[0].allFinite());
  CHECK(stats->W_wls[0].allFinite());
  CHECK(stats->n_levels[0] == std::vector<std::int32_t>({5, 5, 5}));
}

TEST_CASE("Ordinal rows round-trip through lavaan-shaped partables and matrix_rep ignores them") {
  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  int threshold_rows = 0;
  int scale_rows = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] == magmaan::parse::Op::Threshold) {
      ++threshold_rows;
      CHECK_FALSE(mr->cell_for_row[i].used);
    }
    if (pt->op[i] == magmaan::parse::Op::ResponseScale) {
      ++scale_rows;
      CHECK_FALSE(mr->cell_for_row[i].used);
    }
  }
  CHECK(threshold_rows == 6);
  CHECK(scale_rows == 3);
}

TEST_CASE("Ordinal delta preparation fixes response variances and compacts free indices") {
  Eigen::MatrixXd X(320, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 5; ++rep) {
    for (int x1 = 1; x1 <= 4; ++x1) {
      for (int x2 = 1; x2 <= 4; ++x2) {
        for (int x3 = 1; x3 <= 4; ++x3) {
          X(r, 0) = x1;
          X(r, 1) = x2;
          X(r, 2) = x3;
          ++r;
        }
      }
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2 + t3\n"
      "x2 | t1 + t2 + t3\n"
      "x3 | t1 + t2 + t3\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  magmaan::spec::Starts starts;
  auto pt = magmaan::spec::build(*fp, {}, &starts);
  REQUIRE(pt.has_value());
  const std::int32_t old_n = pt->n_free();
  starts.hint.resize(static_cast<std::size_t>(old_n),
                     std::numeric_limits<double>::quiet_NaN());
  for (std::int32_t k = 1; k <= old_n; ++k) {
    starts.hint[static_cast<std::size_t>(k - 1)] = static_cast<double>(k);
  }

  std::int32_t old_latent_var_free = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] == magmaan::parse::Op::Covariance &&
        pt->lhs_var[i] == pt->rhs_var[i] && pt->lhs_var[i] >= 0 &&
        pt->is_user_latent[static_cast<std::size_t>(pt->lhs_var[i])] != 0) {
      old_latent_var_free = pt->free[i];
    }
  }
  REQUIRE(old_latent_var_free > 0);

  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(*pt, *stats, &starts);
  REQUIRE(prep.has_value());
  CHECK(pt->n_free() == old_n - 3);
  CHECK(starts.hint.size() == static_cast<std::size_t>(pt->n_free()));
  CHECK(starts.hint.back() == doctest::Approx(static_cast<double>(old_latent_var_free)));

  int fixed_response_variances = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] != magmaan::parse::Op::Covariance ||
        pt->lhs_var[i] != pt->rhs_var[i] || pt->lhs_var[i] < 0) {
      continue;
    }
    const bool is_observed =
        pt->ov_pos[static_cast<std::size_t>(pt->lhs_var[i])] >= 0;
    const bool is_latent =
        pt->is_user_latent[static_cast<std::size_t>(pt->lhs_var[i])] != 0;
    if (is_observed) {
      ++fixed_response_variances;
      CHECK(pt->free[i] == 0);
      CHECK(pt->fixed_value[i] == doctest::Approx(1.0));
    }
    if (is_latent) {
      CHECK(pt->free[i] == pt->n_free());
    }
  }
  CHECK(fixed_response_variances == 3);
}

TEST_CASE("Ordinal theta parameterization is a valid reparameterization of delta") {
  // Delta and Theta fit the same model to the same data — they are
  // reparameterizations, so the discrepancy at the optimum (fmin / χ²) must
  // agree. Theta differs only in the fit objective: the implied latent-
  // response moments are standardized before comparison with the polychorics.
  std::mt19937 rng(20260518);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(600, 4);
  const double loading[4] = {0.9, 0.8, 0.7, 0.6};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.6) + (y > 0.5);   // 3 categories
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
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

  using magmaan::estimate::OrdinalParameterization;
  using magmaan::estimate::OrdinalWeightKind;

  // The prepared partable is parameterization-independent.
  auto prep = magmaan::estimate::prepare_ordinal_partable(
      *pt, *stats, OrdinalParameterization::Theta);
  REQUIRE(prep.has_value());

  auto delta = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Delta);
  auto theta = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Theta);
  REQUIRE_MESSAGE(delta.has_value(),
      "delta fit failed: " << (delta.has_value() ? "" : delta.error().detail));
  REQUIRE_MESSAGE(theta.has_value(),
      "theta fit failed: " << (theta.has_value() ? "" : theta.error().detail));

  CHECK(std::isfinite(theta->fmin));
  // NB: no iteration / f_eval assertion — `simple_start_values` on this
  // 4-indicator synthetic lands close enough to the optimum that NLopt's
  // gradient at the start point is already under tolerance, returning at
  // n_evals = 0. The test's invariant is reparameterization equivalence
  // (same fmin), not the optimizer's evaluation count.
  // Reparameterization invariance: same minimized discrepancy.
  CHECK(theta->fmin == doctest::Approx(delta->fmin).epsilon(1e-4));
  // The parameter vectors themselves differ — theta loadings are unstandardized.
  CHECK((theta->theta - delta->theta).cwiseAbs().maxCoeff() > 1e-3);
}

TEST_CASE("Cache-aware ordinal theta fits and SNLLS use fit-only workspaces") {
  std::mt19937 rng(20260611);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(640, 4);
  const double loading[4] = {0.88, 0.80, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.65) + (y > 0.45);
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

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

  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS,
      magmaan::data::OrdinalMomentParameterization::Theta);
  auto uls_workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, uls_plan);
  REQUIRE(uls_workspace.has_value());
  CHECK(uls_workspace->gamma_cache.block_count() == 0);
  auto uls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, uls_workspace->moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto uls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, uls_workspace->moments, nullptr, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_bounded.has_value(),
      "theta ULS bounded failed: "
          << (uls_bounded.has_value() ? "" : uls_bounded.error().detail));
  REQUIRE_MESSAGE(uls_snlls.has_value(),
      "theta ULS SNLLS failed: "
          << (uls_snlls.has_value() ? "" : uls_snlls.error().detail));
  CHECK(uls_snlls->fmin == doctest::Approx(uls_bounded->fmin).epsilon(1e-7));
  CHECK((uls_snlls->theta - uls_bounded->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(uls_snlls->n_linear > 0);

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Theta);
  auto dwls_workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, dwls_plan);
  REQUIRE(dwls_workspace.has_value());
  REQUIRE(dwls_workspace->gamma_cache.block_count() == 1);
  CHECK(dwls_workspace->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls_workspace->gamma_cache.blocks[0].has_full);
  auto dwls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts,
      magmaan::estimate::OrdinalParameterization::Theta);
  auto dwls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, dwls_workspace->moments, &dwls_workspace->gamma_cache, {},
      dwls_plan, *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, dwls_workspace->moments, &dwls_workspace->gamma_cache,
      dwls_plan, *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_legacy.has_value(),
      "theta legacy DWLS failed: "
          << (dwls_legacy.has_value() ? "" : dwls_legacy.error().detail));
  REQUIRE_MESSAGE(dwls_cached.has_value(),
      "theta cached DWLS failed: "
          << (dwls_cached.has_value() ? "" : dwls_cached.error().detail));
  REQUIRE_MESSAGE(dwls_snlls.has_value(),
      "theta DWLS SNLLS failed: "
          << (dwls_snlls.has_value() ? "" : dwls_snlls.error().detail));
  CHECK(dwls_cached->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-8));
  CHECK((dwls_cached->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(dwls_snlls->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-7));
  CHECK((dwls_snlls->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        8e-5);
  CHECK(dwls_snlls->n_linear > 0);

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Theta);
  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts,
      magmaan::estimate::OrdinalParameterization::Theta);
  auto wls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  magmaan::data::OrdinalGammaCache wls_snlls_cache;
  wls_snlls_cache.blocks.resize(1);
  wls_snlls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_snlls_cache.blocks[0].has_full = true;
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &wls_snlls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_legacy.has_value(),
      "theta legacy WLS failed: "
          << (wls_legacy.has_value() ? "" : wls_legacy.error().detail));
  REQUIRE_MESSAGE(wls_cached.has_value(),
      "theta cached WLS failed: "
          << (wls_cached.has_value() ? "" : wls_cached.error().detail));
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "theta WLS SNLLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  CHECK(wls_cached->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-7));
  CHECK((wls_cached->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        1e-4);
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-6));
  CHECK((wls_snlls->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        2e-4);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK(wls_snlls_cache.blocks[0].has_wls_weight);
}

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

TEST_CASE("Mixed ordinal stats and DWLS fit use continuous and threshold moments") {
  std::mt19937 rng(20240515);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(600, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    const double e1 = norm(rng);
    const double e2 = norm(rng);
    const double y1 = 0.8 * eta + 0.6 * e1;
    const double y2 = 0.7 * eta + 0.7 * e2;
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = y1 + 0.2;
    X(i, 3) = y2 - 0.1;
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());
  REQUIRE(stats->R.size() == 1);
  CHECK(stats->thresholds[0].size() == 3);
  CHECK(stats->mean[0].size() == 4);
  CHECK(stats->moments[0].size() == 13);
  CHECK(stats->NACOV[0].rows() == 13);
  CHECK(stats->W_dwls[0].rows() == 13);
  CHECK(stats->W_wls[0].rows() == 13);
  CHECK(stats->R[0](0, 0) == doctest::Approx(1.0));
  CHECK(stats->R[0](1, 1) == doctest::Approx(1.0));
  CHECK(stats->R[0](2, 2) > 0.0);
  CHECK(stats->R[0](3, 3) > 0.0);

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;

  const char* th_syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n";
  auto fpt = magmaan::parse::Parser::parse(th_syntax);
  REQUIRE(fpt.has_value());
  auto pt2 = magmaan::spec::build(*fpt, opts);
  REQUIRE(pt2.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt2);
  REQUIRE(mr.has_value());
  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt2, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
  CHECK(std::isfinite(fit->fmin));

  auto rob = magmaan::estimate::robust_mixed_ordinal(
      *pt2, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(rob.has_value());
  CHECK(rob->vcov.rows() == fit->theta.size());
  CHECK(rob->se.size() == fit->theta.size());
  CHECK(rob->se.allFinite());
}

TEST_CASE("Mixed ordinal full-threshold SNLLS matches bounded DWLS/WLS") {
  std::mt19937 rng(20260612);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(560, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.7) + (eta > 0.35);
    X(i, 1) = 1.0 + (0.68 * eta + 0.74 * norm(rng) > 0.05);
    X(i, 2) = 0.82 * eta + 0.57 * norm(rng) + 0.15;
    X(i, 3) = 0.63 * eta + 0.78 * norm(rng) - 0.12;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions build_opts;
  build_opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                          "x1 | t1 + t2\n"
                                          "x2 | t1\n"
                                          "x1 ~*~ 1*x1\n"
                                          "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::mixed_ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto check_weight = [&](magmaan::estimate::OrdinalWeightKind weight) {
    auto bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
        *pt, *mr, *stats, {}, weight, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto snlls = magmaan::estimate::fit_mixed_ordinal_snlls_full_thresholds(
        *pt, *mr, *stats, weight, *x0, magmaan::estimate::Backend::NloptLbfgs,
        opts);
    REQUIRE_MESSAGE(bounded.has_value(),
                    "mixed bounded failed: "
                        << (bounded.has_value() ? "" : bounded.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
                    "mixed SNLLS failed: "
                        << (snlls.has_value() ? "" : snlls.error().detail));
    CHECK(snlls->fmin == doctest::Approx(bounded->fmin).epsilon(2e-6));
    CHECK((snlls->theta - bounded->theta).cwiseAbs().maxCoeff() < 8e-4);
    CHECK(snlls->n_nonlinear > 0);
    CHECK(snlls->n_linear > 0);
  };

  check_weight(magmaan::estimate::OrdinalWeightKind::DWLS);
  check_weight(magmaan::estimate::OrdinalWeightKind::WLS);

  // Theta flows through the same full-threshold stack: the standardized
  // covariance moments make the non-threshold block nonlinear, so only the
  // thresholds stay Golub-Pereyra linear, and the fit must agree with the
  // bounded theta fit. The delta model above keeps a binary indicator, but
  // under theta that model has a near-flat lambda/psi ridge (both optimizers
  // stall at arbitrary ridge points), so the theta parity check uses a
  // well-identified design with three-category ordinal indicators.
  std::mt19937 rng_theta(20260614);
  Eigen::MatrixXd Xt(560, 4);
  for (Eigen::Index i = 0; i < Xt.rows(); ++i) {
    const double eta = norm(rng_theta);
    const double y1 = 0.85 * eta + 0.53 * norm(rng_theta);
    const double y2 = 0.78 * eta + 0.63 * norm(rng_theta);
    Xt(i, 0) = 1.0 + (y1 > -0.7) + (y1 > 0.35);
    Xt(i, 1) = 1.0 + (y2 > -0.5) + (y2 > 0.55);
    Xt(i, 2) = 0.82 * eta + 0.57 * norm(rng_theta) + 0.15;
    Xt(i, 3) = 0.63 * eta + 0.78 * norm(rng_theta) - 0.12;
  }
  auto stats_theta =
      magmaan::data::mixed_ordinal_stats_from_data({Xt}, ordered);
  REQUIRE(stats_theta.has_value());
  auto fp_theta = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                                "x1 | t1 + t2\n"
                                                "x2 | t1 + t2\n"
                                                "x1 ~*~ 1*x1\n"
                                                "x2 ~*~ 1*x2\n");
  REQUIRE(fp_theta.has_value());
  auto pt_theta = magmaan::spec::build(*fp_theta, build_opts);
  REQUIRE(pt_theta.has_value());
  auto mr_theta = magmaan::model::build_matrix_rep(*pt_theta);
  REQUIRE(mr_theta.has_value());
  auto x0_theta = magmaan::estimate::mixed_ordinal_start_values(
      *pt_theta, *mr_theta, *stats_theta, {});
  REQUIRE(x0_theta.has_value());

  auto check_theta = [&](magmaan::estimate::OrdinalWeightKind weight) {
    auto bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
        *pt_theta, *mr_theta, *stats_theta, {}, weight, *x0_theta,
        magmaan::estimate::Backend::NloptLbfgs, opts,
        magmaan::estimate::OrdinalParameterization::Theta);
    auto snlls = magmaan::estimate::fit_mixed_ordinal_snlls_full_thresholds(
        *pt_theta, *mr_theta, *stats_theta, weight, *x0_theta,
        magmaan::estimate::Backend::NloptLbfgs, opts,
        magmaan::estimate::OrdinalParameterization::Theta);
    REQUIRE_MESSAGE(bounded.has_value(),
                    "mixed bounded theta failed: "
                        << (bounded.has_value() ? "" : bounded.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
                    "mixed SNLLS theta failed: "
                        << (snlls.has_value() ? "" : snlls.error().detail));
    CHECK(snlls->fmin == doctest::Approx(bounded->fmin).epsilon(2e-6));
    CHECK((snlls->theta - bounded->theta).cwiseAbs().maxCoeff() < 8e-4);
    CHECK(snlls->n_nonlinear > 0);
    CHECK(snlls->n_linear > 0);
  };
  check_theta(magmaan::estimate::OrdinalWeightKind::DWLS);
  check_theta(magmaan::estimate::OrdinalWeightKind::WLS);
}

TEST_CASE("Mixed ordinal fit-only workspace supplies DWLS diagonal fits") {
  std::mt19937 rng(20260613);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.65) + (eta > 0.3);
    X(i, 1) = 1.0 + (0.66 * eta + 0.75 * norm(rng) > 0.08);
    X(i, 2) = 0.84 * eta + 0.55 * norm(rng) + 0.12;
    X(i, 3) = 0.61 * eta + 0.79 * norm(rng) - 0.08;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto workspace =
      magmaan::data::mixed_ordinal_workspace_from_data({X}, ordered, plan);
  REQUIRE(workspace.has_value());
  REQUIRE(workspace->gamma_cache.blocks.size() == 1);
  CHECK(workspace->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(workspace->gamma_cache.blocks[0].has_full);
  CHECK(workspace->moments.moments[0].size() == stats->moments[0].size());
  const double moment_diff =
      (workspace->moments.moments[0] - stats->moments[0])
          .cwiseAbs()
          .maxCoeff();
  CHECK(moment_diff < 1e-10);
  REQUIRE(stats->NACOV.size() == 1);
  REQUIRE(workspace->gamma_cache.blocks[0].diagonal.size() ==
          stats->NACOV[0].rows());
  CHECK((workspace->gamma_cache.blocks[0].diagonal -
         stats->NACOV[0].diagonal())
            .cwiseAbs()
            .maxCoeff() < 1e-8);

  magmaan::spec::BuildOptions build_opts;
  build_opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                          "x1 | t1 + t2\n"
                                          "x2 | t1\n"
                                          "x1 ~*~ 1*x1\n"
                                          "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0_full =
      magmaan::estimate::mixed_ordinal_start_values(*pt, *mr, *stats, {});
  auto x0_lazy = magmaan::estimate::mixed_ordinal_start_values(
      *pt, *mr, workspace->moments, {});
  REQUIRE(x0_full.has_value());
  REQUIRE(x0_lazy.has_value());
  REQUIRE(x0_full->size() == x0_lazy->size());
  CHECK((*x0_full - *x0_lazy).cwiseAbs().maxCoeff() < 1e-10);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto full_bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS,
      *x0_full, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto lazy_bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan,
      *x0_lazy, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto lazy_snlls =
      magmaan::estimate::fit_mixed_ordinal_snlls_full_thresholds(
          *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0_lazy,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_bounded.has_value(),
                  "full mixed bounded failed: "
                      << (full_bounded.has_value() ? ""
                                                   : full_bounded.error().detail));
  REQUIRE_MESSAGE(lazy_bounded.has_value(),
                  "lazy mixed bounded failed: "
                      << (lazy_bounded.has_value() ? ""
                                                   : lazy_bounded.error().detail));
  REQUIRE_MESSAGE(lazy_snlls.has_value(),
                  "lazy mixed SNLLS failed: "
                      << (lazy_snlls.has_value() ? ""
                                                 : lazy_snlls.error().detail));
  CHECK(lazy_bounded->fmin ==
        doctest::Approx(full_bounded->fmin).epsilon(2e-6));
  CHECK(lazy_snlls->fmin == doctest::Approx(full_bounded->fmin).epsilon(2e-6));
  CHECK((lazy_bounded->theta - full_bounded->theta).cwiseAbs().maxCoeff() <
        8e-4);
  CHECK((lazy_snlls->theta - full_bounded->theta).cwiseAbs().maxCoeff() <
        8e-4);

  // A WLS plan carries the full Gamma but defers the O(m^3) inverse: the
  // workspace must hand back has_full without has_wls_weight, the cache-aware
  // fit builds the weight on demand and matches the eagerly materialized fit,
  // and the cache retains the weight afterwards.
  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto wls_workspace =
      magmaan::data::mixed_ordinal_workspace_from_data({X}, ordered, wls_plan);
  REQUIRE(wls_workspace.has_value());
  REQUIRE(wls_workspace->gamma_cache.blocks.size() == 1);
  CHECK(wls_workspace->gamma_cache.blocks[0].has_full);
  CHECK_FALSE(wls_workspace->gamma_cache.blocks[0].has_wls_weight);
  auto full_wls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS,
      *x0_full, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto lazy_wls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, wls_workspace->moments, &wls_workspace->gamma_cache, {},
      wls_plan, *x0_lazy, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_wls.has_value(),
                  "full mixed WLS failed: "
                      << (full_wls.has_value() ? "" : full_wls.error().detail));
  REQUIRE_MESSAGE(lazy_wls.has_value(),
                  "lazy mixed WLS failed: "
                      << (lazy_wls.has_value() ? "" : lazy_wls.error().detail));
  CHECK(lazy_wls->fmin == doctest::Approx(full_wls->fmin).epsilon(1e-10));
  CHECK((lazy_wls->theta - full_wls->theta).cwiseAbs().maxCoeff() < 1e-8);
  CHECK(wls_workspace->gamma_cache.blocks[0].has_wls_weight);
}

TEST_CASE("Mixed ordinal full-Gamma cache supplies robust reporting") {
  std::mt19937 rng(20260614);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(540, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.7) + (eta > 0.35);
    X(i, 1) = 1.0 + (0.62 * eta + 0.78 * norm(rng) > 0.02);
    X(i, 2) = 0.80 * eta + 0.62 * norm(rng) + 0.16;
    X(i, 3) = 0.58 * eta + 0.82 * norm(rng) - 0.05;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions build_opts;
  build_opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                          "x1 | t1 + t2\n"
                                          "x2 | t1\n"
                                          "x1 ~*~ 1*x1\n"
                                          "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  const auto moments = magmaan::data::mixed_ordinal_moments_from_stats(*stats);
  auto x0 = magmaan::estimate::mixed_ordinal_start_values(
      *pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto full_dwls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS,
      *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_dwls.has_value(),
      "full mixed DWLS failed: " <<
      (full_dwls.has_value() ? "" : full_dwls.error().detail));

  auto dwls_cache = magmaan::data::ordinal_gamma_cache_from_stats(*stats);
  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitPlusInference,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto cached_dwls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, moments, &dwls_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(cached_dwls.has_value(),
      "cached mixed DWLS failed: " <<
      (cached_dwls.has_value() ? "" : cached_dwls.error().detail));
  CHECK(cached_dwls->fmin == doctest::Approx(full_dwls->fmin).epsilon(2e-6));
  CHECK((cached_dwls->theta - full_dwls->theta).cwiseAbs().maxCoeff() < 8e-4);
  CHECK(dwls_cache.blocks[0].has_full);
  CHECK(dwls_cache.blocks[0].has_diagonal);
  CHECK(dwls_cache.blocks[0].has_dwls_weight);

  auto materialized_dwls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *cached_dwls,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  auto cached_dwls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, moments, dwls_cache, *cached_dwls, dwls_plan);
  REQUIRE_MESSAGE(materialized_dwls_rob.has_value(),
      "materialized mixed DWLS robust failed: " <<
      (materialized_dwls_rob.has_value()
           ? ""
           : materialized_dwls_rob.error().detail));
  REQUIRE_MESSAGE(cached_dwls_rob.has_value(),
      "cached mixed DWLS robust failed: " <<
      (cached_dwls_rob.has_value() ? "" : cached_dwls_rob.error().detail));
  CHECK(cached_dwls_rob->chisq_standard ==
        doctest::Approx(materialized_dwls_rob->chisq_standard).epsilon(1e-12));
  CHECK(cached_dwls_rob->df == materialized_dwls_rob->df);
  CHECK((cached_dwls_rob->se - materialized_dwls_rob->se)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_dwls_rob->eigvals - materialized_dwls_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK(cached_dwls_rob->satorra_bentler.chi2_scaled ==
        doctest::Approx(materialized_dwls_rob->satorra_bentler.chi2_scaled));
  CHECK(cached_dwls_rob->mean_var_adjusted.chi2_adj ==
        doctest::Approx(materialized_dwls_rob->mean_var_adjusted.chi2_adj));
  CHECK(cached_dwls_rob->scaled_shifted.chi2_adj ==
        doctest::Approx(materialized_dwls_rob->scaled_shifted.chi2_adj));

  auto full_wls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS,
      *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_wls.has_value(),
      "full mixed WLS failed: " <<
      (full_wls.has_value() ? "" : full_wls.error().detail));
  auto materialized_wls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *full_wls,
      magmaan::estimate::OrdinalWeightKind::WLS);

  auto wls_cache = magmaan::data::ordinal_gamma_cache_from_stats(*stats);
  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::InferenceOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto cached_wls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, moments, wls_cache, *full_wls, wls_plan);
  REQUIRE_MESSAGE(materialized_wls_rob.has_value(),
      "materialized mixed WLS robust failed: " <<
      (materialized_wls_rob.has_value()
           ? ""
           : materialized_wls_rob.error().detail));
  REQUIRE_MESSAGE(cached_wls_rob.has_value(),
      "cached mixed WLS robust failed: " <<
      (cached_wls_rob.has_value() ? "" : cached_wls_rob.error().detail));
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK((cached_wls_rob->vcov - materialized_wls_rob->vcov)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_wls_rob->eigvals - materialized_wls_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
}

TEST_CASE("Mixed ordinal polyserial DPD keeps shared marginals and fits DWLS") {
  std::mt19937 rng(20260517);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng) + 0.2;
    X(i, 3) = 0.7 * eta + 0.7 * norm(rng) - 0.1;
  }
  for (Eigen::Index i = 0; i < 28; ++i) {
    X(i, 0) = 1.0;
    X(i, 2) = 7.0 + 0.02 * static_cast<double>(i);
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};

  auto base = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  auto ml_limit = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
      {X}, ordered, magmaan::data::PolyserialPairDpdOptions{.alpha = 0.0});
  auto robust = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
      {X}, ordered, magmaan::data::PolyserialPairDpdOptions{.alpha = 0.45});
  REQUIRE(base.has_value());
  REQUIRE(ml_limit.has_value());
  REQUIRE(robust.has_value());

  CHECK(ml_limit->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(ml_limit->stats.R[0].isApprox(base->R[0], 1e-10));
  CHECK(ml_limit->stats.NACOV[0].isApprox(base->NACOV[0], 1e-8));
  CHECK(robust->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(robust->stats.mean[0].isApprox(base->mean[0], 0.0));
  CHECK(robust->stats.R[0](0, 0) == doctest::Approx(base->R[0](0, 0)));
  CHECK(robust->stats.R[0](1, 1) == doctest::Approx(base->R[0](1, 1)));
  CHECK(robust->stats.R[0](2, 2) == doctest::Approx(base->R[0](2, 2)));
  CHECK(robust->stats.R[0](3, 3) == doctest::Approx(base->R[0](3, 3)));
  CHECK(std::abs(robust->stats.R[0](2, 0) - base->R[0](2, 0)) > 1e-4);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(robust->block_diagnostics.size() == 1);
  CHECK(robust->block_diagnostics[0].dpd_pairs.size() == 4);
  CHECK(robust->block_diagnostics[0].dpd_fits.size() == 4);
  CHECK(robust->block_diagnostics[0].dpd_fits[0].weights.allFinite());

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
      *pt, *mr, robust->stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
  CHECK(std::isfinite(fit->fmin));
}

TEST_CASE("Huber residual clipping API covers hard, smooth, Tukey, and no-clip") {
  using magmaan::data::HuberResidualClipKind;
  using magmaan::data::HuberResidualClipOptions;

  auto none = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{.kind = HuberResidualClipKind::None});
  REQUIRE(none.has_value());
  CHECK(none->psi == doctest::Approx(2.0));
  CHECK(none->dpsi == doctest::Approx(1.0));

  auto hard = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{
               .kind = HuberResidualClipKind::HardHuber, .k = 1.25});
  REQUIRE(hard.has_value());
  CHECK(hard->psi == doctest::Approx(1.25));
  CHECK(hard->dpsi == doctest::Approx(0.0));
  CHECK(hard->weight == doctest::Approx(0.625));

  auto pseudo = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{
               .kind = HuberResidualClipKind::PseudoHuber, .k = 1.0});
  REQUIRE(pseudo.has_value());
  CHECK(pseudo->psi == doctest::Approx(2.0 / std::sqrt(5.0)));
  CHECK(pseudo->dpsi == doctest::Approx(1.0 / (std::sqrt(5.0) * 5.0)));

  auto tukey = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{
               .kind = HuberResidualClipKind::TukeyBiweight, .k = 1.5});
  REQUIRE(tukey.has_value());
  CHECK(tukey->psi == doctest::Approx(0.0));
  CHECK(tukey->dpsi == doctest::Approx(0.0));
}

TEST_CASE("Mixed ordinal Huber residual no-clip preserves single-ordinal ML Gamma") {
  std::mt19937 rng(20260519);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(360, 3);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.65) + (eta > 0.45);
    X(i, 1) = 0.8 * eta + 0.6 * norm(rng);
    X(i, 2) = 0.55 * eta + 0.84 * norm(rng);
  }
  for (Eigen::Index i = 0; i < 18; ++i) {
    X(i, 0) = 1.0;
    X(i, 1) = 5.5 + 0.02 * static_cast<double>(i);
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 0}};

  auto base = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  auto none = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::None},
          .correlation_repair = {}});
  auto robust = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::HardHuber,
              .k = 1.345},
          .correlation_repair = {}});
  REQUIRE(base.has_value());
  REQUIRE(none.has_value());
  REQUIRE(robust.has_value());

  CHECK(none->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(none->stats.R[0].isApprox(base->R[0], 1e-12));
  CHECK(none->stats.NACOV[0].isApprox(base->NACOV[0], 1e-10));
  CHECK(none->block_diagnostics[0].gamma.isApprox(
      (none->block_diagnostics[0].moment_influence.transpose() *
       none->block_diagnostics[0].moment_influence) /
          static_cast<double>(none->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(robust->block_diagnostics[0].robust_pairs.size() == 2);
  CHECK_FALSE(robust->stats.NACOV[0].isApprox(base->NACOV[0], 1e-6));
}

TEST_CASE("Mixed ordinal Huber residual stats rebuild Gamma and preserve continuous moments") {
  std::mt19937 rng(20260518);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(420, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.55) + (eta > 0.55);
    X(i, 1) = 1.0 + (0.7 * eta + 0.72 * norm(rng) > 0.0);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng);
    X(i, 3) = 0.6 * eta + 0.8 * norm(rng);
  }
  for (Eigen::Index i = 0; i < 20; ++i) {
    X(i, 0) = 1.0;
    X(i, 1) = 2.0;
    X(i, 2) = 6.0 + 0.01 * static_cast<double>(i);
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};

  auto base = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  auto none = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::None},
          .correlation_repair = {}});
  auto robust = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::HardHuber,
              .k = 1.345},
          .correlation_repair = {}});
  REQUIRE(base.has_value());
  REQUIRE(none.has_value());
  REQUIRE(robust.has_value());

  CHECK(none->stats.mean[0].isApprox(base->mean[0], 0.0));
  CHECK(robust->stats.mean[0].isApprox(base->mean[0], 0.0));
  CHECK(robust->stats.R[0](2, 2) == doctest::Approx(base->R[0](2, 2)));
  CHECK(robust->stats.R[0](3, 3) == doctest::Approx(base->R[0](3, 3)));
  CHECK(std::abs(robust->stats.R[0](2, 0) - base->R[0](2, 0)) > 1e-5);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(robust->block_diagnostics[0].robust_pairs.size() == 5);
  CHECK(robust->block_diagnostics[0].rho.size() == 5);

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
      *pt, *mr, robust->stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
}

TEST_CASE("Mixed ordinal Huber residual Monte Carlo keeps sparse contaminated Gamma stable") {
  std::normal_distribution<double> norm(0.0, 1.0);
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  double clean_to_ml = 0.0;
  double clean_to_huber = 0.0;
  double huber_log_condition = 0.0;
  int stable_reps = 0;

  for (int rep = 0; rep < 6; ++rep) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(20260600 + rep));
    Eigen::MatrixXd clean(300, 4);
    for (Eigen::Index i = 0; i < clean.rows(); ++i) {
      const double eta = norm(rng);
      const double z1 = 0.78 * eta + 0.63 * norm(rng);
      const double z2 = 0.62 * eta + 0.78 * norm(rng);
      clean(i, 0) = 1.0 + (z1 > -1.65) + (z1 > -0.25) + (z1 > 1.35);
      clean(i, 1) = 1.0 + (z2 > -1.15) + (z2 > 0.35);
      clean(i, 2) = 0.75 * eta + 0.66 * norm(rng);
      clean(i, 3) = 0.55 * eta + 0.84 * norm(rng);
    }

    Eigen::MatrixXd contaminated(clean.rows() + 24, clean.cols());
    contaminated.topRows(clean.rows()) = clean;
    for (Eigen::Index i = 0; i < 24; ++i) {
      contaminated(clean.rows() + i, 0) = 1.0;
      contaminated(clean.rows() + i, 1) = 3.0;
      contaminated(clean.rows() + i, 2) = 6.0 + 0.03 * static_cast<double>(i);
      contaminated(clean.rows() + i, 3) = -5.5 - 0.02 * static_cast<double>(i);
    }

    auto clean_stats = magmaan::data::mixed_ordinal_stats_from_data(
        {clean}, ordered);
    auto base = magmaan::data::mixed_ordinal_stats_from_data(
        {contaminated}, ordered);
    auto huber = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
        {contaminated}, ordered,
        magmaan::data::MixedOrdinalHuberResidualOptions{
            .clip = magmaan::data::HuberResidualClipOptions{
                .kind = magmaan::data::HuberResidualClipKind::HardHuber,
                .k = 1.20},
            .correlation_repair =
                magmaan::data::MixedOrdinalCorrelationRepairOptions{
                    .kind = magmaan::data::MixedOrdinalCorrelationRepairKind::Ridge,
                    .min_eigenvalue = 1e-6}});
    auto dpd = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
        {contaminated}, ordered,
        magmaan::data::PolyserialPairDpdOptions{.alpha = 0.35});
    REQUIRE(clean_stats.has_value());
    REQUIRE(base.has_value());
    REQUIRE(huber.has_value());
    REQUIRE(dpd.has_value());

    const Eigen::MatrixXd& gamma = huber->stats.NACOV[0];
    CHECK(gamma.allFinite());
    CHECK(huber->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
    CHECK(huber->stats.W_dwls[0].diagonal().allFinite());
    CHECK(huber->block_diagnostics[0].gamma.isApprox(
        (huber->block_diagnostics[0].moment_influence.transpose() *
         huber->block_diagnostics[0].moment_influence) /
            static_cast<double>(huber->stats.n_obs[0]),
        1e-12));
    CHECK(dpd->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
    CHECK(dpd->stats.W_dwls[0].diagonal().allFinite());

    const double base_cond = symmetric_condition_number(base->NACOV[0]);
    const double huber_cond = symmetric_condition_number(gamma);
    CHECK(std::isfinite(base_cond));
    CHECK(std::isfinite(huber_cond));
    CHECK(huber_cond < 1e14);
    huber_log_condition += std::log(huber_cond);

    const double clean_r = clean_stats->R[0](2, 0);
    const double base_r = base->R[0](2, 0);
    const double huber_r = huber->stats.R[0](2, 0);
    clean_to_ml += std::abs(base_r - clean_r);
    clean_to_huber += std::abs(huber_r - clean_r);
    CHECK(std::abs(huber_r - base_r) > 1e-4);
    CHECK(huber->block_diagnostics[0].objective.allFinite());
    ++stable_reps;
  }

  REQUIRE(stable_reps == 6);
  CHECK(clean_to_huber < clean_to_ml);
  CHECK(huber_log_condition / static_cast<double>(stable_reps) < std::log(1e8));
}

TEST_CASE("Mixed ordinal validation rejects malformed stats before fitting") {
  std::mt19937 rng(20240516);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(240, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng);
    X(i, 3) = 0.7 * eta + 0.7 * norm(rng);
  }
  auto stats = magmaan::data::mixed_ordinal_stats_from_data(
      {X}, std::vector<std::vector<std::int32_t>>{{1, 1, 0, 0}});
  REQUIRE(stats.has_value());

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

  auto expect_error = [&](magmaan::data::MixedOrdinalStats bad,
                          const char* needle) {
    // Call the core directly: malformed stats must be rejected by
    // validate_stats, which runs before the x0 size check — so an empty x0
    // is fine here.
    auto fit = magmaan::estimate::fit_mixed_ordinal_bounded(
        *pt, *mr, bad, {}, magmaan::estimate::OrdinalWeightKind::DWLS,
        Eigen::VectorXd{});
    REQUIRE_FALSE(fit.has_value());
    CHECK(fit.error().detail.find(needle) != std::string::npos);
  };

  {
    auto bad = *stats;
    bad.ordered[0][0] = 2;
    expect_error(std::move(bad), "ordered mask");
  }
  {
    auto bad = *stats;
    bad.ordered[0][3] = 1;
    expect_error(std::move(bad), "missing thresholds");
  }
  {
    auto bad = *stats;
    bad.threshold_ov[0][0] = 2;
    expect_error(std::move(bad), "continuous variable");
  }
  {
    auto bad = *stats;
    bad.NACOV[0](0, 0) = -1.0;
    expect_error(std::move(bad), "NACOV diagonal");
  }
  {
    auto bad = *stats;
    bad.mean[0](0) = std::numeric_limits<double>::quiet_NaN();
    expect_error(std::move(bad), "non-finite");
  }
}

TEST_CASE("Mixed ordinal theta parameterization fits and supports post-fit reporting") {
  std::mt19937 rng(20260520);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng) + 0.2;
    X(i, 3) = 0.7 * eta + 0.7 * norm(rng) - 0.1;
  }
  auto stats = magmaan::data::mixed_ordinal_stats_from_data(
      {X}, std::vector<std::vector<std::int32_t>>{{1, 1, 0, 0}});
  REQUIRE(stats.has_value());

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

  using magmaan::estimate::OrdinalParameterization;
  using magmaan::estimate::OrdinalWeightKind;
  auto delta = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Delta);
  auto theta = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Theta);
  REQUIRE_MESSAGE(delta.has_value(),
      "delta fit failed: " << (delta.has_value() ? "" : delta.error().detail));
  REQUIRE_MESSAGE(theta.has_value(),
      "theta fit failed: " << (theta.has_value() ? "" : theta.error().detail));
  CHECK(theta->theta.allFinite());
  CHECK(std::isfinite(theta->fmin));
  CHECK((theta->theta - delta->theta).cwiseAbs().maxCoeff() > 1e-3);

  auto rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *theta, OrdinalWeightKind::DWLS,
      OrdinalParameterization::Theta);
  REQUIRE(rob.has_value());
  CHECK(rob->vcov.rows() == theta->theta.size());
  CHECK(rob->se.allFinite());
  CHECK(rob->eigvals.size() == rob->df);

  magmaan::inference::ModificationIndexOptions mi_opts;
  mi_opts.candidates = magmaan::inference::ScoreCandidateSet::WithAbsentRows;
  auto mi = magmaan::estimate::modification_indices_mixed_ordinal(
      *pt, *mr, *stats, *theta, OrdinalWeightKind::DWLS, mi_opts,
      OrdinalParameterization::Theta);
  REQUIRE(mi.has_value());
  CHECK_FALSE(mi->rows.empty());

  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_mixed_ordinal_partable(
      pt_prepared, *stats, OrdinalParameterization::Theta);
  REQUIRE(prep.has_value());
  auto ev_dbg = magmaan::model::ModelEvaluator::build(pt_prepared, *mr);
  REQUIRE(ev_dbg.has_value());
  CHECK(ev_dbg->param_locations().size() ==
        static_cast<std::size_t>(theta->theta.size()));
  auto std_all = magmaan::measures::standardize::standardize_all(
      pt_prepared, *mr, *theta, rob->vcov);
  REQUIRE_MESSAGE(std_all.has_value(),
      "standardize_all failed: " <<
      (std_all.has_value() ? "" : std_all.error().detail));
  if (std_all.has_value()) {
    CHECK(std_all->theta.size() == theta->theta.size());
    CHECK(std_all->se.allFinite());
  }
}
