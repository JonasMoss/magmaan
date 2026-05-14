#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

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
  auto pt = magmaan::spec::lavaanify(*fp);
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
  auto pt = magmaan::spec::lavaanify(*fp, {}, &starts);
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
