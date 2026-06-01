#include <doctest/doctest.h>

#include <vector>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/error.hpp"
#include "magmaan/sim/norta.hpp"
#include "magmaan/sim/projection.hpp"

TEST_CASE("projection derives normal thresholds from category probabilities") {
  Eigen::VectorXd probabilities(3);
  probabilities << 0.25, 0.50, 0.25;

  auto thresholds_or =
      magmaan::sim::thresholds_from_probabilities(probabilities);
  REQUIRE(thresholds_or.has_value());
  REQUIRE(thresholds_or->size() == 2);
  CHECK(magmaan::sim::normal_cdf((*thresholds_or)(0)) ==
        doctest::Approx(0.25).epsilon(1e-13));
  CHECK(magmaan::sim::normal_cdf((*thresholds_or)(1)) ==
        doctest::Approx(0.75).epsilon(1e-13));
}

TEST_CASE("ordinal projection codes thresholded columns as one-based categories") {
  Eigen::MatrixXd latent(6, 2);
  latent << -2.0, -1.0,
            -0.5, -0.2,
             0.2,  0.1,
             0.8,  0.4,
             1.3,  0.9,
             2.0,  1.4;
  Eigen::VectorXd th0(2);
  th0 << 0.0, 1.0;
  Eigen::VectorXd th1(1);
  th1 << 0.5;

  auto projected_or = magmaan::sim::project_ordinal_matrix(latent, {th0, th1});
  REQUIRE(projected_or.has_value());
  const auto& projected = *projected_or;
  CHECK(projected.ordered == std::vector<std::int32_t>{1, 1});
  CHECK(projected.n_levels == std::vector<std::int32_t>{3, 2});
  CHECK(projected.category_counts[0](0) == 2);
  CHECK(projected.category_counts[0](1) == 2);
  CHECK(projected.category_counts[0](2) == 2);
  CHECK(projected.category_counts[1](0) == 4);
  CHECK(projected.category_counts[1](1) == 2);
  CHECK(projected.X(0, 0) == 1.0);
  CHECK(projected.X(3, 0) == 2.0);
  CHECK(projected.X(5, 0) == 3.0);

  auto workspace_or = magmaan::data::ordinal_workspace_from_integer_data(
      {projected.X},
      magmaan::data::ordinal_weight_plan(
          magmaan::data::OrdinalWorkspacePurpose::FitOnly,
          magmaan::data::OrdinalEstimatorKind::ULS));
  REQUIRE(workspace_or.has_value());
  CHECK(workspace_or->moments.n_levels[0] ==
        std::vector<std::int32_t>{3, 2});
}

TEST_CASE("mixed projection preserves continuous columns and marks ordinal columns") {
  Eigen::MatrixXd latent(6, 3);
  latent << -2.0, 10.0, -1.0,
            -0.5, 11.0, -0.2,
             0.2, 12.0,  0.1,
             0.8, 13.0,  0.4,
             1.3, 14.0,  0.9,
             2.0, 15.0,  1.4;
  Eigen::VectorXd th0(2);
  th0 << 0.0, 1.0;
  Eigen::VectorXd th2(1);
  th2 << 0.5;

  magmaan::sim::MixedProjectionSpec spec;
  spec.kinds = {magmaan::sim::ObservedKind::Ordinal,
                magmaan::sim::ObservedKind::Continuous,
                magmaan::sim::ObservedKind::Ordinal};
  spec.thresholds = {th0, Eigen::VectorXd{}, th2};

  auto projected_or = magmaan::sim::project_mixed_matrix(latent, spec);
  REQUIRE(projected_or.has_value());
  const auto& projected = *projected_or;
  CHECK(projected.ordered == std::vector<std::int32_t>{1, 0, 1});
  CHECK(projected.n_levels == std::vector<std::int32_t>{3, 0, 2});
  CHECK(projected.X.col(1).isApprox(latent.col(1), 0.0));
  CHECK(projected.X(0, 0) == 1.0);
  CHECK(projected.X(5, 0) == 3.0);
  CHECK(projected.X(0, 2) == 1.0);
  CHECK(projected.X(5, 2) == 2.0);

  auto workspace_or = magmaan::data::mixed_ordinal_workspace_from_data(
      {projected.X}, {projected.ordered},
      magmaan::data::ordinal_weight_plan(
          magmaan::data::OrdinalWorkspacePurpose::FitOnly,
          magmaan::data::OrdinalEstimatorKind::ULS));
  REQUIRE(workspace_or.has_value());
  CHECK(workspace_or->moments.ordered[0] ==
        std::vector<std::int32_t>{1, 0, 1});
}

TEST_CASE("projection validates probabilities and thresholds") {
  Eigen::VectorXd bad_prob(2);
  bad_prob << 0.5, 0.0;
  auto probs_or = magmaan::sim::thresholds_from_probabilities(bad_prob);
  REQUIRE_FALSE(probs_or.has_value());
  CHECK(probs_or.error().kind == magmaan::SimError::Kind::InvalidInput);

  Eigen::MatrixXd latent(2, 1);
  latent << 0.0,
            1.0;
  Eigen::VectorXd unsorted(2);
  unsorted << 1.0, 0.0;
  auto projected_or =
      magmaan::sim::project_ordinal_matrix(latent, {unsorted});
  REQUIRE_FALSE(projected_or.has_value());
  CHECK(projected_or.error().kind == magmaan::SimError::Kind::InvalidInput);
}
