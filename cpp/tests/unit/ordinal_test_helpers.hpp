#pragma once

#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "../../src/estimate/detail_theta_threshold_profile.hpp"

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
#include <Eigen/QR>

#include "magmaan/data/h_score.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/pairwise_mixed.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/frontier/pairwise.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/measures/reliability.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/frontier/fmg.hpp"
#include "magmaan/robust/weighted_chisq.hpp"
#include "magmaan/spec/build.hpp"

namespace ordinal_test_detail {
Eigen::MatrixXd ordinal_expected_counts(const Eigen::VectorXd& th_i,
                                        const Eigen::VectorXd& th_j,
                                        double rho,
                                        double total);
double h_score_pair_objective(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho,
    const magmaan::data::PolychoricHScoreOptions& options);
double dpd_pair_objective(const Eigen::MatrixXd& counts,
                          const Eigen::VectorXd& th_i,
                          const Eigen::VectorXd& th_j,
                          double rho,
                          double alpha);
Eigen::MatrixXd ordinal_pair_score_rows_from_counts(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho);
Eigen::MatrixXd ordinal_data_from_pair_counts(const Eigen::MatrixXd& counts);
double std_normal_cdf(double x) noexcept;
double std_normal_pdf(double x) noexcept;
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
    const Eigen::MatrixXd& R);
Eigen::VectorXd finite_diff_gamma_diag_case_influence(
    const GammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps = 1e-6);
Eigen::MatrixXd finite_diff_gamma_case_influence(
    const GammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps = 1e-6);
struct MixedGammaDiagInfluenceProbe {
  Eigen::MatrixXd SC;
  Eigen::MatrixXd B;
  Eigen::MatrixXd H;
  Eigen::MatrixXd Gamma;
  std::vector<Eigen::MatrixXd> b_case;
};
MixedGammaDiagInfluenceProbe mixed_gamma_diag_influence_probe(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R);
Eigen::VectorXd finite_diff_mixed_gamma_diag_case_influence(
    const MixedGammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps = 1e-6);
Eigen::MatrixXd finite_diff_mixed_gamma_case_influence(
    const MixedGammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps = 1e-6);
double symmetric_condition_number(const Eigen::MatrixXd& x);
bool matrix_matches_with_nan(const Eigen::MatrixXd& lhs,
                             const Eigen::MatrixXd& rhs,
                             double tol = 0.0);
// Synthetic one-factor 3-category block for multi-group profiling tests.
Eigen::MatrixXd ordinal_test_block(std::uint32_t seed,
                                   Eigen::Index n,
                                   const std::array<double, 4>& loading,
                                   double cut1,
                                   double cut2);

}  // namespace ordinal_test_detail
using namespace ordinal_test_detail;
