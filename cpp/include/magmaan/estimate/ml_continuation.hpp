#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate::frontier {

enum class MlContinuationTarget : std::uint8_t {
  Diagonal,
  ScaledIdentity,
  Identity,
};

struct MlRidgeContinuationOptions {
  std::vector<double> alphas{0.50, 0.20, 0.10, 0.05, 0.01};
  MlContinuationTarget target = MlContinuationTarget::Diagonal;
  bool include_endpoint = true;
  double diagonal_floor = 1e-8;
};

struct MlRidgeContinuationStep {
  double alpha = 0.0;
  double min_sample_eigen = 0.0;
  double max_sample_eigen = 0.0;
  double sample_condition = 0.0;
  Estimates estimates;
};

struct MlRidgeContinuationResult {
  Estimates final;
  data::SampleStats final_sample_stats;
  std::vector<MlRidgeContinuationStep> steps;
  int total_iterations = 0;
  int total_f_evals = 0;
  int total_g_evals = 0;
};

// Fit complete-data normal-theory ML along a covariance-continuation path:
//
//   S_alpha = (1 - alpha) S + alpha T(S)
//
// where T(S) is diag(S), a scale-matched identity matrix, or the raw identity
// matrix. Each stage warm-starts from the previous stage's estimate; with the
// default options the last stage is alpha = 0, so the returned `final` estimate
// is the ordinary ML endpoint for the caller's original sample statistics.
fit_expected<MlRidgeContinuationResult>
fit_ml_ridge_continuation(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::SampleStats& samp, const Eigen::VectorXd& x0,
    Bounds bounds = {}, Backend backend = Backend::NloptLbfgs,
    optim::OptimOptions opts = {},
    MlRidgeContinuationOptions continuation = {});

}  // namespace magmaan::estimate::frontier
