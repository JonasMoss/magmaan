#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/gmm/moment_quadratic.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/nt/robust.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate {

using data::RawData;
using data::SampleStats;
using nt::robust::MeanVarAdjustedResult;
using nt::robust::SatorraBentlerResult;
using nt::robust::ScaledShiftedResult;

struct WeightedMomentBlock {
  Eigen::MatrixXd jacobian;  // model moments wrt full free theta
  Eigen::MatrixXd weight;    // estimator weight for this moment block
  Eigen::MatrixXd gamma;     // NACOV/meat for this moment block
  std::int64_t n_obs = 0;
};

struct WeightedRobustResult {
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
  Eigen::VectorXd eigvals;
  double chisq_standard = 0.0;
  int df = 0;
  nt::robust::SatorraBentlerResult satorra_bentler;
  nt::robust::MeanVarAdjustedResult mean_var_adjusted;
  nt::robust::ScaledShiftedResult scaled_shifted;
};

post_expected<WeightedRobustResult>
robust_weighted_moments(const std::vector<WeightedMomentBlock>& blocks,
                        const Eigen::MatrixXd& K,
                        double fmin);

// Moment-quadratic least-squares objective F(θ) = ½·r̃(θ)ᵀr̃(θ). The `weight`
// is the only thing that distinguishes the estimators: empty ⇒ ULS; a
// normal-theory weight (`gmm::normal_theory_weight`) ⇒ GLS; a caller-supplied
// weight ⇒ WLS / DWLS.
post_expected<double>
evaluate_ls_objective(spec::LatentStructure pt,
                      const model::MatrixRep& rep,
                      const data::SampleStats& samp,
                      const Eigen::VectorXd& theta,
                      const gmm::Weight& weight);

// Continuous-LS reference χ². An empty `weight` ⇒ ULS, reported via Browne's
// residual-based normal-theory statistic; a non-empty `weight` ⇒ GLS / WLS,
// reported as 2N·fmin.
post_expected<double>
continuous_ls_chisq(data::SampleStats samp,
                    spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const Estimates& est,
                    const gmm::Weight& weight);

// Robust (sandwich) inference for a continuous moment-quadratic fit. `weight`
// follows the same convention as above (empty ⇒ ULS identity weight). `gamma`
// supplies the per-block moment NACOV directly, or `raw` supplies the raw data
// from which it is estimated.
post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     const gmm::Weight& weight,
                     const std::vector<Eigen::MatrixXd>& gamma);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     const gmm::Weight& weight,
                     const data::RawData& raw);

}  // namespace magmaan::estimate
