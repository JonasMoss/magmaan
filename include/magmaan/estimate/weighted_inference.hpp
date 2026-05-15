#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/fit.hpp"
#include "magmaan/fit/gls.hpp"
#include "magmaan/fit/raw_data.hpp"
#include "magmaan/fit/robust.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/fit/uls.hpp"
#include "magmaan/fit/wls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate {

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
  fit::SatorraBentlerResult satorra_bentler;
  fit::MeanVarAdjustedResult mean_var_adjusted;
  fit::ScaledShiftedResult scaled_shifted;
};

post_expected<WeightedRobustResult>
robust_weighted_moments(const std::vector<WeightedMomentBlock>& blocks,
                        const Eigen::MatrixXd& K,
                        double fmin);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::ULS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::GLS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::WLS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::ULS discrepancy,
                     const fit::RawData& raw);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::GLS discrepancy,
                     const fit::RawData& raw);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const fit::SampleStats& samp,
                     const fit::Estimates& est,
                     fit::WLS discrepancy,
                     const fit::RawData& raw);

}  // namespace magmaan::estimate
