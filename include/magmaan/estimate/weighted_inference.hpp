#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/gls/gls.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/nt/robust.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/gls/wls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate {

using data::RawData;
using data::SampleStats;
using gls::GLS;
using gls::ULS;
using gls::WLS;
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

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     gls::ULS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     gls::GLS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     gls::WLS discrepancy,
                     const std::vector<Eigen::MatrixXd>& gamma);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     gls::ULS discrepancy,
                     const data::RawData& raw);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     gls::GLS discrepancy,
                     const data::RawData& raw);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     gls::WLS discrepancy,
                     const data::RawData& raw);

}  // namespace magmaan::estimate
