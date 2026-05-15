#pragma once

#include <string_view>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/nt/robust.hpp"
#include "magmaan/nt/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/concepts.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using nt::infer::ScoreCandidate;
using nt::infer::ScoreCandidateKind;
using nt::infer::ScoreTestResult;
using nt::infer::ScoreTestTable;
using nt::infer::chi2_pvalue;
using nt::robust::MeanVarAdjustedResult;
using nt::robust::SatorraBentlerResult;
using nt::robust::ScaledShiftedResult;

enum class OrdinalWeightKind {
  DWLS,
  WLS,
};

enum class OrdinalParameterization {
  Delta,
  Theta,
};

struct OrdinalRobustResult {
  Eigen::MatrixXd vcov;              // full free-parameter covariance
  Eigen::VectorXd se;                // sqrt(diag(vcov))
  Eigen::VectorXd eigvals;           // nonzero U-Gamma eigenvalues
  double chisq_standard = 0.0;       // N * F_min
  int df = 0;
  nt::robust::SatorraBentlerResult satorra_bentler;
  nt::robust::MeanVarAdjustedResult mean_var_adjusted;
  nt::robust::ScaledShiftedResult scaled_shifted;
};

fit_expected<void>
prepare_ordinal_delta_partable(spec::LatentStructure& pt,
                                const data::OrdinalStats& stats,
                                spec::Starts* starts = nullptr);

fit_expected<void>
prepare_ordinal_partable(spec::LatentStructure& pt,
                         const data::OrdinalStats& stats,
                         OrdinalParameterization parameterization,
                         spec::Starts* starts = nullptr);

fit_expected<void>
prepare_mixed_ordinal_delta_partable(spec::LatentStructure& pt,
                                      const data::MixedOrdinalStats& stats,
                                      spec::Starts* starts = nullptr);

fit_expected<void>
prepare_mixed_ordinal_partable(spec::LatentStructure& pt,
                                const data::MixedOrdinalStats& stats,
                                OrdinalParameterization parameterization,
                                spec::Starts* starts = nullptr);

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalStats& stats,
               const Estimates& est,
               OrdinalWeightKind weights);

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalStats& stats,
                     const Estimates& est,
                     OrdinalWeightKind weights);

post_expected<nt::infer::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights);

post_expected<nt::infer::ScoreTestTable>
score_tests_ordinal(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    const Estimates& est,
                    OrdinalWeightKind weights);

post_expected<nt::infer::ScoreTestTable>
modification_indices_mixed_ordinal(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::MixedOrdinalStats& stats,
                                   const Estimates& est,
                                   OrdinalWeightKind weights);

post_expected<nt::infer::ScoreTestTable>
score_tests_mixed_ordinal(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          const Estimates& est,
                          OrdinalWeightKind weights);

template <optim::LsBoundedOptimizer O = optim::LbfgsBOptimizer>
fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    O optimizer = {},
                    spec::Starts starts = {});

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    optim::LbfgsBOptimizer optimizer,
                    spec::Starts starts = {});

template <optim::LsBoundedOptimizer O = optim::LbfgsBOptimizer>
fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          O optimizer = {},
                          spec::Starts starts = {});

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          optim::LbfgsBOptimizer optimizer,
                          spec::Starts starts = {});

extern template fit_expected<Estimates>
fit_ordinal_bounded<optim::LbfgsBOptimizer>(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    Bounds bounds,
    OrdinalWeightKind weights,
    optim::LbfgsBOptimizer optimizer,
    spec::Starts starts);

extern template fit_expected<Estimates>
fit_mixed_ordinal_bounded<optim::LbfgsBOptimizer>(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    Bounds bounds,
    OrdinalWeightKind weights,
    optim::LbfgsBOptimizer optimizer,
    spec::Starts starts);

}  // namespace magmaan::estimate
