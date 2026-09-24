#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate::frontier {

// Empirical reduced-bias M-estimation after Kosmidis and Lunardon, exposed as
// frontier SEM machinery. Moment-quadratic paths use the same complete
// infinitesimal-jackknife estimating equations as the misspecification-robust
// sandwich code, including the leading influence of data-dependent weights.
struct RBMOptions {
  double fd_rel_step = 1e-4;
  double fd_abs_step = 1e-6;
#ifdef MAGMAAN_WITH_PORT
  Backend backend = Backend::Port;
#else
  Backend backend = Backend::NloptLbfgs;
#endif
  optim::OptimOptions optim = {};
  bool check_admissibility = true;
  double admissibility_tol = 1e-6;
};

struct RBMResult {
  Estimates estimates;

  Eigen::VectorXd correction;   // theta_rbm - theta_start, full theta space
  Eigen::VectorXd adjustment;   // tangent-space grad P(theta), full theta size
  Eigen::MatrixXd information;  // j(theta): base theta for explicit, final for implicit
  Eigen::MatrixXd meat;         // e(theta): base theta for explicit, final for implicit
  Eigen::MatrixXd information_reduced;  // K' j(theta) K, or IJ reduced bread
  Eigen::MatrixXd meat_reduced;         // K-reduced score / estimating meat

  double trace_term = 0.0;      // trace(j^-1 e)
  double penalty = 0.0;         // P(theta) = -0.5 * trace_term
  double penalty_per_observation = 0.0;
  double penalized_fmin = 0.0;  // original fmin + trace_term / (2N)

  bool admissible = true;
  bool bounds_satisfied = true;
  bool sigma_pd = true;
  std::vector<std::string> warnings;
};

fit_expected<RBMResult>
rbm_explicit_ml(spec::LatentStructure pt,
                const model::MatrixRep& rep,
                const data::SampleStats& samp,
                const data::RawData& raw,
                const Estimates& base,
                Bounds bounds = {},
                RBMOptions opts = {});

fit_expected<RBMResult>
rbm_implicit_ml(spec::LatentStructure pt,
                const model::MatrixRep& rep,
                const data::SampleStats& samp,
                const data::RawData& raw,
                const Estimates& start,
                Bounds bounds = {},
                RBMOptions opts = {});

fit_expected<RBMResult>
rbm_explicit_fiml(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::RawData& raw,
                  const fiml::FIMLPack& pack,
                  const Estimates& base,
                  Bounds bounds = {},
                  RBMOptions opts = {});

fit_expected<RBMResult>
rbm_explicit_fiml(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::RawData& raw,
                  const Estimates& base,
                  Bounds bounds = {},
                  RBMOptions opts = {});

fit_expected<RBMResult>
rbm_implicit_fiml(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::RawData& raw,
                  const fiml::FIMLPack& pack,
                  const Estimates& start,
                  Bounds bounds = {},
                  RBMOptions opts = {});

fit_expected<RBMResult>
rbm_implicit_fiml(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::RawData& raw,
                  const Estimates& start,
                  Bounds bounds = {},
                  RBMOptions opts = {});

fit_expected<RBMResult>
rbm_explicit_continuous_ls(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::SampleStats& samp,
                           const Estimates& base,
                           const gmm::Weight& weight,
                           const data::RawData& raw,
                           ContinuousLsIJWeightMode mode =
                               ContinuousLsIJWeightMode::Fixed,
                           DlsWeightOptions dls_opts = {},
                           Bounds bounds = {},
                           RBMOptions opts = {});

fit_expected<RBMResult>
rbm_implicit_continuous_ls(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::SampleStats& samp,
                           const Estimates& start,
                           const gmm::Weight& weight,
                           const data::RawData& raw,
                           ContinuousLsIJWeightMode mode =
                               ContinuousLsIJWeightMode::Fixed,
                           DlsWeightOptions dls_opts = {},
                           Bounds bounds = {},
                           RBMOptions opts = {});

fit_expected<RBMResult>
rbm_explicit_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     const Estimates& base,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization =
                         OrdinalParameterization::Delta,
                     Bounds bounds = {},
                     RBMOptions opts = {});

fit_expected<RBMResult>
rbm_implicit_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     const Estimates& start,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization =
                         OrdinalParameterization::Delta,
                     Bounds bounds = {},
                     RBMOptions opts = {});

fit_expected<RBMResult>
rbm_explicit_mixed_ordinal(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           const Estimates& base,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization =
                               OrdinalParameterization::Delta,
                           Bounds bounds = {},
                           RBMOptions opts = {});

fit_expected<RBMResult>
rbm_implicit_mixed_ordinal(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           const Estimates& start,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization =
                               OrdinalParameterization::Delta,
                           Bounds bounds = {},
                           RBMOptions opts = {});

fit_expected<RBMResult>
rbm_explicit_two_stage(spec::LatentStructure pt,
                       const model::MatrixRep& rep,
                       const data::RawData& raw,
                       const fiml::FIMLPack& pack,
                       const fiml::FIMLH1& h1,
                       const Estimates& base,
                       fiml::TwoStageWeight weight = fiml::TwoStageWeight::Nt,
                       fiml::TwoStageDlsOptions dls = {},
                       Bounds bounds = {},
                       RBMOptions opts = {});

fit_expected<RBMResult>
rbm_implicit_two_stage(spec::LatentStructure pt,
                       const model::MatrixRep& rep,
                       const data::RawData& raw,
                       const fiml::FIMLPack& pack,
                       const fiml::FIMLH1& h1,
                       const Estimates& start,
                       fiml::TwoStageWeight weight = fiml::TwoStageWeight::Nt,
                       fiml::TwoStageDlsOptions dls = {},
                       Bounds bounds = {},
                       RBMOptions opts = {});

}  // namespace magmaan::estimate::frontier
