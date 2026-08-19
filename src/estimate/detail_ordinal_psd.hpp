#pragma once

#include <functional>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate::detail_ordinal {

struct TransformedOrdinalEvaluation {
  model::Evaluation evaluation;
  Eigen::VectorXd theta;
  Eigen::MatrixXd J_theta;
};

using TransformedOrdinalEvaluationFn =
    std::function<fit_expected<TransformedOrdinalEvaluation>(
        const Eigen::VectorXd&, bool)>;

// Build the production ordinal LS residual/Jacobian callbacks when the
// optimizer coordinates x are not the ordinary prepared partable coordinates.
// `evaluate` supplies implied moments and both d(moment)/dx and d(theta)/dx;
// `expand` maps x back to the ordinary parameter vector returned to callers.
fit_expected<optim::GmmProblem>
ordinal_ls_problem_transformed(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    const Eigen::VectorXd& x0,
    Eigen::Index n_param,
    OrdinalWeightKind weights,
    OrdinalParameterization parameterization,
    TransformedOrdinalEvaluationFn evaluate,
    optim::ExpandFn expand);

// Mixed continuous/ordinal counterpart over thresholds, continuous means and
// variances, and pairwise covariance/polyserial/polychoric associations.
fit_expected<optim::GmmProblem>
mixed_ordinal_ls_problem_transformed(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Eigen::VectorXd& x0,
    Eigen::Index n_param,
    OrdinalWeightKind weights,
    OrdinalParameterization parameterization,
    TransformedOrdinalEvaluationFn evaluate,
    optim::ExpandFn expand);

}  // namespace magmaan::estimate::detail_ordinal
