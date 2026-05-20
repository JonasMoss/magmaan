#pragma once

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::estimate::frontier {

// Model-implied fourth-order (MI4 / structured ADF) weight builder.
//
// This is an explicit covariance-moment weight builder for complete-data pure
// CFA models. It estimates the source fourth cumulants implied by
//
//   X = Lambda eta + epsilon
//
// with mutually independent factor and uniqueness sources, plugs them into the
// structured Gamma formula, and returns W = Gamma^{-1}. The returned
// `gmm::Weight` is consumed by the ordinary WLS / moment-quadratic path.
fit_expected<gmm::Weight>
structured_gamma_weight(const model::ModelEvaluator& ev,
                        const model::MatrixRep& rep,
                        const data::SampleStats& samp,
                        const data::RawData& raw,
                        const Eigen::VectorXd& theta0);

}  // namespace magmaan::estimate::frontier
