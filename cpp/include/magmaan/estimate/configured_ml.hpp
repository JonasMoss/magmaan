#pragma once

#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/fitting_options.hpp"
#include "magmaan/estimate/start_pipeline.hpp"

namespace magmaan::estimate {

// Single-level complete continuous ML. General affine/nonlinear constraints,
// composites and categorical/missing-data sources need their own parity gate.
// User hints are in the model's target identification and original units.
fit_expected<Eigen::VectorXd> lavaan_ml_start_values(
    const spec::LatentStructure&, const model::MatrixRep&,
    const SampleStats&, const spec::Starts& = {}, bool simple = false);

// lavaan's acceptance gradient at `theta`: the largest absolute gradient of
// the unnormalized ML objective (½F on `sample`) in optimizer coordinates
// z = scale·θ (unit scale when empty), skipping coordinates at a bound.
// Infinite when the objective or a gradient entry is not finite.
fit_expected<double> lavaan_acceptance_gradient(
    spec::LatentStructure, const model::MatrixRep&, const SampleStats&,
    const Eigen::VectorXd& theta, const Bounds& = {},
    const Eigen::VectorXd& scale = {});

// Both search routes reuse the ordinary ML discrepancy and finalization.
// `explicit_start` supplies values rather than selecting a constructor.
fit_expected<Estimates> fit_ml_configured(
    spec::LatentStructure, const model::MatrixRep&, const SampleStats&,
    const FittingOptions&, const spec::Starts& = {},
    const Eigen::VectorXd& explicit_start = {}, Bounds = {});

} // namespace magmaan::estimate
