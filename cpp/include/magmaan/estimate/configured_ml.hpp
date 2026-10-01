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

// Both search routes reuse the ordinary ML discrepancy and finalization.
// `explicit_start` supplies values rather than selecting a constructor.
fit_expected<Estimates> fit_ml_configured(
    spec::LatentStructure, const model::MatrixRep&, const SampleStats&,
    const FittingOptions&, const spec::Starts& = {},
    const Eigen::VectorXd& explicit_start = {}, Bounds = {});

} // namespace magmaan::estimate
