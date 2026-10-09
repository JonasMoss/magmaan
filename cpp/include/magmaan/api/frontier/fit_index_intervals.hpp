#pragma once

#include "magmaan/api/policy.hpp"

namespace magmaan::api::frontier {
// Lab candidate: V1/V2 are discrepancy-scale variances; incremental index
// intervals use the joint user/baseline likelihood contrast at first order.
struct FitIndexInterval {
  std::string index, variant;
  double estimate = std::numeric_limits<double>::quiet_NaN();
  double lower = std::numeric_limits<double>::quiet_NaN();
  double upper = std::numeric_limits<double>::quiet_NaN();
  double v1 = std::numeric_limits<double>::quiet_NaN();
  double v2 = std::numeric_limits<double>::quiet_NaN();
  InferenceReason reason = InferenceReason::Available;
  std::string detail;
};
struct FitIndexIntervals {
  std::vector<FitIndexInterval> indices;
  PolicyFitMeasures points;
  double casewise_discrepancy = std::numeric_limits<double>::quiet_NaN();
  double quadratic_trace = std::numeric_limits<double>::quiet_NaN();
  double quadratic_squared_trace = std::numeric_limits<double>::quiet_NaN();
};
FitIndexIntervals fit_index_intervals_misspec_unavailable(InferenceReason reason,
    std::string detail);
FitIndexIntervals fit_index_intervals_misspec(const Fit& fit, double level = 0.90);
FitIndexIntervals fit_index_intervals_misspec(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::Estimates& est, const PolicyFitState& state,
    bool fiml = false, double level = 0.90);
} // namespace magmaan::api::frontier
