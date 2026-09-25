#pragma once

// magmaan's inference policy: the one set of choices the ordinary-user
// package applies by default (project/design/r-interface-vision.md). Every
// component is either computed under the policy or carries a reason; no
// component is ever replaced by a result under another convention.

#include <limits>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/robust/prepared_ntml.hpp"

namespace magmaan::api {

enum class InferenceReason {
  Available,
  NotConverged,      // the fit did not pass its convergence verdict
  PsdBoundary,       // PSD-constrained fit on the cone boundary: nonregular
  Saturated,         // zero degrees of freedom: no global test exists
  UnsupportedModel,  // outside the policy's model class (e.g. fixed x)
  NumericFailure,    // singular information or a failed decomposition
};

std::string_view reason_name(InferenceReason reason) noexcept;

// The fit-level facts that gate inference. `policy_fit_state()` derives them
// from the estimates; bindings that rebuild estimates without their
// diagnostics supply them directly.
struct PolicyFitState {
  bool converged = true;
  bool psd_boundary = false;
};

PolicyFitState policy_fit_state(const estimate::Estimates& estimates);

struct PolicyGlobalTest {
  InferenceReason reason = InferenceReason::Available;
  std::string detail;
  double statistic = std::numeric_limits<double>::quiet_NaN();
  int df = 0;
  // Mean eigenvalue of the UGamma spectrum: the SB scaling divisor.
  double sb_scale = std::numeric_limits<double>::quiet_NaN();
  double p_sb = std::numeric_limits<double>::quiet_NaN();
  double p_peba4 = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd eigenvalues;  // ascending, length df
};

struct PolicyInference {
  InferenceReason covariance_reason = InferenceReason::Available;
  std::string covariance_detail;
  Eigen::MatrixXd covariance;  // free parameters, total-sample scale
  PolicyGlobalTest score, lr;
};

// All components unavailable for one reason.
PolicyInference policy_unavailable(InferenceReason reason, std::string detail);

// Complete-data normal-theory ML.
//
// Parameter covariance: the sandwich V (sum_i s_i s_i') V with V the inverse
// observed information and s_i the exact casewise likelihood scores at the
// fitted point (robust::frontier::ntml_score_sandwich).
//
// Global tests against the saturated model: the score statistic and the
// likelihood-ratio statistic, each calibrated with SB (mean scaling) and
// PEBA4 from the UGamma spectrum. Both use the shared expected-information
// geometry; under the global null the observed and expected information
// differ by O_p(n^-1/2), and the SB and PEBA4 evidence behind the policy is
// for this geometry.
PolicyInference policy_inference_ml(robust::frontier::NTMLFit& fit,
                                    const PolicyFitState& state);

}  // namespace magmaan::api
