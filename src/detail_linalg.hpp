#pragma once

// Shared, error-model-agnostic linear-algebra helpers for the fit/data
// translation units. Kept here (a src/-root private header, like detail_vech.hpp)
// so both data/ and estimate/gmm/ can include it with a relative path and then
// wrap the result in their own error type (PostError vs FitError).

#include <Eigen/Core>

namespace magmaan::detail {

struct SymmetricGateResult {
  bool            ok         = false;
  bool            finite     = false;  // input was finite and square
  bool            decomposed = false;  // eigensolver succeeded, eigenvalues finite
  Eigen::Index    dim        = 0;
  Eigen::Index    rank       = 0;      // count of eigenvalues > tol
  double          tol        = 0.0;    // 1e-10 * max(1, max_eval)
  double          min_eval   = 0.0;
  double          max_eval   = 0.0;
  double          rcond      = 0.0;    // min_eval / max_eval (0 if max_eval <= 0)
};

// Result of a gated symmetric positive-definite inverse. `ok == false` means the
// inverse must NOT be used; the diagnostic fields explain why (non-finite/non-
// square input, eigensolver failure, or rank deficiency). The same eigen-floor
// gate is used for every fourth-moment weight inversion in magmaan (continuous
// ADF/WLS Γ̂, structured Γ, ordinal NACOV / A11) so the policy lives in one place.
struct SymInverseResult : SymmetricGateResult {
  Eigen::MatrixXd inverse;             // populated only when ok == true
};

// Symmetric finite/eigendecomposition gate on 0.5*(A + Aᵀ).  `PD` requires
// λ_min > tol; `PSD` allows rank deficiency but rejects λ_min < -tol.  These
// helpers do not invert, so callers can guard generalized eigensolver and square
// root preconditions without constructing an unnecessary inverse.
SymmetricGateResult symmetric_pd_gated(const Eigen::MatrixXd& A);
SymmetricGateResult symmetric_psd_gated(const Eigen::MatrixXd& A);

// Symmetric PD inverse via SelfAdjointEigenSolver on 0.5*(A + Aᵀ), gated on the
// smallest eigenvalue. The gate is `min_eval > tol` with
// `tol = 1e-10 * max(1, max_eval)` (equivalently rcond ≳ 1e-10 once max_eval ≥ 1):
// matrices that are PD only to working precision (e.g. an empirical fourth-moment
// Γ̂ with a structurally degenerate direction at ~1e-17) are rejected rather than
// inverted into a ~1e16 eigendirection. No exceptions; failures are reported via
// the returned struct so each caller can map them onto FitError/PostError.
SymInverseResult symmetric_inverse_pd_gated(const Eigen::MatrixXd& A);

}  // namespace magmaan::detail
