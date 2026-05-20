#include "magmaan/optim/port_optimizer.hpp"

#ifdef MAGMAAN_WITH_PORT

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

// PORT entry points exposed by the vendored third_party/port/ library.
// f2c-translated Fortran: every parameter is a pointer (Fortran's
// pass-by-reference convention), arrays are conceptually 1-based inside the
// routine (f2c inserts `--iv` at function entry to make Fortran's 1-based
// indexing line up with the C array we pass). From the caller's perspective,
// that means our `iv[0]` is what the Fortran source calls `IV(1)`, our `v[0]`
// is `V(1)`, and so on — the offset stays internal to PORT.
//
// The routines below are tagged `extern "C"`, which f2c also requires so it
// can link against pure C call sites.
extern "C" {
int drmngb_(double *b, double *d, double *fx, double *g,
            int *iv, int *liv, int *lv, int *n,
            double *v, double *x);
int divset_(int *alg, int *iv, int *liv, int *lv, double *v);
}

namespace magmaan::optim {

namespace {

FitError make_err(FitError::Kind k, std::string detail,
                  int iter = 0, double fval = 0.0) {
  return FitError{k, std::move(detail), iter, fval};
}

// PORT's IV(1) status codes, named per the docstring at
// third_party/port/cport/drmngb.c:108–123 and the dispatch in lines 325–719.
//
// In the source, IV(1) is the slot the caller reads after each `drmngb_`
// return. After f2c's --iv adjustment, that maps to our `iv[0]`. We never
// touch the indexing math from C; just consult `iv[0]` post-return and
// match against these named constants.
enum PortStatus : int {
  PORT_REQUEST_F  = 1,   // caller compute F(x); fill *fx
  PORT_REQUEST_G  = 2,   // caller compute G(x); fill g
  PORT_OK_XCONV   = 3,
  PORT_OK_RFCONV  = 4,
  PORT_OK_BOTH    = 5,
  PORT_OK_AFCONV  = 6,
  PORT_OK_SINGCONV = 7,
  PORT_FAIL_NOISY = 8,
  PORT_FAIL_FALSE = 9,
  PORT_FAIL_BUDGET = 10,
};

// PORT IV / V subscript layout (Fortran 1-based, mapped to C 0-based).
// Kept as small named constants so the configure step below is readable; no
// dispatch logic depends on them directly.
constexpr int kIv_Status   = 0;   // IV(1)  — status code (R/W)
constexpr int kIv_TooBig   = 1;   // IV(2)  — caller sets to 1 to reject F
constexpr int kIv_MxFCal   = 16;  // IV(17) — max function evaluations
constexpr int kIv_MxIter   = 17;  // IV(18) — max iterations
constexpr int kIv_NIter    = 30;  // IV(31) — iteration count (R)
constexpr int kV_F         = 9;   // V(10)  — current function value (R)
// IV(6)=NFCALL and IV(30)=NGCALL are also available post-fit for callers
// that want to surface eval counts separately from iterations; the present
// LbfgsOutput shape carries only `iterations`, so we don't expose them yet.

// PORT requires bounds; "unbounded" means a sentinel near double limits.
// 1e308 is well inside the dynamic range and matches the convention used in
// the AMPL/ASL test harnesses; choosing infinity would risk overflow in
// PORT's interior arithmetic when it forms (upper - lower).
constexpr double kPortInf = 1e308;

}  // namespace

fit_expected<LbfgsOutput>
PortOptimizer::minimize(Objective f,
                        const Eigen::VectorXd& x0,
                        const Eigen::VectorXd& lower,
                        const Eigen::VectorXd& upper) const {
  if (x0.size() == 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "PortOptimizer: empty parameter vector"));
  }
  if (lower.size() != x0.size() || upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "PortOptimizer: bounds size mismatch (lower=" +
            std::to_string(lower.size()) +
            ", upper=" + std::to_string(upper.size()) +
            ", x0=" + std::to_string(x0.size()) + ")"));
  }

  const int    n   = static_cast<int>(x0.size());
  const auto   nu  = static_cast<std::size_t>(n);
  // PORT documents these minimum work-array sizes in drmngb.c:88-91.
  const int liv = 59 + n;
  const int lv  = 71 + n * (n + 19) / 2;

  std::vector<int>    iv(static_cast<std::size_t>(liv), 0);
  std::vector<double> v(static_cast<std::size_t>(lv), 0.0);
  // Scale vector — identity. PORT recommends scale ≈ inverse magnitude of
  // each x component; identity is the standard default and matches what
  // R's `nlminb` does when the caller leaves `scale = 1`.
  std::vector<double> d(nu, 1.0);
  std::vector<double> g(nu, 0.0);
  // Bounds are stored as interleaved (lower, upper) pairs in PORT's
  // documented layout (drmngb.c:84 "b.... VECTOR OF LOWER AND UPPER
  // BOUNDS ON X."). Translate ±infinity sentinels to ±kPortInf so PORT's
  // internal (upper - lower) arithmetic never sees NaN or Inf.
  std::vector<double> b(2 * nu);
  for (std::size_t i = 0; i < nu; ++i) {
    const double lo = std::isfinite(lower[static_cast<Eigen::Index>(i)])
                          ? lower[static_cast<Eigen::Index>(i)]
                          : -kPortInf;
    const double up = std::isfinite(upper[static_cast<Eigen::Index>(i)])
                          ? upper[static_cast<Eigen::Index>(i)]
                          : +kPortInf;
    b[2 * i]     = lo;
    b[2 * i + 1] = up;
  }

  // Working copy of x; PORT updates it in place.
  Eigen::VectorXd x  = x0;
  double          fx = std::numeric_limits<double>::quiet_NaN();

  // Initialise IV / V to PORT defaults. `alg = 2` selects the general
  // unconstrained-minimization defaults (SUMSL/HUMSL); `alg = 1` would
  // pick NL2SOL's defaults, which is what we'll use in Phase 2's
  // `port_nls` adapter.
  int alg = 2;
  int liv_arg = liv;
  int lv_arg  = lv;
  divset_(&alg, iv.data(), &liv_arg, &lv_arg, v.data());

  // Forward our caller-supplied tuning. PORT's other tolerance defaults
  // (AFCTOL, RFCTOL, XCTOL) match the spirit of LBFGS's defaults closely
  // enough that we leave them; the iteration cap is what callers most
  // commonly tighten.
  iv[kIv_MxIter] = opts_.max_iter;
  iv[kIv_MxFCal] = opts_.max_iter * 10;  // generous; PORT lifts this only if max_iter is otherwise binding

  // Reverse-communication loop. PORT sets `iv[0]` to PORT_REQUEST_F or
  // PORT_REQUEST_G, we compute the requested quantity, and call drmngb_
  // again. Termination is signalled by `iv[0] >= 3`.
  //
  // Our magmaan Objective always supplies both value and gradient in one
  // pass, so on a PORT_REQUEST_G we re-call f (cheaply, since the model
  // evaluator caches recent x in practice) to extract the gradient.
  // (PORT's design assumes value and gradient might be expensive separately;
  // forcing both per inner request is a micro-pessimisation we accept in
  // exchange for one Objective contract across all adapters.)
  int n_arg = n;
  Eigen::VectorXd grad_buf(n);
  // Guard against infinite reverse-communication loops if PORT somehow
  // never signals termination. Each request consumes at most one
  // evaluation, so 100x the documented evaluation cap is comfortably
  // beyond any reasonable horizon.
  const int budget_safeguard = 100 * (opts_.max_iter + 1) * 10;
  int loop_counter = 0;

  while (true) {
    if (++loop_counter > budget_safeguard) {
      return std::unexpected(make_err(FitError::Kind::OptimizerNonConvergence,
          "PortOptimizer: reverse-communication loop ran past safeguard limit "
          "(" + std::to_string(budget_safeguard) + " requests); PORT did not "
          "signal termination",
          iv[kIv_NIter], fx));
    }

    drmngb_(b.data(), d.data(), &fx, g.data(),
            iv.data(), &liv_arg, &lv_arg, &n_arg,
            v.data(), x.data());

    const int status = iv[kIv_Status];

    if (status == PORT_REQUEST_F) {
      grad_buf.setZero();
      const double val = f(x, grad_buf);
      if (!std::isfinite(val)) {
        // Tell PORT to back off and try a smaller step. PORT clears this
        // flag on the next call once it has accepted/rejected the step.
        iv[kIv_TooBig] = 1;
      } else {
        fx = val;
        iv[kIv_TooBig] = 0;
      }
      continue;
    }

    if (status == PORT_REQUEST_G) {
      grad_buf.setZero();
      const double val = f(x, grad_buf);
      // We've already filled `fx` from the previous PORT_REQUEST_F; PORT
      // is asking for the gradient at the same point, so we discard the
      // returned value (it agrees with `fx`).
      (void)val;
      for (std::size_t i = 0; i < nu; ++i) {
        g[i] = grad_buf[static_cast<Eigen::Index>(i)];
      }
      continue;
    }

    // status >= 3: PORT has terminated; map the code to a magmaan outcome.
    break;
  }

  const int    final_status = iv[kIv_Status];
  const int    n_iter       = iv[kIv_NIter];
  const double f_final      = v[kV_F];  // PORT's recorded final function value

  switch (final_status) {
    case PORT_OK_XCONV:
    case PORT_OK_RFCONV:
    case PORT_OK_BOTH:
    case PORT_OK_AFCONV:
    case PORT_OK_SINGCONV:
      break;
    case PORT_FAIL_NOISY:
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "PORT drmngb: noisy objective detected (PORT IV(1)=8) — the model "
          "evaluator's gradient is inconsistent with its function value to "
          "more than PORT's noise tolerance",
          n_iter, f_final));
    case PORT_FAIL_FALSE:
      return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
          "PORT drmngb: false convergence (PORT IV(1)=9) — the step length "
          "fell below the precision PORT can resolve",
          n_iter, f_final));
    case PORT_FAIL_BUDGET:
      return std::unexpected(make_err(FitError::Kind::OptimizerNonConvergence,
          "PORT drmngb: iteration or evaluation budget exhausted "
          "(PORT IV(1)=10, max_iter=" + std::to_string(opts_.max_iter) + ")",
          n_iter, f_final));
    default:
      return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
          "PORT drmngb: solver failure (PORT IV(1)=" +
              std::to_string(final_status) + ")",
          n_iter, f_final));
  }

  if (!std::isfinite(f_final)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "PORT drmngb: final objective value is non-finite",
        n_iter, f_final));
  }

  return LbfgsOutput{std::move(x), f_final, n_iter};
}

fit_expected<LbfgsOutput>
PortOptimizer::minimize(Objective f, const Eigen::VectorXd& x0) const {
  const double inf = std::numeric_limits<double>::infinity();
  return minimize(std::move(f), x0,
                  Eigen::VectorXd::Constant(x0.size(), -inf),
                  Eigen::VectorXd::Constant(x0.size(),  inf));
}

}  // namespace magmaan::optim

#endif  // MAGMAAN_WITH_PORT
