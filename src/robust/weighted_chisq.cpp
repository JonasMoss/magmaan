#include "magmaan/robust/weighted_chisq.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Core>

namespace magmaan::robust {


namespace {

constexpr double pi = 3.14159265358979323846;

// QUADPACK's dqagie takes a bare `double f(double*)` integrand (no user-data
// pointer), so the Imhof parameters are threaded through thread-local state:
// imhof_upper sets them, calls dqagie synchronously, then clears. thread_local
// keeps it re-entrant across parallel callers.
thread_local const Eigen::VectorXd* tls_lambda = nullptr;
thread_local double                 tls_x      = 0.0;

// Imhof (1961) integrand f(u) = sin θ(u) / (u · ρ(u)) on [0, ∞), with
//   θ(u) = ½ Σⱼ atan(λⱼ u) − ½ x u ,   ρ(u) = Πⱼ (1 + λⱼ²u²)^{1/4} .
// dqk15i maps [0,∞) onto (0,1) and supplies the Jacobian; we provide only the
// untransformed integrand. The u=0 limit (sin θ ~ ½(Σλ − x)u, ρ(0)=1) is kept
// for safety though dqk15i's interior Gauss-Kronrod nodes never sample it.
extern "C" double imhof_integrand(double* up) noexcept {
  const Eigen::VectorXd& lam = *tls_lambda;
  const double u = *up;
  if (u == 0.0) return 0.5 * (lam.sum() - tls_x);
  double theta    = -0.5 * tls_x * u;
  double log_rho4 = 0.0;  // = Σⱼ log(1 + λⱼ²u²)
  for (Eigen::Index j = 0; j < lam.size(); ++j) {
    const double lu = lam(j) * u;
    theta    += 0.5 * std::atan(lu);
    log_rho4 += std::log1p(lu * lu);
  }
  return std::sin(theta) / (u * std::exp(0.25 * log_rho4));
}

// Direct (non-thread-local) integrand evaluation for the Simpson fallback.
double imhof_f(const Eigen::VectorXd& lam, double x, double u) noexcept {
  if (u == 0.0) return 0.5 * (lam.sum() - x);
  double theta    = -0.5 * x * u;
  double log_rho4 = 0.0;
  for (Eigen::Index j = 0; j < lam.size(); ++j) {
    const double lu = lam(j) * u;
    theta    += 0.5 * std::atan(lu);
    log_rho4 += std::log1p(lu * lu);
  }
  return std::sin(theta) / (u * std::exp(0.25 * log_rho4));
}

// Composite Simpson on [a, b] with N (even) intervals.
double composite_simpson(const Eigen::VectorXd& lam, double x,
                         double a, double b, long N) {
  if (N < 2)      N = 2;
  if (N % 2 != 0) N += 1;
  const double h = (b - a) / static_cast<double>(N);
  double sum_odd = 0.0, sum_even = 0.0;
  for (long i = 1; i < N; ++i) {
    const double u = a + static_cast<double>(i) * h;
    if (i % 2) sum_odd  += imhof_f(lam, x, u);
    else       sum_even += imhof_f(lam, x, u);
  }
  return (h / 3.0) * (imhof_f(lam, x, a) + 4.0 * sum_odd + 2.0 * sum_even +
                      imhof_f(lam, x, b));
}

// Dense composite-Simpson Imhof tail — the pre-QUADPACK method. Truncates the
// [0,U] range with U from the tail-decay bound and Richardson-refines. Robust
// for the weakly-damped small-df oscillatory tail where qagi's epsilon
// extrapolation breaks down; used only as a fallback (it is the slow path qagi
// replaced for the common df >> 1 case).
double imhof_simpson(const Eigen::VectorXd& lam, double x,
                     double rel_tol, int max_doublings) {
  const double k = static_cast<double>(lam.size());
  double prod_sqrt = 1.0;
  for (Eigen::Index j = 0; j < lam.size(); ++j) {
    prod_sqrt *= std::sqrt(std::max(lam(j), 1e-300));
  }
  const double U_bound =
      std::pow(2.0 / (pi * x * rel_tol * prod_sqrt), 2.0 / (k + 2.0));
  const double U = std::clamp(U_bound, 20.0, 1.0e4);
  const double max_freq = 0.5 * (x + lam.sum());
  const double h_target = pi / (4.0 * std::max(max_freq, 1.0));
  long N0 = static_cast<long>(std::ceil(U / h_target));
  if (N0 < 256)        N0 = 256;
  if (N0 % 2)          N0 += 1;
  if (N0 > (1L << 14)) N0 = (1L << 14);
  double I_prev = composite_simpson(lam, x, 0.0, U, N0);
  long   N      = N0;
  for (int it = 0; it < max_doublings; ++it) {
    N *= 2;
    const double I    = composite_simpson(lam, x, 0.0, U, N);
    const double diff = std::abs(I - I_prev);
    const double tol  = std::max(rel_tol, 1e-13 * std::max(std::abs(I), 1.0));
    I_prev = I;
    if (diff < tol) break;
  }
  return 0.5 + I_prev / pi;
}

}  // namespace

// QUADPACK qagi adaptive Gauss-Kronrod integrator over a (semi-)infinite
// interval (public domain; third_party/quadpack). ABI mirrors
// third_party/quadpack/quadpack.h (f2c `integer`=int, `doublereal`=double).
extern "C" {
int dqagie_(double (*f)(double*), double* bound, int* inf, double* epsabs,
            double* epsrel, int* limit, double* result, double* abserr,
            int* neval, int* ier, double* alist, double* blist, double* rlist,
            double* elist, int* iord, int* last);
}

double imhof_upper(const Eigen::Ref<const Eigen::VectorXd>& lambda,
                   double x,
                   double rel_tol,
                   int    max_doublings) {
  // Degenerate inputs.
  if (lambda.size() == 0) return (x < 0.0) ? 1.0 : 0.0;
  if (!std::isfinite(x)) return (x > 0.0) ? 0.0 : 1.0;
  if (x <= 0.0)          return 1.0;

  // Snapshot λ as an owning vector so the Integrand can hold a const& without
  // worrying about expression-template lifetimes; also lets us replace tiny
  // negatives (numerical noise from a generalised eig) with 0.
  Eigen::VectorXd lam = lambda;
  for (Eigen::Index j = 0; j < lam.size(); ++j) {
    if (!(lam(j) > 0.0)) lam(j) = 0.0;
  }
  const double lmax = lam.maxCoeff();
  if (lmax <= 0.0) return 0.0;

  // ∫₀^∞ f(u) du via QUADPACK qagi (bound = 0, inf = 1). dqk15i performs the
  // [0,∞) transform; the adaptive driver bisects the worst panel and
  // epsilon-extrapolates, concentrating evaluations where the oscillatory
  // integrand actually lives — far cheaper than a fixed grid as df grows.
  tls_lambda = &lam;
  tls_x      = x;

  constexpr int limit = 256;          // max subintervals; the integrand is smooth
  double bound  = 0.0;
  int    inf    = 1;
  double epsabs = std::max(rel_tol, 1e-10);
  double epsrel = std::max(rel_tol, 1e-10);
  int    lim    = limit;
  double result = 0.0, abserr = 0.0;
  int    neval  = 0, ier = 0, last = 0;
  std::array<double, limit * 4> work{};
  std::array<int, limit>        iord{};
  dqagie_(&imhof_integrand, &bound, &inf, &epsabs, &epsrel, &lim, &result,
          &abserr, &neval, &ier, work.data(), work.data() + limit,
          work.data() + 2 * limit, work.data() + 3 * limit, iord.data(), &last);

  tls_lambda = nullptr;

  // qagi is accurate and fast when ρ damps the oscillation (moderate-to-large
  // df, the common SEM case): trust it. But for the weakly-damped small-df tail
  // (df ~ 1-2, large x) its epsilon extrapolation is unreliable — it reports
  // ier != 0 or an abserr that swamps the true (tiny) probability. Fall back to
  // the dense Simpson integral there. The branch is rare; df >> 1 takes the
  // fast path.
  const double p = (ier == 0 && abserr <= 1e-6)
                       ? 0.5 + result / pi
                       : imhof_simpson(lam, x, rel_tol, max_doublings);
  return std::clamp(p, 0.0, 1.0);
}

}  // namespace magmaan::robust
