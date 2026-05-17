#include "magmaan/robust/weighted_chisq.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace magmaan::nt::robust {


namespace {

constexpr double pi = 3.14159265358979323846;

// Imhof integrand f(u) = sin θ(u) / (u · ρ(u)) with the u=0 limit handled
// analytically (sin θ(u) ~ ½(Σλⱼ − x)·u as u→0 and ρ(0) = 1).
struct Integrand {
  const Eigen::VectorXd& lambda;
  double x;

  double operator()(double u) const noexcept {
    if (u == 0.0) {
      return 0.5 * (lambda.sum() - x);
    }
    double theta    = -0.5 * x * u;
    double log_rho4 = 0.0;  // = Σⱼ log(1 + λⱼ²u²)
    for (Eigen::Index j = 0; j < lambda.size(); ++j) {
      const double lu = lambda(j) * u;
      theta    += 0.5 * std::atan(lu);
      log_rho4 += std::log1p(lu * lu);
    }
    const double rho = std::exp(0.25 * log_rho4);
    return std::sin(theta) / (u * rho);
  }
};

// Composite Simpson on [a, b] with N (even) intervals. Returns h/3 · (f(a) +
// 4·Σ_odd + 2·Σ_even + f(b)). Wraps `Integrand` so we can hold k-eigenvalue
// state across function evaluations without re-allocation.
double composite_simpson(const Integrand& f, double a, double b, long N) {
  if (N < 2)       N = 2;
  if (N % 2 != 0)  N += 1;
  const double h = (b - a) / static_cast<double>(N);
  double sum_odd  = 0.0;
  double sum_even = 0.0;
  for (long i = 1; i < N; ++i) {
    const double u = a + static_cast<double>(i) * h;
    if (i % 2) sum_odd  += f(u);
    else       sum_even += f(u);
  }
  return (h / 3.0) * (f(a) + 4.0 * sum_odd + 2.0 * sum_even + f(b));
}

}  // namespace

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

  Integrand f{lam, x};

  // Truncation U.  At large u the integrand magnitude decays at least as
  //   |f(u)| ≤ 1 / (u · ρ(u))  ≤  1 / (u^{1 + k/2} · Πⱼ √λⱼ)
  // and integration-by-parts against the oscillating sin θ further damps the
  // *integral* tail to ~2 / (π · x · U^{1 + k/2} · Πⱼ √λⱼ).  Solve that ≤ ε:
  //   U  ≥  (2 / (π · x · ε · Πⱼ √λⱼ))^{2 / (k + 2)} .
  const double k = static_cast<double>(lam.size());
  double prod_sqrt = 1.0;
  for (Eigen::Index j = 0; j < lam.size(); ++j) {
    prod_sqrt *= std::sqrt(std::max(lam(j), 1e-300));
  }
  const double U_bound =
      std::pow(2.0 / (pi * x * rel_tol * prod_sqrt), 2.0 / (k + 2.0));
  // Conservative ceiling: even with very loose λ / small x the Imhof
  // integrand's oscillation kills the integral well before U = 10⁴.  Going
  // beyond is wasted under-ASAN work that can dominate the test runtime.
  const double U = std::clamp(U_bound, 20.0, 1.0e4);

  // Initial number of Simpson intervals.  The integrand's dominant frequency
  // is bounded by θ'(u) ∈ [−x/2, ½ Σλⱼ], i.e. |θ'(u)| ≤ ½(x + Σλⱼ).  We aim
  // for ~8 Simpson samples per radian initially (step h ≲ π/(4·max_freq))
  // and let the Richardson loop below refine — that keeps the first
  // composite-Simpson call cheap.
  const double max_freq = 0.5 * (x + lam.sum());
  const double h_target = pi / (4.0 * std::max(max_freq, 1.0));
  long N0 = static_cast<long>(std::ceil(U / h_target));
  if (N0 < 256)       N0 = 256;
  if (N0 % 2)         N0 += 1;
  if (N0 > (1L << 14)) N0 = (1L << 14);   // initial cap = 16 384

  // Richardson-style refinement: double N until two consecutive Simpson
  // estimates agree to `rel_tol` or we hit `max_doublings`.
  //
  // Round-off floor: with composite Simpson at large N the integrand sum
  // accumulates round-off ≈ √N · eps_machine, so `diff` cannot decrease past
  // ≈ 1e-13 · max(|I|, 1).  Without this guard the convergence test would
  // never fire and we'd burn through the doubling budget for no gain.
  double I_prev = composite_simpson(f, 0.0, U, N0);
  long   N      = N0;
  for (int it = 0; it < max_doublings; ++it) {
    N *= 2;
    const double I    = composite_simpson(f, 0.0, U, N);
    const double diff = std::abs(I - I_prev);
    const double tol  = std::max(rel_tol, 1e-13 * std::max(std::abs(I), 1.0));
    I_prev = I;
    if (diff < tol) break;
  }

  const double p = 0.5 + I_prev / pi;
  return std::clamp(p, 0.0, 1.0);
}

}  // namespace magmaan::nt::robust
