#include <doctest/doctest.h>

#include <random>

#include <Eigen/Core>

#include "magmaan/inference/inference.hpp"        // chi2_pvalue (oracle for closed forms)
#include "magmaan/robust/robust.hpp"


namespace {

Eigen::VectorXd vec(std::initializer_list<double> v) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
  Eigen::Index i = 0;
  for (double x : v) out(i++) = x;
  return out;
}

}  // namespace

// ── Closed-form cross-checks ────────────────────────────────────────────────

TEST_CASE("imhof_upper: single λ collapses to a scaled χ²(1)") {
  // For a single λ, Q = λ·χ²(1), so Pr(Q > x) = Pr(χ²(1) > x/λ).
  for (double lambda : {0.25, 1.0, 4.0, 17.5}) {
    for (double x : {0.05, 0.5, 1.0, 2.5, 7.0, 15.0}) {
      const double p_imhof = magmaan::robust::imhof_upper(vec({lambda}), x);
      const double p_exact = magmaan::inference::chi2_pvalue(x / lambda, 1);
      INFO("λ=", lambda, "  x=", x);
      CHECK(p_imhof == doctest::Approx(p_exact).epsilon(1e-5));
    }
  }
}

TEST_CASE("imhof_upper: k equal-weight λ=1 collapses to χ²(k)") {
  // Σⱼ 1·χ²₁ⱼ = χ²(k).
  for (int k : {2, 3, 5, 8}) {
    Eigen::VectorXd lam = Eigen::VectorXd::Ones(k);
    for (double x : {0.1, 1.0, 3.0, 8.0, 16.0, 30.0}) {
      const double p_imhof = magmaan::robust::imhof_upper(lam, x);
      const double p_exact = magmaan::inference::chi2_pvalue(x, k);
      INFO("k=", k, "  x=", x);
      CHECK(p_imhof == doctest::Approx(p_exact).epsilon(1e-5));
    }
  }
}

TEST_CASE("imhof_upper: k equal-weight λ=c collapses to a scaled χ²(k)") {
  // Σⱼ c·χ²₁ⱼ = c·χ²(k), so Pr(Q > x) = Pr(χ²(k) > x/c).
  const int k = 4;
  for (double c : {0.5, 2.0, 7.3}) {
    Eigen::VectorXd lam = Eigen::VectorXd::Constant(k, c);
    for (double x : {0.5, 3.0, 10.0, 25.0}) {
      const double p_imhof = magmaan::robust::imhof_upper(lam, x);
      const double p_exact = magmaan::inference::chi2_pvalue(x / c, k);
      INFO("c=", c, "  x=", x);
      CHECK(p_imhof == doctest::Approx(p_exact).epsilon(1e-5));
    }
  }
}

// ── Degenerate / edge cases ─────────────────────────────────────────────────

TEST_CASE("imhof_upper: degenerate inputs") {
  // x ≤ 0  ⇒  full mass to the right.
  CHECK(magmaan::robust::imhof_upper(vec({1.0, 2.0}), 0.0)  == doctest::Approx(1.0));
  CHECK(magmaan::robust::imhof_upper(vec({1.0, 2.0}), -3.0) == doctest::Approx(1.0));

  // All λ = 0  ⇒  Q ≡ 0; upper tail above any x > 0 is exactly 0.
  CHECK(magmaan::robust::imhof_upper(vec({0.0, 0.0, 0.0}), 1.0) == doctest::Approx(0.0));
}

// ── Monte-Carlo cross-check on unequal-λ mixtures ───────────────────────────
//
// Closed-form references only exist for single-λ and all-equal-λ.  For mixed
// spectra (the actual nested-test regime — m generalised eigenvalues of `S` /
// `C` are generically distinct) we cross-check against a Monte-Carlo estimate
// of Pr(Σⱼ λⱼ · χ²₁ⱼ > x).  3M samples gives ≈ 6·10⁻⁴ standard error in the
// body of the distribution and the tolerance widens automatically in the tail
// via the Poisson std-err of the count itself.

namespace {

double mc_upper(const Eigen::VectorXd& lambda, double x,
                long n_samples, std::mt19937& rng) {
  std::chi_squared_distribution<double> chi1(1.0);
  long hits = 0;
  for (long s = 0; s < n_samples; ++s) {
    double q = 0.0;
    for (Eigen::Index j = 0; j < lambda.size(); ++j) q += lambda(j) * chi1(rng);
    if (q > x) ++hits;
  }
  return static_cast<double>(hits) / static_cast<double>(n_samples);
}

}  // namespace

TEST_CASE("imhof_upper: Monte-Carlo agreement on mixed-λ spectra") {
  std::mt19937 rng(0xc0ffeeU);
  // 500k draws × 5 cases keeps the test under ~5 s while the 4·σ envelope
  // below (σ ≈ √(p(1−p)/N) ≈ 7·10⁻⁴ at p = 0.5) stays comfortably above
  // Imhof's quadrature error.
  constexpr long N = 500'000;

  struct Case {
    Eigen::VectorXd lambda;
    double x;
  };
  std::vector<Case> cases = {
    {vec({1.0, 0.5}),       1.0},
    {vec({1.0, 0.5}),       3.0},
    {vec({2.0, 1.0, 0.5}),  2.0},
    {vec({2.0, 1.0, 0.5}),  8.0},
    {vec({3.0, 1.0, 0.3, 0.1}), 5.0},
  };
  for (const auto& c : cases) {
    const double p_imhof = magmaan::robust::imhof_upper(c.lambda, c.x);
    const double p_mc    = mc_upper(c.lambda, c.x, N, rng);
    // Wilson-style envelope: 4·σ where σ = √(p(1−p)/N).
    const double sigma   = std::sqrt(std::max(p_mc * (1.0 - p_mc), 1e-12) / N);
    const double tol     = std::max(4.0 * sigma, 5e-4);
    INFO("λ=[", c.lambda.transpose(), "]  x=", c.x,
         "  p_imhof=", p_imhof, "  p_mc=", p_mc, "  tol=", tol);
    CHECK(std::abs(p_imhof - p_mc) < tol);
  }
}
