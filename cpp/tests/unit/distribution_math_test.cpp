#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "../oracle.hpp"
// Private src/ helper; included by relative path because src/ is PRIVATE to the
// magmaan target. The kernels are header-only (all inline), so no link step is
// required. These goldens are the sanctioned validation for the hand-rolled
// special functions: every new kernel added here must ship a golden vs R.
#include "../../src/detail_distribution_math.hpp"

namespace {

// Per-kernel tolerances set from the measured worst-case error vs base R, with a
// few× margin. They encode each kernel's genuine accuracy, not an aspiration:
//  - gamma_p/q, student_t_cdf, F infinite-df branches: ~machine precision.
//  - regularized_beta saturates to 1 near the upper boundary (its bt prefactor
//    underflows), so its CDF (and the F upper tail wrapping it) carries an abs
//    floor ~1e-8 there. Benign for callers, which use the small-tail side.
//  - quantile kernels invert the CDF by bisection; the t quantile is also
//    limited by CDF conditioning in the tail.
double tol_for(const std::string& fn, double expected) {
  const double a = std::abs(expected);
  if (fn == "student_t_quantile")       return 1e-4 + 1e-5 * a;
  if (fn == "inverse_gamma_p")          return 1e-7 + 1e-7 * a;
  if (fn == "inverse_regularized_beta") return 1e-9 + 1e-9 * a;
  if (fn == "regularized_beta")         return 5e-8 + 1e-9 * a;
  if (fn == "f_upper_tail")             return 2e-8 + 1e-9 * a;
  return 1e-12 + 1e-10 * a;  // gamma_p/q, student_t_cdf, F inf-df branches
}

double eval_case(const nlohmann::json& c) {
  using namespace magmaan::detail;
  const std::string fn = c["fn"].get<std::string>();
  const auto num = [&](const char* k) { return c[k].get<double>(); };
  const double inf = std::numeric_limits<double>::infinity();

  if (fn == "gamma_p") return gamma_p(num("a"), num("x"));
  if (fn == "gamma_q") return gamma_q(num("a"), num("x"));
  if (fn == "regularized_beta") return regularized_beta(num("a"), num("b"), num("x"));
  if (fn == "inverse_regularized_beta")
    return inverse_regularized_beta(num("p"), num("a"), num("b"));
  if (fn == "inverse_gamma_p")
    return inverse_gamma_p(num("p"), num("shape"), num("scale"));
  if (fn == "student_t_cdf") return student_t_cdf(num("x"), num("df"));
  if (fn == "student_t_quantile") return student_t_quantile(num("p"), num("df"));
  if (fn == "f_upper_tail") return f_upper_tail(num("x"), num("d1"), num("d2"));
  if (fn == "f_upper_tail_d1inf") return f_upper_tail(num("x"), inf, num("d2"));
  if (fn == "f_upper_tail_d2inf") return f_upper_tail(num("x"), num("d1"), inf);
  return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

TEST_CASE("special functions match base-R distribution goldens") {
  const std::string path =
      magmaan::test::fixtures_dir() + "/distribution_math.json";
  auto raw = magmaan::test::read_fixture(path);
  REQUIRE(raw.has_value());
  auto fixture = nlohmann::json::parse(*raw, nullptr, false);
  REQUIRE_FALSE(fixture.is_discarded());

  std::size_t checked = 0;
  for (const auto& c : fixture["cases"]) {
    const std::string fn = c["fn"].get<std::string>();
    const double expected = c["expected"].get<double>();
    const double got = eval_case(c);
    const double tol = tol_for(fn, expected);
    INFO("fn=", fn, " expected=", expected, " got=", got, " tol=", tol);
    REQUIRE(std::isfinite(got));
    CHECK(std::abs(got - expected) <= tol);
    ++checked;
  }
  CHECK(checked > 1000);  // guard against an empty/short fixture silently passing
}
