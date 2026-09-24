#include <doctest/doctest.h>

#include <initializer_list>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/robust/frontier/fmg.hpp"

namespace {

Eigen::VectorXd vec(std::initializer_list<double> xs) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(xs.size()));
  Eigen::Index i = 0;
  for (double x : xs) out(i++) = x;
  return out;
}

struct Case {
  const char* method_name;
  magmaan::robust::frontier::FmgMethod method;
  double param;
  double p_value;
  double chi2_equiv;
  std::vector<double> reference_lambdas;
};

}  // namespace

TEST_CASE("FMG fixed-spectrum p-value goldens match independent Imhof oracle") {
  using magmaan::robust::frontier::FmgMethod;
  using magmaan::robust::frontier::FmgOptions;
  using magmaan::robust::frontier::fmg_test;

  const double chi2 = 8.75;
  const int df = 5;
  const Eigen::VectorXd eig = vec({0.35, 3.2, 1.4, 0.9, 2.1, 0.15, 4.6});
  const Eigen::VectorXd expected_lambdas = vec({4.6, 3.2, 2.1, 1.4, 0.9});

  // Oracle p-values were generated once from R's pchisq/pf and
  // CompQuadForm::imhof with epsabs/epsrel = 1e-12. No magmaan code
  // participates in the constants.
  const std::vector<Case> cases = {
      {"standard", FmgMethod::StandardChiSquare, 0.0,
       1.1946101189794246e-01, 8.75, {}},
      {"satorra-bentler", FmgMethod::SatorraBentler, 0.0,
       6.1040669165564887e-01, 3.5860655737704938,
       {4.6, 3.2, 2.1, 1.4, 0.9}},
      {"scaled-shifted", FmgMethod::ScaledShifted, 0.0,
       5.8475744029176702e-01, 3.7579960255809226,
       {4.6, 3.2, 2.1, 1.4, 0.9}},
      {"scaled-F", FmgMethod::ScaledF, 0.0,
       5.7647078093452753e-01, 3.8141587363746727,
       {4.6, 3.2, 2.1, 1.4, 0.9}},
      {"all", FmgMethod::All, 0.0,
       5.7419404373323291e-01, 3.8296476407353186,
       {4.6, 3.2, 2.1, 1.4, 0.9}},
      {"penalized-all", FmgMethod::PenalizedAll, 0.0,
       6.0096741935659492e-01, 3.6490272756409365,
       {3.52, 2.82, 2.27, 1.92, 1.67}},
      {"eba-2", FmgMethod::Eba, 2.0,
       5.8537306895472507e-01, 3.7538365071506354,
       {3.3, 3.3, 3.3, 1.15, 1.15}},
      {"peba-2", FmgMethod::Peba, 2.0,
       6.0415575724181159e-01, 3.6277224347575285,
       {2.87, 2.87, 2.87, 1.795, 1.795}},
      {"pols-2", FmgMethod::Pols, 2.0,
       6.0106994894497567e-01, 3.6483415384489897,
       {3.36, 2.9, 2.44, 1.98, 1.52}},
  };

  for (const auto& c : cases) {
    INFO("method=", std::string(c.method_name));
    const auto got = fmg_test(chi2, df, eig,
                              FmgOptions{.method = c.method, .param = c.param});

    REQUIRE(got.lambdas_raw.size() == expected_lambdas.size());
    REQUIRE(got.lambdas.size() == expected_lambdas.size());
    for (Eigen::Index i = 0; i < expected_lambdas.size(); ++i) {
      CHECK(got.lambdas_raw(i) == doctest::Approx(expected_lambdas(i)));
      CHECK(got.lambdas(i) == doctest::Approx(expected_lambdas(i)));
    }

    REQUIRE(got.lambdas_reference.size() ==
            static_cast<Eigen::Index>(c.reference_lambdas.size()));
    for (Eigen::Index i = 0; i < got.lambdas_reference.size(); ++i) {
      CHECK(got.lambdas_reference(i) ==
            doctest::Approx(c.reference_lambdas[static_cast<std::size_t>(i)])
                .epsilon(1e-12));
    }

    CHECK(got.p_value == doctest::Approx(c.p_value).epsilon(5e-8));
    CHECK(got.chi2_equiv == doctest::Approx(c.chi2_equiv).epsilon(1e-7));
    CHECK(got.n_truncated == 0);
  }
}
