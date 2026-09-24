#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/model/auto_identification.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::compat::lavaan::LavaanParTable;
using magmaan::compat::lavaan::to_lavaan_partable;
using magmaan::model::AdmissibilityVerdict;
using magmaan::model::backconvert_std_lv_to_marker;
using magmaan::model::is_std_lv_admissible;
using magmaan::model::partable_marker_to_std_lv;
using magmaan::parse::Op;
using magmaan::parse::Parser;
using magmaan::spec::BuildOptions;
using magmaan::spec::LatentNames;
using magmaan::spec::LatentStructure;
using magmaan::spec::Starts;

namespace {

LavaanParTable build_pt(std::string_view syntax, bool std_lv) {
  Parser p;
  auto parsed = p.parse(syntax);
  REQUIRE(parsed.has_value());
  BuildOptions opts;
  opts.auto_fix_first = !std_lv;
  opts.std_lv         = std_lv;
  opts.auto_var       = true;
  opts.fixed_x        = true;
  Starts starts;
  LatentNames names;
  auto built = magmaan::spec::build(*parsed, opts, &starts, &names);
  REQUIRE(built.has_value());
  return to_lavaan_partable(*built, names, starts);
}

}  // namespace

TEST_CASE("admissibility: clean two-factor CFA admits") {
  auto m = build_pt("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6", false);
  auto s = build_pt("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6", true);
  auto v = is_std_lv_admissible(m, s);
  CHECK(v.admissible);
  CHECK(v.reason.empty());
}

TEST_CASE("admissibility: cross-factor loading equality is rejected") {
  // `eq` label shared across f1 and f2 loadings -- the longitudinal-
  // invariance pattern. Should be rejected via the `==` row predicate
  // (lavaanify materializes the equality as an `==` constraint).
  auto m = build_pt("f1 =~ y1 + eq*y2\nf2 =~ y3 + eq*y4", false);
  auto v = is_std_lv_admissible(m);
  CHECK_FALSE(v.admissible);
}

TEST_CASE("admissibility: higher-order is rejected") {
  auto m = build_pt(
      "f1 =~ y1 + y2\nf2 =~ y3 + y4\nf3 =~ y5 + y6\ng =~ f1 + f2 + f3",
      false);
  auto v = is_std_lv_admissible(m);
  CHECK_FALSE(v.admissible);
  CHECK(v.reason.find("higher-order") != std::string::npos);
}

TEST_CASE("admissibility: fixed latent variance is rejected") {
  auto m = build_pt("f =~ y1 + y2 + y3\nf ~~ 1*f", false);
  auto v = is_std_lv_admissible(m);
  CHECK_FALSE(v.admissible);
  CHECK(v.reason.find("latent variance") != std::string::npos);
}

TEST_CASE("partable_marker_to_std_lv: matches lavaan std_lv npar") {
  auto m = build_pt("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6", false);
  auto s = build_pt("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6", true);
  auto surg = partable_marker_to_std_lv(m);
  CHECK(surg.n_free() == s.n_free());
  CHECK(surg.size() == m.size());
}

TEST_CASE("partable_marker_to_std_lv: marker anchors become free, "
          "latent variances become fixed at 1") {
  auto m = build_pt("f =~ y1 + y2 + y3", false);
  auto surg = partable_marker_to_std_lv(m);
  bool saw_freed_anchor = false;
  bool saw_fixed_var = false;
  for (std::size_t i = 0; i < surg.size(); ++i) {
    if (surg.op[i] == Op::Measurement && surg.lhs[i] == "f" &&
        surg.rhs[i] == "y1") {
      // anchor row: should now be free, ustart NaN
      CHECK(surg.free[i] > 0);
      CHECK(std::isnan(surg.ustart[i]));
      saw_freed_anchor = true;
    }
    if (surg.op[i] == Op::Covariance && surg.lhs[i] == "f" &&
        surg.rhs[i] == "f") {
      // variance row: should now be fixed at 1
      CHECK(surg.free[i] == 0);
      CHECK(std::abs(surg.ustart[i] - 1.0) < 1e-12);
      saw_fixed_var = true;
    }
  }
  CHECK(saw_freed_anchor);
  CHECK(saw_fixed_var);
}

TEST_CASE("backconvert: per-factor rescaling round-trip on a synthetic est") {
  // Build a single-factor CFA: f =~ y1 + y2 + y3. Marker form has
  // lambda(f, y1) = 1 fixed, lambda(f, y2), lambda(f, y3) free, psi(f) free,
  // residual variances free.
  auto m = build_pt("f =~ y1 + y2 + y3", false);

  // Construct a synthetic std_lv-coord est vector with c_f = 2 implied:
  // - lambda(f, y1)_std = 2  -> backconverts to 1
  // - lambda(f, y2)_std = 4  -> backconverts to 2
  // - lambda(f, y3)_std = 6  -> backconverts to 3
  // - psi(f)_std       = 1  -> backconverts to 4
  // - residual variances pass through unchanged.
  Eigen::VectorXd est(static_cast<Eigen::Index>(m.size()));
  for (std::size_t i = 0; i < m.size(); ++i) {
    const auto k = static_cast<Eigen::Index>(i);
    if (m.op[i] == Op::Measurement && m.rhs[i] == "y1") est[k] = 2.0;
    else if (m.op[i] == Op::Measurement && m.rhs[i] == "y2") est[k] = 4.0;
    else if (m.op[i] == Op::Measurement && m.rhs[i] == "y3") est[k] = 6.0;
    else if (m.op[i] == Op::Covariance && m.lhs[i] == "f" &&
             m.rhs[i] == "f") est[k] = 1.0;
    else est[k] = 0.5;  // residual variances (arbitrary)
  }

  Eigen::VectorXd back = backconvert_std_lv_to_marker(m, est);
  for (std::size_t i = 0; i < m.size(); ++i) {
    const auto k = static_cast<Eigen::Index>(i);
    if (m.op[i] == Op::Measurement && m.rhs[i] == "y1") {
      CHECK(back[k] == doctest::Approx(1.0));
    } else if (m.op[i] == Op::Measurement && m.rhs[i] == "y2") {
      CHECK(back[k] == doctest::Approx(2.0));
    } else if (m.op[i] == Op::Measurement && m.rhs[i] == "y3") {
      CHECK(back[k] == doctest::Approx(3.0));
    } else if (m.op[i] == Op::Covariance && m.lhs[i] == "f" &&
               m.rhs[i] == "f") {
      CHECK(back[k] == doctest::Approx(4.0));
    } else {
      CHECK(back[k] == doctest::Approx(0.5));
    }
  }
}
