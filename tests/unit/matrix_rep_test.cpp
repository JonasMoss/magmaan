#include <doctest/doctest.h>

#include <string>
#include <string_view>

#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::model::build_matrix_rep;
using magmaan::model::Cell;
using magmaan::model::MatId;
using magmaan::model::MatrixRep;
using magmaan::model::RepForm;
using magmaan::parse::Parser;
using magmaan::spec::build;

namespace {

MatrixRep must_build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LatentNames names;
  auto pt = build(*fp, {}, nullptr, &names);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt, &names);
  REQUIRE_MESSAGE(mr.has_value(),
                  "build_matrix_rep failed: " << mr.error().detail);
  return std::move(*mr);
}

// Scan cell_for_row for the first cell matching the predicate.
const Cell* find_cell(const MatrixRep& mr, MatId mat,
                      std::int16_t row, std::int16_t col) {
  for (const auto& c : mr.cell_for_row) {
    if (c.used && c.mat == mat && c.row == row && c.col == col) return &c;
  }
  return nullptr;
}

bool has_structural_cell(const MatrixRep& mr, MatId mat,
                         std::int16_t row, std::int16_t col) {
  for (const auto& c : mr.structural_cells) {
    if (c.mat == mat && c.row == row && c.col == col) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("matrix_rep: 1F CFA — Lambda 3×1, Theta 3×3 diag, Psi 1×1") {
  auto mr = must_build("f =~ x1 + x2 + x3");
  REQUIRE(mr.dims.size() == 1);
  CHECK(mr.dims[0].n_observed == 3);
  CHECK(mr.dims[0].n_latent   == 1);
  REQUIRE(mr.ov_names[0] == std::vector<std::string>{"x1", "x2", "x3"});
  REQUIRE(mr.lv_names[0] == std::vector<std::string>{"f"});

  // Λ entries
  CHECK(find_cell(mr, MatId::Lambda, 0, 0) != nullptr);   // f =~ x1
  CHECK(find_cell(mr, MatId::Lambda, 1, 0) != nullptr);   // f =~ x2
  CHECK(find_cell(mr, MatId::Lambda, 2, 0) != nullptr);   // f =~ x3

  // Θ diagonal
  CHECK(find_cell(mr, MatId::Theta, 0, 0) != nullptr);
  CHECK(find_cell(mr, MatId::Theta, 1, 1) != nullptr);
  CHECK(find_cell(mr, MatId::Theta, 2, 2) != nullptr);

  // Ψ
  CHECK(find_cell(mr, MatId::Psi, 0, 0) != nullptr);
}

TEST_CASE("matrix_rep: 3F Holzinger — Lambda 9×3, Theta 9×9, Psi 3×3 with off-diagonals") {
  auto mr = must_build(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  CHECK(mr.dims[0].n_observed == 9);
  CHECK(mr.dims[0].n_latent   == 3);
  REQUIRE(mr.lv_names[0] == std::vector<std::string>{"visual", "textual", "speed"});

  // Visual loadings on x1, x2, x3.
  CHECK(find_cell(mr, MatId::Lambda, 0, 0) != nullptr);
  CHECK(find_cell(mr, MatId::Lambda, 1, 0) != nullptr);
  CHECK(find_cell(mr, MatId::Lambda, 2, 0) != nullptr);
  // Textual loadings on x4, x5, x6.
  CHECK(find_cell(mr, MatId::Lambda, 3, 1) != nullptr);
  CHECK(find_cell(mr, MatId::Lambda, 4, 1) != nullptr);
  CHECK(find_cell(mr, MatId::Lambda, 5, 1) != nullptr);
  // Speed loadings on x7, x8, x9.
  CHECK(find_cell(mr, MatId::Lambda, 6, 2) != nullptr);
  CHECK(find_cell(mr, MatId::Lambda, 7, 2) != nullptr);
  CHECK(find_cell(mr, MatId::Lambda, 8, 2) != nullptr);

  // Latent covariances from auto.cov.lv.x: visual~~textual, visual~~speed, textual~~speed.
  CHECK(find_cell(mr, MatId::Psi, 0, 1) != nullptr);
  CHECK(find_cell(mr, MatId::Psi, 0, 2) != nullptr);
  CHECK(find_cell(mr, MatId::Psi, 1, 2) != nullptr);

  // Latent variances on the diagonal.
  CHECK(find_cell(mr, MatId::Psi, 0, 0) != nullptr);
  CHECK(find_cell(mr, MatId::Psi, 1, 1) != nullptr);
  CHECK(find_cell(mr, MatId::Psi, 2, 2) != nullptr);
}

TEST_CASE("matrix_rep: residual covariance lands in Theta off-diagonal") {
  auto mr = must_build("x1 ~~ x2");
  CHECK(mr.dims[0].n_observed == 2);
  CHECK(mr.dims[0].n_latent   == 0);
  CHECK(find_cell(mr, MatId::Theta, 0, 1) != nullptr);  // x1 ~~ x2
  CHECK(find_cell(mr, MatId::Theta, 0, 0) != nullptr);  // auto x1 ~~ x1
  CHECK(find_cell(mr, MatId::Theta, 1, 1) != nullptr);  // auto x2 ~~ x2
}

TEST_CASE("matrix_rep: second-order CFA — higher-order =~ lowers to Beta") {
  // `general` is measured by three latents. The first-order factors stay
  // latent (not observed), so the model has 9 observed and 4 latent vars —
  // the higher-order loadings land in Β, not Λ, and force the Reduced form.
  auto mr = must_build(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9\n"
      "general =~ visual + textual + speed");
  CHECK(mr.form == RepForm::Reduced);
  CHECK(mr.dims[0].n_observed == 9);
  CHECK(mr.dims[0].n_latent   == 4);
  REQUIRE(mr.ov_names[0] ==
          std::vector<std::string>{"x1", "x2", "x3", "x4", "x5", "x6",
                                   "x7", "x8", "x9"});
  REQUIRE(mr.lv_names[0] ==
          std::vector<std::string>{"visual", "textual", "speed", "general"});

  // First-order loadings still in Λ.
  CHECK(find_cell(mr, MatId::Lambda, 0, 0) != nullptr);  // visual =~ x1
  CHECK(find_cell(mr, MatId::Lambda, 5, 1) != nullptr);  // textual =~ x6
  CHECK(find_cell(mr, MatId::Lambda, 8, 2) != nullptr);  // speed =~ x9

  // Higher-order loadings in Β[effect, cause]: visual/textual/speed ← general.
  CHECK(find_cell(mr, MatId::Beta, 0, 3) != nullptr);    // general =~ visual
  CHECK(find_cell(mr, MatId::Beta, 1, 3) != nullptr);    // general =~ textual
  CHECK(find_cell(mr, MatId::Beta, 2, 3) != nullptr);    // general =~ speed

  // A first-order factor never appears as a Λ row.
  CHECK(find_cell(mr, MatId::Lambda, 9, 3) == nullptr);
}

TEST_CASE("matrix_rep: measurement of a promoted observed state lowers to Beta") {
  auto mr = must_build(
      "Deta2 =~ 1*bmi2\n"
      "bmi2 ~ 1*bmi1");
  CHECK(mr.form == RepForm::Reduced);
  REQUIRE(mr.ov_names[0] == std::vector<std::string>{"bmi2", "bmi1"});
  REQUIRE(mr.lv_names[0] == std::vector<std::string>{"Deta2", "bmi2", "bmi1"});

  CHECK(has_structural_cell(mr, MatId::Lambda, 0, 1));  // bmi2 measures phantom bmi2
  CHECK(has_structural_cell(mr, MatId::Lambda, 1, 2));  // bmi1 measures phantom bmi1
  CHECK(find_cell(mr, MatId::Beta, 1, 0) != nullptr);   // Deta2 =~ bmi2 => bmi2 <- Deta2
  CHECK(find_cell(mr, MatId::Beta, 1, 2) != nullptr);   // bmi2 ~ bmi1
  CHECK(find_cell(mr, MatId::Lambda, 0, 0) == nullptr);
}

TEST_CASE("matrix_rep: third-order chain g =~ trait =~ state =~ d") {
  // Higher-order lowering composes — every latent-on-latent =~ becomes a Β
  // path, leaving only the bottom =~ rows in Λ.
  auto mr = must_build(
      "state1 =~ d1 + d2\n"
      "state2 =~ d3 + d4\n"
      "trait =~ state1 + state2\n"
      "g =~ trait");
  CHECK(mr.form == RepForm::Reduced);
  CHECK(mr.dims[0].n_observed == 4);   // d1..d4
  CHECK(mr.dims[0].n_latent   == 4);   // state1, state2, trait, g
  CHECK(find_cell(mr, MatId::Lambda, 0, 0) != nullptr);  // state1 =~ d1
  CHECK(find_cell(mr, MatId::Beta, 0, 2) != nullptr);    // trait =~ state1
  CHECK(find_cell(mr, MatId::Beta, 1, 2) != nullptr);    // trait =~ state2
  CHECK(find_cell(mr, MatId::Beta, 2, 3) != nullptr);    // g =~ trait
}

TEST_CASE("matrix_rep: constraint rows are sentinel cells") {
  // An explicit `a == b` line is a constraint row — it has no matrix cell, so
  // its `cell_for_row` entry stays a sentinel (used == false).
  auto mr = must_build("f =~ x1 + a*x2 + b*x3\na == b");
  bool found_unused = false;
  for (const auto& c : mr.cell_for_row) {
    if (!c.used) { found_unused = true; break; }
  }
  CHECK(found_unused);
}
