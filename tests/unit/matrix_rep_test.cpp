#include <doctest/doctest.h>

#include <string>
#include <string_view>

#include "latva/model/matrix_rep.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

using latva::model::build_matrix_rep;
using latva::model::Cell;
using latva::model::MatId;
using latva::model::MatrixRep;
using latva::parse::Parser;
using latva::partable::lavaanify;

namespace {

MatrixRep must_build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
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

TEST_CASE("matrix_rep: constraint rows are sentinel cells") {
  auto mr = must_build("f =~ x1 + a*x2 + a*x3");
  // The shared label triggers an auto-equality `==` row at the end.
  // That row should have used == false.
  bool found_unused = false;
  for (const auto& c : mr.cell_for_row) {
    if (!c.used) { found_unused = true; break; }
  }
  CHECK(found_unused);
}
