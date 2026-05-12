#include "latva/fit/resolve_fixed_x.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/partable/partable.hpp"

namespace latva::fit {

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

}  // namespace

fit_expected<void>
resolve_fixed_x_from_sample(partable::ParTable&      pt,
                            const model::MatrixRep&  rep,
                            const SampleStats&       samp) {
  if (rep.cell_for_row.size() != pt.size()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "MatrixRep cell count doesn't match ParTable"));
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.exo[i] != 1) continue;          // only fixed.x rows
    if (pt.free[i] != 0) continue;
    if (std::isfinite(pt.fixed_value[i])) continue;  // already resolved
    const auto& c = rep.cell_for_row[i];
    if (!c.used) continue;
    const std::size_t b = static_cast<std::size_t>(c.block);
    if (b >= samp.S.size()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "block " + std::to_string(b) + " missing from SampleStats"));
    }
    const auto& S = samp.S[b];
    // Fixed.x cells live in Psi under the Reduced form; row/col are
    // lv_ext indices, but since the phantom-Λ row identifies ov_idx
    // directly via name, we look up the sample S via the rep's
    // ov_names: lv_ext name == ov name for these rows.
    if (b >= rep.lv_names.size() || b >= rep.ov_names.size()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "MatrixRep block dims missing"));
    }
    const auto& lv_n = rep.lv_names[b];
    const auto& ov_n = rep.ov_names[b];
    if (c.row < 0 || c.col < 0 ||
        c.row >= static_cast<std::int16_t>(lv_n.size()) ||
        c.col >= static_cast<std::int16_t>(lv_n.size())) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fixed.x cell indices out of range"));
    }
    // Find ov index of these names. (For phantom-promoted ov.x, the
    // lv_ext name equals the ov name.)
    auto find_ov = [&](const std::string& n) -> int {
      for (std::size_t k = 0; k < ov_n.size(); ++k)
        if (ov_n[k] == n) return static_cast<int>(k);
      return -1;
    };
    const int r = find_ov(lv_n[static_cast<std::size_t>(c.row)]);
    const int s = find_ov(lv_n[static_cast<std::size_t>(c.col)]);
    if (r < 0 || s < 0) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fixed.x lv_ext name not found in ov_names"));
    }
    pt.fixed_value[i] = S(r, s);
  }
  return {};
}

}  // namespace latva::fit
