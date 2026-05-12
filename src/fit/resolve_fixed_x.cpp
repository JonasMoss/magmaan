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
resolve_fixed_x_from_sample(partable::LatentStructure&      pt,
                            const model::MatrixRep&  rep,
                            const SampleStats&       samp) {
  if (rep.cell_for_row.size() != pt.size()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "MatrixRep cell count doesn't match LatentStructure"));
  }
  const std::size_t n_lv_ext = pt.lv_ext_order.size();
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
    // A fixed.x cell lives in Ψ (the Reduced form promotes exogenous observed
    // to phantom latents): c.row / c.col are lv_ext positions. Map them to var
    // ids via lv_ext_order, then to observed positions via ov_pos — for a
    // phantom-promoted ov.x the lv_ext slot and the observed slot are the same
    // variable — and read the sample covariance there.
    if (c.row < 0 || c.col < 0 ||
        static_cast<std::size_t>(c.row) >= n_lv_ext ||
        static_cast<std::size_t>(c.col) >= n_lv_ext) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fixed.x cell indices out of range"));
    }
    const std::int32_t v_r = pt.lv_ext_order[static_cast<std::size_t>(c.row)];
    const std::int32_t v_c = pt.lv_ext_order[static_cast<std::size_t>(c.col)];
    const std::int32_t r = (v_r >= 0 && v_r < pt.n_vars)
                               ? pt.ov_pos[static_cast<std::size_t>(v_r)] : -1;
    const std::int32_t s = (v_c >= 0 && v_c < pt.n_vars)
                               ? pt.ov_pos[static_cast<std::size_t>(v_c)] : -1;
    if (r < 0 || s < 0 ||
        r >= static_cast<std::int32_t>(S.rows()) ||
        s >= static_cast<std::int32_t>(S.cols())) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fixed.x variable not found in the observed ordering"));
    }
    pt.fixed_value[i] = S(r, s);
  }
  return {};
}

}  // namespace latva::fit
