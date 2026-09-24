#pragma once

#include "magmaan/model/matrix_rep.hpp"

namespace magmaan::estimate::detail {

// Start heuristics concern observed indicators, not the phantom states used
// to evaluate a structural model. Never pass these semantic cells to an evaluator.
inline model::Cell start_cell(const spec::LatentStructure& pt,
                              const model::MatrixRep& rep, std::size_t i) {
  auto c = rep.cell_for_row[i];
  if (!c.used || rep.form != model::RepForm::Reduced) return c;
  auto observed = [&](std::int32_t var) {
    return var >= 0 && var < pt.n_vars
        ? pt.ov_pos[static_cast<std::size_t>(var)] : -1;
  };
  const auto left = observed(pt.lhs_var[i]);
  const auto right = observed(pt.rhs_var[i]);
  if (pt.op[i] == parse::Op::Measurement && right >= 0) {
    c.mat = model::MatId::Lambda;
    c.row = static_cast<std::int16_t>(right);
  } else if (pt.op[i] == parse::Op::Covariance && left >= 0 && right >= 0) {
    c.mat = model::MatId::Theta;
    c.row = static_cast<std::int16_t>(left);
    c.col = static_cast<std::int16_t>(right);
  } else if (pt.op[i] == parse::Op::Intercept && left >= 0) {
    c.mat = model::MatId::Nu;
    c.row = static_cast<std::int16_t>(left);
  }
  return c;
}

} // namespace magmaan::estimate::detail
