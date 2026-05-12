#include "latva/model/matrix_rep.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/parse/op.hpp"
#include "latva/partable/partable.hpp"

namespace latva::model {

namespace {

ModelError make_err(ModelError::Kind k, std::string detail) {
  return ModelError{k, std::move(detail)};
}

bool is_constraint_op(parse::Op op) noexcept {
  return op == parse::Op::EqConstraint || op == parse::Op::LtConstraint ||
         op == parse::Op::GtConstraint || op == parse::Op::DefineParam;
}

// Decide which form to use: PureCFA when there are no `~` regressions,
// else Reduced. `~1` (mean structure) does NOT force Reduced — intercepts
// live in Nu / Alpha and don't require phantom latents to represent.
RepForm decide_form(const partable::ParTable& pt) noexcept {
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == parse::Op::Regression) return RepForm::Reduced;
  }
  return RepForm::PureCFA;
}

// Recover var-id → name from the partable's (still-present) string columns:
// every variable appears as the lhs or rhs of at least one formula row.
std::vector<std::string> var_names_of(const partable::ParTable& pt) {
  std::vector<std::string> names(static_cast<std::size_t>(pt.n_vars));
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.is_constraint_row(i)) continue;
    const std::int32_t l = pt.lhs_var[i];
    const std::int32_t r = pt.rhs_var[i];
    if (l >= 0 && l < pt.n_vars) names[static_cast<std::size_t>(l)] = pt.lhs[i];
    if (r >= 0 && r < pt.n_vars) names[static_cast<std::size_t>(r)] = pt.rhs[i];
  }
  return names;
}

}  // namespace

model_expected<MatrixRep> build_matrix_rep(const partable::ParTable& pt) {
  MatrixRep out;
  out.cell_for_row.resize(pt.size());
  out.form = decide_form(pt);

  const std::size_t nb = static_cast<std::size_t>(pt.n_groups());
  out.dims.resize(nb);
  out.ov_names.resize(nb);
  out.lv_names.resize(nb);

  // The canonical orderings + classification are precomputed by lavaanify and
  // ride on the partable's variable inventory — matrix_rep no longer re-derives
  // them. In v0 multi-group, every block has the same variable set; broadcast.
  const std::vector<std::string> names = var_names_of(pt);
  std::vector<std::string> ov_names_b, lv_names_b;
  ov_names_b.reserve(pt.ov_order.size());
  lv_names_b.reserve(pt.lv_ext_order.size());
  for (auto v : pt.ov_order)     ov_names_b.push_back(names[static_cast<std::size_t>(v)]);
  for (auto v : pt.lv_ext_order) lv_names_b.push_back(names[static_cast<std::size_t>(v)]);
  for (std::size_t b = 0; b < nb; ++b) {
    out.ov_names[b] = ov_names_b;
    out.lv_names[b] = lv_names_b;
    out.dims[b] = BlockDims{static_cast<std::int16_t>(pt.ov_order.size()),
                            static_cast<std::int16_t>(pt.lv_ext_order.size())};
  }

  // Phantom-Λ structural identity cells per block: every variable promoted
  // into the extended-latent set (i.e. in lv_ext_order but not a user latent)
  // gets Λ[ov_pos, lv_ext_pos] = 1 in each block. Only Reduced form has these.
  if (out.form == RepForm::Reduced) {
    for (auto v : pt.lv_ext_order) {
      const std::size_t vi = static_cast<std::size_t>(v);
      if (pt.is_user_latent[vi]) continue;
      const std::int32_t ov_i = pt.ov_pos[vi];
      const std::int32_t lv_i = pt.lv_ext_pos[vi];
      if (ov_i < 0 || lv_i < 0) continue;
      for (std::int8_t b = 0; b < static_cast<std::int8_t>(nb); ++b) {
        StructuralCell sc;
        sc.mat   = MatId::Lambda;
        sc.row   = static_cast<std::int16_t>(ov_i);
        sc.col   = static_cast<std::int16_t>(lv_i);
        sc.block = b;
        sc.value = 1.0;
        out.structural_cells.push_back(sc);
      }
    }
  }

  auto ov_idx     = [&](std::int32_t v) -> std::int16_t {
    return (v < 0 || v >= pt.n_vars) ? std::int16_t{-1}
                                     : static_cast<std::int16_t>(pt.ov_pos[static_cast<std::size_t>(v)]);
  };
  auto lv_ext_idx = [&](std::int32_t v) -> std::int16_t {
    return (v < 0 || v >= pt.n_vars) ? std::int16_t{-1}
                                     : static_cast<std::int16_t>(pt.lv_ext_pos[static_cast<std::size_t>(v)]);
  };
  auto is_user_lv = [&](std::int32_t v) -> bool {
    return v >= 0 && v < pt.n_vars && pt.is_user_latent[static_cast<std::size_t>(v)] != 0;
  };

  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto op = pt.op[i];
    if (is_constraint_op(op)) continue;

    const std::int32_t L = pt.lhs_var[i];
    const std::int32_t R = pt.rhs_var[i];
    Cell c; c.block = static_cast<std::int8_t>(pt.block[i] - 1);  // 1-based → 0-based

    auto unknown_var = [&](std::string_view s) {
      return std::unexpected(make_err(ModelError::Kind::UnknownVariable,
          std::string("row references unknown variable: '") + std::string(s) + "'"));
    };

    switch (op) {
      case parse::Op::Measurement: {
        // Λ[ind, lat] regardless of form.
        const auto row_idx = ov_idx(R);
        const auto col_idx = lv_ext_idx(L);
        if (row_idx < 0) return unknown_var(pt.rhs[i]);
        if (col_idx < 0) return unknown_var(pt.lhs[i]);
        c.mat = MatId::Lambda; c.row = row_idx; c.col = col_idx; c.used = true;
        break;
      }
      case parse::Op::Covariance: {
        // PureCFA: lv↔lv → Ψ; ov↔ov → Θ; mixed → error (rare).
        // Reduced:  any operand in lv_ext → Ψ; otherwise Θ.
        if (out.form == RepForm::Reduced) {
          const bool L_ext = lv_ext_idx(L) >= 0;
          const bool R_ext = lv_ext_idx(R) >= 0;
          if (L_ext || R_ext) {
            c.mat = MatId::Psi; c.row = lv_ext_idx(L); c.col = lv_ext_idx(R);
          } else {
            c.mat = MatId::Theta; c.row = ov_idx(L); c.col = ov_idx(R);
          }
        } else {
          const bool L_lv = is_user_lv(L);
          const bool R_lv = is_user_lv(R);
          if (L_lv && R_lv) {
            c.mat = MatId::Psi; c.row = lv_ext_idx(L); c.col = lv_ext_idx(R);
          } else if (!L_lv && !R_lv) {
            c.mat = MatId::Theta; c.row = ov_idx(L); c.col = ov_idx(R);
          } else {
            return std::unexpected(make_err(ModelError::Kind::UnsupportedRowKind,
                std::string("~~ between latent and observed not supported in "
                            "PureCFA: lhs='") + pt.lhs[i] + "', rhs='" + pt.rhs[i] + "'"));
          }
        }
        if (c.row < 0) return unknown_var(pt.lhs[i]);
        if (c.col < 0) return unknown_var(pt.rhs[i]);
        c.used = true;
        break;
      }
      case parse::Op::Regression: {
        // Β[lhs, rhs] in lv_ext indices. Reduced form only.
        if (out.form != RepForm::Reduced) {
          return std::unexpected(make_err(ModelError::Kind::UnsupportedRowKind,
              "~ requires Reduced form (decide_form picked PureCFA)"));
        }
        const auto row_idx = lv_ext_idx(L);
        const auto col_idx = lv_ext_idx(R);
        if (row_idx < 0) return unknown_var(pt.lhs[i]);
        if (col_idx < 0) return unknown_var(pt.rhs[i]);
        c.mat = MatId::Beta; c.row = row_idx; c.col = col_idx; c.used = true;
        break;
      }
      case parse::Op::Intercept: {
        // `lhs ~ 1`: indicator intercept (Nu) if lhs is observed, latent mean
        // (Alpha) if lhs is a user latent. An `ov ~ 1` still maps to Nu even
        // when ov was promoted to a phantom latent in Reduced form.
        if (is_user_lv(L)) {
          const auto row_idx = lv_ext_idx(L);
          if (row_idx < 0) return unknown_var(pt.lhs[i]);
          c.mat = MatId::Alpha; c.row = row_idx; c.col = 0; c.used = true;
        } else {
          const auto row_idx = ov_idx(L);
          if (row_idx < 0) return unknown_var(pt.lhs[i]);
          c.mat = MatId::Nu; c.row = row_idx; c.col = 0; c.used = true;
        }
        break;
      }
      default:
        break;
    }
    out.cell_for_row[i] = c;
  }

  return out;
}

}  // namespace latva::model
