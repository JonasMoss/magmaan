#include "latva/model/matrix_rep.hpp"

#include <algorithm>
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

struct OrderedSet {
  std::vector<std::string> items;
  bool contains(std::string_view s) const noexcept {
    for (const auto& it : items) if (it == s) return true;
    return false;
  }
  void insert(std::string_view s) {
    if (!contains(s)) items.emplace_back(s);
  }
  std::int16_t index_of(std::string_view s) const noexcept {
    for (std::size_t i = 0; i < items.size(); ++i)
      if (items[i] == s) return static_cast<std::int16_t>(i);
    return -1;
  }
};

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
    if (pt.op[i] == parse::Op::Regression) {
      return RepForm::Reduced;
    }
  }
  return RepForm::PureCFA;
}

// Per-form variable classification + ordering.
//
//  PureCFA : lv = user latents (LHS of =~). ov = indicators (RHS of =~)
//            then anything else mentioned in non-constraint rows.
//  Reduced : ov order matches lavaan's [ov.ind..., ov.y..., ov.x...]
//            (when there are user latents) OR [ov.y..., ov.x...] (when
//            there are no latents). lv_extended = [user_lv..., ov.y...,
//            ov.x...] — endogenous and exogenous observed both promoted
//            to phantom latents.
struct VarOrder {
  OrderedSet ov_ind;     // RHS of =~
  OrderedSet ov_y;       // LHS of ~ or ~1, minus latents
  OrderedSet ov_x;       // RHS of ~, minus latents/indicators/ov.y
  OrderedSet lv;         // LHS of =~ (user latents only)
  OrderedSet ov;         // unified observed ordering
  OrderedSet lv_ext;     // unified latent ordering for Λ cols / Β / Ψ
};

VarOrder classify(const partable::ParTable& pt, RepForm form) {
  VarOrder v;
  // Pass 1: lv from =~, ov_ind from =~ rhs.
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == parse::Op::Measurement) {
      v.lv.insert(pt.lhs[i]);
      v.ov_ind.insert(pt.rhs[i]);
    }
  }
  // Pass 2: ov_y from ~ (and ~1) lhs, minus latents.
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == parse::Op::Regression || pt.op[i] == parse::Op::Intercept) {
      if (!v.lv.contains(pt.lhs[i])) v.ov_y.insert(pt.lhs[i]);
    }
  }
  // Pass 3: ov_x from ~ rhs, minus rest.
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == parse::Op::Regression) {
      if (!v.lv.contains(pt.rhs[i]) && !v.ov_ind.contains(pt.rhs[i]) &&
          !v.ov_y.contains(pt.rhs[i])) {
        v.ov_x.insert(pt.rhs[i]);
      }
    }
  }
  // Pass 4: ov_misc (anything else in ~~ rows).
  OrderedSet ov_misc;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Covariance) continue;
    for (auto name : {pt.lhs[i], pt.rhs[i]}) {
      if (!v.lv.contains(name) && !v.ov_ind.contains(name) &&
          !v.ov_y.contains(name) && !v.ov_x.contains(name)) {
        ov_misc.insert(name);
      }
    }
  }

  // Build unified ov. Lavaan's order is:
  //   PureCFA : ov.ind first, then ov_misc.
  //   Reduced with user latents : ov.ind first, then ov.y, then ov.x.
  //   Reduced no user latents   : ov.y first, then ov.x.
  if (form == RepForm::Reduced && v.lv.items.empty()) {
    for (const auto& s : v.ov_y.items)   v.ov.insert(s);
    for (const auto& s : v.ov_x.items)   v.ov.insert(s);
  } else {
    for (const auto& s : v.ov_ind.items) v.ov.insert(s);
    for (const auto& s : v.ov_y.items)   v.ov.insert(s);
    for (const auto& s : v.ov_x.items)   v.ov.insert(s);
    for (const auto& s : ov_misc.items)  v.ov.insert(s);
  }

  // lv_ext order:
  //   PureCFA : just user latents.
  //   Reduced : [user_lv..., ov.y..., ov.x...].
  for (const auto& s : v.lv.items) v.lv_ext.insert(s);
  if (form == RepForm::Reduced) {
    for (const auto& s : v.ov_y.items) v.lv_ext.insert(s);
    for (const auto& s : v.ov_x.items) v.lv_ext.insert(s);
  }
  return v;
}

}  // namespace

model_expected<MatrixRep> build_matrix_rep(const partable::ParTable& pt) {
  MatrixRep out;
  out.cell_for_row.resize(pt.size());

  // Determine number of blocks from pt.block (lavaan-style 1-based).
  // Constraint rows have block = 0 and don't count.
  std::int32_t n_blocks = 0;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.block[i] > n_blocks) n_blocks = pt.block[i];
  }
  if (n_blocks < 1) n_blocks = 1;
  const std::size_t nb = static_cast<std::size_t>(n_blocks);
  out.dims.resize(nb);
  out.ov_names.resize(nb);
  out.lv_names.resize(nb);

  out.form = decide_form(pt);
  // In v0 multi-group, every block has the same variable set (same model
  // structure replicated). Classify once and broadcast.
  const VarOrder v = classify(pt, out.form);
  for (std::size_t b = 0; b < nb; ++b) {
    out.lv_names[b] = v.lv_ext.items;
    out.ov_names[b] = v.ov.items;
    out.dims[b] = BlockDims{
        static_cast<std::int16_t>(v.ov.items.size()),
        static_cast<std::int16_t>(v.lv_ext.items.size())};
  }

  // Phantom-Λ structural identity cells per block. Reduced form: every
  // ov.y and ov.x has Λ[ov_idx, lv_ext_idx] = 1 in each block.
  if (out.form == RepForm::Reduced) {
    auto add_phantom = [&](std::string_view name, std::int8_t block) {
      const auto ov_idx = v.ov.index_of(name);
      const auto lv_idx = v.lv_ext.index_of(name);
      if (ov_idx < 0 || lv_idx < 0) return;
      StructuralCell sc;
      sc.mat   = MatId::Lambda;
      sc.row   = ov_idx;
      sc.col   = lv_idx;
      sc.block = block;
      sc.value = 1.0;
      out.structural_cells.push_back(sc);
    };
    for (std::int8_t b = 0; b < static_cast<std::int8_t>(nb); ++b) {
      for (const auto& y : v.ov_y.items) add_phantom(y, b);
      for (const auto& x : v.ov_x.items) add_phantom(x, b);
    }
  }

  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto op = pt.op[i];
    if (is_constraint_op(op)) continue;

    const std::string_view L = pt.lhs[i];
    const std::string_view R = pt.rhs[i];
    // Convert 1-based pt.block to 0-based cell.block. Constraint rows
    // already filtered above.
    Cell c; c.block = static_cast<std::int8_t>(pt.block[i] - 1);

    auto unknown_var = [&](std::string_view s) {
      return std::unexpected(make_err(ModelError::Kind::UnknownVariable,
          std::string("row references unknown variable: '") +
              std::string(s) + "'"));
    };

    switch (op) {
      case parse::Op::Measurement: {
        // Λ[ind, lat] regardless of form.
        const auto row_idx = v.ov.index_of(R);
        const auto col_idx = v.lv_ext.index_of(L);
        if (row_idx < 0) return unknown_var(R);
        if (col_idx < 0) return unknown_var(L);
        c.mat  = MatId::Lambda;
        c.row  = row_idx;
        c.col  = col_idx;
        c.used = true;
        break;
      }
      case parse::Op::Covariance: {
        // Routing depends on form + variable types:
        //   PureCFA: lv↔lv → Ψ; ov↔ov → Θ; mixed → error (rare).
        //   Reduced: any operand in lv_ext (lv, ov.y, ov.x) → Ψ; otherwise Θ.
        const bool L_in_lv_ext = v.lv_ext.contains(L);
        const bool R_in_lv_ext = v.lv_ext.contains(R);
        if (out.form == RepForm::Reduced) {
          if (L_in_lv_ext || R_in_lv_ext) {
            c.mat = MatId::Psi;
            c.row = v.lv_ext.index_of(L);
            c.col = v.lv_ext.index_of(R);
          } else {
            c.mat = MatId::Theta;
            c.row = v.ov.index_of(L);
            c.col = v.ov.index_of(R);
          }
        } else {
          const bool L_lv = v.lv.contains(L);
          const bool R_lv = v.lv.contains(R);
          if (L_lv && R_lv) {
            c.mat = MatId::Psi;
            c.row = v.lv.index_of(L);
            c.col = v.lv.index_of(R);
          } else if (!L_lv && !R_lv) {
            c.mat = MatId::Theta;
            c.row = v.ov.index_of(L);
            c.col = v.ov.index_of(R);
          } else {
            return std::unexpected(make_err(
                ModelError::Kind::UnsupportedRowKind,
                std::string("~~ between latent and observed not supported "
                            "in PureCFA: lhs='") + std::string(L) +
                    "', rhs='" + std::string(R) + "'"));
          }
        }
        if (c.row < 0) return unknown_var(L);
        if (c.col < 0) return unknown_var(R);
        c.used = true;
        break;
      }
      case parse::Op::Regression: {
        // Β[lhs, rhs] in lv_ext indices. Reduced form only.
        if (out.form != RepForm::Reduced) {
          return std::unexpected(make_err(
              ModelError::Kind::UnsupportedRowKind,
              "~ requires Reduced form (decide_form picked PureCFA)"));
        }
        const auto row_idx = v.lv_ext.index_of(L);
        const auto col_idx = v.lv_ext.index_of(R);
        if (row_idx < 0) return unknown_var(L);
        if (col_idx < 0) return unknown_var(R);
        c.mat  = MatId::Beta;
        c.row  = row_idx;
        c.col  = col_idx;
        c.used = true;
        break;
      }
      case parse::Op::Intercept: {
        // `lhs ~ 1` — indicator intercept (Nu) if lhs is observed,
        // latent mean (Alpha) if lhs is latent. In Reduced form lv_ext
        // includes both user-latents and the phantom-latents promoted
        // from observed y/x; an `ov ~ 1` row still maps to Nu (it sets
        // the observed-side intercept, not the phantom-latent mean).
        const bool L_lv = (out.form == RepForm::Reduced)
                              ? v.lv.contains(L)
                              : v.lv.contains(L);
        if (L_lv) {
          const auto row_idx = v.lv_ext.index_of(L);
          if (row_idx < 0) return unknown_var(L);
          c.mat  = MatId::Alpha;
          c.row  = row_idx;
          c.col  = 0;
          c.used = true;
        } else {
          const auto row_idx = v.ov.index_of(L);
          if (row_idx < 0) return unknown_var(L);
          c.mat  = MatId::Nu;
          c.row  = row_idx;
          c.col  = 0;
          c.used = true;
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
