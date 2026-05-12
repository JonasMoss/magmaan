#include "latva/partable/lavaan_view.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "latva/parse/op.hpp"
#include "latva/partable/lavaanify.hpp"   // compute_eq_groups
#include "latva/partable/partable.hpp"

#include "classify.hpp"

namespace latva::partable {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool is_constraint_op(parse::Op op) noexcept {
  return op == parse::Op::EqConstraint || op == parse::Op::LtConstraint ||
         op == parse::Op::GtConstraint || op == parse::Op::DefineParam;
}

}  // namespace

LavaanParTable to_lavaan_partable(const LatentStructure& s,
                                  const LatentNames& names,
                                  const Starts& starts) {
  const std::size_t n = s.size();
  LavaanParTable out;
  out.id.resize(n);     out.user.resize(n);   out.lhs.resize(n);
  out.op.resize(n);     out.rhs.resize(n);    out.block.resize(n);
  out.group.resize(n);  out.free.resize(n);   out.exo.resize(n);
  out.ustart.resize(n); out.label.resize(n);  out.plabel.resize(n);

  for (std::size_t i = 0; i < n; ++i) {
    out.id[i]     = static_cast<std::int32_t>(i + 1);
    out.user[i]   = names.row_user[i];
    out.lhs[i]    = names.row_lhs[i];
    out.op[i]     = s.op[i];
    out.rhs[i]    = names.row_rhs[i];
    out.block[i]  = s.group[i];   // block == group (no multilevel modeled)
    out.group[i]  = s.group[i];
    out.free[i]   = s.free[i];
    out.exo[i]    = s.exo[i];
    out.label[i]  = names.row_label[i];
    out.plabel[i] = names.row_plabel[i];
    // lavaan's `ustart`: a fixed row's `fixed_value` (NaN for an unresolved
    // fixed.x moment), or a free row's user start hint (NaN where none).
    if (s.free[i] == 0) {
      out.ustart[i] = s.fixed_value[i];
    } else {
      const std::size_t k = static_cast<std::size_t>(s.free[i] - 1);
      out.ustart[i] = (k < starts.hint.size()) ? starts.hint[k] : kNaN;
    }
  }
  out.group_var    = names.group_var;
  out.group_labels = names.group_labels;
  out.extra_real   = s.extra_real;
  out.extra_int    = s.extra_int;
  out.extra_str    = s.extra_str;
  return out;
}

ParsedLavaanParTable from_lavaan_partable(const LavaanParTable& pt) {
  const std::size_t n = pt.size();
  ParsedLavaanParTable out;
  LatentStructure& s     = out.structure;
  LatentNames&     names = out.names;
  Starts&          starts = out.starts;

  // Re-derive the variable inventory from the formula rows. `~` anywhere ⇒
  // Reduced LISREL form (ov.y / ov.x promoted to phantom latents).
  bool reduced = false;
  for (std::size_t i = 0; i < n; ++i)
    if (pt.op[i] == parse::Op::Regression) { reduced = true; break; }
  struct ClassRow { parse::Op op; std::string lhs; std::string rhs; };
  std::vector<ClassRow> formula_rows;
  formula_rows.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (is_constraint_op(pt.op[i])) continue;
    formula_rows.push_back(ClassRow{pt.op[i], pt.lhs[i], pt.rhs[i]});
  }
  const detail::VarSets      vsets = detail::classify_vars(formula_rows);
  const detail::VarInventory inv   = detail::build_var_inventory(vsets, reduced);

  // Variable table.
  s.n_vars         = static_cast<std::int32_t>(inv.names.size());
  s.var_role       = inv.roles;
  s.is_user_latent = inv.is_user_lv;
  s.ov_order       = inv.ov_order;
  s.lv_ext_order   = inv.lv_ext_order;
  s.ov_pos.assign(static_cast<std::size_t>(s.n_vars), -1);
  s.lv_ext_pos.assign(static_cast<std::size_t>(s.n_vars), -1);
  for (std::size_t k = 0; k < inv.ov_order.size(); ++k)
    s.ov_pos[static_cast<std::size_t>(inv.ov_order[k])] = static_cast<std::int32_t>(k);
  for (std::size_t k = 0; k < inv.lv_ext_order.size(); ++k)
    s.lv_ext_pos[static_cast<std::size_t>(inv.lv_ext_order[k])] = static_cast<std::int32_t>(k);

  // Per-row structural columns.
  s.op.resize(n);  s.group.resize(n);  s.free.resize(n);  s.exo.resize(n);
  s.fixed_value.resize(n);  s.lhs_var.resize(n);  s.rhs_var.resize(n);
  std::int32_t n_free = 0;
  for (std::size_t i = 0; i < n; ++i) {
    s.op[i]    = pt.op[i];
    s.group[i] = pt.group[i];
    s.free[i]  = pt.free[i];
    s.exo[i]   = pt.exo[i];
    if (s.free[i] > n_free) n_free = s.free[i];
    if (is_constraint_op(pt.op[i])) {
      s.lhs_var[i] = -1;
      s.rhs_var[i] = -1;
    } else {
      s.lhs_var[i] = inv.lookup(pt.lhs[i]);
      s.rhs_var[i] = (pt.op[i] == parse::Op::Intercept || pt.rhs[i].empty())
                         ? -1 : inv.lookup(pt.rhs[i]);
    }
    // Split lavaan's `ustart`: a fixed row's value → `fixed_value`; a free
    // row's value (if any) → a start hint.
    s.fixed_value[i] = (s.free[i] == 0) ? pt.ustart[i] : kNaN;
  }

  // Start hints (sized n_free; NaN where none).
  starts.hint.assign(static_cast<std::size_t>(n_free < 0 ? 0 : n_free), kNaN);
  for (std::size_t i = 0; i < n; ++i) {
    if (pt.free[i] <= 0) continue;
    const double u = pt.ustart[i];
    if (std::isfinite(u))
      starts.hint[static_cast<std::size_t>(pt.free[i] - 1)] = u;
  }

  // LatentNames companion (verbal model — display / round-trip).
  names.var_name = inv.names;
  names.row_lhs.resize(n);  names.row_rhs.resize(n);  names.row_label.resize(n);
  names.row_plabel.resize(n);  names.row_user.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    names.row_lhs[i]    = pt.lhs[i];
    names.row_rhs[i]    = pt.rhs[i];
    names.row_label[i]  = pt.label[i];
    names.row_plabel[i] = pt.plabel[i];
    names.row_user[i]   = pt.user[i];
  }
  names.group_var    = pt.group_var;
  names.group_labels = pt.group_labels;

  // Methods-developer escape-hatch columns pass straight through.
  s.extra_real = pt.extra_real;
  s.extra_int  = pt.extra_int;
  s.extra_str  = pt.extra_str;

  // Resolve the equality-constraint reparameterization from the (possibly
  // hand-edited) `==` / `<` / `>` rows so fit() honors it.
  compute_eq_groups(s, names);
  return out;
}

}  // namespace latva::partable
