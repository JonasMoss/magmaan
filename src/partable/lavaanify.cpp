#include "latva/partable/lavaanify.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/parse/expr_format.hpp"
#include "latva/parse/flat_partable.hpp"
#include "latva/parse/op.hpp"
#include "latva/partable/partable.hpp"

#include "classify.hpp"

namespace latva::partable {

namespace {

using detail::OrderedSet;
using detail::VarSets;
using detail::VarInventory;
using detail::classify_vars;
using detail::build_var_inventory;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

PartableError make_err(PartableError::Kind k, std::string detail) {
  return PartableError{k, std::move(detail)};
}

// === LatentStructure row builder ===================================================
//
// Every step appends to a single PartableBuilder. After all steps run we
// emit the final LatentStructure, assigning id / plabel / free in a single sweep.
struct PendingRow {
  std::int8_t      user;
  std::string      lhs;
  parse::Op        op;
  std::string      rhs;
  std::int32_t     block = 1;
  std::int32_t     group = 1;
  // Raw modifier intent (resolved to free / fixed_value / start hint at the end).
  bool             user_fixed_value = false;
  double           fixed_value = kNaN;
  bool             user_start_value = false;
  double           start_value = kNaN;
  std::string      label;
  // True when the user supplied any modifier OTHER than a bare label
  // (Free / StartValue / FixedValue). When true, auto.fix.first does NOT
  // override this row — the user has spoken about it explicitly.
  bool             user_explicit = false;
  // Pre-computed flags for free assignment.
  bool             auto_fixed = false;  // auto.fix.first, fixed.x, label-propagated
  std::int8_t      exo = 0;
};

PendingRow make_pending(std::int8_t user, std::string lhs, parse::Op op,
                        std::string rhs) {
  PendingRow p;
  p.user = user;
  p.lhs  = std::move(lhs);
  p.op   = op;
  p.rhs  = std::move(rhs);
  return p;
}

// True if any row of the form `var ~~ var` is already in the partable —
// either user-supplied (e.g. `f ~~ f` in the model) OR previously added
// by auto.var on an earlier ov-set pass. Used to suppress duplicate
// variance rows when a variable appears in both `ov_ind` and `ov_y`
// (e.g. an indicator with an explicit `~ 1` intercept).
bool variance_already_present(const std::vector<PendingRow>& rows,
                              std::string_view var) {
  for (const auto& r : rows) {
    if (r.op == parse::Op::Covariance && r.lhs == var && r.rhs == var) {
      return true;
    }
  }
  return false;
}

// === Step 3: user modifier application =====================================
//
// Helper: pick the effective ModifierAtom for group `group_idx` (0-based).
// Non-GroupVec modifiers apply identically to all groups; GroupVec
// modifiers must have exactly `n_groups` atoms and the right one is
// selected by index.
partable_expected<parse::ModifierAtom>
select_group_atom(const parse::Modifier& m, std::int32_t group_idx,
                  std::int32_t n_groups, std::string_view context) {
  partable_expected<parse::ModifierAtom> out{parse::ModifierAtom{parse::Free{}}};
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::FixedValue>) {
          out = parse::ModifierAtom{v};
        } else if constexpr (std::is_same_v<T, parse::Free>) {
          out = parse::ModifierAtom{v};
        } else if constexpr (std::is_same_v<T, parse::Label>) {
          out = parse::ModifierAtom{v};
        } else if constexpr (std::is_same_v<T, parse::StartValue>) {
          out = parse::ModifierAtom{v};
        } else if constexpr (std::is_same_v<T, parse::GroupVec>) {
          if (static_cast<std::int32_t>(v.per_group.size()) != n_groups) {
            out = std::unexpected(make_err(
                PartableError::Kind::BadGroupSpec,
                std::string("c(...) modifier on '") + std::string(context) +
                    "' has " + std::to_string(v.per_group.size()) +
                    " atoms but n_groups = " + std::to_string(n_groups)));
            return;
          }
          out = v.per_group[static_cast<std::size_t>(group_idx)];
        }
      },
      m);
  return out;
}

void apply_atom(PendingRow& row, const parse::ModifierAtom& atom) {
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::FixedValue>) {
          row.user_fixed_value = true;
          row.fixed_value = v.value;
          row.user_explicit = true;
        } else if constexpr (std::is_same_v<T, parse::Free>) {
          // Explicit NA — user is opting OUT of any auto-fix-first the
          // pipeline would otherwise apply, but doesn't fix or start.
          row.user_explicit = true;
        } else if constexpr (std::is_same_v<T, parse::Label>) {
          row.label = std::string(v.text);
          // Note: a bare label does NOT count as user_explicit — lavaan
          // still applies auto.fix.first when the marker indicator
          // happens to carry a label.
        } else if constexpr (std::is_same_v<T, parse::StartValue>) {
          // start(v)*x and v?x supply a start value but do NOT opt out of
          // auto.fix.first. If the row is also a marker indicator, it gets
          // fixed; the start_value still wins as that row's fixed_value
          // (see finalize()).
          row.user_start_value = true;
          row.start_value = v.value;
        }
      },
      atom);
}

partable_expected<void>
apply_modifiers_to_rows(const parse::FlatPartable& flat,
                        std::vector<PendingRow>& rows,
                        std::int32_t group_idx, std::int32_t n_groups) {
  // rows[i] corresponds 1:1 to flat.rows[i] in this initial slice (before
  // we add any auto rows). Auto rows are appended later.
  for (std::size_t i = 0; i < flat.rows.size(); ++i) {
    const auto& fr = flat.rows[i];
    if (fr.mod_idx == 0) continue;
    auto atom_or = select_group_atom(flat.mods[fr.mod_idx],
                                     group_idx, n_groups,
                                     std::string(fr.lhs) + " " +
                                         std::string(parse::to_string(fr.op)) +
                                         " " + std::string(fr.rhs));
    if (!atom_or.has_value()) return std::unexpected(atom_or.error());
    apply_atom(rows[i], *atom_or);
  }
  return {};
}

// === Step 4 (auto-add default rows) is implemented inline in lavaanify();
// it depends on a re-walk of flat.rows to detect endogenous LVs which is
// cleaner to do at the orchestration layer.

// === Step 5: auto.fix.first ================================================
//
// For each LV: find the first Measurement row with that LV as LHS, set
// fixed_value=1.0 — but only if the user did NOT explicitly modify that
// row (Free, StartValue, or FixedValue). A bare label does NOT count as
// explicit; lavaan still auto-fixes a marker indicator that happens to
// carry a label.
void apply_auto_fix_first(const VarSets& v, std::vector<PendingRow>& rows) {
  for (const auto& lv : v.lv.items) {
    for (auto& row : rows) {
      if (row.op != parse::Op::Measurement) continue;
      if (row.lhs != lv) continue;
      if (!row.user_explicit) {
        row.user_fixed_value = true;
        row.fixed_value      = 1.0;
        row.auto_fixed       = true;
      }
      break;  // only the first matching row, even if we didn't fix it
    }
  }
}

// === Step 5b: label-propagated fix =========================================
//
// After auto.fix.first, look for groups of rows sharing a user label. If
// any row in the group is fixed, propagate that fix to every other row
// in the group. Mirrors lavaan's behaviour: an auto-equality constraint
// is then NOT synthesized for that group (no point — they're all fixed).
void propagate_fix_via_labels(std::vector<PendingRow>& rows) {
  std::unordered_map<std::string, std::vector<std::size_t>> by_label;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].label.empty()) continue;
    by_label[rows[i].label].push_back(i);
  }
  for (const auto& [lbl, idxs] : by_label) {
    if (idxs.size() < 2) continue;
    // Find first fixed row in the group, if any.
    std::size_t fix_idx = idxs.size();
    double fix_val = kNaN;
    for (auto i : idxs) {
      if (rows[i].user_fixed_value) {
        fix_idx = i;
        fix_val = rows[i].fixed_value;
        break;
      }
    }
    if (fix_idx == idxs.size()) continue;   // none fixed
    for (auto i : idxs) {
      if (i == fix_idx) continue;
      if (!rows[i].user_fixed_value) {
        rows[i].user_fixed_value = true;
        rows[i].fixed_value      = fix_val;
        rows[i].auto_fixed       = true;
      }
    }
  }
}

// === Step 6: fixed.x mirror ================================================
//
// For each x in ov_x: add `x ~~ x` row with exo=1, free=0, fixed_value=NaN
// (resolved from the sample at fit time). For each pair (x_i, x_j) with
// i<j: add `x_i ~~ x_j` row, same flags.
void apply_fixed_x(const VarSets& v, std::vector<PendingRow>& rows) {
  for (std::size_t i = 0; i < v.ov_x.items.size(); ++i) {
    PendingRow p = make_pending(0, v.ov_x.items[i], parse::Op::Covariance,
                                v.ov_x.items[i]);
    p.exo = 1;
    p.user_fixed_value = true;   // fixed (not estimated)
    p.fixed_value = kNaN;        // value comes from data, not specified here
    p.auto_fixed = true;
    rows.push_back(std::move(p));
    for (std::size_t j = i + 1; j < v.ov_x.items.size(); ++j) {
      PendingRow q = make_pending(0, v.ov_x.items[i], parse::Op::Covariance,
                                  v.ov_x.items[j]);
      q.exo = 1;
      q.user_fixed_value = true;
      q.fixed_value = kNaN;
      q.auto_fixed = true;
      rows.push_back(std::move(q));
    }
  }
}

// === Step 7: synthesize auto-equality constraint rows ======================
//
// After all rows have label[] populated, find labels appearing on multiple
// rows. For each such label, anchor on the first row and emit one `==`
// constraint row per additional row, referencing the two `plabel`s.
//
// We also need plabels assigned by this point (id-driven). The caller
// assigns plabels just before calling this function.
void add_auto_equality_constraints(const std::vector<PendingRow>& rows,
                                   const std::vector<std::string>& plabels,
                                   std::vector<PendingRow>& out_eq_rows) {
  // Map from label text → list of indices into `rows` where it occurs.
  std::unordered_map<std::string, std::vector<std::size_t>> by_label;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].label.empty()) continue;
    by_label[rows[i].label].push_back(i);
  }
  for (const auto& [lbl, idxs] : by_label) {
    if (idxs.size() < 2) continue;
    // If any row in the group is fixed, lavaan does NOT add a constraint
    // (all parameters are then trivially fixed). Mirror that.
    bool any_fixed = false;
    for (auto i : idxs) if (rows[i].user_fixed_value) any_fixed = true;
    if (any_fixed) continue;
    const std::size_t ref = idxs.front();
    for (std::size_t k = 1; k < idxs.size(); ++k) {
      PendingRow eq;
      eq.user  = 2;                           // auto-equality
      eq.lhs   = plabels[ref];
      eq.op    = parse::Op::EqConstraint;
      eq.rhs   = plabels[idxs[k]];
      eq.block = 0;
      eq.group = 0;
      out_eq_rows.push_back(std::move(eq));
    }
  }
}

// === Step 9: append user constraint rows ===================================
//
// Each constraint in flat.constraints becomes a LatentStructure row with op set
// to its constraint kind, lhs/rhs filled from the canonical text of the
// Expr trees (or for Define, lhs is the new param name).
void append_user_constraints(const parse::FlatPartable& flat,
                             std::vector<PendingRow>& rows) {
  for (const auto& c : flat.constraints) {
    PendingRow p;
    p.user  = 1;
    p.block = 0;
    p.group = 0;
    if (c.kind == parse::ConstraintKind::Define) {
      p.lhs   = std::string(c.name);
      p.op    = parse::Op::DefineParam;
      p.rhs   = parse::expr_to_canonical(c.rhs);
      p.label = std::string(c.name);  // lavaan sets label = name for Define
    } else {
      p.lhs = parse::expr_to_canonical(c.lhs);
      p.rhs = parse::expr_to_canonical(c.rhs);
      switch (c.kind) {
        case parse::ConstraintKind::Eq: p.op = parse::Op::EqConstraint; break;
        case parse::ConstraintKind::Lt: p.op = parse::Op::LtConstraint; break;
        case parse::ConstraintKind::Gt: p.op = parse::Op::GtConstraint; break;
        case parse::ConstraintKind::Define: break;  // unreachable
      }
    }
    rows.push_back(std::move(p));
  }
}

// === Final pass: assign free indices; split fixed_value / start hints =======
//
// Walks rows in order. 1-based free indices go to rows where `auto_fixed ==
// false` AND the user didn't fix the value AND op is not a constraint. The old
// `ustart` is split: a fixed row's value lands in `out.fixed_value[i]`; a free
// row's `start(v)`/`v?x` hint lands in `starts.hint[free-1]` (sized n_free, NaN
// where unspecified). The name-free var-id columns (from `inv`) fill the
// `LatentStructure`; the verbal columns (`lhs`/`rhs`/`label`/`plabel`/`user`) +
// the var-name table + the group metadata fill the `LatentNames` companion.
// `id` (= row position) and `plabel` (= `.p<id>.`) are no longer stored — they
// are pure functions of position, materialized in `to_lavaan_partable`.
void finalize(const std::vector<PendingRow>& src, const VarInventory& inv,
              LatentStructure& out, Starts& starts, LatentNames& names) {
  const std::size_t n = src.size();
  out.op.resize(n);
  out.group.resize(n);
  out.free.resize(n);
  out.exo.resize(n);
  out.fixed_value.resize(n);
  out.lhs_var.resize(n);
  out.rhs_var.resize(n);
  starts.hint.clear();

  names.var_name   = inv.names;
  names.row_lhs.resize(n);
  names.row_rhs.resize(n);
  names.row_label.resize(n);
  names.row_plabel.resize(n);
  names.row_user.resize(n);

  // Variable table.
  out.n_vars       = static_cast<std::int32_t>(inv.names.size());
  out.var_role     = inv.roles;
  out.is_user_latent = inv.is_user_lv;
  out.ov_order     = inv.ov_order;
  out.lv_ext_order = inv.lv_ext_order;
  out.ov_pos.assign(static_cast<std::size_t>(out.n_vars), -1);
  out.lv_ext_pos.assign(static_cast<std::size_t>(out.n_vars), -1);
  for (std::size_t k = 0; k < inv.ov_order.size(); ++k)
    out.ov_pos[static_cast<std::size_t>(inv.ov_order[k])] = static_cast<std::int32_t>(k);
  for (std::size_t k = 0; k < inv.lv_ext_order.size(); ++k)
    out.lv_ext_pos[static_cast<std::size_t>(inv.lv_ext_order[k])] = static_cast<std::int32_t>(k);

  std::int32_t free_idx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const PendingRow& r = src[i];
    out.op[i]    = r.op;
    out.group[i] = r.group;
    out.exo[i]   = r.exo;

    const bool is_constraint =
        (r.op == parse::Op::EqConstraint || r.op == parse::Op::LtConstraint ||
         r.op == parse::Op::GtConstraint || r.op == parse::Op::DefineParam);

    // Var-id columns: only formula rows have variable lhs/rhs. `~1` rows have
    // an empty rhs. Constraint rows have neither (their sides are expressions).
    if (is_constraint) {
      out.lhs_var[i] = -1;
      out.rhs_var[i] = -1;
    } else {
      out.lhs_var[i] = inv.lookup(r.lhs);
      out.rhs_var[i] = (r.op == parse::Op::Intercept || r.rhs.empty())
                           ? -1 : inv.lookup(r.rhs);
    }

    // free: 0 if user-fixed, auto-fixed, exo, or constraint; else assign.
    if (is_constraint || r.user_fixed_value || r.exo == 1) {
      out.free[i] = 0;
      // Fixed/exo/constraint row → fixed_value carries what the old `ustart`
      // would have: user start_value beats user fixed_value beats NaN (so a
      // `start(v)*x` on an auto-fixed marker still pins the cell at v — the
      // long-standing lavaanify behavior); exo/constraint rows get NaN.
      if (r.user_start_value)        out.fixed_value[i] = r.start_value;
      else if (r.user_fixed_value)   out.fixed_value[i] = r.fixed_value;
      else                           out.fixed_value[i] = kNaN;
    } else {
      out.free[i] = ++free_idx;
      out.fixed_value[i] = kNaN;     // free — value comes from estimation
      // Start hint for the (free_idx)-th free parameter, NaN if none. Pushed
      // in free-index order, so starts.hint.size() == free_idx here.
      starts.hint.push_back(r.user_start_value ? r.start_value : kNaN);
    }

    // LatentNames companion (verbal model — display / round-trip only).
    // plabel: ".p<position>." for non-constraint rows; "" for constraint rows.
    names.row_lhs[i]    = r.lhs;
    names.row_rhs[i]    = r.rhs;
    names.row_label[i]  = r.label;
    names.row_plabel[i] = is_constraint ? std::string{}
                                        : (".p" + std::to_string(i + 1) + ".");
    names.row_user[i]   = r.user;
  }
}

}  // namespace

namespace {

// Build the formula + auto-* rows for one group. Steps 1-6 of the
// single-group pipeline (user rows, auto.var, auto.cov.lv.x, auto.fix.first,
// propagate_fix, fixed.x) — the multi-group entry point calls this once per
// group, sets `block`/`group` on each output row, and concatenates.
partable_expected<std::vector<PendingRow>>
build_group_template(const parse::FlatPartable& flat,
                     const LavaanifyOptions&    opts,
                     const VarSets&             v,
                     std::int32_t               group_idx) {
  std::vector<PendingRow> rows;
  rows.reserve(flat.rows.size() + 16);
  for (const auto& r : flat.rows) {
    rows.push_back(make_pending(/*user=*/1, std::string(r.lhs), r.op,
                                std::string(r.rhs)));
  }
  if (auto e = apply_modifiers_to_rows(flat, rows, group_idx, opts.n_groups);
      !e.has_value()) {
    return std::unexpected(e.error());
  }
  if (opts.auto_var) {
    auto add_var = [&](std::string_view name) {
      if (variance_already_present(rows, name)) return;
      rows.push_back(make_pending(/*user=*/0, std::string(name),
                                  parse::Op::Covariance, std::string(name)));
    };
    for (const auto& ov : v.ov_ind.items)  add_var(ov);
    for (const auto& ov : v.ov_y.items)    add_var(ov);
    for (const auto& ov : v.ov_misc.items) add_var(ov);
    for (const auto& lv : v.lv.items)      add_var(lv);
  }
  if (opts.auto_cov_lv_x) {
    OrderedSet endo_lv;
    for (const auto& fr : flat.rows) {
      if (fr.op == parse::Op::Regression && v.lv.contains(fr.lhs)) {
        endo_lv.insert(fr.lhs);
      }
    }
    OrderedSet exo_lv;
    for (const auto& lv : v.lv.items) {
      if (!endo_lv.contains(lv)) exo_lv.insert(lv);
    }
    for (std::size_t i = 0; i + 1 < exo_lv.items.size(); ++i) {
      for (std::size_t j = i + 1; j < exo_lv.items.size(); ++j) {
        rows.push_back(make_pending(/*user=*/0, exo_lv.items[i],
                                    parse::Op::Covariance, exo_lv.items[j]));
      }
    }
  }
  if (opts.auto_fix_first) {
    apply_auto_fix_first(v, rows);
  }
  propagate_fix_via_labels(rows);
  if (opts.fixed_x) {
    apply_fixed_x(v, rows);
  }
  // meanstructure: auto-add ν (free) for every observed variable that
  // doesn't already have an `~1` row, and α (fixed at 0) for every user
  // latent that doesn't either. Matches lavaan's `meanstructure = TRUE`
  // default — α is auto-fixed for identification while ν stays free.
  if (opts.meanstructure) {
    auto intercept_present = [&](std::string_view name) {
      for (const auto& r : rows) {
        if (r.op == parse::Op::Intercept && r.lhs == name) return true;
      }
      return false;
    };
    auto add_intercept = [&](std::string_view name, bool fix_at_zero) {
      if (intercept_present(name)) return;
      PendingRow p = make_pending(/*user=*/0, std::string(name),
                                  parse::Op::Intercept, std::string());
      if (fix_at_zero) {
        p.user_fixed_value = true;
        p.fixed_value      = 0.0;
        p.user_explicit    = true;
      }
      rows.push_back(std::move(p));
    };
    for (const auto& ov : v.ov_ind.items)  add_intercept(ov, false);
    for (const auto& ov : v.ov_y.items)    add_intercept(ov, false);
    for (const auto& ov : v.ov_x.items)    add_intercept(ov, false);
    for (const auto& ov : v.ov_misc.items) add_intercept(ov, false);
    for (const auto& lv : v.lv.items)      add_intercept(lv, true);
  }
  return rows;
}

}  // namespace

partable_expected<LatentStructure> lavaanify(const parse::FlatPartable& flat,
                                      const LavaanifyOptions& opts,
                                      Starts* out_starts,
                                      LatentNames* out_names) {
  if (opts.n_groups < 1) {
    return std::unexpected(make_err(
        PartableError::Kind::BadGroupSpec,
        "n_groups must be >= 1; got " + std::to_string(opts.n_groups)));
  }
  if (!opts.group_labels.empty() &&
      static_cast<std::int32_t>(opts.group_labels.size()) != opts.n_groups) {
    return std::unexpected(make_err(
        PartableError::Kind::BadGroupSpec,
        "group_labels has " + std::to_string(opts.group_labels.size()) +
            " entries but n_groups = " + std::to_string(opts.n_groups)));
  }
  if (flat.rows.empty() && flat.constraints.empty()) {
    return std::unexpected(make_err(
        PartableError::Kind::EmptyModel,
        "model contains no formulas or constraints"));
  }

  // Step 2: classify variables (same set across groups in v0 — no
  // group-specific variable structure yet) and build the name-free inventory
  // (var ids, roles, canonical orderings) that matrix_rep otherwise re-derives.
  const VarSets v = classify_vars(flat.rows);
  bool has_regression = false;
  for (const auto& r : flat.rows)
    if (r.op == parse::Op::Regression) { has_regression = true; break; }
  const VarInventory inv = build_var_inventory(v, has_regression);

  // Steps 1-6 per group, then concatenate. block / group are 1-based per
  // lavaan convention. Each group gets its own copy of formula + auto-*
  // rows; cross-group equality (when the user supplies shared labels) is
  // picked up by the post-replication auto-equality scan below.
  std::vector<PendingRow> rows;
  for (std::int32_t g = 1; g <= opts.n_groups; ++g) {
    auto group_rows_or = build_group_template(flat, opts, v, /*group_idx=*/g - 1);
    if (!group_rows_or.has_value()) {
      return std::unexpected(group_rows_or.error());
    }
    auto& group_rows = *group_rows_or;
    for (auto& r : group_rows) {
      r.block = g;
      r.group = g;
      rows.push_back(std::move(r));
    }
  }

  // Plabels are unique globally so equality constraints can reference any
  // row across all groups.
  std::vector<std::string> plabels(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    plabels[i] = ".p" + std::to_string(i + 1) + ".";
  }
  // Step 7: auto-equality constraints. Scans all rows; when the same user
  // label appears in two rows (within or across groups) it emits a `==`
  // constraint linking their plabels.
  std::vector<PendingRow> eq_rows;
  add_auto_equality_constraints(rows, plabels, eq_rows);

  // Step 9: user constraint rows.
  std::vector<PendingRow> user_constraint_rows;
  append_user_constraints(flat, user_constraint_rows);

  // Step 10: row ordering — formula/auto rows first (already grouped),
  // then user constraints, then auto-equality constraints.
  for (auto& r : user_constraint_rows) rows.push_back(std::move(r));
  for (auto& r : eq_rows)              rows.push_back(std::move(r));

  // Final pass: id/plabel/free assignment + fixed_value / start-hint split +
  // var-id columns + the LatentNames companion.
  LatentStructure out;
  Starts starts;
  LatentNames names;
  finalize(rows, inv, out, starts, names);
  if (out_starts) *out_starts = std::move(starts);

  // Group identity (verbal metadata — lives on `LatentNames`; nothing on the
  // numeric path branches on it). Single-group: leave both empty (the "no
  // groups == one group" sentinel).
  if (opts.n_groups > 1) {
    names.group_var = opts.group_var.empty() ? "group" : opts.group_var;
    if (static_cast<std::int32_t>(opts.group_labels.size()) == opts.n_groups) {
      names.group_labels = opts.group_labels;
    } else {
      names.group_labels.reserve(static_cast<std::size_t>(opts.n_groups));
      for (std::int32_t g = 1; g <= opts.n_groups; ++g) {
        names.group_labels.push_back(std::to_string(g));
      }
    }
  } else {
    names.group_var    = opts.group_var;        // may be a user-named single group
    names.group_labels = opts.group_labels;     // 0 or 1 entries
  }

  // Resolve the linear-equality reparameterization (shared labels, explicit
  // `a == b`) into `out.eq_groups`; flag `<` / `>` / non-bare `==` as
  // unenforced (fit() errors on those). Needs the verbal columns on `names`.
  compute_eq_groups(out, names);
  if (out_names) *out_names = std::move(names);
  return out;
}

namespace {

// A canonical Expr text uses no whitespace; a bare identifier therefore has
// no operator/paren characters. (`a`, `.p3.` are bare; `2*b`, `(a+b)`, `-1`
// are not.)
bool is_bare_identifier(const std::string& s) noexcept {
  if (s.empty()) return false;
  for (char c : s) {
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' ||
        c == '(' || c == ')' || c == ' ' || c == '\t') {
      return false;
    }
  }
  return true;
}

}  // namespace

void compute_eq_groups(LatentStructure& s, const LatentNames& names) {
  const std::int32_t npar = s.n_free();
  s.eq_groups.assign(static_cast<std::size_t>(npar < 0 ? 0 : npar), 0);
  for (std::int32_t k = 0; k < npar; ++k) s.eq_groups[static_cast<std::size_t>(k)] = k;
  s.has_unenforced_constraints = false;

  // `.pN.` plabel → 1-based free index, and `label` → 1-based free index,
  // over free (free > 0), non-constraint rows. Plabels/labels live on `names`.
  std::unordered_map<std::string, std::int32_t> plabel_to_free, label_to_free;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s.free[i] <= 0) continue;
    if (i < names.row_plabel.size() && !names.row_plabel[i].empty())
      plabel_to_free.try_emplace(names.row_plabel[i], s.free[i]);
    if (i < names.row_label.size() && !names.row_label[i].empty())
      label_to_free.try_emplace(names.row_label[i], s.free[i]);
  }
  auto resolve = [&](const std::string& tok) -> std::int32_t {
    if (!is_bare_identifier(tok)) return -1;
    if (auto it = plabel_to_free.find(tok); it != plabel_to_free.end()) return it->second;
    if (auto it = label_to_free.find(tok);  it != label_to_free.end())  return it->second;
    return -1;
  };

  // Union-find over 1-based free indices (index 0 unused).
  std::vector<std::int32_t> parent(static_cast<std::size_t>(npar) + 1);
  for (std::int32_t k = 0; k <= npar; ++k) parent[static_cast<std::size_t>(k)] = k;
  auto find = [&parent](std::int32_t x) -> std::int32_t {
    while (parent[static_cast<std::size_t>(x)] != x) {
      parent[static_cast<std::size_t>(x)] =
          parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
      x = parent[static_cast<std::size_t>(x)];
    }
    return x;
  };
  auto unite = [&](std::int32_t a, std::int32_t b) {
    a = find(a); b = find(b);
    if (a != b) parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
  };

  for (std::size_t i = 0; i < s.size(); ++i) {
    const parse::Op op = s.op[i];
    if (op == parse::Op::LtConstraint || op == parse::Op::GtConstraint) {
      s.has_unenforced_constraints = true;
      continue;
    }
    if (op != parse::Op::EqConstraint) continue;
    const std::string lhs_txt = i < names.row_lhs.size() ? names.row_lhs[i] : std::string{};
    const std::string rhs_txt = i < names.row_rhs.size() ? names.row_rhs[i] : std::string{};
    const std::int32_t li = resolve(lhs_txt);
    const std::int32_t ri = resolve(rhs_txt);
    if (li < 0 || ri < 0) { s.has_unenforced_constraints = true; continue; }
    unite(li, ri);
  }
  if (npar == 0) return;

  // Compact union-find roots into contiguous 0-based group indices, ascending
  // in free-index order.
  std::unordered_map<std::int32_t, std::int32_t> root_to_group;
  std::int32_t next_group = 0;
  for (std::int32_t k = 1; k <= npar; ++k) {
    const std::int32_t r = find(k);
    auto [it, inserted] = root_to_group.try_emplace(r, next_group);
    if (inserted) ++next_group;
    s.eq_groups[static_cast<std::size_t>(k - 1)] = it->second;
  }
}

}  // namespace latva::partable
