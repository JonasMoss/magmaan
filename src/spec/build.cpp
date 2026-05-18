#include "magmaan/spec/build.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/expr_format.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/spec/composite_expand.hpp"
#include "magmaan/spec/lin_constraints.hpp"
#include "magmaan/spec/partable.hpp"

#include "classify.hpp"
#include "detail_constraint_text.hpp"

namespace magmaan::spec {

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
  // equal("...") target: when non-empty, this row is tied to the parameter
  // whose canonical `lhs op rhs` name equals this string. Resolved into a
  // shared label by build_group_template.
  std::string      equal_ref;
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

bool covariance_already_present(const std::vector<PendingRow>& rows,
                                std::string_view lhs,
                                std::string_view rhs) {
  for (const auto& r : rows) {
    if (r.op != parse::Op::Covariance) continue;
    if ((r.lhs == lhs && r.rhs == rhs) || (r.lhs == rhs && r.rhs == lhs)) {
      return true;
    }
  }
  return false;
}

bool regression_already_present(const std::vector<PendingRow>& rows,
                                std::string_view lhs,
                                std::string_view rhs) {
  for (const auto& r : rows) {
    if (r.op != parse::Op::Regression) continue;
    if (r.lhs == lhs && r.rhs == rhs) return true;
  }
  return false;
}

// Build an auto-generated `lhs ~~ rhs` covariance row. Under `orthogonal` the
// row is fixed at 0 instead of free — lavaan's orthogonal identification keeps
// the latent-covariance rows in the partable but pins them (free = 0,
// ustart = 0).
// Whitespace-stripped copy — normalizes an equal("...") target so it matches
// the space-free canonical `lhs op rhs` name of a row.
std::string strip_ws(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    if (c != ' ' && c != '\t') out += c;
  return out;
}

// Canonical `lhs op rhs` name of a row, no spaces (`visual=~x2`, `x1~~x1`,
// `y~x`, `x1~1`). The form an equal("...") target string is matched against.
std::string row_canonical_name(const PendingRow& r) {
  std::string s = r.lhs;
  s += parse::to_string(r.op);
  s += r.rhs;
  return s;
}

PendingRow make_auto_cov(std::string lhs, std::string rhs, bool orthogonal) {
  PendingRow p = make_pending(/*user=*/0, std::move(lhs),
                              parse::Op::Covariance, std::move(rhs));
  if (orthogonal) {
    p.user_fixed_value = true;
    p.fixed_value      = 0.0;
    p.auto_fixed       = true;
  }
  return p;
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
        } else if constexpr (std::is_same_v<T, parse::EqualRef>) {
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
        } else if constexpr (std::is_same_v<T, parse::EqualRef>) {
          // equal("...") ties this parameter to another — an explicit
          // equality statement, so the row opts out of auto.fix.first (it
          // stays free and tied rather than being pinned to a marker 1.0).
          row.equal_ref = strip_ws(v.text);
          row.user_explicit = true;
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

// === Step 4 (auto-add default rows) is implemented inline in build();
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

// === Step 5 (std.lv variant) ===============================================
//
// `std.lv` identification: instead of fixing a marker loading, scale each
// latent by fixing its `~~`-self variance to 1.0 — but only if the user
// didn't already speak about that row (Free / StartValue / FixedValue), in
// which case the user wins (and the latent may then be unidentified, exactly
// as in lavaan). auto.var has normally already added the `lv ~~ lv` row by
// the time we run; if not (auto.var off and the user didn't write one), we
// append one fixed at 1.0 so the latent is still scaled. The first loading is
// left free — `lavaanify` does not call `apply_auto_fix_first` under std.lv.
void apply_std_lv(const VarSets& v, std::vector<PendingRow>& rows) {
  for (const auto& lv : v.lv.items) {
    PendingRow* var_row = nullptr;
    for (auto& row : rows) {
      if (row.op == parse::Op::Covariance && row.lhs == lv && row.rhs == lv) {
        var_row = &row;
        break;
      }
    }
    if (var_row != nullptr) {
      if (!var_row->user_explicit) {
        var_row->user_fixed_value = true;
        var_row->fixed_value      = 1.0;
        var_row->auto_fixed       = true;
      }
    } else {
      PendingRow p = make_pending(/*user=*/0, lv, parse::Op::Covariance, lv);
      p.user_fixed_value = true;
      p.fixed_value      = 1.0;
      p.auto_fixed       = true;
      rows.push_back(std::move(p));
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

void apply_random_x(const VarSets& v, std::vector<PendingRow>& rows) {
  for (std::size_t i = 0; i < v.ov_x.items.size(); ++i) {
    if (!variance_already_present(rows, v.ov_x.items[i])) {
      rows.push_back(make_pending(0, v.ov_x.items[i],
                                  parse::Op::Covariance, v.ov_x.items[i]));
    }
    for (std::size_t j = i + 1; j < v.ov_x.items.size(); ++j) {
      if (covariance_already_present(rows, v.ov_x.items[i], v.ov_x.items[j])) {
        continue;
      }
      rows.push_back(make_pending(0, v.ov_x.items[i],
                                  parse::Op::Covariance, v.ov_x.items[j]));
    }
  }
}

// === Step 7: (no rows here) =================================================
//
// Auto-equality from a shared user label used to be a synthesized `==` row
// (`user=2`) referencing two `.pN.` plabels. It isn't stored anymore — the
// equality lives in `LatentStructure.eq_groups` (computed by `compute_eq_groups`,
// which scans shared labels directly). `to_lavaan_partable()` re-synthesizes
// the lavaan-shaped `.pN.`-pair rows from the labels for display / round-trip.

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
                     const BuildOptions&    opts,
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
    if (!opts.fixed_x) apply_random_x(v, rows);
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
        // Skip pairs the user already wrote: an explicit `f_i ~~ f_j` row
        // (labelled, fixed, or bare) wins, exactly as auto.var and random.x
        // defer to user variance/covariance rows. Without this guard a
        // user-specified exogenous-latent covariance gets a duplicate auto
        // row — a second free parameter for the same moment, which makes the
        // information matrix rank-deficient.
        if (covariance_already_present(rows, exo_lv.items[i], exo_lv.items[j])) {
          continue;
        }
        rows.push_back(make_auto_cov(exo_lv.items[i], exo_lv.items[j],
                                     opts.orthogonal));
      }
    }
  }
  if (opts.auto_cov_y) {
    OrderedSet endo_lv;
    for (const auto& fr : flat.rows) {
      if (fr.op == parse::Op::Regression && v.lv.contains(fr.lhs)) {
        endo_lv.insert(fr.lhs);
      }
    }
    for (std::size_t i = 0; i + 1 < endo_lv.items.size(); ++i) {
      for (std::size_t j = i + 1; j < endo_lv.items.size(); ++j) {
        const auto& lhs = endo_lv.items[i];
        const auto& rhs = endo_lv.items[j];
        if (covariance_already_present(rows, lhs, rhs)) continue;
        if (regression_already_present(rows, lhs, rhs) ||
            regression_already_present(rows, rhs, lhs)) {
          continue;
        }
        rows.push_back(make_auto_cov(lhs, rhs, opts.orthogonal));
      }
    }
  }
  // Latent scaling — three mutually-exclusive conventions. `std.lv`: fix each
  // LV variance at 1 (auto.fix.first forced off, lavaan-style). `effect_coding`:
  // leave *everything* free (loadings + LV variance) — the scale is set instead
  // by the `Σλ == #indicators` constraint rows synthesized in `build()`.
  // Otherwise: the marker convention (fix the first loading per latent at 1).
  if (opts.std_lv) {
    apply_std_lv(v, rows);
  } else if (opts.effect_coding) {
    // nothing: all loadings free, `lv ~~ lv` free (from auto.var).
  } else if (opts.auto_fix_first) {
    apply_auto_fix_first(v, rows);
  }
  // Resolve equal(...) references. A row carrying an `equal_ref` is tied to
  // the parameter it names (canonical `lhs op rhs`): magmaan expresses the
  // tie as a shared label — both rows get the same label, which
  // `compute_eq_groups` later merges into one free parameter. That is the
  // equivalent fitted model to lavaan's `.pX. == .pY.` constraint row (same
  // estimates, same degrees of freedom). Runs before `propagate_fix_via_labels`
  // so a tie to a fixed parameter still propagates the fix.
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].equal_ref.empty()) continue;
    std::size_t ref = rows.size();
    for (std::size_t j = 0; j < rows.size(); ++j) {
      if (j == i) continue;
      if (row_canonical_name(rows[j]) == rows[i].equal_ref) { ref = j; break; }
    }
    if (ref == rows.size()) {
      return std::unexpected(make_err(
          PartableError::Kind::UnknownLabelInConstraint,
          "equal(\"" + rows[i].equal_ref +
              "\") references a parameter that is not in the model"));
    }
    const std::string shared =
        rows[ref].label.empty() ? rows[i].equal_ref : rows[ref].label;
    rows[ref].label = shared;
    rows[i].label   = shared;
  }
  propagate_fix_via_labels(rows);
  if (opts.fixed_x) {
    apply_fixed_x(v, rows);
  }
  // meanstructure: auto-add ν for every observed variable that doesn't already
  // have an `~1` row, and α for every user latent that doesn't either. SEM/CFA
  // defaults are free ν / fixed-zero α; growth mode flips those defaults.
  // Fixed.x observed exogenous means remain data-given.
  if (opts.meanstructure) {
    auto intercept_present = [&](std::string_view name) {
      for (const auto& r : rows) {
        if (r.op == parse::Op::Intercept && r.lhs == name) return true;
      }
      return false;
    };
    auto add_intercept = [&](std::string_view name, bool fix_at_zero,
                             bool fixed_x_exo = false) {
      if (intercept_present(name)) return;
      PendingRow p = make_pending(/*user=*/0, std::string(name),
                                  parse::Op::Intercept, std::string());
      if (fixed_x_exo) {
        p.exo = 1;
        p.user_fixed_value = true;
        p.fixed_value      = kNaN;
        p.auto_fixed       = true;
      } else if (fix_at_zero) {
        p.user_fixed_value = true;
        p.fixed_value      = 0.0;
        p.user_explicit    = true;
      }
      rows.push_back(std::move(p));
    };
    const bool fix_ov = !opts.int_ov_free;
    const bool fix_lv = !opts.int_lv_free;
    for (const auto& ov : v.ov_ind.items)  add_intercept(ov, fix_ov);
    for (const auto& ov : v.ov_y.items)    add_intercept(ov, fix_ov);
    for (const auto& ov : v.ov_x.items)
      add_intercept(ov, fix_ov, opts.fixed_x);
    for (const auto& ov : v.ov_misc.items) add_intercept(ov, fix_ov);
    for (const auto& lv : v.lv.items)      add_intercept(lv, fix_lv);
  }
  return rows;
}

}  // namespace

partable_expected<LatentStructure> build(const parse::FlatPartable& flat,
                                      const BuildOptions& opts,
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
  if (opts.effect_coding && opts.std_lv) {
    return std::unexpected(make_err(
        PartableError::Kind::BadGroupSpec,
        "effect_coding and std_lv are mutually exclusive identification conventions"));
  }

  // Composite (`<~`) support: rewrite each composite into a Henseler-Ogasawara
  // reflective sub-model up front, so every step below sees only an ordinary
  // reflective model. The expanded FlatPartable borrows string storage from
  // `flat` for its unmodified rows — fine here, `flat` outlives `build()`.
  CompositeExpansion comp_exp;
  bool has_composites = false;
  for (const auto& r : flat.rows)
    if (r.op == parse::Op::Composite) { has_composites = true; break; }
  const parse::FlatPartable* flatp = &flat;
  if (has_composites) {
    auto exp = expand_composites(flat);
    if (!exp.has_value()) return std::unexpected(exp.error());
    comp_exp = std::move(*exp);
    flatp = &comp_exp.flat;
  }
  const parse::FlatPartable& eflat = *flatp;

  // Step 2: classify variables (same set across groups in v0 — no
  // group-specific variable structure yet) and build the name-free inventory
  // (var ids, roles, canonical orderings) that matrix_rep otherwise re-derives.
  const VarSets v = classify_vars(eflat.rows);
  bool has_regression = false;
  for (const auto& r : eflat.rows)
    if (r.op == parse::Op::Regression) { has_regression = true; break; }
  const VarInventory inv = build_var_inventory(v, has_regression);

  // Steps 1-6 per group, then concatenate. block / group are 1-based per
  // lavaan convention. Each group gets its own copy of formula + auto-*
  // rows; cross-group equality (when the user supplies shared labels) is
  // picked up by the post-replication auto-equality scan below.
  std::vector<PendingRow> rows;
  for (std::int32_t g = 1; g <= opts.n_groups; ++g) {
    auto group_rows_or = build_group_template(eflat, opts, v, /*group_idx=*/g - 1);
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

  // Step 9: constraint rows. First the `effect.coding` rows synthesized below,
  // then the user's `==` / `<` / `>` / `:=` statements. (Auto-equality from
  // shared labels is NOT a row anymore — it's `eq_groups`.)
  std::vector<PendingRow> user_constraint_rows;

  // effect.coding: one linear-equality row per (latent, group),
  // `Σ (loading plabels) == #indicators`. The `=~` rows are all already in
  // `rows` at their final positions (the constraint rows below are appended
  // after them), so `.p<i+1>.` for the i-th row of `rows` is its plabel.
  if (opts.effect_coding) {
    // Henseler-Ogasawara loading blocks are self-scaling — they must not pick
    // up `Σλ == #indicators` effect-coding rows. Skip emergent + excrescent.
    std::unordered_set<std::string> ho_latents;
    for (const auto& c : comp_exp.composites) {
      ho_latents.insert(c.composite);
      for (const auto& e : c.excrescent) ho_latents.insert(e);
    }
    struct Bucket { std::string lhs; std::int32_t group = 0; std::vector<std::size_t> plabel_idx; };
    std::vector<Bucket> buckets;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (rows[i].op != parse::Op::Measurement) continue;
      if (ho_latents.count(rows[i].lhs) != 0) continue;
      std::size_t b = 0;
      for (; b < buckets.size(); ++b)
        if (buckets[b].lhs == rows[i].lhs && buckets[b].group == rows[i].group) break;
      if (b == buckets.size()) buckets.push_back(Bucket{rows[i].lhs, rows[i].group, {}});
      buckets[b].plabel_idx.push_back(i + 1);  // 1-based ⇒ `.p<i+1>.`
    }
    for (const auto& bk : buckets) {
      std::string lhs_txt;
      for (std::size_t j = 0; j < bk.plabel_idx.size(); ++j) {
        if (j != 0) lhs_txt += '+';
        lhs_txt += ".p" + std::to_string(bk.plabel_idx[j]) + ".";
      }
      PendingRow p;
      p.user  = 1;
      p.op    = parse::Op::EqConstraint;
      p.block = 0;
      p.group = 0;
      p.lhs   = std::move(lhs_txt);
      p.rhs   = std::to_string(bk.plabel_idx.size());
      user_constraint_rows.push_back(std::move(p));
    }
  }

  append_user_constraints(eflat, user_constraint_rows);

  // Step 10: row ordering — formula/auto rows first (already grouped), then
  // the constraint rows. `to_lavaan_partable` re-synthesizes the lavaan-shaped
  // shared-label `.pN.`-pair rows from the labels when projecting.
  for (auto& r : user_constraint_rows) rows.push_back(std::move(r));

  // Final pass: id/plabel/free assignment + fixed_value / start-hint split +
  // var-id columns + the LatentNames companion.
  LatentStructure out;
  Starts starts;
  LatentNames names;
  finalize(rows, inv, out, starts, names);
  names.composites = std::move(comp_exp.composites);
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

  // Resolve the linear-equality reparameterization. `compute_eq_groups` folds
  // shared labels + bare `a == b` into `out.eq_groups` and flags `<`/`>` /
  // non-bare `==` as unenforced; `resolve_lin_constraints` then reduces every
  // *linear* non-bare `==` row (`a == 2*b+c`, …) to a `lin_constraint_R` row and
  // clears the flag if only those remained (fit() still errors on `<`/`>` /
  // genuinely-nonlinear `==`). Both need the verbal columns on `names`.
  compute_eq_groups(out, names);
  resolve_lin_constraints(out, names);
  if (out_names) *out_names = std::move(names);
  return out;
}

void compute_eq_groups(LatentStructure& s, const LatentNames& names) {
  const std::int32_t npar = s.n_free();
  s.eq_groups.assign(static_cast<std::size_t>(npar < 0 ? 0 : npar), 0);
  for (std::int32_t k = 0; k < npar; ++k) s.eq_groups[static_cast<std::size_t>(k)] = k;
  // The constraint-classification flags (`has_inequality_constraints`,
  // `nonlinear_eq_rows`, `has_unenforced_constraints`) are owned entirely by
  // `resolve_lin_constraints`, which always runs right after this. This
  // function only computes the `eq_groups` merge partition.

  // Over free (free > 0), non-constraint rows: `.pN.` plabel → 1-based free
  // index (first wins), and user label → list of 1-based free indices, in row
  // order. Plabels/labels live on `names`.
  std::unordered_map<std::string, std::int32_t> plabel_to_free;
  std::unordered_map<std::string, std::vector<std::int32_t>> free_by_label;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s.free[i] <= 0) continue;
    if (i < names.row_plabel.size() && !names.row_plabel[i].empty())
      plabel_to_free.try_emplace(names.row_plabel[i], s.free[i]);
    if (i < names.row_label.size() && !names.row_label[i].empty())
      free_by_label[names.row_label[i]].push_back(s.free[i]);
  }
  auto resolve = [&](const std::string& tok) -> std::int32_t {
    if (!detail::is_bare_identifier(tok)) return -1;
    if (auto it = plabel_to_free.find(tok); it != plabel_to_free.end()) return it->second;
    if (auto it = free_by_label.find(tok); it != free_by_label.end() && !it->second.empty())
      return it->second.front();
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

  // Shared-label equality: every free row carrying the same user label is one
  // group. (Lavaan also synthesizes the `.pN.`-pair `==` rows for this — those
  // are projected back by `to_lavaan_partable`, not stored here.) A label group
  // with a fixed member had the fix propagated to all of them, so none are free
  // and the group simply doesn't show up here.
  for (const auto& [lbl, idxs] : free_by_label) {
    for (std::size_t k = 1; k < idxs.size(); ++k) unite(idxs.front(), idxs[k]);
  }

  for (std::size_t i = 0; i < s.size(); ++i) {
    const parse::Op op = s.op[i];
    if (op == parse::Op::LtConstraint || op == parse::Op::GtConstraint) {
      continue;  // not a merge; resolve_lin_constraints classifies inequalities
    }
    if (op != parse::Op::EqConstraint) continue;
    const std::string lhs_txt = i < names.row_lhs.size() ? names.row_lhs[i] : std::string{};
    const std::string rhs_txt = i < names.row_rhs.size() ? names.row_rhs[i] : std::string{};
    const std::int32_t li = resolve(lhs_txt);
    const std::int32_t ri = resolve(rhs_txt);
    // Not a pure parameter merge (one side isn't a bare free reference) —
    // leave it for resolve_lin_constraints to reduce or classify.
    if (li < 0 || ri < 0) continue;
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

}  // namespace magmaan::spec
