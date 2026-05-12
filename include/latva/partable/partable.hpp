#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "latva/parse/op.hpp"

namespace latva::partable {

// === Variable roles =========================================================
// Lavaan's variable-classification buckets, in the order they affect the
// canonical observed/latent ordering. Each variable has exactly one role —
// the *first* bucket it lands in when scanned in this order.
//   Indicator : RHS of `=~`                          (ov.ind)
//   EndoOv    : LHS of `~` / `~1` and not latent     (ov.y) — promoted to a
//               phantom latent in the Reduced LISREL form
//   ExoOv     : RHS of `~` and not latent/ov.ind/ov.y (ov.x) — likewise
//               promoted; also the `fixed.x` data-given variables
//   MiscOv    : observed, mentioned only in `~~`     (ov.misc) — gets a
//               variance but is *not* promoted to a phantom latent
//   Latent    : LHS of `=~`                          (lv)
enum class VarRole : std::uint8_t { Indicator, EndoOv, ExoOv, MiscOv, Latent };

// Names that go with a `LatentStructure` — the *verbal* model. Everything in here is
// for display / round-tripping / parsing; nothing on the model's numeric path
// (matrix_rep, evaluator, fit, inference) reads it. Sizes: `var_name` is
// `n_vars`; `row_*` are `n_rows`; `group_labels` is `n_groups`.
struct LatentNames {
  std::vector<std::string>  var_name;     // var id → name
  std::vector<std::string>  row_lhs;      // per-row textual lhs (var name for
                                          // formula rows; expression text /
                                          // defined-param name for constraint rows)
  std::vector<std::string>  row_rhs;      // per-row textual rhs ("" for `~1`)
  std::vector<std::string>  row_label;    // user-supplied label, or ""
  std::vector<std::string>  row_plabel;   // .pN. synthetic; "" for constraint rows
  std::vector<std::int8_t>  row_user;     // 0=auto, 1=user-supplied, 2=auto-equality
  std::string               group_var;    // grouping-variable name; "" ⇒ unnamed single group
  std::vector<std::string>  group_labels; // per-group level labels ("1".."n" if unsupplied)
};

// Lavaanified LatentStructure: model description in struct-of-arrays form.
//
// Mirrors lavaan's parTable() columns, but only the model-description ones.
// Estimation outputs (est, se, vcov) and the user's start *hints* live on
// separate types (`Estimates`, `Starts`, ...) and are composed by the user,
// not stored here. The one start-flavored thing the LatentStructure does carry is
// `fixed_value`: the value at which a *fixed* parameter sits — that's a model
// fact (`f =~ 1*x1`, a `std.lv` variance fix, a `fixed.x` covariance), not an
// optimization hint. See docs/agents/rules.md (or the project plan) for the
// rationale on the model/estimation split.
//
// Constraint statements (==, <, >, :=) are real LatentStructure rows just like in
// lavaan, with `block=0` and `group=0`. Auto-equality from a shared user
// label is encoded as a synthesized `==` row with `user=2` referencing two
// `plabel`s; the source rows keep distinct `free` indices.
//
// Group identity is part of the model: `n_groups()` ≥ 1 always; the
// single-group case is just `n_groups()==1` with `group_var` empty — not a
// different shape. `group_var` / `group_labels` are pure metadata (no code
// path branches on them); they round-trip the grouping variable's name and
// per-group level labels (lavaan keeps these on `@Data`, but latva treats
// them as belonging to the model).
//
// NOTE (refactor in progress, stage B1): the structural columns below are
// being migrated to a name-free form — `var_role` / `ov_order` / `lv_ext_order`
// / `ov_pos` / `lv_ext_pos` / `lhs_var` / `rhs_var` mirror the string columns
// `lhs` / `rhs` and the variable classification that `matrix_rep` currently
// re-derives. The string columns and `LatentNames`-bound columns (`label`,
// `plabel`, `user`, `group_var`, `group_labels`) still live here for now;
// a later stage moves them to `LatentNames` and renames this type.
struct LatentStructure {
  // Identity
  std::vector<std::int32_t> id;     // 1-based, matches lavaan
  std::vector<std::int8_t>  user;   // 0=auto, 1=user-supplied, 2=auto-equality

  // Statement (strings — being migrated to the var-id columns below)
  std::vector<std::string>  lhs;
  std::vector<parse::Op>    op;
  std::vector<std::string>  rhs;

  // Variable table (name-free mirror of the model — populated by lavaanify;
  // consumed by matrix_rep). `lhs_var` / `rhs_var` are var ids (or -1 for a
  // `~1` rhs / a constraint-row side that isn't a variable).
  std::int32_t              n_vars = 0;
  std::vector<VarRole>      var_role;        // size n_vars: first-bucket role
  std::vector<std::int8_t>  is_user_latent;  // size n_vars: 1 if LHS of some `=~`
  std::vector<std::int32_t> ov_order;        // var ids in canonical observed order
  std::vector<std::int32_t> lv_ext_order;    // var ids in extended-latent order
                                             // (PureCFA: latents; Reduced: latents,
                                             //  then ov.y, then ov.x)
  std::vector<std::int32_t> ov_pos;          // size n_vars: index in ov_order, or -1
  std::vector<std::int32_t> lv_ext_pos;      // size n_vars: index in lv_ext_order, or -1
  std::vector<std::int32_t> lhs_var;         // size n_rows: var id of lhs, or -1
  std::vector<std::int32_t> rhs_var;         // size n_rows: var id of rhs, or -1

  // Block / group context. `block`/`group` are 1-based per row (always equal
  // — no multilevel yet); 0 for constraint rows. `group_var` is the grouping
  // variable name ("" ⇒ unnamed single group); `group_labels[g-1]` is the
  // label of group `g` (empty ⇒ trivial single group).
  std::vector<std::int32_t> block;
  std::vector<std::int32_t> group;
  std::string               group_var;
  std::vector<std::string>  group_labels;

  // Estimation contract
  std::vector<std::int32_t> free;        // 0=fixed (or constraint row); else 1-based θ index
  std::vector<std::int8_t>  exo;         // 1 if data-given exogenous moment (fixed.x)
  std::vector<double>       fixed_value;  // value of a fixed param (free==0); NaN for free
                                          // rows, constraint rows, and not-yet-resolved
                                          // fixed.x moments (filled from the sample later).
                                          // Free-param start *hints* live on `Starts`, not here.

  // Linear-equality reparameterization, precomputed by lavaanify (or
  // `compute_eq_groups`): `eq_groups[k]` (0-based) is the merged-parameter
  // group of the (k+1)-th free param — free params in the same group are equal
  // (shared label, explicit `a == b`, cross-group invariance). Empty ⇒ identity
  // (every param its own group). `has_unenforced_constraints` is set when the
  // model carries an `<` / `>` row or a non-bare `==` expression (`a == 2*b`):
  // `fit()` then errors (those phases aren't implemented). See fit/constraints.
  std::vector<std::int32_t> eq_groups;
  bool                      has_unenforced_constraints = false;

  // Naming
  std::vector<std::string>  label;  // user-supplied or empty
  std::vector<std::string>  plabel; // .pN. synthetic; empty for constraint rows

  // Methods-developer extensibility — for columns latva itself doesn't ship.
  std::unordered_map<std::string, std::vector<double>>      extra_real;
  std::unordered_map<std::string, std::vector<std::int32_t>> extra_int;
  std::unordered_map<std::string, std::vector<std::string>>  extra_str;

  std::size_t size() const noexcept { return id.size(); }

  // Number of free parameters (the largest `free` index, or 0 if all fixed).
  std::int32_t n_free() const noexcept {
    std::int32_t n = 0;
    for (auto f : free) if (f > n) n = f;
    return n;
  }

  // Number of groups (the largest `group` index over formula rows; ≥ 1
  // always — constraint rows carry 0 and are ignored). Single-group ⇒ 1.
  std::int32_t n_groups() const noexcept {
    std::int32_t n = 1;
    for (auto g : group) if (g > n) n = g;
    return n;
  }

  bool is_constraint_row(std::size_t i) const noexcept {
    const parse::Op o = op[i];
    return o == parse::Op::EqConstraint || o == parse::Op::LtConstraint ||
           o == parse::Op::GtConstraint || o == parse::Op::DefineParam;
  }
};

// Lightweight const view into a single row.
struct RowView {
  const LatentStructure* pt;
  std::size_t     i;
  std::int32_t     id()     const noexcept { return pt->id[i]; }
  std::int8_t      user()   const noexcept { return pt->user[i]; }
  std::string_view lhs()    const noexcept { return pt->lhs[i]; }
  parse::Op        op()     const noexcept { return pt->op[i]; }
  std::string_view rhs()    const noexcept { return pt->rhs[i]; }
  std::int32_t     block()  const noexcept { return pt->block[i]; }
  std::int32_t     group()  const noexcept { return pt->group[i]; }
  std::int32_t     free()        const noexcept { return pt->free[i]; }
  std::int8_t      exo()         const noexcept { return pt->exo[i]; }
  double           fixed_value() const noexcept { return pt->fixed_value[i]; }
  std::string_view label()  const noexcept { return pt->label[i]; }
  std::string_view plabel() const noexcept { return pt->plabel[i]; }
};

}  // namespace latva::partable
