#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "magmaan/parse/op.hpp"

namespace magmaan::spec {

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

// Names that go with a `LatentStructure` — the *verbal* model. Everything in here
// is for display / round-tripping / parsing; nothing on the model's numeric path
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

// Lavaanified model structure — what to estimate, in struct-of-arrays form,
// *modulo* estimator (ML/GLS/WLS/polychorics), identification convention
// (marker / std.lv / effect coding), and names. It carries integer variable
// ids + per-variable roles + the canonical orderings, so `matrix_rep` reads
// the variable classification rather than re-deriving it from name strings.
//
// Estimation outputs (est, se, vcov) live on `Estimates` and as free
// matrices/vectors from `information_*` / `vcov` / `se`; the
// user's start *hints* live on `Starts`; the verbal model (names, user labels,
// group var + level labels, `.pN.` plabels) lives on `LatentNames`. A
// lavaan-shaped projection (`LavaanParTable`, see `lavaan_view.hpp`) bundles
// them back up for display and round-tripping. The one start-flavored thing
// the structure *does* carry is `fixed_value`: the value at which a *fixed*
// parameter sits — a model fact (`f =~ 1*x1`, a `std.lv` variance fix, a
// `fixed.x` covariance), not an optimization hint.
//
// Constraint statements (==, <, >, :=) are real rows just like in lavaan, with
// `group = 0`. Auto-equality from a shared user label is encoded as a
// synthesized `==` row referencing two `.pN.` plabels (held on `LatentNames`);
// the source rows keep distinct `free` indices. The resolved equality groups
// are precomputed into `eq_groups`.
//
// Group identity is part of the model only as `n_groups()` ≥ 1 plus the
// per-row `group` index; the *named* part (which variable, what the level
// labels are) is a `LatentNames` concern. The single-group case is just
// `n_groups() == 1`.
struct LatentStructure {
  // Statement kind, per row.
  std::vector<parse::Op>    op;

  // Variable table — name-free mirror of the model, populated by `lavaanify`
  // (or `from_lavaan_partable`) and consumed by `matrix_rep`. `lhs_var` /
  // `rhs_var` are var ids (or -1 for a `~1` rhs and for constraint-row sides,
  // which are expressions, not variables).
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

  // Per-row group index — 1-based (matches lavaan); 0 for constraint rows.
  // (No multilevel yet: lavaan's `block` equals `group` here. When multilevel
  // lands, re-add a `block` column alongside.)
  std::vector<std::int32_t> group;

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
  // (shared label, explicit bare `a == b`, cross-group invariance). Empty ⇒
  // identity (every param its own group).
  std::vector<std::int32_t> eq_groups;

  // General-*linear* equality constraints that can't be expressed as a mere
  // parameter merge (`a == 2*b + c`, `b2 + b3 == 1.5`, `Σλ == k` for effect
  // coding): `R·θ = d` over the `n_free()` free params. `lin_constraint_R` is
  // row-major, `(lin_constraint_d.size()) × n_free()`. Filled by
  // `resolve_lin_constraints` (which re-parses the `==` rows `compute_eq_groups`
  // left flagged); empty in the common case. `build_eq_constraints` stacks these
  // on top of the `eq_groups` merge rows to form the affine reparam `θ = θ₀ + Kα`.
  std::vector<double>       lin_constraint_R;
  std::vector<double>       lin_constraint_d;

  // Set when the model carries an `<` / `>` row or a genuinely *nonlinear* `==`
  // expression (`a == b*c`) — `fit()` then errors (those phases aren't
  // implemented). Cleared by `resolve_lin_constraints` once every non-bare `==`
  // row has been reduced to a `lin_constraint_R` row (and no `<`/`>` remain).
  bool                      has_unenforced_constraints = false;

  // Methods-developer extensibility — for columns magmaan itself doesn't ship.
  std::unordered_map<std::string, std::vector<double>>      extra_real;
  std::unordered_map<std::string, std::vector<std::int32_t>> extra_int;
  std::unordered_map<std::string, std::vector<std::string>>  extra_str;

  std::size_t size() const noexcept { return op.size(); }

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

}  // namespace magmaan::spec
