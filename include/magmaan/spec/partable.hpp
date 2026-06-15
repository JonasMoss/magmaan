#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/parse/op.hpp"

namespace magmaan::spec {

// === Nonlinear equality constraints =========================================
//
// A nonlinear `==` row (`a == b*c`, `b1 == (b2+b3)^2`) cannot be reduced to the
// linear `R·θ = d` form. It is instead compiled — at lavaanify time, by
// `resolve_lin_constraints`, where identifier resolution is available — into a
// *name-free* expression tree: every parameter reference becomes a free θ index
// or a fixed constant. Fit-time evaluation then needs only `LatentStructure`,
// keeping numeric fitters name-free.

// One node of a flat expression-tree pool. `lhs` / `rhs` index back into the
// pool; a Unary node uses `lhs` only; Const / Param are leaves.
struct NlExprNode {
  enum class Kind : std::uint8_t { Const, Param, Unary, Binary };
  Kind         kind     = Kind::Const;
  double       constant = 0.0;            // Const literal, or a fixed Param's value
  std::int32_t free_idx = -1;             // Param: 0-based θ index; -1 ⇒ fixed Param
  parse::UnOp  un_op    = parse::UnOp::Pos;
  parse::BinOp bin_op   = parse::BinOp::Add;
  std::int32_t lhs      = -1;             // child pool indices
  std::int32_t rhs      = -1;
};

// One nonlinear equality constraint as `h(θ) == 0`, with `h = lhs − rhs` of the
// original `==`. `nodes` is the flat pool (children precede parents); `root` is
// h's top node.
struct NlConstraint {
  std::vector<NlExprNode> nodes;
  std::int32_t            root = -1;
};

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

// Cross-group equality families, mirroring lavaan's `group.equal` string
// vector. `build` ties the data-free families across groups (shared labels →
// `eq_groups` merge). The categorical prep step (`prepare_ordinal_delta_partable`)
// reads the resolved set off `LatentStructure::group_equal` and, when
// `Thresholds` is present, applies the Wu-Estabrook (2016) release that lavaan
// triggers for ordered-categorical delta models: free the group-2+ ordinal
// response-scale variances (non-binary) and indicator intercepts.
enum class GroupEqual : std::uint8_t {
  Loadings,             // =~ rows
  Thresholds,           // | rows (also triggers the categorical scale/intercept release)
  Intercepts,           // ~1 on observed indicators
  Means,                // ~1 on latent variables
  Residuals,            // ~~ self-variances of observed indicators
  ResidualCovariances,  // ~~ off-diagonals among observed indicators
  LvVariances,          // ~~ self-variances of latent variables
  LvCovariances,        // ~~ off-diagonals among latent variables
  Regressions,          // ~ structural paths
};

// Composite implementation selected at lavaanify time. `None` means the caller
// has not opened a `<~` door; any composite syntax must then fail at
// `spec::build`. H-O is the historical desugaring path; FC-SEM is the native
// W/T path introduced in lavaan 0.6-20.
enum class CompositeMode : std::uint8_t {
  None,
  HenselerOgasawara,
  FcSem,
};

// One user composite (`C <~ x1 + x2 + ...`). In Henseler-Ogasawara mode,
// `spec::build` rewrites every composite into a reflective sub-model: the
// emergent latent keeps the user's composite name and the `excrescent` nuisance
// latents (K-1 of them, for K indicators) span the rest of the indicator space.
// In FC-SEM mode, no expansion happens and `excrescent` is empty.
//
// The verbal copy lives on `LatentNames` for output/round-tripping. Native
// FC-SEM also mirrors the same composite in name-free `LatentStructure`
// `composite_blocks`, because W/T semantics are part of the numeric contract.
struct CompositeInfo {
  std::string              composite;    // emergent latent = the user's name
  std::vector<std::string> indicators;   // composite indicators x1..xK, in order
  std::vector<std::string> excrescent;   // H-O synthesized nuisance latents
};

// Name-free FC-SEM composite block: `composite_var` is the construct whose
// weights live in W; `indicator_vars` are the observed variables forming its T
// block. Empty for ordinary models and for the Henseler-Ogasawara expansion.
struct CompositeBlock {
  std::int32_t              composite_var = -1;
  std::vector<std::int32_t> indicator_vars;
};

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
  std::vector<CompositeInfo> composites;   // one per `<~` composite; empty if none
  CompositeMode             composite_mode = CompositeMode::None;
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

  // Native FC-SEM composites. Empty unless `composite_mode == FcSem` and the
  // model contains `<~` rows. Henseler-Ogasawara composites are represented by
  // ordinary expanded `=~`/`~~` rows instead.
  CompositeMode             composite_mode = CompositeMode::None;
  std::vector<CompositeBlock> composite_blocks;

  // Per-row group index — 1-based (matches lavaan); 0 for constraint rows.
  // (No multilevel yet: lavaan's `block` equals `group` here. When multilevel
  // lands, re-add a `block` column alongside.)
  std::vector<std::int32_t> group;

  // Resolved `group.equal` families (≙ lavaan `group.equal`), stamped by
  // `build` from `BuildOptions::group_equal`. `build` ties the data-free
  // families across groups directly; the categorical prep step
  // (`prepare_ordinal_delta_partable`) reads this to apply the Wu-Estabrook
  // scale/intercept release when `Thresholds` is present. Empty in the common
  // single-group / fully-free-across-groups case.
  std::vector<GroupEqual>   group_equal;

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

  // Constraint classification, all set by `resolve_lin_constraints` (which
  // re-parses every non-pure-merge `==` row and scans the `<` / `>` rows):
  //
  //   has_inequality_constraints — the model carries a `<` / `>` row. `fit()`
  //     always errors: inequality-constrained estimation would need boundary
  //     (chi-bar-squared) asymptotics machinery magmaan does not have.
  //   nonlinear_eq_rows — partable row indices of well-formed *nonlinear* `==`
  //     constraints (`a == b*c`, `b1 == (b2+b3)^2`): every identifier is known,
  //     the expression is not affine. Enforced via the nonlinear-equality path.
  //   has_unenforced_constraints — a `==` row is *malformed*: it failed to
  //     re-parse, or references an unknown identifier. `fit()` always errors.
  //
  // A plain *linear* `==` either folds into `eq_groups` (a bare merge) or
  // becomes a `lin_constraint_R` row — neither sets any of these flags.
  bool                      has_inequality_constraints = false;
  std::vector<std::int32_t> nonlinear_eq_rows;
  bool                      has_unenforced_constraints = false;

  // Compiled nonlinear equality constraints, parallel to `nonlinear_eq_rows`
  // (same size and order): `nl_constraints[k]` is the name-free expression
  // tree for the `==` row at `nonlinear_eq_rows[k]`. Built by
  // `resolve_lin_constraints`; consumed by the IPOPT constrained fit path and
  // the constrained vcov / df projection.
  std::vector<NlConstraint> nl_constraints;

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
