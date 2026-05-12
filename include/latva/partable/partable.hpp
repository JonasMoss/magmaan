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

// Lavaanified ParTable: model description in struct-of-arrays form.
//
// Mirrors lavaan's parTable() columns, but only the model-description ones.
// Estimation outputs (est, se, vcov) and the user's start *hints* live on
// separate types (`Estimates`, `Starts`, ...) and are composed by the user,
// not stored here. The one start-flavored thing the ParTable does carry is
// `fixed_value`: the value at which a *fixed* parameter sits — that's a model
// fact (`f =~ 1*x1`, a `std.lv` variance fix, a `fixed.x` covariance), not an
// optimization hint. See docs/agents/rules.md (or the project plan) for the
// rationale on the model/estimation split.
//
// Constraint statements (==, <, >, :=) are real ParTable rows just like in
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
struct ParTable {
  // Identity
  std::vector<std::int32_t> id;     // 1-based, matches lavaan
  std::vector<std::int8_t>  user;   // 0=auto, 1=user-supplied, 2=auto-equality

  // Statement
  std::vector<std::string>  lhs;
  std::vector<parse::Op>    op;
  std::vector<std::string>  rhs;

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

  // Number of groups (the largest `block`/`group` index over formula rows;
  // ≥ 1 always — constraint rows carry 0 and are ignored). Single-group ⇒ 1.
  std::int32_t n_groups() const noexcept {
    std::int32_t n = 1;
    for (auto g : group) if (g > n) n = g;
    return n;
  }
};

// Lightweight const view into a single row.
struct RowView {
  const ParTable* pt;
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
