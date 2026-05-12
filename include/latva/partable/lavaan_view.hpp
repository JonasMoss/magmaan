#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "latva/parse/op.hpp"
#include "latva/partable/partable.hpp"
#include "latva/partable/start_hints.hpp"

namespace latva::partable {

// A lavaan-shaped projection of the model: the columns lavaan's `parTable()`
// carries, in struct-of-arrays form. This is *not* the in-memory model
// representation — that's the (`LatentStructure`, `LatentNames`, `Starts`)
// triple. It exists so the R layer can keep a familiar data.frame surface and
// the golden fixtures can keep diffing against `parTable(lavaanify(...))`.
//
// `ustart` reconstructs lavaan's column: a fixed row's `LatentStructure.fixed_value`
// (NaN for a not-yet-resolved fixed.x moment), or a free row's start *hint*
// (NaN where the user gave none). `block`/`group` are both the per-row group
// index (1-based; 0 for constraint rows) — they differ only under multilevel,
// which latva doesn't model yet. `plabel` is `.p<id>.` for formula rows, "" for
// constraint rows. Group identity (`group_var`, `group_labels`) is table-level
// metadata, mirroring `LatentNames`. `extra_*` carry the methods-developer
// escape-hatch columns straight through from `LatentStructure`.
struct LavaanParTable {
  std::vector<std::int32_t> id;       // 1-based, = row position
  std::vector<std::int8_t>  user;     // 0=auto, 1=user, 2=auto-equality
  std::vector<std::string>  lhs;
  std::vector<parse::Op>    op;
  std::vector<std::string>  rhs;       // "" for `~1` rows
  std::vector<std::int32_t> block;
  std::vector<std::int32_t> group;
  std::vector<std::int32_t> free;       // 0=fixed/constraint; else 1-based θ index
  std::vector<std::int8_t>  exo;        // 1 if data-given exogenous moment (fixed.x)
  std::vector<double>       ustart;     // fixed value or start hint; NaN ⇒ neither
  std::vector<std::string>  label;      // user-supplied or ""
  std::vector<std::string>  plabel;     // .pN. synthetic; "" for constraint rows

  std::string               group_var;
  std::vector<std::string>  group_labels;

  std::unordered_map<std::string, std::vector<double>>      extra_real;
  std::unordered_map<std::string, std::vector<std::int32_t>> extra_int;
  std::unordered_map<std::string, std::vector<std::string>>  extra_str;

  std::size_t size() const noexcept { return id.size(); }

  std::int32_t n_free() const noexcept {
    std::int32_t n = 0;
    for (auto f : free) if (f > n) n = f;
    return n;
  }
  std::int32_t n_groups() const noexcept {
    std::int32_t n = 1;
    for (auto g : group) if (g > n) n = g;
    return n;
  }
};

// Build the lavaan-shaped projection from the model triple. `starts` carries
// the free-parameter start hints (pass `{}` if you don't have/want them — the
// `ustart` column then has NaN for every free row).
LavaanParTable to_lavaan_partable(const LatentStructure& structure,
                                  const LatentNames& names,
                                  const Starts& starts = {});

// The model triple, as produced by `from_lavaan_partable`.
struct ParsedLavaanParTable {
  LatentStructure structure;
  LatentNames     names;
  Starts          starts;
};

// Invert `to_lavaan_partable`: re-derive the variable inventory (ids, roles,
// canonical orderings), the equality-constraint reparameterization, and the
// `LatentNames` companion from a lavaan-shaped partable. Used by the R
// `parse_partable_df` path (a possibly hand-edited data.frame) and by anything
// that round-trips. Total (never fails) — a malformed table just yields a
// malformed model the way a hand-edited lavaan parTable would.
ParsedLavaanParTable from_lavaan_partable(const LavaanParTable& pt);

}  // namespace latva::partable
