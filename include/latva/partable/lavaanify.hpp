#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "latva/expected.hpp"
#include "latva/parse/flat_partable.hpp"
#include "latva/partable/partable.hpp"
#include "latva/partable/start_hints.hpp"

namespace latva::partable {

// Knobs that lavaan exposes via cfa() / sem() / lavaan() entry points.
// We surface them on a single LavaanifyOptions because we only have one
// entry point. Defaults match lavaan's cfa()/sem() defaults (the choices
// practitioners expect): variances and exo-LV covariances auto-added,
// marker indicator auto-fixed to 1, fixed.x mirrors lavaan exactly.
struct LavaanifyOptions {
  bool auto_var       = true;   // residual variances for endogenous OV; LV variances
  bool auto_cov_lv_x  = true;   // covariances among exogenous latents
  bool auto_cov_y     = false;  // covariances among endogenous LV/OV (for SEM with multiple y)
  bool auto_fix_first = true;   // fix first loading per LV to 1.0 (marker indicator)
  bool fixed_x        = true;   // mirror lavaan: data-given exogenous OV moments
  bool meanstructure  = false;  // auto-add ν (free) for OV and α (fixed at 0) for LV
  // Group identity. `n_groups` (≙ lavaan `ngroups`) drives row replication;
  // `group_var` (≙ `group`) names the grouping variable — defaults to "group"
  // when n_groups > 1 and left empty; `group_labels` (≙ `group.label`) names
  // the levels — auto-generated as "1".."n_groups" when empty/wrong-arity.
  // All three are stored verbatim into the resulting ParTable.
  std::int32_t             n_groups = 1;   // 1 = single-group; >1 replicates rows per group
  std::string              group_var;      // "" ⇒ "group" (if n_groups>1) or unnamed (if ==1)
  std::vector<std::string> group_labels;   // size must be 0 or n_groups
};

// production: lavaanify (the P3 pipeline; see project plan for steps)
//
// Turns a parsed FlatPartable into a complete ParTable: classifies variables,
// applies user modifiers, auto-adds default rows, fixes marker indicators,
// fixed.x mirrors, synthesizes auto-equality constraints, assigns id /
// plabel / free indices, appends user constraints as ParTable rows, and
// stamps `group_var` / `group_labels` onto the result.
//
// `c(v1,…,vk)*x` per-group modifiers are applied (the g-th atom for group g);
// `PartableError::Kind::BadGroupSpec` if `n_groups < 1`, a `c(...)` arity ≠
// `n_groups`, or `group_labels` is non-empty with size ≠ `n_groups`.
//
// `out_starts` (optional): if non-null, receives the user's start *hints* —
// the `start(v)*x` / `v?x` modifiers on free parameters — sized `n_free`
// (NaN where no hint was given). Pass it through to `simple_start_values` /
// `fit`. Left null when the caller only needs the structural ParTable.
partable_expected<ParTable>
lavaanify(const parse::FlatPartable& flat, const LavaanifyOptions& opts = {},
          Starts* out_starts = nullptr);

}  // namespace latva::partable
