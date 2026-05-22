#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::spec {

// Knobs that lavaan exposes via cfa() / sem() / lavaan() entry points.
// We surface them on a single BuildOptions because we only have one
// entry point. Defaults match lavaan's cfa()/sem() defaults (the choices
// practitioners expect): variances and exo-LV covariances auto-added,
// marker indicator auto-fixed to 1, fixed.x mirrors lavaan exactly.
struct BuildOptions {
  bool auto_var       = true;   // residual variances for endogenous OV; LV variances
  bool auto_cov_lv_x  = true;   // covariances among exogenous latents
  bool auto_cov_y     = false;  // covariances among endogenous LV/OV (for SEM with multiple y)
  // `orthogonal` identification (≙ lavaan `cfa(…, orthogonal = TRUE)`): every
  // *auto-added* covariance among latent variables is fixed at 0 instead of
  // freely estimated. The covariance rows are still emitted (free = 0, value
  // 0) so the partable mirrors lavaan's. An explicit user `f1 ~~ f2` row still
  // wins. Only affects the rows `auto_cov_lv_x` / `auto_cov_y` generate.
  bool orthogonal     = false;  // fix auto latent-variable covariances at 0
  bool auto_fix_first = true;   // fix first loading per LV to 1.0 (marker indicator)
  // `std.lv` identification: scale each latent by fixing its `~~`-self
  // variance at 1.0 instead of fixing a marker loading. When true,
  // auto.fix.first is forced off regardless of `auto_fix_first` (lavaan does
  // the same: `if (std.lv) auto.fix.first <- FALSE`) and the first loading
  // stays free. Marker (`auto_fix_first`) remains the default — the oracle
  // contract is that bare `cfa(model, data)` is the marker parameterization.
  bool std_lv         = false;  // fix each LV variance at 1.0 instead of a marker loading
  // `effect.coding` identification (Little/Slegers/Card 2006; ≙ lavaan
  // `cfa(…, effect.coding = "loadings")`): leave *everything* free — every
  // loading and the LV variance — and add one linear-equality row per latent,
  // `Σλ == #indicators`, so the loadings average to 1 and the latent's metric
  // is the average of its indicators'. Forces auto.fix.first and std.lv off;
  // mutually exclusive with `std_lv` (lavaanify errors on both). Mean-structure
  // effect coding (`Σν == 0`) is a future sub-step, not yet implemented.
  bool effect_coding  = false;  // free all loadings + LV var; add `Σλ == #indicators`
  bool fixed_x        = true;   // mirror lavaan: data-given exogenous OV moments
  bool meanstructure  = false;  // auto-add ν / α rows, with defaults below
  bool int_ov_free    = true;   // observed intercepts free by default; growth() sets false
  bool int_lv_free    = false;  // latent means fixed at 0 by default; growth() sets true
  // Composite handling. The default remains the historical
  // Henseler-Ogasawara desugaring so existing callers keep today's behavior;
  // the native FC-SEM W/T path is opt-in while its evaluator lands.
  CompositeMode composite_mode = CompositeMode::HenselerOgasawara;
  // Group identity. `n_groups` (≙ lavaan `ngroups`) drives row replication;
  // `group_var` (≙ `group`) names the grouping variable — defaults to "group"
  // when n_groups > 1 and left empty; `group_labels` (≙ `group.label`) names
  // the levels — auto-generated as "1".."n_groups" when empty/wrong-arity.
  // All three are stored verbatim into the resulting LatentStructure.
  std::int32_t             n_groups = 1;   // 1 = single-group; >1 replicates rows per group
  std::string              group_var;      // "" ⇒ "group" (if n_groups>1) or unnamed (if ==1)
  std::vector<std::string> group_labels;   // size must be 0 or n_groups
};

// production: lavaanify (the P3 pipeline; see project plan for steps)
//
// Turns a parsed FlatPartable into a complete LatentStructure: classifies variables,
// applies user modifiers, auto-adds default rows, applies the chosen latent
// scaling (marker indicator via auto.fix.first, or `std.lv` LV-variance fix),
// fixed.x mirrors, synthesizes auto-equality constraints, assigns id /
// plabel / free indices, appends user constraints as LatentStructure rows, and
// stamps `group_var` / `group_labels` onto the result.
//
// `c(v1,…,vk)*x` per-group modifiers are applied (the g-th atom for group g);
// `PartableError::Kind::BadGroupSpec` if `n_groups < 1`, a `c(...)` arity ≠
// `n_groups`, or `group_labels` is non-empty with size ≠ `n_groups`.
//
// `out_starts` (optional): if non-null, receives the user's start *hints* —
// the `start(v)*x` / `v?x` modifiers on free parameters — sized `n_free`
// (NaN where no hint was given). Pass it through to `simple_start_values` /
// `fit`. Left null when the caller only needs the structural LatentStructure.
//
// `out_names` (optional): if non-null, receives the `LatentNames` companion —
// the verbal model (variable names by id, per-row labels/plabels, group var +
// level labels). Left null when the caller only needs the structural LatentStructure.
partable_expected<LatentStructure>
build(const parse::FlatPartable& flat, const BuildOptions& opts = {},
          Starts* out_starts = nullptr, LatentNames* out_names = nullptr);

// Recompute `s.eq_groups` (the parameter-merge partition) from the model's
// pure-merge `==` rows — auto-equality via `.pN.` plabel pairs and explicit
// bare `a == b` via row labels, both resolved through `names`. `lavaanify`
// calls this; callers that build a `LatentStructure` from outside (e.g. via
// `from_lavaan_partable` on a hand-edited partable) call it too so equality
// constraints are honored. It must always be followed by
// `resolve_lin_constraints`, which owns the constraint-classification flags
// (`has_inequality_constraints` / `nonlinear_eq_rows` /
// `has_unenforced_constraints`). Safe to call repeatedly.
void compute_eq_groups(LatentStructure& s, const LatentNames& names);

}  // namespace magmaan::spec
