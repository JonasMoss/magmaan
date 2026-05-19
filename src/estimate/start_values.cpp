#include "magmaan/estimate/start_values.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

bool is_constraint_op(parse::Op op) noexcept {
  return op == parse::Op::EqConstraint || op == parse::Op::LtConstraint ||
         op == parse::Op::GtConstraint || op == parse::Op::DefineParam;
}

}  // namespace

fit_expected<Eigen::VectorXd>
simple_start_values(const spec::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    const SampleStats& samp,
                    const spec::Starts& starts) {
  const std::int32_t n_free = pt.n_free();
  Eigen::VectorXd start = Eigen::VectorXd::Zero(n_free);
  if (n_free == 0) return start;
  const std::size_t n_hint = starts.hint.size();

  // Per-block table of latent (Ψ) columns whose own variance is *fixed* — the
  // `std.lv` case (and any explicit `f ~~ c*f`). A loading into such a latent
  // can't be scaled by the LV variance, so we start it from the indicator's
  // own variance (≈ lavaan's `sqrt(0.5 · var_indicator)`) rather than the
  // marker-world 0.7. Built once up front by a single sweep over the rows.
  std::vector<std::vector<bool>> lv_var_fixed(rep.dims.size());
  std::vector<std::vector<std::int16_t>> phantom_ov_for_lv(rep.dims.size());
  // Marker indicator per latent: the observed row whose loading on that latent
  // is fixed to 1 via a user `=~` marker. Markers are partable rows, so this
  // is read from `cell_for_row` (not `structural_cells`, which holds only
  // synthesized fixed cells). Used to sign and scale that latent's free
  // loadings.
  std::vector<std::vector<std::int16_t>> marker_for_lv(rep.dims.size());
  for (std::size_t b = 0; b < rep.dims.size(); ++b) {
    lv_var_fixed[b].assign(static_cast<std::size_t>(rep.dims[b].n_latent), false);
    phantom_ov_for_lv[b].assign(static_cast<std::size_t>(rep.dims[b].n_latent), -1);
    marker_for_lv[b].assign(static_cast<std::size_t>(rep.dims[b].n_latent), -1);
  }
  for (const auto& sc : rep.structural_cells) {
    if (sc.mat != model::MatId::Lambda || sc.value != 1.0) continue;
    const std::size_t b = static_cast<std::size_t>(sc.block);
    if (b < phantom_ov_for_lv.size() &&
        sc.col >= 0 &&
        static_cast<std::size_t>(sc.col) < phantom_ov_for_lv[b].size()) {
      phantom_ov_for_lv[b][static_cast<std::size_t>(sc.col)] = sc.row;
    }
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto& c = rep.cell_for_row[i];
    if (!c.used || pt.free[i] != 0) continue;  // estimated → neither fact holds
    const std::size_t b = static_cast<std::size_t>(c.block);
    // std.lv: a fixed latent variance.
    if (c.mat == model::MatId::Psi && c.row == c.col &&
        b < lv_var_fixed.size() &&
        static_cast<std::size_t>(c.col) < lv_var_fixed[b].size()) {
      lv_var_fixed[b][static_cast<std::size_t>(c.col)] = true;
    }
    // marker: a loading fixed to 1.
    if (c.mat == model::MatId::Lambda &&
        !std::isnan(pt.fixed_value[i]) &&
        std::abs(pt.fixed_value[i] - 1.0) < 1e-12 &&
        b < marker_for_lv.size() &&
        static_cast<std::size_t>(c.col) < marker_for_lv[b].size()) {
      marker_for_lv[b][static_cast<std::size_t>(c.col)] = c.row;
    }
  }

  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (is_constraint_op(pt.op[i])) continue;
    if (pt.free[i] == 0) continue;
    const std::int32_t k = pt.free[i] - 1;

    // 1) honor a user-supplied start hint.
    if (static_cast<std::size_t>(k) < n_hint && std::isfinite(starts.hint[static_cast<std::size_t>(k)])) {
      start(k) = starts.hint[static_cast<std::size_t>(k)];
      continue;
    }

    // 2) infer by cell.
    const auto& c = rep.cell_for_row[i];
    if (!c.used) {
      start(k) = 0.0;
      continue;
    }
    const std::size_t b = static_cast<std::size_t>(c.block);
    if (b >= samp.S.size()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "block " + std::to_string(b) +
              " has no SampleStats entry"));
    }
    const auto& S = samp.S[b];

    switch (c.mat) {
      case model::MatId::Lambda: {
        // Under std.lv (LV variance fixed), scale from the indicator's own
        // variance — λ ≈ √(½·Var(indicator)) — so a unit-variance latent
        // reproduces ~half of each indicator's spread. Otherwise (marker
        // world) use the default magnitude 0.7, signed by the indicator's
        // covariance with the factor's marker: reverse-keyed indicators then
        // start in the correct orthant, where a flat +0.7 stranded them in
        // the wrong one and could stall the optimizer (see
        // docs/validation/convergence_diagnostics.md). The magnitude stays 0.7, so
        // equality-constrained loadings of the same sign keep a consistent
        // start.
        const bool lv_fixed =
            b < lv_var_fixed.size() &&
            static_cast<std::size_t>(c.col) < lv_var_fixed[b].size() &&
            lv_var_fixed[b][static_cast<std::size_t>(c.col)];
        if (lv_fixed && c.row < S.rows() && S(c.row, c.row) > 0.0) {
          start(k) = std::sqrt(0.5 * S(c.row, c.row));
          break;
        }
        const std::int16_t marker =
            (b < marker_for_lv.size() &&
             static_cast<std::size_t>(c.col) < marker_for_lv[b].size())
                ? marker_for_lv[b][static_cast<std::size_t>(c.col)]
                : static_cast<std::int16_t>(-1);
        const bool reverse_keyed =
            marker >= 0 && marker < S.rows() && c.row < S.rows() &&
            S(c.row, marker) < 0.0;
        start(k) = reverse_keyed ? -0.7 : 0.7;
        break;
      }
      case model::MatId::Theta:
        if (c.row == c.col) {
          if (c.row >= S.rows()) {
            return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
                "theta row index " + std::to_string(c.row) +
                    " out of range for sample S (p=" +
                    std::to_string(S.rows()) + ")"));
          }
          start(k) = 0.5 * S(c.row, c.row);
        } else {
          start(k) = 0.0;
        }
        break;
      case model::MatId::Psi:
        if (c.row == c.col) {
          const bool is_phantom_ov =
              b < phantom_ov_for_lv.size() &&
              static_cast<std::size_t>(c.row) < phantom_ov_for_lv[b].size() &&
              phantom_ov_for_lv[b][static_cast<std::size_t>(c.row)] >= 0;
          if (is_phantom_ov) {
            const auto ov = phantom_ov_for_lv[b][static_cast<std::size_t>(c.row)];
            start(k) = ov < S.rows() ? S(ov, ov) : 0.05;
          } else {
            start(k) = 0.05;
          }
        } else {
          start(k) = 0.0;
        }
        break;
      case model::MatId::Beta:
        // Regression coefficients (latent → latent OR via phantom-Λ
        // observed → observed). Lavaan's "simple" defaults regressions to
        // 0; we mirror that.
        start(k) = 0.0;
        break;
      case model::MatId::Nu:
        // Indicator intercept ν_i: start at the sample mean of indicator
        // i if `samp.mean` is populated; otherwise 0. (For mean-structure
        // models the user is expected to supply sample means.)
        if (c.row < static_cast<std::int16_t>(samp.mean.size() > b
                ? samp.mean[b].size() : 0)) {
          start(k) = samp.mean[b](c.row);
        } else {
          start(k) = 0.0;
        }
        break;
      case model::MatId::Alpha:
        // Latent mean α_j — start at 0. Lavaan typically auto-fixes
        // latent means at 0 anyway; users specifying α as free usually
        // want it estimated.
        start(k) = 0.0;
        break;
    }
  }
  return start;
}

}  // namespace magmaan::estimate
