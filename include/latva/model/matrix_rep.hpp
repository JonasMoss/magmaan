#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "latva/expected.hpp"
#include "latva/partable/partable.hpp"

namespace latva::model {

// LISREL matrix kinds.
enum class MatId : std::uint8_t {
  Lambda,   // p × m   factor loadings (=~)
  Theta,    // p × p   residual covariances of indicators (~~ obs/obs)
  Psi,      // m × m   covariances among latent factors (~~ lv/lv)
  Beta,     // m × m   regressions among latents (~ lv on lv)
  Nu,       // p × 1   indicator intercepts (~1 on ov)
  Alpha,    // m × 1   latent means         (~1 on lv)
};

constexpr std::string_view to_string(MatId m) noexcept {
  switch (m) {
    case MatId::Lambda: return "Lambda";
    case MatId::Theta:  return "Theta";
    case MatId::Psi:    return "Psi";
    case MatId::Beta:   return "Beta";
    case MatId::Nu:     return "Nu";
    case MatId::Alpha:  return "Alpha";
  }
  return "?";
}

// One Cell per LatentStructure row. `used == false` means the partable row does
// not correspond to a matrix entry (constraint rows, define-param, and
// rows the current build does not yet know how to place).
struct Cell {
  MatId        mat   = MatId::Lambda;
  std::int16_t row   = -1;   // 0-based; -1 when used == false
  std::int16_t col   = -1;
  std::int8_t  block = 0;    // single-block in v0
  bool         used  = false;
};

struct BlockDims {
  std::int16_t n_observed = 0;   // p (Λ rows; Θ dim)
  std::int16_t n_latent   = 0;   // m_extended (Λ cols; Β/Ψ dim).
                                 // Includes phantom latents for ov.y / ov.x
                                 // when the model has any `~` operator.
};

// A "structural" cell — an entry that latva inserts mechanically (not
// from a partable row), with a fixed value. Used for the phantom-latent
// identity entries in Λ that map endogenous/exogenous observed to their
// own latent slot (Λ[ov_idx, lv_extended_idx] = 1).
struct StructuralCell {
  MatId        mat   = MatId::Lambda;
  std::int16_t row   = -1;
  std::int16_t col   = -1;
  std::int8_t  block = 0;
  double       value = 0.0;
};

// Which lavaan-style matrix representation the model uses.
//   PureCFA : no `~` rows. Σ = Λ Ψ Λᵀ + Θ. lv_names = user latents.
//   Reduced : has `~`. lv_names = [user_lv, ov.y, ov.x]; Λ has phantom
//             identity columns for ov.y and ov.x (added as
//             structural_cells); Σ = Λ (I−B)⁻¹ Ψ (I−B)⁻ᵀ Λᵀ + Θ.
enum class RepForm : std::uint8_t { PureCFA, Reduced };

// Static map from a LatentStructure to LISREL matrix layout. Built once per
// partable; consumed by ModelEvaluator. No floating-point math here —
// just integer indexing and variable orderings.
struct MatrixRep {
  RepForm                form = RepForm::PureCFA;
  std::vector<Cell>      cell_for_row;     // cell_for_row.size() == ptable.size()
  std::vector<StructuralCell> structural_cells;  // phantom-Λ identity, etc.
  std::vector<BlockDims> dims;             // dims.size() == n_blocks (1 in v0)
  std::vector<std::vector<std::string>> ov_names;   // per block
  std::vector<std::vector<std::string>> lv_names;   // per block (extended)
};

// production: build_matrix_rep
//
// Pure-CFA scope (P5.1):
//   supported : =~ (→ Lambda), ~~ between observed (→ Theta),
//               ~~ between latents (→ Psi), constraint rows (→ unused).
//   unused    : ~ (→ Beta) and ~1 (→ Nu/Alpha) — those rows get a
//               `used = false` Cell so the partable is still indexable
//               by row but the cell is a sentinel. The full LISREL
//               representation including phantom-latent regressions
//               lands in P5.2.
model_expected<MatrixRep>
build_matrix_rep(const partable::LatentStructure& pt);

}  // namespace latva::model
