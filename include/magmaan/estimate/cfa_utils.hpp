#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

// Shared building blocks for the non-iterative CFA start-value producers
// (Guttman/MGM, Bentler-1982). Pure numerical helpers — no fitting.

namespace magmaan::estimate {

// Per-variable residual variance via the Spearman communality heuristic —
// port of lavaan's `lav_cfa_theta_spearman` with "wide" bounds. `S` should be
// a single factor's indicator (co)variance submatrix. For < 3 indicators the
// communality is undefined; falls back to 0.5 * variance.
Eigen::VectorXd theta_spearman(const Eigen::MatrixXd& S);

// Smallest generalized eigenvalue of the pencil (A, B): the smallest λ with
// det(A − λ·B) = 0. Returns nullopt when B is not numerically positive
// definite (the caller then skips the PD correction, as lavaan does).
std::optional<double> smallest_gen_root(const Eigen::MatrixXd& A,
                                        const Eigen::MatrixXd& B);

// Nearest positive-definite matrix: symmetrize, then clamp eigenvalues to tol.
Eigen::MatrixXd force_pd(const Eigen::MatrixXd& M, double tol);

// One CFA block's factor-loading layout, extracted from a MatrixRep. Only
// factors that carry at least one observed Lambda indicator are listed.
struct CfaBlockLayout {
  // One per Lambda cell of the block.
  struct Load {
    std::int16_t ov_row = -1;    // observed-variable row in S
    std::int16_t factor = -1;    // 0-based index into `factor_col`
    std::int32_t free_idx = -1;  // 0-based free-parameter index, -1 if fixed
  };
  std::int16_t n_observed = 0;
  std::vector<std::int16_t> factor_col;  // size n_factor: original Λ column
  std::vector<Load> loads;
  std::vector<std::int16_t> marker_ov;   // size n_factor: marker row, -1 if none
  bool crossloadings = false;            // an indicator loads on > 1 factor
  bool all_have_marker = true;

  std::int16_t n_factor() const {
    return static_cast<std::int16_t>(factor_col.size());
  }
};

// Per-block CFA loading layouts (one entry per MatrixRep block).
std::vector<CfaBlockLayout>
cfa_block_layouts(const spec::LatentStructure& pt, const model::MatrixRep& rep);

}  // namespace magmaan::estimate
