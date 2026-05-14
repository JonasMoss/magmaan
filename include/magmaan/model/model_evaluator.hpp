#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/partable/partable.hpp"

namespace magmaan::model {

// Implied moments — what the model says about the population.
//   sigma[b] : implied covariance matrix for block b. (p_b × p_b.)
//   mu[b]    : implied mean vector for block b. Empty in v0
//              (mean structure is v0.4 work).
struct ImpliedMoments {
  std::vector<Eigen::MatrixXd> sigma;
  std::vector<Eigen::VectorXd> mu;
};

struct Evaluation {
  ImpliedMoments  moments;
  Eigen::MatrixXd J_sigma;
  Eigen::MatrixXd J_mu;
};

// Per-block assembled LISREL matrices at a given θ, plus the derived
// intermediates (A = (I−B)⁻¹, LamA = Λ A, Mid = A Ψ Aᵀ). Consumers that
// need to differentiate Σ analytically (observed-info SE) build their
// closed-form expressions out of these.
struct BlockMatrices {
  Eigen::MatrixXd Lambda;   // p × m
  Eigen::MatrixXd Psi;      // m × m (symmetric)
  Eigen::MatrixXd Theta;    // p × p (symmetric)
  Eigen::MatrixXd Beta;     // m × m
  Eigen::MatrixXd A;        // m × m, (I − B)⁻¹
  Eigen::MatrixXd LamA;     // p × m, Λ A
  Eigen::MatrixXd Mid;      // m × m, A Ψ Aᵀ (symmetric)
  Eigen::VectorXd Nu;       // p × 1, indicator intercepts; empty when
                            // the model has no mean structure
  Eigen::VectorXd Alpha;    // m × 1, latent means; empty when no mean
                            // structure or all α params absent
};

struct AssembledMatrices {
  std::vector<BlockMatrices> blocks;
};

// Where a free parameter θ_k lands in the LISREL matrices. Used by
// analytic post-fit inference to specialize ∂²Σ/∂θ_a∂θ_b case-by-case
// on the matrix pair without having to look inside ModelEvaluator.
struct ParamLocation {
  MatId        mat   = MatId::Lambda;
  std::int16_t row   = -1;
  std::int16_t col   = -1;
  std::int8_t  block = 0;
};

// ModelEvaluator: turns a parameter vector θ into implied moments and
// the analytic Jacobian dvech(Σ)/dθ.
//
// General LISREL form (P6.2):
//   Σ(θ) = Λ (I − B)⁻¹ Ψ (I − B)⁻ᵀ Λᵀ + Θ
//
// Pure CFA is a special case where B = 0 (so (I−B)⁻¹ = I); we still run
// the full pipeline because the cost is negligible at v0 sizes. Reduced
// LISREL handles regressions and CFA+structural via phantom-Λ identity
// columns inserted by build_matrix_rep().
//
// Construction is heavyweight (sets up parameter→cell tables, allocates
// per-block Λ/Θ/Ψ buffers). sigma() / dsigma_dtheta() are designed for
// hot-path use: no heap allocation, just fills pre-allocated buffers and
// returns Eigen views.
class ModelEvaluator {
 public:
  // Borrows LatentStructure + MatrixRep references — caller must keep them alive
  // for the lifetime of the evaluator. Build is fallible because some
  // partables (mismatched MatrixRep, missing variables) can't be evaluated.
  static model_expected<ModelEvaluator>
  build(const partable::LatentStructure& pt, const MatrixRep& rep);

  // Number of free parameters this evaluator expects in θ.
  std::size_t n_free() const noexcept { return n_free_; }
  std::size_t n_blocks() const noexcept { return blocks_.size(); }

  // Σ(θ). The returned ImpliedMoments references internal buffers that
  // get overwritten on the next sigma() call — copy if you need to keep
  // it across a subsequent evaluation.
  model_expected<ImpliedMoments>
  sigma(Eigen::Ref<const Eigen::VectorXd> theta) const;

  // dvech(Σ) / dθ. Shape: (sum_b p_b * (p_b + 1) / 2) × n_free.
  // Block ordering matches sigma(); within a block, vech is column-major
  // lower-triangle (matches Eigen's default).
  model_expected<Eigen::MatrixXd>
  dsigma_dtheta(Eigen::Ref<const Eigen::VectorXd> theta) const;

  // dμ / dθ. Shape: (sum_b p_b) × n_free, stacked per block.
  // Returns a 0×0 matrix when no block has mean structure — consumers
  // (ML::gradient with mean term) detect "covariance-only" via that
  // sentinel and skip the mean-term contribution. Blocks without their
  // own mean structure contribute zero rows but still occupy p_b slots
  // so the row offsets are consistent with `ImpliedMoments::mu`.
  model_expected<Eigen::MatrixXd>
  dmu_dtheta(Eigen::Ref<const Eigen::VectorXd> theta) const;

  // Combined hot-path evaluation. Assembles θ and computes shared LISREL
  // intermediates once, then materializes implied moments and requested
  // Jacobians. Existing single-purpose calls above remain the stable API for
  // callers that only need one output.
  model_expected<Evaluation>
  evaluate(Eigen::Ref<const Eigen::VectorXd> theta,
           bool with_sigma_jacobian,
           bool with_mu_jacobian) const;

  // Per-block Λ, Ψ, Θ, B at θ along with the derived A, LamA, Mid.
  // Returned by value (copies of the internal buffers) so the caller can
  // hold it across subsequent ModelEvaluator calls.
  model_expected<AssembledMatrices>
  assembled(Eigen::Ref<const Eigen::VectorXd> theta) const;

  // Location of each free parameter in the matrices, indexed by 0-based
  // free-parameter ordinal (matches θ ordering). Length == n_free().
  std::vector<ParamLocation> param_locations() const;

 private:
  ModelEvaluator() = default;

  // Per-block working buffers + per-cell index tables.
  struct BlockState {
    std::int16_t p = 0;             // n_observed
    std::int16_t m = 0;             // n_latent (extended)
    bool         has_means = false; // true when partable has any ~1 row
                                    // touching this block — then Nu/Alpha
                                    // are sized and Mu is populated.
    mutable Eigen::MatrixXd Lambda; // p × m
    mutable Eigen::MatrixXd Theta;  // p × p (symmetric)
    mutable Eigen::MatrixXd Psi;    // m × m (symmetric)
    mutable Eigen::MatrixXd Beta;   // m × m  (regressions among lv_ext)
    mutable Eigen::MatrixXd ImB;    // I − B
    mutable Eigen::MatrixXd ImB_inv;// (I − B)⁻¹
    mutable Eigen::MatrixXd LamA;   // Λ (I − B)⁻¹  (used in Jacobian)
    mutable Eigen::MatrixXd Mid;    // (I − B)⁻¹ Ψ (I − B)⁻ᵀ
    mutable Eigen::MatrixXd Sigma;  // p × p — output buffer
    mutable Eigen::VectorXd Nu;     // p × 1, indicator intercepts (only
                                    // sized when has_means)
    mutable Eigen::VectorXd Alpha;  // m × 1, latent means (only sized
                                    // when has_means)
    mutable Eigen::VectorXd Mu;     // p × 1, μ = ν + Λ A α — output buffer
  };

  // For each LatentStructure row, the (mat, row, col, block) cell + its source:
  // either a free θ index (1-based) or a fixed value.
  struct CellWrite {
    std::uint8_t  mat;        // matches MatId enum value
    std::int16_t  row;
    std::int16_t  col;
    std::int8_t   block;
    bool          is_free;    // true → take value from theta[free_idx-1]
    std::int32_t  free_idx;   // 1-based, only valid if is_free
    double        fixed_value;
  };

  std::vector<BlockState> blocks_;
  std::vector<CellWrite>  writes_;     // one per non-constraint, used cell
  // Structural (auto-inserted) cells with fixed values, e.g. phantom-Λ
  // identity entries for ov.y / ov.x in Reduced form.
  std::vector<CellWrite>  structural_writes_;
  std::size_t             n_free_ = 0;

  // Fills Λ/Θ/Ψ from theta + writes_. Used by both sigma() and the
  // Jacobian (which also needs the matrices instantiated).
  model_expected<void> assemble(Eigen::Ref<const Eigen::VectorXd> theta) const;

  void compute_intermediates(bool with_sigma, bool with_mu) const;
  ImpliedMoments current_moments() const;
  Eigen::MatrixXd current_dsigma_dtheta() const;
  Eigen::MatrixXd current_dmu_dtheta() const;
};

}  // namespace magmaan::model
