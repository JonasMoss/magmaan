#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::inference {

using estimate::build_eq_constraints;
using estimate::EqConstraints;
using estimate::resolve_fixed_x_from_sample;
using estimate::fiml::FIML;
using estimate::fiml::fiml_start_sample_stats;
using estimate::fiml::validate_fiml_fixed_x_missing_policy;

enum class ScoreCandidateKind : std::uint8_t {
  FixedParam,
  EqualityRelease,
};

enum class ScoreInformation : std::uint8_t {
  Expected,
  Observed,
};

struct ScoreCandidate {
  ScoreCandidateKind kind = ScoreCandidateKind::FixedParam;
  std::size_t        row = 0;          // partable row for fixed params; A_eq row for equality releases
  parse::Op          op = parse::Op::Measurement;
  std::int32_t       lhs_var = -1;
  std::int32_t       rhs_var = -1;
  std::int32_t       group = 0;
};

struct ScoreTestResult {
  ScoreCandidate candidate;
  double score = 0.0;
  double information = 0.0;
  double mi = 0.0;
  int    df = 1;
  double p_value = 1.0;
  double epc = 0.0;        // raw expected parameter change
  double epc_lv = 0.0;     // std.lv-standardized EPC (latent SDs)
  double epc_all = 0.0;    // std.all-standardized EPC (latent + indicator SDs)

  // Robust extension (filled only by `inference::frontier`; the normal-theory
  // path leaves them at these defaults, so `mi_scaled == mi` and
  // `scaling_factor == 1` there). `mi` stays the ordinary statistic; `mi_scaled`
  // is the generalized / SB-scaled robust statistic `mi / scaling_factor`, with
  // `p_value` referenced to `mi_scaled` on the robust path.
  double v_eff = 0.0;          // robust efficient-score variance (NT info units)
  double mi_scaled = 0.0;      // mi / scaling_factor
  double scaling_factor = 1.0; // c = gᵀB1g / gᵀA1g (→ 1 under normality)
};

struct ScoreTestTable {
  std::vector<ScoreTestResult> rows;
};

// Which candidate parameters a modification-index sweep scores.
enum class ScoreCandidateSet : std::uint8_t {
  FixedRowsOnly,   // only parameters already present in the model as fixed rows
  WithAbsentRows,  // also enumerate statements absent from the model entirely
};

// Controls a modification-index sweep. The default reproduces the legacy
// `modification_indices(..., ScoreInformation)` behaviour. The absent-row flags
// scope the enumeration (mirroring lavaan's `modindices()`: cross-loadings and
// covariances). Structural regressions are not enumerated — adding a `~` row
// changes the model form, which needs Reduced-LISREL variable-table support.
struct ModificationIndexOptions {
  ScoreCandidateSet candidates = ScoreCandidateSet::FixedRowsOnly;
  ScoreInformation  information = ScoreInformation::Expected;
  bool include_loadings = true;     // absent cross-loadings  f =~ x
  bool include_covariances = true;  // absent covariances     x ~~ y
};

// Normal-theory ML modification indices / score tests.
post_expected<ScoreTestTable>
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     ScoreInformation information = ScoreInformation::Expected);

// Normal-theory ML modification indices with explicit options: optionally
// enumerates absent statements (cross-loadings, covariances) as fixed-at-0
// candidates, and always reports standardized EPC (`epc_lv` / `epc_all`)
// alongside the raw `epc`.
post_expected<ScoreTestTable>
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     const ModificationIndexOptions& options);

// Least-squares (moment-quadratic) modification indices / score tests. The
// `weight` is the only estimator selector: empty ⇒ ULS; a normal-theory
// weight ⇒ GLS; a caller-supplied weight ⇒ WLS. LS score tests always use the
// expected (residual-Jacobian) information.
post_expected<ScoreTestTable>
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     const estimate::gmm::Weight& weight);

post_expected<ScoreTestTable>
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     const estimate::gmm::Weight& weight,
                     const ModificationIndexOptions& options);

post_expected<ScoreTestTable>
modification_indices_fiml(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          FIML discrepancy = {},
                          double h_step = 1e-4);

post_expected<ScoreTestTable>
modification_indices_fiml(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          const ModificationIndexOptions& options,
                          FIML discrepancy = {},
                          double h_step = 1e-4);

post_expected<ScoreTestTable>
score_tests(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            const estimate::gmm::Weight& weight);

post_expected<ScoreTestTable>
score_tests_fiml(spec::LatentStructure pt,
                 const model::MatrixRep& rep,
                 const RawData& raw,
                 const Estimates& est,
                 FIML discrepancy = {},
                 double h_step = 1e-4);

// ── Robust (generalized / Satorra-Bentler-scaled) score tests ───────────────
//
// Frontier surface: goes beyond lavaan, which falls back to the ordinary
// statistic when `se != "standard"` (see lav_test_score.R). For every fixed-row
// / equality-release candidate these report `mi` (the ordinary NT statistic) and
// `mi_scaled = mi / c` with the per-direction scaling `c = gᵀB1g / gᵀA1g`, where
// A1/B1 are the parameter-space sandwich bread/meat (`robust::param_space_sandwich`)
// and g is the efficient-score direction. v1 covers continuous ML, both breads
// (`Information::Expected` ≈ robust.sem/MLM; `Information::Observed` ≈
// robust.huber.white/MLR). Single-group only.
namespace frontier {

struct RobustScoreOptions {
  // bread (Expected/Observed) + meat moments + Γ̂ source, shared with `robust_se`.
  robust::InferenceSpec spec{robust::Information::Expected,
                             robust::WeightMoments::Structured,
                             robust::ScoreCovariance::Empirical};
  ModificationIndexOptions base{};  // candidate set, loadings/covariances, info
};

// Robust ML modification indices. Γ̂ from raw data; or model-implied Γ_NT when
// `options.spec.cov == ScoreCovariance::ModelImplied` (raw is then ignored and
// may be omitted via the sample-stats overload).
post_expected<ScoreTestTable>
modification_indices_robust(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const SampleStats& samp,
                            const RawData& raw,
                            const Estimates& est,
                            const RobustScoreOptions& options = {});

// Model-implied (Γ_NT) meat only — `options.spec.cov` is treated as ModelImplied.
// For the Expected bread this reproduces the ordinary MI exactly (reduction-to-NT
// baseline); requires only sample statistics.
post_expected<ScoreTestTable>
modification_indices_robust(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const SampleStats& samp,
                            const Estimates& est,
                            const RobustScoreOptions& options = {});

// Caller-supplied Γ̂ (e.g. lavaan's NACOV); the R-assembled oracle path.
post_expected<ScoreTestTable>
modification_indices_robust(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const SampleStats& samp,
                            const Eigen::MatrixXd& gamma_hat,
                            const Estimates& est,
                            const RobustScoreOptions& options = {});

// Robust equality-release score tests (B1 built once in θ-space, df=1 per row).
post_expected<ScoreTestTable>
score_tests_robust(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const SampleStats& samp,
                   const RawData& raw,
                   const Estimates& est,
                   const RobustScoreOptions& options = {});

post_expected<ScoreTestTable>
score_tests_robust(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const SampleStats& samp,
                   const Eigen::MatrixXd& gamma_hat,
                   const Estimates& est,
                   const RobustScoreOptions& options = {});

// ── Continuous-LS (moment-quadratic) robust score tests ─────────────────────
// The same per-direction scaling in the moment metric: A1 = Σ_b (n_b/N)·Δ'WΔ,
// B1 = Σ_b (n_b/N)·Δ'WΓ̂WΔ with W the ESTIMATION weight, following the
// non-robust `modification_indices(.., weight)` convention (empty ⇒ ULS
// identity; `gmm::normal_theory_weight` ⇒ GLS; caller-supplied ⇒ WLS/DWLS).
// The bread is always the expected (Δ'WΔ) one — `options.spec.bread` must be
// Expected. Γ̂ sources mirror the ML overloads: empirical from complete-data
// raw, model-implied Γ_NT (per `options.spec.moments`: Structured ⇒ Γ_NT(Σ̂),
// Unstructured ⇒ Γ_NT(S)), or caller-supplied per-block Γ̂. With the GLS
// weight and the Γ_NT(S) meat the sandwich collapses (B1 = A1, c ≡ 1) — the
// exact reduction-to-NT baseline. Single-group only (v1).

post_expected<ScoreTestTable>
modification_indices_robust(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const SampleStats& samp,
                            const RawData& raw,
                            const Estimates& est,
                            const estimate::gmm::Weight& weight,
                            const RobustScoreOptions& options = {});

// Model-implied (Γ_NT) meat only — `options.spec.cov` is treated as ModelImplied.
post_expected<ScoreTestTable>
modification_indices_robust(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const SampleStats& samp,
                            const Estimates& est,
                            const estimate::gmm::Weight& weight,
                            const RobustScoreOptions& options = {});

// Caller-supplied per-block Γ̂ in the [μ ; vech σ] moment layout.
post_expected<ScoreTestTable>
modification_indices_robust(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const SampleStats& samp,
                            const std::vector<Eigen::MatrixXd>& gamma_blocks,
                            const Estimates& est,
                            const estimate::gmm::Weight& weight,
                            const RobustScoreOptions& options = {});

post_expected<ScoreTestTable>
score_tests_robust(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const SampleStats& samp,
                   const RawData& raw,
                   const Estimates& est,
                   const estimate::gmm::Weight& weight,
                   const RobustScoreOptions& options = {});

post_expected<ScoreTestTable>
score_tests_robust(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const SampleStats& samp,
                   const Estimates& est,
                   const estimate::gmm::Weight& weight,
                   const RobustScoreOptions& options = {});

post_expected<ScoreTestTable>
score_tests_robust(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const SampleStats& samp,
                   const std::vector<Eigen::MatrixXd>& gamma_blocks,
                   const Estimates& est,
                   const estimate::gmm::Weight& weight,
                   const RobustScoreOptions& options = {});

// Per-direction robust statistic worker, shared with the ordinal robust score
// path (`estimate::frontier`): computes the NT statistic for `direction` and
// rescales by c = gᵀB1g / gᵀA1g, with g the efficient-score direction implied
// by the info-metric nuisance projection. A1/B1 must be on the same weight
// scale as the score/information evaluation (c is not W-scale-invariant).
post_expected<ScoreTestResult>
score_for_direction_robust(const ScoreCandidate& candidate,
                           const Eigen::VectorXd& score_full,
                           const Eigen::MatrixXd& info_full,
                           const Eigen::MatrixXd& A1,
                           const Eigen::MatrixXd& B1,
                           const Eigen::MatrixXd& K_nuisance,
                           const Eigen::VectorXd& direction);

}  // namespace frontier

}  // namespace magmaan::inference
