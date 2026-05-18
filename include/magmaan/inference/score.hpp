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

}  // namespace magmaan::inference
