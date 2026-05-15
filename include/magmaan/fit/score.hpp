#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/fiml.hpp"
#include "magmaan/fit/fit.hpp"
#include "magmaan/fit/gls.hpp"
#include "magmaan/fit/ml.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/fit/uls.hpp"
#include "magmaan/fit/wls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/partable/partable.hpp"

namespace magmaan::fit {

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
  double epc = 0.0;
};

struct ScoreTestTable {
  std::vector<ScoreTestResult> rows;
};

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     ML discrepancy = {},
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     ULS discrepancy,
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     GLS discrepancy,
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     WLS discrepancy,
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices_fiml(partable::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          FIML discrepancy = {},
                          double h_step = 1e-4);

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            ML discrepancy = {},
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            ULS discrepancy,
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            GLS discrepancy,
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            WLS discrepancy,
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests_fiml(partable::LatentStructure pt,
                 const model::MatrixRep& rep,
                 const RawData& raw,
                 const Estimates& est,
                 FIML discrepancy = {},
                 double h_step = 1e-4);

}  // namespace magmaan::fit
