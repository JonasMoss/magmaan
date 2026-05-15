#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/nt/fiml.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/gls/gls.hpp"
#include "magmaan/nt/ml.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/gls/wls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::nt::infer {

using estimate::build_eq_constraints;
using estimate::EqConstraints;
using estimate::resolve_fixed_x_from_sample;
using gls::GLS;
using gls::ULS;
using gls::WLS;
using nt::ml::ML;
using fiml::FIML;
using fiml::fiml_start_sample_stats;
using fiml::validate_fiml_fixed_x_missing_policy;

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
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     ML discrepancy = {},
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     ULS discrepancy,
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     GLS discrepancy,
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     WLS discrepancy,
                     ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
modification_indices_fiml(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          FIML discrepancy = {},
                          double h_step = 1e-4);

post_expected<ScoreTestTable>
score_tests(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            ML discrepancy = {},
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            ULS discrepancy,
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            GLS discrepancy,
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            WLS discrepancy,
            ScoreInformation information = ScoreInformation::Expected);

post_expected<ScoreTestTable>
score_tests_fiml(spec::LatentStructure pt,
                 const model::MatrixRep& rep,
                 const RawData& raw,
                 const Estimates& est,
                 FIML discrepancy = {},
                 double h_step = 1e-4);

}  // namespace magmaan::nt::infer
