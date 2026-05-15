#pragma once

#include "magmaan/fit/inference.hpp"
#include "magmaan/fit/score.hpp"

namespace magmaan::nt::infer {

using fit::Estimates;
using fit::ScoreCandidate;
using fit::ScoreCandidateKind;
using fit::ScoreInformation;
using fit::ScoreTestResult;
using fit::ScoreTestTable;
using fit::WaldTestResult;
using fit::ZTestResult;
using fit::browne_residual_adf;
using fit::browne_residual_nt;
using fit::chi2_pvalue;
using fit::chi2_stat;
using fit::df_stat;
using fit::information_expected;
using fit::information_observed_analytic;
using fit::information_observed_fd;
using fit::modification_indices;
using fit::modification_indices_fiml;
using fit::noncentral_chisq_cdf;
using fit::rls_chi2;
using fit::score_tests;
using fit::score_tests_fiml;
using fit::se;
using fit::vcov;
using fit::wald_test;
using fit::z_test;

}  // namespace magmaan::nt::infer
