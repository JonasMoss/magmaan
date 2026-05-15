#pragma once

#include "magmaan/nt/infer.hpp"

namespace magmaan::fit {
using nt::infer::WaldTestResult;
using nt::infer::ZTestResult;
using nt::infer::browne_residual_adf;
using nt::infer::browne_residual_nt;
using nt::infer::chi2_pvalue;
using nt::infer::chi2_stat;
using nt::infer::df_stat;
using nt::infer::information_expected;
using nt::infer::information_observed_analytic;
using nt::infer::information_observed_fd;
using nt::infer::noncentral_chisq_cdf;
using nt::infer::rls_chi2;
using nt::infer::se;
using nt::infer::vcov;
using nt::infer::wald_test;
using nt::infer::z_test;
}
