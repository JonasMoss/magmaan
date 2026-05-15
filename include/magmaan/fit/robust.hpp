#pragma once

#include "magmaan/nt/robust.hpp"

namespace magmaan::fit {
using nt::robust::Information;
using nt::robust::InferenceSpec;
using nt::robust::MeanVarAdjustedResult;
using nt::robust::RobustSeResult;
using nt::robust::SatorraBentlerResult;
using nt::robust::ScaledShiftedResult;
using nt::robust::ScoreCovariance;
using nt::robust::UFactor;
using nt::robust::WeightMoments;
using nt::robust::build_u_factor;
using nt::robust::casewise_contributions;
using nt::robust::mean_var_adjusted;
using nt::robust::reduced_gamma_nt;
using nt::robust::reduced_gamma_sample;
using nt::robust::reduced_gamma_sample_streaming;
using nt::robust::reduced_gamma_unbiased;
using nt::robust::reduced_gamma_unbiased_casewise;
using nt::robust::robust_se;
using nt::robust::satorra_bentler;
using nt::robust::scaled_shifted;
using nt::robust::ugamma_eigenvalues;
}
