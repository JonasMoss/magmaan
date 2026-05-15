#pragma once

#include "magmaan/estimate/snlls.hpp"

namespace magmaan::fit {
using estimate::SnllsDiagnostics;
using estimate::SnllsEstimates;
using estimate::fit_snlls_bounded;
}

namespace magmaan::fit::detail_snlls {
using estimate::detail_snlls::BlockKind;
using estimate::detail_snlls::Classification;
using estimate::detail_snlls::Counters;
using estimate::detail_snlls::Profile;
using estimate::detail_snlls::ProfileCache;
using estimate::detail_snlls::cached_profile_at;
using estimate::detail_snlls::classify;
using estimate::detail_snlls::expand_beta;
using estimate::detail_snlls::fit_err;
using estimate::detail_snlls::profile_at;
using estimate::detail_snlls::profile_gradient_at;
using estimate::detail_snlls::profile_jacobian_at;
}

namespace magmaan::fit {
}
