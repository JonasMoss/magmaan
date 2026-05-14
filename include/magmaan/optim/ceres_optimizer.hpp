#pragma once

#include "magmaan/fit/ceres_optimizer.hpp"

namespace magmaan::optim {

using fit::CeresBoundedOptimizer;
using fit::CeresOptimizer;
#ifdef MAGMAAN_WITH_CERES
using fit::CeresOptions;
#endif

}  // namespace magmaan::optim
