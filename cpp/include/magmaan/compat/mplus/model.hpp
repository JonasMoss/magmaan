#pragma once

#include "magmaan/parse/mplus_parser.hpp"
#include "magmaan/spec/build.hpp"

namespace magmaan::compat::mplus {

// MODEL lowering materializes Mplus defaults. Only fixed-x sample moments
// may be added by spec::build; starts remain explicit caller hints.
spec::BuildOptions build_options(const parse::MplusInput& input);

}  // namespace magmaan::compat::mplus
