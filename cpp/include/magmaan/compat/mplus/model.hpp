#pragma once

#include "magmaan/parse/mplus_parser.hpp"
#include "magmaan/spec/build.hpp"

namespace magmaan::compat::mplus {

// MODEL lowering materializes Mplus defaults. Only fixed-x sample moments
// may be added by spec::build; starts remain explicit caller hints.
std::string to_lavaan_syntax(const parse::FlatPartable& flat);

void apply_provenance(const parse::MplusModel& parsed, const spec::LatentStructure& structure, spec::LatentNames& names);

spec::BuildOptions build_options(const parse::MplusInput& input);

}  // namespace magmaan::compat::mplus
