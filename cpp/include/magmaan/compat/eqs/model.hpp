#pragma once

#include <string>

#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/build.hpp"

namespace magmaan::compat::eqs {

// The EQS frontend materializes every independent variance and specified
// covariance. Do not let lavaan defaults change this fully specified model.
inline spec::BuildOptions build_options() {
  spec::BuildOptions options;
  options.auto_var = false;
  options.auto_cov_lv_x = false;
  options.auto_cov_y = false;
  options.auto_fix_first = false;
  options.auto_fix_single = false;
  options.fixed_x = false;
  return options;
}

// A rebuildable textual projection of already-lowered formula rows. Used at
// the R boundary to preserve ordinary model-spec rebuilding/refitting support.
std::string to_lavaan_syntax(const parse::FlatPartable& flat);

}  // namespace magmaan::compat::eqs
