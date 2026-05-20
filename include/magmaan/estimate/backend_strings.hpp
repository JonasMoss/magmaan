#pragma once

#include <string_view>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"

namespace magmaan::estimate {

// Bidirectional string ↔ Backend mapping. The kebab-case spellings match the
// `static constexpr std::string_view name` member each optimizer adapter
// already carries (e.g. `LbfgsOptimizer::name == "lbfgs"`,
// `PortNlsOptimizer::name == "port-nls"`), so the table is human-recognisable
// and stays in sync with the adapter naming convention.
//
// Used by the R bindings layer to thread a user-facing `optimizer = "..."`
// string argument through one Rcpp shim per fit family — replacing the older
// pattern of separate `fit_uls_impl` / `fit_uls_ceres_impl` / ...
// per-Backend shims.
//
// Recognised strings (kebab-case):
//   "lbfgs"          → Backend::Lbfgs        (L-BFGS / L-BFGS-B, the default)
//   "ceres"          → Backend::Ceres        (Ceres LM; needs MAGMAAN_WITH_CERES)
//   "ceres-bfgs"     → Backend::CeresBfgs    (Ceres dense-BFGS line search)
//   "nlopt-slsqp"    → Backend::NloptSlsqp   (Kraft 1988 SQP)
//   "nlopt-bobyqa"   → Backend::NloptBobyqa  (Powell 2009 DF TR)
//   "nlopt-tnewton"  → Backend::NloptTnewton (Nash 1985 truncated Newton)
//   "nlopt-var2"     → Backend::NloptVar2    (Shanno-Phua 1980 full BFGS)
//   "nlopt-lbfgs"    → Backend::NloptLbfgs   (NLopt's own L-BFGS)
//   "port"           → Backend::Port         (drmngb_, = R nlminb)
//   "port-nls"       → Backend::PortNls      (drn2gb_, = R nls)
//
// Unrecognised input returns FitError::Kind::NumericIssue with a detail
// string listing the accepted names.
fit_expected<Backend>
backend_from_string(std::string_view name);

// Inverse: Backend → canonical kebab-case name. Used by error messages and
// post-fit reporting so the optimizer label that comes back to R matches
// what the caller wrote in `optimizer = "..."`.
std::string_view
backend_name(Backend backend);

}  // namespace magmaan::estimate
