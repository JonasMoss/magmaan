#include "magmaan/estimate/backend_strings.hpp"

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace magmaan::estimate {

namespace {

// The canonical kebab-case mapping. Entry order matches the Backend enum
// declaration order in `fit.hpp` for grep-ability — when a new backend lands
// in the enum, this table should grow alongside it.
constexpr std::array<std::pair<std::string_view, Backend>, 10> kBackendTable{{
    {"ceres",         Backend::Ceres},
    {"ceres-bfgs",    Backend::CeresBfgs},
    {"nlopt-slsqp",   Backend::NloptSlsqp},
    {"nlopt-bobyqa",  Backend::NloptBobyqa},
    {"nlopt-tnewton", Backend::NloptTnewton},
    {"nlopt-var2",    Backend::NloptVar2},
    {"nlopt-lbfgs",   Backend::NloptLbfgs},
    {"ipopt",         Backend::Ipopt},
    {"port",          Backend::Port},
    {"port-nls",      Backend::PortNls},
}};

}  // namespace

fit_expected<Backend>
backend_from_string(std::string_view name) {
  for (const auto& [s, b] : kBackendTable) {
    if (s == name) return b;
  }
  // Construct the "accepted names" list once, in the error path; no
  // performance concern.
  std::string accepted;
  for (std::size_t i = 0; i < kBackendTable.size(); ++i) {
    if (i > 0) accepted += ", ";
    accepted += '"';
    accepted += std::string(kBackendTable[i].first);
    accepted += '"';
  }
  return std::unexpected(FitError{
      FitError::Kind::NumericIssue,
      "unknown optimizer name \"" + std::string(name) +
          "\"; accepted: " + accepted,
      0, 0.0});
}

std::string_view
backend_name(Backend backend) {
  for (const auto& [s, b] : kBackendTable) {
    if (b == backend) return s;
  }
  return "unknown";  // unreachable for any value in the enum
}

}  // namespace magmaan::estimate
