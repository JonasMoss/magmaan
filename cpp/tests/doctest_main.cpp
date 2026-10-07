#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>

#include <Eigen/Core>

#include "../src/estimate/detail_identification_probe.hpp"

// Identification sweep (board TASK-33.3). With MAGMAAN_IDENTIFICATION_SWEEP
// set, every structural identification check run by any test prints one line
// to stderr; `ctest -V` prefixes it with the test number, so
// cpp/tests/tools/identification_sweep.py can attribute each report to its
// test. Without the variable nothing is registered.
namespace {

void print_values(const char* name, const Eigen::VectorXd& values) {
  std::fprintf(stderr, " %s=", name);
  for (Eigen::Index k = 0; k < values.size(); ++k)
    std::fprintf(stderr, "%s%.3g", k == 0 ? "" : ",", values(k));
}

void print_identification(const magmaan::estimate::IdentificationReport& r,
                          double seconds) {
  std::fprintf(stderr,
               "[identification] status=%s reason=%s map=%s q=%d moments=%d "
               "rank=%d points=%d nullity=%d seconds=%.6g",
               magmaan::estimate::to_string(r.status).data(),
               magmaan::estimate::to_string(r.reason).data(),
               magmaan::estimate::to_string(r.map).data(), r.n_parameters,
               r.n_moments, r.rank, r.n_points,
               static_cast<int>(r.null_directions.cols()), seconds);
  print_values("mins", r.min_relative_singular_values);
  print_values("smallest", r.smallest_singular_values);
  std::fprintf(stderr, "\n");
}

struct IdentificationSweep {
  IdentificationSweep() {
    if (std::getenv("MAGMAAN_IDENTIFICATION_SWEEP") != nullptr)
      magmaan::estimate::identification_test::set_observer(&print_identification);
  }
};
const IdentificationSweep identification_sweep;

}  // namespace
