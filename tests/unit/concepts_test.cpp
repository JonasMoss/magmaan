#include <doctest/doctest.h>

#include "magmaan/fit/concepts.hpp"
#include "magmaan/fit/lbfgs_optimizer.hpp"
#include "magmaan/fit/ml.hpp"
#include "magmaan/fit/uls.hpp"

// Compile-time guards that the shipped Discrepancy / Optimizer implementations
// satisfy the concepts in `magmaan/fit/concepts.hpp`. If a value/gradient
// signature or a minimize signature ever drifts, this fails at the
// static_assert — well before any call site.

TEST_CASE("Discrepancy concept is satisfied by ML and ULS") {
  static_assert(magmaan::fit::Discrepancy<magmaan::fit::ML>,
                "ML must model the Discrepancy concept "
                "(see include/magmaan/fit/concepts.hpp)");
  static_assert(magmaan::fit::Discrepancy<magmaan::fit::ULS>,
                "ULS must model the Discrepancy concept "
                "(see include/magmaan/fit/concepts.hpp)");
  CHECK(true);  // body presence so the test shows up in the discovered list
}

TEST_CASE("Optimizer concept is satisfied by LbfgsOptimizer") {
  static_assert(magmaan::fit::Optimizer<magmaan::fit::LbfgsOptimizer>,
                "LbfgsOptimizer must model the Optimizer concept "
                "(see include/magmaan/fit/concepts.hpp)");
  CHECK(true);
}
