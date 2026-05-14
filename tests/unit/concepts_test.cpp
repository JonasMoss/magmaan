#include <doctest/doctest.h>

#include "magmaan/optim/concepts.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/nt/ml.hpp"
#include "magmaan/gls/uls.hpp"

// Compile-time guards that the shipped Discrepancy / Optimizer implementations
// satisfy the concepts in `magmaan/optim/concepts.hpp`. If a value/gradient
// signature or a minimize signature ever drifts, this fails at the
// static_assert — well before any call site.

TEST_CASE("Discrepancy concept is satisfied by ML and ULS") {
  static_assert(magmaan::optim::Discrepancy<magmaan::nt::ml::ML>,
                "ML must model the Discrepancy concept "
                "(see include/magmaan/optim/concepts.hpp)");
  static_assert(magmaan::optim::Discrepancy<magmaan::gls::ULS>,
                "ULS must model the Discrepancy concept "
                "(see include/magmaan/optim/concepts.hpp)");
  CHECK(true);  // body presence so the test shows up in the discovered list
}

TEST_CASE("Optimizer concept is satisfied by LbfgsOptimizer") {
  static_assert(magmaan::optim::Optimizer<magmaan::optim::LbfgsOptimizer>,
                "LbfgsOptimizer must model the Optimizer concept "
                "(see include/magmaan/optim/concepts.hpp)");
  CHECK(true);
}
