#pragma once

#include "magmaan/optim/concepts.hpp"

namespace magmaan::fit {
template <class D> concept Discrepancy = optim::Discrepancy<D>;
template <class O> concept Optimizer = optim::Optimizer<O>;
template <class O> concept BoundedOptimizer = optim::BoundedOptimizer<O>;
template <class D> concept LsDiscrepancy = optim::LsDiscrepancy<D>;
template <class O> concept LsBoundedOptimizer = optim::LsBoundedOptimizer<O>;
using optim::LsResidualFn;
using optim::LsJacobianFn;
}
