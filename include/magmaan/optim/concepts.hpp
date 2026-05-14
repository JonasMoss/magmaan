#pragma once

#include "magmaan/fit/concepts.hpp"

namespace magmaan::optim {

template <class D>
concept Discrepancy = fit::Discrepancy<D>;

template <class O>
concept Optimizer = fit::Optimizer<O>;

template <class O>
concept BoundedOptimizer = fit::BoundedOptimizer<O>;

template <class D>
concept LsDiscrepancy = fit::LsDiscrepancy<D>;

template <class O>
concept LsBoundedOptimizer = fit::LsBoundedOptimizer<O>;

using fit::LsJacobianFn;
using fit::LsResidualFn;

}  // namespace magmaan::optim

