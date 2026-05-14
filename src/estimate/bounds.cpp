#include "magmaan/fit/bounds.hpp"

#include <limits>
#include <string>
#include <utility>

#include "magmaan/error.hpp"
#include "magmaan/parse/op.hpp"

namespace magmaan::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

post_expected<Bounds>
bounds_from_partable(const partable::LatentStructure& pt) {
  const auto n_free = pt.n_free();
  Bounds b;
  b.lower = Eigen::VectorXd::Constant(n_free, -kInf);
  b.upper = Eigen::VectorXd::Constant(n_free,  kInf);
  if (n_free == 0) return b;

  // A variance row in lavaan-speak: `lhs ~~ rhs` with `lhs == rhs`. Could be a
  // latent variance (Ψ-diag, e.g. `f ~~ f`) or a residual variance (Θ-diag,
  // e.g. `x1 ~~ x1`). Either way: free → lower bound at zero (Heywood barrier).
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Covariance) continue;
    if (pt.lhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i]) continue;
    const auto fr = pt.free[i];
    if (fr <= 0) continue;                  // fixed param: no bound needed
    if (fr > n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "bounds_from_partable: free[" + std::to_string(i) + "] = " +
              std::to_string(fr) + " exceeds n_free = " +
              std::to_string(n_free)));
    }
    // `free` is 1-based; theta index is `fr - 1`. Setting the same coordinate
    // twice (e.g. shared-label invariance across groups) is idempotent.
    b.lower(static_cast<Eigen::Index>(fr - 1)) = 0.0;
  }
  return b;
}

}  // namespace magmaan::fit
