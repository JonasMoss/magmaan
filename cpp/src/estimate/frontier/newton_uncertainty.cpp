#include "magmaan/estimate/frontier/newton_accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include "detail_stable_norm.hpp"

namespace magmaan::estimate::frontier {
namespace {
using Wide = long double;
constexpr Wide wide_epsilon = std::numeric_limits<Wide>::epsilon();

Wide gamma(Eigen::Index operations) {
  const Wide t = static_cast<Wide>(operations) * wide_epsilon;
  return t < 1 ? t / (1 - t) : std::numeric_limits<Wide>::infinity();
}
// Use explicit wide accumulations, not BLAS or an unverified eigensolver, for
// verification. Gamma bounds include product and summation roundoff; the tiny
// additive allowance also covers gradual underflow. Overflow stays unresolved.
Wide norm_upper(const Eigen::MatrixXd& a) {
  Wide sum = 0;
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    const Wide x = static_cast<Wide>(a.data()[i]); sum += x * x;
  }
  const Wide tiny = std::numeric_limits<Wide>::min();
  return std::sqrt((sum + tiny * static_cast<Wide>(2 * a.size() + 1)) /
                   (1 - gamma(2 * a.size() + 2))) * (1 + 32 * wide_epsilon);
}
Wide product_error(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                   const Eigen::MatrixXd& c) {
  Wide squares = 0;
  const Wide tiny = std::numeric_limits<Wide>::min();
  for (Eigen::Index j = 0; j < c.cols(); ++j) {
    for (Eigen::Index i = 0; i < c.rows(); ++i) {
      Wide dot = 0, absolute = std::abs(static_cast<Wide>(c(i, j)));
      for (Eigen::Index k = 0; k < a.cols(); ++k) {
        const Wide term = static_cast<Wide>(a(i, k)) * static_cast<Wide>(b(k, j));
        dot += term; absolute += std::abs(term);
      }
      const Wide error = std::abs(dot - static_cast<Wide>(c(i, j))) +
          gamma(2 * a.cols() + 3) * absolute /
              (1 - gamma(a.cols() + 1)) + tiny * (2 * a.cols() + 3);
      squares += error * error;
    }
  }
  return std::sqrt((squares + tiny * (2 * c.size() + 1)) /
      (1 - gamma(2 * c.size() + 2))) * (1 + 32 * wide_epsilon);
}
Wide singular_lower(const Eigen::MatrixXd& triangular, bool lower) {
  const Eigen::Index q = triangular.rows();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(q, q);
  const Eigen::MatrixXd inverse = lower
      ? Eigen::MatrixXd(triangular.triangularView<Eigen::Lower>().solve(identity))
      : Eigen::MatrixXd(triangular.triangularView<Eigen::Upper>().solve(identity));
  if (!inverse.allFinite()) return 0;
  const Wide defect = product_error(inverse, triangular, identity);
  const Wide norm = norm_upper(inverse);
  // Neumann's lemma: ||I-XR|| < 1 proves R invertible and bounds ||R^-1||.
  return defect < 1 && norm > 0 ?
      std::nextafter(((1 - defect) / norm) * (1 - 32 * wide_epsilon), Wide(0)) : 0;
}
Wide binary_norm_gamma(Eigen::Index size) {
  const Wide t = static_cast<Wide>(4 * size + 8) * static_cast<Wide>(std::numeric_limits<double>::epsilon());
  return t < 1 ? t / (1 - t) : std::numeric_limits<Wide>::infinity();
}
bool valid_bounds(double a, double b, double budget) {
  return std::isfinite(a) && a >= 0 && std::isfinite(b) && b >= 0 &&
      std::isfinite(budget) && budget >= 0;
}
double up(Wide value) {
  return std::nextafter(static_cast<double>(value), std::numeric_limits<double>::infinity());
}
void finish(NewtonDistanceInterval& out, Wide lower, Wide upper, double budget) {
  // Cover the few scalar operations forming the final endpoints, then round
  // outward when converting back to binary64. No fixed condition cap is used.
  const Wide allowance = gamma(64) * (1 + std::abs(lower) + std::abs(upper));
  out.lower = std::max(0.0, std::nextafter(static_cast<double>(lower - allowance),
                                         -std::numeric_limits<double>::infinity()));
  out.upper = up(upper + allowance);
  if (!std::isfinite(out.upper) || !std::isfinite(out.distance)) return;
  out.error_bound = up(std::max(static_cast<Wide>(out.distance) - static_cast<Wide>(out.lower),
                               static_cast<Wide>(out.upper) - static_cast<Wide>(out.distance)));
  out.status = NewtonAccuracyStatus::Available;
  if (out.upper <= budget) out.decision = NewtonBudgetDecision::WithinBudget;
  else if (out.lower > budget) out.decision = NewtonBudgetDecision::AboveBudget;
}
}  // namespace

std::string_view to_string(NewtonBudgetDecision d) noexcept {
  switch (d) {
    case NewtonBudgetDecision::WithinBudget: return "within_budget";
    case NewtonBudgetDecision::AboveBudget: return "above_budget";
    default: return "unresolved";
  }
}

NewtonDistanceInterval newton_metric_distance_interval(
    const NewtonMetricSystem& system, const Eigen::VectorXd& residual,
    double factor_error, double residual_error, double budget) {
  NewtonDistanceInterval out;
  const auto& a = system.equilibrated_factor;
  const Eigen::Index q = a.cols();
  if (!valid_bounds(factor_error, residual_error, budget) || !residual.allFinite() ||
      residual.size() != a.rows() || !a.allFinite()) return out;
  out.status = system.status;
  if (system.status != NewtonAccuracyStatus::Available) return out;
  out.status = NewtonAccuracyStatus::IllConditioned;
  if (q == 0) {
    out.status = NewtonAccuracyStatus::Available;
    out.distance = out.lower = out.upper = out.error_bound = 0;
    out.decision = NewtonBudgetDecision::WithinBudget;
    return out;
  }
  const Eigen::MatrixXd Q = system.factorization.householderQ() *
      Eigen::MatrixXd::Identity(a.rows(), q);
  const Eigen::MatrixXd R = system.factorization.matrixR().topLeftCorner(q, q)
      .triangularView<Eigen::Upper>();
  const Wide orthogonal = product_error(Q.transpose(), Q, Eigen::MatrixXd::Identity(q, q));
  out.orthogonality_error_bound = up(orthogonal);
  const Eigen::MatrixXd permuted = a * system.factorization.colsPermutation();
  const Wide delta_a = (static_cast<Wide>(factor_error) + product_error(Q, R, permuted)) *
      (1 + 32 * wide_epsilon);
  out.factor_error_bound = up(delta_a);
  if (!(orthogonal < 1)) return out;
  const Wide smin = std::sqrt(1 - orthogonal) * singular_lower(R, false) *
      (1 - 32 * wide_epsilon);
  const Wide margin = smin - delta_a;
  out.rank_margin = std::max(0.0, std::nextafter(static_cast<double>(margin), 0.0));
  if (!(margin > 0)) return out;
  const Eigen::VectorXd projected = Q.transpose() * residual;
  out.distance = detail::stable_norm(projected);
  const Wide bnorm = norm_upper(residual);
  const Wide arithmetic = product_error(Q.transpose(), residual, projected) +
      binary_norm_gamma(q) * norm_upper(projected);
  // P_Q is the orthogonal projector onto the *computed* Q's column space.
  // Its singular values account for Q's orthogonality defect. The equal-rank
  // projector perturbation is bounded by delta_a / (smin(QR)-delta_a).
  const Wide error = bnorm * (delta_a / margin +
      orthogonal / (1 + std::sqrt(1 - orthogonal))) + static_cast<Wide>(residual_error) + arithmetic;
  finish(out, std::max(Wide(0), static_cast<Wide>(out.distance) - error),
         static_cast<Wide>(out.distance) + error, budget);
  return out;
}

NewtonDistanceInterval newton_hessian_distance_interval(
    const NewtonSystem& system, const Eigen::VectorXd& gradient,
    double hessian_error, double gradient_error, double budget) {
  NewtonDistanceInterval out;
  const Eigen::Index q = system.scale.size();
  if (!valid_bounds(hessian_error, gradient_error, budget)) return out;
  out.status = system.status;
  if (system.status != NewtonAccuracyStatus::Available) return out;
  if (gradient.size() != q || !gradient.allFinite()) {
    out.status = NewtonAccuracyStatus::Unavailable; return out;
  }
  if (system.coordinate_map.size()) { out.status = NewtonAccuracyStatus::Unsupported; return out; }
  out.status = NewtonAccuracyStatus::IllConditioned;
  if (q == 0) {
    out.status = NewtonAccuracyStatus::Available;
    out.distance = out.lower = out.upper = out.error_bound = 0;
    out.decision = NewtonBudgetDecision::WithinBudget;
    return out;
  }
  const Eigen::MatrixXd L = system.factorization.matrixL();
  const Wide minimum = singular_lower(L, true);
  const Wide eigen_lower = minimum * minimum * (1 - 32 * wide_epsilon);
  const Wide delta_h = (static_cast<Wide>(hessian_error) +
      product_error(L, L.transpose(), system.equilibrated_hessian)) * (1 + 32 * wide_epsilon);
  out.factor_error_bound = up(delta_h);
  const Wide margin = eigen_lower - delta_h;
  out.rank_margin = std::max(0.0, std::nextafter(static_cast<double>(margin), 0.0));
  if (!(margin > 0)) return out;
  const Eigen::VectorXd b = system.scale.cwiseProduct(gradient);
  const Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(b);
  if (!y.allFinite()) return out;
  out.distance = detail::stable_norm(y);
  const Wide solve_error = product_error(L, y, b) / minimum +
      binary_norm_gamma(q) * norm_upper(y);
  const Wide b_error = static_cast<Wide>(gradient_error) + product_error(system.scale.asDiagonal().toDenseMatrix(), gradient, b);
  const Wide alpha = delta_h / eigen_lower;
  finish(out, std::max(Wide(0), (static_cast<Wide>(out.distance) - solve_error) / std::sqrt(1 + alpha) -
                             b_error / std::sqrt(margin)),
         (static_cast<Wide>(out.distance) + solve_error) / std::sqrt(1 - alpha) + b_error / std::sqrt(margin), budget);
  return out;
}
double newton_curvature_lower_bound(const NewtonSystem& system, double error) {
  if (!std::isfinite(error) || error < 0) return std::numeric_limits<double>::quiet_NaN();
  if (system.status != NewtonAccuracyStatus::Available) return 0;
  if (system.scale.size() == 0) return std::numeric_limits<double>::infinity();
  const Eigen::MatrixXd L = system.factorization.matrixL();
  const Wide minimum = singular_lower(L, true);
  const Wide delta = (static_cast<Wide>(error) +
      product_error(L, L.transpose(), system.equilibrated_hessian)) * (1 + 32 * wide_epsilon);
  return std::max(0.0, std::nextafter(static_cast<double>(minimum * minimum *
      (1 - 32 * wide_epsilon) - delta), 0.0));
}
}  // namespace magmaan::estimate::frontier
