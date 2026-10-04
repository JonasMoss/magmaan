#include "magmaan/api/policy.hpp"

#include <algorithm>

#include <Eigen/QR>
#include <Eigen/SVD>
#include "magmaan/estimate/constraints.hpp"

namespace magmaan::api::frontier {

post_expected<MomentNestedTangent> moment_nested_tangent(
    spec::LatentStructure null_pt, const model::MatrixRep& null_rep,
    const estimate::Estimates& null_estimates,
    spec::LatentStructure alternative_pt, const model::MatrixRep& alternative_rep,
    const estimate::Estimates& alternative_estimates, const data::OrdinalStats& stats,
    estimate::OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* null_row_user,
    const std::vector<std::int8_t>* alternative_row_user) {
  auto fail = [](PostError::Kind kind, const std::string& detail)
      -> post_expected<MomentNestedTangent> {
    return std::unexpected(PostError{kind, "DWLS moment nesting: " + detail});
  };
  if (null_rep.ov_names != alternative_rep.ov_names ||
      null_pt.n_groups() != alternative_pt.n_groups() ||
      null_pt.n_levels() != alternative_pt.n_levels())
    return fail(PostError::Kind::NotNested, "observed variables or group/level layouts differ");
  auto o0 = estimate::frontier::ordinal_ls_objective(null_pt, null_rep, stats,
      null_estimates, estimate::OrdinalWeightKind::DWLS, parameterization, null_row_user);
  auto o1 = estimate::frontier::ordinal_ls_objective(alternative_pt, alternative_rep, stats,
      alternative_estimates, estimate::OrdinalWeightKind::DWLS, parameterization, alternative_row_user);
  if (!o0 || !o1) return fail(PostError::Kind::NumericIssue,
      !o0 ? o0.error().detail : o1.error().detail);
  auto c0 = estimate::build_eq_constraints(o0->pt);
  auto c1 = estimate::build_eq_constraints(o1->pt);
  if (!c0 || !c1) return fail(PostError::Kind::UnsupportedNesting,
      !c0 ? c0.error().detail : c1.error().detail);
  const auto q0 = c0->K().cols(), q1 = c1->K().cols();
  if (q0 > q1) return fail(PostError::Kind::NotNested, "null has more free directions");
  auto target = o0->problem.r(null_estimates.theta);
  auto j0 = o0->problem.J(null_estimates.theta);
  if (!target || !j0) return fail(PostError::Kind::NumericIssue,
      !target ? target.error().detail : j0.error().detail);
  Eigen::VectorXd alpha = c1->contract(alternative_estimates.theta);
  Eigen::VectorXd theta = c1->expand(alpha);
  // Fitting to the null's implied moments is equivalent to matching weighted
  // residual vectors: the shared observed moments cancel. Backtracking keeps
  // every trial inside the evaluator's covariance domain.
  const double tol = 1e-9 * std::max(1.0, target->norm());
  for (int iter = 0; iter < 100; ++iter) {
    auto ev = o1->problem.eval(theta);
    if (!ev) return fail(PostError::Kind::NumericIssue, ev.error().detail);
    const Eigen::VectorXd residual = ev->residual - *target;
    if (residual.norm() <= tol) break;
    const Eigen::MatrixXd J = ev->jacobian * c1->K();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J);
    qr.setThreshold(1e-10);
    if (qr.rank() != q1) return fail(PostError::Kind::BoundaryNesting,
        "alternative moment Jacobian is rank deficient");
    const Eigen::VectorXd step = qr.solve(-residual);
    bool accepted = false;
    for (double scale = 1.0; scale >= 1e-8; scale *= 0.5) {
      const Eigen::VectorXd trial = c1->expand(alpha + scale * step);
      auto r = o1->problem.r(trial);
      if (r && (*r - *target).norm() < residual.norm()) {
        alpha += scale * step;
        theta = trial;
        accepted = true;
        break;
      }
    }
    if (!accepted) break;
  }
  auto ev = o1->problem.eval(theta);
  if (!ev) return fail(PostError::Kind::NumericIssue, ev.error().detail);
  if ((ev->residual - *target).norm() > tol)
    return fail(PostError::Kind::NotNested, "no zero-residual implied-moment embedding found");
  const Eigen::MatrixXd D1 = ev->jacobian * c1->K(), D0 = *j0 * c0->K();
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(D1);
  qr.setThreshold(1e-10);
  if (qr.rank() != q1) return fail(PostError::Kind::BoundaryNesting,
      "alternative moment Jacobian is rank deficient at embedding");
  const Eigen::MatrixXd T = qr.solve(D0);
  if ((D1 * T - D0).norm() > 1e-8 * std::max(1.0, D0.norm()))
    return fail(PostError::Kind::NotNested, "null moment tangent is not included");
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeFullU);
  svd.setThreshold(1e-10);
  if (svd.rank() != q0) return fail(PostError::Kind::BoundaryNesting,
      "null tangent is rank deficient");
  return MomentNestedTangent{std::move(theta), T,
      svd.matrixU().rightCols(q1 - q0).transpose()};
}

} // namespace magmaan::api::frontier
