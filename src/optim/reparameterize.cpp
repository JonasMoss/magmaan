#include "magmaan/optim/reparameterize.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/problem.hpp"

namespace magmaan::optim {

GmmProblem
reparameterize(const GmmProblem& prob,
               const estimate::EqConstraints& con) {
  GmmProblem out;
  out.n_resid = prob.n_resid;
  out.n_param = static_cast<Eigen::Index>(con.n_alpha);

  out.r = [r = prob.r, con](
              const Eigen::VectorXd& a) -> fit_expected<Eigen::VectorXd> {
    return r(con.expand(a));
  };
  out.J = [J = prob.J, con](
              const Eigen::VectorXd& a) -> fit_expected<Eigen::MatrixXd> {
    auto Jt = J(con.expand(a));
    if (!Jt.has_value()) return std::unexpected(Jt.error());
    return Eigen::MatrixXd(*Jt * con.Kmat);
  };
  if (prob.eval) {
    out.eval = [eval = prob.eval, con](
                   const Eigen::VectorXd& a) -> fit_expected<LsEvaluation> {
      auto e = eval(con.expand(a));
      if (!e.has_value()) return std::unexpected(e.error());
      return LsEvaluation{std::move(e->residual),
                          Eigen::MatrixXd(e->jacobian * con.Kmat)};
    };
  }
  out.expand = [e = prob.expand, con](const Eigen::VectorXd& a) {
    return e(con.expand(a));
  };
  return out;
}

ScalarProblem
reparameterize(const ScalarProblem& prob,
               const estimate::EqConstraints& con) {
  ScalarProblem out;
  out.n_param = static_cast<Eigen::Index>(con.n_alpha);

  out.f = [f = prob.f, con](const Eigen::VectorXd& a,
                            Eigen::VectorXd& grad) -> double {
    const Eigen::VectorXd theta = con.expand(a);
    Eigen::VectorXd grad_theta(theta.size());
    const double v = f(theta, grad_theta);
    grad = con.reduce_gradient(grad_theta);
    return v;
  };
  out.expand = [e = prob.expand, con](const Eigen::VectorXd& a) {
    return e(con.expand(a));
  };
  return out;
}

estimate::Bounds
fold_alpha_bounds(const estimate::EqConstraints& con,
                  const estimate::Bounds& b) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const Eigen::Index na = con.Kmat.cols();
  estimate::Bounds out;
  out.lower = Eigen::VectorXd::Constant(na, -kInf);
  out.upper = Eigen::VectorXd::Constant(na, kInf);
  for (Eigen::Index k = 0; k < b.lower.size(); ++k) {
    const auto g =
        static_cast<Eigen::Index>(con.group[static_cast<std::size_t>(k)]);
    out.lower(g) = std::max(out.lower(g), b.lower(k));
    out.upper(g) = std::min(out.upper(g), b.upper(k));
  }
  return out;
}

}  // namespace magmaan::optim
