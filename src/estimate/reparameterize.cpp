#include "magmaan/estimate/reparameterize.hpp"

#include <utility>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/problem.hpp"

namespace magmaan::estimate {

optim::GmmProblem
reparameterize(const optim::GmmProblem& prob, const EqConstraints& con) {
  optim::GmmProblem out;
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
  out.expand = [e = prob.expand, con](const Eigen::VectorXd& a) {
    return e(con.expand(a));
  };
  return out;
}

optim::ScalarProblem
reparameterize(const optim::ScalarProblem& prob, const EqConstraints& con) {
  optim::ScalarProblem out;
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

}  // namespace magmaan::estimate
