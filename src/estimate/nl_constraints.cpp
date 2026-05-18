#include "magmaan/estimate/nl_constraints.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace magmaan::estimate {

namespace {

// Forward-mode AD over one constraint's node pool. Children always precede
// their parent in the pool (the compiler appends them first), so a single
// forward sweep yields each node's value and θ-gradient; the root carries h
// and ∂h/∂θ.
void eval_pool(const spec::NlConstraint& c, const Eigen::VectorXd& theta,
               double& value, Eigen::VectorXd& grad) {
  const Eigen::Index npar = theta.size();
  const std::size_t  n    = c.nodes.size();
  std::vector<double>          val(n, 0.0);
  std::vector<Eigen::VectorXd> g(n, Eigen::VectorXd::Zero(npar));
  for (std::size_t i = 0; i < n; ++i) {
    const spec::NlExprNode& nd = c.nodes[i];
    switch (nd.kind) {
      case spec::NlExprNode::Kind::Const:
        val[i] = nd.constant;                  // gradient stays zero
        break;
      case spec::NlExprNode::Kind::Param:
        if (nd.free_idx >= 0 && nd.free_idx < npar) {
          val[i] = theta(nd.free_idx);
          g[i](nd.free_idx) = 1.0;
        } else {
          val[i] = nd.constant;                // a fixed-parameter reference
        }
        break;
      case spec::NlExprNode::Kind::Unary: {
        const std::size_t L = static_cast<std::size_t>(nd.lhs);
        const double a = val[L];
        switch (nd.un_op) {
          case parse::UnOp::Neg:
            val[i] = -a;  g[i] = -g[L];
            break;
          case parse::UnOp::Pos:
            val[i] =  a;  g[i] =  g[L];
            break;
          case parse::UnOp::Exp: {
            // (exp u)' = exp(u)·u'
            const double e = std::exp(a);
            val[i] = e;  g[i] = e * g[L];
            break;
          }
          case parse::UnOp::Log: {
            // (log u)' = u'/u. A transient u ≤ 0 mid-optimization surfaces as
            // a NaN value to the augmented-Lagrangian feasibility check (as
            // the Pow node does); the gradient denominator is floored so the
            // L-BFGS step direction stays finite.
            const double d = (a > 0.0) ? a : 1e-300;
            val[i] = std::log(a);
            g[i]   = g[L] / d;
            break;
          }
        }
        break;
      }
      case spec::NlExprNode::Kind::Binary: {
        const std::size_t L = static_cast<std::size_t>(nd.lhs);
        const std::size_t R = static_cast<std::size_t>(nd.rhs);
        const double a = val[L], b = val[R];
        const Eigen::VectorXd& ga = g[L];
        const Eigen::VectorXd& gb = g[R];
        switch (nd.bin_op) {
          case parse::BinOp::Add:
            val[i] = a + b;  g[i] = ga + gb;
            break;
          case parse::BinOp::Sub:
            val[i] = a - b;  g[i] = ga - gb;
            break;
          case parse::BinOp::Mul:
            val[i] = a * b;  g[i] = b * ga + a * gb;   // (ab)' = b·a' + a·b'
            break;
          case parse::BinOp::Div: {
            const double d = (b != 0.0) ? b : 1e-300;  // guard a transient 0
            val[i] = a / d;
            g[i]   = (ga * d - a * gb) / (d * d);       // (a/b)' = (a'b − ab')/b²
            break;
          }
          case parse::BinOp::Pow: {
            // (a^b)' = b·a^(b-1)·a' + a^b·log(a)·b'. The log term vanishes
            // when the exponent is constant in θ (gb == 0).
            val[i] = std::pow(a, b);
            g[i]   = b * std::pow(a, b - 1.0) * ga;
            if (gb.cwiseAbs().maxCoeff() != 0.0 && a > 0.0) {
              g[i] += val[i] * std::log(a) * gb;
            }
            break;
          }
        }
        break;
      }
    }
  }
  value = val[static_cast<std::size_t>(c.root)];
  grad  = g[static_cast<std::size_t>(c.root)];
}

}  // namespace

Eigen::VectorXd
NonlinearEqConstraints::h(const Eigen::Ref<const Eigen::VectorXd>& theta) const {
  Eigen::VectorXd out(static_cast<Eigen::Index>(rows.size()));
  Eigen::VectorXd grad;
  double v = 0.0;
  for (std::size_t j = 0; j < rows.size(); ++j) {
    eval_pool(rows[j], theta, v, grad);
    out(static_cast<Eigen::Index>(j)) = v;
  }
  return out;
}

Eigen::MatrixXd
NonlinearEqConstraints::jacobian(
    const Eigen::Ref<const Eigen::VectorXd>& theta) const {
  Eigen::MatrixXd J(static_cast<Eigen::Index>(rows.size()), npar);
  Eigen::VectorXd grad;
  double v = 0.0;
  for (std::size_t j = 0; j < rows.size(); ++j) {
    eval_pool(rows[j], theta, v, grad);
    J.row(static_cast<Eigen::Index>(j)) = grad.transpose();
  }
  return J;
}

NonlinearEqConstraints build_nl_constraints(const spec::LatentStructure& pt) {
  NonlinearEqConstraints out;
  out.rows = pt.nl_constraints;
  out.npar = pt.n_free();
  return out;
}

}  // namespace magmaan::estimate
