#include "magmaan/measures/reliability.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "../detail_vech.hpp"

namespace magmaan::measures::frontier::reliability {

namespace {

PostError err(std::string detail) {
  return PostError{PostError::Kind::NumericIssue, std::move(detail)};
}

bool finite_matrix(const Eigen::Ref<const Eigen::MatrixXd>& M) {
  return M.allFinite();
}

post_expected<Eigen::MatrixXd>
sym_cov(const Eigen::Ref<const Eigen::MatrixXd>& S, Eigen::Index p_min,
        const char* call) {
  if (S.rows() != S.cols()) {
    return std::unexpected(err(std::string(call) + ": covariance matrix must be square"));
  }
  if (S.rows() < p_min) {
    return std::unexpected(err(std::string(call) + ": too few items"));
  }
  if (!finite_matrix(S)) {
    return std::unexpected(err(std::string(call) + ": covariance matrix is not finite"));
  }
  return Eigen::MatrixXd(0.5 * (S + S.transpose()).eval());
}

double total_variance(const Eigen::Ref<const Eigen::MatrixXd>& S) {
  return S.sum();
}

post_expected<double>
check_total(const Eigen::Ref<const Eigen::MatrixXd>& S, const char* call) {
  const double t = total_variance(S);
  if (!std::isfinite(t) || t <= 0.0) {
    return std::unexpected(err(std::string(call) + ": total-score variance must be positive"));
  }
  return t;
}

Eigen::VectorXd total_gradient(Eigen::Index p) {
  Eigen::VectorXd out(detail::vech_len(p));
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      out(k++) = (i == j) ? 1.0 : 2.0;
    }
  }
  return out;
}

Eigen::VectorXd diag_gradient(Eigen::Index p) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(detail::vech_len(p));
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      if (i == j) out(k) = 1.0;
      ++k;
    }
  }
  return out;
}

post_expected<Eigen::MatrixXd>
inverse_spd(const Eigen::Ref<const Eigen::MatrixXd>& S, const char* call) {
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(err(std::string(call) + ": covariance matrix is not positive definite"));
  }
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(S.rows(), S.cols());
  Eigen::MatrixXd inv = llt.solve(I);
  if (!inv.allFinite()) {
    return std::unexpected(err(std::string(call) + ": covariance inverse is not finite"));
  }
  return Eigen::MatrixXd(0.5 * (inv + inv.transpose()).eval());
}

post_expected<double>
lambda6_residual_sum(const Eigen::Ref<const Eigen::MatrixXd>& S,
                     const Eigen::Ref<const Eigen::MatrixXd>& Sinv) {
  double e_sum = 0.0;
  for (Eigen::Index i = 0; i < S.rows(); ++i) {
    const double kii = Sinv(i, i);
    if (!std::isfinite(kii) || kii <= 0.0) {
      return std::unexpected(err("guttman_lambda6: nonpositive inverse-covariance diagonal"));
    }
    e_sum += 1.0 / kii;
  }
  return e_sum;
}

Eigen::VectorXd vech_symmetric_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& G) {
  const Eigen::Index p = G.rows();
  Eigen::VectorXd out(detail::vech_len(p));
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      out(k++) = (i == j) ? G(i, i) : G(i, j) + G(j, i);
    }
  }
  return out;
}

post_expected<Eigen::VectorXd>
finite_difference_gradient(Coefficient coef,
                           const Eigen::Ref<const Eigen::MatrixXd>& S) {
  const Eigen::Index p = S.rows();
  auto base_or = value(coef, S);
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  Eigen::VectorXd g(detail::vech_len(p));
  Eigen::MatrixXd Sp = S;
  Eigen::MatrixXd Sm = S;
  Eigen::Index k = 0;
  constexpr double rel = 1e-6;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      const double scale = std::max(1.0, std::abs(S(i, j)));
      const double h = rel * scale;
      Sp = S;
      Sm = S;
      Sp(i, j) += h;
      Sm(i, j) -= h;
      if (i != j) {
        Sp(j, i) += h;
        Sm(j, i) -= h;
      }
      auto fp_or = value(coef, Sp);
      if (!fp_or.has_value()) return std::unexpected(fp_or.error());
      auto fm_or = value(coef, Sm);
      if (!fm_or.has_value()) return std::unexpected(fm_or.error());
      g(k++) = (*fp_or - *fm_or) / (2.0 * h);
    }
  }
  return g;
}

}  // namespace

post_expected<double>
cronbach_alpha(const Eigen::Ref<const Eigen::MatrixXd>& S) {
  auto S_or = sym_cov(S, 2, "cronbach_alpha");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  auto t_or = check_total(*S_or, "cronbach_alpha");
  if (!t_or.has_value()) return std::unexpected(t_or.error());
  const double p = static_cast<double>(S_or->rows());
  return (p / (p - 1.0)) * (1.0 - S_or->trace() / *t_or);
}

post_expected<double>
guttman_lambda6(const Eigen::Ref<const Eigen::MatrixXd>& S) {
  auto S_or = sym_cov(S, 2, "guttman_lambda6");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  auto t_or = check_total(*S_or, "guttman_lambda6");
  if (!t_or.has_value()) return std::unexpected(t_or.error());
  auto Sinv_or = inverse_spd(*S_or, "guttman_lambda6");
  if (!Sinv_or.has_value()) return std::unexpected(Sinv_or.error());
  auto e_or = lambda6_residual_sum(*S_or, *Sinv_or);
  if (!e_or.has_value()) return std::unexpected(e_or.error());
  return 1.0 - *e_or / *t_or;
}

post_expected<double>
spearman_guttman_omega(const Eigen::Ref<const Eigen::MatrixXd>& S) {
  auto S_or = sym_cov(S, 3, "spearman_guttman_omega");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  auto t_or = check_total(*S_or, "spearman_guttman_omega");
  if (!t_or.has_value()) return std::unexpected(t_or.error());

  const Eigen::Index p = S_or->rows();
  const double denom = static_cast<double>((p - 1) * (p - 2) / 2);
  double h_sum = 0.0;
  for (Eigen::Index i = 0; i < p; ++i) {
    double h_i = 0.0;
    for (Eigen::Index ell = 0; ell < p; ++ell) {
      for (Eigen::Index j = 0; j < ell; ++j) {
        if (j == i || ell == i) continue;
        const double d = (*S_or)(j, ell);
        if (!std::isfinite(d) || d == 0.0) {
          return std::unexpected(err(
              "spearman_guttman_omega: zero pair covariance in communality denominator"));
        }
        h_i += (*S_or)(i, j) * (*S_or)(i, ell) / d;
      }
    }
    h_sum += h_i / denom;
  }
  const double numerator = *t_or - S_or->trace() + h_sum;
  return numerator / *t_or;
}

post_expected<double>
value(Coefficient coef, const Eigen::Ref<const Eigen::MatrixXd>& S) {
  switch (coef) {
    case Coefficient::Alpha:
      return cronbach_alpha(S);
    case Coefficient::Lambda6:
      return guttman_lambda6(S);
    case Coefficient::SpearmanGuttmanOmega:
      return spearman_guttman_omega(S);
  }
  return std::unexpected(err("reliability value: unknown coefficient"));
}

post_expected<Eigen::VectorXd>
gradient(Coefficient coef, const Eigen::Ref<const Eigen::MatrixXd>& S) {
  auto S_or = sym_cov(S, coef == Coefficient::SpearmanGuttmanOmega ? 3 : 2,
                      "reliability gradient");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  auto t_or = check_total(*S_or, "reliability gradient");
  if (!t_or.has_value()) return std::unexpected(t_or.error());

  const Eigen::Index p = S_or->rows();
  const Eigen::VectorXd gt = total_gradient(p);
  if (coef == Coefficient::Alpha) {
    const double c = static_cast<double>(p) / static_cast<double>(p - 1);
    const double d = S_or->trace();
    return Eigen::VectorXd((-c / *t_or) * diag_gradient(p) +
                           (c * d / (*t_or * *t_or)) * gt);
  }
  if (coef == Coefficient::Lambda6) {
    auto Sinv_or = inverse_spd(*S_or, "guttman_lambda6 gradient");
    if (!Sinv_or.has_value()) return std::unexpected(Sinv_or.error());
    auto e_or = lambda6_residual_sum(*S_or, *Sinv_or);
    if (!e_or.has_value()) return std::unexpected(e_or.error());

    Eigen::MatrixXd dE = Eigen::MatrixXd::Zero(p, p);
    for (Eigen::Index i = 0; i < p; ++i) {
      const double kii = (*Sinv_or)(i, i);
      const Eigen::VectorXd kcol = Sinv_or->col(i);
      dE.noalias() += (kcol * kcol.transpose()) / (kii * kii);
    }
    return Eigen::VectorXd((-1.0 / *t_or) * vech_symmetric_gradient(dE) +
                           (*e_or / (*t_or * *t_or)) * gt);
  }
  return finite_difference_gradient(coef, *S_or);
}

post_expected<DeltaResult>
delta_method(Coefficient coef,
             const Eigen::Ref<const Eigen::MatrixXd>& S,
             const Eigen::Ref<const Eigen::MatrixXd>& gamma,
             std::int64_t n) {
  if (n <= 0) {
    return std::unexpected(err("reliability delta_method: n must be positive"));
  }
  auto S_or = sym_cov(S, coef == Coefficient::SpearmanGuttmanOmega ? 3 : 2,
                      "reliability delta_method");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  const Eigen::Index pstar = detail::vech_len(S_or->rows());
  if (gamma.rows() != pstar || gamma.cols() != pstar || !gamma.allFinite()) {
    return std::unexpected(err("reliability delta_method: gamma has invalid shape or non-finite entries"));
  }
  auto v_or = value(coef, *S_or);
  if (!v_or.has_value()) return std::unexpected(v_or.error());
  auto g_or = gradient(coef, *S_or);
  if (!g_or.has_value()) return std::unexpected(g_or.error());
  const double avar = g_or->dot(gamma * *g_or);
  if (!std::isfinite(avar)) {
    return std::unexpected(err("reliability delta_method: asymptotic variance is not finite"));
  }
  DeltaResult out;
  out.value = *v_or;
  out.gradient = std::move(*g_or);
  out.avar = std::max(0.0, avar);
  out.se = std::sqrt(out.avar / static_cast<double>(n));
  return out;
}

}  // namespace magmaan::measures::frontier::reliability
