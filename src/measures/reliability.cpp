#include "magmaan/measures/reliability.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

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

namespace {

// Spearman (1927) ratio-of-sums communality of index i within an index set
// `others` (the co-members of i's factor, or the other factors for the second
// stage), over a symmetric matrix M:
//   h_i = sum_{j<l in others} M_ij M_il / sum_{j<l in others} M_jl.
// On the congeneric manifold each ratio equals lambda_i^2 (or gamma_f^2); the
// pooled ratio-of-sums is the numerically stable Hancock & An (2020) form.
post_expected<double>
ratio_of_sums_communality(const Eigen::MatrixXd& M, Eigen::Index i,
                          const std::vector<Eigen::Index>& others,
                          const char* call) {
  if (others.size() < 2) {
    return std::unexpected(err(std::string(call) +
        ": a factor needs at least three indicators for a closed-form loading"));
  }
  double num = 0.0;
  double den = 0.0;
  for (std::size_t a = 0; a < others.size(); ++a) {
    for (std::size_t b = a + 1; b < others.size(); ++b) {
      const Eigen::Index j = others[a];
      const Eigen::Index l = others[b];
      num += M(i, j) * M(i, l);
      den += M(j, l);
    }
  }
  if (!std::isfinite(den) || std::abs(den) < 1e-12) {
    return std::unexpected(err(std::string(call) +
        ": degenerate communality denominator (near-zero within-factor covariance)"));
  }
  const double h = num / den;
  if (!std::isfinite(h)) {
    return std::unexpected(err(std::string(call) + ": non-finite communality"));
  }
  return h;
}

// Value of the multidimensional closed form on an already-symmetric S.
post_expected<double>
omega_value_impl(OmegaTarget target, const Eigen::MatrixXd& S,
                 const OmegaSpec& spec) {
  const char* call = (target == OmegaTarget::Hierarchical)
                         ? "omega_hierarchical"
                         : "omega_total";
  const Eigen::Index p = S.rows();
  if (spec.block.size() != p) {
    return std::unexpected(err(std::string(call) + ": block length must equal p"));
  }
  int kmax = -1;
  for (Eigen::Index i = 0; i < p; ++i) {
    if (spec.block(i) < 0) {
      return std::unexpected(err(std::string(call) + ": block ids must be non-negative"));
    }
    kmax = std::max(kmax, spec.block(i));
  }
  const Eigen::Index k = static_cast<Eigen::Index>(kmax) + 1;
  std::vector<std::vector<Eigen::Index>> items(static_cast<std::size_t>(k));
  for (Eigen::Index i = 0; i < p; ++i) {
    items[static_cast<std::size_t>(spec.block(i))].push_back(i);
  }
  for (Eigen::Index f = 0; f < k; ++f) {
    if (items[static_cast<std::size_t>(f)].empty()) {
      return std::unexpected(err(std::string(call) + ": empty factor (gap in block ids)"));
    }
  }
  if (target == OmegaTarget::Hierarchical && k < 3) {
    return std::unexpected(err("omega_hierarchical: needs at least three factors"));
  }

  Eigen::VectorXd w;
  if (spec.weights.size() == 0) {
    w = Eigen::VectorXd::Ones(p);
  } else if (spec.weights.size() == p) {
    w = spec.weights;
  } else {
    return std::unexpected(err(std::string(call) + ": weights length must equal p"));
  }

  // Stage 1: C = S with the diagonal replaced by within-block communalities.
  Eigen::MatrixXd C = S;
  for (Eigen::Index f = 0; f < k; ++f) {
    const auto& blk = items[static_cast<std::size_t>(f)];
    for (std::size_t a = 0; a < blk.size(); ++a) {
      std::vector<Eigen::Index> others;
      others.reserve(blk.size() - 1);
      for (std::size_t b = 0; b < blk.size(); ++b) {
        if (b != a) others.push_back(blk[b]);
      }
      auto h_or = ratio_of_sums_communality(S, blk[a], others, call);
      if (!h_or.has_value()) return std::unexpected(h_or.error());
      C(blk[a], blk[a]) = *h_or;
    }
  }

  // B = X'CX, simple-structure block sums.
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(k, k);
  for (Eigen::Index f = 0; f < k; ++f) {
    for (Eigen::Index g = 0; g < k; ++g) {
      double s = 0.0;
      for (Eigen::Index i : items[static_cast<std::size_t>(f)]) {
        for (Eigen::Index j : items[static_cast<std::size_t>(g)]) s += C(i, j);
      }
      B(f, g) = s;
    }
  }
  const double T = w.dot(S * w);
  if (!std::isfinite(T) || T <= 0.0) {
    return std::unexpected(err(std::string(call) + ": total-score variance must be positive"));
  }
  Eigen::LLT<Eigen::MatrixXd> Bllt(B);
  if (Bllt.info() != Eigen::Success) {
    return std::unexpected(err(std::string(call) +
        ": X'CX is not positive definite (weak or collinear factors)"));
  }

  if (target == OmegaTarget::Total) {
    const Eigen::VectorXd Cw = C * w;
    Eigen::VectorXd c = Eigen::VectorXd::Zero(k);
    for (Eigen::Index f = 0; f < k; ++f) {
      double s = 0.0;
      for (Eigen::Index i : items[static_cast<std::size_t>(f)]) s += Cw(i);
      c(f) = s;
    }
    const double num = c.dot(Bllt.solve(c));
    return num / T;
  }

  // Stage 2 (hierarchical): centroid extraction on the factor correlation P.
  const Eigen::VectorXd d = B.diagonal();
  for (Eigen::Index f = 0; f < k; ++f) {
    if (!(d(f) > 0.0)) {
      return std::unexpected(err("omega_hierarchical: non-positive factor variance"));
    }
  }
  Eigen::MatrixXd P(k, k);
  for (Eigen::Index f = 0; f < k; ++f) {
    for (Eigen::Index g = 0; g < k; ++g) {
      P(f, g) = B(f, g) / (std::sqrt(d(f)) * std::sqrt(d(g)));
    }
  }
  Eigen::MatrixXd CP = P;
  for (Eigen::Index f = 0; f < k; ++f) {
    std::vector<Eigen::Index> others;
    others.reserve(static_cast<std::size_t>(k - 1));
    for (Eigen::Index g = 0; g < k; ++g) {
      if (g != f) others.push_back(g);
    }
    auto hf_or = ratio_of_sums_communality(P, f, others, call);
    if (!hf_or.has_value()) return std::unexpected(hf_or.error());
    CP(f, f) = *hf_or;
  }

  const Eigen::VectorXd C1 = C.rowwise().sum();
  Eigen::VectorXd a = Eigen::VectorXd::Zero(k);
  for (Eigen::Index f = 0; f < k; ++f) {
    double s = 0.0;
    for (Eigen::Index i : items[static_cast<std::size_t>(f)]) s += C1(i);
    a(f) = s;
  }
  const Eigen::VectorXd CP1 = CP.rowwise().sum();
  const double denom_cp = CP1.sum();  // 1'C_P 1
  if (!std::isfinite(denom_cp) || denom_cp <= 0.0) {
    return std::unexpected(err("omega_hierarchical: non-positive second-order normalizer"));
  }
  const Eigen::VectorXd v = d.cwiseSqrt().cwiseProduct(CP1);  // D^{1/2} C_P 1
  const double t = a.dot(Bllt.solve(v));                      // a'B^{-1}D^{1/2}C_P 1
  return (t * t) / (denom_cp * T);
}

}  // namespace

post_expected<double>
omega_multidim(OmegaTarget target,
               const Eigen::Ref<const Eigen::MatrixXd>& S,
               const OmegaSpec& spec) {
  auto S_or = sym_cov(S, 2,
                      target == OmegaTarget::Hierarchical ? "omega_hierarchical"
                                                          : "omega_total");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  return omega_value_impl(target, *S_or, spec);
}

post_expected<Eigen::VectorXd>
omega_multidim_gradient(OmegaTarget target,
                        const Eigen::Ref<const Eigen::MatrixXd>& S,
                        const OmegaSpec& spec) {
  auto S_or = sym_cov(S, 2,
                      target == OmegaTarget::Hierarchical ? "omega_hierarchical"
                                                          : "omega_total");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  const Eigen::Index p = S_or->rows();
  auto base_or = omega_value_impl(target, *S_or, spec);
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  Eigen::VectorXd g(detail::vech_len(p));
  Eigen::Index idx = 0;
  constexpr double rel = 1e-6;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      const double h = rel * std::max(1.0, std::abs((*S_or)(i, j)));
      Eigen::MatrixXd Sp = *S_or;
      Eigen::MatrixXd Sm = *S_or;
      Sp(i, j) += h;
      Sm(i, j) -= h;
      if (i != j) {
        Sp(j, i) += h;
        Sm(j, i) -= h;
      }
      auto fp_or = omega_value_impl(target, Sp, spec);
      if (!fp_or.has_value()) return std::unexpected(fp_or.error());
      auto fm_or = omega_value_impl(target, Sm, spec);
      if (!fm_or.has_value()) return std::unexpected(fm_or.error());
      g(idx++) = (*fp_or - *fm_or) / (2.0 * h);
    }
  }
  return g;
}

post_expected<DeltaResult>
omega_multidim_delta(OmegaTarget target,
                     const Eigen::Ref<const Eigen::MatrixXd>& S,
                     const OmegaSpec& spec,
                     const Eigen::Ref<const Eigen::MatrixXd>& gamma,
                     std::int64_t n) {
  if (n <= 0) {
    return std::unexpected(err("omega_multidim_delta: n must be positive"));
  }
  auto S_or = sym_cov(S, 2,
                      target == OmegaTarget::Hierarchical ? "omega_hierarchical"
                                                          : "omega_total");
  if (!S_or.has_value()) return std::unexpected(S_or.error());
  const Eigen::Index pstar = detail::vech_len(S_or->rows());
  if (gamma.rows() != pstar || gamma.cols() != pstar || !gamma.allFinite()) {
    return std::unexpected(err("omega_multidim_delta: gamma has invalid shape or non-finite entries"));
  }
  auto v_or = omega_value_impl(target, *S_or, spec);
  if (!v_or.has_value()) return std::unexpected(v_or.error());
  auto g_or = omega_multidim_gradient(target, *S_or, spec);
  if (!g_or.has_value()) return std::unexpected(g_or.error());
  const double avar = g_or->dot(gamma * *g_or);
  if (!std::isfinite(avar)) {
    return std::unexpected(err("omega_multidim_delta: asymptotic variance is not finite"));
  }
  DeltaResult out;
  out.value = *v_or;
  out.gradient = std::move(*g_or);
  out.avar = std::max(0.0, avar);
  out.se = std::sqrt(out.avar / static_cast<double>(n));
  return out;
}

}  // namespace magmaan::measures::frontier::reliability
