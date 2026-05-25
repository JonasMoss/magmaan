#include "magmaan/estimate/fiml.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"

#include "detail_vech.hpp"

namespace magmaan::estimate::fiml {

using data::RawData;
using data::SampleStats;
using estimate::Estimates;
using estimate::EqConstraints;
using estimate::build_eq_constraints;
using estimate::resolve_fixed_x_from_sample;
using estimate::simple_start_values;
using measures::BaselineFit;

namespace {

FitError make_fit_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

PostError make_post_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

PostError fit_to_post(const FitError& err, std::string prefix) {
  return make_post_err(PostError::Kind::NumericIssue,
                       std::move(prefix) + ": " + err.detail);
}

using detail::vech_index;
using detail::vech_len;

constexpr double two_pi = 6.283185307179586476925286766559;

bool finite_observed_row(const Eigen::MatrixXd& X,
                         const std::vector<Eigen::Index>& obs,
                         Eigen::Index row) {
  for (Eigen::Index c : obs) {
    if (!std::isfinite(X(row, c))) return false;
  }
  return true;
}

fit_expected<void>
validate_raw_shape(const RawData& raw) {
  if (raw.X.empty()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML: RawData.X is empty"));
  }
  if (!raw.mask.empty() && raw.mask.size() != raw.X.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML: RawData.mask must be empty or have one block per X block"));
  }
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    if (X.rows() <= 0) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: block " + std::to_string(b) + " has no rows"));
    }
    if (X.cols() <= 0) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: block " + std::to_string(b) + " has no columns"));
    }
    if (!raw.mask.empty() &&
        (raw.mask[b].rows() != X.rows() || raw.mask[b].cols() != X.cols())) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: mask shape mismatch in block " + std::to_string(b)));
    }
  }
  return {};
}

Eigen::MatrixXd select_square(const Eigen::MatrixXd& M,
                              const std::vector<Eigen::Index>& idx) {
  const Eigen::Index q = static_cast<Eigen::Index>(idx.size());
  Eigen::MatrixXd out(q, q);
  for (Eigen::Index j = 0; j < q; ++j)
    for (Eigen::Index i = 0; i < q; ++i)
      out(i, j) = M(idx[static_cast<std::size_t>(i)],
                    idx[static_cast<std::size_t>(j)]);
  return out;
}

Eigen::VectorXd select_vector(const Eigen::VectorXd& v,
                              const std::vector<Eigen::Index>& idx) {
  const Eigen::Index q = static_cast<Eigen::Index>(idx.size());
  Eigen::VectorXd out(q);
  for (Eigen::Index i = 0; i < q; ++i) {
    out(i) = v(idx[static_cast<std::size_t>(i)]);
  }
  return out;
}

double log_det_from_llt(const Eigen::LLT<Eigen::MatrixXd>& llt) {
  double out = 0.0;
  const auto L = llt.matrixL();
  for (Eigen::Index i = 0; i < L.rows(); ++i) out += std::log(L(i, i));
  return 2.0 * out;
}

std::vector<Eigen::Index>
fixed_x_observed_indices(const spec::LatentStructure& pt) {
  std::unordered_set<std::int32_t> exo_vars;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.exo[i] != 1) continue;
    if (i < pt.lhs_var.size() && pt.lhs_var[i] >= 0) exo_vars.insert(pt.lhs_var[i]);
    if (i < pt.rhs_var.size() && pt.rhs_var[i] >= 0) exo_vars.insert(pt.rhs_var[i]);
  }
  std::vector<Eigen::Index> out;
  std::unordered_set<Eigen::Index> seen;
  for (std::int32_t v : exo_vars) {
    if (v < 0 || static_cast<std::size_t>(v) >= pt.ov_pos.size()) continue;
    const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
    if (pos < 0) continue;
    const Eigen::Index idx = static_cast<Eigen::Index>(pos);
    if (seen.insert(idx).second) out.push_back(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<Eigen::Index>
observed_exogenous_indices(const spec::LatentStructure& pt) {
  std::vector<Eigen::Index> out;
  for (std::int32_t v = 0; v < pt.n_vars; ++v) {
    if (static_cast<std::size_t>(v) >= pt.var_role.size() ||
        pt.var_role[static_cast<std::size_t>(v)] != spec::VarRole::ExoOv ||
        static_cast<std::size_t>(v) >= pt.ov_pos.size()) {
      continue;
    }
    const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
    if (pos >= 0) out.push_back(static_cast<Eigen::Index>(pos));
  }
  std::sort(out.begin(), out.end());
  return out;
}

post_expected<double>
fixed_x_saturated_logl(const spec::LatentStructure& pt,
                       const SampleStats& samp) {
  const std::vector<Eigen::Index> exo_idx = fixed_x_observed_indices(pt);
  if (exo_idx.empty()) return 0.0;

  double logl = 0.0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd& S = samp.S[b];
    const Eigen::Index p = S.rows();
    if (exo_idx.back() >= p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fixed.x exogenous sub-block index out of range"));
    }
    const Eigen::Index px = static_cast<Eigen::Index>(exo_idx.size());
    Eigen::MatrixXd Sxx(px, px);
    for (Eigen::Index r = 0; r < px; ++r) {
      for (Eigen::Index c = 0; c < px; ++c) {
        Sxx(r, c) = S(exo_idx[static_cast<std::size_t>(r)],
                      exo_idx[static_cast<std::size_t>(c)]);
      }
    }
    Eigen::LLT<Eigen::MatrixXd> llt_xx(Sxx);
    if (llt_xx.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fixed.x exogenous sub-block for block " + std::to_string(b) +
              " is not positive definite"));
    }
    const double n_b = static_cast<double>(samp.n_obs[b]);
    logl += -0.5 * n_b *
        (static_cast<double>(px) * std::log(two_pi) +
         log_det_from_llt(llt_xx) + static_cast<double>(px));
  }
  return logl;
}

double observed_constant(const FIMLCache& cache) {
  double c = 0.0;
  for (const FIMLPattern& pat : cache.patterns) {
    c += static_cast<double>(pat.n_obs)
       * static_cast<double>(pat.observed.size()) * std::log(two_pi);
  }
  return c;
}

fit_expected<double>
h1_complete_data_value(const SampleStats& samp) {
  std::int64_t n_total = 0;
  for (auto n : samp.n_obs) n_total += n;
  if (n_total <= 0) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1: no observations"));
  }

  double f = 0.0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd S = 0.5 * (samp.S[b] + samp.S[b].transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSample,
          "FIML H1: complete-data sample covariance is not positive definite"));
    }
    const double scale = static_cast<double>(samp.n_obs[b]) /
                         static_cast<double>(n_total);
    f += scale * (log_det_from_llt(llt) + static_cast<double>(S.rows()));
  }
  return f;
}

Eigen::Index h1_chol_len(Eigen::Index p) {
  return p * (p + 1) / 2;
}

Eigen::VectorXd h1_start_for_block(const SampleStats& samp,
                                   std::size_t block) {
  const Eigen::Index p = samp.S[block].rows();
  Eigen::VectorXd x(p + h1_chol_len(p));
  x.head(p) = samp.mean[block];

  Eigen::MatrixXd S = 0.5 * (samp.S[block] + samp.S[block].transpose());
  const double max_diag = (p > 0) ? S.diagonal().cwiseAbs().maxCoeff() : 1.0;
  const double base = std::max(1.0, max_diag);
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  for (int k = 0; llt.info() != Eigen::Success && k < 10; ++k) {
    S.diagonal().array() += base * std::pow(10.0, -8.0 + static_cast<double>(k));
    llt.compute(S);
  }
  Eigen::MatrixXd L;
  if (llt.info() == Eigen::Success) {
    L = llt.matrixL();
  } else {
    L = Eigen::MatrixXd::Identity(p, p) * std::sqrt(base);
  }

  Eigen::Index off = p;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      x(off++) = (r == c) ? std::log(std::max(L(r, c), 1e-8)) : L(r, c);
    }
  }
  return x;
}

void h1_decode(const Eigen::VectorXd& x,
               Eigen::Index p,
               Eigen::VectorXd& mu,
               Eigen::MatrixXd& L,
               Eigen::MatrixXd& Sigma) {
  mu = x.head(p);
  L = Eigen::MatrixXd::Zero(p, p);
  Eigen::Index off = p;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      L(r, c) = (r == c) ? std::exp(x(off)) : x(off);
      ++off;
    }
  }
  Sigma.noalias() = L * L.transpose();
}

fit_expected<double>
h1_block_value_from_moments(const FIMLCache& cache,
                            std::size_t block,
                            const Eigen::VectorXd& mu,
                            const Eigen::MatrixXd& Sigma) {
  double f = 0.0;
  for (const FIMLPattern& pat : cache.patterns) {
    if (pat.block != block) continue;
    const Eigen::MatrixXd Sigma_o = select_square(Sigma, pat.observed);
    const Eigen::VectorXd Mu_o = select_vector(mu, pat.observed);
    const Eigen::VectorXd d = pat.mean - Mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "FIML H1: saturated observed-pattern covariance is not positive definite"));
    }
    const Eigen::MatrixXd SigmaInv_A = llt.solve(A);
    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);
    f += scale * (log_det_from_llt(llt) + SigmaInv_A.trace());
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_fit_err(FitError::Kind::NonFiniteObjective,
        "FIML H1 objective evaluated to non-finite"));
  }
  return f;
}

std::vector<Eigen::Index>
missing_indices(Eigen::Index p, const std::vector<Eigen::Index>& observed) {
  std::vector<unsigned char> seen(static_cast<std::size_t>(p), 0);
  for (Eigen::Index idx : observed) seen[static_cast<std::size_t>(idx)] = 1;
  std::vector<Eigen::Index> out;
  for (Eigen::Index i = 0; i < p; ++i) {
    if (!seen[static_cast<std::size_t>(i)]) out.push_back(i);
  }
  return out;
}

Eigen::MatrixXd select_rect(const Eigen::MatrixXd& M,
                            const std::vector<Eigen::Index>& rows,
                            const std::vector<Eigen::Index>& cols) {
  Eigen::MatrixXd out(static_cast<Eigen::Index>(rows.size()),
                      static_cast<Eigen::Index>(cols.size()));
  for (Eigen::Index j = 0; j < out.cols(); ++j) {
    for (Eigen::Index i = 0; i < out.rows(); ++i) {
      out(i, j) = M(rows[static_cast<std::size_t>(i)],
                    cols[static_cast<std::size_t>(j)]);
    }
  }
  return out;
}

fit_expected<void>
h1_em_update_block(const FIMLCache& cache,
                   std::size_t block,
                   Eigen::VectorXd& mu,
                   Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = cache.block_p[block];
  Eigen::VectorXd sum_x = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd sum_xx = Eigen::MatrixXd::Zero(p, p);
  std::int64_t n_block = 0;

  for (const FIMLPattern& pat : cache.patterns) {
    if (pat.block != block) continue;
    n_block += pat.n_obs;
    const auto& obs = pat.observed;
    const std::vector<Eigen::Index> miss = missing_indices(p, obs);
    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    const Eigen::Index m = static_cast<Eigen::Index>(miss.size());

    Eigen::VectorXd avg_x = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd avg_xx = Eigen::MatrixXd::Zero(p, p);

    for (Eigen::Index i = 0; i < q; ++i) {
      const Eigen::Index ri = obs[static_cast<std::size_t>(i)];
      avg_x(ri) = pat.mean(i);
      for (Eigen::Index j = 0; j < q; ++j) {
        const Eigen::Index cj = obs[static_cast<std::size_t>(j)];
        avg_xx(ri, cj) = pat.cov(i, j) + pat.mean(i) * pat.mean(j);
      }
    }

    if (m > 0) {
      const Eigen::MatrixXd Sigma_oo = select_square(Sigma, obs);
      Eigen::LLT<Eigen::MatrixXd> llt(Sigma_oo);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
            "FIML H1 EM: observed-pattern covariance is not positive definite"));
      }
      const Eigen::MatrixXd Sigma_mo = select_rect(Sigma, miss, obs);
      const Eigen::MatrixXd Sigma_om = Sigma_mo.transpose();
      const Eigen::MatrixXd Sigma_mm = select_square(Sigma, miss);
      const Eigen::MatrixXd Sigma_oo_inv =
          llt.solve(Eigen::MatrixXd::Identity(q, q));
      const Eigen::MatrixXd B = Sigma_mo * Sigma_oo_inv;
      const Eigen::MatrixXd C = Sigma_mm - B * Sigma_om;

      const Eigen::VectorXd mu_o = select_vector(mu, obs);
      const Eigen::VectorXd mu_m = select_vector(mu, miss);
      const Eigen::VectorXd mbar = mu_m + B * (pat.mean - mu_o);
      const Eigen::MatrixXd mcov = B * pat.cov * B.transpose();
      const Eigen::MatrixXd m2 = C + mcov + mbar * mbar.transpose();
      const Eigen::MatrixXd cross =
          pat.mean * mbar.transpose() + pat.cov * B.transpose();

      for (Eigen::Index i = 0; i < m; ++i) {
        const Eigen::Index ri = miss[static_cast<std::size_t>(i)];
        avg_x(ri) = mbar(i);
        for (Eigen::Index j = 0; j < m; ++j) {
          const Eigen::Index cj = miss[static_cast<std::size_t>(j)];
          avg_xx(ri, cj) = m2(i, j);
        }
      }
      for (Eigen::Index i = 0; i < q; ++i) {
        const Eigen::Index oi = obs[static_cast<std::size_t>(i)];
        for (Eigen::Index j = 0; j < m; ++j) {
          const Eigen::Index mj = miss[static_cast<std::size_t>(j)];
          avg_xx(oi, mj) = cross(i, j);
          avg_xx(mj, oi) = cross(i, j);
        }
      }
    }

    sum_x.noalias() += static_cast<double>(pat.n_obs) * avg_x;
    sum_xx.noalias() += static_cast<double>(pat.n_obs) * avg_xx;
  }

  if (n_block <= 0) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: block has no observations"));
  }
  mu = sum_x / static_cast<double>(n_block);
  Sigma = sum_xx / static_cast<double>(n_block) - mu * mu.transpose();
  Sigma = 0.5 * (Sigma + Sigma.transpose());
  return {};
}

fit_expected<double>
h1_missing_data_value(const FIMLCache& cache, const SampleStats& starts) {
  if (starts.S.size() != cache.block_p.size() ||
      starts.mean.size() != cache.block_p.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1: start statistics do not match raw-data blocks"));
  }

  double total = 0.0;
  for (std::size_t b = 0; b < cache.block_p.size(); ++b) {
    const Eigen::Index p = cache.block_p[b];
    const Eigen::VectorXd x0 = h1_start_for_block(starts, b);
    Eigen::VectorXd mu;
    Eigen::MatrixXd L;
    Eigen::MatrixXd Sigma;
    h1_decode(x0, p, mu, L, Sigma);

    double prev = std::numeric_limits<double>::infinity();
    double cur = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < 10000; ++iter) {
      auto val = h1_block_value_from_moments(cache, b, mu, Sigma);
      if (!val.has_value()) return std::unexpected(val.error());
      cur = *val;
      if (std::isfinite(prev) &&
          std::abs(prev - cur) <= 1e-11 * (1.0 + std::abs(cur))) {
        break;
      }
      prev = cur;
      auto upd = h1_em_update_block(cache, b, mu, Sigma);
      if (!upd.has_value()) return std::unexpected(upd.error());
    }
    auto final_val = h1_block_value_from_moments(cache, b, mu, Sigma);
    if (!final_val.has_value()) return std::unexpected(final_val.error());
    total += *final_val;
  }
  return total;
}

fit_expected<double>
fiml_h1_value(const RawData& raw,
              const FIMLCache& cache,
              const SampleStats& starts) {
  return raw.mask.empty() ? h1_complete_data_value(starts)
                          : h1_missing_data_value(cache, starts);
}

fit_expected<void>
h1_moments_block(const RawData& raw,
                 const FIMLCache& cache,
                 const SampleStats& starts,
                 std::size_t block,
                 Eigen::VectorXd& mu,
                 Eigen::MatrixXd& Sigma) {
  if (block >= cache.block_p.size() || block >= starts.S.size() ||
      block >= starts.mean.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1: block index out of range"));
  }
  if (raw.mask.empty()) {
    mu = starts.mean[block];
    Sigma = 0.5 * (starts.S[block] + starts.S[block].transpose());
    return {};
  }

  const Eigen::Index p = cache.block_p[block];
  const Eigen::VectorXd x0 = h1_start_for_block(starts, block);
  Eigen::MatrixXd L;
  h1_decode(x0, p, mu, L, Sigma);

  double prev = std::numeric_limits<double>::infinity();
  double cur = std::numeric_limits<double>::infinity();
  for (int iter = 0; iter < 10000; ++iter) {
    auto val = h1_block_value_from_moments(cache, block, mu, Sigma);
    if (!val.has_value()) return std::unexpected(val.error());
    cur = *val;
    if (std::isfinite(prev) &&
        std::abs(prev - cur) <= 1e-11 * (1.0 + std::abs(cur))) {
      break;
    }
    prev = cur;
    auto upd = h1_em_update_block(cache, block, mu, Sigma);
    if (!upd.has_value()) return std::unexpected(upd.error());
  }
  return {};
}

post_expected<Eigen::MatrixXd>
invert_symmetric(const Eigen::MatrixXd& A, std::string what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  const Eigen::Index q = A.rows();
  const Eigen::MatrixXd S = 0.5 * (A + A.transpose());
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  if (llt.info() == Eigen::Success) {
    return Eigen::MatrixXd(llt.solve(Eigen::MatrixXd::Identity(q, q)));
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        std::move(what) + " is not invertible"));
  }
  return Eigen::MatrixXd(ldlt.solve(Eigen::MatrixXd::Identity(q, q)));
}

post_expected<Eigen::VectorXd>
observed_row_score(const Eigen::VectorXd& x_o,
                   const std::vector<Eigen::Index>& obs,
                   Eigen::Index p,
                   const Eigen::VectorXd& Mu,
                   const Eigen::MatrixXd& Sigma,
                   const Eigen::MatrixXd& J_sigma,
                   const Eigen::MatrixXd& J_mu,
                   Eigen::Index sigma_off,
                   Eigen::Index mu_off) {
  const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
  const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
  const Eigen::VectorXd Mu_o = select_vector(Mu, obs);
  const Eigen::VectorXd d = x_o - Mu_o;
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust: implied observed-pattern Sigma is not positive definite"));
  }
  const Eigen::MatrixXd SigmaInv =
      llt.solve(Eigen::MatrixXd::Identity(q, q));
  const Eigen::VectorXd z = llt.solve(d);
  Eigen::MatrixXd G = SigmaInv - z * z.transpose();
  G = 0.5 * (G + G.transpose()).eval();

  Eigen::VectorXd w = Eigen::VectorXd::Zero(J_sigma.rows());
  Eigen::VectorXd u = Eigen::VectorXd::Zero(J_mu.rows());
  for (Eigen::Index cj = 0; cj < q; ++cj) {
    const Eigen::Index c = obs[static_cast<std::size_t>(cj)];
    for (Eigen::Index ri = cj; ri < q; ++ri) {
      const Eigen::Index r = obs[static_cast<std::size_t>(ri)];
      const Eigen::Index rr = std::max(r, c);
      const Eigen::Index cc = std::min(r, c);
      const Eigen::Index idx = sigma_off + vech_index(p, rr, cc);
      w(idx) += (ri == cj) ? G(ri, cj) : 2.0 * G(ri, cj);
    }
  }
  for (Eigen::Index i = 0; i < q; ++i) {
    const Eigen::Index r = obs[static_cast<std::size_t>(i)];
    u(mu_off + r) += -2.0 * z(i);
  }
  Eigen::VectorXd score = J_sigma.transpose() * w;
  score.noalias() += J_mu.transpose() * u;
  return score;
}

post_expected<Eigen::MatrixXd>
fiml_casewise_scores(const RawData& raw,
                     const FIMLCache& cache,
                     const model::ImpliedMoments& moments,
                     const Eigen::MatrixXd& J_sigma,
                     const Eigen::MatrixXd& J_mu) {
  if (raw.X.size() != moments.sigma.size() ||
      moments.mu.size() != moments.sigma.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust: raw and implied moment block count mismatch"));
  }
  Eigen::Index n_total = 0;
  for (const auto& X : raw.X) n_total += X.rows();
  Eigen::MatrixXd scores(n_total, J_sigma.cols());

  Eigen::Index row_out = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    const Eigen::Index p = X.cols();
    for (Eigen::Index r = 0; r < X.rows(); ++r) {
      std::vector<Eigen::Index> obs;
      obs.reserve(static_cast<std::size_t>(p));
      if (raw.mask.empty()) {
        for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
      } else {
        for (Eigen::Index c = 0; c < p; ++c)
          if (raw.mask[b](r, c) != 0) obs.push_back(c);
      }
      if (obs.empty()) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML robust: row has no observed values"));
      }
      if (!finite_observed_row(X, obs, r)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML robust: non-finite observed value"));
      }
      Eigen::VectorXd x_o(static_cast<Eigen::Index>(obs.size()));
      for (Eigen::Index j = 0; j < x_o.size(); ++j) {
        x_o(j) = X(r, obs[static_cast<std::size_t>(j)]);
      }
      auto s_or = observed_row_score(
          x_o, obs, p, moments.mu[b], moments.sigma[b], J_sigma, J_mu,
          cache.sigma_offsets[b], cache.mu_offsets[b]);
      if (!s_or.has_value()) return std::unexpected(s_or.error());
      scores.row(row_out++) = s_or->transpose();
    }
  }
  return scores;
}

post_expected<Eigen::MatrixXd>
fiml_observed_hessian_fd(spec::LatentStructure pt,
                         const model::MatrixRep& rep,
                         const RawData& raw,
                         const FIMLCache& cache,
                         const SampleStats& start_samp,
                         const Estimates& est,
                         FIML discrepancy,
                         double h_step) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "resolve_fixed_x_from_sample"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  const Eigen::Index q = static_cast<Eigen::Index>(ev.n_free());
  auto grad_at = [&](const Eigen::VectorXd& theta)
      -> post_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ModelEvaluator::evaluate failed: " + eval.error().detail));
    }
    auto g_or = discrepancy.gradient(raw, cache, eval->moments,
                                     eval->J_sigma, eval->J_mu);
    if (!g_or.has_value()) {
      return std::unexpected(fit_to_post(g_or.error(), "FIML gradient"));
    }
    return *g_or;
  };

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  for (Eigen::Index k = 0; k < q; ++k) {
    Eigen::VectorXd tp = est.theta;
    Eigen::VectorXd tm = est.theta;
    tp(k) += h_step;
    tm(k) -= h_step;
    auto gp = grad_at(tp);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = grad_at(tm);
    if (!gm.has_value()) return std::unexpected(gm.error());
    H.col(k) = (*gp - *gm) / (2.0 * h_step);
  }
  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

Eigen::Index vech_row(Eigen::Index p, Eigen::Index target) {
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      if (k == target) return r;
      ++k;
    }
  }
  return 0;
}

Eigen::Index vech_col(Eigen::Index p, Eigen::Index target) {
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      if (k == target) return c;
      ++k;
    }
  }
  return 0;
}

post_expected<Eigen::MatrixXd>
fiml_saturated_scores_block(const RawData& raw,
                            std::size_t block,
                            const Eigen::VectorXd& mu,
                            const Eigen::MatrixXd& Sigma) {
  if (block >= raw.X.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: block index out of range"));
  }
  if (!raw.mask.empty() && block >= raw.mask.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: mask block index out of range"));
  }
  const Eigen::MatrixXd& X = raw.X[block];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (mu.size() != p || Sigma.rows() != p || Sigma.cols() != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: saturated moments do not match raw block"));
  }
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd scores = Eigen::MatrixXd::Zero(n, p + pstar);
  for (Eigen::Index r = 0; r < n; ++r) {
    std::vector<Eigen::Index> obs;
    obs.reserve(static_cast<std::size_t>(p));
    if (raw.mask.empty()) {
      for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
    } else {
      for (Eigen::Index c = 0; c < p; ++c)
        if (raw.mask[block](r, c) != 0) obs.push_back(c);
    }
    if (obs.empty() || !finite_observed_row(X, obs, r)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML robust H1: invalid observed row"));
    }
    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    Eigen::VectorXd x_o(q);
    for (Eigen::Index j = 0; j < q; ++j) {
      x_o(j) = X(r, obs[static_cast<std::size_t>(j)]);
    }
    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    const Eigen::VectorXd Mu_o = select_vector(mu, obs);
    const Eigen::VectorXd d = x_o - Mu_o;
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML robust H1: Sigma_oo is not positive definite"));
    }
    const Eigen::MatrixXd SigmaInv =
        llt.solve(Eigen::MatrixXd::Identity(q, q));
    const Eigen::VectorXd z = llt.solve(d);
    Eigen::MatrixXd G = SigmaInv - z * z.transpose();
    G = 0.5 * (G + G.transpose()).eval();
    for (Eigen::Index i = 0; i < q; ++i) {
      const Eigen::Index rr = obs[static_cast<std::size_t>(i)];
      scores(r, rr) = -2.0 * z(i);
    }
    for (Eigen::Index cj = 0; cj < q; ++cj) {
      const Eigen::Index c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = cj; ri < q; ++ri) {
        const Eigen::Index rr0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index rr = std::max(rr0, c);
        const Eigen::Index cc = std::min(rr0, c);
        const Eigen::Index idx = p + vech_index(p, rr, cc);
        scores(r, idx) += (ri == cj) ? G(ri, cj) : 2.0 * G(ri, cj);
      }
    }
  }
  return scores;
}

post_expected<Eigen::MatrixXd>
fiml_saturated_hessian_fd_block(const RawData& raw,
                                std::size_t block,
                                const Eigen::VectorXd& mu,
                                const Eigen::MatrixXd& Sigma,
                                double h_step) {
  const Eigen::Index p = mu.size();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index q = p + pstar;
  auto grad_at = [&](const Eigen::VectorXd& mu0,
                     const Eigen::MatrixXd& Sigma0)
      -> post_expected<Eigen::VectorXd> {
    Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (Sigma0 + Sigma0.transpose()));
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML robust H1: perturbed Sigma is not positive definite"));
    }
    auto S_or = fiml_saturated_scores_block(raw, block, mu0, Sigma0);
    if (!S_or.has_value()) return std::unexpected(S_or.error());
    return Eigen::VectorXd(S_or->colwise().mean().transpose());
  };

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  for (Eigen::Index k = 0; k < q; ++k) {
    Eigen::VectorXd mup = mu;
    Eigen::VectorXd mum = mu;
    Eigen::MatrixXd Sp = Sigma;
    Eigen::MatrixXd Sm = Sigma;
    if (k < p) {
      mup(k) += h_step;
      mum(k) -= h_step;
    } else {
      const Eigen::Index vk = k - p;
      const Eigen::Index r = vech_row(p, vk);
      const Eigen::Index c = vech_col(p, vk);
      Sp(r, c) += h_step;
      Sm(r, c) -= h_step;
      if (r != c) {
        Sp(c, r) += h_step;
        Sm(c, r) -= h_step;
      }
    }
    auto gp = grad_at(mup, Sp);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = grad_at(mum, Sm);
    if (!gm.has_value()) return std::unexpected(gm.error());
    H.col(k) = (*gp - *gm) / (2.0 * h_step);
  }
  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

post_expected<double>
fiml_saturated_trace_h1(const RawData& raw,
                        const FIMLCache& cache,
                        const SampleStats& start_samp,
                        double h_step) {
  if (cache.block_p.size() != raw.X.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: cache and raw block count mismatch"));
  }

  double trace = 0.0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    Eigen::VectorXd h1_mu;
    Eigen::MatrixXd h1_sigma;
    auto h1_mom_or = h1_moments_block(raw, cache, start_samp, b,
                                      h1_mu, h1_sigma);
    if (!h1_mom_or.has_value()) {
      return std::unexpected(fit_to_post(h1_mom_or.error(),
          "FIML H1 moments"));
    }
    auto h1_scores_or = fiml_saturated_scores_block(raw, b,
                                                    h1_mu, h1_sigma);
    if (!h1_scores_or.has_value()) return std::unexpected(h1_scores_or.error());
    auto h1_H_or = fiml_saturated_hessian_fd_block(raw, b, h1_mu,
                                                  h1_sigma, h_step);
    if (!h1_H_or.has_value()) return std::unexpected(h1_H_or.error());

    const double n_block = static_cast<double>(raw.X[b].rows());
    const Eigen::MatrixXd h1_meat =
        (h1_scores_or->transpose() * (*h1_scores_or)) / n_block;
    auto h1_Hinv_or = invert_symmetric(*h1_H_or,
                                       "fiml_robust_mlr: H1 observed Hessian");
    if (!h1_Hinv_or.has_value()) return std::unexpected(h1_Hinv_or.error());
    trace += 0.5 * ((*h1_Hinv_or) * h1_meat).trace();
  }
  return trace;
}

post_expected<void>
independence_moments_from_raw(const RawData& raw,
                              std::vector<Eigen::VectorXd>& means,
                              std::vector<Eigen::VectorXd>& vars) {
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(fit_to_post(ok.error(), "FIML baseline"));
  }

  means.clear();
  vars.clear();
  means.reserve(raw.X.size());
  vars.reserve(raw.X.size());

  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd count = Eigen::VectorXd::Zero(p);

    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index c = 0; c < p; ++c) {
        const bool observed = raw.mask.empty() || raw.mask[b](r, c) != 0;
        if (!observed) continue;
        const double x = X(r, c);
        if (!std::isfinite(x)) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "FIML baseline: non-finite observed value in block " +
                  std::to_string(b)));
        }
        mean(c) += x;
        count(c) += 1.0;
      }
    }

    for (Eigen::Index c = 0; c < p; ++c) {
      if (count(c) <= 0.0) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML baseline: block " + std::to_string(b) + " column " +
                std::to_string(c) + " has no observed values"));
      }
      mean(c) /= count(c);
    }

    Eigen::VectorXd var = Eigen::VectorXd::Zero(p);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index c = 0; c < p; ++c) {
        const bool observed = raw.mask.empty() || raw.mask[b](r, c) != 0;
        if (!observed) continue;
        const double d = X(r, c) - mean(c);
        var(c) += d * d;
      }
    }
    for (Eigen::Index c = 0; c < p; ++c) {
      var(c) /= count(c);
      if (!(var(c) > 0.0) || !std::isfinite(var(c))) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML baseline: block " + std::to_string(b) + " column " +
                std::to_string(c) + " variance is not positive"));
      }
    }

    means.push_back(std::move(mean));
    vars.push_back(std::move(var));
  }
  return {};
}

post_expected<double>
independence_value_from_patterns(const FIMLCache& cache,
                                 const std::vector<Eigen::VectorXd>& means,
                                 const std::vector<Eigen::VectorXd>& vars,
                                 const std::vector<Eigen::MatrixXd>& covs,
                                 const std::vector<Eigen::Index>& exo_idx) {
  if (cache.n_total <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML baseline: cache has no observations"));
  }
  if (means.size() != cache.block_p.size() ||
      vars.size() != cache.block_p.size() ||
      covs.size() != cache.block_p.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML baseline: moment block count mismatch"));
  }

  double f = 0.0;
  for (const FIMLPattern& pat : cache.patterns) {
    const auto& mu = means[pat.block];
    const auto& var = vars[pat.block];
    Eigen::MatrixXd Sigma =
        Eigen::MatrixXd::Zero(pat.mean.size(), pat.mean.size());
    for (Eigen::Index j = 0; j < pat.mean.size(); ++j) {
      const Eigen::Index c = pat.observed[static_cast<std::size_t>(j)];
      Sigma(j, j) = var(c);
    }
    for (Eigen::Index a = 0;
         a < static_cast<Eigen::Index>(exo_idx.size()); ++a) {
      const Eigen::Index ca = exo_idx[static_cast<std::size_t>(a)];
      auto ia = std::find(pat.observed.begin(), pat.observed.end(), ca);
      if (ia == pat.observed.end()) continue;
      const Eigen::Index ja = static_cast<Eigen::Index>(
          std::distance(pat.observed.begin(), ia));
      for (Eigen::Index b = a + 1;
           b < static_cast<Eigen::Index>(exo_idx.size()); ++b) {
        const Eigen::Index cb = exo_idx[static_cast<std::size_t>(b)];
        auto ib = std::find(pat.observed.begin(), pat.observed.end(), cb);
        if (ib == pat.observed.end()) continue;
        const Eigen::Index jb = static_cast<Eigen::Index>(
            std::distance(pat.observed.begin(), ib));
        Sigma(ja, jb) = covs[pat.block](ca, cb);
        Sigma(jb, ja) = covs[pat.block](cb, ca);
      }
    }
    Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (Sigma + Sigma.transpose()));
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML baseline: covariance is not positive definite"));
    }
    const double log_det = log_det_from_llt(llt);
    Eigen::VectorXd d(pat.mean.size());
    for (Eigen::Index j = 0; j < pat.mean.size(); ++j) {
      const Eigen::Index c = pat.observed[static_cast<std::size_t>(j)];
      if (!(Sigma(j, j) > 0.0) || !std::isfinite(Sigma(j, j))) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML baseline: non-positive variance"));
      }
      d(j) = pat.mean(j) - mu(c);
    }
    const double quad = (pat.cov + d * d.transpose())
                            .cwiseProduct(llt.solve(
                                Eigen::MatrixXd::Identity(Sigma.rows(),
                                                          Sigma.cols())))
                            .sum();
    if (!std::isfinite(log_det) || !std::isfinite(quad)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML baseline: non-finite pattern contribution"));
    }
    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);
    f += scale * (log_det + quad);
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML baseline objective evaluated to non-finite"));
  }
  return f;
}

}  // namespace

fit_expected<FIMLCache>
FIML::prepare(const RawData& raw) const {
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  FIMLCache cache;
  cache.patterns.reserve(raw.X.size());
  cache.sigma_offsets.resize(raw.X.size());
  cache.mu_offsets.resize(raw.X.size());
  cache.block_p.resize(raw.X.size());

  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    cache.sigma_offsets[b] = sigma_off;
    cache.mu_offsets[b] = mu_off;
    cache.block_p[b] = p;
    sigma_off += vech_len(p);
    mu_off += p;

    std::map<std::vector<Eigen::Index>, std::vector<Eigen::Index>> rows_by_obs;
    for (Eigen::Index r = 0; r < n; ++r) {
      std::vector<Eigen::Index> obs;
      obs.reserve(static_cast<std::size_t>(p));
      if (raw.mask.empty()) {
        for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
      } else {
        for (Eigen::Index c = 0; c < p; ++c) {
          if (raw.mask[b](r, c) != 0) obs.push_back(c);
        }
      }
      if (obs.empty()) {
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            "FIML: block " + std::to_string(b) + " row " +
                std::to_string(r) + " has no observed values"));
      }
      if (!finite_observed_row(X, obs, r)) {
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            "FIML: non-finite observed value in block " + std::to_string(b)));
      }
      rows_by_obs[std::move(obs)].push_back(r);
      ++cache.n_total;
    }

    for (const auto& [obs, rows] : rows_by_obs) {
      const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
      Eigen::VectorXd mean = Eigen::VectorXd::Zero(q);
      for (Eigen::Index r : rows) {
        for (Eigen::Index j = 0; j < q; ++j) {
          mean(j) += X(r, obs[static_cast<std::size_t>(j)]);
        }
      }
      mean /= static_cast<double>(rows.size());

      Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(q, q);
      for (Eigen::Index r : rows) {
        Eigen::VectorXd d(q);
        for (Eigen::Index j = 0; j < q; ++j) {
          d(j) = X(r, obs[static_cast<std::size_t>(j)]) - mean(j);
        }
        cov.noalias() += d * d.transpose();
      }
      cov /= static_cast<double>(rows.size());

      cache.patterns.push_back(FIMLPattern{
          b, obs, static_cast<std::int64_t>(rows.size()),
          std::move(mean), std::move(cov)});
    }
  }

  if (cache.n_total <= 0) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML: no observations"));
  }
  return cache;
}

fit_expected<double>
FIML::value(const RawData& raw, const FIMLCache& cache,
            const model::ImpliedMoments& moments) const {
  Eigen::MatrixXd J0;
  auto vg = value_gradient(raw, cache, moments, J0, J0);
  if (!vg.has_value()) return std::unexpected(vg.error());
  return vg->value;
}

fit_expected<Eigen::VectorXd>
FIML::gradient(const RawData& raw, const FIMLCache& cache,
               const model::ImpliedMoments& moments,
               const Eigen::MatrixXd& J_sigma,
               const Eigen::MatrixXd& J_mu) const {
  auto vg = value_gradient(raw, cache, moments, J_sigma, J_mu);
  if (!vg.has_value()) return std::unexpected(vg.error());
  return std::move(vg->gradient);
}

fit_expected<FIMLValueGradient>
FIML::value_gradient(const RawData&,
                     const FIMLCache& cache,
                     const model::ImpliedMoments& moments,
                     const Eigen::MatrixXd& J_sigma,
                     const Eigen::MatrixXd& J_mu) const {
  if (cache.n_total <= 0) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIMLCache has non-positive n_total"));
  }
  if (moments.sigma.size() != cache.block_p.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML: cache and implied moments have different block counts"));
  }
  if (moments.mu.size() != moments.sigma.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML: mean structure is required"));
  }

  const bool want_gradient = (J_sigma.size() > 0 || J_mu.size() > 0);
  Eigen::Index total_vech = 0;
  Eigen::Index total_p = 0;
  for (std::size_t b = 0; b < moments.sigma.size(); ++b) {
    const Eigen::Index p = moments.sigma[b].rows();
    if (moments.sigma[b].cols() != p ||
        p != cache.block_p[b] ||
        moments.mu[b].size() != p) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: implied moment shape mismatch in block " + std::to_string(b)));
    }
    total_vech += vech_len(p);
    total_p += p;
  }

  if (want_gradient) {
    if (J_sigma.rows() != total_vech) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: J_sigma row count mismatch"));
    }
    if (J_mu.rows() != total_p || J_mu.cols() != J_sigma.cols()) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: J_mu shape mismatch"));
    }
  }

  Eigen::VectorXd w(want_gradient ? total_vech : 0);
  Eigen::VectorXd u(want_gradient ? total_p : 0);
  if (want_gradient) {
    w.setZero();
    u.setZero();
  }

  double f = 0.0;
  for (const FIMLPattern& pat : cache.patterns) {
    const auto& Sigma = moments.sigma[pat.block];
    const auto& Mu = moments.mu[pat.block];
    const Eigen::Index q = static_cast<Eigen::Index>(pat.observed.size());
    const Eigen::MatrixXd Sigma_o = select_square(Sigma, pat.observed);
    const Eigen::VectorXd Mu_o = select_vector(Mu, pat.observed);
    const Eigen::VectorXd d = pat.mean - Mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();

    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "FIML: implied observed-pattern Σ is not positive definite"));
    }
    const auto& L = llt.matrixL();
    double log_det = 0.0;
    for (Eigen::Index i = 0; i < q; ++i) log_det += std::log(L(i, i));
    log_det *= 2.0;

    const Eigen::MatrixXd SigmaInv = llt.solve(Eigen::MatrixXd::Identity(q, q));
    const Eigen::MatrixXd SigmaInv_A = llt.solve(A);
    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);
    f += scale * (log_det + SigmaInv_A.trace());

    if (want_gradient) {
      const Eigen::MatrixXd SigmaInv_A_Inv =
          llt.solve(SigmaInv_A.transpose()).transpose();
      Eigen::MatrixXd G = SigmaInv - SigmaInv_A_Inv;
      G = 0.5 * (G + G.transpose());
      const Eigen::VectorXd z = llt.solve(d);

      const Eigen::Index sigma_off = cache.sigma_offsets[pat.block];
      const Eigen::Index mu_off = cache.mu_offsets[pat.block];
      const Eigen::Index p = cache.block_p[pat.block];
      for (Eigen::Index cj = 0; cj < q; ++cj) {
        const Eigen::Index c = pat.observed[static_cast<std::size_t>(cj)];
        for (Eigen::Index ri = cj; ri < q; ++ri) {
          const Eigen::Index r = pat.observed[static_cast<std::size_t>(ri)];
          const Eigen::Index rr = std::max(r, c);
          const Eigen::Index cc = std::min(r, c);
          const Eigen::Index idx = sigma_off + vech_index(p, rr, cc);
          w(idx) += scale * ((ri == cj) ? G(ri, cj) : 2.0 * G(ri, cj));
        }
      }
      for (Eigen::Index i = 0; i < q; ++i) {
        const Eigen::Index r = pat.observed[static_cast<std::size_t>(i)];
        u(mu_off + r) += -2.0 * scale * z(i);
      }
    }
  }

  if (!std::isfinite(f)) {
    return std::unexpected(make_fit_err(FitError::Kind::NonFiniteObjective,
        "FIML objective evaluated to non-finite"));
  }

  Eigen::VectorXd g;
  if (want_gradient) {
    g = J_sigma.transpose() * w;
    g.noalias() += J_mu.transpose() * u;
  }
  return FIMLValueGradient{f, std::move(g)};
}

fit_expected<SampleStats>
fiml_start_sample_stats(const RawData& raw) {
  if (raw.mask.empty()) {
    auto samp = sample_stats_from_raw(raw);
    if (!samp.has_value()) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "fiml_start_sample_stats: " + samp.error().detail));
    }
    return std::move(*samp);
  }
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  SampleStats out;
  out.S.reserve(raw.X.size());
  out.mean.reserve(raw.X.size());
  out.n_obs.reserve(raw.X.size());

  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    const auto& M = raw.mask[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd count = Eigen::VectorXd::Zero(p);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index c = 0; c < p; ++c) {
        if (M(r, c) == 0) continue;
        if (!std::isfinite(X(r, c))) {
          return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
              "fiml_start_sample_stats: non-finite observed value"));
        }
        mean(c) += X(r, c);
        count(c) += 1.0;
      }
    }
    for (Eigen::Index c = 0; c < p; ++c) {
      if (count(c) <= 0.0) {
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            "fiml_start_sample_stats: column " + std::to_string(c) +
                " has no observed values"));
      }
      mean(c) /= count(c);
    }

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(p, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j; i < p; ++i) {
        double acc = 0.0;
        double nij = 0.0;
        for (Eigen::Index r = 0; r < n; ++r) {
          if (M(r, i) == 0 || M(r, j) == 0) continue;
          acc += (X(r, i) - mean(i)) * (X(r, j) - mean(j));
          nij += 1.0;
        }
        if (nij <= 0.0) {
          return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
              "fiml_start_sample_stats: variable pair has no joint observations"));
        }
        S(i, j) = acc / nij;
        S(j, i) = S(i, j);
      }
    }
    out.S.push_back(std::move(S));
    out.mean.push_back(std::move(mean));
    out.n_obs.push_back(static_cast<std::int64_t>(n));
  }
  return out;
}

fit_expected<void>
validate_fiml_fixed_x_missing_policy(const spec::LatentStructure& pt,
                                     const RawData& raw) {
  if (raw.mask.empty()) return {};
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  const std::vector<Eigen::Index> fixed_x = fixed_x_observed_indices(pt);
  if (fixed_x.empty()) return {};

  for (std::size_t b = 0; b < raw.mask.size(); ++b) {
    const auto& M = raw.mask[b];
    for (Eigen::Index c : fixed_x) {
      if (c < 0 || c >= M.cols()) {
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            "FIML: fixed.x observed variable index out of range"));
      }
      for (Eigen::Index r = 0; r < M.rows(); ++r) {
        if (M(r, c) == 0) {
          return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
              "FIML: fixed.x with missing observed exogenous variables is "
              "not supported yet"));
        }
      }
    }
  }
  return {};
}

post_expected<FIMLExtras>
fiml_extras(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const RawData& raw,
            const Estimates& est,
            FIML discrepancy) {
  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) {
    return std::unexpected(fit_to_post(cache_or.error(), "FIML::prepare"));
  }
  const FIMLCache& cache = *cache_or;

  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) {
    return std::unexpected(fit_to_post(start_samp_or.error(),
        "fiml_start_sample_stats"));
  }
  SampleStats start_samp = std::move(*start_samp_or);

  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }

  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "resolve_fixed_x_from_sample"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "Estimates.theta size " + std::to_string(est.theta.size()) +
            " != evaluator n_free " + std::to_string(ev.n_free())));
  }

  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ev.sigma(theta) failed: " + sm_or.error().detail));
  }
  auto h0_or = discrepancy.value(raw, cache, *sm_or);
  if (!h0_or.has_value()) {
    return std::unexpected(fit_to_post(h0_or.error(), "FIML H0 likelihood"));
  }

  fit_expected<double> h1_or = fiml_h1_value(raw, cache, start_samp);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 likelihood"));
  }

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(con_or.error());
  }
  const int npar = static_cast<int>(con_or->n_alpha);

  FIMLExtras out;
  out.ntotal = cache.n_total;
  out.npar = npar;

  const double c = observed_constant(cache);
  const double N = static_cast<double>(cache.n_total);
  out.logl = -0.5 * (N * (*h0_or) + c);
  out.unrestricted_logl = -0.5 * (N * (*h1_or) + c);
  auto fixed_x_marg_or = fixed_x_saturated_logl(pt, start_samp);
  if (!fixed_x_marg_or.has_value()) return std::unexpected(fixed_x_marg_or.error());
  out.logl -= *fixed_x_marg_or;
  out.unrestricted_logl -= *fixed_x_marg_or;
  out.chi2 = -2.0 * (out.logl - out.unrestricted_logl);
  out.aic = -2.0 * out.logl + 2.0 * static_cast<double>(npar);
  out.bic = -2.0 * out.logl + static_cast<double>(npar) * std::log(N);
  out.bic2 = -2.0 * out.logl + static_cast<double>(npar)
      * std::log((N + 2.0) / 24.0);

  // SRMR — Bentler-type, standardizing the residual of the model-implied
  // moments against the FIML saturated (H1, EM) moments by the H1 SDs. The
  // FIML model carries a mean structure, so the p mean residuals join the
  // p(p+1)/2 covariance residuals.
  if (sm_or->sigma.size() != cache.block_p.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_extras: implied-moment block count does not match the data"));
  }
  double srmr_acc = 0.0;
  for (std::size_t b = 0; b < cache.block_p.size(); ++b) {
    Eigen::VectorXd h1_mu;
    Eigen::MatrixXd h1_sigma;
    if (auto e = h1_moments_block(raw, cache, start_samp, b, h1_mu, h1_sigma);
        !e.has_value()) {
      return std::unexpected(fit_to_post(e.error(), "FIML H1 moments"));
    }
    const Eigen::MatrixXd S = 0.5 * (h1_sigma + h1_sigma.transpose());
    const Eigen::MatrixXd Sigma =
        0.5 * (sm_or->sigma[b] + sm_or->sigma[b].transpose());
    const Eigen::Index p = S.rows();
    if (p == 0) continue;

    double sum_sq = 0.0;
    for (Eigen::Index col = 0; col < p; ++col) {
      const double dc = (S(col, col) - Sigma(col, col)) / S(col, col);
      sum_sq += dc * dc;
      for (Eigen::Index row = col + 1; row < p; ++row) {
        const double rij = (S(row, col) - Sigma(row, col)) /
                           std::sqrt(S(row, row) * S(col, col));
        sum_sq += rij * rij;
      }
    }
    double pstar = static_cast<double>(p) * static_cast<double>(p + 1) / 2.0;
    if (b < sm_or->mu.size() && sm_or->mu[b].size() == p &&
        h1_mu.size() == p) {
      for (Eigen::Index i = 0; i < p; ++i) {
        const double mr = (h1_mu(i) - sm_or->mu[b](i)) / std::sqrt(S(i, i));
        sum_sq += mr * mr;
      }
      pstar += static_cast<double>(p);
    }
    const double srmr_b = (pstar > 0.0) ? std::sqrt(sum_sq / pstar) : 0.0;
    const double n_b = (b < start_samp.n_obs.size())
                           ? static_cast<double>(start_samp.n_obs[b])
                           : 0.0;
    if (N > 0.0) srmr_acc += (n_b / N) * srmr_b;
  }
  out.srmr = srmr_acc;
  return out;
}

post_expected<Eigen::MatrixXd>
fiml_observed_information(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          FIML discrepancy,
                          double h_step) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_observed_information: h_step must be positive"));
  }
  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) {
    return std::unexpected(fit_to_post(cache_or.error(), "FIML::prepare"));
  }
  auto start_or = fiml_start_sample_stats(raw);
  if (!start_or.has_value()) {
    return std::unexpected(fit_to_post(start_or.error(),
        "fiml_start_sample_stats"));
  }
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  // H = ∂²F/∂θ² of the per-observation-averaged deviance F. The FIML
  // log-likelihood is logl = −½(N·F + c), so the observed information is
  // −∂²logl/∂θ² = ½·N·H.
  auto H_or = fiml_observed_hessian_fd(pt, rep, raw, *cache_or, *start_or, est,
                                       discrepancy, h_step);
  if (!H_or.has_value()) return std::unexpected(H_or.error());
  const double N = static_cast<double>(cache_or->n_total);
  return Eigen::MatrixXd(0.5 * N * (*H_or));
}

post_expected<FIMLRobustMLR>
fiml_robust_mlr(spec::LatentStructure pt,
                const model::MatrixRep& rep,
                const RawData& raw,
                const Estimates& est,
                int df,
                double chisq,
                FIML discrepancy,
                double h_step) {
  if (df <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_robust_mlr: robust scaled test requires df > 0"));
  }
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_robust_mlr: h_step must be > 0"));
  }

  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) {
    return std::unexpected(fit_to_post(cache_or.error(), "FIML::prepare"));
  }
  const FIMLCache& cache = *cache_or;

  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) {
    return std::unexpected(fit_to_post(start_samp_or.error(),
        "fiml_start_sample_stats"));
  }
  SampleStats start_samp = std::move(*start_samp_or);

  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "resolve_fixed_x_from_sample"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "Estimates.theta size " + std::to_string(est.theta.size()) +
            " != evaluator n_free " + std::to_string(ev.n_free())));
  }

  auto eval_or = ev.evaluate(est.theta, true, true);
  if (!eval_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::evaluate failed: " + eval_or.error().detail));
  }
  const auto& eval = *eval_or;
  auto scores_or = fiml_casewise_scores(raw, cache, eval.moments,
                                        eval.J_sigma, eval.J_mu);
  if (!scores_or.has_value()) return std::unexpected(scores_or.error());
  Eigen::MatrixXd scores = std::move(*scores_or);

  auto H_or = fiml_observed_hessian_fd(pt, rep, raw, cache, start_samp,
                                       est, discrepancy, h_step);
  if (!H_or.has_value()) return std::unexpected(H_or.error());
  Eigen::MatrixXd H = std::move(*H_or);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  Eigen::MatrixXd K;
  if (con_or->active()) {
    K = con_or->K();
    scores = (scores * K).eval();
    H = (K.transpose() * H * K).eval();
    H = 0.5 * (H + H.transpose()).eval();
  }
  const Eigen::Index q = H.rows();
  const double N = static_cast<double>(cache.n_total);
  const Eigen::MatrixXd meat =
      (scores.transpose() * scores) / N;
  auto Hinv_or = invert_symmetric(H, "fiml_robust_mlr: observed Hessian");
  if (!Hinv_or.has_value()) return std::unexpected(Hinv_or.error());
  const Eigen::MatrixXd& Hinv = *Hinv_or;

  Eigen::MatrixXd vcov_q = (Hinv * meat * Hinv) / N;
  vcov_q = 0.5 * (vcov_q + vcov_q.transpose()).eval();

  FIMLRobustMLR out;
  out.ntotal = cache.n_total;
  out.df = df;
  out.vcov = (K.size() > 0)
      ? Eigen::MatrixXd(K * vcov_q * K.transpose())
      : std::move(vcov_q);
  out.vcov = 0.5 * (out.vcov + out.vcov.transpose()).eval();
  out.se.resize(out.vcov.rows());
  for (Eigen::Index k = 0; k < out.vcov.rows(); ++k) {
    const double d = out.vcov(k, k);
    out.se(k) = (d >= 0.0) ? std::sqrt(d)
                           : std::numeric_limits<double>::quiet_NaN();
  }

  const Eigen::MatrixXd h0 = Hinv * meat;
  out.trace_ugamma_h0 = 0.5 * h0.trace();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      0.5 * (h0 + h0.transpose()), Eigen::EigenvaluesOnly);
  if (es.info() == Eigen::Success) out.eigvals = es.eigenvalues();

  auto h1_trace_or = fiml_saturated_trace_h1(raw, cache, start_samp, h_step);
  if (!h1_trace_or.has_value()) return std::unexpected(h1_trace_or.error());
  out.trace_ugamma_h1 = *h1_trace_or;
  out.trace_ugamma = out.trace_ugamma_h1 - out.trace_ugamma_h0;
  out.scaling_factor = out.trace_ugamma / static_cast<double>(df);
  out.chisq_scaled = (out.scaling_factor > 0.0)
      ? chisq / out.scaling_factor
      : std::numeric_limits<double>::quiet_NaN();
  (void)q;
  return out;
}

namespace {

post_expected<BaselineFit>
fiml_baseline_chi2_impl(const RawData& raw,
                        const std::vector<Eigen::Index>& exo_idx,
                        FIML discrepancy) {
  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) {
    return std::unexpected(fit_to_post(cache_or.error(), "FIML::prepare"));
  }
  const FIMLCache& cache = *cache_or;

  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) {
    return std::unexpected(fit_to_post(start_samp_or.error(),
        "fiml_start_sample_stats"));
  }
  const SampleStats& start_samp = *start_samp_or;

  auto h1_or = fiml_h1_value(raw, cache, start_samp);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 likelihood"));
  }

  std::vector<Eigen::VectorXd> means;
  std::vector<Eigen::VectorXd> vars;
  if (auto ok = independence_moments_from_raw(raw, means, vars); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto baseline_or = independence_value_from_patterns(cache, means, vars,
                                                      start_samp.S, exo_idx);
  if (!baseline_or.has_value()) {
    return std::unexpected(baseline_or.error());
  }

  BaselineFit out;
  out.chi2 = static_cast<double>(cache.n_total) * (*baseline_or - *h1_or);
  if (out.chi2 < 0.0 && out.chi2 > -1e-8) out.chi2 = 0.0;
  for (Eigen::Index p : cache.block_p) {
    out.df += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
    const int px = static_cast<int>(exo_idx.size());
    out.df -= px * (px - 1) / 2;
  }
  return out;
}

}  // namespace

post_expected<BaselineFit>
fiml_baseline_chi2(const RawData& raw,
                   FIML discrepancy) {
  return fiml_baseline_chi2_impl(raw, {}, discrepancy);
}

post_expected<BaselineFit>
fiml_baseline_chi2(const spec::LatentStructure& pt,
                   const RawData& raw,
                   FIML discrepancy) {
  return fiml_baseline_chi2_impl(raw, observed_exogenous_indices(pt),
                                 discrepancy);
}

post_expected<SaturatedMoments>
saturated_em_moments(const RawData& raw, double h_step) {
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(fit_to_post(ok.error(), "saturated_em_moments"));
  }

  FIML discrepancy{};
  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) {
    return std::unexpected(fit_to_post(cache_or.error(),
        "saturated_em_moments: FIML::prepare"));
  }
  const FIMLCache& cache = *cache_or;

  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) {
    return std::unexpected(fit_to_post(start_samp_or.error(),
        "saturated_em_moments: fiml_start_sample_stats"));
  }
  const SampleStats& start_samp = *start_samp_or;

  const std::size_t B = raw.X.size();
  if (B == 0 || cache.block_p.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "saturated_em_moments: empty or inconsistent block layout"));
  }

  // Block-stacked η = (μ_1, vech(Σ_1), μ_2, vech(Σ_2), …); column-major
  // lower-triangle vech matches `fiml_saturated_scores_block` and
  // `robust::casewise_contributions(include_means = true)`.
  std::vector<Eigen::Index> q_b(B);
  std::vector<Eigen::Index> off_b(B + 1, 0);
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = cache.block_p[b];
    q_b[b] = p + detail::vech_len(p);
    off_b[b + 1] = off_b[b] + q_b[b];
  }
  const Eigen::Index Q = off_b.back();

  SaturatedMoments out;
  out.mean.resize(B);
  out.cov.resize(B);
  out.n_obs.resize(B);
  out.H = Eigen::MatrixXd::Zero(Q, Q);
  out.J = Eigen::MatrixXd::Zero(Q, Q);

  for (std::size_t b = 0; b < B; ++b) {
    Eigen::VectorXd mu_b;
    Eigen::MatrixXd Sigma_b;
    if (auto e = h1_moments_block(raw, cache, start_samp, b, mu_b, Sigma_b);
        !e.has_value()) {
      return std::unexpected(fit_to_post(e.error(),
          "saturated_em_moments: H1 EM block " + std::to_string(b)));
    }

    auto scores_or = fiml_saturated_scores_block(raw, b, mu_b, Sigma_b);
    if (!scores_or.has_value()) {
      return std::unexpected(scores_or.error());
    }
    auto H_or = fiml_saturated_hessian_fd_block(raw, b, mu_b, Sigma_b, h_step);
    if (!H_or.has_value()) {
      return std::unexpected(H_or.error());
    }

    // The block helpers report scores as deviance gradients (∂(−2logL)/∂η) and
    // H as the FD Hessian of mean deviance. Convert to log-likelihood scale:
    //   information  H_b = (n_b / 2) · H_dev_mean
    //   score cov    J_b = (1 / 4) · scoresᵀ scores
    const Eigen::MatrixXd& scores_dev = *scores_or;
    const Eigen::MatrixXd& H_dev_mean = *H_or;
    const double n_b = static_cast<double>(raw.X[b].rows());
    const Eigen::Index q = q_b[b];
    const Eigen::Index off = off_b[b];
    out.H.block(off, off, q, q) = (n_b / 2.0) * H_dev_mean;
    out.J.block(off, off, q, q) =
        0.25 * (scores_dev.transpose() * scores_dev);

    out.mean[b] = std::move(mu_b);
    out.cov[b]  = std::move(Sigma_b);
    out.n_obs[b] = static_cast<std::int64_t>(raw.X[b].rows());
  }

  auto Hinv_or = invert_symmetric(out.H,
      "saturated_em_moments: aggregated saturated information");
  if (!Hinv_or.has_value()) {
    return std::unexpected(Hinv_or.error());
  }
  const Eigen::MatrixXd Hinv = *Hinv_or;
  out.acov = Hinv * out.J * Hinv;
  out.acov = (0.5 * (out.acov + out.acov.transpose())).eval();
  return out;
}

fit_expected<Estimates>
fit_fiml(spec::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         const Eigen::VectorXd& x0,
         FIML discrepancy,
         Backend backend,
         optim::OptimOptions opts) {
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(e.error());
  }

  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) return std::unexpected(start_samp_or.error());
  const SampleStats& start_samp = *start_samp_or;

  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp);
      !e.has_value()) {
    return std::unexpected(e.error());
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail, 0, 0.0});
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(FitError{
        FitError::Kind::InvalidStartValues,
        "fit_fiml: x0 size (" + std::to_string(x0.size()) + ") != n_free (" +
            std::to_string(pt.n_free()) + ")", 0, 0.0});
  }

  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) return std::unexpected(cache_or.error());

  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail, 0, 0.0});
  }
  const EqConstraints& con = *con_or;
  const NonlinearEqConstraints nl = build_nl_constraints(pt);

  auto eval_at = [&](const Eigen::VectorXd& x,
                     Eigen::VectorXd& grad) -> double {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    auto vg = discrepancy.value_gradient(raw, *cache_or, eval->moments,
                                         eval->J_sigma, eval->J_mu);
    if (!vg.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    grad = std::move(vg->gradient);
    return vg->value;
  };

  auto run_fiml_scalar = [&](const optim::ScalarProblem& prob,
                             const Eigen::VectorXd& start)
      -> fit_expected<optim::OptimResult> {
    switch (backend) {
      case Backend::NloptSlsqp:
        return optim::nlopt_slsqp(prob, start, {}, opts);
      case Backend::NloptLbfgs:
        return optim::nlopt_lbfgs(prob, start, {}, opts);
      case Backend::Ipopt:
#ifdef MAGMAAN_WITH_IPOPT
        return optim::ipopt(prob, start, {}, opts);
#else
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            "fit_fiml: IPOPT backend requested but MAGMAAN_WITH_IPOPT is off"));
#endif
      default:
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            "fit_fiml: requested optimizer backend is not supported; use "
            "nlopt-lbfgs, nlopt-slsqp, or ipopt"));
    }
  };

  auto run_fiml_constrained = [&](const optim::ScalarProblem& prob,
                                  const optim::ConstraintFn& h,
                                  const optim::ConstraintJacFn& J_h,
                                  std::int32_t m,
                                  const Eigen::VectorXd& start)
      -> fit_expected<optim::OptimResult> {
    optim::ConstrainedScalarProblem cprob;
    cprob.objective = prob;
    cprob.h = h;
    cprob.J_h = J_h;
    cprob.n_constraint = m;
    cprob.constraint_lower = Eigen::VectorXd::Zero(m);
    cprob.constraint_upper = Eigen::VectorXd::Zero(m);
    if (backend == Backend::NloptSlsqp) {
      return optim::nlopt_slsqp_constrained(cprob, start, {}, opts);
    }
#ifdef MAGMAAN_WITH_IPOPT
    if (backend == Backend::Ipopt) {
      return optim::ipopt_constrained(cprob, start, {}, opts);
    }
#else
    if (backend == Backend::Ipopt) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "fit_fiml: nonlinear equality constraints require optimizer "
          "\"nlopt-slsqp\" or an IPOPT-enabled build with optimizer "
          "\"ipopt\""));
    }
#endif
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "fit_fiml: nonlinear equality constraints require optimizer "
        "\"nlopt-slsqp\" or \"ipopt\""));
  };

  if (!con.active()) {
    optim::ScalarProblem prob;
    prob.f       = eval_at;
    prob.n_param = x0.size();
    prob.expand  = [](const Eigen::VectorXd& x) { return x; };
    fit_expected<optim::OptimResult> out_or;
    if (nl.active()) {
      auto h_fn   = [&nl](const Eigen::VectorXd& th) { return nl.h(th); };
      auto jac_fn = [&nl](const Eigen::VectorXd& th) { return nl.jacobian(th); };
      out_or = run_fiml_constrained(prob, h_fn, jac_fn, nl.m(), x0);
    } else {
      out_or = run_fiml_scalar(prob, x0);
    }
    if (!out_or.has_value()) return std::unexpected(out_or.error());
    return Estimates{prob.expand(out_or->x), out_or->fmin, out_or->iterations,
                     out_or->f_evals, out_or->g_evals, out_or->status,
                     out_or->grad_inf_norm, std::move(out_or->audit)};
  }

  if (con.n_alpha == 0) {
    Eigen::VectorXd theta = con.expand(Eigen::VectorXd(0));
    if (nl.active()) {
      const Eigen::VectorXd h = nl.h(theta);
      const double viol = h.size() > 0 ? h.cwiseAbs().maxCoeff() : 0.0;
      if (viol > 1e-6 || !std::isfinite(viol)) {
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            "fit_fiml: nonlinear equality constraints are infeasible at the "
            "fully linear-constrained point; violation " +
                std::to_string(viol)));
      }
    }
    Eigen::VectorXd scratch(theta.size());
    const double f = eval_at(theta, scratch);
    return Estimates{std::move(theta), f, 0};
  }

  optim::ScalarProblem prob_a;
  prob_a.n_param = con.n_alpha;
  prob_a.expand  = [&con](const Eigen::VectorXd& a) { return con.expand(a); };
  prob_a.f = [&con, &eval_at](const Eigen::VectorXd& a,
                              Eigen::VectorXd& grad_a) -> double {
    const Eigen::VectorXd x = con.expand(a);
    Eigen::VectorXd grad_x(x.size());
    const double v = eval_at(x, grad_x);
    grad_a = con.reduce_gradient(grad_x);
    return v;
  };
  const Eigen::VectorXd alpha0 = con.contract(x0);
  fit_expected<optim::OptimResult> out_or;
  if (nl.active()) {
    auto h_a = [&nl, &con](const Eigen::VectorXd& a) {
      return nl.h(con.expand(a));
    };
    auto jac_a = [&nl, &con](const Eigen::VectorXd& a) {
      return Eigen::MatrixXd(nl.jacobian(con.expand(a)) * con.K());
    };
    out_or = run_fiml_constrained(prob_a, h_a, jac_a, nl.m(), alpha0);
  } else {
    out_or = run_fiml_scalar(prob_a, alpha0);
  }
  if (!out_or.has_value()) return std::unexpected(out_or.error());
  return Estimates{con.expand(out_or->x), out_or->fmin, out_or->iterations,
                   out_or->f_evals, out_or->g_evals, out_or->status,
                   out_or->grad_inf_norm, std::move(out_or->audit)};
}

}  // namespace magmaan::estimate::fiml
