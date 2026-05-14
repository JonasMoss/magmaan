#include "magmaan/fit/fiml.hpp"

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

#include "detail_vech.hpp"

namespace magmaan::fit {

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
fixed_x_observed_indices(const partable::LatentStructure& pt) {
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
fiml_observed_hessian_fd(partable::LatentStructure pt,
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
                                 const std::vector<Eigen::VectorXd>& vars) {
  if (cache.n_total <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML baseline: cache has no observations"));
  }
  if (means.size() != cache.block_p.size() || vars.size() != cache.block_p.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML baseline: moment block count mismatch"));
  }

  double f = 0.0;
  for (const FIMLPattern& pat : cache.patterns) {
    const auto& mu = means[pat.block];
    const auto& var = vars[pat.block];
    double log_det = 0.0;
    double quad = 0.0;
    for (Eigen::Index j = 0;
         j < static_cast<Eigen::Index>(pat.observed.size()); ++j) {
      const Eigen::Index c = pat.observed[static_cast<std::size_t>(j)];
      const double v = var(c);
      if (!(v > 0.0) || !std::isfinite(v)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML baseline: non-positive variance"));
      }
      const double d = pat.mean(j) - mu(c);
      log_det += std::log(v);
      quad += (pat.cov(j, j) + d * d) / v;
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
validate_fiml_fixed_x_missing_policy(const partable::LatentStructure& pt,
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
fiml_extras(partable::LatentStructure pt,
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
  out.chi2 = -2.0 * (out.logl - out.unrestricted_logl);
  out.aic = -2.0 * out.logl + 2.0 * static_cast<double>(npar);
  out.bic = -2.0 * out.logl + static_cast<double>(npar) * std::log(N);
  out.bic2 = -2.0 * out.logl + static_cast<double>(npar)
      * std::log((N + 2.0) / 24.0);
  return out;
}

post_expected<FIMLRobustMLR>
fiml_robust_mlr(partable::LatentStructure pt,
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

post_expected<BaselineFit>
fiml_baseline_chi2(const RawData& raw,
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

  auto baseline_or = independence_value_from_patterns(cache, means, vars);
  if (!baseline_or.has_value()) {
    return std::unexpected(baseline_or.error());
  }

  BaselineFit out;
  out.chi2 = static_cast<double>(cache.n_total) * (*baseline_or - *h1_or);
  if (out.chi2 < 0.0 && out.chi2 > -1e-8) out.chi2 = 0.0;
  for (Eigen::Index p : cache.block_p) {
    out.df += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
  }
  return out;
}

}  // namespace magmaan::fit
