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

  if (!raw.mask.empty() && !fixed_x_observed_indices(pt).empty()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_extras: fixed.x with missing observed exogenous variables is "
        "not supported yet"));
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

  fit_expected<double> h1_or =
      raw.mask.empty() ? h1_complete_data_value(start_samp)
                       : h1_missing_data_value(cache, start_samp);
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

}  // namespace magmaan::fit
