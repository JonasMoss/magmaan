#include "magmaan/fit/fiml.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
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

using detail::vech_index;
using detail::vech_len;

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

}  // namespace magmaan::fit
