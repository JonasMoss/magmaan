#include "magmaan/estimate/fiml.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/weighted_inference.hpp"

#include "detail_second_order.hpp"
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

struct IndexVectorHash {
  std::size_t operator()(const std::vector<Eigen::Index>& v) const noexcept {
    std::size_t seed = 0;
    for (Eigen::Index idx : v) {
      seed ^= std::hash<Eigen::Index>{}(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

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
using detail::vech_lower;
using detail::vech_unpack;

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

// One EM sweep over the block's patterns: returns the H1 objective evaluated
// at the *input* (mu, Sigma) — sharing each pattern's Cholesky with the
// E-step — and writes the M-step update into (mu_next, Sigma_next).
fit_expected<double>
h1_em_step_block(const FIMLCache& cache,
                 std::size_t block,
                 const Eigen::VectorXd& mu,
                 const Eigen::MatrixXd& Sigma,
                 Eigen::VectorXd& mu_next,
                 Eigen::MatrixXd& Sigma_next) {
  const Eigen::Index p = cache.block_p[block];
  Eigen::VectorXd sum_x = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd sum_xx = Eigen::MatrixXd::Zero(p, p);
  std::int64_t n_block = 0;
  double f = 0.0;

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

    const Eigen::MatrixXd Sigma_oo = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_oo);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "FIML H1: saturated observed-pattern covariance is not positive definite"));
    }
    const Eigen::VectorXd mu_o = select_vector(mu, obs);
    const Eigen::VectorXd d = pat.mean - mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();
    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);

    if (m > 0) {
      const Eigen::MatrixXd Sigma_mo = select_rect(Sigma, miss, obs);
      const Eigen::MatrixXd Sigma_om = Sigma_mo.transpose();
      const Eigen::MatrixXd Sigma_mm = select_square(Sigma, miss);
      const Eigen::MatrixXd Sigma_oo_inv =
          llt.solve(Eigen::MatrixXd::Identity(q, q));
      f += scale * (log_det_from_llt(llt) + Sigma_oo_inv.cwiseProduct(A).sum());
      const Eigen::MatrixXd B = Sigma_mo * Sigma_oo_inv;
      const Eigen::MatrixXd C = Sigma_mm - B * Sigma_om;

      const Eigen::VectorXd mu_m = select_vector(mu, miss);
      const Eigen::VectorXd mbar = mu_m + B * d;
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
    } else {
      f += scale * (log_det_from_llt(llt) + llt.solve(A).trace());
    }

    sum_x.noalias() += static_cast<double>(pat.n_obs) * avg_x;
    sum_xx.noalias() += static_cast<double>(pat.n_obs) * avg_xx;
  }

  if (n_block <= 0) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: block has no observations"));
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_fit_err(FitError::Kind::NonFiniteObjective,
        "FIML H1 objective evaluated to non-finite"));
  }
  mu_next = sum_x / static_cast<double>(n_block);
  Sigma_next = sum_xx / static_cast<double>(n_block) -
               mu_next * mu_next.transpose();
  Sigma_next = 0.5 * (Sigma_next + Sigma_next.transpose());
  return f;
}

// Runs the EM iteration for one block. On return (mu, Sigma) hold the
// converged moments and the returned value is the H1 objective at exactly
// those moments — the same value/update ordering as the previous two-pass
// loop, with one Cholesky per pattern per iteration instead of two.
fit_expected<double>
h1_em_iterate_block(const FIMLCache& cache,
                    std::size_t block,
                    Eigen::VectorXd& mu,
                    Eigen::MatrixXd& Sigma) {
  Eigen::VectorXd mu_next;
  Eigen::MatrixXd Sigma_next;
  double prev = std::numeric_limits<double>::infinity();
  double cur = std::numeric_limits<double>::infinity();
  for (int iter = 0; iter < 10000; ++iter) {
    auto step = h1_em_step_block(cache, block, mu, Sigma, mu_next, Sigma_next);
    if (!step.has_value()) return std::unexpected(step.error());
    cur = *step;
    if (std::isfinite(prev) &&
        std::abs(prev - cur) <= 1e-11 * (1.0 + std::abs(cur))) {
      return cur;
    }
    prev = cur;
    mu.swap(mu_next);
    Sigma.swap(Sigma_next);
  }
  // Iteration cap reached: (mu, Sigma) carry one more update than `cur`
  // was evaluated at, so re-evaluate to keep value/moment consistency.
  auto final_val = h1_block_value_from_moments(cache, block, mu, Sigma);
  if (!final_val.has_value()) return std::unexpected(final_val.error());
  return *final_val;
}

// Light consistency check before indexing a caller-supplied FIMLH1 against a
// caller-supplied pack: the pack overloads trust the caller to have built both
// from the same raw data, but a block-count mismatch is always a usage bug.
fit_expected<void>
validate_h1_blocks(const FIMLCache& cache, const FIMLH1& h1) {
  if (h1.mu.size() != cache.block_p.size() ||
      h1.sigma.size() != cache.block_p.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1: moments and pattern cache have different block counts"));
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

// Per-observed-pattern moment factorization shared by all rows carrying that
// pattern: rows in the same pattern see the same Σ_oo, so the Cholesky and
// explicit inverse are computed once per pattern, not once per row.
struct PatternMoments {
  Eigen::MatrixXd SigmaInv;
  Eigen::VectorXd Mu_o;
};

using PatternMomentMap =
    std::unordered_map<std::vector<Eigen::Index>, PatternMoments,
                       IndexVectorHash>;

struct WeightedFIMLPattern {
  std::size_t block = 0;
  std::vector<Eigen::Index> observed;
  double n_weight = 0.0;
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
};

struct WeightedPatternBlock {
  std::size_t block = 0;
  Eigen::Index p = 0;
  double n_weight = 0.0;
  std::vector<WeightedFIMLPattern> patterns;
  Eigen::VectorXd row_weight;
};

post_expected<const PatternMoments*>
pattern_moments_for(PatternMomentMap& cache_map,
                    const std::vector<Eigen::Index>& obs,
                    const Eigen::VectorXd& Mu,
                    const Eigen::MatrixXd& Sigma,
                    const char* what) {
  auto it = cache_map.find(obs);
  if (it == cache_map.end()) {
    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(what) + " is not positive definite"));
    }
    PatternMoments pm;
    pm.SigmaInv = llt.solve(Eigen::MatrixXd::Identity(q, q));
    pm.SigmaInv = 0.5 * (pm.SigmaInv + pm.SigmaInv.transpose()).eval();
    pm.Mu_o = select_vector(Mu, obs);
    it = cache_map.emplace(obs, std::move(pm)).first;
  }
  return &it->second;
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

  Eigen::VectorXd w(J_sigma.rows());
  Eigen::VectorXd u(J_mu.rows());
  Eigen::Index row_out = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    const Eigen::Index p = X.cols();
    const Eigen::Index sigma_off = cache.sigma_offsets[b];
    const Eigen::Index mu_off = cache.mu_offsets[b];
    PatternMomentMap bypat;
    std::vector<Eigen::Index> obs;
    for (Eigen::Index r = 0; r < X.rows(); ++r) {
      obs.clear();
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
      auto pm_or = pattern_moments_for(
          bypat, obs, moments.mu[b], moments.sigma[b],
          "FIML robust: implied observed-pattern Sigma");
      if (!pm_or.has_value()) return std::unexpected(pm_or.error());
      const PatternMoments& pm = **pm_or;

      const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
      Eigen::VectorXd d(q);
      for (Eigen::Index j = 0; j < q; ++j) {
        d(j) = X(r, obs[static_cast<std::size_t>(j)]) - pm.Mu_o(j);
      }
      const Eigen::VectorXd z = pm.SigmaInv * d;
      const Eigen::MatrixXd G = pm.SigmaInv - z * z.transpose();

      w.setZero();
      u.setZero();
      for (Eigen::Index cj = 0; cj < q; ++cj) {
        const Eigen::Index c = obs[static_cast<std::size_t>(cj)];
        for (Eigen::Index ri = cj; ri < q; ++ri) {
          const Eigen::Index rr0 = obs[static_cast<std::size_t>(ri)];
          const Eigen::Index rr = std::max(rr0, c);
          const Eigen::Index cc = std::min(rr0, c);
          const Eigen::Index idx = sigma_off + vech_index(p, rr, cc);
          w(idx) += (ri == cj) ? G(ri, cj) : 2.0 * G(ri, cj);
        }
      }
      for (Eigen::Index i = 0; i < q; ++i) {
        const Eigen::Index rr = obs[static_cast<std::size_t>(i)];
        u(mu_off + rr) += -2.0 * z(i);
      }
      scores.row(row_out).noalias() = w.transpose() * J_sigma;
      scores.row(row_out).noalias() += u.transpose() * J_mu;
      ++row_out;
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

struct SigmaBasis {
  Eigen::Index full_index = 0;
  Eigen::Index a = 0;  // local observed row, a >= b
  Eigen::Index b = 0;  // local observed column
};

// trace(D_l · X · D_k · Y) where D is the symmetric vech basis matrix of the
// (a, b) entry (≤2 nonzeros), expanded entry-wise in O(1).
double basis_trace_xy(const Eigen::MatrixXd& X,
                      const Eigen::MatrixXd& Y,
                      const SigmaBasis& l,
                      const SigmaBasis& k) {
  const Eigen::Index lu[2] = {l.a, l.b};
  const Eigen::Index lv[2] = {l.b, l.a};
  const Eigen::Index ku[2] = {k.a, k.b};
  const Eigen::Index kv[2] = {k.b, k.a};
  const int ln = (l.a == l.b) ? 1 : 2;
  const int kn = (k.a == k.b) ? 1 : 2;

  double out = 0.0;
  for (int i = 0; i < ln; ++i) {
    for (int j = 0; j < kn; ++j) {
      out += X(lv[i], ku[j]) * Y(kv[j], lu[i]);
    }
  }
  return out;
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
  PatternMomentMap bypat;
  std::vector<Eigen::Index> obs;
  for (Eigen::Index r = 0; r < n; ++r) {
    obs.clear();
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
    auto pm_or = pattern_moments_for(bypat, obs, mu, Sigma,
                                     "FIML robust H1: Sigma_oo");
    if (!pm_or.has_value()) return std::unexpected(pm_or.error());
    const PatternMoments& pm = **pm_or;

    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    Eigen::VectorXd d(q);
    for (Eigen::Index j = 0; j < q; ++j) {
      d(j) = X(r, obs[static_cast<std::size_t>(j)]) - pm.Mu_o(j);
    }
    const Eigen::VectorXd z = pm.SigmaInv * d;
    const Eigen::MatrixXd G = pm.SigmaInv - z * z.transpose();
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

// Analytic observed Hessian of the mean saturated deviance, aggregated per
// observed-value pattern instead of per row: with z_r = Σ_oo⁻¹(x_r − μ_o),
// every row contribution is at most quadratic in z_r, so a pattern enters
// only through n_r, Σ_r z_r = A·Σ_r d_r and Σ_r z_r z_rᵀ = A(Σ_r d_r d_rᵀ)A —
// all closed-form in the pattern's stored mean and covariance.
post_expected<Eigen::MatrixXd>
fiml_saturated_hessian_analytic_block(const FIMLCache& cache,
                                      std::size_t block,
                                      const Eigen::VectorXd& mu,
                                      const Eigen::MatrixXd& Sigma) {
  if (block >= cache.block_p.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1 analytic: block index out of range"));
  }
  const Eigen::Index p = cache.block_p[block];
  if (mu.size() != p || Sigma.rows() != p || Sigma.cols() != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1 analytic: saturated moments do not match raw block"));
  }

  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index qfull = p + pstar;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(qfull, qfull);
  std::int64_t n_block = 0;

  for (const FIMLPattern& pat : cache.patterns) {
    if (pat.block != block) continue;
    n_block += pat.n_obs;
    const auto& obs = pat.observed;
    const Eigen::Index qobs = static_cast<Eigen::Index>(obs.size());
    const double nr = static_cast<double>(pat.n_obs);

    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML robust H1 analytic: Sigma_oo is not positive definite"));
    }
    Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(qobs, qobs));
    A = 0.5 * (A + A.transpose()).eval();
    const Eigen::VectorXd Mu_o = select_vector(mu, obs);
    const Eigen::VectorXd dbar = pat.mean - Mu_o;

    const Eigen::VectorXd Zsum = nr * (A * dbar);
    const Eigen::MatrixXd sum_ddT = nr * (pat.cov + dbar * dbar.transpose());
    const Eigen::MatrixXd M = A * sum_ddT * A;

    for (Eigen::Index j = 0; j < qobs; ++j) {
      const Eigen::Index fj = obs[static_cast<std::size_t>(j)];
      for (Eigen::Index i = 0; i < qobs; ++i) {
        const Eigen::Index fi = obs[static_cast<std::size_t>(i)];
        H(fi, fj) += 2.0 * nr * A(i, j);
      }
    }

    std::vector<SigmaBasis> basis;
    basis.reserve(static_cast<std::size_t>(qobs * (qobs + 1) / 2));
    for (Eigen::Index cj = 0; cj < qobs; ++cj) {
      const Eigen::Index full_c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = cj; ri < qobs; ++ri) {
        const Eigen::Index full_r0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index full_r = std::max(full_r0, full_c);
        const Eigen::Index full_l = std::min(full_r0, full_c);

        SigmaBasis e;
        e.full_index = p + vech_index(p, full_r, full_l);
        e.a = ri;
        e.b = cj;

        // Σ_r A·D_e·z_r = A·D_e·Zsum, with D_e·Zsum having ≤2 nonzeros.
        Eigen::VectorXd ADz;
        if (ri == cj) {
          ADz = A.col(ri) * Zsum(ri);
        } else {
          ADz = A.col(ri) * Zsum(cj) + A.col(cj) * Zsum(ri);
        }
        for (Eigen::Index i = 0; i < qobs; ++i) {
          const Eigen::Index full_i = obs[static_cast<std::size_t>(i)];
          const double h = 2.0 * ADz(i);
          H(full_i, e.full_index) += h;
          H(e.full_index, full_i) += h;
        }
        basis.push_back(e);
      }
    }

    for (std::size_t kk = 0; kk < basis.size(); ++kk) {
      const SigmaBasis& k = basis[kk];
      for (std::size_t ll = 0; ll <= kk; ++ll) {
        const SigmaBasis& l = basis[ll];
        // Σ_r [−tr(D_l A D_k A) + z_rᵀD_l A D_k z_r + z_rᵀD_k A D_l z_r]
        const double h = -nr * basis_trace_xy(A, A, l, k) +
                         basis_trace_xy(A, M, l, k) +
                         basis_trace_xy(A, M, k, l);
        H(k.full_index, l.full_index) += h;
        if (kk != ll) H(l.full_index, k.full_index) += h;
      }
    }
  }

  if (n_block <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1 analytic: block has no observations"));
  }
  H /= static_cast<double>(n_block);
  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

post_expected<WeightedPatternBlock>
weighted_pattern_block(const RawData& raw,
                       std::size_t block,
                       Eigen::Index perturb_row,
                       double perturb_delta) {
  if (block >= raw.X.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: block index out of range"));
  }
  if (!raw.mask.empty() && block >= raw.mask.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: mask block index out of range"));
  }
  if (!(perturb_delta > -1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: case perturbation gives non-positive weight"));
  }

  const Eigen::MatrixXd& X = raw.X[block];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (perturb_row < 0 || perturb_row >= n) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: row index out of range"));
  }

  struct Accum {
    double w = 0.0;
    Eigen::VectorXd sum;
    Eigen::MatrixXd sumsq;
  };
  std::unordered_map<std::vector<Eigen::Index>, Accum, IndexVectorHash> acc;
  Eigen::VectorXd row_weight(n);
  std::vector<Eigen::Index> obs;
  for (Eigen::Index r = 0; r < n; ++r) {
    obs.clear();
    obs.reserve(static_cast<std::size_t>(p));
    if (raw.mask.empty()) {
      for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
    } else {
      for (Eigen::Index c = 0; c < p; ++c) {
        if (raw.mask[block](r, c) != 0) obs.push_back(c);
      }
    }
    if (obs.empty() || !finite_observed_row(X, obs, r)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: invalid observed row"));
    }

    const double w = 1.0 + ((r == perturb_row) ? perturb_delta : 0.0);
    if (!(w > 0.0) || !std::isfinite(w)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: non-positive row weight"));
    }
    row_weight(r) = w;

    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    auto [it, inserted] = acc.try_emplace(obs);
    if (inserted) {
      it->second.sum = Eigen::VectorXd::Zero(q);
      it->second.sumsq = Eigen::MatrixXd::Zero(q, q);
    }
    Eigen::VectorXd x(q);
    for (Eigen::Index j = 0; j < q; ++j) {
      x(j) = X(r, obs[static_cast<std::size_t>(j)]);
    }
    it->second.w += w;
    it->second.sum.noalias() += w * x;
    it->second.sumsq.noalias() += w * (x * x.transpose());
  }

  WeightedPatternBlock out;
  out.block = block;
  out.p = p;
  out.row_weight = std::move(row_weight);
  out.n_weight = out.row_weight.sum();
  if (!(out.n_weight > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: empty weighted block"));
  }
  out.patterns.reserve(acc.size());
  for (auto& [key, a] : acc) {
    if (!(a.w > 0.0)) continue;
    Eigen::VectorXd mean = a.sum / a.w;
    Eigen::MatrixXd cov = a.sumsq / a.w - mean * mean.transpose();
    cov = 0.5 * (cov + cov.transpose()).eval();
    out.patterns.push_back(WeightedFIMLPattern{
        block, std::move(key), a.w, std::move(mean), std::move(cov)});
  }
  return out;
}

post_expected<SampleStats>
weighted_start_sample_stats_block(const RawData& raw,
                                  const WeightedPatternBlock& wp) {
  const Eigen::MatrixXd& X = raw.X[wp.block];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (wp.row_weight.size() != n || wp.p != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: weighted start shape mismatch"));
  }

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd count = Eigen::VectorXd::Zero(p);
  for (Eigen::Index r = 0; r < n; ++r) {
    const double w = wp.row_weight(r);
    for (Eigen::Index c = 0; c < p; ++c) {
      const bool observed = raw.mask.empty() || raw.mask[wp.block](r, c) != 0;
      if (!observed) continue;
      mean(c) += w * X(r, c);
      count(c) += w;
    }
  }
  for (Eigen::Index c = 0; c < p; ++c) {
    if (!(count(c) > 0.0)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: column has no weighted observed values"));
    }
    mean(c) /= count(c);
  }

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(p, p);
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      double acc = 0.0;
      double nij = 0.0;
      for (Eigen::Index r = 0; r < n; ++r) {
        const bool oi = raw.mask.empty() || raw.mask[wp.block](r, i) != 0;
        const bool oj = raw.mask.empty() || raw.mask[wp.block](r, j) != 0;
        if (!oi || !oj) continue;
        const double w = wp.row_weight(r);
        acc += w * (X(r, i) - mean(i)) * (X(r, j) - mean(j));
        nij += w;
      }
      if (!(nij > 0.0)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "ML2S Gamma influence: variable pair has no weighted joint "
            "observations"));
      }
      S(i, j) = acc / nij;
      S(j, i) = S(i, j);
    }
  }

  SampleStats out;
  out.mean.push_back(std::move(mean));
  out.S.push_back(std::move(S));
  out.n_obs.push_back(static_cast<std::int64_t>(n));
  return out;
}

fit_expected<double>
weighted_h1_em_step_block(const WeightedPatternBlock& wp,
                          const Eigen::VectorXd& mu,
                          const Eigen::MatrixXd& Sigma,
                          Eigen::VectorXd& mu_next,
                          Eigen::MatrixXd& Sigma_next) {
  const Eigen::Index p = wp.p;
  Eigen::VectorXd sum_x = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd sum_xx = Eigen::MatrixXd::Zero(p, p);
  double n_block = 0.0;
  double f = 0.0;

  for (const WeightedFIMLPattern& pat : wp.patterns) {
    n_block += pat.n_weight;
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

    const Eigen::MatrixXd Sigma_oo = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_oo);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ML2S Gamma influence: saturated observed-pattern covariance is "
          "not positive definite"));
    }
    const Eigen::VectorXd mu_o = select_vector(mu, obs);
    const Eigen::VectorXd d = pat.mean - mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();
    const double scale = pat.n_weight / wp.n_weight;

    if (m > 0) {
      const Eigen::MatrixXd Sigma_mo = select_rect(Sigma, miss, obs);
      const Eigen::MatrixXd Sigma_om = Sigma_mo.transpose();
      const Eigen::MatrixXd Sigma_mm = select_square(Sigma, miss);
      const Eigen::MatrixXd Sigma_oo_inv =
          llt.solve(Eigen::MatrixXd::Identity(q, q));
      f += scale * (log_det_from_llt(llt) + Sigma_oo_inv.cwiseProduct(A).sum());
      const Eigen::MatrixXd B = Sigma_mo * Sigma_oo_inv;
      const Eigen::MatrixXd C = Sigma_mm - B * Sigma_om;

      const Eigen::VectorXd mu_m = select_vector(mu, miss);
      const Eigen::VectorXd mbar = mu_m + B * d;
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
    } else {
      f += scale * (log_det_from_llt(llt) + llt.solve(A).trace());
    }

    sum_x.noalias() += pat.n_weight * avg_x;
    sum_xx.noalias() += pat.n_weight * avg_xx;
  }

  if (!(n_block > 0.0) || !(wp.n_weight > 0.0)) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "ML2S Gamma influence: weighted block has no observations"));
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_fit_err(FitError::Kind::NonFiniteObjective,
        "ML2S Gamma influence: H1 objective evaluated to non-finite"));
  }
  mu_next = sum_x / n_block;
  Sigma_next = sum_xx / n_block - mu_next * mu_next.transpose();
  Sigma_next = 0.5 * (Sigma_next + Sigma_next.transpose());
  return f;
}

fit_expected<void>
weighted_h1_em_iterate_block(const WeightedPatternBlock& wp,
                             Eigen::VectorXd& mu,
                             Eigen::MatrixXd& Sigma) {
  Eigen::VectorXd mu_next;
  Eigen::MatrixXd Sigma_next;
  double prev = std::numeric_limits<double>::infinity();
  for (int iter = 0; iter < 10000; ++iter) {
    auto step = weighted_h1_em_step_block(wp, mu, Sigma, mu_next, Sigma_next);
    if (!step.has_value()) return std::unexpected(step.error());
    const double cur = *step;
    if (std::isfinite(prev) &&
        std::abs(prev - cur) <= 1e-11 * (1.0 + std::abs(cur))) {
      return {};
    }
    prev = cur;
    mu.swap(mu_next);
    Sigma.swap(Sigma_next);
  }
  return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
      "ML2S Gamma influence: weighted H1 EM did not converge"));
}

post_expected<Eigen::MatrixXd>
weighted_saturated_hessian_analytic_block(const WeightedPatternBlock& wp,
                                          const Eigen::VectorXd& mu,
                                          const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = wp.p;
  if (mu.size() != p || Sigma.rows() != p || Sigma.cols() != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: saturated moments do not match raw block"));
  }

  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index qfull = p + pstar;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(qfull, qfull);
  double n_block = 0.0;

  for (const WeightedFIMLPattern& pat : wp.patterns) {
    n_block += pat.n_weight;
    const auto& obs = pat.observed;
    const Eigen::Index qobs = static_cast<Eigen::Index>(obs.size());
    const double nr = pat.n_weight;

    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: Sigma_oo is not positive definite"));
    }
    Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(qobs, qobs));
    A = 0.5 * (A + A.transpose()).eval();
    const Eigen::VectorXd Mu_o = select_vector(mu, obs);
    const Eigen::VectorXd dbar = pat.mean - Mu_o;

    const Eigen::VectorXd Zsum = nr * (A * dbar);
    const Eigen::MatrixXd sum_ddT = nr * (pat.cov + dbar * dbar.transpose());
    const Eigen::MatrixXd M = A * sum_ddT * A;

    for (Eigen::Index j = 0; j < qobs; ++j) {
      const Eigen::Index fj = obs[static_cast<std::size_t>(j)];
      for (Eigen::Index i = 0; i < qobs; ++i) {
        const Eigen::Index fi = obs[static_cast<std::size_t>(i)];
        H(fi, fj) += 2.0 * nr * A(i, j);
      }
    }

    std::vector<SigmaBasis> basis;
    basis.reserve(static_cast<std::size_t>(qobs * (qobs + 1) / 2));
    for (Eigen::Index cj = 0; cj < qobs; ++cj) {
      const Eigen::Index full_c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = cj; ri < qobs; ++ri) {
        const Eigen::Index full_r0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index full_r = std::max(full_r0, full_c);
        const Eigen::Index full_l = std::min(full_r0, full_c);

        SigmaBasis e;
        e.full_index = p + vech_index(p, full_r, full_l);
        e.a = ri;
        e.b = cj;

        Eigen::VectorXd ADz;
        if (ri == cj) {
          ADz = A.col(ri) * Zsum(ri);
        } else {
          ADz = A.col(ri) * Zsum(cj) + A.col(cj) * Zsum(ri);
        }
        for (Eigen::Index i = 0; i < qobs; ++i) {
          const Eigen::Index full_i = obs[static_cast<std::size_t>(i)];
          const double h = 2.0 * ADz(i);
          H(full_i, e.full_index) += h;
          H(e.full_index, full_i) += h;
        }
        basis.push_back(e);
      }
    }

    for (std::size_t kk = 0; kk < basis.size(); ++kk) {
      const SigmaBasis& k = basis[kk];
      for (std::size_t ll = 0; ll <= kk; ++ll) {
        const SigmaBasis& l = basis[ll];
        const double h = -nr * basis_trace_xy(A, A, l, k) +
                         basis_trace_xy(A, M, l, k) +
                         basis_trace_xy(A, M, k, l);
        H(k.full_index, l.full_index) += h;
        if (kk != ll) H(l.full_index, k.full_index) += h;
      }
    }
  }

  if (!(n_block > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: weighted block has no observations"));
  }
  H /= n_block;
  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

post_expected<Eigen::MatrixXd>
weighted_two_stage_gamma_block(const RawData& raw,
                               std::size_t block,
                               Eigen::Index perturb_row,
                               double perturb_delta) {
  auto wp_or = weighted_pattern_block(raw, block, perturb_row, perturb_delta);
  if (!wp_or.has_value()) return std::unexpected(wp_or.error());
  const WeightedPatternBlock& wp = *wp_or;

  auto start_or = weighted_start_sample_stats_block(raw, wp);
  if (!start_or.has_value()) return std::unexpected(start_or.error());
  const Eigen::VectorXd x0 = h1_start_for_block(*start_or, 0);
  Eigen::VectorXd mu;
  Eigen::MatrixXd L;
  Eigen::MatrixXd Sigma;
  h1_decode(x0, wp.p, mu, L, Sigma);

  auto em_or = weighted_h1_em_iterate_block(wp, mu, Sigma);
  if (!em_or.has_value()) {
    return std::unexpected(fit_to_post(em_or.error(),
        "ML2S Gamma influence: weighted H1 EM"));
  }

  auto scores_or = fiml_saturated_scores_block(raw, block, mu, Sigma);
  if (!scores_or.has_value()) return std::unexpected(scores_or.error());
  auto Hdev_or = weighted_saturated_hessian_analytic_block(wp, mu, Sigma);
  if (!Hdev_or.has_value()) return std::unexpected(Hdev_or.error());

  const Eigen::Index q = wp.p + vech_len(wp.p);
  const Eigen::MatrixXd H = (wp.n_weight / 2.0) * (*Hdev_or);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(q, q);
  for (Eigen::Index i = 0; i < scores_or->rows(); ++i) {
    const Eigen::VectorXd s = scores_or->row(i).transpose();
    J.noalias() += wp.row_weight(i) * (s * s.transpose());
  }
  J *= 0.25;

  auto Hinv_or = invert_symmetric(
      H, "ML2S Gamma influence: weighted saturated information");
  if (!Hinv_or.has_value()) return std::unexpected(Hinv_or.error());
  Eigen::MatrixXd acov = (*Hinv_or) * J * (*Hinv_or);
  acov = 0.5 * (acov + acov.transpose()).eval();
  Eigen::MatrixXd gamma = wp.n_weight * acov;
  gamma = 0.5 * (gamma + gamma.transpose()).eval();
  if (!gamma.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: non-finite weighted Gamma"));
  }
  return gamma;
}

post_expected<Eigen::MatrixXd>
weighted_two_stage_gamma_influence_fd_block(const RawData& raw,
                                            std::size_t block,
                                            Eigen::Index row,
                                            double eps = 1e-4) {
  if (!(eps > 0.0 && eps < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: finite-difference step must lie in (0, 1)"));
  }
  auto plus_or = weighted_two_stage_gamma_block(raw, block, row, eps);
  if (!plus_or.has_value()) return std::unexpected(plus_or.error());
  auto minus_or = weighted_two_stage_gamma_block(raw, block, row, -eps);
  if (!minus_or.has_value()) return std::unexpected(minus_or.error());

  const double n = static_cast<double>(raw.X[block].rows());
  Eigen::MatrixXd out = (n / (2.0 * eps)) * (*plus_or - *minus_or);
  out = 0.5 * (out + out.transpose()).eval();
  return out;
}

// === Analytic ML2S missing-data Γ-influence ================================
//
// The Stage-1 saturated-FIML sandwich Γ_b = n·H⁻¹JH⁻¹ feeds the non-NT Stage-2
// weight (DWLS/ADF/DLS); the estimated-weight IJ correction needs the per-case
// influence dΓ_b/dw_i. Writing θ = (μ, vech Σ), with s_j the deviance score
// (row j of `fiml_saturated_scores_block`) and g_j = −½ s_j the log-likelihood
// gradient, the weighted saturated fit has
//   H = (n/2)·H̄_dev = Σ_j w_j I_j   (total observed info; I_j = ½ per-case
//                                     deviance Hessian)
//   J = ¼ Σ_j w_j s_j s_jᵀ = Σ_j w_j g_j g_jᵀ      Γ = n·H⁻¹JH⁻¹
// and the saturated-moment influence is Δθ_i = H⁻¹ g_i (= `saturated_em_moment_
// influence` row i). The total weight-derivative is direct (explicit weight on
// case i) plus the θ-movement of the fit:
//   dH/dw_i = I_i + (∂H/∂θ)·Δθ_i ,   dJ/dw_i = g_i g_iᵀ + (∂J/∂θ)·Δθ_i
//   dΓ/dw_i = H⁻¹JH⁻¹·(dn/dw_i = 1)
//           + n·[ −H⁻¹(dH)H⁻¹JH⁻¹ + H⁻¹(dJ)H⁻¹ − H⁻¹JH⁻¹(dH)H⁻¹ ].
// The J-movement collapses without any per-case info matrix:
//   (∂J/∂θ)·v = ¼ (dSᵀ S + Sᵀ dS) ,  dS = directional deviance score along v.
// The H-movement is the per-pattern directional third derivative of the
// analytic saturated Hessian. `weighted_two_stage_gamma_influence_fd_block`
// multiplies its central difference by n, so this returns n·dΓ/dw_i to match it.

// ∂s/∂θ·v for the deviance scores: directional twin of
// `fiml_saturated_scores_block` along v = (dmu, dSigma).
post_expected<Eigen::MatrixXd>
fiml_saturated_scores_directional_block(const RawData& raw, std::size_t block,
                                        const Eigen::VectorXd& mu,
                                        const Eigen::MatrixXd& Sigma,
                                        const Eigen::VectorXd& dmu,
                                        const Eigen::MatrixXd& dSigma) {
  const Eigen::MatrixXd& X = raw.X[block];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd dscores = Eigen::MatrixXd::Zero(n, p + pstar);
  std::vector<Eigen::Index> obs;
  for (Eigen::Index r = 0; r < n; ++r) {
    obs.clear();
    obs.reserve(static_cast<std::size_t>(p));
    if (raw.mask.empty()) {
      for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
    } else {
      for (Eigen::Index c = 0; c < p; ++c)
        if (raw.mask[block](r, c) != 0) obs.push_back(c);
    }
    if (obs.empty() || !finite_observed_row(X, obs, r)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: invalid observed row"));
    }
    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: Sigma_oo is not positive definite"));
    }
    const Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(q, q));
    const Eigen::MatrixXd dS_o = select_square(dSigma, obs);
    const Eigen::VectorXd dmu_o = select_vector(dmu, obs);
    const Eigen::VectorXd Mu_o = select_vector(mu, obs);
    Eigen::VectorXd d(q);
    for (Eigen::Index j = 0; j < q; ++j) {
      d(j) = X(r, obs[static_cast<std::size_t>(j)]) - Mu_o(j);
    }
    const Eigen::VectorXd z = A * d;
    const Eigen::MatrixXd dA = -(A * dS_o * A);
    const Eigen::VectorXd dz = dA * d - A * dmu_o;  // dd/dθ·v = −dmu_o
    for (Eigen::Index i = 0; i < q; ++i) {
      dscores(r, obs[static_cast<std::size_t>(i)]) = -2.0 * dz(i);
    }
    // G = A − z zᵀ ⇒ dG = dA − (dz zᵀ + z dzᵀ); same vech packing as the base.
    const Eigen::MatrixXd dG =
        dA - (dz * z.transpose() + z * dz.transpose());
    for (Eigen::Index cj = 0; cj < q; ++cj) {
      const Eigen::Index c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = cj; ri < q; ++ri) {
        const Eigen::Index rr0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index rr = std::max(rr0, c);
        const Eigen::Index cc = std::min(rr0, c);
        const Eigen::Index idx = p + vech_index(p, rr, cc);
        dscores(r, idx) += (ri == cj) ? dG(ri, cj) : 2.0 * dG(ri, cj);
      }
    }
  }
  return dscores;
}

// Per-case observed information I_i = ½·(case deviance Hessian), in the full
// q×q [μ; vech Σ] layout — the single-case slice of the analytic Hessian
// assembly (nr = 1, Zsum = z_i, M = z_i z_iᵀ).
post_expected<Eigen::MatrixXd>
fiml_saturated_case_info_block(const RawData& raw, std::size_t block,
                               Eigen::Index row, const Eigen::VectorXd& mu,
                               const Eigen::MatrixXd& Sigma) {
  const Eigen::MatrixXd& X = raw.X[block];
  const Eigen::Index p = X.cols();
  const Eigen::Index q = p + vech_len(p);
  std::vector<Eigen::Index> obs;
  obs.reserve(static_cast<std::size_t>(p));
  if (raw.mask.empty()) {
    for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
  } else {
    for (Eigen::Index c = 0; c < p; ++c)
      if (raw.mask[block](row, c) != 0) obs.push_back(c);
  }
  if (obs.empty() || !finite_observed_row(X, obs, row)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: invalid observed row"));
  }
  const Eigen::Index m = static_cast<Eigen::Index>(obs.size());
  const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: Sigma_oo is not positive definite"));
  }
  Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(m, m));
  A = 0.5 * (A + A.transpose()).eval();
  const Eigen::VectorXd Mu_o = select_vector(mu, obs);
  Eigen::VectorXd d(m);
  for (Eigen::Index j = 0; j < m; ++j) {
    d(j) = X(row, obs[static_cast<std::size_t>(j)]) - Mu_o(j);
  }
  const Eigen::VectorXd Zsum = A * d;             // z_i (nr = 1)
  const Eigen::MatrixXd M = Zsum * Zsum.transpose();

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  for (Eigen::Index j = 0; j < m; ++j) {
    const Eigen::Index fj = obs[static_cast<std::size_t>(j)];
    for (Eigen::Index i = 0; i < m; ++i) {
      H(fj, obs[static_cast<std::size_t>(i)]) += 2.0 * A(i, j);
    }
  }
  std::vector<SigmaBasis> basis;
  basis.reserve(static_cast<std::size_t>(m * (m + 1) / 2));
  for (Eigen::Index cj = 0; cj < m; ++cj) {
    for (Eigen::Index ri = cj; ri < m; ++ri) {
      const Eigen::Index full_r0 = obs[static_cast<std::size_t>(ri)];
      const Eigen::Index full_c = obs[static_cast<std::size_t>(cj)];
      SigmaBasis e;
      e.full_index =
          p + vech_index(p, std::max(full_r0, full_c), std::min(full_r0, full_c));
      e.a = ri;
      e.b = cj;
      Eigen::VectorXd ADz =
          (ri == cj) ? Eigen::VectorXd(A.col(ri) * Zsum(ri))
                     : Eigen::VectorXd(A.col(ri) * Zsum(cj) +
                                       A.col(cj) * Zsum(ri));
      for (Eigen::Index i = 0; i < m; ++i) {
        const Eigen::Index full_i = obs[static_cast<std::size_t>(i)];
        const double h = 2.0 * ADz(i);
        H(full_i, e.full_index) += h;
        H(e.full_index, full_i) += h;
      }
      basis.push_back(e);
    }
  }
  for (std::size_t kk = 0; kk < basis.size(); ++kk) {
    const SigmaBasis& k = basis[kk];
    for (std::size_t ll = 0; ll <= kk; ++ll) {
      const SigmaBasis& l = basis[ll];
      const double h = -basis_trace_xy(A, A, l, k) +
                       basis_trace_xy(A, M, l, k) + basis_trace_xy(A, M, k, l);
      H(k.full_index, l.full_index) += h;
      if (kk != ll) H(l.full_index, k.full_index) += h;
    }
  }
  H = 0.5 * (H + H.transpose()).eval();  // deviance Hessian (nr = 1)
  return Eigen::MatrixXd(0.5 * H);       // I_i = ½ · deviance Hessian
}

// ∂(H̄_dev)/∂θ·v: per-pattern directional twin of
// `fiml_saturated_hessian_analytic_block` (pattern data moments held fixed).
// Caller scales by n/2 for the total-information derivative ∂H/∂θ·v.
post_expected<Eigen::MatrixXd>
fiml_saturated_hessian_directional_block(const FIMLCache& cache,
                                         std::size_t block,
                                         const Eigen::VectorXd& mu,
                                         const Eigen::MatrixXd& Sigma,
                                         const Eigen::VectorXd& dmu,
                                         const Eigen::MatrixXd& dSigma) {
  const Eigen::Index p = cache.block_p[block];
  const Eigen::Index q = p + vech_len(p);
  Eigen::MatrixXd dH = Eigen::MatrixXd::Zero(q, q);
  std::int64_t n_block = 0;

  for (const FIMLPattern& pat : cache.patterns) {
    if (pat.block != block) continue;
    n_block += pat.n_obs;
    const auto& obs = pat.observed;
    const Eigen::Index m = static_cast<Eigen::Index>(obs.size());
    const double nr = static_cast<double>(pat.n_obs);

    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: Sigma_oo is not positive definite"));
    }
    Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(m, m));
    A = 0.5 * (A + A.transpose()).eval();
    const Eigen::MatrixXd dS_o = select_square(dSigma, obs);
    const Eigen::MatrixXd dA = -(A * dS_o * A);
    const Eigen::VectorXd dmu_o = select_vector(dmu, obs);
    const Eigen::VectorXd Mu_o = select_vector(mu, obs);
    const Eigen::VectorXd dbar = pat.mean - Mu_o;
    const Eigen::VectorXd ddbar = -dmu_o;

    const Eigen::VectorXd Zsum = nr * (A * dbar);
    const Eigen::VectorXd dZsum = nr * (dA * dbar + A * ddbar);
    const Eigen::MatrixXd sum_ddT = nr * (pat.cov + dbar * dbar.transpose());
    const Eigen::MatrixXd dsum_ddT =
        nr * (ddbar * dbar.transpose() + dbar * ddbar.transpose());
    const Eigen::MatrixXd M = A * sum_ddT * A;
    const Eigen::MatrixXd dM =
        dA * sum_ddT * A + A * dsum_ddT * A + A * sum_ddT * dA;

    for (Eigen::Index j = 0; j < m; ++j) {
      const Eigen::Index fj = obs[static_cast<std::size_t>(j)];
      for (Eigen::Index i = 0; i < m; ++i) {
        dH(obs[static_cast<std::size_t>(i)], fj) += 2.0 * nr * dA(i, j);
      }
    }

    std::vector<SigmaBasis> basis;
    basis.reserve(static_cast<std::size_t>(m * (m + 1) / 2));
    for (Eigen::Index cj = 0; cj < m; ++cj) {
      for (Eigen::Index ri = cj; ri < m; ++ri) {
        const Eigen::Index full_r0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index full_c = obs[static_cast<std::size_t>(cj)];
        SigmaBasis e;
        e.full_index = p + vech_index(p, std::max(full_r0, full_c),
                                      std::min(full_r0, full_c));
        e.a = ri;
        e.b = cj;
        Eigen::VectorXd dADz;
        if (ri == cj) {
          dADz = dA.col(ri) * Zsum(ri) + A.col(ri) * dZsum(ri);
        } else {
          dADz = dA.col(ri) * Zsum(cj) + A.col(ri) * dZsum(cj) +
                 dA.col(cj) * Zsum(ri) + A.col(cj) * dZsum(ri);
        }
        for (Eigen::Index i = 0; i < m; ++i) {
          const Eigen::Index full_i = obs[static_cast<std::size_t>(i)];
          const double h = 2.0 * dADz(i);
          dH(full_i, e.full_index) += h;
          dH(e.full_index, full_i) += h;
        }
        basis.push_back(e);
      }
    }

    for (std::size_t kk = 0; kk < basis.size(); ++kk) {
      const SigmaBasis& k = basis[kk];
      for (std::size_t ll = 0; ll <= kk; ++ll) {
        const SigmaBasis& l = basis[ll];
        const double h =
            -nr * (basis_trace_xy(dA, A, l, k) + basis_trace_xy(A, dA, l, k)) +
            (basis_trace_xy(dA, M, l, k) + basis_trace_xy(A, dM, l, k)) +
            (basis_trace_xy(dA, M, k, l) + basis_trace_xy(A, dM, k, l));
        dH(k.full_index, l.full_index) += h;
        if (kk != ll) dH(l.full_index, k.full_index) += h;
      }
    }
  }

  if (n_block <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: block has no observations"));
  }
  dH /= static_cast<double>(n_block);
  return Eigen::MatrixXd(0.5 * (dH + dH.transpose()));
}

// Per-block precompute for the analytic Γ-influence: base scores S, total info
// inverse H⁻¹, and the sandwich H⁻¹JH⁻¹ at the (unperturbed) saturated fit.
struct Ml2sGammaInfluencePrep {
  Eigen::Index n = 0;
  Eigen::Index q = 0;
  Eigen::VectorXd mu;
  Eigen::MatrixXd Sigma;
  Eigen::MatrixXd scores;  // n×q deviance scores
  Eigen::MatrixXd Hinv;    // q×q
  Eigen::MatrixXd HiJHi;   // q×q (= H⁻¹ J H⁻¹)
};

post_expected<Ml2sGammaInfluencePrep>
prep_ml2s_gamma_influence(const RawData& raw, const FIMLCache& cache,
                          std::size_t block, const Eigen::VectorXd& mu,
                          const Eigen::MatrixXd& Sigma) {
  auto scores_or = fiml_saturated_scores_block(raw, block, mu, Sigma);
  if (!scores_or.has_value()) return std::unexpected(scores_or.error());
  auto Hbar_or = fiml_saturated_hessian_analytic_block(cache, block, mu, Sigma);
  if (!Hbar_or.has_value()) return std::unexpected(Hbar_or.error());

  const double n = static_cast<double>(raw.X[block].rows());
  const Eigen::MatrixXd H = (n / 2.0) * (*Hbar_or);
  const Eigen::MatrixXd J = 0.25 * (scores_or->transpose() * (*scores_or));
  auto Hinv_or = invert_symmetric(
      H, "ML2S Gamma influence: weighted saturated information");
  if (!Hinv_or.has_value()) return std::unexpected(Hinv_or.error());

  Ml2sGammaInfluencePrep prep;
  prep.n = raw.X[block].rows();
  prep.q = scores_or->cols();
  prep.mu = mu;
  prep.Sigma = Sigma;
  prep.scores = std::move(*scores_or);
  prep.Hinv = std::move(*Hinv_or);
  prep.HiJHi = prep.Hinv * J * prep.Hinv;
  prep.HiJHi = 0.5 * (prep.HiJHi + prep.HiJHi.transpose()).eval();
  return prep;
}

// n·dΓ_b/dw_i for case `i` — the analytic drop-in for
// `weighted_two_stage_gamma_influence_fd_block`.
post_expected<Eigen::MatrixXd>
weighted_two_stage_gamma_influence_analytic_block(
    const RawData& raw, const FIMLCache& cache, std::size_t block,
    const Ml2sGammaInfluencePrep& prep, Eigen::Index i) {
  const Eigen::Index p = prep.Sigma.rows();
  const double n = static_cast<double>(prep.n);
  const Eigen::VectorXd s_i = prep.scores.row(i).transpose();
  const Eigen::VectorXd g_i = -0.5 * s_i;
  const Eigen::VectorXd dtheta = prep.Hinv * g_i;  // Δθ_i = H⁻¹ g_i
  Eigen::VectorXd dmu = dtheta.head(p);
  Eigen::MatrixXd dSigma(p, p);
  vech_unpack(dtheta.segment(p, vech_len(p)), p, dSigma);

  auto Ii_or = fiml_saturated_case_info_block(raw, block, i, prep.mu, prep.Sigma);
  if (!Ii_or.has_value()) return std::unexpected(Ii_or.error());
  auto dS_or = fiml_saturated_scores_directional_block(
      raw, block, prep.mu, prep.Sigma, dmu, dSigma);
  if (!dS_or.has_value()) return std::unexpected(dS_or.error());
  auto dHbar_or = fiml_saturated_hessian_directional_block(
      cache, block, prep.mu, prep.Sigma, dmu, dSigma);
  if (!dHbar_or.has_value()) return std::unexpected(dHbar_or.error());

  // J̇_i = g_i g_iᵀ + ¼(dSᵀS + SᵀdS);  Ḣ_i = I_i + (n/2)·∂H̄/∂θ·Δθ_i.
  const Eigen::MatrixXd Jdot =
      g_i * g_i.transpose() +
      0.25 * (dS_or->transpose() * prep.scores +
              prep.scores.transpose() * (*dS_or));
  const Eigen::MatrixXd Hdot = *Ii_or + (n / 2.0) * (*dHbar_or);

  const Eigen::MatrixXd& Hinv = prep.Hinv;
  const Eigen::MatrixXd& HiJHi = prep.HiJHi;
  Eigen::MatrixXd dGamma =
      HiJHi + n * (-(Hinv * Hdot * HiJHi) + (Hinv * Jdot * Hinv) -
                   (HiJHi * Hdot * Hinv));
  dGamma = 0.5 * (dGamma + dGamma.transpose()).eval();
  return Eigen::MatrixXd(n * dGamma);
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

// Gradient of the FIML deviance F with respect to the FULL per-block moments,
// aggregated over observed-value patterns: ∂F/∂Σ_b = G_b in the symmetric
// matrix-derivative sense (dF = tr(G_b · dΣ_b) + u_bᵀ dμ_b), with the
// per-pattern weight Σ_oo⁻¹ − Σ_oo⁻¹(S_r + ddᵀ)Σ_oo⁻¹ scattered into the full
// p×p index space. These are the weights the second-order chain-rule term of
// the observed Hessian contracts against ∂²Σ/∂θ_a∂θ_b and ∂²μ/∂θ_a∂θ_b.
struct FIMLMomentGradient {
  std::vector<Eigen::MatrixXd> G;  // per block, p×p symmetric
  std::vector<Eigen::VectorXd> u;  // per block, p
};

post_expected<FIMLMomentGradient>
fiml_moment_gradient(const FIMLCache& cache,
                     const model::ImpliedMoments& moments) {
  FIMLMomentGradient out;
  const std::size_t n_blocks = cache.block_p.size();
  out.G.resize(n_blocks);
  out.u.resize(n_blocks);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::Index p = cache.block_p[b];
    out.G[b] = Eigen::MatrixXd::Zero(p, p);
    out.u[b] = Eigen::VectorXd::Zero(p);
  }

  for (const FIMLPattern& pat : cache.patterns) {
    const auto& obs = pat.observed;
    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    const Eigen::MatrixXd Sigma_o = select_square(moments.sigma[pat.block], obs);
    const Eigen::VectorXd Mu_o = select_vector(moments.mu[pat.block], obs);
    const Eigen::VectorXd d = pat.mean - Mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();

    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML observed Hessian: implied observed-pattern Σ is not "
          "positive definite"));
    }
    const Eigen::MatrixXd SigmaInv =
        llt.solve(Eigen::MatrixXd::Identity(q, q));
    Eigen::MatrixXd Gp = SigmaInv - SigmaInv * A * SigmaInv;
    Gp = 0.5 * (Gp + Gp.transpose()).eval();
    const Eigen::VectorXd z = SigmaInv * d;

    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);
    auto& Gb = out.G[pat.block];
    auto& ub = out.u[pat.block];
    for (Eigen::Index cj = 0; cj < q; ++cj) {
      const Eigen::Index c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = 0; ri < q; ++ri) {
        const Eigen::Index r = obs[static_cast<std::size_t>(ri)];
        Gb(r, c) += scale * Gp(ri, cj);
      }
      ub(c) += -2.0 * scale * z(cj);
    }
  }
  return out;
}

// Analytic observed Hessian of the per-observation-averaged FIML deviance F
// with respect to θ — same object as `fiml_observed_hessian_fd`, exact instead
// of central-difference. Chain rule through the moment map (μ(θ), Σ(θ)):
//
//   H = Σ_b (n_b/N) · Δ_bᵀ · W_b · Δ_b                       (first order)
//     + Σ_b [ tr(G_b · ∂²Σ_b/∂θ_a∂θ_b) + u_bᵀ ∂²μ_b/∂θ_a∂θ_b ]  (second order)
//
// where W_b is the per-block moment-space Hessian of the mean deviance
// (`fiml_saturated_hessian_analytic_block` evaluated at the model-implied
// moments), Δ_b stacks [∂μ_b/∂θ ; ∂vech(Σ_b)/∂θ], and (G_b, u_b) is the
// pattern-aggregated moment gradient above contracted with the closed-form
// LISREL second derivatives (`detail::second_sigma_trace` / `detail::second_mu`).
post_expected<Eigen::MatrixXd>
fiml_observed_hessian_analytic(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const FIMLCache& cache,
                               const SampleStats& start_samp,
                               const Estimates& est) {
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
  if (est.theta.size() != q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML observed Hessian: theta size mismatch"));
  }

  auto am_or = ev.assembled(est.theta);
  if (!am_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::assembled failed: " + am_or.error().detail));
  }
  auto eval_or = ev.evaluate(est.theta, true, true);
  if (!eval_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::evaluate failed: " + eval_or.error().detail));
  }
  const auto& eval = *eval_or;
  const auto locs = ev.param_locations();

  const std::size_t n_blocks = cache.block_p.size();
  Eigen::Index total_p = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) total_p += cache.block_p[b];
  if (eval.J_mu.rows() != total_p || eval.J_mu.cols() != q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML observed Hessian: mean structure is required"));
  }

  std::vector<std::int64_t> n_block(n_blocks, 0);
  for (const FIMLPattern& pat : cache.patterns) n_block[pat.block] += pat.n_obs;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);

  // First-order term: per-block moment-space Hessian at the implied moments,
  // contracted with the stacked [μ; vech Σ] Jacobian.
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::Index p = cache.block_p[b];
    const Eigen::Index pstar = vech_len(p);
    auto W_or = fiml_saturated_hessian_analytic_block(cache, b,
                                                      eval.moments.mu[b],
                                                      eval.moments.sigma[b]);
    if (!W_or.has_value()) return std::unexpected(W_or.error());

    Eigen::MatrixXd Delta(p + pstar, q);
    Delta.topRows(p) = eval.J_mu.middleRows(cache.mu_offsets[b], p);
    Delta.bottomRows(pstar) =
        eval.J_sigma.middleRows(cache.sigma_offsets[b], pstar);

    const double weight = static_cast<double>(n_block[b]) /
                          static_cast<double>(cache.n_total);
    H.noalias() += weight * (Delta.transpose() * ((*W_or) * Delta));
  }

  // Second-order term: pattern-aggregated moment gradient contracted with the
  // closed-form LISREL second derivatives.
  auto mg_or = fiml_moment_gradient(cache, eval.moments);
  if (!mg_or.has_value()) return std::unexpected(mg_or.error());
  const FIMLMomentGradient& mg = *mg_or;

  std::vector<detail::SecondOrderWeights> sow;
  sow.reserve(n_blocks);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    sow.push_back(detail::SecondOrderWeights::build(mg.G[b],
                                                    am_or->blocks[b],
                                                    /*has_means=*/true));
  }

  for (Eigen::Index a = 0; a < q; ++a) {
    for (Eigen::Index b = a; b < q; ++b) {
      const auto& la = locs[static_cast<std::size_t>(a)];
      const auto& lb = locs[static_cast<std::size_t>(b)];
      if (la.block != lb.block) continue;
      const auto blk = static_cast<std::size_t>(la.block);
      const auto& bm = am_or->blocks[blk];
      double h2 = detail::second_sigma_trace(la, lb, sow[blk], bm);
      h2 += detail::second_mu(la, lb, bm, sow[blk].A_alpha).dot(mg.u[blk]);
      H(a, b) += h2;
      if (a != b) H(b, a) += h2;
    }
  }

  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

post_expected<double>
fiml_saturated_trace_h1(const RawData& raw,
                        const FIMLCache& cache,
                        const FIMLH1& h1) {
  if (cache.block_p.size() != raw.X.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: cache and raw block count mismatch"));
  }
  if (auto e = validate_h1_blocks(cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "FIML H1 moments"));
  }

  double trace = 0.0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const Eigen::VectorXd& h1_mu = h1.mu[b];
    const Eigen::MatrixXd& h1_sigma = h1.sigma[b];
    auto h1_scores_or = fiml_saturated_scores_block(raw, b,
                                                    h1_mu, h1_sigma);
    if (!h1_scores_or.has_value()) return std::unexpected(h1_scores_or.error());
    auto h1_H_or = fiml_saturated_hessian_analytic_block(cache, b, h1_mu,
                                                         h1_sigma);
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

post_expected<FIMLScoreMeatBread>
fiml_score_meat_bread(const spec::LatentStructure& pt,
                      const model::MatrixRep& rep,
                      const RawData& raw,
                      const FIMLPack& pack,
                      const Estimates& est) {
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
  auto scores_or = fiml_casewise_scores(raw, pack.cache, eval.moments,
                                        eval.J_sigma, eval.J_mu);
  if (!scores_or.has_value()) return std::unexpected(scores_or.error());
  auto H_or =
      fiml_observed_hessian_analytic(pt, rep, pack.cache, pack.start_stats, est);
  if (!H_or.has_value()) return std::unexpected(H_or.error());

  FIMLScoreMeatBread out;
  out.scores = std::move(*scores_or);
  out.hessian = std::move(*H_or);
  return out;
}

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

    std::unordered_map<std::vector<Eigen::Index>, std::vector<Eigen::Index>,
                       IndexVectorHash> rows_by_obs;
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

fit_expected<FIMLPack>
fiml_pack(const RawData& raw) {
  auto cache_or = FIML{}.prepare(raw);
  if (!cache_or.has_value()) return std::unexpected(cache_or.error());
  auto start_or = fiml_start_sample_stats(raw);
  if (!start_or.has_value()) return std::unexpected(start_or.error());
  return FIMLPack{std::move(*cache_or), std::move(*start_or)};
}

fit_expected<FIMLH1>
fiml_h1_moments(const RawData& raw, const FIMLPack& pack) {
  const FIMLCache& cache = pack.cache;
  const SampleStats& starts = pack.start_stats;
  const std::size_t B = cache.block_p.size();
  if (starts.S.size() != B || starts.mean.size() != B) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1: start statistics do not match raw-data blocks"));
  }

  FIMLH1 out;
  out.mu.resize(B);
  out.sigma.resize(B);

  if (raw.mask.empty()) {
    auto value_or = h1_complete_data_value(starts);
    if (!value_or.has_value()) return std::unexpected(value_or.error());
    out.value = *value_or;
    for (std::size_t b = 0; b < B; ++b) {
      out.mu[b] = starts.mean[b];
      out.sigma[b] = 0.5 * (starts.S[b] + starts.S[b].transpose());
    }
    return out;
  }

  double total = 0.0;
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = cache.block_p[b];
    const Eigen::VectorXd x0 = h1_start_for_block(starts, b);
    Eigen::VectorXd mu;
    Eigen::MatrixXd L;
    Eigen::MatrixXd Sigma;
    h1_decode(x0, p, mu, L, Sigma);

    // The converged value is evaluated at the same (mu, Sigma) the iteration
    // returns, so the H1 value and moments stay mutually consistent.
    auto final_val = h1_em_iterate_block(cache, b, mu, Sigma);
    if (!final_val.has_value()) return std::unexpected(final_val.error());
    total += *final_val;
    out.mu[b] = std::move(mu);
    out.sigma[b] = std::move(Sigma);
  }
  out.value = total;
  return out;
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
    Eigen::MatrixXd Sigma_o(q, q);
    Eigen::VectorXd d(q);
    for (Eigen::Index j = 0; j < q; ++j) {
      const Eigen::Index cj = pat.observed[static_cast<std::size_t>(j)];
      d(j) = pat.mean(j) - Mu(cj);
      for (Eigen::Index i = 0; i < q; ++i) {
        Sigma_o(i, j) =
            Sigma(pat.observed[static_cast<std::size_t>(i)], cj);
      }
    }

    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "FIML: implied observed-pattern Σ is not positive definite"));
    }
    const auto& L = llt.matrixL();
    double log_det = 0.0;
    for (Eigen::Index i = 0; i < q; ++i) log_det += std::log(L(i, i));
    log_det *= 2.0;

    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);
    if (!want_gradient) {
      const Eigen::MatrixXd A = pat.cov + d * d.transpose();
      f += scale * (log_det + llt.solve(A).trace());
    } else {
      Eigen::MatrixXd SigmaInv = Eigen::MatrixXd::Identity(q, q);
      llt.solveInPlace(SigmaInv);
      const Eigen::VectorXd z = SigmaInv * d;
      const double quad = SigmaInv.cwiseProduct(pat.cov).sum() + d.dot(z);
      f += scale * (log_det + quad);

      Eigen::MatrixXd SigmaInvCov(q, q);
      SigmaInvCov.noalias() = SigmaInv * pat.cov;
      Eigen::MatrixXd G(q, q);
      G.noalias() = SigmaInv - SigmaInvCov * SigmaInv;
      G.noalias() -= z * z.transpose();
      G = 0.5 * (G + G.transpose());

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

namespace {

post_expected<FIMLExtras>
fiml_extras_impl(spec::LatentStructure pt,
                 const model::MatrixRep& rep,
                 const RawData& raw,
                 const Estimates& est,
                 const FIMLPack& pack,
                 const FIMLH1& h1,
                 FIML discrepancy) {
  const FIMLCache& cache = pack.cache;
  const SampleStats& start_samp = pack.start_stats;
  if (auto e = validate_h1_blocks(cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "FIML H1 moments"));
  }

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
  out.unrestricted_logl = -0.5 * (N * h1.value + c);
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
    const Eigen::VectorXd& h1_mu = h1.mu[b];
    const Eigen::MatrixXd& h1_sigma = h1.sigma[b];
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

}  // namespace

post_expected<FIMLExtras>
fiml_extras(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const RawData& raw,
            const Estimates& est,
            FIML discrepancy) {
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 likelihood"));
  }
  return fiml_extras_impl(std::move(pt), rep, raw, est, *pack_or, *h1_or,
                          discrepancy);
}

post_expected<FIMLExtras>
fiml_extras(spec::LatentStructure pt,
            const model::MatrixRep& rep,
            const RawData& raw,
            const Estimates& est,
            const FIMLPack& pack,
            const FIMLH1& h1) {
  return fiml_extras_impl(std::move(pt), rep, raw, est, pack, h1, FIML{});
}

namespace {

post_expected<Eigen::MatrixXd>
fiml_observed_information_impl(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const RawData& raw,
                               const Estimates& est,
                               const FIMLPack& pack) {
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  // H = ∂²F/∂θ² of the per-observation-averaged deviance F. The FIML
  // log-likelihood is logl = −½(N·F + c), so the observed information is
  // −∂²logl/∂θ² = ½·N·H. F here is the FULL-scale kernel deviance, NOT the
  // ½F optimiser objective — est.fmin is halved only in the fit adapter, so
  // this ½·N·H stays correct unchanged.
  auto H_or = fiml_observed_hessian_analytic(std::move(pt), rep, pack.cache,
                                             pack.start_stats, est);
  if (!H_or.has_value()) return std::unexpected(H_or.error());
  const double N = static_cast<double>(pack.cache.n_total);
  return Eigen::MatrixXd(0.5 * N * (*H_or));
}

}  // namespace

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
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  return fiml_observed_information_impl(std::move(pt), rep, raw, est,
                                        *pack_or);
}

post_expected<Eigen::MatrixXd>
fiml_observed_information(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          const FIMLPack& pack) {
  return fiml_observed_information_impl(std::move(pt), rep, raw, est, pack);
}

namespace diagnostic {

post_expected<Eigen::MatrixXd>
fiml_observed_information_fd(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const RawData& raw,
                             const Estimates& est,
                             FIML discrepancy,
                             double h_step) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_observed_information_fd: h_step must be positive"));
  }
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  auto H_or = fiml_observed_hessian_fd(pt, rep, raw, pack_or->cache,
                                       pack_or->start_stats, est,
                                       discrepancy, h_step);
  if (!H_or.has_value()) return std::unexpected(H_or.error());
  const double N = static_cast<double>(pack_or->cache.n_total);
  return Eigen::MatrixXd(0.5 * N * (*H_or));
}

}  // namespace diagnostic

namespace {

post_expected<FIMLRobustMLR>
fiml_robust_mlr_impl(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const RawData& raw,
                     const Estimates& est,
                     int df,
                     double chisq,
                     const FIMLPack& pack,
                     const FIMLH1& h1) {
  const FIMLCache& cache = pack.cache;
  const SampleStats& start_samp = pack.start_stats;
  if (auto e = validate_h1_blocks(cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "FIML H1 moments"));
  }

  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "resolve_fixed_x_from_sample"));
  }

  auto parts_or = fiml_score_meat_bread(pt, rep, raw, pack, est);
  if (!parts_or.has_value()) return std::unexpected(parts_or.error());
  Eigen::MatrixXd scores = std::move(parts_or->scores);
  Eigen::MatrixXd H = std::move(parts_or->hessian);

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

  auto h1_trace_or = fiml_saturated_trace_h1(raw, cache, h1);
  if (!h1_trace_or.has_value()) return std::unexpected(h1_trace_or.error());
  out.trace_ugamma_h1 = *h1_trace_or;
  out.trace_ugamma = out.trace_ugamma_h1 - out.trace_ugamma_h0;
  if (df > 0) {
    out.scaling_factor = out.trace_ugamma / static_cast<double>(df);
    out.chisq_scaled = (out.scaling_factor > 0.0)
        ? chisq / out.scaling_factor
        : std::numeric_limits<double>::quiet_NaN();
  } else {
    out.scaling_factor = std::numeric_limits<double>::quiet_NaN();
    out.chisq_scaled = std::numeric_limits<double>::quiet_NaN();
  }
  (void)q;
  return out;
}

}  // namespace

post_expected<FIMLRobustMLR>
fiml_robust_mlr(spec::LatentStructure pt,
                const model::MatrixRep& rep,
                const RawData& raw,
                const Estimates& est,
                int df,
                double chisq,
                FIML discrepancy,
                double h_step) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_robust_mlr: h_step must be > 0"));
  }
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 moments"));
  }
  return fiml_robust_mlr_impl(std::move(pt), rep, raw, est, df, chisq,
                              *pack_or, *h1_or);
}

post_expected<FIMLRobustMLR>
fiml_robust_mlr(spec::LatentStructure pt,
                const model::MatrixRep& rep,
                const RawData& raw,
                const Estimates& est,
                int df,
                double chisq,
                const FIMLPack& pack,
                const FIMLH1& h1) {
  return fiml_robust_mlr_impl(std::move(pt), rep, raw, est, df, chisq,
                              pack, h1);
}

namespace {

post_expected<FIMLEtaJacobian>
fiml_eta_jacobian_impl(spec::LatentStructure pt,
                       const model::MatrixRep& rep,
                       const RawData& raw,
                       const Estimates& est,
                       const FIMLPack& pack) {
  const FIMLCache& cache = pack.cache;
  const SampleStats& start_samp = pack.start_stats;
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "resolve_fixed_x_from_sample"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_eta_jacobian: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_eta_jacobian: theta size mismatch"));
  }
  auto eval_or = ev.evaluate(est.theta, /*with_sigma_jacobian=*/true,
                             /*with_mu_jacobian=*/true);
  if (!eval_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_eta_jacobian: ModelEvaluator::evaluate failed: " +
            eval_or.error().detail));
  }
  const Eigen::MatrixXd& Js = eval_or->J_sigma;
  const Eigen::MatrixXd& Jm = eval_or->J_mu;
  const Eigen::Index qpar = static_cast<Eigen::Index>(ev.n_free());
  const bool has_mu = Jm.rows() > 0;

  Eigen::Index Q = 0;
  for (Eigen::Index p : cache.block_p) Q += p + p * (p + 1) / 2;

  Eigen::MatrixXd Delta = Eigen::MatrixXd::Zero(Q, qpar);
  const std::size_t B = raw.X.size();
  Eigen::Index eta_off = 0, mu_off = 0, sig_off = 0;
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = raw.X[b].cols();
    const Eigen::Index ps = p * (p + 1) / 2;
    if (has_mu) Delta.block(eta_off, 0, p, qpar) = Jm.block(mu_off, 0, p, qpar);
    Delta.block(eta_off + p, 0, ps, qpar) = Js.block(sig_off, 0, ps, qpar);
    eta_off += p + ps;
    mu_off += p;
    sig_off += ps;
  }
  if (eta_off != Q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_eta_jacobian: assembled Δ rows (" + std::to_string(eta_off) +
            ") != saturated dim (" + std::to_string(Q) + ")"));
  }

  return FIMLEtaJacobian{std::move(Delta)};
}

post_expected<Eigen::MatrixXd>
fiml_structured_h1_information(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const RawData& raw,
                               const Estimates& est,
                               const FIMLPack& pack) {
  const FIMLCache& cache = pack.cache;
  const SampleStats& start_samp = pack.start_stats;
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "resolve_fixed_x_from_sample"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_structured_h1_information: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_structured_h1_information: theta size mismatch"));
  }
  auto eval_or = ev.evaluate(est.theta, /*with_sigma_jacobian=*/false,
                             /*with_mu_jacobian=*/false);
  if (!eval_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_structured_h1_information: ModelEvaluator::evaluate failed: " +
            eval_or.error().detail));
  }

  const std::size_t B = raw.X.size();
  if (B == 0 || cache.block_p.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_structured_h1_information: empty or inconsistent block layout"));
  }

  std::vector<Eigen::Index> q_b(B);
  std::vector<Eigen::Index> off_b(B + 1, 0);
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = cache.block_p[b];
    q_b[b] = p + vech_len(p);
    off_b[b + 1] = off_b[b] + q_b[b];
  }
  const Eigen::Index Q = off_b.back();
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(Q, Q);

  for (std::size_t b = 0; b < B; ++b) {
    auto H_or = fiml_saturated_hessian_analytic_block(
        cache, b, eval_or->moments.mu[b], eval_or->moments.sigma[b]);
    if (!H_or.has_value()) return std::unexpected(H_or.error());

    const double n_b = static_cast<double>(raw.X[b].rows());
    const Eigen::Index q = q_b[b];
    const Eigen::Index off = off_b[b];
    V.block(off, off, q, q) = (n_b / 2.0) * (*H_or);
  }

  return Eigen::MatrixXd(0.5 * (V + V.transpose()));
}

}  // namespace

post_expected<FIMLEtaJacobian>
fiml_eta_jacobian(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const RawData& raw,
                  const Estimates& est,
                  FIML discrepancy) {
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  return fiml_eta_jacobian_impl(std::move(pt), rep, raw, est, *pack_or);
}

post_expected<FIMLEtaJacobian>
fiml_eta_jacobian(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const RawData& raw,
                  const Estimates& est,
                  const FIMLPack& pack) {
  return fiml_eta_jacobian_impl(std::move(pt), rep, raw, est, pack);
}

namespace {

// Single-model FIML residual projector U = V − VΔ(ΔᵀVΔ)⁻¹ΔᵀV in the saturated
// η-metric, with Δ the constraint-collapsed model Jacobian ∂[μ; vech Σ]/∂θ. The
// weight V is supplied by the caller (sm.H saturated, or the structured H1
// curvature) so a nested difference U0 − U1 can share a common V. Shared by the
// GOF spectrum (fiml_ugamma_spectrum_impl) and the method-2001 nested spectrum.
post_expected<Eigen::MatrixXd>
fiml_residual_projector_impl(const spec::LatentStructure& pt,
                             const model::MatrixRep& rep,
                             const RawData& raw,
                             const Estimates& est,
                             const FIMLPack& pack,
                             const Eigen::Ref<const Eigen::MatrixXd>& V) {
  const Eigen::Index Q = V.rows();
  if (V.cols() != Q || Q == 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_residual_projector: V must be square and non-empty"));
  }

  // Δ = ∂[μ(θ); vech Σ(θ)]/∂θ at θ̂.
  auto jac_or = fiml_eta_jacobian_impl(pt, rep, raw, est, pack);
  if (!jac_or.has_value()) return std::unexpected(jac_or.error());
  Eigen::MatrixXd Delta = std::move(jac_or->Delta_theta);
  if (Delta.rows() != Q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_residual_projector: Δ rows (" + std::to_string(Delta.rows()) +
            ") != saturated dim (" + std::to_string(Q) + ")"));
  }

  // Equality constraints: collapse Δ to local free coordinates. Linear
  // constraints use their affine K; nonlinear use null([A_eq; ∂h/∂θ]) at θ̂.
  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  if (pt.nl_constraints.empty()) {
    if (con_or->active()) Delta = (Delta * con_or->K()).eval();
  } else {
    if (est.theta.size() != Delta.cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fiml_residual_projector: theta size mismatch for nonlinear equality "
          "constraint tangent"));
    }
    const NonlinearEqConstraints nl = build_nl_constraints(pt);
    const Eigen::MatrixXd H = nl.jacobian(est.theta);
    if (!H.allFinite()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fiml_residual_projector: nonlinear equality constraint Jacobian is "
          "not finite at theta"));
    }
    const Eigen::Index npar = Delta.cols();
    Eigen::MatrixXd C(con_or->A_eq.rows() + H.rows(), npar);
    if (con_or->A_eq.rows() > 0) {
      C.topRows(con_or->A_eq.rows()) = con_or->A_eq;
    }
    C.bottomRows(H.rows()) = H;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);
    svd.setThreshold(1e-9);
    const Eigen::Index nz = npar - svd.rank();
    if (nz <= 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fiml_residual_projector: equality constraints leave no tangent "
          "directions"));
    }
    Delta = (Delta * svd.matrixV().rightCols(nz)).eval();
  }

  // U = V − VΔ(ΔᵀVΔ)⁻¹ΔᵀV (rank df).
  const Eigen::MatrixXd VD = V * Delta;
  Eigen::MatrixXd DtVD = Delta.transpose() * VD;
  DtVD = (0.5 * (DtVD + DtVD.transpose())).eval();
  auto DtVDinv_or = invert_symmetric(DtVD, "fiml_residual_projector: ΔᵀVΔ");
  if (!DtVDinv_or.has_value()) return std::unexpected(DtVDinv_or.error());
  Eigen::MatrixXd U = V - VD * (*DtVDinv_or) * VD.transpose();
  U = (0.5 * (U + U.transpose())).eval();
  return U;
}

post_expected<FIMLUGammaSpectrum>
fiml_ugamma_spectrum_impl(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          int df,
                          double chi2_lrt,
                          const FIMLPack& pack,
                          const FIMLH1& h1,
                          const SaturatedMoments* sm_precomputed) {
  if (df <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_ugamma_spectrum: requires df > 0"));
  }

  // (1) Saturated-moment ingredients (block-diagonal η-space, multi-group safe):
  //     Γ_mis = acov = H⁻¹ J H⁻¹. V is the saturated observed H1 information
  //     (the FMG-spectrum convention, PD at the saturated optimum). A caller
  //     holding the Stage-1 saturated moments passes them in to skip the rebuild.
  SaturatedMoments sm_owned;
  if (!sm_precomputed) {
    auto sm_or = saturated_em_moments(raw, pack, h1);
    if (!sm_or.has_value()) return std::unexpected(sm_or.error());
    sm_owned = std::move(*sm_or);
  }
  const SaturatedMoments& sm = sm_precomputed ? *sm_precomputed : sm_owned;
  const Eigen::MatrixXd& V = sm.H;
  const Eigen::MatrixXd& G = sm.acov;
  const Eigen::Index Q = V.rows();

  // (2-4) Residual projector U = V − VΔ(ΔᵀVΔ)⁻¹ΔᵀV (constraint-collapsed Δ).
  auto U_or = fiml_residual_projector_impl(pt, rep, raw, est, pack, V);
  if (!U_or.has_value()) return std::unexpected(U_or.error());
  const Eigen::MatrixXd U = std::move(*U_or);

  // (5) Eigenvalues of the non-symmetric U·Γ_mis via the symmetric reduction
  //     RᵀUR with Γ_mis = R Rᵀ (Γ_mis is symmetric PSD); eig(RᵀUR) = eig(U·Γ).
  Eigen::MatrixXd reduced;
  {
    const Eigen::MatrixXd Gs = 0.5 * (G + G.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(Gs);
    if (llt.info() == Eigen::Success) {
      const Eigen::MatrixXd R = llt.matrixL();
      reduced = R.transpose() * U * R;
    } else {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_g(Gs);
      const Eigen::VectorXd d = es_g.eigenvalues().cwiseMax(0.0).cwiseSqrt();
      const Eigen::MatrixXd sq =
          es_g.eigenvectors() * d.asDiagonal() * es_g.eigenvectors().transpose();
      reduced = sq * U * sq;
    }
    reduced = (0.5 * (reduced + reduced.transpose())).eval();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(reduced,
                                                    Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_ugamma_spectrum: eigen-solve failed"));
  }
  const Eigen::VectorXd all = es.eigenvalues();  // Q ascending
  if (static_cast<Eigen::Index>(df) > all.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_ugamma_spectrum: df exceeds spectrum size"));
  }

  FIMLUGammaSpectrum out;
  out.df = df;
  out.chi2_lrt = chi2_lrt;
  out.eigvals = all.tail(df);  // top-df nonzero eigenvalues, ascending
  out.trace_xcheck = out.eigvals.sum();

  // Sanity: the largest projected-out eigenvalue must be ~0 — a non-trivial
  // value means `df` is inconsistent with the U-rank (layout / df bug).
  if (Q > static_cast<Eigen::Index>(df)) {
    const double zero_top = std::abs(all(Q - df - 1));
    const double scale = std::max(1.0, out.eigvals.cwiseAbs().maxCoeff());
    if (zero_top > 1e-6 * scale) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fiml_ugamma_spectrum: df=" + std::to_string(df) +
              " inconsistent with the U·Γ rank (projected-out eigenvalue " +
              std::to_string(zero_top) + " not ~0)"));
    }
  }
  return out;
}

}  // namespace

post_expected<FIMLUGammaSpectrum>
fiml_ugamma_spectrum(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const RawData& raw,
                     const Estimates& est,
                     int df,
                     double chi2_lrt,
                     FIML discrepancy,
                     double h_step) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_ugamma_spectrum: h_step must be > 0"));
  }
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 moments"));
  }
  return fiml_ugamma_spectrum_impl(std::move(pt), rep, raw, est, df, chi2_lrt,
                                   *pack_or, *h1_or, nullptr);
}

post_expected<FIMLUGammaSpectrum>
fiml_ugamma_spectrum(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const RawData& raw,
                     const Estimates& est,
                     int df,
                     double chi2_lrt,
                     const FIMLPack& pack,
                     const FIMLH1& h1) {
  return fiml_ugamma_spectrum_impl(std::move(pt), rep, raw, est, df, chi2_lrt,
                                   pack, h1, nullptr);
}

post_expected<FIMLUGammaSpectrum>
fiml_ugamma_spectrum(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const RawData& raw,
                     const Estimates& est,
                     int df,
                     double chi2_lrt,
                     const FIMLPack& pack,
                     const FIMLH1& h1,
                     const SaturatedMoments& sm) {
  return fiml_ugamma_spectrum_impl(std::move(pt), rep, raw, est, df, chi2_lrt,
                                   pack, h1, &sm);
}

namespace {

post_expected<double>
normal_theory_chisq_from_moments(
    const std::vector<Eigen::VectorXd>& mu_model,
    const std::vector<Eigen::MatrixXd>& sigma_model,
    const std::vector<Eigen::VectorXd>& mu_sat,
    const std::vector<Eigen::MatrixXd>& sigma_sat,
    const std::vector<std::int64_t>& n_obs,
    std::string_view label) {
  if (sigma_model.size() != sigma_sat.size() ||
      mu_model.size() != mu_sat.size() ||
      sigma_model.size() != n_obs.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": moment block count mismatch"));
  }

  double out = 0.0;
  for (std::size_t b = 0; b < sigma_sat.size(); ++b) {
    const Eigen::MatrixXd Sigma =
        0.5 * (sigma_model[b] + sigma_model[b].transpose());
    const Eigen::MatrixXd S = 0.5 * (sigma_sat[b] + sigma_sat[b].transpose());
    if (Sigma.rows() != Sigma.cols() || S.rows() != S.cols() ||
        Sigma.rows() != S.rows() || mu_model[b].size() != Sigma.rows() ||
        mu_sat[b].size() != Sigma.rows()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": moment dimension mismatch"));
    }
    Eigen::LLT<Eigen::MatrixXd> llt_sigma(Sigma);
    Eigen::LLT<Eigen::MatrixXd> llt_s(S);
    if (llt_sigma.info() != Eigen::Success || llt_s.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": covariance is not positive definite"));
    }
    const Eigen::VectorXd d = mu_sat[b] - mu_model[b];
    const double f = log_det_from_llt(llt_sigma) - log_det_from_llt(llt_s) +
        llt_sigma.solve(S).trace() + d.dot(llt_sigma.solve(d)) -
        static_cast<double>(S.rows());
    if (!std::isfinite(f)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(label) + ": non-finite discrepancy"));
    }
    out += static_cast<double>(n_obs[b]) * f;
  }
  if (out < 0.0 && out > -1e-8) out = 0.0;
  return out;
}

std::vector<Eigen::MatrixXd>
independence_covariances_from_saturated(
    const std::vector<Eigen::MatrixXd>& sigma_sat,
    const std::vector<Eigen::Index>& exo_idx) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(sigma_sat.size());
  for (const Eigen::MatrixXd& S : sigma_sat) {
    Eigen::MatrixXd Sigma = S.diagonal().asDiagonal();
    for (Eigen::Index a = 0;
         a < static_cast<Eigen::Index>(exo_idx.size()); ++a) {
      const Eigen::Index ia = exo_idx[static_cast<std::size_t>(a)];
      if (ia < 0 || ia >= S.rows()) continue;
      for (Eigen::Index b = a + 1;
           b < static_cast<Eigen::Index>(exo_idx.size()); ++b) {
        const Eigen::Index ib = exo_idx[static_cast<std::size_t>(b)];
        if (ib < 0 || ib >= S.rows()) continue;
        Sigma(ia, ib) = S(ia, ib);
        Sigma(ib, ia) = S(ib, ia);
      }
    }
    out.push_back(std::move(Sigma));
  }
  return out;
}

int baseline_df_from_block_p(const std::vector<Eigen::Index>& block_p,
                             const std::vector<Eigen::Index>& exo_idx) {
  int out = 0;
  const int px = static_cast<int>(exo_idx.size());
  for (Eigen::Index p : block_p) {
    out += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
    out -= px * (px - 1) / 2;
  }
  return out;
}

Eigen::MatrixXd baseline_eta_delta_from_block_p(
    const std::vector<Eigen::Index>& block_p,
    const std::vector<Eigen::Index>& exo_idx) {
  Eigen::Index n_rows = 0;
  Eigen::Index n_cols = 0;
  const Eigen::Index n_exo = static_cast<Eigen::Index>(exo_idx.size());
  const Eigen::Index n_exo_cov = n_exo * (n_exo - 1) / 2;
  for (Eigen::Index p : block_p) {
    n_rows += p + vech_len(p);
    n_cols += p + p + n_exo_cov;
  }

  Eigen::MatrixXd Delta = Eigen::MatrixXd::Zero(n_rows, n_cols);
  Eigen::Index row0 = 0;
  Eigen::Index col0 = 0;
  for (Eigen::Index p : block_p) {
    for (Eigen::Index j = 0; j < p; ++j) {
      Delta(row0 + j, col0 + j) = 1.0;
      Delta(row0 + p + vech_index(p, j, j), col0 + p + j) = 1.0;
    }
    Eigen::Index cov_col = col0 + 2 * p;
    for (Eigen::Index a = 0; a < n_exo; ++a) {
      const Eigen::Index ia = exo_idx[static_cast<std::size_t>(a)];
      if (ia < 0 || ia >= p) continue;
      for (Eigen::Index b = a + 1; b < n_exo; ++b) {
        const Eigen::Index ib = exo_idx[static_cast<std::size_t>(b)];
        if (ib >= 0 && ib < p) {
          const Eigen::Index r = std::max(ia, ib);
          const Eigen::Index c = std::min(ia, ib);
          Delta(row0 + p + vech_index(p, r, c), cov_col) = 1.0;
        }
        ++cov_col;
      }
    }
    row0 += p + vech_len(p);
    col0 += 2 * p + n_exo_cov;
  }
  return Delta;
}

std::vector<Eigen::Index> block_p_from_saturated(const SaturatedMoments& sm) {
  std::vector<Eigen::Index> out;
  out.reserve(sm.cov.size());
  for (const Eigen::MatrixXd& S : sm.cov) out.push_back(S.rows());
  return out;
}

post_expected<double>
residual_projector_trace(const Eigen::MatrixXd& V,
                         const Eigen::MatrixXd& G,
                         const Eigen::MatrixXd& Delta,
                         std::string_view label) {
  if (V.rows() != V.cols() || G.rows() != G.cols() ||
      V.rows() != G.rows() || Delta.rows() != V.rows()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": eta-space dimension mismatch"));
  }
  const Eigen::MatrixXd VD = V * Delta;
  Eigen::MatrixXd DtVD = Delta.transpose() * VD;
  DtVD = 0.5 * (DtVD + DtVD.transpose()).eval();
  auto DtVDinv_or = invert_symmetric(DtVD,
                                     std::string(label) + ": DtVD");
  if (!DtVDinv_or.has_value()) return std::unexpected(DtVDinv_or.error());
  Eigen::MatrixXd U = V - VD * (*DtVDinv_or) * VD.transpose();
  U = 0.5 * (U + U.transpose()).eval();
  const double tr = (U * G).trace();
  if (!std::isfinite(tr)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(label) + ": non-finite trace"));
  }
  return tr;
}

post_expected<FIMLCorrectedFitMeasures>
fiml_corrected_fit_measures_impl(spec::LatentStructure pt,
                                 const model::MatrixRep& rep,
                                 const RawData& raw,
                                 const Estimates& est,
                                 int df,
                                 const FIMLPack& pack,
                                 const FIMLH1& h1) {
  if (df <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_corrected_fit_measures: requires df > 0"));
  }
  if (auto e = validate_h1_blocks(pack.cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "FIML H1 moments"));
  }
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "validate_fiml_fixed_x_missing_policy"));
  }
  if (auto e = resolve_fixed_x_from_sample(pt, rep, pack.start_stats);
      !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "resolve_fixed_x_from_sample"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto implied_or = ev_or->evaluate(est.theta, true, true);
  if (!implied_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::evaluate failed: " + implied_or.error().detail));
  }

  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  const SaturatedMoments& sm = *sm_or;

  auto xx3_or = normal_theory_chisq_from_moments(
      implied_or->moments.mu, implied_or->moments.sigma, h1.mu, h1.sigma, sm.n_obs,
      "fiml_corrected_fit_measures: user XX3");
  if (!xx3_or.has_value()) return std::unexpected(xx3_or.error());

  auto spec_or = fiml_ugamma_spectrum_impl(
      pt, rep, raw, est, df, *xx3_or, pack, h1, &sm);
  if (!spec_or.has_value()) return std::unexpected(spec_or.error());

  const std::vector<Eigen::Index> exo_idx = observed_exogenous_indices(pt);
  const std::vector<Eigen::MatrixXd> sigma_null =
      independence_covariances_from_saturated(h1.sigma, exo_idx);
  auto xx3_null_or = normal_theory_chisq_from_moments(
      h1.mu, sigma_null, h1.mu, h1.sigma, sm.n_obs,
      "fiml_corrected_fit_measures: baseline XX3");
  if (!xx3_null_or.has_value()) return std::unexpected(xx3_null_or.error());

  const int df_null = baseline_df_from_block_p(pack.cache.block_p, exo_idx);
  if (df_null <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_corrected_fit_measures: baseline df must be > 0"));
  }
  const Eigen::MatrixXd D_null =
      baseline_eta_delta_from_block_p(pack.cache.block_p, exo_idx);
  auto tr_null_or = residual_projector_trace(
      sm.H, sm.acov, D_null, "fiml_corrected_fit_measures: baseline");
  if (!tr_null_or.has_value()) return std::unexpected(tr_null_or.error());

  FIMLCorrectedFitMeasures out;
  out.xx3 = *xx3_or;
  out.df3 = df;
  out.c_hat3 = spec_or->trace_xcheck / static_cast<double>(df);
  out.xx3_scaled = (out.c_hat3 > 0.0)
      ? out.xx3 / out.c_hat3
      : std::numeric_limits<double>::quiet_NaN();
  out.xx3_null = *xx3_null_or;
  out.df3_null = df_null;
  out.c_hat3_null = *tr_null_or / static_cast<double>(df_null);
  out.xx3_null_scaled = (out.c_hat3_null > 0.0)
      ? out.xx3_null / out.c_hat3_null
      : std::numeric_limits<double>::quiet_NaN();

  measures::RobustFitMeasureInputs inputs;
  inputs.chi2 = out.xx3;
  inputs.df = out.df3;
  inputs.chi2_scaled = out.xx3_scaled;
  inputs.scaling_factor = out.c_hat3;
  inputs.baseline_chi2 = out.xx3_null;
  inputs.baseline_df = out.df3_null;
  inputs.baseline_chi2_scaled = out.xx3_null_scaled;
  inputs.baseline_scaling_factor = out.c_hat3_null;
  inputs.n_total = pack.cache.n_total;
  inputs.n_groups = pack.cache.block_p.size();
  out.indices = measures::robust_fit_measures(inputs);
  return out;
}

}  // namespace

post_expected<FIMLCorrectedFitMeasures>
fiml_corrected_fit_measures(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const RawData& raw,
                            const Estimates& est,
                            int df,
                            FIML discrepancy) {
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 moments"));
  }
  return fiml_corrected_fit_measures_impl(std::move(pt), rep, raw, est, df,
                                          *pack_or, *h1_or);
}

post_expected<FIMLCorrectedFitMeasures>
fiml_corrected_fit_measures(spec::LatentStructure pt,
                            const model::MatrixRep& rep,
                            const RawData& raw,
                            const Estimates& est,
                            int df,
                            const FIMLPack& pack,
                            const FIMLH1& h1) {
  return fiml_corrected_fit_measures_impl(std::move(pt), rep, raw, est, df,
                                          pack, h1);
}

post_expected<Eigen::MatrixXd>
fiml_residual_projector(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const RawData& raw,
                        const Estimates& est,
                        const Eigen::Ref<const Eigen::MatrixXd>& V,
                        FIML discrepancy) {
  (void)discrepancy;
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  return fiml_residual_projector_impl(pt, rep, raw, est, *pack_or, V);
}

namespace {

post_expected<BaselineFit>
fiml_baseline_chi2_impl(const RawData& raw,
                        const std::vector<Eigen::Index>& exo_idx,
                        const FIMLPack& pack,
                        const FIMLH1& h1) {
  const FIMLCache& cache = pack.cache;
  const SampleStats& start_samp = pack.start_stats;

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
  out.chi2 = static_cast<double>(cache.n_total) * (*baseline_or - h1.value);
  if (out.chi2 < 0.0 && out.chi2 > -1e-8) out.chi2 = 0.0;
  for (Eigen::Index p : cache.block_p) {
    out.df += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
    const int px = static_cast<int>(exo_idx.size());
    out.df -= px * (px - 1) / 2;
  }
  return out;
}

post_expected<BaselineFit>
fiml_baseline_chi2_from_raw(const RawData& raw,
                            const std::vector<Eigen::Index>& exo_idx) {
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 likelihood"));
  }
  return fiml_baseline_chi2_impl(raw, exo_idx, *pack_or, *h1_or);
}

}  // namespace

post_expected<BaselineFit>
fiml_baseline_chi2(const RawData& raw,
                   FIML discrepancy) {
  (void)discrepancy;
  return fiml_baseline_chi2_from_raw(raw, {});
}

post_expected<BaselineFit>
fiml_baseline_chi2(const spec::LatentStructure& pt,
                   const RawData& raw,
                   FIML discrepancy) {
  (void)discrepancy;
  return fiml_baseline_chi2_from_raw(raw, observed_exogenous_indices(pt));
}

post_expected<BaselineFit>
fiml_baseline_chi2(const spec::LatentStructure& pt,
                   const RawData& raw,
                   const FIMLPack& pack,
                   const FIMLH1& h1) {
  return fiml_baseline_chi2_impl(raw, observed_exogenous_indices(pt),
                                 pack, h1);
}

namespace {

enum class SaturatedHessianKind { Analytic, FiniteDifference };

post_expected<SaturatedMoments>
saturated_em_moments_impl(const RawData& raw,
                          const FIMLPack& pack,
                          const FIMLH1& h1,
                          double h_step,
                          SaturatedHessianKind hessian_kind,
                          const char* caller) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": h_step must be > 0"));
  }
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(fit_to_post(ok.error(), caller));
  }

  const FIMLCache& cache = pack.cache;
  if (auto e = validate_h1_blocks(cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), caller));
  }

  const std::size_t B = raw.X.size();
  if (B == 0 || cache.block_p.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": empty or inconsistent block layout"));
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
    const Eigen::VectorXd& mu_b = h1.mu[b];
    const Eigen::MatrixXd& Sigma_b = h1.sigma[b];

    auto scores_or = fiml_saturated_scores_block(raw, b, mu_b, Sigma_b);
    if (!scores_or.has_value()) {
      return std::unexpected(scores_or.error());
    }
    auto H_or = (hessian_kind == SaturatedHessianKind::Analytic)
        ? fiml_saturated_hessian_analytic_block(cache, b, mu_b, Sigma_b)
        : fiml_saturated_hessian_fd_block(raw, b, mu_b, Sigma_b, h_step);
    if (!H_or.has_value()) {
      return std::unexpected(H_or.error());
    }

    // The block helpers report scores as deviance gradients (∂(−2logL)/∂η) and
    // H as the Hessian of mean deviance. Convert to log-likelihood scale:
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

    out.mean[b] = mu_b;
    out.cov[b]  = Sigma_b;
    out.n_obs[b] = static_cast<std::int64_t>(raw.X[b].rows());
  }

  auto Hinv_or = invert_symmetric(out.H,
      std::string(caller) + ": aggregated saturated information");
  if (!Hinv_or.has_value()) {
    return std::unexpected(Hinv_or.error());
  }
  const Eigen::MatrixXd Hinv = *Hinv_or;
  out.acov = Hinv * out.J * Hinv;
  out.acov = (0.5 * (out.acov + out.acov.transpose())).eval();
  return out;
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence_impl(const RawData& raw,
                                   const FIMLCache& cache,
                                   const FIMLH1& h1,
                                   const SaturatedMoments& sm,
                                   const char* caller) {
  if (auto ok = validate_raw_shape(raw); !ok.has_value()) {
    return std::unexpected(fit_to_post(ok.error(), caller));
  }
  if (auto e = validate_h1_blocks(cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), caller));
  }

  const std::size_t B = raw.X.size();
  if (B == 0 || cache.block_p.size() != B || sm.mean.size() != B ||
      sm.cov.size() != B || sm.n_obs.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": empty or inconsistent block layout"));
  }

  std::vector<Eigen::Index> q_b(B);
  std::vector<Eigen::Index> off_b(B + 1, 0);
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index n = raw.X[b].rows();
    const Eigen::Index p = cache.block_p[b];
    if (raw.X[b].cols() != p || h1.mu[b].size() != p ||
        h1.sigma[b].rows() != p || h1.sigma[b].cols() != p ||
        sm.mean[b].size() != p || sm.cov[b].rows() != p ||
        sm.cov[b].cols() != p || sm.n_obs[b] != n) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": malformed saturated block " +
              std::to_string(b)));
    }
    q_b[b] = p + detail::vech_len(p);
    off_b[b + 1] = off_b[b] + q_b[b];
    n_total += n;
  }
  const Eigen::Index Q = off_b.back();
  if (sm.H.rows() != Q || sm.H.cols() != Q || sm.acov.rows() != Q ||
      sm.acov.cols() != Q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": saturated information shape mismatch"));
  }

  auto Hinv_or = invert_symmetric(sm.H,
      std::string(caller) + ": aggregated saturated information");
  if (!Hinv_or.has_value()) return std::unexpected(Hinv_or.error());
  const Eigen::MatrixXd& Hinv = *Hinv_or;

  Eigen::MatrixXd influence = Eigen::MatrixXd::Zero(n_total, Q);
  Eigen::Index row_off = 0;
  for (std::size_t b = 0; b < B; ++b) {
    auto scores_or = fiml_saturated_scores_block(raw, b, h1.mu[b], h1.sigma[b]);
    if (!scores_or.has_value()) return std::unexpected(scores_or.error());

    const Eigen::Index n = raw.X[b].rows();
    const Eigen::Index q = q_b[b];
    const Eigen::Index off = off_b[b];
    if (scores_or->rows() != n || scores_or->cols() != q) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": saturated score shape mismatch in block " +
              std::to_string(b)));
    }

    const Eigen::MatrixXd scores_log = -0.5 * (*scores_or);
    influence.middleRows(row_off, n).noalias() =
        scores_log * Hinv.middleRows(off, q);
    row_off += n;
  }
  if (!influence.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": non-finite influence rows"));
  }
  return influence;
}

post_expected<SaturatedMoments>
saturated_em_moments_from_raw(const RawData& raw,
                              double h_step,
                              SaturatedHessianKind hessian_kind,
                              const char* caller) {
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(),
        std::string(caller) + ": fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(),
        std::string(caller) + ": H1 EM"));
  }
  return saturated_em_moments_impl(raw, *pack_or, *h1_or, h_step,
                                   hessian_kind, caller);
}

}  // namespace

post_expected<SaturatedMoments>
saturated_em_moments(const RawData& raw, double h_step) {
  return saturated_em_moments_from_raw(raw, h_step,
                                       SaturatedHessianKind::Analytic,
                                       "saturated_em_moments");
}

post_expected<SaturatedMoments>
saturated_em_moments(const RawData& raw,
                     const FIMLPack& pack,
                     const FIMLH1& h1) {
  return saturated_em_moments_impl(raw, pack, h1, /*h_step=*/1e-4,
                                   SaturatedHessianKind::Analytic,
                                   "saturated_em_moments");
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence(const RawData& raw, double h_step) {
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(),
        "saturated_em_moment_influence: fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(),
        "saturated_em_moment_influence: H1 EM"));
  }
  auto sm_or = saturated_em_moments_impl(
      raw, *pack_or, *h1_or, h_step, SaturatedHessianKind::Analytic,
      "saturated_em_moment_influence");
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  return saturated_em_moment_influence_impl(
      raw, pack_or->cache, *h1_or, *sm_or, "saturated_em_moment_influence");
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence(const RawData& raw,
                              const FIMLPack& pack,
                              const FIMLH1& h1) {
  auto sm_or = saturated_em_moments_impl(
      raw, pack, h1, /*h_step=*/1e-4, SaturatedHessianKind::Analytic,
      "saturated_em_moment_influence");
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  return saturated_em_moment_influence_impl(
      raw, pack.cache, h1, *sm_or, "saturated_em_moment_influence");
}

post_expected<Eigen::MatrixXd>
saturated_em_moment_influence(const RawData& raw,
                              const FIMLPack& pack,
                              const FIMLH1& h1,
                              const SaturatedMoments& sm) {
  return saturated_em_moment_influence_impl(
      raw, pack.cache, h1, sm, "saturated_em_moment_influence");
}

post_expected<data::MixedOrdinalStats>
mixed_ordinal_stats_hybrid_fiml_from_observed_data(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& ordered,
    bool full_wls_weight,
    double h_step) {
  auto stats_or =
      data::mixed_ordinal_stats_from_observed_data(X, ordered, full_wls_weight);
  if (!stats_or.has_value()) return std::unexpected(stats_or.error());
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_hybrid_fiml_from_observed_data: h_step must be > 0"));
  }

  data::MixedOrdinalStats stats = std::move(*stats_or);
  stats.gamma_diag_influence.clear();
  stats.gamma_full_influence.clear();
  stats.gamma_diag_influence.reserve(X.size());
  stats.gamma_full_influence.reserve(X.size());

  for (std::size_t b = 0; b < X.size(); ++b) {
    const Eigen::MatrixXd& Xb = X[b];
    const Eigen::Index n = Xb.rows();
    const Eigen::Index p = Xb.cols();
    std::vector<Eigen::Index> cont;
    cont.reserve(static_cast<std::size_t>(p));
    std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        cont_pos[static_cast<std::size_t>(j)] =
            static_cast<Eigen::Index>(cont.size());
        cont.push_back(j);
      }
    }
    const Eigen::Index q = static_cast<Eigen::Index>(cont.size());
    if (q == 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: block " +
              std::to_string(b) + " has no continuous variables"));
    }

    std::vector<Eigen::Index> keep_rows;
    keep_rows.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index r = 0; r < n; ++r) {
      bool any = false;
      for (Eigen::Index k = 0; k < q; ++k) {
        any = any || std::isfinite(Xb(r, cont[static_cast<std::size_t>(k)]));
      }
      if (any) keep_rows.push_back(r);
    }
    if (keep_rows.size() < 2) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: block " +
              std::to_string(b) +
              " has fewer than 2 rows with continuous observations"));
    }

    RawData raw_cont;
    const Eigen::Index nf = static_cast<Eigen::Index>(keep_rows.size());
    Eigen::MatrixXd Xc(nf, q);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask(nf, q);
    for (Eigen::Index rr = 0; rr < nf; ++rr) {
      const Eigen::Index src = keep_rows[static_cast<std::size_t>(rr)];
      for (Eigen::Index k = 0; k < q; ++k) {
        const double v = Xb(src, cont[static_cast<std::size_t>(k)]);
        const bool obs = std::isfinite(v);
        Xc(rr, k) = obs ? v : std::numeric_limits<double>::quiet_NaN();
        mask(rr, k) = obs ? 1 : 0;
      }
    }
    raw_cont.X.push_back(std::move(Xc));
    raw_cont.mask.push_back(std::move(mask));

    auto pack_or = fiml_pack(raw_cont);
    if (!pack_or.has_value()) {
      return std::unexpected(fit_to_post(pack_or.error(),
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: fiml_pack"));
    }
    auto h1_or = fiml_h1_moments(raw_cont, *pack_or);
    if (!h1_or.has_value()) {
      return std::unexpected(fit_to_post(h1_or.error(),
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: H1 EM"));
    }
    auto sm_or = saturated_em_moments_impl(
        raw_cont, *pack_or, *h1_or, h_step, SaturatedHessianKind::Analytic,
        "mixed_ordinal_stats_hybrid_fiml_from_observed_data");
    if (!sm_or.has_value()) return std::unexpected(sm_or.error());
    auto infl_or = saturated_em_moment_influence(
        raw_cont, *pack_or, *h1_or, *sm_or);
    if (!infl_or.has_value()) return std::unexpected(infl_or.error());
    if (sm_or->mean.size() != 1 || sm_or->cov.size() != 1 ||
        sm_or->mean[0].size() != q || sm_or->cov[0].rows() != q ||
        sm_or->cov[0].cols() != q) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: malformed "
          "continuous saturated moments"));
    }

    const Eigen::Index qmom = q + detail::vech_len(q);
    if (infl_or->rows() != nf || infl_or->cols() != qmom) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: continuous "
          "influence shape mismatch"));
    }
    Eigen::MatrixXd cont_if = Eigen::MatrixXd::Zero(n, qmom);
    for (Eigen::Index rr = 0; rr < nf; ++rr) {
      const Eigen::Index src = keep_rows[static_cast<std::size_t>(rr)];
      cont_if.row(src) = static_cast<double>(n) * infl_or->row(rr);
    }

    Eigen::MatrixXd R_old = stats.R[b];
    Eigen::MatrixXd IF_old = stats.moment_influence[b];
    Eigen::MatrixXd& R = stats.R[b];
    Eigen::VectorXd& mean = stats.mean[b];
    Eigen::VectorXd& moments = stats.moments[b];
    Eigen::MatrixXd& G = stats.moment_influence[b];
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index mb = moments.size();
    if (G.rows() != n || G.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_hybrid_fiml_from_observed_data: mixed "
          "influence shape mismatch"));
    }

    std::vector<Eigen::Index> mean_col(static_cast<std::size_t>(p), -1);
    std::vector<Eigen::Index> var_col(static_cast<std::size_t>(p), -1);
    Eigen::Index pos = nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (cont_pos[static_cast<std::size_t>(j)] >= 0) mean_col[static_cast<std::size_t>(j)] = pos++;
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (cont_pos[static_cast<std::size_t>(j)] >= 0) var_col[static_cast<std::size_t>(j)] = pos++;
    }
    const Eigen::Index assoc_start = pos;

    for (Eigen::Index j = 0; j < p; ++j) {
      const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
      if (cp < 0) continue;
      const Eigen::Index mc = mean_col[static_cast<std::size_t>(j)];
      const Eigen::Index vc = var_col[static_cast<std::size_t>(j)];
      mean(j) = sm_or->mean[0](cp);
      R(j, j) = sm_or->cov[0](cp, cp);
      moments(mc) = -mean(j);
      moments(vc) = R(j, j);
      G.col(mc) = -cont_if.col(cp);
      G.col(vc) = cont_if.col(q + detail::vech_index(q, cp, cp));
    }

    Eigen::Index assoc = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const Eigen::Index col = assoc_start + assoc;
        const Eigen::Index ci = cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index cj = cont_pos[static_cast<std::size_t>(j)];
        if (ci >= 0 && cj >= 0) {
          R(i, j) = R(j, i) = sm_or->cov[0](ci, cj);
          moments(col) = R(i, j);
          G.col(col) = cont_if.col(q + detail::vech_index(q, ci, cj));
        } else if (ci >= 0 || cj >= 0) {
          const Eigen::Index c = ci >= 0 ? i : j;
          const Eigen::Index cp = ci >= 0 ? ci : cj;
          const double old_var = R_old(c, c);
          const double new_var = R(c, c);
          if (!(old_var > 0.0) || !(new_var > 0.0)) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_stats_hybrid_fiml_from_observed_data: "
                "non-positive continuous variance"));
          }
          const double old_sd = std::sqrt(old_var);
          const double new_sd = std::sqrt(new_var);
          const double rho = R_old(i, j) / old_sd;
          const Eigen::Index vc = var_col[static_cast<std::size_t>(c)];
          const Eigen::VectorXd rho_if =
              (IF_old.col(col) - (rho / (2.0 * old_sd)) * IF_old.col(vc)) /
              old_sd;
          R(i, j) = R(j, i) = rho * new_sd;
          moments(col) = R(i, j);
          G.col(col) = new_sd * rho_if +
              (rho / (2.0 * new_sd)) *
                  cont_if.col(q + detail::vech_index(q, cp, cp));
        }
        ++assoc;
      }
    }

    Eigen::MatrixXd NACOV = (G.transpose() * G) / static_cast<double>(n);
    NACOV = 0.5 * (NACOV + NACOV.transpose()).eval();
    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mb, mb);
    for (Eigen::Index k = 0; k < mb; ++k) {
      const double v = NACOV(k, k);
      if (!(v > 0.0) || !std::isfinite(v)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_hybrid_fiml_from_observed_data: non-positive "
            "hybrid NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    Eigen::MatrixXd W_wls;
    if (full_wls_weight) {
      Eigen::LLT<Eigen::MatrixXd> llt(NACOV);
      if (llt.info() == Eigen::Success) {
        W_wls = llt.solve(Eigen::MatrixXd::Identity(mb, mb));
        W_wls = 0.5 * (W_wls + W_wls.transpose()).eval();
      }
    }

    Eigen::MatrixXd gamma_diag_if(n, mb);
    Eigen::MatrixXd gamma_full_if(n, mb * mb);
    const Eigen::VectorXd diag = NACOV.diagonal();
    for (Eigen::Index r = 0; r < n; ++r) {
      const Eigen::VectorXd gr = G.row(r).transpose();
      gamma_diag_if.row(r) = (gr.array().square() - diag.array()).matrix();
      const Eigen::MatrixXd outer = gr * gr.transpose() - NACOV;
      for (Eigen::Index cc = 0; cc < mb; ++cc) {
        for (Eigen::Index rr = 0; rr < mb; ++rr) {
          gamma_full_if(r, rr + cc * mb) = outer(rr, cc);
        }
      }
    }

    stats.NACOV[b] = std::move(NACOV);
    stats.W_dwls[b] = std::move(W_dwls);
    stats.W_wls[b] = std::move(W_wls);
    stats.gamma_diag_influence.push_back(std::move(gamma_diag_if));
    stats.gamma_full_influence.push_back(std::move(gamma_full_if));
  }
  return stats;
}

namespace {

SampleStats
sample_stats_from_saturated(const SaturatedMoments& sm) {
  SampleStats samp;
  samp.S     = sm.cov;
  samp.mean  = sm.mean;
  samp.n_obs = sm.n_obs;
  return samp;
}

bool ml2s_has_mean_rows(const SampleStats& samp,
                        const model::ImpliedMoments& moments) {
  for (std::size_t b = 0; b < moments.mu.size(); ++b) {
    if (moments.mu[b].size() > 0 && b < samp.mean.size() &&
        samp.mean[b].size() > 0) {
      return true;
    }
  }
  return false;
}

struct Ml2sMomentLayout {
  bool has_means = false;
  std::vector<Eigen::Index> block_rows;
  std::vector<Eigen::Index> mu_offsets;
  std::vector<Eigen::Index> sigma_offsets;
  Eigen::Index total_mu_rows = 0;
  Eigen::Index total_sigma_rows = 0;
};

post_expected<void>
ml2s_validate_moment_shapes(const SampleStats& samp,
                            const model::ImpliedMoments& moments) {
  if (samp.S.size() != moments.sigma.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: SampleStats and ImpliedMoments block "
        "counts differ"));
  }
  if (samp.n_obs.size() != samp.S.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: n_obs block count does not match sample "
        "covariances"));
  }
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    if (samp.n_obs[b] <= 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: non-positive n_obs in block " +
              std::to_string(b)));
    }
    const auto& S = samp.S[b];
    const auto& Sigma = moments.sigma[b];
    if (S.rows() != S.cols() || Sigma.rows() != Sigma.cols() ||
        S.rows() != Sigma.rows()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: covariance shape mismatch in block " +
              std::to_string(b)));
    }
    if (b < samp.mean.size() && b < moments.mu.size() &&
        samp.mean[b].size() > 0 && moments.mu[b].size() > 0 &&
        samp.mean[b].size() != moments.mu[b].size()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: mean shape mismatch in block " +
              std::to_string(b)));
    }
  }
  return {};
}

Ml2sMomentLayout
ml2s_make_layout(const SampleStats& samp,
                 const model::ImpliedMoments& moments) {
  Ml2sMomentLayout layout;
  layout.has_means = ml2s_has_mean_rows(samp, moments);
  layout.block_rows.resize(moments.sigma.size());
  layout.mu_offsets.resize(moments.sigma.size());
  layout.sigma_offsets.resize(moments.sigma.size());
  for (std::size_t b = 0; b < moments.sigma.size(); ++b) {
    const Eigen::Index p = moments.sigma[b].rows();
    layout.mu_offsets[b] = layout.total_mu_rows;
    if (layout.has_means) layout.total_mu_rows += p;
    layout.sigma_offsets[b] = layout.total_sigma_rows;
    layout.total_sigma_rows += vech_len(p);
    layout.block_rows[b] = (layout.has_means ? p : 0) + vech_len(p);
  }
  return layout;
}

post_expected<Eigen::MatrixXd>
ml2s_moment_jacobian_block(const Ml2sMomentLayout& layout,
                           const model::ImpliedMoments& moments,
                           const Eigen::MatrixXd& J_sigma,
                           const Eigen::MatrixXd& J_mu,
                           std::size_t b) {
  const Eigen::Index p = moments.sigma[b].rows();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index n_free = J_sigma.cols();
  if (J_sigma.rows() != layout.total_sigma_rows) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: J_sigma row count does not match moment "
        "layout"));
  }
  if (layout.has_means) {
    if (J_mu.rows() != layout.total_mu_rows || J_mu.cols() != n_free) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: J_mu shape does not match moment "
          "layout"));
    }
  }

  Eigen::MatrixXd Jb(layout.block_rows[b], n_free);
  Eigen::Index out = 0;
  if (layout.has_means) {
    Jb.topRows(p).setZero();
    if (b < moments.mu.size() && moments.mu[b].size() > 0) {
      Jb.topRows(p) = J_mu.block(layout.mu_offsets[b], 0, p, n_free);
    }
    out = p;
  }
  Jb.block(out, 0, pstar, n_free) =
      J_sigma.block(layout.sigma_offsets[b], 0, pstar, n_free);
  return Jb;
}

post_expected<Eigen::MatrixXd>
ml2s_weight_block(const gmm::Weight& weight,
                  const Ml2sMomentLayout& layout,
                  std::size_t b) {
  if (weight.size() <= b) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: missing Stage-2 weight block " +
            std::to_string(b)));
  }
  const auto& W = weight[b];
  if (W.rows() != layout.block_rows[b] || W.cols() != layout.block_rows[b]) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: Stage-2 weight dimension mismatch in "
        "block " + std::to_string(b)));
  }
  return W;
}

Eigen::VectorXd
ml2s_block_residual(const SampleStats& samp,
                    const model::ImpliedMoments& moments,
                    const Ml2sMomentLayout& layout,
                    std::size_t b) {
  const Eigen::Index p = moments.sigma[b].rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::VectorXd d(layout.block_rows[b]);
  Eigen::Index off = 0;
  if (layout.has_means) {
    const bool have_mu =
        b < moments.mu.size() && moments.mu[b].size() == p;
    const Eigen::VectorXd mu_model =
        have_mu ? moments.mu[b] : Eigen::VectorXd::Zero(p);
    const Eigen::VectorXd mean_s =
        (b < samp.mean.size() && samp.mean[b].size() == p)
            ? samp.mean[b]
            : Eigen::VectorXd::Zero(p);
    d.head(p) = mu_model - mean_s;
    off = p;
  }
  d.segment(off, pstar) = vech_lower(moments.sigma[b] - samp.S[b]);
  return d;
}

post_expected<double>
ml2s_total_n(const SampleStats& samp) {
  double n = 0.0;
  for (auto nb : samp.n_obs) n += static_cast<double>(nb);
  if (!(n > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: non-positive total n_obs"));
  }
  return n;
}

post_expected<Eigen::VectorXd>
ml2s_gradient(const model::ModelEvaluator& ev,
              const SampleStats& samp,
              const gmm::Weight& weight,
              double N_total,
              const Eigen::VectorXd& theta) {
  auto eval = ev.evaluate(theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: ModelEvaluator::evaluate failed: " +
            eval.error().detail));
  }
  if (auto v = ml2s_validate_moment_shapes(samp, eval->moments);
      !v.has_value()) {
    return std::unexpected(v.error());
  }
  const auto layout = ml2s_make_layout(samp, eval->moments);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(theta.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    auto Jb = ml2s_moment_jacobian_block(layout, eval->moments,
                                         eval->J_sigma, eval->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = ml2s_weight_block(weight, layout, b);
    if (!Wb.has_value()) return std::unexpected(Wb.error());
    const Eigen::VectorXd d =
        ml2s_block_residual(samp, eval->moments, layout, b);
    const double w_b = static_cast<double>(samp.n_obs[b]) / N_total;
    g.noalias() += w_b * (Jb->transpose() * (*Wb) * d);
  }
  if (!g.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: non-finite LS gradient"));
  }
  return g;
}

post_expected<Eigen::MatrixXd>
ml2s_observed_bread(const spec::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    const SampleStats& samp,
                    const Estimates& est,
                    const gmm::Weight& weight,
                    const Eigen::MatrixXd& K) {
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto n_or = ml2s_total_n(samp);
  if (!n_or.has_value()) return std::unexpected(n_or.error());
  const model::ModelEvaluator ev = std::move(*ev_or);
  auto grad_at = [&](const Eigen::VectorXd& theta)
      -> post_expected<Eigen::VectorXd> {
    return ml2s_gradient(ev, samp, weight, *n_or, theta);
  };
  return estimate::observed_moment_bread_fd(grad_at, est.theta, K);
}

Eigen::MatrixXd
ml2s_gamma_nt_directional(const Eigen::MatrixXd& S,
                          const Eigen::MatrixXd& H) {
  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(pstar, pstar);
  for (Eigen::Index c1 = 0; c1 < p; ++c1) {
    for (Eigen::Index r1 = c1; r1 < p; ++r1) {
      const Eigen::Index k1 = vech_index(p, r1, c1);
      for (Eigen::Index c2 = 0; c2 < p; ++c2) {
        for (Eigen::Index r2 = c2; r2 < p; ++r2) {
          const Eigen::Index k2 = vech_index(p, r2, c2);
          out(k1, k2) =
              H(r1, r2) * S(c1, c2) + S(r1, r2) * H(c1, c2) +
              H(r1, c2) * S(c1, r2) + S(r1, c2) * H(c1, r2);
        }
      }
    }
  }
  return out;
}

post_expected<Eigen::MatrixXd>
ml2s_nt_gamma_influence(const SampleStats& samp,
                        const Eigen::RowVectorXd& moment_row,
                        const Ml2sMomentLayout& layout,
                        std::size_t b) {
  const Eigen::Index p = samp.S[b].rows();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index mb = layout.block_rows[b];
  const Eigen::Index cov_off = layout.has_means ? p : 0;
  if (moment_row.size() != mb || cov_off + pstar != mb) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: NT Gamma influence shape mismatch in "
        "block " + std::to_string(b)));
  }

  Eigen::MatrixXd dS(p, p);
  vech_unpack(moment_row.segment(cov_off, pstar).transpose(), p, dS);
  Eigen::MatrixXd dG = Eigen::MatrixXd::Zero(mb, mb);
  if (layout.has_means) dG.topLeftCorner(p, p) = dS;
  dG.block(cov_off, cov_off, pstar, pstar) =
      ml2s_gamma_nt_directional(samp.S[b], dS);
  return dG;
}

post_expected<Eigen::MatrixXd>
ml2s_weight_correction_block(const RawData& raw,
                             const FIMLCache& cache,
                             const SampleStats& samp,
                             const Eigen::VectorXd& residual,
                             const Eigen::MatrixXd& moment_influence,
                             const Eigen::MatrixXd& weight,
                             const Ml2sMomentLayout& layout,
                             std::size_t b,
                             TwoStageWeight kind,
                             TwoStageDlsOptions dls) {
  const Eigen::Index mb = layout.block_rows[b];
  if (residual.size() != mb || moment_influence.cols() != mb ||
      moment_influence.rows() != raw.X[b].rows() ||
      weight.rows() != mb || weight.cols() != mb) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: ML2S weight correction shape mismatch in "
        "block " + std::to_string(b)));
  }
  if (kind == TwoStageWeight::Dls && !(dls.a >= 0.0 && dls.a <= 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: DLS mixing scalar a must lie in [0, 1]"));
  }

  // The data-dependent weights (DWLS/ADF/DLS with a>0) carry the estimated-weight
  // channel through the analytic Stage-1 sandwich-Γ influence; build its
  // per-block precompute once. ULS is a fixed identity weight.
  const bool needs_gamma = kind == TwoStageWeight::Dwls ||
                           kind == TwoStageWeight::Adf ||
                           (kind == TwoStageWeight::Dls && dls.a > 0.0);
  Ml2sGammaInfluencePrep prep;
  if (needs_gamma) {
    if (b >= samp.mean.size() || samp.mean[b].size() != raw.X[b].cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: estimated-weight ML2S Gamma influence "
          "requires saturated means"));
    }
    auto prep_or =
        prep_ml2s_gamma_influence(raw, cache, b, samp.mean[b], samp.S[b]);
    if (!prep_or.has_value()) return std::unexpected(prep_or.error());
    prep = std::move(*prep_or);
  }

  const Eigen::RowVectorXd lhs = residual.transpose() * weight;
  Eigen::MatrixXd correction(moment_influence.rows(), mb);
  for (Eigen::Index i = 0; i < moment_influence.rows(); ++i) {
    Eigen::MatrixXd dG = Eigen::MatrixXd::Zero(mb, mb);
    if (needs_gamma) {
      auto gf_or = weighted_two_stage_gamma_influence_analytic_block(
          raw, cache, b, prep, i);
      if (!gf_or.has_value()) return std::unexpected(gf_or.error());
      if (kind == TwoStageWeight::Dwls) {
        dG.diagonal() = gf_or->diagonal();
      } else if (kind == TwoStageWeight::Adf) {
        dG = std::move(*gf_or);
      } else {
        dG.noalias() += dls.a * (*gf_or);
      }
    }
    if (kind == TwoStageWeight::Dls && dls.a < 1.0) {
      auto nt_or = ml2s_nt_gamma_influence(
          samp, moment_influence.row(i), layout, b);
      if (!nt_or.has_value()) return std::unexpected(nt_or.error());
      dG.noalias() += (1.0 - dls.a) * (*nt_or);
    }
    correction.row(i) = lhs * dG * weight;
  }
  return correction;
}

}  // namespace

post_expected<Eigen::MatrixXd>
two_stage_gamma_from_acov(const SaturatedMoments& sm, bool se_weighted) {
  double N = 0.0;
  Eigen::Index Q = 0;
  for (std::size_t b = 0; b < sm.cov.size(); ++b) {
    if (b >= sm.n_obs.size()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: saturated moment block count mismatch"));
    }
    const Eigen::Index p = sm.cov[b].rows();
    if (sm.cov[b].cols() != p || sm.mean[b].size() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: malformed saturated moments in block " +
              std::to_string(b)));
    }
    if (sm.n_obs[b] <= 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: non-positive block sample size"));
    }
    Q += p + detail::vech_len(p);
    N += static_cast<double>(sm.n_obs[b]);
  }
  if (sm.acov.rows() != Q || sm.acov.cols() != Q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: saturated ACOV shape mismatch"));
  }
  if (!(N > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: total sample size must be positive"));
  }

  Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(Q, Q);
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < sm.cov.size(); ++b) {
    const Eigen::Index p = sm.cov[b].rows();
    const Eigen::Index q = p + detail::vech_len(p);
    const double n = static_cast<double>(sm.n_obs[b]);
    const double scale = se_weighted ? (n * n / N) : n;
    gamma.block(off, off, q, q) = scale * sm.acov.block(off, off, q, q);
    off += q;
  }
  return Eigen::MatrixXd(0.5 * (gamma + gamma.transpose()).eval());
}

post_expected<Eigen::MatrixXd>
two_stage_saturated_gamma_influence(const RawData& raw, std::size_t block,
                                    Eigen::Index row,
                                    GammaInfluenceRegime regime) {
  if (block >= raw.X.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_saturated_gamma_influence: block index out of range"));
  }
  if (row < 0 || row >= raw.X[block].rows()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_saturated_gamma_influence: row index out of range"));
  }
  if (regime == GammaInfluenceRegime::FiniteDifference) {
    return weighted_two_stage_gamma_influence_fd_block(raw, block, row);
  }
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(),
        "two_stage_saturated_gamma_influence: fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(),
        "two_stage_saturated_gamma_influence: H1 EM"));
  }
  auto prep_or = prep_ml2s_gamma_influence(raw, pack_or->cache, block,
                                           h1_or->mu[block], h1_or->sigma[block]);
  if (!prep_or.has_value()) return std::unexpected(prep_or.error());
  return weighted_two_stage_gamma_influence_analytic_block(
      raw, pack_or->cache, block, *prep_or, row);
}

namespace {

post_expected<Eigen::MatrixXd>
sym_pd_inverse(const Eigen::MatrixXd& A, std::string what) {
  const Eigen::MatrixXd S = 0.5 * (A + A.transpose());
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        "two_stage_stage2_weight: " + what + " is not invertible"));
  }
  const Eigen::MatrixXd inv =
      ldlt.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  return Eigen::MatrixXd(0.5 * (inv + inv.transpose()).eval());
}

robust::Information robust_bread(TwoStageBread bread) {
  return bread == TwoStageBread::Observed ? robust::Information::Observed
                                          : robust::Information::Expected;
}

bool raw_has_no_missing_mask(const RawData& raw) {
  return raw.mask.empty();
}

}  // namespace

post_expected<std::vector<Eigen::MatrixXd>>
two_stage_stage2_weight_blocks(const SaturatedMoments& sm, TwoStageWeight kind,
                               TwoStageDlsOptions dls) {
  if (kind == TwoStageWeight::Dls && !(dls.a >= 0.0 && dls.a <= 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_stage2_weight: DLS mixing scalar a must lie in [0, 1]"));
  }

  // Γ_FIML in the n-scaled per-unit (test-statistic) convention, block-diagonal
  // over [mean ; vech(cov)]. The meat for ADF/DLS, and (its diagonal) for DWLS.
  auto gamma_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_or.has_value()) return std::unexpected(gamma_or.error());
  const Eigen::MatrixXd& Gfiml = *gamma_or;

  std::vector<Eigen::MatrixXd> blocks;
  blocks.reserve(sm.cov.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < sm.cov.size(); ++b) {
    const Eigen::Index p = sm.cov[b].rows();
    const Eigen::Index ps = vech_len(p);
    const Eigen::Index q = p + ps;
    const Eigen::MatrixXd& Sigma = sm.cov[b];

    // NT covariance-moment ACOV; the NT mean-moment ACOV is Σ itself.
    auto gnt_or = data::gamma_nt(Sigma);
    if (!gnt_or.has_value()) return std::unexpected(gnt_or.error());

    const Eigen::MatrixXd Gf = Gfiml.block(off, off, q, q);

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(q, q);
    switch (kind) {
      case TwoStageWeight::Nt: {
        // V = blockdiag(Σ⁻¹, gamma_nt(Σ)⁻¹), the lavaan robust.two.stage weight.
        auto sinv = sym_pd_inverse(Sigma, "saturated covariance block");
        if (!sinv.has_value()) return std::unexpected(sinv.error());
        auto cinv = sym_pd_inverse(*gnt_or, "covariance NT Gamma block");
        if (!cinv.has_value()) return std::unexpected(cinv.error());
        W.topLeftCorner(p, p) = *sinv;
        W.bottomRightCorner(ps, ps) = *cinv;
        break;
      }
      case TwoStageWeight::Uls: {
        W.setIdentity();
        break;
      }
      case TwoStageWeight::Adf: {
        auto winv = sym_pd_inverse(Gf, "Gamma_FIML block");
        if (!winv.has_value()) return std::unexpected(winv.error());
        W = std::move(*winv);
        break;
      }
      case TwoStageWeight::Dwls: {
        const Eigen::VectorXd d = Gf.diagonal();
        for (Eigen::Index i = 0; i < q; ++i) {
          if (!(d(i) > 0.0)) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "two_stage_stage2_weight: non-positive Gamma_FIML diagonal; "
                "DWLS weight undefined"));
          }
          W(i, i) = 1.0 / d(i);
        }
        break;
      }
      case TwoStageWeight::Dls: {
        // Browne mix of the two moment ACOVs over the full block, then invert:
        // a = 0 recovers Nt (Γ_NT⁻¹), a = 1 recovers Adf (Γ_FIML⁻¹).
        Eigen::MatrixXd Gnt = Eigen::MatrixXd::Zero(q, q);
        Gnt.topLeftCorner(p, p) = Sigma;
        Gnt.bottomRightCorner(ps, ps) = *gnt_or;
        const Eigen::MatrixXd Gmix = (1.0 - dls.a) * Gnt + dls.a * Gf;
        auto winv = sym_pd_inverse(Gmix, "DLS-mixed Gamma block");
        if (!winv.has_value()) return std::unexpected(winv.error());
        W = std::move(*winv);
        break;
      }
    }
    blocks.push_back(std::move(W));
    off += q;
  }
  return blocks;
}

post_expected<Eigen::MatrixXd>
two_stage_stage2_weight(const SaturatedMoments& sm, TwoStageWeight kind,
                        TwoStageDlsOptions dls) {
  auto blocks_or = two_stage_stage2_weight_blocks(sm, kind, dls);
  if (!blocks_or.has_value()) return std::unexpected(blocks_or.error());
  const auto& blocks = *blocks_or;
  Eigen::Index Q = 0;
  for (const auto& W : blocks) Q += W.rows();
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(Q, Q);
  Eigen::Index off = 0;
  for (const auto& W : blocks) {
    V.block(off, off, W.rows(), W.cols()) = W;
    off += W.rows();
  }
  return Eigen::MatrixXd(0.5 * (V + V.transpose()).eval());
}

namespace {

// Non-NT Stage-2 weighted two-stage inference. The DWLS / ADF / DLS weight is
// not the normal-theory weight that build_u_factor bakes in, so the robust
// sandwich runs through the explicit-weight moment-quadratic path. `est` must
// minimize ½ rᵀ W r for the SAME weight (the caller fits Stage 2 with
// `two_stage_stage2_weight_blocks(sm, kind, dls)`); the meat is the per-block
// Γ_FIML in the n-scaled test-statistic convention.
post_expected<TwoStageEMMLInference>
two_stage_em_weighted_inference_from_sm(spec::LatentStructure pt,
                                        const model::MatrixRep& rep,
                                        const Estimates& est,
                                        const SaturatedMoments& sm,
                                        TwoStageWeight kind,
                                        TwoStageDlsOptions dls,
                                        TwoStageBread bread) {
  SampleStats samp = sample_stats_from_saturated(sm);

  auto weight_or = two_stage_stage2_weight_blocks(sm, kind, dls);
  if (!weight_or.has_value()) return std::unexpected(weight_or.error());

  auto gamma_full_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_full_or.has_value()) return std::unexpected(gamma_full_or.error());
  std::vector<Eigen::MatrixXd> gamma;
  gamma.reserve(sm.cov.size());
  Eigen::Index goff = 0;
  for (std::size_t b = 0; b < sm.cov.size(); ++b) {
    const Eigen::Index q = sm.cov[b].rows() + vech_len(sm.cov[b].rows());
    gamma.push_back(gamma_full_or->block(goff, goff, q, q));
    goff += q;
  }

  auto rr_or = robust_continuous_ls(std::move(pt), rep, samp, est,
                                    *weight_or, gamma, robust_bread(bread));
  if (!rr_or.has_value()) return std::unexpected(rr_or.error());

  TwoStageEMMLInference out;
  out.vcov = std::move(rr_or->vcov);
  out.se = std::move(rr_or->se);
  out.eigvals = std::move(rr_or->eigvals);
  out.df = rr_or->df;
  out.chisq = rr_or->chisq_standard;
  out.scaling_factor = rr_or->satorra_bentler.scale_c;
  out.chisq_scaled = rr_or->satorra_bentler.chi2_scaled;
  out.trace_ugamma = (out.eigvals.size() > 0) ? out.eigvals.sum() : 0.0;
  out.ntotal = 0;
  for (auto n : samp.n_obs) out.ntotal += n;
  return out;
}

post_expected<TwoStageEMMLInference>
two_stage_complete_data_weighted_ij_from_sm(spec::LatentStructure pt,
                                            const model::MatrixRep& rep,
                                            const RawData& raw,
                                            const Estimates& est,
                                            const SaturatedMoments& sm,
                                            TwoStageWeight kind,
                                            TwoStageDlsOptions dls,
                                            TwoStageBread bread) {
  if (bread != TwoStageBread::Observed || !raw_has_no_missing_mask(raw) ||
      kind == TwoStageWeight::Nt || kind == TwoStageWeight::Uls) {
    return two_stage_em_weighted_inference_from_sm(
        std::move(pt), rep, est, sm, kind, dls, bread);
  }

  auto base_or = two_stage_em_weighted_inference_from_sm(
      pt, rep, est, sm, kind, dls, bread);
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  SampleStats samp = sample_stats_from_saturated(sm);
  post_expected<WeightedRobustResult> ij_or =
      std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: unsupported Stage-2 weight"));
  switch (kind) {
    case TwoStageWeight::Dwls:
      ij_or = estimate::robust_continuous_ls_dwls_ij(
          std::move(pt), rep, samp, est, raw);
      break;
    case TwoStageWeight::Adf:
      ij_or = estimate::robust_continuous_ls_wls_ij(
          std::move(pt), rep, samp, est, raw);
      break;
    case TwoStageWeight::Dls: {
      frontier::DlsWeightOptions opts;
      opts.a = dls.a;
      ij_or = estimate::robust_continuous_ls_dls_ij(
          std::move(pt), rep, samp, est, raw, opts);
      break;
    }
    case TwoStageWeight::Uls:
    case TwoStageWeight::Nt:
      break;
  }
  if (!ij_or.has_value()) return std::unexpected(ij_or.error());

  TwoStageEMMLInference out = std::move(*base_or);
  out.vcov = std::move(ij_or->vcov);
  out.se = std::move(ij_or->se);
  return out;
}

struct Ml2sIjAssembly {
  std::vector<WeightedMomentIJBlock> blocks;
  Eigen::MatrixXd K;
  Eigen::MatrixXd observed_bread;
};

// Per-case ML2S infinitesimal-jackknife blocks plus K and the observed bread,
// shared by the missing-data SE sandwich and the casewise-influence accessor so
// both decompose the SAME complete-sandwich meat. Each block's moment_influence
// is the Stage-1 saturated-moment per-case influence (`saturated_em_moment_influence`,
// which reduces to centered mean/covariance rows under complete data); the
// weight_correction is the data-dependent-weight `IF(Ŵ)` term
// (`ml2s_weight_correction_block`, zero for the NT weight, which lavaan
// robust.two.stage treats as fixed). Always the observed bread.
post_expected<Ml2sIjAssembly>
build_ml2s_ij_blocks(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const RawData& raw,
                     const Estimates& est,
                     const FIMLPack& pack,
                     const FIMLH1& h1,
                     const SaturatedMoments& sm,
                     TwoStageWeight kind,
                     TwoStageDlsOptions dls) {
  SampleStats samp = sample_stats_from_saturated(sm);
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(),
        "two_stage_em_ml_inference: fixed.x resolution"));
  }
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  Eigen::MatrixXd K = con_or->K();
  if (K.rows() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: constraint reparameterization has "
        "incompatible shape"));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: fitted theta length does not match "
        "partable"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval_or = ev_or->evaluate(est.theta, true, true);
  if (!eval_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: ModelEvaluator::evaluate failed: " +
            eval_or.error().detail));
  }
  if (auto v = ml2s_validate_moment_shapes(samp, eval_or->moments);
      !v.has_value()) {
    return std::unexpected(v.error());
  }
  const Ml2sMomentLayout layout =
      ml2s_make_layout(samp, eval_or->moments);

  auto weight_or = two_stage_stage2_weight_blocks(sm, kind, dls);
  if (!weight_or.has_value()) return std::unexpected(weight_or.error());
  auto influence_or = saturated_em_moment_influence(raw, pack, h1, sm);
  if (!influence_or.has_value()) return std::unexpected(influence_or.error());

  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(samp.S.size());
  Eigen::Index row_off = 0;
  Eigen::Index col_off = 0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p = samp.S[b].rows();
    const Eigen::Index qb = p + vech_len(p);
    const Eigen::Index n = raw.X[b].rows();
    if (layout.block_rows[b] != qb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: ML2S estimated-weight IJ requires "
          "mean and covariance moment rows"));
    }
    if (row_off + n > influence_or->rows() ||
        col_off + qb > influence_or->cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_em_ml_inference: saturated influence shape mismatch"));
    }

    auto Jb = ml2s_moment_jacobian_block(
        layout, eval_or->moments, eval_or->J_sigma, eval_or->J_mu, b);
    if (!Jb.has_value()) return std::unexpected(Jb.error());
    auto Wb = ml2s_weight_block(*weight_or, layout, b);
    if (!Wb.has_value()) return std::unexpected(Wb.error());
    const Eigen::VectorXd residual =
        ml2s_block_residual(samp, eval_or->moments, layout, b);
    Eigen::MatrixXd moment_rows =
        static_cast<double>(n) *
        influence_or->block(row_off, col_off, n, qb);
    auto correction_or = ml2s_weight_correction_block(
        raw, pack.cache, samp, residual, moment_rows, *Wb, layout, b, kind,
        dls);
    if (!correction_or.has_value()) return std::unexpected(correction_or.error());

    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = std::move(*Jb),
        .weight = std::move(*Wb),
        .moment_influence = std::move(moment_rows),
        .weight_correction = std::move(*correction_or),
        .n_obs = samp.n_obs[b]});
    row_off += n;
    col_off += qb;
  }

  auto ob_or = ml2s_observed_bread(pt, rep, samp, est, *weight_or, K);
  if (!ob_or.has_value()) return std::unexpected(ob_or.error());
  return Ml2sIjAssembly{std::move(ij_blocks), std::move(K), std::move(*ob_or)};
}

post_expected<TwoStageEMMLInference>
two_stage_missing_data_weighted_ij_from_sm(spec::LatentStructure pt,
                                           const model::MatrixRep& rep,
                                           const RawData& raw,
                                           const Estimates& est,
                                           const FIMLPack& pack,
                                           const FIMLH1& h1,
                                           const SaturatedMoments& sm,
                                           TwoStageWeight kind,
                                           TwoStageDlsOptions dls,
                                           TwoStageBread bread) {
  if (bread != TwoStageBread::Observed || raw_has_no_missing_mask(raw) ||
      kind == TwoStageWeight::Nt || kind == TwoStageWeight::Uls) {
    return two_stage_em_weighted_inference_from_sm(
        std::move(pt), rep, est, sm, kind, dls, bread);
  }

  auto base_or = two_stage_em_weighted_inference_from_sm(
      pt, rep, est, sm, kind, dls, bread);
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1, sm,
                                     kind, dls);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  auto ij_or = robust_weighted_moment_ij(
      asm_or->blocks, asm_or->K, 2.0 * est.fmin, asm_or->observed_bread);
  if (!ij_or.has_value()) return std::unexpected(ij_or.error());

  TwoStageEMMLInference out = std::move(*base_or);
  out.vcov = std::move(ij_or->vcov);
  out.se = std::move(ij_or->se);
  return out;
}

// Core two-stage ML inference from already-computed Stage-1 saturated moments.
// `raw` is intentionally absent: the only thing the inference ever needs from
// the data are the saturated moments and their ACOV, both carried by `sm`.
// Callers that hold a Stage-1 `SaturatedMoments` should route here directly
// rather than recomputing the EM + observed information + ACOV. `kind == Nt`
// is the normal-theory path; non-NT weights dispatch to the explicit-weight
// robust sandwich above.
post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference_from_sm(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const Estimates& est,
                                  const SaturatedMoments& sm,
                                  TwoStageWeight kind,
                                  TwoStageDlsOptions dls,
                                  TwoStageBread bread) {
  if (kind != TwoStageWeight::Nt) {
    return two_stage_em_weighted_inference_from_sm(std::move(pt), rep, est, sm,
                                                   kind, dls, bread);
  }
  SampleStats samp = sample_stats_from_saturated(sm);

  auto df_or = inference::df_stat(pt, samp, est.theta);
  if (!df_or.has_value()) return std::unexpected(df_or.error());

  // The Satorra-Bentler weight uses Unstructured (sample/saturated h1) moments,
  // not Structured (model-implied). lavaan's two.stage / robust.two.stage path
  // hard-forces h1.information = "unstructured", and magmaan's FIML FMG spectrum
  // follows the same convention; with Unstructured the SE and test-statistic
  // scaling match lavaan robust.two.stage to machine precision. Structured here
  // is a finite-sample inconsistency (off by a few percent under non-normality).
  auto gamma_se_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/true);
  if (!gamma_se_or.has_value()) return std::unexpected(gamma_se_or.error());
  auto se_or = robust::robust_se(
      pt, rep, samp, est, *gamma_se_or,
      robust::InferenceSpec{robust_bread(bread),
                            robust::WeightMoments::Unstructured,
                            robust::ScoreCovariance::Empirical});
  if (!se_or.has_value()) return std::unexpected(se_or.error());

  TwoStageEMMLInference out;
  out.vcov = std::move(se_or->vcov);
  out.se = std::move(se_or->se);
  out.df = *df_or;
  out.chisq = inference::chi2_stat(samp, est);
  out.ntotal = 0;
  for (auto n : samp.n_obs) out.ntotal += n;

  if (out.df <= 0) return out;

  auto gamma_test_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_test_or.has_value()) return std::unexpected(gamma_test_or.error());
  auto uf_or = robust::build_u_factor(
      std::move(pt), rep, samp, est,
      robust::InferenceSpec{robust_bread(bread),
                            robust::WeightMoments::Unstructured,
                            robust::ScoreCovariance::Empirical});
  if (!uf_or.has_value()) return std::unexpected(uf_or.error());
  if (static_cast<int>(uf_or->df) != out.df) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: U-factor df (" +
            std::to_string(uf_or->df) + ") != df_stat (" +
            std::to_string(out.df) + ")"));
  }
  auto M_or = robust::reduced_gamma_sample_from_gamma(*uf_or, *gamma_test_or);
  if (!M_or.has_value()) return std::unexpected(M_or.error());
  auto ev_or = robust::ugamma_eigenvalues(*M_or);
  if (!ev_or.has_value()) return std::unexpected(ev_or.error());

  out.eigvals = std::move(*ev_or);
  out.trace_ugamma = out.eigvals.sum();
  out.scaling_factor = out.trace_ugamma / static_cast<double>(out.df);
  out.chisq_scaled = (out.scaling_factor > 0.0)
      ? out.chisq / out.scaling_factor
      : std::numeric_limits<double>::quiet_NaN();
  return out;
}

post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference_impl(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const RawData& raw,
                               const Estimates& est,
                               const FIMLPack& pack,
                               const FIMLH1& h1,
                               TwoStageWeight kind,
                               TwoStageDlsOptions dls,
                               TwoStageBread bread) {
  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  if (kind != TwoStageWeight::Nt && bread == TwoStageBread::Observed) {
    if (raw_has_no_missing_mask(raw)) {
      return two_stage_complete_data_weighted_ij_from_sm(
          std::move(pt), rep, raw, est, *sm_or, kind, dls, bread);
    }
    return two_stage_missing_data_weighted_ij_from_sm(
        std::move(pt), rep, raw, est, pack, h1, *sm_or, kind, dls, bread);
  }
  return two_stage_em_ml_inference_from_sm(std::move(pt), rep, est, *sm_or,
                                           kind, dls, bread);
}

}  // namespace

post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const Estimates& est,
                          const SaturatedMoments& sm,
                          TwoStageWeight kind,
                          TwoStageDlsOptions dls,
                          TwoStageBread bread) {
  return two_stage_em_ml_inference_from_sm(std::move(pt), rep, est, sm,
                                           kind, dls, bread);
}

post_expected<CasewiseInfluenceIJ>
two_stage_casewise_influence_ij(spec::LatentStructure pt,
                                const model::MatrixRep& rep,
                                const RawData& raw,
                                const Estimates& est,
                                const FIMLPack& pack,
                                const FIMLH1& h1,
                                TwoStageWeight kind,
                                TwoStageDlsOptions dls) {
  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());

  // Mirror the SE dispatch: a non-NT Stage-2 weight on complete data routes
  // through the continuous-LS casewise accessor (the same route the SE takes to
  // `continuous_ls_*_ij`, so the per-case rows reproduce that vcov). Everything
  // else — non-NT missing data, and the NT weight (correction zero) — goes
  // through the shared IJ-block assembly with the observed bread.
  if (kind != TwoStageWeight::Nt && kind != TwoStageWeight::Uls &&
      raw_has_no_missing_mask(raw)) {
    SampleStats samp = sample_stats_from_saturated(*sm_or);
    ContinuousLsIJWeightMode mode = ContinuousLsIJWeightMode::SampleEmpiricalWls;
    if (kind == TwoStageWeight::Dwls)
      mode = ContinuousLsIJWeightMode::SampleEmpiricalDwls;
    else if (kind == TwoStageWeight::Dls)
      mode = ContinuousLsIJWeightMode::SampleDls;
    frontier::DlsWeightOptions opts;
    opts.a = dls.a;
    return continuous_ls_casewise_influence_ij(
        std::move(pt), rep, samp, est, gmm::Weight{}, raw, mode, opts);
  }

  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1,
                                     *sm_or, kind, dls);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  return casewise_influence_from_ij_blocks(asm_or->blocks, asm_or->K,
                                           asm_or->observed_bread);
}

post_expected<WeightedMomentRBMParts>
two_stage_rbm_parts(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const RawData& raw,
                    const Estimates& est,
                    const FIMLPack& pack,
                    const FIMLH1& h1,
                    TwoStageWeight kind,
                    TwoStageDlsOptions dls) {
  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1,
                                     *sm_or, kind, dls);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  return weighted_moment_rbm_parts(asm_or->blocks, asm_or->K,
                                   asm_or->observed_bread);
}

post_expected<WeightedMomentRBMParts>
two_stage_rbm_parts(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const RawData& raw,
                    const Estimates& est,
                    const FIMLPack& pack,
                    const FIMLH1& h1,
                    const SaturatedMoments& sm,
                    TwoStageWeight kind,
                    TwoStageDlsOptions dls) {
  auto asm_or = build_ml2s_ij_blocks(std::move(pt), rep, raw, est, pack, h1,
                                     sm, kind, dls);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  return weighted_moment_rbm_parts(asm_or->blocks, asm_or->K,
                                   asm_or->observed_bread);
}

namespace {

post_expected<std::vector<Eigen::MatrixXd>>
two_stage_gamma_blocks_from_acov(const SaturatedMoments& sm) {
  auto gamma_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_or.has_value()) return std::unexpected(gamma_or.error());

  std::vector<Eigen::MatrixXd> blocks;
  blocks.reserve(sm.cov.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < sm.cov.size(); ++b) {
    const Eigen::Index p = sm.cov[b].rows();
    const Eigen::Index q = p + vech_len(p);
    if (off + q > gamma_or->rows()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "two_stage_nt_profile_rmsea: saturated Gamma block layout "
          "overruns assembled Gamma"));
    }
    blocks.push_back(gamma_or->block(off, off, q, q));
    off += q;
  }
  if (off != gamma_or->rows()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_nt_profile_rmsea: saturated Gamma has trailing rows"));
  }
  return blocks;
}

post_expected<double>
total_n_from_saturated(const SaturatedMoments& sm, const char* caller) {
  double out = 0.0;
  for (std::int64_t n : sm.n_obs) out += static_cast<double>(n);
  if (!(out > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": saturated moments have non-positive total N"));
  }
  return out;
}

post_expected<std::vector<WeightedProfileMomentBlock>>
fiml_profile_blocks(const spec::LatentStructure& pt,
                    const model::MatrixRep& rep,
                    const RawData& raw,
                    const Estimates& est,
                    const FIMLPack& pack,
                    const SaturatedMoments& sm) {
  const FIMLCache& cache = pack.cache;
  const std::size_t B = raw.X.size();
  if (B == 0 || cache.block_p.size() != B || sm.mean.size() != B ||
      sm.cov.size() != B || sm.n_obs.size() != B) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_profile_rmsea: empty or inconsistent block layout"));
  }

  auto Delta_or = fiml_eta_jacobian_impl(pt, rep, raw, est, pack);
  if (!Delta_or.has_value()) return std::unexpected(Delta_or.error());
  auto Wstar_or = fiml_structured_h1_information(pt, rep, raw, est, pack);
  if (!Wstar_or.has_value()) return std::unexpected(Wstar_or.error());
  auto gamma_full_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_full_or.has_value()) return std::unexpected(gamma_full_or.error());

  const Eigen::MatrixXd& Delta = Delta_or->Delta_theta;
  const Eigen::MatrixXd& Wstar_full = *Wstar_or;
  const Eigen::MatrixXd& gamma_full = *gamma_full_or;

  Eigen::Index Q = 0;
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = cache.block_p[b];
    if (p <= 0 || raw.X[b].cols() != p || sm.cov[b].rows() != p ||
        sm.cov[b].cols() != p || sm.mean[b].size() != p ||
        sm.n_obs[b] != raw.X[b].rows() || sm.n_obs[b] <= 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fiml_profile_rmsea: malformed saturated block " +
              std::to_string(b)));
    }
    Q += p + vech_len(p);
  }
  if (Delta.rows() != Q || sm.H.rows() != Q || sm.H.cols() != Q ||
      Wstar_full.rows() != Q || Wstar_full.cols() != Q ||
      gamma_full.rows() != Q || gamma_full.cols() != Q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_profile_rmsea: saturated/profile matrix shape mismatch"));
  }

  std::vector<WeightedProfileMomentBlock> blocks;
  blocks.reserve(B);
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < B; ++b) {
    const Eigen::Index p = cache.block_p[b];
    const Eigen::Index q = p + vech_len(p);
    const double nb = static_cast<double>(sm.n_obs[b]);
    blocks.push_back(WeightedProfileMomentBlock{
        .jacobian = Delta.block(off, 0, q, Delta.cols()),
        .data_metric = sm.H.block(off, off, q, q) / nb,
        .projection_metric = Wstar_full.block(off, off, q, q) / nb,
        .gamma = gamma_full.block(off, off, q, q),
        .n_obs = sm.n_obs[b]});
    off += q;
  }
  return blocks;
}

}  // namespace

post_expected<WeightedProfileRMSEAResult>
fiml_profile_rmsea(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const RawData& raw,
                   const Estimates& est,
                   double chi2_lrt,
                   const FIMLPack& pack,
                   const FIMLH1& h1,
                   const SaturatedMoments& sm,
                   double eig_tol) {
  if (!std::isfinite(chi2_lrt)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_profile_rmsea: non-finite chi-square"));
  }
  if (auto e = validate_h1_blocks(pack.cache, h1); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "FIML H1 moments"));
  }
  auto n_or = total_n_from_saturated(sm, "fiml_profile_rmsea");
  if (!n_or.has_value()) return std::unexpected(n_or.error());

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  if (con_or->K().rows() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_profile_rmsea: constraint reparameterization has incompatible "
        "shape"));
  }

  auto blocks = fiml_profile_blocks(pt, rep, raw, est, pack, sm);
  if (!blocks.has_value()) return std::unexpected(blocks.error());
  auto info = fiml_observed_information_impl(pt, rep, raw, est, pack);
  if (!info.has_value()) return std::unexpected(info.error());
  Eigen::MatrixXd bread =
      con_or->K().transpose() * (*info) * con_or->K() / *n_or;
  bread = 0.5 * (bread + bread.transpose()).eval();

  return ::magmaan::estimate::weighted_moment_profile_rmsea_two_metric(
      *blocks, con_or->K(), chi2_lrt / *n_or, bread, raw.X.size(), eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
fiml_profile_rmsea(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const RawData& raw,
                   const Estimates& est,
                   double chi2_lrt,
                   const FIMLPack& pack,
                   const FIMLH1& h1,
                   double eig_tol) {
  auto sm = saturated_em_moments(raw, pack, h1);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return fiml_profile_rmsea(
      std::move(pt), rep, raw, est, chi2_lrt, pack, h1, *sm, eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
fiml_profile_rmsea(spec::LatentStructure pt,
                   const model::MatrixRep& rep,
                   const RawData& raw,
                   const Estimates& est,
                   double chi2_lrt,
                   FIML discrepancy,
                   double h_step,
                   double eig_tol) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_profile_rmsea: h_step must be > 0"));
  }
  (void)discrepancy;
  auto pack = fiml_pack(raw);
  if (!pack.has_value()) {
    return std::unexpected(fit_to_post(pack.error(), "fiml_pack"));
  }
  auto h1 = fiml_h1_moments(raw, *pack);
  if (!h1.has_value()) {
    return std::unexpected(fit_to_post(h1.error(), "FIML H1 moments"));
  }
  auto sm = saturated_em_moments(raw, *pack, *h1);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return fiml_profile_rmsea(
      std::move(pt), rep, raw, est, chi2_lrt, *pack, *h1, *sm, eig_tol);
}

post_expected<WeightedProfileLRTResult>
fiml_profile_lrt(spec::LatentStructure pt_H1,
                 const model::MatrixRep& rep_H1,
                 const RawData& raw,
                 const Estimates& est_H1,
                 double chi2_lrt_H1,
                 spec::LatentStructure pt_H0,
                 const model::MatrixRep& rep_H0,
                 const Estimates& est_H0,
                 double chi2_lrt_H0,
                 const FIMLPack& pack,
                 const FIMLH1& h1,
                 const SaturatedMoments& sm,
                 double eig_tol) {
  auto h1_profile = fiml_profile_rmsea(
      std::move(pt_H1), rep_H1, raw, est_H1, chi2_lrt_H1, pack, h1, sm,
      eig_tol);
  if (!h1_profile.has_value()) return std::unexpected(h1_profile.error());
  auto h0_profile = fiml_profile_rmsea(
      std::move(pt_H0), rep_H0, raw, est_H0, chi2_lrt_H0, pack, h1, sm,
      eig_tol);
  if (!h0_profile.has_value()) return std::unexpected(h0_profile.error());
  return ::magmaan::estimate::weighted_moment_profile_lrt(
      *h1_profile, *h0_profile, eig_tol);
}

post_expected<WeightedProfileLRTResult>
fiml_profile_lrt(spec::LatentStructure pt_H1,
                 const model::MatrixRep& rep_H1,
                 const RawData& raw,
                 const Estimates& est_H1,
                 double chi2_lrt_H1,
                 spec::LatentStructure pt_H0,
                 const model::MatrixRep& rep_H0,
                 const Estimates& est_H0,
                 double chi2_lrt_H0,
                 const FIMLPack& pack,
                 const FIMLH1& h1,
                 double eig_tol) {
  auto sm = saturated_em_moments(raw, pack, h1);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return fiml_profile_lrt(
      std::move(pt_H1), rep_H1, raw, est_H1, chi2_lrt_H1,
      std::move(pt_H0), rep_H0, est_H0, chi2_lrt_H0, pack, h1, *sm, eig_tol);
}

post_expected<WeightedProfileLRTResult>
fiml_profile_lrt(spec::LatentStructure pt_H1,
                 const model::MatrixRep& rep_H1,
                 const RawData& raw,
                 const Estimates& est_H1,
                 double chi2_lrt_H1,
                 spec::LatentStructure pt_H0,
                 const model::MatrixRep& rep_H0,
                 const Estimates& est_H0,
                 double chi2_lrt_H0,
                 FIML discrepancy,
                 double h_step,
                 double eig_tol) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fiml_profile_lrt: h_step must be > 0"));
  }
  (void)discrepancy;
  auto pack = fiml_pack(raw);
  if (!pack.has_value()) {
    return std::unexpected(fit_to_post(pack.error(), "fiml_pack"));
  }
  auto h1 = fiml_h1_moments(raw, *pack);
  if (!h1.has_value()) {
    return std::unexpected(fit_to_post(h1.error(), "FIML H1 moments"));
  }
  auto sm = saturated_em_moments(raw, *pack, *h1);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return fiml_profile_lrt(
      std::move(pt_H1), rep_H1, raw, est_H1, chi2_lrt_H1,
      std::move(pt_H0), rep_H0, est_H0, chi2_lrt_H0, *pack, *h1, *sm,
      eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
two_stage_nt_profile_rmsea(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const Estimates& est,
                           const SaturatedMoments& sm,
                           double eig_tol) {
  SampleStats samp = sample_stats_from_saturated(sm);
  auto gamma = two_stage_gamma_blocks_from_acov(sm);
  if (!gamma.has_value()) return std::unexpected(gamma.error());
  return ::magmaan::estimate::ml_profile_rmsea(
      std::move(pt), rep, samp, est, *gamma, eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
two_stage_nt_profile_rmsea(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const RawData& raw,
                           const Estimates& est,
                           double h_step,
                           double eig_tol) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_nt_profile_rmsea: h_step must be > 0"));
  }
  auto sm = saturated_em_moments(raw, h_step);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return two_stage_nt_profile_rmsea(
      std::move(pt), rep, est, *sm, eig_tol);
}

post_expected<WeightedProfileRMSEAResult>
two_stage_nt_profile_rmsea(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const RawData& raw,
                           const Estimates& est,
                           const FIMLPack& pack,
                           const FIMLH1& h1,
                           double eig_tol) {
  auto sm = saturated_em_moments(raw, pack, h1);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return two_stage_nt_profile_rmsea(
      std::move(pt), rep, est, *sm, eig_tol);
}

post_expected<WeightedProfileLRTResult>
two_stage_nt_profile_lrt(spec::LatentStructure pt_H1,
                         const model::MatrixRep& rep_H1,
                         const Estimates& est_H1,
                         spec::LatentStructure pt_H0,
                         const model::MatrixRep& rep_H0,
                         const Estimates& est_H0,
                         const SaturatedMoments& sm,
                         double eig_tol) {
  auto h1 = two_stage_nt_profile_rmsea(
      std::move(pt_H1), rep_H1, est_H1, sm, eig_tol);
  if (!h1.has_value()) return std::unexpected(h1.error());
  auto h0 = two_stage_nt_profile_rmsea(
      std::move(pt_H0), rep_H0, est_H0, sm, eig_tol);
  if (!h0.has_value()) return std::unexpected(h0.error());
  return ::magmaan::estimate::weighted_moment_profile_lrt(
      *h1, *h0, eig_tol);
}

post_expected<WeightedProfileLRTResult>
two_stage_nt_profile_lrt(spec::LatentStructure pt_H1,
                         const model::MatrixRep& rep_H1,
                         const RawData& raw,
                         const Estimates& est_H1,
                         spec::LatentStructure pt_H0,
                         const model::MatrixRep& rep_H0,
                         const Estimates& est_H0,
                         double h_step,
                         double eig_tol) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_nt_profile_lrt: h_step must be > 0"));
  }
  auto sm = saturated_em_moments(raw, h_step);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return two_stage_nt_profile_lrt(
      std::move(pt_H1), rep_H1, est_H1, std::move(pt_H0), rep_H0, est_H0,
      *sm, eig_tol);
}

post_expected<WeightedProfileLRTResult>
two_stage_nt_profile_lrt(spec::LatentStructure pt_H1,
                         const model::MatrixRep& rep_H1,
                         const RawData& raw,
                         const Estimates& est_H1,
                         spec::LatentStructure pt_H0,
                         const model::MatrixRep& rep_H0,
                         const Estimates& est_H0,
                         const FIMLPack& pack,
                         const FIMLH1& h1,
                         double eig_tol) {
  auto sm = saturated_em_moments(raw, pack, h1);
  if (!sm.has_value()) return std::unexpected(sm.error());
  return two_stage_nt_profile_lrt(
      std::move(pt_H1), rep_H1, est_H1, std::move(pt_H0), rep_H0, est_H0,
      *sm, eig_tol);
}

namespace {

post_expected<TwoStageFitMeasures>
two_stage_fit_measures_from_sm(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const Estimates& est,
                               const SaturatedMoments& sm,
                               TwoStageWeight kind,
                               TwoStageDlsOptions dls) {
  SampleStats samp = sample_stats_from_saturated(sm);
  auto user_or = two_stage_em_ml_inference_from_sm(
      pt, rep, est, sm, kind, dls, TwoStageBread::Expected);
  if (!user_or.has_value()) return std::unexpected(user_or.error());
  const TwoStageEMMLInference& user = *user_or;
  if (user.df <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_fit_measures: requires df > 0"));
  }

  const measures::BaselineFit baseline =
      measures::baseline_chi2(pt, samp);
  if (baseline.df <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_fit_measures: baseline df must be > 0"));
  }

  auto gamma_or = two_stage_gamma_from_acov(sm, /*se_weighted=*/false);
  if (!gamma_or.has_value()) return std::unexpected(gamma_or.error());
  auto weight_or = two_stage_stage2_weight(sm, kind, dls);
  if (!weight_or.has_value()) return std::unexpected(weight_or.error());

  const std::vector<Eigen::Index> exo_idx = observed_exogenous_indices(pt);
  const std::vector<Eigen::Index> block_p = block_p_from_saturated(sm);
  const Eigen::MatrixXd D_null =
      baseline_eta_delta_from_block_p(block_p, exo_idx);
  auto tr_null_or = residual_projector_trace(
      *weight_or, *gamma_or, D_null, "two_stage_fit_measures: baseline");
  if (!tr_null_or.has_value()) return std::unexpected(tr_null_or.error());
  const double c_null = *tr_null_or / static_cast<double>(baseline.df);
  const double baseline_scaled = (c_null > 0.0)
      ? baseline.chi2 / c_null
      : std::numeric_limits<double>::quiet_NaN();

  measures::RobustFitMeasureInputs inputs;
  inputs.chi2 = user.chisq;
  inputs.df = user.df;
  inputs.chi2_scaled = user.chisq_scaled;
  inputs.scaling_factor = user.scaling_factor;
  inputs.baseline_chi2 = baseline.chi2;
  inputs.baseline_df = baseline.df;
  inputs.baseline_chi2_scaled = baseline_scaled;
  inputs.baseline_scaling_factor = c_null;
  inputs.n_total = user.ntotal;
  inputs.n_groups = samp.S.size();

  TwoStageFitMeasures out;
  out.baseline = baseline;
  out.indices = measures::robust_fit_measures(inputs);
  return out;
}

}  // namespace

post_expected<TwoStageFitMeasures>
two_stage_fit_measures(spec::LatentStructure pt,
                       const model::MatrixRep& rep,
                       const Estimates& est,
                       const SaturatedMoments& sm,
                       TwoStageWeight kind,
                       TwoStageDlsOptions dls) {
  return two_stage_fit_measures_from_sm(std::move(pt), rep, est, sm,
                                        kind, dls);
}

post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          double h_step,
                          TwoStageWeight kind,
                          TwoStageDlsOptions dls,
                          TwoStageBread bread) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_em_ml_inference: h_step must be > 0"));
  }
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 moments"));
  }
  return two_stage_em_ml_inference_impl(std::move(pt), rep, raw, est,
                                        *pack_or, *h1_or, kind, dls, bread);
}

post_expected<TwoStageEMMLInference>
two_stage_em_ml_inference(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          const FIMLPack& pack,
                          const FIMLH1& h1,
                          TwoStageWeight kind,
                          TwoStageDlsOptions dls,
                          TwoStageBread bread) {
  return two_stage_em_ml_inference_impl(std::move(pt), rep, raw, est,
                                        pack, h1, kind, dls, bread);
}

post_expected<TwoStageFitMeasures>
two_stage_fit_measures(spec::LatentStructure pt,
                       const model::MatrixRep& rep,
                       const RawData& raw,
                       const Estimates& est,
                       double h_step,
                       TwoStageWeight kind,
                       TwoStageDlsOptions dls) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "two_stage_fit_measures: h_step must be > 0"));
  }
  auto pack_or = fiml_pack(raw);
  if (!pack_or.has_value()) {
    return std::unexpected(fit_to_post(pack_or.error(), "fiml_pack"));
  }
  auto h1_or = fiml_h1_moments(raw, *pack_or);
  if (!h1_or.has_value()) {
    return std::unexpected(fit_to_post(h1_or.error(), "FIML H1 moments"));
  }
  return two_stage_fit_measures(std::move(pt), rep, raw, est, *pack_or, *h1_or,
                                kind, dls);
}

post_expected<TwoStageFitMeasures>
two_stage_fit_measures(spec::LatentStructure pt,
                       const model::MatrixRep& rep,
                       const RawData& raw,
                       const Estimates& est,
                       const FIMLPack& pack,
                       const FIMLH1& h1,
                       TwoStageWeight kind,
                       TwoStageDlsOptions dls) {
  auto sm_or = saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) return std::unexpected(sm_or.error());
  return two_stage_fit_measures_from_sm(std::move(pt), rep, est, *sm_or,
                                        kind, dls);
}

namespace diagnostic {

post_expected<SaturatedMoments>
saturated_em_moments_fd(const RawData& raw, double h_step) {
  return saturated_em_moments_from_raw(
      raw, h_step, SaturatedHessianKind::FiniteDifference,
      "saturated_em_moments_fd");
}

}  // namespace diagnostic

namespace {

fit_expected<Estimates>
fit_fiml_impl(spec::LatentStructure pt,
              const model::MatrixRep& rep,
              const RawData& raw,
              const Eigen::VectorXd& x0,
              const FIMLCache& cache,
              const SampleStats& start_samp,
              FIML discrepancy,
              Backend backend,
              optim::OptimOptions opts) {
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(e.error());
  }

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
    auto vg = discrepancy.value_gradient(raw, cache, eval->moments,
                                         eval->J_sigma, eval->J_mu);
    if (!vg.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    // ½·F scale: est.fmin = ½F, uniform with ML/LS. The FIML χ² stays the LRT
    // in fiml_extras, which recomputes the full-F deviance from est.theta and
    // is unaffected. INVARIANT: halve ONLY here in the optimiser adapter —
    // FIML::value / value_gradient must stay full-F because fiml_extras and
    // the observed-Hessian paths (analytic and FD) differentiate them.
    grad = 0.5 * vg->gradient;
    return 0.5 * vg->value;
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

}  // namespace

fit_expected<Estimates>
fit_fiml(spec::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         const Eigen::VectorXd& x0,
         FIML discrepancy,
         Backend backend,
         optim::OptimOptions opts) {
  // Policy validation precedes prepare so unsupported fixed.x-missing models
  // keep reporting the policy error, not a downstream pattern error.
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(e.error());
  }
  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) return std::unexpected(cache_or.error());
  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) return std::unexpected(start_samp_or.error());
  return fit_fiml_impl(std::move(pt), rep, raw, x0, *cache_or, *start_samp_or,
                       discrepancy, backend, std::move(opts));
}

fit_expected<Estimates>
fit_fiml(spec::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         const Eigen::VectorXd& x0,
         const FIMLPack& pack,
         Backend backend,
         optim::OptimOptions opts) {
  return fit_fiml_impl(std::move(pt), rep, raw, x0, pack.cache,
                       pack.start_stats, FIML{}, backend, std::move(opts));
}

}  // namespace magmaan::estimate::fiml
