#include "magmaan/data/frontier/shrinkage.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"

#include "detail_linalg.hpp"

namespace magmaan::data::frontier {

namespace {

PostError make_err(std::string detail) {
  return PostError{PostError::Kind::NumericIssue, std::move(detail)};
}

// Smallest eigenvalue of a symmetric matrix (NaN on solver failure / empty).
double min_eigenvalue(const Eigen::MatrixXd& M) {
  if (M.rows() == 0) return std::numeric_limits<double>::quiet_NaN();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return es.eigenvalues().minCoeff();
}

post_expected<CovarianceRepairResult>
shrink_covariance_matrix(const Eigen::MatrixXd& S0,
                         std::int64_t n_obs,
                         CovarianceShrinkageOptions opts,
                         Eigen::MatrixXd& Sp,
                         const std::string& what) {
  if (S0.rows() != S0.cols()) {
    return std::unexpected(make_err(what + " is not square"));
  }
  const Eigen::Index p = S0.rows();
  const Eigen::MatrixXd S = 0.5 * (S0 + S0.transpose());

  CovarianceRepairResult diag;
  diag.raw_min_eigen = min_eigenvalue(S);

  double s = 0.0;
  if (opts.kind != CovarianceShrinkageKind::None) {
    if (opts.estimate_intensity) {
      if (opts.kind != CovarianceShrinkageKind::DiagonalTarget) {
        return std::unexpected(make_err(
            "covariance shrinkage: estimate_intensity is supported only for "
            "DiagonalTarget"));
      }
      if (n_obs < 1) {
        return std::unexpected(make_err(
            "covariance shrinkage: estimate_intensity needs a positive n_obs"));
      }
      const double n = static_cast<double>(n_obs);
      double num = 0.0;
      double den = 0.0;
      for (Eigen::Index i = 0; i < p; ++i) {
        if (!(S(i, i) > 0.0)) {
          return std::unexpected(make_err(
              "covariance shrinkage: non-positive variance in " + what));
        }
      }
      for (Eigen::Index i = 0; i < p; ++i) {
        for (Eigen::Index j = i + 1; j < p; ++j) {
          const double r = S(i, j) / std::sqrt(S(i, i) * S(j, j));
          num += (1.0 - r * r) * (1.0 - r * r) / n;
          den += r * r;
        }
      }
      s = (den > 0.0) ? std::clamp(num / den, 0.0, 1.0) : 0.0;
    } else {
      s = std::clamp(opts.intensity, 0.0, 1.0);
    }
  }

  Sp = S;
  if (opts.kind != CovarianceShrinkageKind::None && s > 0.0) {
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(p, p);
    switch (opts.kind) {
      case CovarianceShrinkageKind::None:
        break;
      case CovarianceShrinkageKind::Ridge:
        T = Eigen::MatrixXd::Identity(p, p);
        break;
      case CovarianceShrinkageKind::IdentityTarget: {
        const double mu = (p > 0) ? S.trace() / static_cast<double>(p) : 0.0;
        T = mu * Eigen::MatrixXd::Identity(p, p);
        break;
      }
      case CovarianceShrinkageKind::DiagonalTarget:
        T = S.diagonal().asDiagonal();
        break;
      case CovarianceShrinkageKind::ConstantCorrelation: {
        for (Eigen::Index i = 0; i < p; ++i) {
          if (!(S(i, i) > 0.0)) {
            return std::unexpected(make_err(
                "covariance shrinkage: non-positive variance in " + what));
          }
        }
        double rbar = 0.0;
        Eigen::Index npair = 0;
        for (Eigen::Index i = 0; i < p; ++i) {
          for (Eigen::Index j = i + 1; j < p; ++j) {
            rbar += S(i, j) / std::sqrt(S(i, i) * S(j, j));
            ++npair;
          }
        }
        if (npair > 0) rbar /= static_cast<double>(npair);
        for (Eigen::Index i = 0; i < p; ++i) {
          T(i, i) = S(i, i);
          for (Eigen::Index j = i + 1; j < p; ++j) {
            const double t = rbar * std::sqrt(S(i, i) * S(j, j));
            T(i, j) = t;
            T(j, i) = t;
          }
        }
        break;
      }
    }
    Sp = (1.0 - s) * S + s * T;
  }

  diag.min_eigen = min_eigenvalue(Sp);
  diag.intensity = s;
  diag.shrunk = (opts.kind != CovarianceShrinkageKind::None) && (s > 0.0);
  return diag;
}

post_expected<Eigen::MatrixXd> symmetric_inverse_pd(const Eigen::MatrixXd& A,
                                                    std::string what) {
  detail::SymInverseResult r = detail::symmetric_inverse_pd_gated(A);
  if (r.ok) return std::move(r.inverse);
  if (!r.finite) {
    return std::unexpected(make_err(std::move(what) +
                                    " is not a finite square matrix"));
  }
  if (!r.decomposed) {
    return std::unexpected(make_err(std::move(what) +
                                    " eigendecomposition failed"));
  }
  return std::unexpected(make_err(std::move(what) + " is not positive definite"));
}

Eigen::VectorXd mixed_moment_vector(const Eigen::MatrixXd& R,
                                    const Eigen::VectorXd& mean,
                                    const std::vector<std::int32_t>& ordered,
                                    const Eigen::VectorXd& thresholds) {
  const Eigen::Index p = R.rows();
  Eigen::Index n_cont = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) ++n_cont;
  }
  Eigen::VectorXd out(thresholds.size() + 2 * n_cont + p * (p - 1) / 2);
  Eigen::Index k = 0;
  out.segment(k, thresholds.size()) = thresholds;
  k += thresholds.size();
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) out(k++) = -mean(j);
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) out(k++) = R(j, j);
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) out(k++) = R(i, j);
  }
  return out;
}

struct MixedMomentCoordinate {
  Eigen::Index row = 0;
  Eigen::Index i = 0;
  Eigen::Index j = 0;
};

std::vector<MixedMomentCoordinate>
mixed_covariance_moment_coordinates(const MixedOrdinalStats& stats,
                                    std::size_t b) {
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  Eigen::Index row = nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) ++row;
  }

  std::vector<MixedMomentCoordinate> out;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
      out.push_back(MixedMomentCoordinate{row, j, j});
      ++row;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out.push_back(MixedMomentCoordinate{row, i, j});
      ++row;
    }
  }
  return out;
}

post_expected<Eigen::VectorXd>
shrunk_mixed_moments_for_block(const MixedOrdinalStats& stats,
                               std::size_t b,
                               const Eigen::MatrixXd& R,
                               CovarianceShrinkageOptions opts,
                               Eigen::MatrixXd* shrunk_R,
                               CovarianceRepairResult* diag) {
  Eigen::MatrixXd Sp;
  auto d_or = shrink_covariance_matrix(
      R, stats.n_obs[b], opts,
      Sp, "MixedOrdinalStats block " + std::to_string(b) + " R");
  if (!d_or.has_value()) return std::unexpected(d_or.error());
  if (shrunk_R != nullptr) *shrunk_R = Sp;
  if (diag != nullptr) *diag = *d_or;
  return mixed_moment_vector(Sp, stats.mean[b], stats.ordered[b],
                             stats.thresholds[b]);
}

}  // namespace

post_expected<ShrunkSampleStats>
shrink_sample_stats(const SampleStats& samp, CovarianceShrinkageOptions opts) {
  ShrunkSampleStats out;
  out.stats.mean = samp.mean;     // means pass through unchanged
  out.stats.n_obs = samp.n_obs;   // n_obs pass through unchanged
  out.stats.S.reserve(samp.S.size());
  out.block_diagnostics.reserve(samp.S.size());

  for (std::size_t blk = 0; blk < samp.S.size(); ++blk) {
    const std::int64_t n_obs =
        blk < samp.n_obs.size() ? samp.n_obs[blk] : 0;
    Eigen::MatrixXd Sp;
    auto diag_or = shrink_covariance_matrix(
        samp.S[blk], n_obs, opts, Sp,
        "SampleStats block " + std::to_string(blk) + " covariance");
    if (!diag_or.has_value()) return std::unexpected(diag_or.error());

    out.stats.S.push_back(std::move(Sp));
    out.block_diagnostics.push_back(*diag_or);
  }

  return out;
}

post_expected<ShrunkMixedOrdinalStats>
shrink_mixed_ordinal_stats(const MixedOrdinalStats& stats,
                           CovarianceShrinkageOptions opts) {
  ShrunkMixedOrdinalStats out;
  out.stats = stats;
  out.block_diagnostics.reserve(stats.R.size());

  const std::size_t nb = stats.R.size();
  if (stats.mean.size() != nb || stats.ordered.size() != nb ||
      stats.thresholds.size() != nb || stats.moments.size() != nb ||
      stats.NACOV.size() != nb || stats.n_obs.size() != nb) {
    return std::unexpected(make_err(
        "shrink_mixed_ordinal_stats: inconsistent block counts"));
  }

  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mdim = stats.moments[b].size();
    if (stats.R[b].cols() != p || stats.mean[b].size() != p ||
        stats.ordered[b].size() != static_cast<std::size_t>(p) ||
        stats.NACOV[b].rows() != mdim || stats.NACOV[b].cols() != mdim) {
      return std::unexpected(make_err(
          "shrink_mixed_ordinal_stats: dimension mismatch in block " +
          std::to_string(b)));
    }

    Eigen::MatrixXd shrunk_R;
    CovarianceRepairResult diag;
    auto moment_or = shrunk_mixed_moments_for_block(
        stats, b, stats.R[b], opts, &shrunk_R, &diag);
    if (!moment_or.has_value()) return std::unexpected(moment_or.error());

    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(mdim, mdim);
    if (diag.shrunk || opts.estimate_intensity) {
      const auto coords = mixed_covariance_moment_coordinates(stats, b);
      for (const MixedMomentCoordinate& c : coords) {
        const double base = stats.R[b](c.i, c.j);
        const double h = 1e-6 * std::max(1.0, std::abs(base));
        Eigen::MatrixXd Rp = stats.R[b];
        Eigen::MatrixXd Rm = stats.R[b];
        Rp(c.i, c.j) += h;
        Rm(c.i, c.j) -= h;
        if (c.i != c.j) {
          Rp(c.j, c.i) += h;
          Rm(c.j, c.i) -= h;
        }
        auto mp_or = shrunk_mixed_moments_for_block(
            stats, b, Rp, opts, nullptr, nullptr);
        if (!mp_or.has_value()) return std::unexpected(mp_or.error());
        auto mm_or = shrunk_mixed_moments_for_block(
            stats, b, Rm, opts, nullptr, nullptr);
        if (!mm_or.has_value()) return std::unexpected(mm_or.error());
        J.col(c.row) = (*mp_or - *mm_or) / (2.0 * h);
      }
    }

    Eigen::MatrixXd NACOV = J * stats.NACOV[b] * J.transpose();
    NACOV = 0.5 * (NACOV + NACOV.transpose());

    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(
            "shrink_mixed_ordinal_stats: block " + std::to_string(b) +
            " has a non-positive NACOV diagonal after shrinkage"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    auto W_wls_or = symmetric_inverse_pd(
        NACOV, "shrunk mixed ordinal NACOV matrix");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());

    out.stats.R[b] = std::move(shrunk_R);
    out.stats.moments[b] = std::move(*moment_or);
    out.stats.NACOV[b] = std::move(NACOV);
    out.stats.W_dwls[b] = std::move(W_dwls);
    out.stats.W_wls[b] = std::move(*W_wls_or);
    out.block_diagnostics.push_back(diag);
  }

  return out;
}

}  // namespace magmaan::data::frontier
