#include "magmaan/estimate/ordinal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/gmm/gp.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/robust/satorra2000.hpp"
#include "magmaan/robust/weighted_chisq.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/optim/reparameterize.hpp"
#include "magmaan/parse/op.hpp"

#include "detail_second_order.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using inference::ScoreCandidate;
using inference::ScoreCandidateKind;
using inference::ScoreTestResult;
using inference::ScoreTestTable;
using inference::chi2_pvalue;
using robust::MeanVarAdjustedResult;
using robust::SatorraBentlerResult;
using robust::ScaledShiftedResult;
using optim::OptimOptions;

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

PostError make_post_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

PostError fit_to_post(FitError e) {
  return make_post_err(PostError::Kind::NumericIssue, std::move(e.detail));
}

PostError model_to_post(ModelError e) {
  return make_post_err(PostError::Kind::NumericIssue, std::move(e.detail));
}

FitError post_to_fit(PostError e) {
  return make_err(FitError::Kind::NumericIssue, std::move(e.detail));
}

OrdinalParameterization to_estimate_parameterization(
    data::OrdinalMomentParameterization parameterization) {
  return parameterization == data::OrdinalMomentParameterization::Theta
      ? OrdinalParameterization::Theta
      : OrdinalParameterization::Delta;
}

Eigen::Index vech_index(Eigen::Index p, Eigen::Index r, Eigen::Index c) noexcept {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

Eigen::Index vech_len(Eigen::Index p) noexcept {
  return p * (p + 1) / 2;
}

bool matrix_all_finite(const Eigen::MatrixXd& M) {
  for (Eigen::Index c = 0; c < M.cols(); ++c)
    for (Eigen::Index r = 0; r < M.rows(); ++r)
      if (!std::isfinite(M(r, c))) return false;
  return true;
}

bool vector_all_finite(const Eigen::VectorXd& v) {
  for (Eigen::Index i = 0; i < v.size(); ++i)
    if (!std::isfinite(v(i))) return false;
  return true;
}

post_expected<double> log_det_pd_post(const Eigen::MatrixXd& A,
                                      std::string what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  const Eigen::MatrixXd S = 0.5 * (A + A.transpose());
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        std::move(what) + " is not positive definite"));
  }
  double out = 0.0;
  const auto L = llt.matrixL();
  for (Eigen::Index i = 0; i < L.rows(); ++i) out += std::log(L(i, i));
  return 2.0 * out;
}

post_expected<Eigen::MatrixXd>
sym_inverse_pd_post(const Eigen::MatrixXd& A, std::string what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  const Eigen::MatrixXd S = 0.5 * (A + A.transpose());
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        std::move(what) + " is not positive definite"));
  }
  Eigen::MatrixXd inv =
      llt.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  return Eigen::MatrixXd(0.5 * (inv + inv.transpose()).eval());
}

post_expected<Eigen::MatrixXd>
ordinal_nt_gamma_block(const data::OrdinalStats& stats, std::size_t b) {
  const Eigen::MatrixXd& R = stats.R[b];
  const Eigen::Index p = R.rows();
  if (R.cols() != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_stage2_weight: R block is not square"));
  }
  const Eigen::Index nth = stats.thresholds[b].size();
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;
  if (stats.NACOV[b].rows() != mdim || stats.NACOV[b].cols() != mdim) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_stage2_weight: NACOV dimension mismatch in block " +
            std::to_string(b)));
  }

  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(mdim, mdim);
  if (nth > 0) {
    G.topLeftCorner(nth, nth) = stats.NACOV[b].topLeftCorner(nth, nth);
  }
  if (ncorr == 0) return Eigen::MatrixXd(0.5 * (G + G.transpose()).eval());

  auto gamma_cov_or = data::gamma_nt(R);
  if (!gamma_cov_or.has_value()) return std::unexpected(gamma_cov_or.error());

  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ncorr, pstar);
  Eigen::Index row = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const double rho = R(i, j);
      D(row, vech_index(p, i, j)) = 1.0;
      D(row, vech_index(p, i, i)) = -0.5 * rho;
      D(row, vech_index(p, j, j)) = -0.5 * rho;
      ++row;
    }
  }
  Eigen::MatrixXd Gcorr = D * (*gamma_cov_or) * D.transpose();
  G.bottomRightCorner(ncorr, ncorr) =
      0.5 * (Gcorr + Gcorr.transpose()).eval();
  return Eigen::MatrixXd(0.5 * (G + G.transpose()).eval());
}

post_expected<Eigen::MatrixXd>
dwls_weight_from_gamma(const Eigen::MatrixXd& G, std::string what) {
  if (G.rows() != G.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(G.rows(), G.cols());
  for (Eigen::Index k = 0; k < G.rows(); ++k) {
    const double v = G(k, k);
    if (!(v > 0.0) || !std::isfinite(v)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::move(what) + " has a non-positive diagonal"));
    }
    W(k, k) = 1.0 / v;
  }
  return W;
}

post_expected<Eigen::MatrixXd>
correlation_matrix(const Eigen::MatrixXd& Sigma, std::string what) {
  if (Sigma.rows() != Sigma.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  const Eigen::Index p = Sigma.rows();
  Eigen::VectorXd sd(p);
  for (Eigen::Index i = 0; i < p; ++i) {
    const double v = Sigma(i, i);
    if (!(v > 0.0) || !std::isfinite(v)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::move(what) + " has a non-positive diagonal"));
    }
    sd(i) = std::sqrt(v);
  }
  Eigen::MatrixXd R(p, p);
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = 0; i < p; ++i) {
      R(i, j) = Sigma(i, j) / (sd(i) * sd(j));
    }
  }
  R.diagonal().setOnes();
  return Eigen::MatrixXd(0.5 * (R + R.transpose()).eval());
}

post_expected<Eigen::MatrixXd>
ordinal_catml_correlation_matrix(const Eigen::MatrixXd& Sigma,
                                 OrdinalParameterization parameterization,
                                 std::string what) {
  if (parameterization == OrdinalParameterization::Theta) {
    return correlation_matrix(Sigma, std::move(what));
  }
  if (Sigma.rows() != Sigma.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  Eigen::MatrixXd R = 0.5 * (Sigma + Sigma.transpose()).eval();
  R.diagonal().setOnes();
  return R;
}

post_expected<Eigen::MatrixXd>
catml_correlation_information(const Eigen::MatrixXd& R) {
  auto Rinv_or = sym_inverse_pd_post(R, "catML implied correlation");
  if (!Rinv_or.has_value()) return std::unexpected(Rinv_or.error());
  const Eigen::MatrixXd& Rinv = *Rinv_or;
  const Eigen::Index p = R.rows();
  const Eigen::Index ncorr = p * (p - 1) / 2;
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(ncorr, ncorr);
  Eigen::Index a = 0;
  for (Eigen::Index c1 = 0; c1 < p; ++c1) {
    for (Eigen::Index r1 = c1 + 1; r1 < p; ++r1) {
      Eigen::MatrixXd E1 = Eigen::MatrixXd::Zero(p, p);
      E1(r1, c1) = 1.0;
      E1(c1, r1) = 1.0;
      const Eigen::MatrixXd A1 = Rinv * E1 * Rinv;
      Eigen::Index b = 0;
      for (Eigen::Index c2 = 0; c2 < p; ++c2) {
        for (Eigen::Index r2 = c2 + 1; r2 < p; ++r2) {
          Eigen::MatrixXd E2 = Eigen::MatrixXd::Zero(p, p);
          E2(r2, c2) = 1.0;
          E2(c2, r2) = 1.0;
          V(a, b) = 0.5 * (A1 * E2).trace();
          ++b;
        }
      }
      ++a;
    }
  }
  return Eigen::MatrixXd(0.5 * (V + V.transpose()).eval());
}

post_expected<double>
catml_correlation_discrepancy(const Eigen::MatrixXd& sample_R,
                              const Eigen::MatrixXd& implied_R) {
  auto logS_or = log_det_pd_post(sample_R, "catML sample polychoric R");
  if (!logS_or.has_value()) return std::unexpected(logS_or.error());
  auto logM_or = log_det_pd_post(implied_R, "catML implied R");
  if (!logM_or.has_value()) return std::unexpected(logM_or.error());
  auto invM_or = sym_inverse_pd_post(implied_R, "catML implied R");
  if (!invM_or.has_value()) return std::unexpected(invM_or.error());
  const double tr = sample_R.cwiseProduct(*invM_or).sum();
  return *logM_or + tr - *logS_or - static_cast<double>(sample_R.rows());
}

Eigen::Index numerical_rank(const Eigen::MatrixXd& A) {
  if (A.size() == 0) return 0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.singularValues().size() == 0) return 0;
  const double s0 = svd.singularValues()(0);
  const double tol = std::numeric_limits<double>::epsilon() *
                     static_cast<double>(std::max(A.rows(), A.cols())) *
                     std::max(1.0, s0);
  Eigen::Index r = 0;
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
    if (svd.singularValues()(i) > tol) ++r;
  }
  return r;
}

fit_expected<std::int64_t> total_n_obs(const data::OrdinalStats& s) {
  std::int64_t total = 0;
  for (auto n : s.n_obs) total += n;
  if (total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats has non-positive total n_obs"));
  }
  return total;
}

fit_expected<std::int64_t> total_n_obs(const data::OrdinalMoments& s) {
  std::int64_t total = 0;
  for (auto n : s.n_obs) total += n;
  if (total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalMoments has non-positive total n_obs"));
  }
  return total;
}

fit_expected<std::int64_t> total_n_obs(const data::MixedOrdinalStats& s) {
  std::int64_t total = 0;
  for (auto n : s.n_obs) total += n;
  if (total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats has non-positive total n_obs"));
  }
  return total;
}

data::OrdinalStats stats_adapter(const data::OrdinalMoments& moments) {
  data::OrdinalStats stats;
  stats.R = moments.R;
  stats.thresholds = moments.thresholds;
  stats.threshold_ov = moments.threshold_ov;
  stats.threshold_level = moments.threshold_level;
  stats.n_obs = moments.n_obs;
  stats.n_levels = moments.n_levels;
  stats.ov_names = moments.ov_names;
  return stats;
}

data::MixedOrdinalStats stats_adapter(const data::MixedOrdinalMoments& moments) {
  data::MixedOrdinalStats stats;
  stats.R = moments.R;
  stats.mean = moments.mean;
  stats.ordered = moments.ordered;
  stats.thresholds = moments.thresholds;
  stats.threshold_ov = moments.threshold_ov;
  stats.threshold_level = moments.threshold_level;
  stats.moments = moments.moments;
  stats.n_obs = moments.n_obs;
  stats.n_levels = moments.n_levels;
  stats.ov_names = moments.ov_names;
  return stats;
}

fit_expected<void> validate_stats(const data::OrdinalStats& s,
                                  const model::MatrixRep& rep,
                                  OrdinalWeightKind kind) {
  const std::size_t nb = rep.dims.size();
  if (s.R.size() != nb || s.thresholds.size() != nb ||
      s.threshold_ov.size() != nb || s.threshold_level.size() != nb ||
      s.n_obs.size() != nb || s.n_levels.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats block counts do not match MatrixRep"));
  }
  // ULS uses the identity weight (materialized on demand in weight_factors /
  // robust_ordinal), so only the NACOV / moment blocks are validated here; the
  // weight is trivially positive definite and sized to NACOV.
  const bool uls = kind == OrdinalWeightKind::ULS;
  static const std::vector<Eigen::MatrixXd> kNoWeights;
  const auto& Ws = uls ? kNoWeights
                 : (kind == OrdinalWeightKind::DWLS ? s.W_dwls : s.W_wls);
  if (s.NACOV.size() != nb || (!uls && Ws.size() != nb)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats weight/NACOV block count does not match MatrixRep"));
  }
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = rep.dims[b].n_observed;
    if (s.R[b].rows() != p || s.R[b].cols() != p) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats R dimension mismatch in block " + std::to_string(b)));
    }
    if (static_cast<Eigen::Index>(s.threshold_ov[b].size()) != s.thresholds[b].size() ||
        static_cast<Eigen::Index>(s.threshold_level[b].size()) != s.thresholds[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats threshold metadata mismatch in block " + std::to_string(b)));
    }
    const Eigen::Index mdim = s.thresholds[b].size() + p * (p - 1) / 2;
    if (s.NACOV[b].rows() != mdim || s.NACOV[b].cols() != mdim ||
        (!uls && (Ws[b].rows() != mdim || Ws[b].cols() != mdim))) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats moment/weight dimension mismatch in block " +
              std::to_string(b)));
    }
    if (s.n_obs[b] <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats n_obs must be positive in block " + std::to_string(b)));
    }
    if (!matrix_all_finite(s.R[b]) || !vector_all_finite(s.thresholds[b]) ||
        !matrix_all_finite(s.NACOV[b]) ||
        (!uls && !matrix_all_finite(Ws[b]))) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats contains non-finite values in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < s.NACOV[b].rows(); ++k) {
      if (!(s.NACOV[b](k, k) > 0.0)) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "OrdinalStats NACOV diagonal is not positive in block " +
                std::to_string(b)));
      }
    }
    if (!uls) {
      Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "OrdinalStats weight matrix is not positive definite in block " +
                std::to_string(b)));
      }
    }
  }
  return {};
}

fit_expected<void> validate_moments(const data::OrdinalMoments& s,
                                    const model::MatrixRep& rep) {
  const std::size_t nb = rep.dims.size();
  if (s.R.size() != nb || s.thresholds.size() != nb ||
      s.threshold_ov.size() != nb || s.threshold_level.size() != nb ||
      s.n_obs.size() != nb || s.n_levels.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalMoments block counts do not match MatrixRep"));
  }
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = rep.dims[b].n_observed;
    if (s.R[b].rows() != p || s.R[b].cols() != p) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalMoments R dimension mismatch in block " + std::to_string(b)));
    }
    if (static_cast<Eigen::Index>(s.threshold_ov[b].size()) !=
            s.thresholds[b].size() ||
        static_cast<Eigen::Index>(s.threshold_level[b].size()) !=
            s.thresholds[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalMoments threshold metadata mismatch in block " +
              std::to_string(b)));
    }
    if (s.n_obs[b] <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalMoments n_obs must be positive in block " +
              std::to_string(b)));
    }
    if (!matrix_all_finite(s.R[b]) || !vector_all_finite(s.thresholds[b])) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalMoments contains non-finite values in block " +
              std::to_string(b)));
    }
  }
  return {};
}

fit_expected<void> validate_stats(const data::MixedOrdinalStats& s,
                                  const model::MatrixRep& rep,
                                  OrdinalWeightKind kind) {
  const std::size_t nb = rep.dims.size();
  if (s.R.size() != nb || s.mean.size() != nb || s.ordered.size() != nb ||
      s.thresholds.size() != nb || s.threshold_ov.size() != nb ||
      s.threshold_level.size() != nb || s.moments.size() != nb ||
      s.n_obs.size() != nb || s.n_levels.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats block counts do not match MatrixRep"));
  }
  const bool uls = kind == OrdinalWeightKind::ULS;
  static const std::vector<Eigen::MatrixXd> kNoWeights;
  const auto& Ws = uls ? kNoWeights
                 : (kind == OrdinalWeightKind::DWLS ? s.W_dwls : s.W_wls);
  if (s.NACOV.size() != nb || (!uls && Ws.size() != nb)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats weight/NACOV block count does not match MatrixRep"));
  }
  if (kind == OrdinalWeightKind::WLS) {
    for (std::size_t b = 0; b < nb; ++b) {
      if (Ws[b].size() == 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats full-WLS weight was not materialized (stats "
            "built with full_wls_weight = false); rebuild with "
            "full_wls_weight = true to fit WLS"));
      }
    }
  }
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = rep.dims[b].n_observed;
    if (s.R[b].rows() != p || s.R[b].cols() != p || s.mean[b].size() != p ||
        s.ordered[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats dimension mismatch in block " + std::to_string(b)));
    }
    if (static_cast<Eigen::Index>(s.threshold_ov[b].size()) != s.thresholds[b].size() ||
        static_cast<Eigen::Index>(s.threshold_level[b].size()) != s.thresholds[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats threshold metadata mismatch in block " +
              std::to_string(b)));
    }
    Eigen::Index n_cont = 0;
    std::vector<char> has_threshold(static_cast<std::size_t>(p), 0);
    for (Eigen::Index j = 0; j < p; ++j) {
      const std::int32_t flag = s.ordered[b][static_cast<std::size_t>(j)];
      if (flag != 0 && flag != 1) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats ordered mask must contain only 0/1 values in block " +
                std::to_string(b)));
      }
      if (flag == 0) ++n_cont;
    }
    for (std::int32_t ov : s.threshold_ov[b]) {
      if (ov < 0 || ov >= p) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats threshold metadata references an invalid observed "
            "variable in block " + std::to_string(b)));
      }
      if (s.ordered[b][static_cast<std::size_t>(ov)] == 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats threshold metadata references a continuous variable "
            "in block " + std::to_string(b)));
      }
      has_threshold[static_cast<std::size_t>(ov)] = 1;
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (s.ordered[b][static_cast<std::size_t>(j)] != 0 &&
          has_threshold[static_cast<std::size_t>(j)] == 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats missing thresholds for ordered variable in block " +
                std::to_string(b)));
      }
    }
    const Eigen::Index mdim = s.thresholds[b].size() + 2 * n_cont + p * (p - 1) / 2;
    if (s.moments[b].size() != mdim || s.NACOV[b].rows() != mdim ||
        s.NACOV[b].cols() != mdim ||
        (!uls && (Ws[b].rows() != mdim || Ws[b].cols() != mdim))) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats moment/weight dimension mismatch in block " +
              std::to_string(b)));
    }
    if (s.n_obs[b] <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats n_obs must be positive in block " + std::to_string(b)));
    }
    if (!matrix_all_finite(s.R[b]) || !vector_all_finite(s.mean[b]) ||
        !vector_all_finite(s.thresholds[b]) ||
        !vector_all_finite(s.moments[b]) ||
        !matrix_all_finite(s.NACOV[b]) ||
        (!uls && !matrix_all_finite(Ws[b]))) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats contains non-finite values in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < s.NACOV[b].rows(); ++k) {
      if (!(s.NACOV[b](k, k) > 0.0)) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats NACOV diagonal is not positive in block " +
                std::to_string(b)));
      }
    }
    if (!uls) {
      Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats weight matrix is not positive definite in block " +
                std::to_string(b)));
      }
    }
  }
  return {};
}

fit_expected<void> validate_moments(const data::MixedOrdinalMoments& s,
                                    const model::MatrixRep& rep) {
  const std::size_t nb = rep.dims.size();
  if (s.R.size() != nb || s.mean.size() != nb || s.ordered.size() != nb ||
      s.thresholds.size() != nb || s.threshold_ov.size() != nb ||
      s.threshold_level.size() != nb || s.moments.size() != nb ||
      s.n_obs.size() != nb || s.n_levels.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalMoments block counts do not match MatrixRep"));
  }
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = rep.dims[b].n_observed;
    if (s.R[b].rows() != p || s.R[b].cols() != p || s.mean[b].size() != p ||
        s.ordered[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalMoments dimension mismatch in block " +
              std::to_string(b)));
    }
    if (static_cast<Eigen::Index>(s.threshold_ov[b].size()) !=
            s.thresholds[b].size() ||
        static_cast<Eigen::Index>(s.threshold_level[b].size()) !=
            s.thresholds[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalMoments threshold metadata mismatch in block " +
              std::to_string(b)));
    }
    Eigen::Index n_cont = 0;
    std::vector<char> has_threshold(static_cast<std::size_t>(p), 0);
    for (Eigen::Index j = 0; j < p; ++j) {
      const std::int32_t flag = s.ordered[b][static_cast<std::size_t>(j)];
      if (flag != 0 && flag != 1) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalMoments ordered mask must contain only 0/1 values in block " +
                std::to_string(b)));
      }
      if (flag == 0) ++n_cont;
    }
    for (std::int32_t ov : s.threshold_ov[b]) {
      if (ov < 0 || ov >= p) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalMoments threshold metadata references an invalid "
            "observed variable in block " +
                std::to_string(b)));
      }
      if (s.ordered[b][static_cast<std::size_t>(ov)] == 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalMoments threshold metadata references a continuous "
            "variable in block " +
                std::to_string(b)));
      }
      has_threshold[static_cast<std::size_t>(ov)] = 1;
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (s.ordered[b][static_cast<std::size_t>(j)] != 0 &&
          has_threshold[static_cast<std::size_t>(j)] == 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalMoments missing thresholds for ordered variable in block " +
                std::to_string(b)));
      }
    }
    const Eigen::Index mdim =
        s.thresholds[b].size() + 2 * n_cont + p * (p - 1) / 2;
    if (s.moments[b].size() != mdim) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalMoments moment dimension mismatch in block " +
              std::to_string(b)));
    }
    if (s.n_obs[b] <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalMoments n_obs must be positive in block " +
              std::to_string(b)));
    }
    if (!matrix_all_finite(s.R[b]) || !vector_all_finite(s.mean[b]) ||
        !vector_all_finite(s.thresholds[b]) ||
        !vector_all_finite(s.moments[b])) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalMoments contains non-finite values in block " +
              std::to_string(b)));
    }
  }
  return {};
}

fit_expected<std::vector<std::vector<char>>>
ordered_indicator_layout(const spec::LatentStructure& pt,
                         const data::OrdinalStats& stats) {
  const std::size_t nb = stats.R.size();
  std::vector<std::vector<char>> out(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    out[b].assign(static_cast<std::size_t>(stats.R[b].rows()), 0);
    for (std::int32_t ov : stats.threshold_ov[b]) {
      if (ov < 0 || static_cast<std::size_t>(ov) >= out[b].size()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "OrdinalStats threshold metadata references an invalid observed variable"));
      }
      out[b][static_cast<std::size_t>(ov)] = 1;
    }
  }
  const std::size_t ng = static_cast<std::size_t>(pt.n_groups());
  if (ng != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats block count does not match partable group count"));
  }
  return out;
}

fit_expected<std::vector<std::vector<char>>>
ordered_indicator_layout(const spec::LatentStructure& pt,
                         const data::MixedOrdinalStats& stats) {
  const std::size_t nb = stats.R.size();
  std::vector<std::vector<char>> out(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    out[b].assign(static_cast<std::size_t>(stats.R[b].rows()), 0);
    for (std::size_t j = 0; j < stats.ordered[b].size(); ++j) {
      out[b][j] = stats.ordered[b][j] != 0 ? 1 : 0;
    }
  }
  const std::size_t ng = static_cast<std::size_t>(pt.n_groups());
  if (ng != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats block count does not match partable group count"));
  }
  return out;
}

fit_expected<std::vector<std::vector<char>>>
ordered_indicator_layout(const spec::LatentStructure& pt,
                         const data::MixedOrdinalMoments& moments) {
  const std::size_t nb = moments.R.size();
  std::vector<std::vector<char>> out(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    out[b].assign(static_cast<std::size_t>(moments.R[b].rows()), 0);
    for (std::size_t j = 0; j < moments.ordered[b].size(); ++j) {
      out[b][j] = moments.ordered[b][j] != 0 ? 1 : 0;
    }
  }
  const std::size_t ng = static_cast<std::size_t>(pt.n_groups());
  if (ng != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalMoments block count does not match partable group count"));
  }
  return out;
}

fit_expected<void>
compact_free_set(spec::LatentStructure& pt,
                 const std::vector<char>& remove_free,
                 spec::Starts* starts) {
  // remove_free is indexed by the ORIGINAL free index (1..old_n); the caller
  // sizes it from pt.n_free() *before* zeroing the removed rows. Recomputing
  // old_n from pt.n_free() here undercounts whenever a removed parameter held
  // the top free index — e.g. std.lv ordinal CFA, where the eliminated
  // response-scale variances are numbered above the (now all-free) loadings —
  // so take old_n from the metadata and only sanity-check the surviving range.
  //
  // Regression: std.lv ordinal delta fits aborted with "received inconsistent
  // metadata" because n_free() (the max free index) dropped after the top
  // response-scale variance was zeroed for elimination.
  // Guard: tests/unit/api_sem_test.cpp "ordinal EAP factor-score precision
  // tracks Monte-Carlo PRMSE" (fits a std.lv one-factor ordinal CFA).
  const std::int32_t old_n = static_cast<std::int32_t>(remove_free.size()) - 1;
  if (old_n < 0 || pt.n_free() > old_n) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "ordinal delta free-set compaction received inconsistent metadata"));
  }

  std::vector<char> seen(static_cast<std::size_t>(old_n) + 1, 0);
  std::vector<std::int32_t> new_to_old;
  new_to_old.reserve(static_cast<std::size_t>(old_n));
  auto append_by_op = [&](std::int32_t group, parse::Op op, bool matching) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.group[i] != group) continue;
      if ((pt.op[i] == op) != matching) continue;
      const std::int32_t old = pt.free[i];
      if (old <= 0 || remove_free[static_cast<std::size_t>(old)] != 0 ||
          seen[static_cast<std::size_t>(old)] != 0) {
        continue;
      }
      seen[static_cast<std::size_t>(old)] = 1;
      new_to_old.push_back(old);
    }
  };
  for (std::int32_t g = 1; g <= pt.n_groups(); ++g) {
    append_by_op(g, parse::Op::Measurement, true);
    append_by_op(g, parse::Op::Threshold, true);
    append_by_op(g, parse::Op::Threshold, false);
  }

  std::vector<std::int32_t> old_to_new(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t neu = 0; neu < new_to_old.size(); ++neu) {
    old_to_new[static_cast<std::size_t>(new_to_old[neu])] =
        static_cast<std::int32_t>(neu) + 1;
  }

  for (std::int32_t& fr : pt.free) {
    if (fr <= 0) continue;
    const std::int32_t neu = old_to_new[static_cast<std::size_t>(fr)];
    if (neu <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal delta cannot remove a response-scale variance that shares a free index"));
    }
    fr = neu;
  }

  std::vector<std::int32_t> eq_new(new_to_old.size(), 0);
  const bool have_eq =
      static_cast<std::int32_t>(pt.eq_groups.size()) == old_n;
  for (std::size_t k = 0; k < new_to_old.size(); ++k) {
    const std::int32_t old = new_to_old[k];
    eq_new[k] = have_eq ? pt.eq_groups[static_cast<std::size_t>(old - 1)]
                        : old - 1;
  }
  pt.eq_groups = std::move(eq_new);

  const std::size_t n_lin = pt.lin_constraint_d.size();
  if (n_lin > 0) {
    const std::size_t old_cols = static_cast<std::size_t>(old_n);
    if (pt.lin_constraint_R.size() != n_lin * old_cols) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal delta free-set compaction found malformed linear constraints"));
    }
    std::vector<double> R_new(n_lin * new_to_old.size(), 0.0);
    for (std::size_t r = 0; r < n_lin; ++r) {
      for (std::int32_t old = 1; old <= old_n; ++old) {
        const double v =
            pt.lin_constraint_R[r * old_cols + static_cast<std::size_t>(old - 1)];
        const std::int32_t neu = old_to_new[static_cast<std::size_t>(old)];
        if (neu > 0) {
          R_new[r * new_to_old.size() + static_cast<std::size_t>(neu - 1)] = v;
        } else if (std::abs(v) > 1e-12) {
          return std::unexpected(make_err(FitError::Kind::NumericIssue,
              "ordinal delta does not support constraints on derived response-scale variances"));
        }
      }
    }
    pt.lin_constraint_R = std::move(R_new);
  }

  if (starts != nullptr && !starts->hint.empty()) {
    std::vector<double> hint_new(new_to_old.size(),
                                 std::numeric_limits<double>::quiet_NaN());
    for (std::size_t k = 0; k < new_to_old.size(); ++k) {
      const std::size_t old = static_cast<std::size_t>(new_to_old[k] - 1);
      if (old < starts->hint.size()) hint_new[k] = starts->hint[old];
    }
    starts->hint = std::move(hint_new);
  }
  return {};
}

struct ThresholdLayout {
  std::vector<std::vector<std::int32_t>> free;
  std::vector<std::vector<double>> fixed;
  std::vector<std::vector<char>> present;
  // Per block, per observed index: 1 if that ordinal indicator's `~~`
  // self-variance is a free parameter — i.e. its response scale is released
  // (Wu-Estabrook multigroup invariance, or the theta parameterization). The
  // moment code then standardizes that indicator's implied covariances and
  // thresholds by √Σᵢᵢ instead of comparing the raw delta moment. Empty / all-0
  // for the single-group delta convention (raw comparison).
  std::vector<std::vector<char>> scale_free;
};

// Affine threshold model tau_b = c[b] + H[b] * gamma over all blocks, plus
// free-set bookkeeping for the profiled paths. H carries free thresholds as
// unit columns, equality-label merges (including cross-group invariance) as
// shared columns, and threshold-only linear equality constraints folded
// through a null-space basis. gamma is the joint reduced threshold
// coordinate; one gamma column may span several blocks.
struct ThresholdDesign {
  std::vector<std::int32_t> active_full;
  std::vector<char> is_threshold;
  Eigen::VectorXd full_template;
  std::vector<Eigen::MatrixXd> H;  // per block: nth_b x n_gamma
  std::vector<Eigen::VectorXd> c;  // per block: nth_b
  Eigen::Index n_gamma = 0;
  // Linear-constraint rows absorbed into H/c (threshold-only rows). They are
  // satisfied identically by the profiled thresholds and must be dropped from
  // any reduced partable handed to downstream constraint machinery.
  std::vector<char> absorbed_lin_row;
};

struct ThresholdFixedProfile {
  spec::LatentStructure pt;
  Eigen::VectorXd x0;
  std::vector<std::int32_t> reduced_to_full;
  ThresholdDesign design;
  Eigen::VectorXd full_template;
};

// Profiled-threshold weight workspace. The threshold map is joint across
// blocks: with shared gamma coordinates the profiled thresholds of block b
// depend on every block's correlation residual, so threshold_from_corr maps
// the stacked correlation residual (length ncorr_total) into block b's
// threshold rows.
struct ProfiledWeightBlock {
  Eigen::MatrixXd factor;               // mdim x mdim weight factor
  Eigen::VectorXd threshold_offset;     // nth: intercept - sample thresholds
  Eigen::MatrixXd threshold_from_corr;  // nth x ncorr_total
  Eigen::VectorXd threshold_intercept;  // nth
};

struct ProfiledWeightWorkspace {
  std::vector<ProfiledWeightBlock> blocks;
  std::vector<Eigen::Index> corr_offset;  // per-block start in stacked d_corr
  Eigen::Index ncorr_total = 0;
};

fit_expected<ThresholdLayout>
make_threshold_layout(const spec::LatentStructure& pt,
                      const model::MatrixRep& rep,
                      const data::OrdinalStats& stats) {
  ThresholdLayout out;
  const std::size_t nb = rep.dims.size();
  out.free.resize(nb);
  out.fixed.resize(nb);
  out.present.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t nth = static_cast<std::size_t>(stats.thresholds[b].size());
    out.free[b].assign(nth, 0);
    out.fixed[b].assign(nth, 0.0);
    out.present[b].assign(nth, 0);
  }

  // Match rows by row order per variable. The R helper generates rows in this
  // deterministic order (`x | t1`, `x | t2`, ...).
  std::vector<std::vector<std::int32_t>> seen;
  seen.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    seen[b].assign(static_cast<std::size_t>(rep.dims[b].n_observed), 0);
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Threshold || pt.group[i] <= 0) continue;
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= nb) continue;
    const std::int32_t ov = pt.lhs_var[i] >= 0
        ? pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])]
        : -1;
    if (ov < 0 || static_cast<std::size_t>(ov) >= seen[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal threshold row references a non-observed variable"));
    }
    const std::int32_t lev = ++seen[b][static_cast<std::size_t>(ov)];
    Eigen::Index pos = -1;
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      if (stats.threshold_ov[b][static_cast<std::size_t>(k)] == ov &&
          stats.threshold_level[b][static_cast<std::size_t>(k)] == lev) {
        pos = k;
        break;
      }
    }
    if (pos < 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal threshold row has no matching sample threshold"));
    }
    out.free[b][static_cast<std::size_t>(pos)] = pt.free[i];
    out.fixed[b][static_cast<std::size_t>(pos)] =
        std::isfinite(pt.fixed_value[i]) ? pt.fixed_value[i] : 0.0;
    out.present[b][static_cast<std::size_t>(pos)] = 1;
  }
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t k = 0; k < out.free[b].size(); ++k) {
      if (!out.present[b][k]) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "missing ordinal threshold row for block " + std::to_string(b)));
      }
    }
  }

  // Response-scale-release mask: an ordinal indicator whose `~~` self-variance
  // survived `prepare_*` as a free parameter (Wu-Estabrook invariance release,
  // or theta parameterization) is compared in standardized form.
  out.scale_free.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    out.scale_free[b].assign(
        static_cast<std::size_t>(rep.dims[b].n_observed), 0);
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Covariance || pt.group[i] <= 0 ||
        pt.free[i] <= 0) {
      continue;
    }
    if (pt.lhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i]) continue;
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= nb) continue;
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
    if (ov < 0 ||
        static_cast<std::size_t>(ov) >= out.scale_free[b].size()) {
      continue;
    }
    out.scale_free[b][static_cast<std::size_t>(ov)] = 1;
  }
  return out;
}

fit_expected<ThresholdDesign>
make_threshold_design(const spec::LatentStructure& pt,
                      const ThresholdLayout& layout,
                      const data::OrdinalStats& stats,
                      const Eigen::VectorXd& x0) {
  const std::int32_t n_free = pt.n_free();
  if (x0.size() != n_free) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ordinal threshold profiling received a start vector with the wrong size"));
  }
  ThresholdDesign design;
  design.is_threshold.assign(static_cast<std::size_t>(n_free), 0);
  design.full_template = x0;

  const std::size_t nb = stats.thresholds.size();
  for (std::size_t b = 0; b < nb; ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr < 0 || fr > n_free) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "ordinal threshold profiling found an invalid threshold free index"));
      }
      if (fr == 0) continue;
      design.is_threshold[static_cast<std::size_t>(fr - 1)] = 1;
      design.full_template(fr - 1) = stats.thresholds[b](k);
    }
  }
  // A free index driving both a threshold row and a non-threshold row cannot
  // be profiled out.
  for (std::size_t row = 0; row < pt.size(); ++row) {
    const std::int32_t fr = pt.free[row];
    if (fr <= 0 || fr > n_free) continue;
    if (design.is_threshold[static_cast<std::size_t>(fr - 1)] != 0 &&
        pt.op[row] != parse::Op::Threshold) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal threshold profiling does not support a free parameter shared between threshold and non-threshold rows"));
    }
  }

  const bool have_eq =
      static_cast<std::int32_t>(pt.eq_groups.size()) == n_free;
  std::int32_t max_group = -1;
  if (have_eq) {
    for (std::int32_t k = 0; k < n_free; ++k) {
      const std::int32_t group = pt.eq_groups[static_cast<std::size_t>(k)];
      if (group < 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "ordinal threshold profiling found malformed equality groups"));
      }
      max_group = std::max(max_group, group);
    }
    std::vector<char> group_has_other(static_cast<std::size_t>(max_group + 1), 0);
    for (std::int32_t k = 0; k < n_free; ++k) {
      const std::int32_t group = pt.eq_groups[static_cast<std::size_t>(k)];
      if (design.is_threshold[static_cast<std::size_t>(k)] == 0) {
        group_has_other[static_cast<std::size_t>(group)] = 1;
      }
    }
    for (std::int32_t t = 0; t < n_free; ++t) {
      if (design.is_threshold[static_cast<std::size_t>(t)] == 0) continue;
      const std::int32_t group = pt.eq_groups[static_cast<std::size_t>(t)];
      if (group_has_other[static_cast<std::size_t>(group)] != 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "ordinal threshold profiling does not support equality constraints between thresholds and non-threshold parameters"));
      }
    }
  }

  // Preliminary coordinate per threshold merge class (equality-label group,
  // possibly spanning blocks), in deterministic (block, row) first-appearance
  // order. Without equality groups every distinct free index is its own
  // class.
  std::vector<std::int32_t> coord_of_free(static_cast<std::size_t>(n_free), -1);
  std::vector<std::int32_t> group_to_coord(
      have_eq ? static_cast<std::size_t>(max_group + 1) : 0, -1);
  Eigen::Index m = 0;
  for (std::size_t b = 0; b < nb; ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr <= 0) continue;
      std::int32_t& coord = coord_of_free[static_cast<std::size_t>(fr - 1)];
      if (coord >= 0) continue;
      if (have_eq) {
        const std::int32_t group = pt.eq_groups[static_cast<std::size_t>(fr - 1)];
        std::int32_t& gc = group_to_coord[static_cast<std::size_t>(group)];
        if (gc < 0) gc = static_cast<std::int32_t>(m++);
        coord = gc;
      } else {
        coord = static_cast<std::int32_t>(m++);
      }
    }
  }

  // Preliminary affine model tau_b = c0_b + H0_b * gamma0 over the merge
  // coordinates.
  std::vector<Eigen::MatrixXd> H0(nb);
  std::vector<Eigen::VectorXd> c0(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index nth = stats.thresholds[b].size();
    H0[b] = Eigen::MatrixXd::Zero(nth, m);
    c0[b] = Eigen::VectorXd::Zero(nth);
    for (Eigen::Index k = 0; k < nth; ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0) {
        H0[b](k, coord_of_free[static_cast<std::size_t>(fr - 1)]) = 1.0;
      } else {
        c0[b](k) = layout.fixed[b][static_cast<std::size_t>(k)];
      }
    }
  }

  // Threshold-only linear equality rows fold into the design through a
  // null-space basis; rows mixing threshold and non-threshold columns are
  // rejected.
  const std::size_t n_lin = pt.lin_constraint_d.size();
  design.absorbed_lin_row.assign(n_lin, 0);
  Eigen::MatrixXd Rc;
  Eigen::VectorXd dc;
  Eigen::Index n_abs = 0;
  if (n_lin > 0) {
    const std::size_t n_cols = static_cast<std::size_t>(n_free);
    if (pt.lin_constraint_R.size() != n_lin * n_cols) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal threshold profiling found malformed linear constraints"));
    }
    for (std::size_t r = 0; r < n_lin; ++r) {
      bool has_threshold = false;
      bool has_other = false;
      for (std::int32_t k = 0; k < n_free; ++k) {
        const double v =
            pt.lin_constraint_R[r * n_cols + static_cast<std::size_t>(k)];
        if (std::abs(v) <= 1e-12) continue;
        if (design.is_threshold[static_cast<std::size_t>(k)] != 0) {
          has_threshold = true;
        } else {
          has_other = true;
        }
      }
      if (has_threshold && has_other) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "ordinal threshold profiling does not support linear constraints mixing thresholds and non-threshold parameters"));
      }
      if (has_threshold) {
        design.absorbed_lin_row[r] = 1;
        ++n_abs;
      }
    }
    if (n_abs > 0) {
      Rc = Eigen::MatrixXd::Zero(n_abs, m);
      dc = Eigen::VectorXd::Zero(n_abs);
      Eigen::Index rr = 0;
      for (std::size_t r = 0; r < n_lin; ++r) {
        if (design.absorbed_lin_row[r] == 0) continue;
        for (std::int32_t k = 0; k < n_free; ++k) {
          const double v =
              pt.lin_constraint_R[r * n_cols + static_cast<std::size_t>(k)];
          if (std::abs(v) <= 1e-12) continue;
          Rc(rr, coord_of_free[static_cast<std::size_t>(k)]) += v;
        }
        dc(rr) = pt.lin_constraint_d[r];
        ++rr;
      }
    }
  }

  if (n_abs == 0) {
    design.H = std::move(H0);
    design.c = std::move(c0);
    design.n_gamma = m;
  } else {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(Rc);
    const Eigen::VectorXd gamma_p = lu.solve(dc);
    const double feas_tol = 1e-8 * (1.0 + dc.cwiseAbs().maxCoeff());
    if (!gamma_p.allFinite() ||
        (Rc * gamma_p - dc).cwiseAbs().maxCoeff() > feas_tol) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal threshold linear equality constraints are infeasible"));
    }
    const Eigen::Index ng = lu.dimensionOfKernel();
    const Eigen::MatrixXd kernel = lu.kernel();
    design.n_gamma = ng;
    design.H.resize(nb);
    design.c.resize(nb);
    for (std::size_t b = 0; b < nb; ++b) {
      design.c[b] = c0[b] + H0[b] * gamma_p;
      design.H[b] = ng > 0
          ? Eigen::MatrixXd(H0[b] * kernel.leftCols(ng))
          : Eigen::MatrixXd::Zero(stats.thresholds[b].size(), 0);
    }
  }

  design.active_full.reserve(static_cast<std::size_t>(n_free));
  for (std::int32_t k = 0; k < n_free; ++k) {
    if (design.is_threshold[static_cast<std::size_t>(k)] == 0) {
      design.active_full.push_back(k);
    }
  }
  return design;
}

Eigen::VectorXd profile_contract(const ThresholdDesign& profile,
                                 const Eigen::VectorXd& theta) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(profile.active_full.size()));
  for (Eigen::Index k = 0; k < out.size(); ++k) {
    out(k) = theta(profile.active_full[static_cast<std::size_t>(k)]);
  }
  return out;
}

Eigen::VectorXd profile_expand(const ThresholdDesign& profile,
                               const Eigen::VectorXd& active) {
  Eigen::VectorXd out = profile.full_template;
  for (Eigen::Index k = 0; k < active.size(); ++k) {
    out(profile.active_full[static_cast<std::size_t>(k)]) = active(k);
  }
  return out;
}

Eigen::MatrixXd profile_jacobian(const ThresholdDesign& profile,
                                 const Eigen::MatrixXd& J_full) {
  Eigen::MatrixXd out(J_full.rows(),
                      static_cast<Eigen::Index>(profile.active_full.size()));
  for (Eigen::Index k = 0; k < out.cols(); ++k) {
    out.col(k) = J_full.col(profile.active_full[static_cast<std::size_t>(k)]);
  }
  return out;
}

Bounds profile_bounds(const ThresholdDesign& profile, const Bounds& bounds) {
  if (bounds.empty()) return {};
  Bounds out;
  const Eigen::Index n = static_cast<Eigen::Index>(profile.active_full.size());
  out.lower.resize(n);
  out.upper.resize(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    const Eigen::Index full = profile.active_full[static_cast<std::size_t>(k)];
    out.lower(k) = bounds.lower(full);
    out.upper(k) = bounds.upper(full);
  }
  return out;
}

fit_expected<std::vector<gmm::GpBlockKind>>
ordinal_gp_block_kinds(const spec::LatentStructure& pt,
                       const ThresholdLayout& layout,
                       const model::ModelEvaluator& ev,
                       OrdinalParameterization parameterization) {
  std::vector<gmm::GpBlockKind> out(
      static_cast<std::size_t>(pt.n_free()), gmm::GpBlockKind::Nonlinear);
  std::vector<char> is_threshold(static_cast<std::size_t>(pt.n_free()), 0);
  for (std::size_t b = 0; b < layout.free.size(); ++b) {
    for (const std::int32_t fr : layout.free[b]) {
      if (fr <= 0) continue;
      if (fr > pt.n_free()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_snlls_full_thresholds: threshold free index out of range"));
      }
      is_threshold[static_cast<std::size_t>(fr - 1)] = 1;
      out[static_cast<std::size_t>(fr - 1)] = gmm::GpBlockKind::Linear;
    }
  }
  if (parameterization == OrdinalParameterization::Theta) return out;

  const auto locs = ev.param_locations();
  if (static_cast<std::int32_t>(locs.size()) != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_snlls_full_thresholds: parameter-location count does not match n_free"));
  }
  for (std::int32_t k = 0; k < pt.n_free(); ++k) {
    if (is_threshold[static_cast<std::size_t>(k)] != 0) continue;
    switch (locs[static_cast<std::size_t>(k)].mat) {
      case model::MatId::Lambda:
      case model::MatId::Beta:
        out[static_cast<std::size_t>(k)] = gmm::GpBlockKind::Nonlinear;
        break;
      case model::MatId::Theta:
      case model::MatId::Psi:
      case model::MatId::Nu:
      case model::MatId::Alpha:
        out[static_cast<std::size_t>(k)] = gmm::GpBlockKind::Linear;
        break;
    }
  }
  return out;
}

fit_expected<void>
ensure_no_unprofiled_equality_constraints(const spec::LatentStructure& pt,
                                          const ThresholdDesign& profile) {
  const std::int32_t n_free = pt.n_free();
  if (!pt.lin_constraint_d.empty()) {
    // Threshold-only linear rows are absorbed into the threshold design and
    // satisfied identically by the profiled thresholds; rows touching
    // non-threshold parameters would be silently dropped by the profiled
    // solver and must be rejected.
    const std::size_t n_lin = pt.lin_constraint_d.size();
    const std::size_t n_cols = static_cast<std::size_t>(n_free);
    if (pt.lin_constraint_R.size() != n_lin * n_cols) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: malformed linear constraints"));
    }
    for (std::size_t r = 0; r < n_lin; ++r) {
      for (std::int32_t k = 0; k < n_free; ++k) {
        const double v =
            pt.lin_constraint_R[r * n_cols + static_cast<std::size_t>(k)];
        if (std::abs(v) > 1e-12 &&
            profile.is_threshold[static_cast<std::size_t>(k)] == 0) {
          return std::unexpected(make_err(FitError::Kind::NumericIssue,
              "fit_ordinal_bounded: profiled ordinal fit-only path does not yet support linear equality constraints on non-threshold parameters"));
        }
      }
    }
  }
  if (static_cast<std::int32_t>(pt.eq_groups.size()) != n_free) return {};

  std::int32_t max_group = -1;
  for (std::int32_t k = 0; k < n_free; ++k) {
    const std::int32_t group = pt.eq_groups[static_cast<std::size_t>(k)];
    if (group < 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: malformed equality groups"));
    }
    max_group = std::max(max_group, group);
  }
  std::vector<std::int32_t> group_size(static_cast<std::size_t>(max_group + 1), 0);
  std::vector<char> group_has_other(static_cast<std::size_t>(max_group + 1), 0);
  for (std::int32_t k = 0; k < n_free; ++k) {
    const std::int32_t group = pt.eq_groups[static_cast<std::size_t>(k)];
    ++group_size[static_cast<std::size_t>(group)];
    if (profile.is_threshold[static_cast<std::size_t>(k)] == 0) {
      group_has_other[static_cast<std::size_t>(group)] = 1;
    }
  }
  for (std::size_t g = 0; g < group_size.size(); ++g) {
    if (group_size[g] > 1 && group_has_other[g] != 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: profiled ordinal fit-only path does not yet support non-threshold equality constraints"));
    }
  }
  return {};
}

fit_expected<ThresholdFixedProfile>
fix_thresholds_for_snlls(spec::LatentStructure pt,
                         const ThresholdLayout& layout,
                         const data::OrdinalStats& stats,
                         const Eigen::VectorXd& x0) {
  auto design_or = make_threshold_design(pt, layout, stats, x0);
  if (!design_or.has_value()) return std::unexpected(design_or.error());
  ThresholdDesign design = std::move(*design_or);

  // Threshold-only linear rows are absorbed by the threshold design; drop
  // them before free-set compaction so the remaining constraint machinery
  // only sees non-threshold rows.
  if (!pt.lin_constraint_d.empty()) {
    const std::size_t n_lin = pt.lin_constraint_d.size();
    const std::size_t n_cols = static_cast<std::size_t>(pt.n_free());
    std::vector<double> R_keep;
    std::vector<double> d_keep;
    for (std::size_t r = 0; r < n_lin; ++r) {
      if (r < design.absorbed_lin_row.size() &&
          design.absorbed_lin_row[r] != 0) {
        continue;
      }
      for (std::size_t k = 0; k < n_cols; ++k) {
        R_keep.push_back(pt.lin_constraint_R[r * n_cols + k]);
      }
      d_keep.push_back(pt.lin_constraint_d[r]);
    }
    pt.lin_constraint_R = std::move(R_keep);
    pt.lin_constraint_d = std::move(d_keep);
  }

  const std::int32_t old_n = pt.n_free();
  const std::vector<std::int32_t> old_free = pt.free;
  std::vector<char> remove_free(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t row = 0; row < pt.size(); ++row) {
    if (pt.op[row] != parse::Op::Threshold || pt.group[row] <= 0) continue;
    const std::size_t b = static_cast<std::size_t>(pt.group[row] - 1);
    if (b >= layout.free.size()) continue;
    const std::int32_t fr = pt.free[row];
    if (fr <= 0 || fr > old_n) continue;
    Eigen::Index kth = -1;
    for (Eigen::Index k = 0;
         k < static_cast<Eigen::Index>(layout.free[b].size()); ++k) {
      if (layout.free[b][static_cast<std::size_t>(k)] == fr) {
        kth = k;
        break;
      }
    }
    if (kth < 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_snlls: threshold row has no profiling metadata"));
    }
    remove_free[static_cast<std::size_t>(fr)] = 1;
    pt.free[row] = 0;
    pt.fixed_value[row] = stats.thresholds[b](kth);
  }

  auto compact = compact_free_set(pt, remove_free, nullptr);
  if (!compact.has_value()) return std::unexpected(compact.error());

  ThresholdFixedProfile out;
  out.pt = std::move(pt);
  out.x0 = Eigen::VectorXd::Zero(out.pt.n_free());
  out.reduced_to_full.assign(static_cast<std::size_t>(out.pt.n_free()), 0);
  out.full_template = design.full_template;
  out.design = std::move(design);
  std::vector<char> seen(static_cast<std::size_t>(out.pt.n_free()), 0);
  for (std::size_t row = 0; row < out.pt.size(); ++row) {
    const std::int32_t old = old_free[row];
    const std::int32_t neu = out.pt.free[row];
    if (old <= 0 || neu <= 0) continue;
    if (old > x0.size() || neu > out.x0.size()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fit_ordinal_snlls: threshold-fixed start mapping is inconsistent"));
    }
    const std::size_t j = static_cast<std::size_t>(neu - 1);
    out.x0(neu - 1) = x0(old - 1);
    out.reduced_to_full[j] = old;
    seen[j] = 1;
  }
  for (std::size_t j = 0; j < seen.size(); ++j) {
    if (seen[j] == 0 || out.reduced_to_full[j] <= 0) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fit_ordinal_snlls: threshold-fixed start vector is incomplete"));
    }
  }
  return out;
}

fit_expected<Eigen::VectorXd>
expand_threshold_fixed_theta(const ThresholdFixedProfile& profile,
                             const Eigen::VectorXd& theta_reduced) {
  if (theta_reduced.size() !=
      static_cast<Eigen::Index>(profile.reduced_to_full.size())) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_snlls: profiled solution has the wrong length"));
  }
  Eigen::VectorXd out = profile.full_template;
  for (Eigen::Index k = 0; k < theta_reduced.size(); ++k) {
    const std::int32_t full = profile.reduced_to_full[static_cast<std::size_t>(k)];
    if (full <= 0 || full > out.size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_snlls: profiled solution mapping is inconsistent"));
    }
    out(full - 1) = theta_reduced(k);
  }
  return out;
}

fit_expected<ThresholdLayout>
make_threshold_layout(const spec::LatentStructure& pt,
                      const model::MatrixRep& rep,
                      const data::MixedOrdinalStats& stats) {
  ThresholdLayout out;
  const std::size_t nb = rep.dims.size();
  out.free.resize(nb);
  out.fixed.resize(nb);
  out.present.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t nth = static_cast<std::size_t>(stats.thresholds[b].size());
    out.free[b].assign(nth, 0);
    out.fixed[b].assign(nth, 0.0);
    out.present[b].assign(nth, 0);
  }

  std::vector<std::vector<std::int32_t>> seen(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    seen[b].assign(static_cast<std::size_t>(rep.dims[b].n_observed), 0);
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Threshold || pt.group[i] <= 0) continue;
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= nb) continue;
    const std::int32_t ov = pt.lhs_var[i] >= 0
        ? pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])]
        : -1;
    if (ov < 0 || static_cast<std::size_t>(ov) >= seen[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "mixed ordinal threshold row references a non-observed variable"));
    }
    const std::int32_t lev = ++seen[b][static_cast<std::size_t>(ov)];
    Eigen::Index pos = -1;
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      if (stats.threshold_ov[b][static_cast<std::size_t>(k)] == ov &&
          stats.threshold_level[b][static_cast<std::size_t>(k)] == lev) {
        pos = k;
        break;
      }
    }
    if (pos < 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "mixed ordinal threshold row has no matching sample threshold"));
    }
    out.free[b][static_cast<std::size_t>(pos)] = pt.free[i];
    out.fixed[b][static_cast<std::size_t>(pos)] =
        std::isfinite(pt.fixed_value[i]) ? pt.fixed_value[i] : 0.0;
    out.present[b][static_cast<std::size_t>(pos)] = 1;
  }
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t k = 0; k < out.free[b].size(); ++k) {
      if (!out.present[b][k]) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "missing mixed ordinal threshold row for block " + std::to_string(b)));
      }
    }
  }
  return out;
}

fit_expected<std::vector<Eigen::MatrixXd>>
weight_factors(const data::OrdinalStats& stats, OrdinalWeightKind kind) {
  // ULS uses the identity weight, so its Cholesky factor is the identity. No
  // NACOV inverse is involved; the residual is the raw moment residual s − σ.
  if (kind == OrdinalWeightKind::ULS) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      out.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
    return out;
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<Eigen::MatrixXd> out;
  out.reserve(Ws.size());
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal weight matrix is not positive definite in block " +
              std::to_string(b)));
    }
    out.push_back(llt.matrixL());
  }
  return out;
}

fit_expected<std::vector<Eigen::MatrixXd>>
full_weight_factors(const data::OrdinalMoments& moments,
                    data::OrdinalGammaCache* cache,
                    const data::OrdinalWeightPlan& plan) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(moments.R.size());
  if (plan.purpose == data::OrdinalWorkspacePurpose::InferenceOnly) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_snlls_full_thresholds: OrdinalWeightPlan purpose cannot be inference-only"));
  }
  const bool fit_plus_inference =
      plan.purpose == data::OrdinalWorkspacePurpose::FitPlusInference;

  if (plan.estimator == data::OrdinalEstimatorKind::ULS) {
    if (fit_plus_inference && cache == nullptr) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_snlls_full_thresholds: ULS fit-plus-inference requires an OrdinalGammaCache"));
    }
    for (std::size_t b = 0; b < moments.R.size(); ++b) {
      const Eigen::Index p = moments.R[b].rows();
      const Eigen::Index mdim =
          moments.thresholds[b].size() + p * (p - 1) / 2;
      out.push_back(Eigen::MatrixXd::Identity(mdim, mdim));
    }
    return out;
  }

  if (cache == nullptr) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_snlls_full_thresholds: weighted ordinal fit requires an OrdinalGammaCache"));
  }
  if (cache->blocks.size() != moments.R.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_snlls_full_thresholds: OrdinalGammaCache block count mismatch"));
  }

  if (plan.estimator == data::OrdinalEstimatorKind::WLS) {
    if (plan.materialization != data::OrdinalGammaMaterialization::Full) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_snlls_full_thresholds: WLS requires full Gamma"));
    }
    auto ok = data::ordinal_gamma_cache_ensure_wls_weights(*cache);
    if (!ok.has_value()) return std::unexpected(post_to_fit(ok.error()));
    for (std::size_t b = 0; b < moments.R.size(); ++b) {
      const Eigen::Index p = moments.R[b].rows();
      const Eigen::Index mdim =
          moments.thresholds[b].size() + p * (p - 1) / 2;
      const Eigen::MatrixXd& W = cache->blocks[b].w_wls;
      if (W.rows() != mdim || W.cols() != mdim || !matrix_all_finite(W)) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_snlls_full_thresholds: WLS weight dimension mismatch in block " +
                std::to_string(b)));
      }
      Eigen::LLT<Eigen::MatrixXd> llt(W);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_snlls_full_thresholds: WLS weight is not positive definite in block " +
                std::to_string(b)));
      }
      out.push_back(llt.matrixL());
    }
    return out;
  }

  const data::OrdinalGammaMaterialization expected =
      fit_plus_inference ? data::OrdinalGammaMaterialization::Full
                         : data::OrdinalGammaMaterialization::Diagonal;
  if (plan.materialization != expected) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        fit_plus_inference
            ? "fit_ordinal_snlls_full_thresholds: DWLS fit-plus-inference requires full Gamma"
            : "fit_ordinal_snlls_full_thresholds: DWLS fit-only requires diagonal Gamma"));
  }
  post_expected<void> ok =
      fit_plus_inference ? data::ordinal_gamma_cache_ensure_dwls_weights(*cache)
                         : data::ordinal_gamma_cache_ensure_diagonal(*cache);
  if (!ok.has_value()) return std::unexpected(post_to_fit(ok.error()));
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const Eigen::Index p = moments.R[b].rows();
    const Eigen::Index mdim =
        moments.thresholds[b].size() + p * (p - 1) / 2;
    const Eigen::VectorXd diagonal = fit_plus_inference
                                         ? cache->blocks[b].w_dwls.diagonal()
                                         : cache->blocks[b].diagonal;
    if (diagonal.size() != mdim || !vector_all_finite(diagonal)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_snlls_full_thresholds: DWLS diagonal dimension mismatch in block " +
              std::to_string(b)));
    }
    Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = fit_plus_inference ? diagonal(k) : 1.0 / diagonal(k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_snlls_full_thresholds: DWLS diagonal is not positive in block " +
                std::to_string(b)));
      }
      factor(k, k) = std::sqrt(v);
    }
    out.push_back(std::move(factor));
  }
  return out;
}

fit_expected<std::pair<data::MixedOrdinalStats, OrdinalWeightKind>>
mixed_stats_from_moments_cache(const data::MixedOrdinalMoments& moments,
                               data::OrdinalGammaCache* cache,
                               const data::OrdinalWeightPlan& plan,
                               std::string_view context) {
  if (plan.purpose == data::OrdinalWorkspacePurpose::InferenceOnly) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(context) + ": OrdinalWeightPlan purpose cannot be inference-only"));
  }
  if (plan.estimator != data::OrdinalEstimatorKind::DWLS &&
      plan.estimator != data::OrdinalEstimatorKind::WLS) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(context) + ": mixed ordinal cache-aware fitting supports DWLS/WLS"));
  }
  if (cache == nullptr) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(context) + ": weighted mixed ordinal fit requires an OrdinalGammaCache"));
  }
  if (cache->blocks.size() != moments.R.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(context) + ": OrdinalGammaCache block count mismatch"));
  }

  data::MixedOrdinalStats stats = stats_adapter(moments);
  stats.NACOV.reserve(moments.R.size());
  const bool fit_plus_inference =
      plan.purpose == data::OrdinalWorkspacePurpose::FitPlusInference;

  if (plan.estimator == data::OrdinalEstimatorKind::WLS) {
    if (plan.materialization != data::OrdinalGammaMaterialization::Full) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(context) + ": WLS requires full Gamma"));
    }
    auto ok = data::ordinal_gamma_cache_ensure_wls_weights(*cache);
    if (!ok.has_value()) return std::unexpected(post_to_fit(ok.error()));
    stats.W_wls.reserve(moments.R.size());
    for (std::size_t b = 0; b < moments.R.size(); ++b) {
      const Eigen::Index mdim = moments.moments[b].size();
      const auto& block = cache->blocks[b];
      if (!block.has_full || block.gamma.rows() != mdim ||
          block.gamma.cols() != mdim || !matrix_all_finite(block.gamma) ||
          block.w_wls.rows() != mdim || block.w_wls.cols() != mdim ||
          !matrix_all_finite(block.w_wls)) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            std::string(context) + ": WLS cache dimension mismatch in block " +
                std::to_string(b)));
      }
      stats.NACOV.push_back(block.gamma);
      stats.W_wls.push_back(block.w_wls);
    }
    return std::pair{std::move(stats), OrdinalWeightKind::WLS};
  }

  const data::OrdinalGammaMaterialization expected =
      fit_plus_inference ? data::OrdinalGammaMaterialization::Full
                         : data::OrdinalGammaMaterialization::Diagonal;
  if (plan.materialization != expected) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        fit_plus_inference
            ? std::string(context) + ": DWLS fit-plus-inference requires full Gamma"
            : std::string(context) + ": DWLS fit-only requires diagonal Gamma"));
  }
  post_expected<void> ok =
      fit_plus_inference ? data::ordinal_gamma_cache_ensure_dwls_weights(*cache)
                         : data::ordinal_gamma_cache_ensure_diagonal(*cache);
  if (!ok.has_value()) return std::unexpected(post_to_fit(ok.error()));
  stats.W_dwls.reserve(moments.R.size());
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const Eigen::Index mdim = moments.moments[b].size();
    const auto& block = cache->blocks[b];
    const Eigen::VectorXd diagonal = fit_plus_inference
                                         ? block.w_dwls.diagonal()
                                         : block.diagonal;
    if (diagonal.size() != mdim || !vector_all_finite(diagonal)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(context) + ": DWLS diagonal dimension mismatch in block " +
              std::to_string(b)));
    }
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(mdim, mdim);
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = fit_plus_inference ? diagonal(k) : 1.0 / diagonal(k);
      const double g = fit_plus_inference ? 1.0 / diagonal(k) : diagonal(k);
      if (!std::isfinite(v) || v <= 0.0 || !std::isfinite(g) || g <= 0.0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            std::string(context) + ": DWLS diagonal is not positive in block " +
                std::to_string(b)));
      }
      W(k, k) = v;
      gamma(k, k) = g;
    }
    if (fit_plus_inference && block.has_full) {
      stats.NACOV.push_back(block.gamma);
    } else {
      stats.NACOV.push_back(std::move(gamma));
    }
    stats.W_dwls.push_back(std::move(W));
  }
  return std::pair{std::move(stats), OrdinalWeightKind::DWLS};
}

// Joint cross-block threshold profiling. Minimizing the threshold block of
// the weighted objective sum_b w_b * r_b' W_b r_b over the shared gamma
// coordinate gives the normal equations
//   A gamma = sum_b w_b H_b' ( W_tt,b (tau_hat_b - c_b) - W_tr,b d_rho,b ),
//   A       = sum_b w_b H_b' W_tt,b H_b,
// with w_b = n_b / N. The block sample weights do not cancel once a gamma
// coordinate spans blocks, and block b's profiled thresholds depend on every
// block's correlation residual through A^{-1}.
fit_expected<ProfiledWeightWorkspace>
build_joint_profiled_workspace(const data::OrdinalMoments& moments,
                               const ThresholdDesign& design,
                               std::vector<Eigen::MatrixXd> Ws,
                               std::vector<Eigen::MatrixXd> factors,
                               std::string_view context) {
  const std::size_t nb = moments.R.size();
  if (design.H.size() != nb || design.c.size() != nb || Ws.size() != nb ||
      factors.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        std::string(context) + ": threshold design block count mismatch"));
  }
  auto N = total_n_obs(moments);
  if (!N.has_value()) return std::unexpected(N.error());

  ProfiledWeightWorkspace out;
  out.blocks.reserve(nb);
  out.corr_offset.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = moments.R[b].rows();
    const Eigen::Index nth = moments.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;
    out.corr_offset[b] = out.ncorr_total;
    out.ncorr_total += ncorr;
    if (Ws[b].rows() != mdim || Ws[b].cols() != mdim ||
        !matrix_all_finite(Ws[b])) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(context) + ": weight dimension mismatch in block " +
              std::to_string(b)));
    }
    if (factors[b].rows() != mdim || factors[b].cols() != mdim ||
        !matrix_all_finite(factors[b])) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(context) + ": weight factor dimension mismatch in block " +
              std::to_string(b)));
    }
    if (design.H[b].rows() != nth || design.H[b].cols() != design.n_gamma ||
        design.c[b].size() != nth) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(context) + ": threshold design dimension mismatch in block " +
              std::to_string(b)));
    }
  }

  const Eigen::Index ng = design.n_gamma;
  Eigen::VectorXd G_const;
  Eigen::MatrixXd G_corr;
  if (ng > 0) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ng, ng);
    Eigen::VectorXd r_const = Eigen::VectorXd::Zero(ng);
    Eigen::MatrixXd R_corr = Eigen::MatrixXd::Zero(ng, out.ncorr_total);
    for (std::size_t b = 0; b < nb; ++b) {
      const Eigen::Index nth = moments.thresholds[b].size();
      const Eigen::Index ncorr = Ws[b].rows() - nth;
      const double w = static_cast<double>(moments.n_obs[b]) /
                       static_cast<double>(*N);
      const Eigen::MatrixXd HtW =
          design.H[b].transpose() * Ws[b].block(0, 0, nth, nth);
      A.noalias() += w * (HtW * design.H[b]);
      r_const.noalias() +=
          w * (HtW * (moments.thresholds[b] - design.c[b]));
      if (ncorr > 0) {
        R_corr.middleCols(out.corr_offset[b], ncorr).noalias() =
            w * (design.H[b].transpose() * Ws[b].block(0, nth, nth, ncorr));
      }
    }
    A = 0.5 * (A + A.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt_A(A);
    if (llt_A.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(context) +
              ": threshold profiling normal matrix is not positive definite "
              "(rank-deficient or contradictory threshold design)"));
    }
    G_const = llt_A.solve(r_const);
    G_corr = llt_A.solve(R_corr);
    if (!G_const.allFinite() || !G_corr.allFinite()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          std::string(context) + ": threshold profiling solve failed"));
    }
  }

  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index nth = moments.thresholds[b].size();
    Eigen::VectorXd intercept = design.c[b];
    Eigen::MatrixXd from_corr = Eigen::MatrixXd::Zero(nth, out.ncorr_total);
    if (ng > 0) {
      intercept.noalias() += design.H[b] * G_const;
      from_corr.noalias() = design.H[b] * G_corr;
    }
    Eigen::VectorXd offset = intercept - moments.thresholds[b];
    out.blocks.push_back(ProfiledWeightBlock{
        .factor = std::move(factors[b]),
        .threshold_offset = std::move(offset),
        .threshold_from_corr = std::move(from_corr),
        .threshold_intercept = std::move(intercept)});
  }
  return out;
}

fit_expected<ProfiledWeightWorkspace>
profiled_weight_workspace(const data::OrdinalMoments& moments,
                          const ThresholdLayout& layout,
                          const ThresholdDesign& design,
                          data::OrdinalGammaCache* cache,
                          const data::OrdinalWeightPlan& plan) {
  (void)layout;
  if (plan.purpose == data::OrdinalWorkspacePurpose::InferenceOnly) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: OrdinalWeightPlan purpose cannot be inference-only"));
  }
  const bool fit_plus_inference =
      plan.purpose == data::OrdinalWorkspacePurpose::FitPlusInference;

  std::vector<Eigen::MatrixXd> Ws;
  std::vector<Eigen::MatrixXd> factors;
  Ws.reserve(moments.R.size());
  factors.reserve(moments.R.size());

  if (plan.estimator == data::OrdinalEstimatorKind::ULS) {
    const data::OrdinalGammaMaterialization expected =
        fit_plus_inference ? data::OrdinalGammaMaterialization::Full
                           : data::OrdinalGammaMaterialization::None;
    if (plan.materialization != expected) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          fit_plus_inference
              ? "fit_ordinal_bounded: ULS fit-plus-inference plan must request full Gamma"
              : "fit_ordinal_bounded: ULS fit-only plan must request no Gamma"));
    }
    if (fit_plus_inference) {
      if (cache == nullptr) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_bounded: ULS fit-plus-inference requires an OrdinalGammaCache"));
      }
      if (cache->blocks.size() != moments.R.size()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_bounded: OrdinalGammaCache block count mismatch"));
      }
      for (std::size_t b = 0; b < moments.R.size(); ++b) {
        const Eigen::Index p = moments.R[b].rows();
        const Eigen::Index mdim =
            moments.thresholds[b].size() + p * (p - 1) / 2;
        const auto& block = cache->blocks[b];
        if (!block.has_full || block.gamma.rows() != mdim ||
            block.gamma.cols() != mdim || !matrix_all_finite(block.gamma)) {
          return std::unexpected(make_err(FitError::Kind::NumericIssue,
              "fit_ordinal_bounded: ULS fit-plus-inference requires full Gamma in block " +
                  std::to_string(b)));
        }
      }
    }
    for (std::size_t b = 0; b < moments.R.size(); ++b) {
      const Eigen::Index p = moments.R[b].rows();
      const Eigen::Index mdim =
          moments.thresholds[b].size() + p * (p - 1) / 2;
      Ws.push_back(Eigen::MatrixXd::Identity(mdim, mdim));
      factors.push_back(Eigen::MatrixXd::Identity(mdim, mdim));
    }
    return build_joint_profiled_workspace(moments, design, std::move(Ws),
                                          std::move(factors),
                                          "fit_ordinal_bounded");
  }

  if (plan.estimator == data::OrdinalEstimatorKind::WLS) {
    if (plan.materialization != data::OrdinalGammaMaterialization::Full) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: WLS fit-only plan must request full Gamma"));
    }
    if (cache == nullptr) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: WLS requires an OrdinalGammaCache"));
    }
    if (cache->blocks.size() != moments.R.size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: OrdinalGammaCache block count mismatch"));
    }
    auto ok = data::ordinal_gamma_cache_ensure_wls_weights(*cache);
    if (!ok.has_value()) return std::unexpected(post_to_fit(ok.error()));

    for (std::size_t b = 0; b < moments.R.size(); ++b) {
      const Eigen::Index p = moments.R[b].rows();
      const Eigen::Index nth = moments.thresholds[b].size();
      const Eigen::Index mdim = nth + p * (p - 1) / 2;
      const Eigen::MatrixXd& W = cache->blocks[b].w_wls;
      if (W.rows() != mdim || W.cols() != mdim || !matrix_all_finite(W)) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_bounded: cached WLS weight dimension mismatch in block " +
                std::to_string(b)));
      }
      Eigen::LLT<Eigen::MatrixXd> llt(W);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_bounded: cached WLS weight is not positive definite in block " +
                std::to_string(b)));
      }
      Ws.push_back(W);
      factors.push_back(llt.matrixL());
    }
    return build_joint_profiled_workspace(moments, design, std::move(Ws),
                                          std::move(factors),
                                          "fit_ordinal_bounded");
  }

  const data::OrdinalGammaMaterialization expected =
      fit_plus_inference ? data::OrdinalGammaMaterialization::Full
                         : data::OrdinalGammaMaterialization::Diagonal;
  if (plan.materialization != expected) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        fit_plus_inference
            ? "fit_ordinal_bounded: DWLS fit-plus-inference plan must request full Gamma"
            : "fit_ordinal_bounded: DWLS fit-only plan must request diagonal Gamma"));
  }
  if (cache == nullptr) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: DWLS requires an OrdinalGammaCache"));
  }
  if (cache->blocks.size() != moments.R.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: OrdinalGammaCache block count mismatch"));
  }
  post_expected<void> ok =
      fit_plus_inference ? data::ordinal_gamma_cache_ensure_dwls_weights(*cache)
                         : data::ordinal_gamma_cache_ensure_diagonal(*cache);
  if (!ok.has_value()) return std::unexpected(post_to_fit(ok.error()));

  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const Eigen::Index p = moments.R[b].rows();
    const Eigen::Index nth = moments.thresholds[b].size();
    const Eigen::Index mdim = nth + p * (p - 1) / 2;
    const Eigen::VectorXd& diagonal = cache->blocks[b].diagonal;
    if (diagonal.size() != mdim || !vector_all_finite(diagonal)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: cached Gamma diagonal dimension mismatch in block " +
              std::to_string(b)));
    }
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(mdim, mdim);
    Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = diagonal(k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_bounded: cached Gamma diagonal is not positive in block " +
                std::to_string(b)));
      }
      const double w = 1.0 / v;
      W(k, k) = w;
      factor(k, k) = std::sqrt(w);
    }
    Ws.push_back(std::move(W));
    factors.push_back(std::move(factor));
  }
  return build_joint_profiled_workspace(moments, design, std::move(Ws),
                                        std::move(factors),
                                        "fit_ordinal_bounded");
}

fit_expected<std::vector<Eigen::MatrixXd>>
weight_factors(const data::MixedOrdinalStats& stats, OrdinalWeightKind kind) {
  if (kind == OrdinalWeightKind::ULS) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      out.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
    return out;
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<Eigen::MatrixXd> out;
  out.reserve(Ws.size());
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    if (Ws[b].size() == 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "mixed ordinal full-WLS weight unavailable in block " +
              std::to_string(b) +
              " (NACOV not positive definite); use DWLS"));
    }
    Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "mixed ordinal weight matrix is not positive definite in block " +
              std::to_string(b)));
    }
    out.push_back(llt.matrixL());
  }
  return out;
}

Eigen::VectorXd implied_thresholds(const ThresholdLayout& layout,
                                   const Eigen::VectorXd& theta,
                                   std::size_t b) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(layout.free[b].size()));
  for (Eigen::Index k = 0; k < out.size(); ++k) {
    const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
    out(k) = fr > 0 ? theta(fr - 1) : layout.fixed[b][static_cast<std::size_t>(k)];
  }
  return out;
}

Eigen::VectorXd corr_lower(const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = Sigma.rows();
  Eigen::VectorXd out(p * (p - 1) / 2);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out(k++) = Sigma(i, j);
    }
  }
  return out;
}

Eigen::MatrixXd corr_jacobian(const Eigen::MatrixXd& Sigma,
                              const Eigen::MatrixXd& J_sigma,
                              Eigen::Index sigma_off) {
  (void)Sigma;
  const Eigen::Index p = Sigma.rows();
  const Eigen::Index n_free = J_sigma.cols();
  Eigen::MatrixXd J(p * (p - 1) / 2, n_free);
  Eigen::Index row = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      J.row(row) = J_sigma.row(sigma_off + vech_index(p, i, j));
      ++row;
    }
  }
  return J;
}

// === Theta-parameterization helpers =========================================
// Under theta the latent-response residual variances are fixed to 1, so the
// implied total variances Σ*ᵢᵢ float; the implied moments are standardized
// before comparison with the (always unit-variance) polychoric sample moments.

// Σᵢⱼ / √(Σᵢᵢ Σⱼⱼ) over the strict lower triangle.
Eigen::VectorXd std_corr_lower(const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = Sigma.rows();
  Eigen::VectorXd sd(p);
  for (Eigen::Index i = 0; i < p; ++i) sd(i) = std::sqrt(Sigma(i, i));
  Eigen::VectorXd out(p * (p - 1) / 2);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out(k++) = Sigma(i, j) / (sd(i) * sd(j));
    }
  }
  return out;
}

// ∂[Σᵢⱼ/√(ΣᵢᵢΣⱼⱼ)]/∂θ over the strict lower triangle, from ∂vech(Σ)/∂θ.
Eigen::MatrixXd std_corr_jacobian(const Eigen::MatrixXd& Sigma,
                                  const Eigen::MatrixXd& J_sigma,
                                  Eigen::Index sigma_off) {
  const Eigen::Index p = Sigma.rows();
  Eigen::MatrixXd J(p * (p - 1) / 2, J_sigma.cols());
  Eigen::Index row = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const double sij = Sigma(i, j);
      const double sii = Sigma(i, i);
      const double sjj = Sigma(j, j);
      const double inv_i = 1.0 / std::sqrt(sii);
      const double inv_j = 1.0 / std::sqrt(sjj);
      J.row(row) =
          (inv_i * inv_j) * J_sigma.row(sigma_off + vech_index(p, i, j)) -
          (0.5 * sij * inv_i / sii * inv_j) *
              J_sigma.row(sigma_off + vech_index(p, i, i)) -
          (0.5 * sij * inv_i * inv_j / sjj) *
              J_sigma.row(sigma_off + vech_index(p, j, j));
      ++row;
    }
  }
  return J;
}

double mixed_assoc_moment(const Eigen::MatrixXd& Sigma,
                          const std::vector<std::int32_t>& ordered,
                          Eigen::Index i,
                          Eigen::Index j,
                          OrdinalParameterization param) {
  if (param == OrdinalParameterization::Delta) return Sigma(i, j);
  const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
  const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
  if (oi && oj) return Sigma(i, j) / std::sqrt(Sigma(i, i) * Sigma(j, j));
  if (oi != oj) {
    const Eigen::Index o = oi ? i : j;
    return Sigma(i, j) / std::sqrt(Sigma(o, o));
  }
  return Sigma(i, j);
}

Eigen::RowVectorXd mixed_assoc_jacobian_row(
    const Eigen::MatrixXd& Sigma,
    const Eigen::MatrixXd& J_sigma,
    Eigen::Index sigma_off,
    const std::vector<std::int32_t>& ordered,
    Eigen::Index i,
    Eigen::Index j,
    OrdinalParameterization param) {
  const Eigen::Index p = Sigma.rows();
  if (param == OrdinalParameterization::Delta) {
    return J_sigma.row(sigma_off + vech_index(p, i, j));
  }
  const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
  const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
  const double sij = Sigma(i, j);
  if (oi && oj) {
    const double sii = Sigma(i, i);
    const double sjj = Sigma(j, j);
    const double inv_i = 1.0 / std::sqrt(sii);
    const double inv_j = 1.0 / std::sqrt(sjj);
    return (inv_i * inv_j) * J_sigma.row(sigma_off + vech_index(p, i, j)) -
           (0.5 * sij * inv_i / sii * inv_j) *
               J_sigma.row(sigma_off + vech_index(p, i, i)) -
           (0.5 * sij * inv_i * inv_j / sjj) *
               J_sigma.row(sigma_off + vech_index(p, j, j));
  }
  if (oi != oj) {
    const Eigen::Index o = oi ? i : j;
    const double soo = Sigma(o, o);
    const double inv_o = 1.0 / std::sqrt(soo);
    return inv_o * J_sigma.row(sigma_off + vech_index(p, i, j)) -
           (0.5 * sij * inv_o / soo) *
               J_sigma.row(sigma_off + vech_index(p, o, o));
  }
  return J_sigma.row(sigma_off + vech_index(p, i, j));
}

fit_expected<Eigen::VectorXd>
ordinal_residuals(const data::OrdinalStats& stats,
                  const ThresholdLayout& layout,
                  const model::ImpliedMoments& moments,
                  const std::vector<Eigen::MatrixXd>& factors,
                  const Eigen::VectorXd& theta,
                  OrdinalParameterization param) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  const bool theta_param = param == OrdinalParameterization::Theta;
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    n_total += stats.thresholds[b].size() + p * (p - 1) / 2;
  }
  Eigen::VectorXd out(n_total);
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::MatrixXd& Sig = moments.sigma[b];
    Eigen::VectorXd d(nth + ncorr);
    Eigen::VectorXd it = implied_thresholds(layout, theta, b);
    const bool block_released =
        b < layout.scale_free.size() &&
        std::any_of(layout.scale_free[b].begin(), layout.scale_free[b].end(),
                    [](char c) { return c != 0; });
    if (block_released && !theta_param) {
      // Wu-Estabrook released block: per-indicator response scale δᵢ = Σ*ᵢᵢ^{-½}
      // where the scale is free (δᵢ = 1 otherwise, e.g. binary-vetoed items and
      // the reference group), with the implied indicator mean μᵢ subtracted
      // from the thresholds. `mixed_assoc_moment(.., Theta)` keyed on the
      // free-scale mask gives the per-pair implied polychoric δᵢδⱼΣ*ᵢⱼ.
      std::vector<std::int32_t> sf(static_cast<std::size_t>(p), 0);
      for (Eigen::Index i = 0; i < p; ++i)
        if (static_cast<std::size_t>(i) < layout.scale_free[b].size())
          sf[static_cast<std::size_t>(i)] =
              layout.scale_free[b][static_cast<std::size_t>(i)];
      const bool have_mu = b < moments.mu.size() && moments.mu[b].size() == p;
      for (Eigen::Index k = 0; k < nth; ++k) {
        const Eigen::Index ov =
            stats.threshold_ov[b][static_cast<std::size_t>(k)];
        const double mu = have_mu ? moments.mu[b](ov) : 0.0;
        const double delta = sf[static_cast<std::size_t>(ov)]
                                 ? 1.0 / std::sqrt(Sig(ov, ov))
                                 : 1.0;
        it(k) = (it(k) - mu) * delta;
      }
      d.head(nth) = it - stats.thresholds[b];
      Eigen::Index r = 0;
      for (Eigen::Index j = 0; j < p; ++j)
        for (Eigen::Index i = j + 1; i < p; ++i)
          d(nth + r++) = mixed_assoc_moment(Sig, sf, i, j,
                                            OrdinalParameterization::Theta) -
                         stats.R[b](i, j);
    } else if (theta_param) {
      // Standardize: implied thresholds (τ_θ − μᵢ)/√Σ*ᵢᵢ, implied correlations.
      // μᵢ is the model-implied indicator mean: 0 in the reference group / when
      // there is no mean structure, the freed group-2+ intercept under
      // Wu-Estabrook (2016) threshold+loading invariance otherwise.
      const bool have_mu = b < moments.mu.size() && moments.mu[b].size() == p;
      for (Eigen::Index k = 0; k < nth; ++k) {
        const Eigen::Index ov =
            stats.threshold_ov[b][static_cast<std::size_t>(k)];
        const double mu = have_mu ? moments.mu[b](ov) : 0.0;
        it(k) = (it(k) - mu) / std::sqrt(Sig(ov, ov));
      }
      d.head(nth) = it - stats.thresholds[b];
      d.tail(ncorr) = std_corr_lower(Sig) - corr_lower(stats.R[b]);
    } else {
      d.head(nth) = it - stats.thresholds[b];
      d.tail(ncorr) = corr_lower(Sig) - corr_lower(stats.R[b]);
    }
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.segment(off, d.size()) = sw * (factors[b].transpose() * d);
    off += d.size();
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "ordinal LS residuals contain non-finite values"));
  }
  return out;
}

// Raw per-block ordinal moment residual d_b = model − sample in the
// [thresholds; lower-triangular correlations] metric, BEFORE the W-whitening
// and sqrt(n/N) scaling that `ordinal_residuals` applies. Mirrors the per-block
// branch of `ordinal_residuals` (keep the two in sync); used only to
// finite-difference the misspecification-robust observed-Hessian bread.
Eigen::VectorXd ordinal_block_residual(const data::OrdinalStats& stats,
                                       const ThresholdLayout& layout,
                                       const model::ImpliedMoments& moments,
                                       const Eigen::VectorXd& theta,
                                       OrdinalParameterization param,
                                       std::size_t b) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::MatrixXd& Sig = moments.sigma[b];
  Eigen::VectorXd d(nth + ncorr);
  Eigen::VectorXd it = implied_thresholds(layout, theta, b);
  const bool block_released =
      b < layout.scale_free.size() &&
      std::any_of(layout.scale_free[b].begin(), layout.scale_free[b].end(),
                  [](char c) { return c != 0; });
  if (block_released && !theta_param) {
    std::vector<std::int32_t> sf(static_cast<std::size_t>(p), 0);
    for (Eigen::Index i = 0; i < p; ++i)
      if (static_cast<std::size_t>(i) < layout.scale_free[b].size())
        sf[static_cast<std::size_t>(i)] =
            layout.scale_free[b][static_cast<std::size_t>(i)];
    const bool have_mu = b < moments.mu.size() && moments.mu[b].size() == p;
    for (Eigen::Index k = 0; k < nth; ++k) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(k)];
      const double mu = have_mu ? moments.mu[b](ov) : 0.0;
      const double delta = sf[static_cast<std::size_t>(ov)]
                               ? 1.0 / std::sqrt(Sig(ov, ov))
                               : 1.0;
      it(k) = (it(k) - mu) * delta;
    }
    d.head(nth) = it - stats.thresholds[b];
    Eigen::Index r = 0;
    for (Eigen::Index j = 0; j < p; ++j)
      for (Eigen::Index i = j + 1; i < p; ++i)
        d(nth + r++) = mixed_assoc_moment(Sig, sf, i, j,
                                          OrdinalParameterization::Theta) -
                       stats.R[b](i, j);
  } else if (theta_param) {
    const bool have_mu = b < moments.mu.size() && moments.mu[b].size() == p;
    for (Eigen::Index k = 0; k < nth; ++k) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(k)];
      const double mu = have_mu ? moments.mu[b](ov) : 0.0;
      it(k) = (it(k) - mu) / std::sqrt(Sig(ov, ov));
    }
    d.head(nth) = it - stats.thresholds[b];
    d.tail(ncorr) = std_corr_lower(Sig) - corr_lower(stats.R[b]);
  } else {
    d.head(nth) = it - stats.thresholds[b];
    d.tail(ncorr) = corr_lower(Sig) - corr_lower(stats.R[b]);
  }
  return d;
}

Eigen::MatrixXd ordinal_moment_jacobian_block(
    const data::OrdinalStats& stats,
    const ThresholdLayout& layout,
    const model::ImpliedMoments& moments,
    const Eigen::MatrixXd& J_sigma,
    const Eigen::VectorXd& theta,
    OrdinalParameterization param,
    const Eigen::MatrixXd& J_mu,
    std::size_t b,
    Eigen::Index sigma_off,
    Eigen::Index mu_off) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::MatrixXd& Sig = moments.sigma[b];
  Eigen::MatrixXd Jb(nth + ncorr, J_sigma.cols());
  Jb.setZero();
  const bool block_released =
      b < layout.scale_free.size() &&
      std::any_of(layout.scale_free[b].begin(), layout.scale_free[b].end(),
                  [](char c) { return c != 0; });
  if (block_released && !theta_param) {
    // Released delta block: thresholds are (tau - mu) * delta_i and association
    // moments use the same scale mask as the fitting residuals.
    std::vector<std::int32_t> sf(static_cast<std::size_t>(p), 0);
    for (Eigen::Index i = 0; i < p; ++i) {
      if (static_cast<std::size_t>(i) < layout.scale_free[b].size()) {
        sf[static_cast<std::size_t>(i)] =
            layout.scale_free[b][static_cast<std::size_t>(i)];
      }
    }
    const bool have_mu = b < moments.mu.size() && moments.mu[b].size() == p;
    const bool have_jmu = J_mu.rows() > 0;
    const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
    for (Eigen::Index k = 0; k < nth; ++k) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(k)];
      const bool std_i = sf[static_cast<std::size_t>(ov)] != 0;
      const double delta = std_i ? 1.0 / std::sqrt(Sig(ov, ov)) : 1.0;
      const double mu = have_mu ? moments.mu[b](ov) : 0.0;
      const double a = it(k) - mu;
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0) Jb(k, fr - 1) += delta;
      if (have_jmu) Jb.row(k) -= delta * J_mu.row(mu_off + ov);
      if (std_i) {
        Jb.row(k) += (-0.5 * a * delta * delta * delta) *
                     J_sigma.row(sigma_off + vech_index(p, ov, ov));
      }
    }
    Eigen::Index r = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        Jb.row(nth + r++) = mixed_assoc_jacobian_row(
            Sig, J_sigma, sigma_off, sf, i, j,
            OrdinalParameterization::Theta);
      }
    }
  } else if (theta_param) {
    // Threshold rows: d[(tau - mu_i) / sqrt(Sigma*_ii)] / d theta.
    const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
    const bool have_mu = b < moments.mu.size() && moments.mu[b].size() == p;
    const bool have_jmu = J_mu.rows() > 0;
    for (Eigen::Index k = 0; k < nth; ++k) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(k)];
      const double sii = Sig(ov, ov);
      const double inv_sd = 1.0 / std::sqrt(sii);
      const double mu = have_mu ? moments.mu[b](ov) : 0.0;
      const double a = it(k) - mu;
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0) Jb(k, fr - 1) += inv_sd;
      if (have_jmu) Jb.row(k) -= inv_sd * J_mu.row(mu_off + ov);
      Jb.row(k) += (-0.5 * a * inv_sd / sii) *
                   J_sigma.row(sigma_off + vech_index(p, ov, ov));
    }
    Jb.bottomRows(ncorr) = std_corr_jacobian(Sig, J_sigma, sigma_off);
  } else {
    for (Eigen::Index k = 0; k < nth; ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0) Jb(k, fr - 1) = 1.0;
    }
    Jb.bottomRows(ncorr) = corr_jacobian(Sig, J_sigma, sigma_off);
  }
  return Jb;
}

fit_expected<Eigen::MatrixXd>
ordinal_jacobian(const data::OrdinalStats& stats,
                 const ThresholdLayout& layout,
                 const model::ImpliedMoments& moments,
                 const Eigen::MatrixXd& J_sigma,
                 const std::vector<Eigen::MatrixXd>& factors,
                 const Eigen::VectorXd& theta,
                 OrdinalParameterization param,
                 const Eigen::MatrixXd& J_mu = Eigen::MatrixXd()) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    n_total += stats.thresholds[b].size() + p * (p - 1) / 2;
  }
  Eigen::MatrixXd out(n_total, J_sigma.cols());
  Eigen::Index out_off = 0;
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::MatrixXd Jb = ordinal_moment_jacobian_block(
        stats, layout, moments, J_sigma, theta, param, J_mu, b, sigma_off,
        mu_off);
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.block(out_off, 0, Jb.rows(), Jb.cols()) =
        sw * (factors[b].transpose() * Jb);
    out_off += Jb.rows();
    sigma_off += vech_len(p);
    mu_off += p;
  }
  return out;
}

// Stacked correlation residual d_corr over all blocks, in workspace order.
// The joint threshold map consumes the full stacked vector: with shared gamma
// coordinates block b's profiled thresholds depend on every block's
// correlation residual.
template <typename Stats>
fit_expected<Eigen::VectorXd>
stacked_corr_residual(const Stats& stats,
                      const model::ImpliedMoments& moments,
                      const ProfiledWeightWorkspace& weights) {
  if (weights.blocks.size() != stats.R.size() ||
      weights.corr_offset.size() != stats.R.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "profiled ordinal residuals received mismatched weight blocks"));
  }
  Eigen::VectorXd d(weights.ncorr_total);
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    d.segment(weights.corr_offset[b], ncorr) =
        corr_lower(moments.sigma[b]) - corr_lower(stats.R[b]);
  }
  return d;
}

fit_expected<Eigen::VectorXd>
profiled_ordinal_residuals(const data::OrdinalMoments& stats,
                           const model::ImpliedMoments& moments,
                           const ProfiledWeightWorkspace& weights) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  auto d_corr_or = stacked_corr_residual(stats, moments, weights);
  if (!d_corr_or.has_value()) return std::unexpected(d_corr_or.error());
  const Eigen::VectorXd& d_corr = *d_corr_or;

  Eigen::Index n_total = 0;
  for (const auto& block : weights.blocks) n_total += block.factor.rows();
  Eigen::VectorXd out(n_total);
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const auto& block = weights.blocks[b];
    const Eigen::Index nth = block.threshold_offset.size();
    const Eigen::Index mdim = nth + ncorr;
    if (block.factor.rows() != mdim || block.factor.cols() != mdim ||
        block.threshold_from_corr.rows() != nth ||
        block.threshold_from_corr.cols() != weights.ncorr_total) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "profiled ordinal residuals found a weight dimension mismatch"));
    }
    Eigen::VectorXd d(mdim);
    d.head(nth) = block.threshold_offset - block.threshold_from_corr * d_corr;
    d.tail(ncorr) = d_corr.segment(weights.corr_offset[b], ncorr);
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.segment(off, mdim) = sw * (block.factor.transpose() * d);
    off += mdim;
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "profiled ordinal LS residuals contain non-finite values"));
  }
  return out;
}

fit_expected<Eigen::MatrixXd>
profiled_ordinal_jacobian(const data::OrdinalMoments& stats,
                          const model::ImpliedMoments& moments,
                          const Eigen::MatrixXd& J_sigma,
                          const ProfiledWeightWorkspace& weights) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  if (weights.blocks.size() != stats.R.size() ||
      weights.corr_offset.size() != stats.R.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "profiled ordinal Jacobian received mismatched weight blocks"));
  }
  // Stacked correlation Jacobian over all blocks, matching the stacked d_corr
  // layout.
  Eigen::MatrixXd J_corr(weights.ncorr_total, J_sigma.cols());
  {
    Eigen::Index sigma_off = 0;
    for (std::size_t b = 0; b < stats.R.size(); ++b) {
      const Eigen::Index p = stats.R[b].rows();
      const Eigen::Index ncorr = p * (p - 1) / 2;
      J_corr.middleRows(weights.corr_offset[b], ncorr) =
          corr_jacobian(moments.sigma[b], J_sigma, sigma_off);
      sigma_off += vech_len(p);
    }
  }

  Eigen::Index n_total = 0;
  for (const auto& block : weights.blocks) n_total += block.factor.rows();
  Eigen::MatrixXd out(n_total, J_sigma.cols());
  Eigen::Index out_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const auto& block = weights.blocks[b];
    const Eigen::Index nth = block.threshold_offset.size();
    const Eigen::Index mdim = nth + ncorr;
    if (block.factor.rows() != mdim || block.factor.cols() != mdim ||
        block.threshold_from_corr.rows() != nth ||
        block.threshold_from_corr.cols() != weights.ncorr_total) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "profiled ordinal Jacobian found a weight dimension mismatch"));
    }
    Eigen::MatrixXd Jp(mdim, J_sigma.cols());
    Jp.topRows(nth).noalias() = -block.threshold_from_corr * J_corr;
    Jp.bottomRows(ncorr) = J_corr.middleRows(weights.corr_offset[b], ncorr);
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.block(out_off, 0, mdim, J_sigma.cols()) =
        sw * (block.factor.transpose() * Jp);
    out_off += mdim;
  }
  return out;
}

fit_expected<Eigen::VectorXd>
reconstruct_profiled_thresholds(const data::OrdinalStats& stats,
                                const ThresholdLayout& layout,
                                const model::ImpliedMoments& moments,
                                const ProfiledWeightWorkspace& weights,
                                const Eigen::VectorXd& theta) {
  auto d_corr_or = stacked_corr_residual(stats, moments, weights);
  if (!d_corr_or.has_value()) return std::unexpected(d_corr_or.error());
  const Eigen::VectorXd& d_corr = *d_corr_or;
  Eigen::VectorXd out = theta;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index nth = stats.thresholds[b].size();
    if (weights.blocks[b].threshold_intercept.size() != nth ||
        weights.blocks[b].threshold_from_corr.rows() != nth ||
        weights.blocks[b].threshold_from_corr.cols() != weights.ncorr_total) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "profiled ordinal reconstruction found a threshold map dimension mismatch"));
    }
    const Eigen::VectorXd tau =
        weights.blocks[b].threshold_intercept -
        weights.blocks[b].threshold_from_corr * d_corr;
    if (!tau.allFinite()) {
      return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
          "profiled ordinal reconstruction produced non-finite thresholds"));
    }
    for (Eigen::Index k = 0; k < nth; ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0) out(fr - 1) = tau(k);
    }
  }
  return out;
}

Eigen::Index ordinal_moment_rows(const data::OrdinalStats& stats) {
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    n_total += stats.thresholds[b].size() + p * (p - 1) / 2;
  }
  return n_total;
}

Eigen::MatrixXd ordinal_moment_jacobian(const data::OrdinalStats& stats,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::VectorXd& theta,
                                        OrdinalParameterization param,
                                        const Eigen::MatrixXd& J_mu =
                                            Eigen::MatrixXd()) {
  Eigen::MatrixXd out(ordinal_moment_rows(stats), J_sigma.cols());
  Eigen::Index out_off = 0;
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::MatrixXd Jb = ordinal_moment_jacobian_block(
        stats, layout, moments, J_sigma, theta, param, J_mu, b, sigma_off,
        mu_off);
    out.block(out_off, 0, Jb.rows(), Jb.cols()) = Jb;
    out_off += Jb.rows();
    sigma_off += vech_len(p);
    mu_off += p;
  }
  return out;
}

Eigen::Index mixed_moment_rows(const data::MixedOrdinalStats& stats) {
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    n_total += stats.moments[b].size();
  }
  return n_total;
}

Eigen::VectorXd mixed_model_moments(const data::MixedOrdinalStats& stats,
                                    const ThresholdLayout& layout,
                                    const model::ImpliedMoments& moments,
                                    const Eigen::VectorXd& theta,
                                    std::size_t b,
                                    OrdinalParameterization param) {
  const Eigen::Index p = stats.R[b].rows();
  Eigen::VectorXd out(stats.moments[b].size());
  Eigen::Index k = 0;
  const Eigen::Index nth = stats.thresholds[b].size();
  Eigen::VectorXd it = implied_thresholds(layout, theta, b);
  if (param == OrdinalParameterization::Theta) {
    for (Eigen::Index h = 0; h < nth; ++h) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(h)];
      it(h) /= std::sqrt(moments.sigma[b](ov, ov));
    }
  }
  out.segment(k, nth) = it;
  k += nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
      const double mu = b < moments.mu.size() && moments.mu[b].size() == p
          ? moments.mu[b](j)
          : 0.0;
      out(k++) = -mu;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
      out(k++) = moments.sigma[b](j, j);
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out(k++) = mixed_assoc_moment(moments.sigma[b], stats.ordered[b], i, j,
                                    param);
    }
  }
  return out;
}

Eigen::MatrixXd mixed_moment_jacobian(const data::MixedOrdinalStats& stats,
                                      const ThresholdLayout& layout,
                                      const model::ImpliedMoments& moments,
                                      const Eigen::MatrixXd& J_sigma,
                                      const Eigen::MatrixXd& J_mu,
                                      const Eigen::VectorXd& theta,
                                      OrdinalParameterization param) {
  Eigen::MatrixXd out(mixed_moment_rows(stats), J_sigma.cols());
  Eigen::Index out_off = 0;
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    Eigen::MatrixXd Jb(stats.moments[b].size(), J_sigma.cols());
    Jb.setZero();
    Eigen::Index row = 0;
    for (Eigen::Index k = 0; k < nth; ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (param == OrdinalParameterization::Theta) {
        const Eigen::Index ov =
            stats.threshold_ov[b][static_cast<std::size_t>(k)];
        const double sii = moments.sigma[b](ov, ov);
        const double inv_sd = 1.0 / std::sqrt(sii);
        const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
        if (fr > 0) Jb(row, fr - 1) += inv_sd;
        Jb.row(row) += (-0.5 * it(k) * inv_sd / sii) *
                       J_sigma.row(sigma_off + vech_index(p, ov, ov));
      } else if (fr > 0) {
        Jb(row, fr - 1) = 1.0;
      }
      ++row;
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
        if (J_mu.rows() > 0) Jb.row(row) = -J_mu.row(mu_off + j);
        ++row;
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
        Jb.row(row++) = J_sigma.row(sigma_off + vech_index(p, j, j));
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        Jb.row(row++) = mixed_assoc_jacobian_row(
            moments.sigma[b], J_sigma, sigma_off, stats.ordered[b], i, j,
            param);
      }
    }
    out.block(out_off, 0, Jb.rows(), Jb.cols()) = Jb;
    out_off += Jb.rows();
    sigma_off += vech_len(p);
    mu_off += p;
  }
  return out;
}

struct MomentCurvatureWeights {
  Eigen::MatrixXd G;  // trace weight for Sigma
  Eigen::VectorXd u;  // dot weight for mu
};

Eigen::Index vech_sym_index(Eigen::Index p, Eigen::Index r,
                            Eigen::Index c) noexcept {
  if (r < c) std::swap(r, c);
  return vech_index(p, r, c);
}

void add_sigma_trace_weight(Eigen::Ref<Eigen::MatrixXd> G, Eigen::Index r,
                            Eigen::Index c, double value) {
  if (value == 0.0) return;
  if (r == c) {
    G(r, c) += value;
  } else {
    const double half = 0.5 * value;
    G(r, c) += half;
    G(c, r) += half;
  }
}

double sigma_deriv(const Eigen::MatrixXd& J_sigma, Eigen::Index sigma_off,
                   Eigen::Index p, Eigen::Index r, Eigen::Index c,
                   Eigen::Index param) {
  const Eigen::Index row = sigma_off + vech_sym_index(p, r, c);
  if (row < 0 || row >= J_sigma.rows() || param < 0 || param >= J_sigma.cols()) {
    return 0.0;
  }
  return J_sigma(row, param);
}

double mu_deriv(const Eigen::MatrixXd& J_mu, Eigen::Index mu_off,
                Eigen::Index idx, Eigen::Index param) {
  const Eigen::Index row = mu_off + idx;
  if (row < 0 || row >= J_mu.rows() || param < 0 || param >= J_mu.cols()) {
    return 0.0;
  }
  return J_mu(row, param);
}

double threshold_deriv(const ThresholdLayout& layout, std::size_t b,
                       Eigen::Index threshold, Eigen::Index param) {
  const std::int32_t fr = layout.free[b][static_cast<std::size_t>(threshold)];
  return fr > 0 && static_cast<Eigen::Index>(fr - 1) == param ? 1.0 : 0.0;
}

std::vector<char> threshold_parameter_mask(const ThresholdLayout& layout,
                                           Eigen::Index q) {
  std::vector<char> out(static_cast<std::size_t>(q), 0);
  for (std::size_t b = 0; b < layout.free.size(); ++b) {
    for (const std::int32_t fr : layout.free[b]) {
      if (fr > 0 && static_cast<Eigen::Index>(fr - 1) < q) {
        out[static_cast<std::size_t>(fr - 1)] = 1;
      }
    }
  }
  return out;
}

void add_assoc_gradient(Eigen::Ref<Eigen::MatrixXd> G,
                        const Eigen::MatrixXd& Sigma, Eigen::Index i,
                        Eigen::Index j, bool std_i, bool std_j,
                        double scale) {
  if (scale == 0.0) return;
  const double sij = Sigma(i, j);
  if (!std_i && !std_j) {
    add_sigma_trace_weight(G, i, j, scale);
    return;
  }
  if (std_i && std_j) {
    const double sii = Sigma(i, i);
    const double sjj = Sigma(j, j);
    const double inv_i = 1.0 / std::sqrt(sii);
    const double inv_j = 1.0 / std::sqrt(sjj);
    add_sigma_trace_weight(G, i, j, scale * inv_i * inv_j);
    add_sigma_trace_weight(G, i, i,
                           scale * (-0.5 * sij * inv_i / sii * inv_j));
    add_sigma_trace_weight(G, j, j,
                           scale * (-0.5 * sij * inv_i * inv_j / sjj));
    return;
  }
  const Eigen::Index o = std_i ? i : j;
  const double soo = Sigma(o, o);
  const double inv_o = 1.0 / std::sqrt(soo);
  add_sigma_trace_weight(G, i, j, scale * inv_o);
  add_sigma_trace_weight(G, o, o, scale * (-0.5 * sij * inv_o / soo));
}

double assoc_extra(const Eigen::MatrixXd& Sigma, const Eigen::MatrixXd& J_sigma,
                   Eigen::Index sigma_off, Eigen::Index p, Eigen::Index i,
                   Eigen::Index j, bool std_i, bool std_j, Eigen::Index a,
                   Eigen::Index b) {
  if (!std_i && !std_j) return 0.0;
  const double dx_a = sigma_deriv(J_sigma, sigma_off, p, i, j, a);
  const double dx_b = sigma_deriv(J_sigma, sigma_off, p, i, j, b);
  const double sij = Sigma(i, j);
  if (std_i && std_j) {
    const double sii = Sigma(i, i);
    const double sjj = Sigma(j, j);
    const double inv_i = 1.0 / std::sqrt(sii);
    const double inv_j = 1.0 / std::sqrt(sjj);
    const double di_a = sigma_deriv(J_sigma, sigma_off, p, i, i, a);
    const double di_b = sigma_deriv(J_sigma, sigma_off, p, i, i, b);
    const double dj_a = sigma_deriv(J_sigma, sigma_off, p, j, j, a);
    const double dj_b = sigma_deriv(J_sigma, sigma_off, p, j, j, b);
    const double f_xi = -0.5 * inv_i / sii * inv_j;
    const double f_xj = -0.5 * inv_i * inv_j / sjj;
    const double f_ii = 0.75 * sij * inv_i / (sii * sii) * inv_j;
    const double f_jj = 0.75 * sij * inv_i * inv_j / (sjj * sjj);
    const double f_ij = 0.25 * sij * inv_i / sii * inv_j / sjj;
    return f_xi * (dx_a * di_b + dx_b * di_a) +
           f_xj * (dx_a * dj_b + dx_b * dj_a) +
           f_ii * di_a * di_b + f_jj * dj_a * dj_b +
           f_ij * (di_a * dj_b + di_b * dj_a);
  }
  const Eigen::Index o = std_i ? i : j;
  const double soo = Sigma(o, o);
  const double inv_o = 1.0 / std::sqrt(soo);
  const double do_a = sigma_deriv(J_sigma, sigma_off, p, o, o, a);
  const double do_b = sigma_deriv(J_sigma, sigma_off, p, o, o, b);
  const double f_xo = -0.5 * inv_o / soo;
  const double f_oo = 0.75 * sij * inv_o / (soo * soo);
  return f_xo * (dx_a * do_b + dx_b * do_a) + f_oo * do_a * do_b;
}

double scaled_threshold_extra(double value_minus_mu, double sigma_ii,
                              double ds_a, double ds_b, double dtau_a,
                              double dtau_b, double dmu_a, double dmu_b) {
  const double inv = 1.0 / std::sqrt(sigma_ii);
  const double d1 = -0.5 * inv / sigma_ii;
  const double d2 = 0.75 * inv / (sigma_ii * sigma_ii);
  return d1 * ((dtau_a - dmu_a) * ds_b + (dtau_b - dmu_b) * ds_a) +
         value_minus_mu * d2 * ds_a * ds_b;
}

MomentCurvatureWeights ordinal_curvature_weights(
    const data::OrdinalStats& stats,
    const ThresholdLayout& layout,
    const model::ImpliedMoments& moments,
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& h,
    OrdinalParameterization param,
    std::size_t b) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  const Eigen::MatrixXd& Sig = moments.sigma[b];
  MomentCurvatureWeights out{Eigen::MatrixXd::Zero(p, p),
                             Eigen::VectorXd::Zero(p)};

  const bool block_released =
      b < layout.scale_free.size() &&
      std::any_of(layout.scale_free[b].begin(), layout.scale_free[b].end(),
                  [](char c) { return c != 0; });
  std::vector<std::int32_t> sf(static_cast<std::size_t>(p),
                               theta_param ? 1 : 0);
  if (block_released && !theta_param) {
    std::fill(sf.begin(), sf.end(), 0);
    for (Eigen::Index i = 0; i < p; ++i) {
      if (static_cast<std::size_t>(i) < layout.scale_free[b].size()) {
        sf[static_cast<std::size_t>(i)] =
            layout.scale_free[b][static_cast<std::size_t>(i)];
      }
    }
  } else if (!theta_param) {
    std::fill(sf.begin(), sf.end(), 0);
  }

  const bool subtract_mu =
      theta_param || (block_released && !theta_param);
  const bool have_mu = subtract_mu && b < moments.mu.size() &&
                       moments.mu[b].size() == p;
  const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
  for (Eigen::Index k = 0; k < nth; ++k) {
    const double hk = h(k);
    const Eigen::Index ov = stats.threshold_ov[b][static_cast<std::size_t>(k)];
    const bool scaled = sf[static_cast<std::size_t>(ov)] != 0;
    const double mu = have_mu ? moments.mu[b](ov) : 0.0;
    if (scaled) {
      const double s = Sig(ov, ov);
      const double inv = 1.0 / std::sqrt(s);
      const double v = it(k) - mu;
      add_sigma_trace_weight(out.G, ov, ov, hk * (-0.5 * v * inv / s));
      if (have_mu) out.u(ov) -= hk * inv;
    } else if (have_mu) {
      out.u(ov) -= hk;
    }
  }

  Eigen::Index row = nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      add_assoc_gradient(out.G, Sig, i, j,
                         sf[static_cast<std::size_t>(i)] != 0,
                         sf[static_cast<std::size_t>(j)] != 0,
                         h(row++));
    }
  }
  return out;
}

double ordinal_curvature_extra(const data::OrdinalStats& stats,
                               const ThresholdLayout& layout,
                               const model::ImpliedMoments& moments,
                               const Eigen::MatrixXd& J_sigma,
                               const Eigen::MatrixXd& J_mu,
                               const Eigen::VectorXd& theta,
                               const Eigen::VectorXd& h,
                               OrdinalParameterization param,
                               std::size_t blk,
                               Eigen::Index sigma_off,
                               Eigen::Index mu_off,
                               Eigen::Index a,
                               Eigen::Index c) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[blk].rows();
  const Eigen::Index nth = stats.thresholds[blk].size();
  const Eigen::MatrixXd& Sig = moments.sigma[blk];
  const bool block_released =
      blk < layout.scale_free.size() &&
      std::any_of(layout.scale_free[blk].begin(), layout.scale_free[blk].end(),
                  [](char x) { return x != 0; });
  std::vector<std::int32_t> sf(static_cast<std::size_t>(p),
                               theta_param ? 1 : 0);
  if (block_released && !theta_param) {
    std::fill(sf.begin(), sf.end(), 0);
    for (Eigen::Index i = 0; i < p; ++i) {
      if (static_cast<std::size_t>(i) < layout.scale_free[blk].size()) {
        sf[static_cast<std::size_t>(i)] =
            layout.scale_free[blk][static_cast<std::size_t>(i)];
      }
    }
  } else if (!theta_param) {
    std::fill(sf.begin(), sf.end(), 0);
  }

  double out = 0.0;
  const bool subtract_mu =
      theta_param || (block_released && !theta_param);
  const bool have_mu = subtract_mu && blk < moments.mu.size() &&
                       moments.mu[blk].size() == p;
  const Eigen::VectorXd it = implied_thresholds(layout, theta, blk);
  for (Eigen::Index k = 0; k < nth; ++k) {
    const Eigen::Index ov =
        stats.threshold_ov[blk][static_cast<std::size_t>(k)];
    if (sf[static_cast<std::size_t>(ov)] == 0) continue;
    const double tau_a = threshold_deriv(layout, blk, k, a);
    const double tau_c = threshold_deriv(layout, blk, k, c);
    const double mu_a = have_mu ? mu_deriv(J_mu, mu_off, ov, a) : 0.0;
    const double mu_c = have_mu ? mu_deriv(J_mu, mu_off, ov, c) : 0.0;
    const double ds_a = sigma_deriv(J_sigma, sigma_off, p, ov, ov, a);
    const double ds_c = sigma_deriv(J_sigma, sigma_off, p, ov, ov, c);
    const double mu = have_mu ? moments.mu[blk](ov) : 0.0;
    out += h(k) * scaled_threshold_extra(
                      it(k) - mu, Sig(ov, ov), ds_a, ds_c,
                      tau_a, tau_c, mu_a, mu_c);
  }

  Eigen::Index row = nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out += h(row) * assoc_extra(
                          Sig, J_sigma, sigma_off, p, i, j,
                          sf[static_cast<std::size_t>(i)] != 0,
                          sf[static_cast<std::size_t>(j)] != 0,
                          a, c);
      ++row;
    }
  }
  return out;
}

MomentCurvatureWeights mixed_curvature_weights(
    const data::MixedOrdinalStats& stats,
    const ThresholdLayout& layout,
    const model::ImpliedMoments& moments,
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& h,
    OrdinalParameterization param,
    std::size_t b) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  const Eigen::MatrixXd& Sig = moments.sigma[b];
  MomentCurvatureWeights out{Eigen::MatrixXd::Zero(p, p),
                             Eigen::VectorXd::Zero(p)};

  Eigen::Index row = 0;
  const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
  for (Eigen::Index k = 0; k < nth; ++k) {
    if (theta_param) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(k)];
      const double s = Sig(ov, ov);
      const double inv = 1.0 / std::sqrt(s);
      add_sigma_trace_weight(out.G, ov, ov,
                             h(row) * (-0.5 * it(k) * inv / s));
    }
    ++row;
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) out.u(j) -= h(row++);
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0)
      add_sigma_trace_weight(out.G, j, j, h(row++));
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool std_i = theta_param &&
                         stats.ordered[b][static_cast<std::size_t>(i)] != 0;
      const bool std_j = theta_param &&
                         stats.ordered[b][static_cast<std::size_t>(j)] != 0;
      add_assoc_gradient(out.G, Sig, i, j, std_i, std_j, h(row++));
    }
  }
  return out;
}

double mixed_curvature_extra(const data::MixedOrdinalStats& stats,
                             const ThresholdLayout& layout,
                             const model::ImpliedMoments& moments,
                             const Eigen::MatrixXd& J_sigma,
                             const Eigen::VectorXd& theta,
                             const Eigen::VectorXd& h,
                             OrdinalParameterization param,
                             std::size_t blk,
                             Eigen::Index sigma_off,
                             Eigen::Index a,
                             Eigen::Index c) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  if (!theta_param) return 0.0;
  const Eigen::Index p = stats.R[blk].rows();
  const Eigen::Index nth = stats.thresholds[blk].size();
  const Eigen::MatrixXd& Sig = moments.sigma[blk];
  const Eigen::VectorXd it = implied_thresholds(layout, theta, blk);

  double out = 0.0;
  Eigen::Index row = 0;
  for (Eigen::Index k = 0; k < nth; ++k) {
    const Eigen::Index ov =
        stats.threshold_ov[blk][static_cast<std::size_t>(k)];
    const double tau_a = threshold_deriv(layout, blk, k, a);
    const double tau_c = threshold_deriv(layout, blk, k, c);
    const double ds_a = sigma_deriv(J_sigma, sigma_off, p, ov, ov, a);
    const double ds_c = sigma_deriv(J_sigma, sigma_off, p, ov, ov, c);
    out += h(row) * scaled_threshold_extra(
                       it(k), Sig(ov, ov), ds_a, ds_c,
                       tau_a, tau_c, 0.0, 0.0);
    ++row;
  }
  for (Eigen::Index j = 0; j < p; ++j)
    if (stats.ordered[blk][static_cast<std::size_t>(j)] == 0) ++row;
  for (Eigen::Index j = 0; j < p; ++j)
    if (stats.ordered[blk][static_cast<std::size_t>(j)] == 0) ++row;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool std_i =
          stats.ordered[blk][static_cast<std::size_t>(i)] != 0;
      const bool std_j =
          stats.ordered[blk][static_cast<std::size_t>(j)] != 0;
      out += h(row) * assoc_extra(Sig, J_sigma, sigma_off, p, i, j,
                                  std_i, std_j, a, c);
      ++row;
    }
  }
  return out;
}

template <typename ExtraFn>
void add_lisrel_second_order(Eigen::Ref<Eigen::MatrixXd> H,
                             const MomentCurvatureWeights& curv,
                             const model::BlockMatrices& bm,
                             const std::vector<model::ParamLocation>& locs,
                             const std::vector<char>& is_threshold,
                             std::size_t block,
                             bool has_mu_rows,
                             double weight,
                             ExtraFn extra) {
  const Eigen::Index q = H.rows();
  const auto blk_i = static_cast<std::int8_t>(block);
  const auto sow =
      detail::SecondOrderWeights::build(curv.G, bm, has_mu_rows);
  for (Eigen::Index a = 0; a < q; ++a) {
    for (Eigen::Index c = a; c < q; ++c) {
      double h2 = extra(a, c);
      if (is_threshold[static_cast<std::size_t>(a)] == 0 &&
          is_threshold[static_cast<std::size_t>(c)] == 0 &&
          locs[static_cast<std::size_t>(a)].block == blk_i &&
          locs[static_cast<std::size_t>(c)].block == blk_i) {
        const auto& la = locs[static_cast<std::size_t>(a)];
        const auto& lb = locs[static_cast<std::size_t>(c)];
        h2 += detail::second_sigma_trace(la, lb, sow, bm);
        if (has_mu_rows) {
          h2 += detail::second_mu(la, lb, bm, sow.A_alpha).dot(curv.u);
        }
      }
      const double val = weight * h2;
      H(a, c) += val;
      if (a != c) H(c, a) += val;
    }
  }
}

post_expected<Eigen::MatrixXd>
ordinal_observed_bread_analytic(const spec::LatentStructure& pt,
                                const model::MatrixRep& rep,
                                const data::OrdinalStats& stats,
                                const Estimates& est,
                                const ThresholdLayout& layout,
                                const std::vector<Eigen::MatrixXd>& Ws,
                                const Eigen::MatrixXd& K,
                                OrdinalParameterization parameterization) {
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  auto assembled = ev_or->assembled(est.theta);
  if (!assembled.has_value()) return std::unexpected(model_to_post(assembled.error()));
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(fit_to_post(n_or.error()));

  const Eigen::Index q = est.theta.size();
  if (K.rows() != q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal observed bread: K row count does not match theta length"));
  }
  const auto locs = ev_or->param_locations();
  if (locs.size() != static_cast<std::size_t>(q)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal observed bread: parameter-location count mismatch"));
  }
  const std::vector<char> threshold_mask = threshold_parameter_mask(layout, q);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    const Eigen::MatrixXd Delta = ordinal_moment_jacobian_block(
        stats, layout, eval->moments, eval->J_sigma, est.theta,
        parameterization, eval->J_mu, b, sigma_off, mu_off);
    const Eigen::VectorXd d =
        ordinal_block_residual(stats, layout, eval->moments, est.theta,
                               parameterization, b);
    if (Ws[b].rows() != mb || Ws[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal observed bread: weight shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd h = Ws[b] * d;
    const double w_b = static_cast<double>(stats.n_obs[b]) /
                       static_cast<double>(*n_or);
    H.noalias() += w_b * (Delta.transpose() * Ws[b] * Delta);
    const auto curv = ordinal_curvature_weights(
        stats, layout, eval->moments, est.theta, h, parameterization, b);
    const bool has_mu_rows =
        eval->J_mu.rows() > 0 && mu_off + p <= eval->J_mu.rows();
    add_lisrel_second_order(
        H, curv, assembled->blocks[b], locs, threshold_mask, b, has_mu_rows,
        w_b, [&](Eigen::Index a, Eigen::Index c) {
          return ordinal_curvature_extra(
              stats, layout, eval->moments, eval->J_sigma, eval->J_mu,
              est.theta, h, parameterization, b, sigma_off, mu_off, a, c);
        });
    sigma_off += vech_len(p);
    mu_off += p;
  }
  if (!H.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal observed bread: non-finite Hessian"));
  }
  H = 0.5 * (H + H.transpose()).eval();
  Eigen::MatrixXd Halpha = K.transpose() * H * K;
  return Eigen::MatrixXd(0.5 * (Halpha + Halpha.transpose()).eval());
}

post_expected<Eigen::MatrixXd>
mixed_observed_bread_analytic(const spec::LatentStructure& pt,
                              const model::MatrixRep& rep,
                              const data::MixedOrdinalStats& stats,
                              const Estimates& est,
                              const ThresholdLayout& layout,
                              const std::vector<Eigen::MatrixXd>& Ws,
                              const Eigen::MatrixXd& K,
                              OrdinalParameterization parameterization) {
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  auto assembled = ev_or->assembled(est.theta);
  if (!assembled.has_value()) return std::unexpected(model_to_post(assembled.error()));
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(fit_to_post(n_or.error()));

  const Eigen::Index q = est.theta.size();
  if (K.rows() != q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal observed bread: K row count does not match theta length"));
  }
  const auto locs = ev_or->param_locations();
  if (locs.size() != static_cast<std::size_t>(q)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal observed bread: parameter-location count mismatch"));
  }
  const std::vector<char> threshold_mask = threshold_parameter_mask(layout, q);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  const Eigen::MatrixXd Delta_full = mixed_moment_jacobian(
      stats, layout, eval->moments, eval->J_sigma, eval->J_mu, est.theta,
      parameterization);
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  Eigen::Index moment_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.moments[b].size();
    const Eigen::MatrixXd Delta = Delta_full.block(moment_off, 0, mb, q);
    const Eigen::VectorXd d =
        mixed_model_moments(stats, layout, eval->moments, est.theta, b,
                            parameterization) -
        stats.moments[b];
    if (Ws[b].rows() != mb || Ws[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed ordinal observed bread: weight shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd h = Ws[b] * d;
    const double w_b = static_cast<double>(stats.n_obs[b]) /
                       static_cast<double>(*n_or);
    H.noalias() += w_b * (Delta.transpose() * Ws[b] * Delta);
    const auto curv = mixed_curvature_weights(
        stats, layout, eval->moments, est.theta, h, parameterization, b);
    const bool has_mu_rows =
        eval->J_mu.rows() > 0 && mu_off + p <= eval->J_mu.rows();
    add_lisrel_second_order(
        H, curv, assembled->blocks[b], locs, threshold_mask, b, has_mu_rows,
        w_b, [&](Eigen::Index a, Eigen::Index c) {
          return mixed_curvature_extra(
              stats, layout, eval->moments, eval->J_sigma, est.theta, h,
              parameterization, b, sigma_off, a, c);
        });
    sigma_off += vech_len(p);
    mu_off += p;
    moment_off += mb;
  }
  if (!H.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal observed bread: non-finite Hessian"));
  }
  H = 0.5 * (H + H.transpose()).eval();
  Eigen::MatrixXd Halpha = K.transpose() * H * K;
  return Eigen::MatrixXd(0.5 * (Halpha + Halpha.transpose()).eval());
}

fit_expected<Eigen::VectorXd>
mixed_ordinal_residuals(const data::MixedOrdinalStats& stats,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const std::vector<Eigen::MatrixXd>& factors,
                        const Eigen::VectorXd& theta,
                        OrdinalParameterization param) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::VectorXd out(mixed_moment_rows(stats));
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    Eigen::VectorXd d =
        mixed_model_moments(stats, layout, moments, theta, b, param) -
        stats.moments[b];
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.segment(off, d.size()) = sw * (factors[b].transpose() * d);
    off += d.size();
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "mixed ordinal LS residuals contain non-finite values"));
  }
  return out;
}

fit_expected<Eigen::MatrixXd>
mixed_ordinal_jacobian(const data::MixedOrdinalStats& stats,
                       const ThresholdLayout& layout,
                       const model::ImpliedMoments& moments,
                       const Eigen::MatrixXd& J_sigma,
                       const Eigen::MatrixXd& J_mu,
                       const std::vector<Eigen::MatrixXd>& factors,
                       const Eigen::VectorXd& theta,
                       OrdinalParameterization param) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::MatrixXd Jfull =
      mixed_moment_jacobian(stats, layout, moments, J_sigma, J_mu, theta,
                            param);
  Eigen::MatrixXd out(Jfull.rows(), Jfull.cols());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.block(off, 0, mb, Jfull.cols()) =
        sw * (factors[b].transpose() * Jfull.block(off, 0, mb, Jfull.cols()));
    off += mb;
  }
  return out;
}

data::SampleStats sample_stats_for_starts(const data::OrdinalStats& stats) {
  data::SampleStats samp;
  samp.S = stats.R;
  samp.n_obs = stats.n_obs;
  return samp;
}

data::SampleStats sample_stats_for_starts(const data::MixedOrdinalStats& stats) {
  data::SampleStats samp;
  samp.S = stats.R;
  samp.mean = stats.mean;
  samp.n_obs = stats.n_obs;
  return samp;
}

data::SampleStats sample_stats_for_starts(const data::MixedOrdinalMoments& moments) {
  data::SampleStats samp;
  samp.S = moments.R;
  samp.mean = moments.mean;
  samp.n_obs = moments.n_obs;
  return samp;
}

void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::OrdinalStats& stats) {
  for (std::size_t b = 0; b < stats.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = stats.thresholds[b](k);
    }
  }
}

void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::MixedOrdinalStats& stats) {
  for (std::size_t b = 0; b < stats.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = stats.thresholds[b](k);
    }
  }
}

void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::MixedOrdinalMoments& moments) {
  for (std::size_t b = 0; b < moments.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < moments.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = moments.thresholds[b](k);
    }
  }
}

OrdinalRobustResult ordinal_result_from_weighted(const WeightedRobustResult& r) {
  OrdinalRobustResult out;
  out.vcov = r.vcov;
  out.se = r.se;
  out.eigvals = r.eigvals;
  out.chisq_standard = r.chisq_standard;
  out.df = r.df;
  out.satorra_bentler = r.satorra_bentler;
  out.mean_var_adjusted = r.mean_var_adjusted;
  out.scaled_shifted = r.scaled_shifted;
  return out;
}

// Per-block missing-pattern flags for the estimated-weight ordinal IJ, plus the
// int_data validation `robust_ordinal_ij` requires (0-based category codes;
// missing entries only with pairwise-overlap Gamma). ULS carries no estimated
// weight, so it returns all-false without touching int_data.
post_expected<std::vector<bool>>
ordinal_ij_block_missing(const data::OrdinalStats& stats,
                         OrdinalWeightKind weights) {
  std::vector<bool> block_has_missing(stats.R.size(), false);
  if (weights == OrdinalWeightKind::ULS) return block_has_missing;
  if (stats.int_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: integer data unavailable; recompute ordinal "
        "stats with int_data to include the estimated-weight influence"));
  }
  const bool allow_pairwise_missing = stats.pairwise_gamma == "overlap";
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::MatrixXi& Xcat = stats.int_data[b];
    if (stats.n_levels[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal_ij: n_levels length mismatch in block " +
              std::to_string(b)));
    }
    if (Xcat.rows() != stats.n_obs[b] || Xcat.cols() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal_ij: int_data shape mismatch in block " +
              std::to_string(b)));
    }
    for (Eigen::Index r = 0; r < Xcat.rows(); ++r) {
      for (Eigen::Index j = 0; j < Xcat.cols(); ++j) {
        const int c = Xcat(r, j);
        if (c < 0) {
          if (!allow_pairwise_missing) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "robust_ordinal_ij: missing ordinal int_data requires "
                "pairwise overlap Gamma in block " + std::to_string(b)));
          }
          block_has_missing[b] = true;
          continue;
        }
        const int max_level = stats.n_levels[b][static_cast<std::size_t>(j)];
        if (c >= max_level) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_ordinal_ij: int_data must contain 0-based category "
              "codes in block " + std::to_string(b)));
        }
      }
    }
  }
  return block_has_missing;
}

// Per-case IJ blocks (Δ_b, W_b, moment_influence, IF(Ŵ) correction) for an
// all-ordinal DWLS/WLS fit, evaluated at `theta`/`moments` (the fitted point for
// the SE path, the freed-candidate null for the score-test sandwich). Shared by
// `robust_ordinal_ij` and `ordinal_param_space_sandwich_ij` so the SE and MI
// paths build identical meat. The DWLS/WLS weight-influence channels are
// recomputed per call (theta-independent but inexpensive enough for the v1
// frontier sweep).
post_expected<std::vector<WeightedMomentIJBlock>>
build_ordinal_ij_blocks(const data::OrdinalStats& stats,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const Eigen::VectorXd& theta,
                        const std::vector<Eigen::MatrixXd>& Ws,
                        const Eigen::MatrixXd& Delta_full,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization,
                        const std::vector<bool>& block_has_missing) {
  if (stats.moment_influence.size() != stats.R.size() ||
      stats.NACOV.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: per-case influence functions unavailable; recompute "
        "ordinal stats (moment_influence is required for the IJ)"));
  }
  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    const Eigen::MatrixXd& G = stats.moment_influence[b];   // n_b × mb
    if (G.cols() != mb || G.rows() != stats.n_obs[b]) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal_ij: moment_influence shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd d_b = ordinal_block_residual(
        stats, layout, moments, theta, parameterization, b);
    Eigen::MatrixXd correction;
    if (weights == OrdinalWeightKind::DWLS) {
      // The IF of the estimated weight Ŵ=diag(Γ̂)⁻¹ enters as corr_{i,k} =
      // d_k·IF_{i,k}(Γ̂)/Γ̂_kk², with IF(Γ̂) = [data-direct sandwich influence at
      // fixed κ] + [κ-movement Σ_l(∂Γ̂_kk/∂κ_l)g_{i,l}]. Both need the integer
      // data; pairwise-overlap MCAR blocks use the observed-support helpers.
      Eigen::MatrixXd IFG;  // n_b × mb: data-direct IF of Γ̂_kk (V̂ + Â variation)
      Eigen::MatrixXd GD;   // n_b × mb: κ-movement IF of Γ̂_kk (FD of Γ̂ over κ)
      auto inf_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_diag_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_diag_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!inf_or.has_value()) return std::unexpected(inf_or.error());
      IFG = std::move(*inf_or);
      auto D_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_diag_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_diag_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!D_or.has_value()) return std::unexpected(D_or.error());
      GD.noalias() = G * D_or->transpose();
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index k = 0; k < mb; ++k) {
        const double gkk = stats.NACOV[b](k, k);
        if (!(gkk > 0.0)) continue;
        const Eigen::VectorXd if_k = (IFG.col(k) + GD.col(k)).eval();
        correction.col(k) = (d_b(k) / (gkk * gkk)) * if_k;
      }
    } else if (weights == OrdinalWeightKind::WLS) {
      // Full-WLS analogue: IF(Ŵ_i) = -W IF_i(Γ̂) W, so the row correction added
      // to g_i W is d' W IF_i(Γ̂) W. `IF_i(Γ̂)` combines the data-direct
      // sandwich channel with the κ-movement channel DΓ/Dκ · IF_i(κ).
      auto inf_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_data_influence(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!inf_or.has_value()) return std::unexpected(inf_or.error());
      auto D_or = block_has_missing[b]
          ? data::ordinal_observed_gamma_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b])
          : data::ordinal_gamma_jacobian_fd(
                stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
                stats.R[b]);
      if (!D_or.has_value()) return std::unexpected(D_or.error());
      if (inf_or->rows() != G.rows() || inf_or->cols() != mb * mb ||
          D_or->rows() != mb * mb || D_or->cols() != mb) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "robust_ordinal_ij: full Gamma influence shape mismatch in block " +
                std::to_string(b)));
      }
      const Eigen::RowVectorXd lhs = d_b.transpose() * Ws[b];
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index i = 0; i < G.rows(); ++i) {
        Eigen::VectorXd if_vec = inf_or->row(i).transpose();
        if_vec.noalias() += (*D_or) * G.row(i).transpose();
        Eigen::Map<const Eigen::MatrixXd> IFGamma(if_vec.data(), mb, mb);
        correction.row(i) = lhs * IFGamma * Ws[b];
      }
    }
    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .moment_influence = G,
        .weight_correction = std::move(correction),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }
  return ij_blocks;
}

// Estimated-weight ("complete-sandwich") parameter-space sandwich {A1, B1} for
// an all-ordinal DWLS/WLS fit: the IJ counterpart of
// `ordinal_param_space_sandwich`, carrying the IF(Ŵ) meat term. Used by the
// estimated-weight robust modification-index / score-test path.
post_expected<robust::ParamSpaceSandwich>
ordinal_param_space_sandwich_ij(const data::OrdinalStats& stats,
                                const ThresholdLayout& layout,
                                const model::ImpliedMoments& moments,
                                const Eigen::VectorXd& theta,
                                const std::vector<Eigen::MatrixXd>& Ws,
                                const Eigen::MatrixXd& Delta_full,
                                OrdinalWeightKind weights,
                                OrdinalParameterization parameterization,
                                const std::vector<bool>& block_has_missing) {
  auto blocks = build_ordinal_ij_blocks(stats, layout, moments, theta, Ws,
                                        Delta_full, weights, parameterization,
                                        block_has_missing);
  if (!blocks.has_value()) return std::unexpected(blocks.error());
  if (Delta_full.rows() != [&] {
        Eigen::Index s = 0;
        for (const auto& blk : *blocks) s += blk.jacobian.rows();
        return s;
      }()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal estimated-weight score tests: moment Jacobian row count does "
        "not match the block layout"));
  }
  return weighted_param_space_sandwich_ij(*blocks);
}

}  // namespace

fit_expected<void>
prepare_ordinal_delta_partable(spec::LatentStructure& pt,
                                const data::OrdinalStats& stats,
                                spec::Starts* starts) {
  auto ordered_or = ordered_indicator_layout(pt, stats);
  if (!ordered_or.has_value()) return std::unexpected(ordered_or.error());
  const auto& ordered = *ordered_or;

  // Wu-Estabrook (2016) multigroup categorical invariance: when `Thresholds`
  // is equated across groups, lavaan releases the group-2+ ordinal response
  // scale and indicator intercept that the single-group convention otherwise
  // pins (residual variance 1, intercept 0). The standard parameterization for
  // ordinal invariance is THETA: the released scale is the free residual
  // variance `~~`, exactly how lavaan-theta reports it (the `~*~` row stays
  // fixed at 1), so the moment path standardizes the released block by its
  // implied √Σ*ᵢᵢ and no `~*~` partable projection is needed. (Under delta the
  // released `~*~` scale is unidentified — it stays pinned at 1 with a singular
  // vcov — so delta invariance is not gated; the dormant delta released-block
  // branch in `ordinal_residuals`/`ordinal_jacobian` is kept but untested.)
  // Binary items keep a fixed scale (one threshold leaves no room to identify
  // a separate scale; lavaan does the same), so the release is vetoed there.
  // At the scalar ordinal rung (`intercepts` also equated), lavaan fixes those
  // group-2+ indicator intercepts back to 0 and frees the group-2+ latent means.
  const bool release_invariant =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Thresholds) != pt.group_equal.end();
  const bool intercepts_equal =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Intercepts) != pt.group_equal.end();
  const bool means_equal =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Means) != pt.group_equal.end();
  auto is_binary = [&](std::size_t b, std::int32_t ov) {
    if (b >= stats.threshold_ov.size()) return false;
    int n = 0;
    for (std::int32_t t : stats.threshold_ov[b])
      if (t == ov) ++n;
    return n <= 1;
  };

  const std::int32_t initial_n = pt.n_free();
  std::int32_t old_n = initial_n;
  if (release_invariant && intercepts_equal && !means_equal) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] != parse::Op::Intercept || pt.group[i] < 2 ||
          pt.free[i] > 0 || pt.lhs_var[i] < 0) {
        continue;
      }
      const std::int32_t ov =
          pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
      if (ov >= 0) continue;  // observed intercept; scalar fixes these
      pt.free[i] = ++old_n;
      pt.fixed_value[i] = std::numeric_limits<double>::quiet_NaN();
      if (static_cast<std::int32_t>(pt.eq_groups.size()) == pt.free[i] - 1) {
        pt.eq_groups.push_back(pt.free[i] - 1);
      }
    }
    if (starts != nullptr &&
        starts->hint.size() < static_cast<std::size_t>(old_n)) {
      starts->hint.resize(static_cast<std::size_t>(old_n),
                          std::numeric_limits<double>::quiet_NaN());
    }
    const std::size_t n_lin = pt.lin_constraint_d.size();
    if (n_lin > 0 && old_n > initial_n &&
        pt.lin_constraint_R.size() ==
            n_lin * static_cast<std::size_t>(initial_n)) {
      std::vector<double> R_new(n_lin * static_cast<std::size_t>(old_n), 0.0);
      for (std::size_t r = 0; r < n_lin; ++r) {
        for (std::int32_t c = 0; c < initial_n; ++c) {
          R_new[r * static_cast<std::size_t>(old_n) +
                static_cast<std::size_t>(c)] =
              pt.lin_constraint_R[r * static_cast<std::size_t>(initial_n) +
                                  static_cast<std::size_t>(c)];
        }
      }
      pt.lin_constraint_R = std::move(R_new);
    }
  }
  std::vector<char> remove_free(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if ((pt.op[i] != parse::Op::Covariance &&
         pt.op[i] != parse::Op::Intercept) ||
        pt.group[i] <= 0) {
      continue;
    }
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= ordered.size()) continue;
    if (pt.lhs_var[i] < 0) continue;
    if (pt.op[i] == parse::Op::Covariance &&
        (pt.rhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i])) {
      continue;
    }
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
    if (ov < 0 || static_cast<std::size_t>(ov) >= ordered[b].size() ||
        ordered[b][static_cast<std::size_t>(ov)] == 0) {
      continue;
    }
    // Release: keep this group-2+ ordinal scale/intercept free instead of
    // pinning it. The scale is vetoed (re-fixed) for binary indicators; the
    // intercept release is suppressed when scalar intercept equality is active.
    if (release_invariant && pt.group[i] >= 2 && pt.free[i] > 0) {
      const bool is_scale = pt.op[i] == parse::Op::Covariance;
      if (is_scale) {
        if (!is_binary(b, ov)) continue;  // honor the free scale row
      } else if (!intercepts_equal) {
        continue;  // honor the free indicator intercept row
      }
    }
    if (pt.free[i] > 0) remove_free[static_cast<std::size_t>(pt.free[i])] = 1;
    pt.free[i] = 0;
    pt.fixed_value[i] = pt.op[i] == parse::Op::Covariance ? 1.0 : 0.0;
  }
  return compact_free_set(pt, remove_free, starts);
}

fit_expected<void>
prepare_ordinal_delta_partable(spec::LatentStructure& pt,
                                const data::OrdinalMoments& moments,
                                spec::Starts* starts) {
  data::OrdinalStats stats = stats_adapter(moments);
  return prepare_ordinal_delta_partable(pt, stats, starts);
}

fit_expected<void>
prepare_ordinal_partable(spec::LatentStructure& pt,
                         const data::OrdinalStats& stats,
                         OrdinalParameterization parameterization,
                         spec::Starts* starts) {
  // The prepared partable is identical for Delta and Theta: magmaan fixes the
  // ordinal-indicator residual variances and intercepts the same way for both.
  // The Delta/Theta distinction is realized in the fit objective (whether the
  // implied moments are standardized), not in the partable layout.
  (void)parameterization;
  return prepare_ordinal_delta_partable(pt, stats, starts);
}

fit_expected<void>
prepare_ordinal_partable(spec::LatentStructure& pt,
                         const data::OrdinalMoments& moments,
                         OrdinalParameterization parameterization,
                         spec::Starts* starts) {
  (void)parameterization;
  return prepare_ordinal_delta_partable(pt, moments, starts);
}

fit_expected<void>
prepare_mixed_ordinal_delta_partable(spec::LatentStructure& pt,
                                      const data::MixedOrdinalStats& stats,
                                      spec::Starts* starts) {
  auto ordered_or = ordered_indicator_layout(pt, stats);
  if (!ordered_or.has_value()) return std::unexpected(ordered_or.error());
  const auto& ordered = *ordered_or;

  const bool release_invariant =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Thresholds) != pt.group_equal.end();
  const bool intercepts_equal =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Intercepts) != pt.group_equal.end();
  const bool means_equal =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Means) != pt.group_equal.end();
  auto is_binary = [&](std::size_t b, std::int32_t ov) {
    if (b >= stats.threshold_ov.size()) return false;
    int n = 0;
    for (std::int32_t t : stats.threshold_ov[b])
      if (t == ov) ++n;
    return n <= 1;
  };

  const std::int32_t initial_n = pt.n_free();
  std::int32_t old_n = initial_n;
  if (release_invariant && intercepts_equal && !means_equal) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] != parse::Op::Intercept || pt.group[i] < 2 ||
          pt.free[i] > 0 || pt.lhs_var[i] < 0) {
        continue;
      }
      const std::int32_t ov =
          pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
      if (ov >= 0) continue;
      pt.free[i] = ++old_n;
      pt.fixed_value[i] = std::numeric_limits<double>::quiet_NaN();
      if (static_cast<std::int32_t>(pt.eq_groups.size()) == pt.free[i] - 1) {
        pt.eq_groups.push_back(pt.free[i] - 1);
      }
    }
    if (starts != nullptr &&
        starts->hint.size() < static_cast<std::size_t>(old_n)) {
      starts->hint.resize(static_cast<std::size_t>(old_n),
                          std::numeric_limits<double>::quiet_NaN());
    }
    const std::size_t n_lin = pt.lin_constraint_d.size();
    if (n_lin > 0 && old_n > initial_n &&
        pt.lin_constraint_R.size() ==
            n_lin * static_cast<std::size_t>(initial_n)) {
      std::vector<double> R_new(n_lin * static_cast<std::size_t>(old_n), 0.0);
      for (std::size_t r = 0; r < n_lin; ++r) {
        for (std::int32_t c = 0; c < initial_n; ++c) {
          R_new[r * static_cast<std::size_t>(old_n) +
                static_cast<std::size_t>(c)] =
              pt.lin_constraint_R[r * static_cast<std::size_t>(initial_n) +
                                  static_cast<std::size_t>(c)];
        }
      }
      pt.lin_constraint_R = std::move(R_new);
    }
  }
  std::vector<char> remove_free(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if ((pt.op[i] != parse::Op::Covariance &&
         pt.op[i] != parse::Op::Intercept) ||
        pt.group[i] <= 0) {
      continue;
    }
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= ordered.size()) continue;
    if (pt.lhs_var[i] < 0) continue;
    if (pt.op[i] == parse::Op::Covariance &&
        (pt.rhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i])) continue;
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
    if (ov < 0 || static_cast<std::size_t>(ov) >= ordered[b].size() ||
        ordered[b][static_cast<std::size_t>(ov)] == 0) {
      continue;
    }
    if (release_invariant && pt.group[i] >= 2 && pt.free[i] > 0) {
      const bool is_scale = pt.op[i] == parse::Op::Covariance;
      if (is_scale) {
        if (!is_binary(b, ov)) continue;
      } else if (!intercepts_equal) {
        continue;
      }
    }
    if (pt.free[i] > 0) remove_free[static_cast<std::size_t>(pt.free[i])] = 1;
    pt.free[i] = 0;
    pt.fixed_value[i] = pt.op[i] == parse::Op::Covariance ? 1.0 : 0.0;
  }
  return compact_free_set(pt, remove_free, starts);
}

fit_expected<void>
prepare_mixed_ordinal_delta_partable(spec::LatentStructure& pt,
                                      const data::MixedOrdinalMoments& moments,
                                      spec::Starts* starts) {
  auto ordered_or = ordered_indicator_layout(pt, moments);
  if (!ordered_or.has_value()) return std::unexpected(ordered_or.error());
  const auto& ordered = *ordered_or;

  const bool release_invariant =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Thresholds) != pt.group_equal.end();
  const bool intercepts_equal =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Intercepts) != pt.group_equal.end();
  const bool means_equal =
      std::find(pt.group_equal.begin(), pt.group_equal.end(),
                spec::GroupEqual::Means) != pt.group_equal.end();
  auto is_binary = [&](std::size_t b, std::int32_t ov) {
    if (b >= moments.threshold_ov.size()) return false;
    int n = 0;
    for (std::int32_t t : moments.threshold_ov[b])
      if (t == ov) ++n;
    return n <= 1;
  };

  const std::int32_t initial_n = pt.n_free();
  std::int32_t old_n = initial_n;
  if (release_invariant && intercepts_equal && !means_equal) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.op[i] != parse::Op::Intercept || pt.group[i] < 2 ||
          pt.free[i] > 0 || pt.lhs_var[i] < 0) {
        continue;
      }
      const std::int32_t ov =
          pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
      if (ov >= 0) continue;
      pt.free[i] = ++old_n;
      pt.fixed_value[i] = std::numeric_limits<double>::quiet_NaN();
      if (static_cast<std::int32_t>(pt.eq_groups.size()) == pt.free[i] - 1) {
        pt.eq_groups.push_back(pt.free[i] - 1);
      }
    }
    if (starts != nullptr &&
        starts->hint.size() < static_cast<std::size_t>(old_n)) {
      starts->hint.resize(static_cast<std::size_t>(old_n),
                          std::numeric_limits<double>::quiet_NaN());
    }
    const std::size_t n_lin = pt.lin_constraint_d.size();
    if (n_lin > 0 && old_n > initial_n &&
        pt.lin_constraint_R.size() ==
            n_lin * static_cast<std::size_t>(initial_n)) {
      std::vector<double> R_new(n_lin * static_cast<std::size_t>(old_n), 0.0);
      for (std::size_t r = 0; r < n_lin; ++r) {
        for (std::int32_t c = 0; c < initial_n; ++c) {
          R_new[r * static_cast<std::size_t>(old_n) +
                static_cast<std::size_t>(c)] =
              pt.lin_constraint_R[r * static_cast<std::size_t>(initial_n) +
                                  static_cast<std::size_t>(c)];
        }
      }
      pt.lin_constraint_R = std::move(R_new);
    }
  }
  std::vector<char> remove_free(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if ((pt.op[i] != parse::Op::Covariance &&
         pt.op[i] != parse::Op::Intercept) ||
        pt.group[i] <= 0) {
      continue;
    }
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= ordered.size()) continue;
    if (pt.lhs_var[i] < 0) continue;
    if (pt.op[i] == parse::Op::Covariance &&
        (pt.rhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i])) {
      continue;
    }
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
    if (ov < 0 || static_cast<std::size_t>(ov) >= ordered[b].size() ||
        ordered[b][static_cast<std::size_t>(ov)] == 0) {
      continue;
    }
    if (release_invariant && pt.group[i] >= 2 && pt.free[i] > 0) {
      const bool is_scale = pt.op[i] == parse::Op::Covariance;
      if (is_scale) {
        if (!is_binary(b, ov)) continue;
      } else if (!intercepts_equal) {
        continue;
      }
    }
    if (pt.free[i] > 0) {
      remove_free[static_cast<std::size_t>(pt.free[i])] = 1;
    }
    pt.free[i] = 0;
    pt.fixed_value[i] = pt.op[i] == parse::Op::Covariance ? 1.0 : 0.0;
  }
  return compact_free_set(pt, remove_free, starts);
}

fit_expected<void>
prepare_mixed_ordinal_partable(spec::LatentStructure& pt,
                                const data::MixedOrdinalStats& stats,
                                OrdinalParameterization parameterization,
                                spec::Starts* starts) {
  (void)parameterization;
  return prepare_mixed_ordinal_delta_partable(pt, stats, starts);
}

// Start-value producer for the ordinal delta path. Prepares the partable
// (delta parameterization fixes the ordinal indicator variances/intercepts —
// this is what changes n_free), seeds the structural parameters via the simple
// scheme, then overwrites the free thresholds from the sample thresholds. The
// returned vector is sized for the *prepared* partable, which is exactly what
// `fit_ordinal_bounded` rebuilds internally.
fit_expected<Eigen::VectorXd>
ordinal_start_values(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     spec::Starts starts) {
  if (auto p = prepare_ordinal_delta_partable(pt, stats, &starts);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());

  data::SampleStats samp = sample_stats_for_starts(stats);
  auto x0_or = simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());
  Eigen::VectorXd x0 = std::move(*x0_or);
  seed_threshold_starts(x0, *layout_or, stats);
  return x0;
}

fit_expected<Eigen::VectorXd>
ordinal_start_values(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalMoments& moments,
                     spec::Starts starts) {
  data::OrdinalStats stats = stats_adapter(moments);
  return ordinal_start_values(std::move(pt), rep, stats, std::move(starts));
}

fit_expected<Eigen::VectorXd>
mixed_ordinal_start_values(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           spec::Starts starts) {
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, &starts);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());

  data::SampleStats samp = sample_stats_for_starts(stats);
  auto x0_or = simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());
  Eigen::VectorXd x0 = std::move(*x0_or);
  seed_threshold_starts(x0, *layout_or, stats);
  return x0;
}

fit_expected<Eigen::VectorXd>
mixed_ordinal_start_values(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalMoments& moments,
                           spec::Starts starts) {
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  data::MixedOrdinalStats stats = stats_adapter(moments);
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, moments, &starts);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());

  data::SampleStats samp = sample_stats_for_starts(moments);
  auto x0_or = simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());
  Eigen::VectorXd x0 = std::move(*x0_or);
  seed_threshold_starts(x0, *layout_or, moments);
  return x0;
}

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalStats& stats,
               const Estimates& est,
               OrdinalWeightKind weights,
               OrdinalParameterization parameterization,
               robust::Information bread) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "OrdinalStats NACOV block count does not match MatrixRep"));
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: fitted theta length does not match ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: constraint reparameterization has incompatible shape"));
  }

  // ULS: the model weight is the identity (the ULSMV sandwich uses NACOV
  // directly). Build it once so `Ws[b]` references a live matrix through the
  // robust_weighted_moments() call below.
  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);
  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    if (stats.NACOV[b].rows() != mb || stats.NACOV[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "OrdinalStats NACOV dimension mismatch in block " + std::to_string(b)));
    }
    blocks.push_back(WeightedMomentBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .gamma = stats.NACOV[b],
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  std::optional<Eigen::MatrixXd> bread_override;
  if (bread == robust::Information::Observed) {
    auto ob = ordinal_observed_bread_analytic(
        pt, rep, stats, est, *layout_or, Ws, K, parameterization);
    if (!ob.has_value()) return std::unexpected(ob.error());
    bread_override = std::move(*ob);
  }

  // 2·est.fmin = F (est.fmin = ½F): robust_weighted_moments forms N·F.
  auto out = robust_weighted_moments(blocks, K, 2.0 * est.fmin, bread_override);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<OrdinalRobustResult>
robust_ordinal_ij(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalStats& stats,
                  const Estimates& est,
                  OrdinalWeightKind weights,
                  OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: per-case influence functions unavailable; recompute "
        "ordinal stats (moment_influence is required for the IJ)"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(missing_or.error());
  const std::vector<bool> block_has_missing = std::move(*missing_or);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: fitted theta length does not match ordinal delta "
        "partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal_ij: constraint reparameterization has incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  auto ij_blocks = build_ordinal_ij_blocks(
      stats, *layout_or, eval->moments, est.theta, Ws, Delta_full, weights,
      parameterization, block_has_missing);
  if (!ij_blocks.has_value()) return std::unexpected(ij_blocks.error());

  auto out = robust_weighted_moment_ij(*ij_blocks, K, 2.0 * est.fmin, A);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<WeightedMomentRBMParts>
ordinal_rbm_parts(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalStats& stats,
                  const Estimates& est,
                  OrdinalWeightKind weights,
                  OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: per-case influence functions unavailable; "
        "recompute ordinal stats (moment_influence is required for RBM)"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(missing_or.error());
  const std::vector<bool> block_has_missing = std::move(*missing_or);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: fitted theta length does not match ordinal delta "
        "partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rbm_parts: constraint reparameterization has incompatible "
        "shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  auto ij_blocks = build_ordinal_ij_blocks(
      stats, *layout_or, eval->moments, est.theta, Ws, Delta_full, weights,
      parameterization, block_has_missing);
  if (!ij_blocks.has_value()) return std::unexpected(ij_blocks.error());

  return weighted_moment_rbm_parts(*ij_blocks, K, A);
}

post_expected<CasewiseInfluenceIJ>
ordinal_casewise_influence_ij(spec::LatentStructure pt,
                              const model::MatrixRep& rep,
                              const data::OrdinalStats& stats,
                              const Estimates& est,
                              OrdinalWeightKind weights,
                              OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: per-case influence functions "
        "unavailable; recompute ordinal stats (moment_influence is required for "
        "the IJ)"));
  }
  auto missing_or = ordinal_ij_block_missing(stats, weights);
  if (!missing_or.has_value()) return std::unexpected(missing_or.error());
  const std::vector<bool> block_has_missing = std::move(*missing_or);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: fitted theta length does not match "
        "ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_casewise_influence_ij: constraint reparameterization has "
        "incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  auto ij_blocks = build_ordinal_ij_blocks(
      stats, *layout_or, eval->moments, est.theta, Ws, Delta_full, weights,
      parameterization, block_has_missing);
  if (!ij_blocks.has_value()) return std::unexpected(ij_blocks.error());

  return casewise_influence_from_ij_blocks(*ij_blocks, K, A);
}

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalMoments& moments,
               data::OrdinalGammaCache& gamma_cache,
               const Estimates& est,
               data::OrdinalWeightPlan plan) {
  if (plan.purpose == data::OrdinalWorkspacePurpose::FitOnly) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: cache-aware inference requires a fit-plus-inference "
        "or inference-only plan"));
  }
  if (plan.materialization != data::OrdinalGammaMaterialization::Full) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: cache-aware inference currently requires full Gamma"));
  }
  OrdinalWeightKind weight_kind;
  if (plan.estimator == data::OrdinalEstimatorKind::DWLS) {
    weight_kind = OrdinalWeightKind::DWLS;
  } else if (plan.estimator == data::OrdinalEstimatorKind::WLS) {
    weight_kind = OrdinalWeightKind::WLS;
  } else {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: cache-aware inference currently supports DWLS/WLS"));
  }

  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (gamma_cache.blocks.size() != moments.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: OrdinalGammaCache block count mismatch"));
  }
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const Eigen::Index p = moments.R[b].rows();
    const Eigen::Index mdim =
        moments.thresholds[b].size() + p * (p - 1) / 2;
    auto& block = gamma_cache.blocks[b];
    if (!block.has_full || block.gamma.rows() != mdim ||
        block.gamma.cols() != mdim || !matrix_all_finite(block.gamma)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal: full Gamma missing or malformed in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < mdim; ++k) {
      if (!(block.gamma(k, k) > 0.0)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "robust_ordinal: full Gamma diagonal is not positive in block " +
                std::to_string(b)));
      }
    }
  }

  post_expected<void> weights_ok =
      weight_kind == OrdinalWeightKind::DWLS
          ? data::ordinal_gamma_cache_ensure_dwls_weights(gamma_cache)
          : data::ordinal_gamma_cache_ensure_wls_weights(gamma_cache);
  if (!weights_ok.has_value()) return std::unexpected(weights_ok.error());

  data::OrdinalStats stats = stats_adapter(moments);
  stats.NACOV.reserve(gamma_cache.blocks.size());
  if (weight_kind == OrdinalWeightKind::DWLS) {
    stats.W_dwls.reserve(gamma_cache.blocks.size());
  } else {
    stats.W_wls.reserve(gamma_cache.blocks.size());
  }
  for (const auto& block : gamma_cache.blocks) {
    stats.NACOV.push_back(block.gamma);
    if (weight_kind == OrdinalWeightKind::DWLS) {
      stats.W_dwls.push_back(block.w_dwls);
    } else {
      stats.W_wls.push_back(block.w_wls);
    }
  }

  return robust_ordinal(std::move(pt), rep, stats, est, weight_kind,
                        to_estimate_parameterization(plan.parameterization));
}

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalStats& stats,
                     const Estimates& est,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization,
                     robust::Information bread) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: fitted theta length does not match mixed delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu, est.theta,
                            parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: constraint reparameterization has incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);
  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    blocks.push_back(WeightedMomentBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .gamma = stats.NACOV[b],
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  std::optional<Eigen::MatrixXd> bread_override;
  if (bread == robust::Information::Observed) {
    auto ob = mixed_observed_bread_analytic(
        pt, rep, stats, est, *layout_or, Ws, K, parameterization);
    if (!ob.has_value()) return std::unexpected(ob.error());
    bread_override = std::move(*ob);
  }

  // 2·est.fmin = F (est.fmin = ½F): robust_weighted_moments forms N·F.
  auto out = robust_weighted_moments(blocks, K, 2.0 * est.fmin, bread_override);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<OrdinalRobustResult>
robust_mixed_ordinal_ij(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const data::MixedOrdinalStats& stats,
                        const Estimates& est,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: per-case influence functions unavailable; "
        "recompute mixed ordinal stats (moment_influence is required for the IJ)"));
  }
  const bool has_diag_gamma_if =
      stats.gamma_diag_influence.size() == stats.R.size();
  const bool has_full_gamma_if =
      stats.gamma_full_influence.size() == stats.R.size();
  if (weights == OrdinalWeightKind::DWLS && !has_diag_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: DWLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_diag_influence or raw_data"));
  }
  if (weights == OrdinalWeightKind::WLS && !has_full_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: WLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_full_influence or raw_data"));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: fitted theta length does not match mixed "
        "delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu, est.theta,
                            parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal_ij: constraint reparameterization has "
        "incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = mixed_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const Eigen::MatrixXd& G = stats.moment_influence[b];
    if (G.rows() != stats.n_obs[b] || G.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_mixed_ordinal_ij: moment_influence shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd model_m = mixed_model_moments(
        stats, *layout_or, eval->moments, est.theta, b, parameterization);
    const Eigen::VectorXd d_b = model_m - stats.moments[b];
    Eigen::MatrixXd correction;
    if (weights == OrdinalWeightKind::DWLS) {
      Eigen::MatrixXd if_gamma;
      if (has_diag_gamma_if) {
        if_gamma = stats.gamma_diag_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: precomputed Gamma diagonal influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb ||
            D_or->rows() != mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: mixed Gamma diagonal influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index k = 0; k < mb; ++k) {
        const double gkk = stats.NACOV[b](k, k);
        if (!(gkk > 0.0)) continue;
        correction.col(k) = (d_b(k) / (gkk * gkk)) * if_gamma.col(k);
      }
    } else if (weights == OrdinalWeightKind::WLS) {
      Eigen::MatrixXd if_gamma;
      if (has_full_gamma_if) {
        if_gamma = stats.gamma_full_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb * mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: precomputed Gamma full influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb * mb ||
            D_or->rows() != mb * mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "robust_mixed_ordinal_ij: mixed full Gamma influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      const Eigen::RowVectorXd lhs = d_b.transpose() * Ws[b];
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index i = 0; i < G.rows(); ++i) {
        const Eigen::VectorXd if_vec = if_gamma.row(i).transpose();
        Eigen::Map<const Eigen::MatrixXd> IFGamma(if_vec.data(), mb, mb);
        correction.row(i) = lhs * IFGamma * Ws[b];
      }
    }
    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .moment_influence = G,
        .weight_correction = std::move(correction),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  auto out = robust_weighted_moment_ij(ij_blocks, K, 2.0 * est.fmin, A);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

post_expected<WeightedMomentRBMParts>
mixed_ordinal_rbm_parts(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const data::MixedOrdinalStats& stats,
                        const Estimates& est,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: per-case influence functions unavailable; "
        "recompute mixed ordinal stats (moment_influence is required for RBM)"));
  }
  const bool has_diag_gamma_if =
      stats.gamma_diag_influence.size() == stats.R.size();
  const bool has_full_gamma_if =
      stats.gamma_full_influence.size() == stats.R.size();
  if (weights == OrdinalWeightKind::DWLS && !has_diag_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: DWLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_diag_influence or raw_data"));
  }
  if (weights == OrdinalWeightKind::WLS && !has_full_gamma_if &&
      stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: WLS estimated-weight influence unavailable; "
        "recompute mixed ordinal stats with gamma_full_influence or raw_data"));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: fitted theta length does not match mixed "
        "delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu, est.theta,
                            parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rbm_parts: constraint reparameterization has "
        "incompatible shape"));
  }

  std::vector<Eigen::MatrixXd> uls_identity;
  if (weights == OrdinalWeightKind::ULS) {
    uls_identity.reserve(stats.NACOV.size());
    for (const auto& G : stats.NACOV)
      uls_identity.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  }
  const auto& Ws = weights == OrdinalWeightKind::ULS ? uls_identity
                 : (weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls);

  auto ob = mixed_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, Ws, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  Eigen::MatrixXd A = 0.5 * (*ob + ob->transpose()).eval();

  std::vector<WeightedMomentIJBlock> ij_blocks;
  ij_blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const Eigen::MatrixXd& G = stats.moment_influence[b];
    if (G.rows() != stats.n_obs[b] || G.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_rbm_parts: moment_influence shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd model_m = mixed_model_moments(
        stats, *layout_or, eval->moments, est.theta, b, parameterization);
    const Eigen::VectorXd d_b = model_m - stats.moments[b];
    Eigen::MatrixXd correction;
    if (weights == OrdinalWeightKind::DWLS) {
      Eigen::MatrixXd if_gamma;
      if (has_diag_gamma_if) {
        if_gamma = stats.gamma_diag_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: precomputed Gamma diagonal influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_diag_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb ||
            D_or->rows() != mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: mixed Gamma diagonal influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index k = 0; k < mb; ++k) {
        const double gkk = stats.NACOV[b](k, k);
        if (!(gkk > 0.0)) continue;
        correction.col(k) = (d_b(k) / (gkk * gkk)) * if_gamma.col(k);
      }
    } else if (weights == OrdinalWeightKind::WLS) {
      Eigen::MatrixXd if_gamma;
      if (has_full_gamma_if) {
        if_gamma = stats.gamma_full_influence[b];
        if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb * mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: precomputed Gamma full influence "
              "shape mismatch in block " + std::to_string(b)));
        }
      } else {
        const bool observed_raw = !stats.raw_data[b].allFinite();
        auto inf_or = observed_raw
            ? data::mixed_observed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_data_influence(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!inf_or.has_value()) return std::unexpected(inf_or.error());
        auto D_or = observed_raw
            ? data::mixed_observed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b])
            : data::mixed_gamma_jacobian_fd(
                  stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
                  stats.thresholds[b], stats.mean[b], stats.R[b]);
        if (!D_or.has_value()) return std::unexpected(D_or.error());
        if (inf_or->rows() != G.rows() || inf_or->cols() != mb * mb ||
            D_or->rows() != mb * mb || D_or->cols() != mb) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_rbm_parts: mixed full Gamma influence shape "
              "mismatch in block " + std::to_string(b)));
        }
        if_gamma = (*inf_or + G * D_or->transpose()).eval();
      }
      const Eigen::RowVectorXd lhs = d_b.transpose() * Ws[b];
      correction = Eigen::MatrixXd::Zero(G.rows(), mb);
      for (Eigen::Index i = 0; i < G.rows(); ++i) {
        const Eigen::VectorXd if_vec = if_gamma.row(i).transpose();
        Eigen::Map<const Eigen::MatrixXd> IFGamma(if_vec.data(), mb, mb);
        correction.row(i) = lhs * IFGamma * Ws[b];
      }
    }
    ij_blocks.push_back(WeightedMomentIJBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .moment_influence = G,
        .weight_correction = std::move(correction),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  return weighted_moment_rbm_parts(ij_blocks, K, A);
}

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalMoments& moments,
                     data::OrdinalGammaCache& gamma_cache,
                     const Estimates& est,
                     data::OrdinalWeightPlan plan) {
  if (plan.purpose == data::OrdinalWorkspacePurpose::FitOnly) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: cache-aware inference requires a "
        "fit-plus-inference or inference-only plan"));
  }
  if (plan.materialization != data::OrdinalGammaMaterialization::Full) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: cache-aware inference currently requires full "
        "Gamma"));
  }
  OrdinalWeightKind weight_kind;
  if (plan.estimator == data::OrdinalEstimatorKind::DWLS) {
    weight_kind = OrdinalWeightKind::DWLS;
  } else if (plan.estimator == data::OrdinalEstimatorKind::WLS) {
    weight_kind = OrdinalWeightKind::WLS;
  } else {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: cache-aware inference currently supports "
        "DWLS/WLS"));
  }

  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (gamma_cache.blocks.size() != moments.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: OrdinalGammaCache block count mismatch"));
  }
  for (std::size_t b = 0; b < moments.R.size(); ++b) {
    const Eigen::Index mdim = moments.moments[b].size();
    auto& block = gamma_cache.blocks[b];
    if (!block.has_full || block.gamma.rows() != mdim ||
        block.gamma.cols() != mdim || !matrix_all_finite(block.gamma)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_mixed_ordinal: full Gamma missing or malformed in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < mdim; ++k) {
      if (!(block.gamma(k, k) > 0.0)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "robust_mixed_ordinal: full Gamma diagonal is not positive in "
            "block " +
                std::to_string(b)));
      }
    }
  }

  post_expected<void> weights_ok =
      weight_kind == OrdinalWeightKind::DWLS
          ? data::ordinal_gamma_cache_ensure_dwls_weights(gamma_cache)
          : data::ordinal_gamma_cache_ensure_wls_weights(gamma_cache);
  if (!weights_ok.has_value()) return std::unexpected(weights_ok.error());

  data::MixedOrdinalStats stats = stats_adapter(moments);
  stats.NACOV.reserve(gamma_cache.blocks.size());
  if (weight_kind == OrdinalWeightKind::DWLS) {
    stats.W_dwls.reserve(gamma_cache.blocks.size());
  } else {
    stats.W_wls.reserve(gamma_cache.blocks.size());
  }
  for (const auto& block : gamma_cache.blocks) {
    stats.NACOV.push_back(block.gamma);
    if (weight_kind == OrdinalWeightKind::DWLS) {
      stats.W_dwls.push_back(block.w_dwls);
    } else {
      stats.W_wls.push_back(block.w_wls);
    }
  }

  return robust_mixed_ordinal(
      std::move(pt), rep, stats, est, weight_kind,
      to_estimate_parameterization(plan.parameterization));
}

namespace {

post_expected<Eigen::MatrixXd> invert_score_spd(const Eigen::MatrixXd& A,
                                                std::string_view what) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (A + A.transpose()));
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        std::string(what) + " is not positive definite"));
  }
  return ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
}

post_expected<inference::ScoreTestResult>
ordinal_score_for_direction(const inference::ScoreCandidate& candidate,
                            const Eigen::VectorXd& score_full,
                            const Eigen::MatrixXd& info_full,
                            const Eigen::MatrixXd& K_nuisance,
                            const Eigen::VectorXd& direction) {
  const Eigen::VectorXd I_d = info_full * direction;
  double score_eff = direction.dot(score_full);
  double info_eff = direction.dot(I_d);
  if (K_nuisance.cols() > 0) {
    const Eigen::MatrixXd I_aa =
        K_nuisance.transpose() * info_full * K_nuisance;
    const Eigen::VectorXd I_ab = K_nuisance.transpose() * I_d;
    const Eigen::VectorXd score_a = K_nuisance.transpose() * score_full;
    auto inv = invert_score_spd(I_aa, "ordinal score nuisance information");
    if (!inv.has_value()) return std::unexpected(inv.error());
    score_eff -= I_ab.dot((*inv) * score_a);
    info_eff -= I_ab.dot((*inv) * I_ab);
  }
  if (!(info_eff > 1e-10 * std::max<double>(1.0, std::abs(info_eff)))) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        "ordinal score efficient information is not positive"));
  }
  inference::ScoreTestResult out;
  out.candidate = candidate;
  out.score = score_eff;
  out.information = info_eff;
  out.mi = (score_eff * score_eff) / info_eff;
  out.df = 1;
  out.p_value = inference::chi2_pvalue(out.mi, 1);
  out.epc = score_eff / info_eff;
  return out;
}

post_expected<Eigen::MatrixXd> ordinal_null_space(const Eigen::MatrixXd& A,
                                                  Eigen::Index n_cols) {
  if (A.rows() == 0) return Eigen::MatrixXd::Identity(n_cols, n_cols);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  svd.setThreshold(1e-9);
  return Eigen::MatrixXd(svd.matrixV().rightCols(n_cols - svd.rank()));
}

post_expected<Eigen::VectorXd>
ordinal_release_direction(const EqConstraints& con, Eigen::Index release_row) {
  Eigen::MatrixXd A_rel(con.A_eq.rows() - 1, con.A_eq.cols());
  Eigen::Index out = 0;
  for (Eigen::Index r = 0; r < con.A_eq.rows(); ++r) {
    if (r == release_row) continue;
    A_rel.row(out++) = con.A_eq.row(r);
  }
  auto K_rel = ordinal_null_space(A_rel, con.npar);
  if (!K_rel.has_value()) return std::unexpected(K_rel.error());
  const Eigen::MatrixXd M = K_rel->transpose() * con.K();
  auto z = ordinal_null_space(M.transpose(), K_rel->cols());
  if (!z.has_value()) return std::unexpected(z.error());
  if (z->cols() != 1) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score equality release is not one-dimensional"));
  }
  Eigen::VectorXd d = (*K_rel) * z->col(0);
  const double norm = d.norm();
  if (!(norm > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score equality-release direction is degenerate"));
  }
  d /= norm;
  return d;
}

bool ordinal_fixed_candidate(const spec::LatentStructure& pt,
                             const model::MatrixRep& rep,
                             std::size_t row) {
  if (row >= pt.size()) return false;
  if (pt.is_constraint_row(row)) return false;
  if (pt.free[row] != 0) return false;
  if (row < pt.exo.size() && pt.exo[row] != 0) return false;
  if (row >= pt.fixed_value.size() || !std::isfinite(pt.fixed_value[row])) {
    return false;
  }
  return pt.op[row] == parse::Op::Threshold ||
         (row < rep.cell_for_row.size() && rep.cell_for_row[row].used);
}

bool ordinal_var_is_latent(const spec::LatentStructure& pt, std::int32_t v) {
  return v >= 0 && static_cast<std::size_t>(v) < pt.var_role.size() &&
         pt.var_role[static_cast<std::size_t>(v)] == spec::VarRole::Latent;
}

bool ordinal_var_is_indicator(const spec::LatentStructure& pt, std::int32_t v) {
  return v >= 0 && static_cast<std::size_t>(v) < pt.var_role.size() &&
         pt.var_role[static_cast<std::size_t>(v)] == spec::VarRole::Indicator;
}

struct OrdinalAbsentRow {
  parse::Op    op;
  std::int32_t lhs;
  std::int32_t rhs;
  std::int32_t group;
};

std::vector<OrdinalAbsentRow>
enumerate_ordinal_absent_rows(
    const spec::LatentStructure& pt,
    const inference::ModificationIndexOptions& opts) {
  std::vector<OrdinalAbsentRow> out;
  std::vector<std::int32_t> latents;
  std::vector<std::int32_t> indicators;
  for (std::int32_t v = 0; v < pt.n_vars; ++v) {
    if (ordinal_var_is_latent(pt, v)) latents.push_back(v);
    else if (ordinal_var_is_indicator(pt, v)) indicators.push_back(v);
  }

  using Key = std::array<std::int32_t, 3>;
  for (std::int32_t g = 1; g <= pt.n_groups(); ++g) {
    std::set<Key> present;
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.group[i] != g) continue;
      const std::int32_t a = pt.lhs_var[i];
      const std::int32_t b = pt.rhs_var[i];
      if (pt.op[i] == parse::Op::Measurement) {
        present.insert({0, a, b});
      } else if (pt.op[i] == parse::Op::Covariance) {
        present.insert({1, std::min(a, b), std::max(a, b)});
      } else if (pt.op[i] == parse::Op::Regression) {
        present.insert({2, a, b});
      }
    }

    if (opts.include_loadings) {
      for (const std::int32_t f : latents) {
        for (const std::int32_t x : indicators) {
          if (!present.count({0, f, x})) {
            out.push_back({parse::Op::Measurement, f, x, g});
          }
        }
      }
    }
    if (opts.include_covariances) {
      auto cov_pairs = [&](const std::vector<std::int32_t>& vs) {
        for (std::size_t i = 0; i < vs.size(); ++i) {
          for (std::size_t j = i + 1; j < vs.size(); ++j) {
            const std::int32_t a = std::min(vs[i], vs[j]);
            const std::int32_t b = std::max(vs[i], vs[j]);
            if (!present.count({1, a, b})) {
              out.push_back({parse::Op::Covariance, a, b, g});
            }
          }
        }
      };
      cov_pairs(indicators);
      cov_pairs(latents);
    }
  }
  return out;
}

spec::LatentStructure
append_ordinal_absent_rows(spec::LatentStructure pt,
                           const std::vector<OrdinalAbsentRow>& rows) {
  for (const OrdinalAbsentRow& r : rows) {
    pt.op.push_back(r.op);
    pt.lhs_var.push_back(r.lhs);
    pt.rhs_var.push_back(r.rhs);
    pt.group.push_back(r.group);
    pt.free.push_back(0);
    pt.exo.push_back(0);
    pt.fixed_value.push_back(0.0);
  }
  return pt;
}

struct OrdinalModificationIndexModel {
  spec::LatentStructure pt;
  model::MatrixRep rep;
  std::size_t original_rows = 0;
};

post_expected<OrdinalModificationIndexModel>
prepare_ordinal_modification_index_model(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const inference::ModificationIndexOptions& options) {
  if (options.candidates == inference::ScoreCandidateSet::FixedRowsOnly) {
    const std::size_t original_rows = pt.size();
    return OrdinalModificationIndexModel{std::move(pt), rep, original_rows};
  }

  const std::size_t original_rows = pt.size();
  const std::vector<OrdinalAbsentRow> absent =
      enumerate_ordinal_absent_rows(pt, options);
  pt = append_ordinal_absent_rows(std::move(pt), absent);
  auto mr = model::build_matrix_rep(pt);
  if (!mr.has_value()) return std::unexpected(model_to_post(mr.error()));
  return OrdinalModificationIndexModel{std::move(pt), std::move(*mr),
                                       original_rows};
}

void ordinal_add_free_group(spec::LatentStructure& pt, std::int32_t old_n) {
  if (static_cast<std::int32_t>(pt.eq_groups.size()) == old_n) {
    pt.eq_groups.push_back(old_n);
  } else if (!pt.eq_groups.empty()) {
    pt.eq_groups.clear();
  }
}

template <class Stats, class ResidualFn, class JacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_modification_indices_impl(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const Stats& stats,
                                  const Estimates& est,
                                  OrdinalWeightKind weights,
                                  const inference::ModificationIndexOptions& options,
                                  ResidualFn residual_fn,
                                  JacobianFn jacobian_fn,
                                  PrepareFn prepare_fn) {
  auto work = prepare_ordinal_modification_index_model(std::move(pt), rep,
                                                       options);
  if (!work.has_value()) return std::unexpected(work.error());

  if (auto v = validate_stats(stats, work->rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(work->pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != work->pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal modification indices: fitted theta length does not match delta partable"));
  }
  auto con0 = build_eq_constraints(work->pt);
  if (!con0.has_value()) return std::unexpected(con0.error());
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);

  inference::ScoreTestTable table;
  for (std::size_t row = 0; row < work->pt.size(); ++row) {
    if (!ordinal_fixed_candidate(work->pt, work->rep, row)) continue;
    spec::LatentStructure aug = work->pt;
    const double fixed_value = aug.fixed_value[row];
    const std::int32_t old_n = aug.n_free();
    aug.free[row] = old_n + 1;
    aug.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
    ordinal_add_free_group(aug, old_n);

    Eigen::VectorXd theta(est.theta.size() + 1);
    if (est.theta.size() > 0) theta.head(est.theta.size()) = est.theta;
    theta(est.theta.size()) = fixed_value;

    auto layout = make_threshold_layout(aug, work->rep, stats);
    if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
    auto factors = weight_factors(stats, weights);
    if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
    auto ev = model::ModelEvaluator::build(aug, work->rep);
    if (!ev.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal modification indices: ModelEvaluator::build failed: " +
              ev.error().detail));
    }
    auto eval = ev->evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal modification indices: fitted evaluation failed: " +
              eval.error().detail));
    }
    auto r = residual_fn(stats, *layout, eval->moments, *factors, theta);
    if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
    auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                         eval->J_mu, *factors, theta);
    if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));

    const bool generated_absent = row >= work->original_rows;
    // lavaan scores generated all-ordinal absent rows on the single-counted
    // moment scale; explicit fixed partable rows retain the fixed-row scale.
    const double moment_scale =
        (generated_absent && std::is_same_v<Stats, data::OrdinalStats>)
            ? 1.0
            : 2.0;
    const Eigen::VectorXd score =
        -moment_scale * n_total * (J->transpose() * *r);
    Eigen::MatrixXd info = moment_scale * n_total * (J->transpose() * *J);
    info = 0.5 * (info + info.transpose());

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(score.size());
    direction(score.size() - 1) = 1.0;
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::FixedParam;
    cand.row = row;
    cand.op = work->pt.op[row];
    cand.lhs_var = work->pt.lhs_var[row];
    cand.rhs_var = work->pt.rhs_var[row];
    cand.group = work->pt.group[row];
    Eigen::MatrixXd K_aug = Eigen::MatrixXd::Zero(score.size(), con0->K().cols());
    if (con0->K().rows() > 0) K_aug.topRows(con0->K().rows()) = con0->K();
    auto res = ordinal_score_for_direction(cand, score, info, K_aug, direction);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}

template <class Stats, class ResidualFn, class JacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_score_tests_impl(spec::LatentStructure pt,
                         const model::MatrixRep& rep,
                         const Stats& stats,
                         const Estimates& est,
                         OrdinalWeightKind weights,
                         ResidualFn residual_fn,
                         JacobianFn jacobian_fn,
                         PrepareFn prepare_fn) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: fitted theta length does not match delta partable"));
  }
  auto con = build_eq_constraints(pt);
  if (!con.has_value()) return std::unexpected(con.error());
  inference::ScoreTestTable table;
  if (!con->active()) return table;
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);

  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: ModelEvaluator::build failed: " + ev.error().detail));
  }
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: fitted evaluation failed: " + eval.error().detail));
  }
  auto r = residual_fn(stats, *layout, eval->moments, *factors, est.theta);
  if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
  auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                       eval->J_mu, *factors, est.theta);
  if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));
  const Eigen::VectorXd score = -n_total * (J->transpose() * *r);
  Eigen::MatrixXd info = n_total * (J->transpose() * *J);
  info = 0.5 * (info + info.transpose());

  for (Eigen::Index row = 0; row < con->A_eq.rows(); ++row) {
    auto d = ordinal_release_direction(*con, row);
    if (!d.has_value()) return std::unexpected(d.error());
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::EqualityRelease;
    cand.row = static_cast<std::size_t>(row);
    cand.op = parse::Op::EqConstraint;
    auto res = ordinal_score_for_direction(cand, score, info, con->K(), *d);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}

// ── Robust (generalized / SB-scaled) ordinal score-test machinery ────────────
// The robust twins of the two sweeps above additionally build the moment-metric
// parameter-space sandwich {A1, B1} at each (augmented) null point — the same
// {Δ_b, W_b, Γ̂_b = NACOV_b, n_b} blocks `robust_ordinal` assembles — and hand
// the per-direction scaling to `inference::frontier::score_for_direction_robust`.
// The sandwich uses the unwhitened estimation weight; c carries no
// `moment_scale` factor, so `mi_scaled` inherits the row-type scale convention
// of the ordinary `mi` and reduces to it exactly under WLS (W = Γ̂⁻¹).

Eigen::Index ordinal_sandwich_block_rows(const data::OrdinalStats& stats,
                                         std::size_t b) {
  const Eigen::Index p = stats.R[b].rows();
  return static_cast<Eigen::Index>(stats.thresholds[b].size()) +
         p * (p - 1) / 2;
}

Eigen::Index ordinal_sandwich_block_rows(const data::MixedOrdinalStats& stats,
                                         std::size_t b) {
  return stats.moments[b].size();
}

template <class Stats>
post_expected<void> validate_ordinal_nacov(const Stats& stats) {
  if (stats.NACOV.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: stats.NACOV is not populated"));
  }
  for (std::size_t b = 0; b < stats.NACOV.size(); ++b) {
    const Eigen::Index mb = ordinal_sandwich_block_rows(stats, b);
    if (stats.NACOV[b].rows() != mb || stats.NACOV[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust score tests: NACOV dimension mismatch in block " +
              std::to_string(b)));
    }
  }
  return {};
}

post_expected<std::vector<Eigen::MatrixXd>>
ordinal_sandwich_weights(const data::OrdinalStats& stats,
                         OrdinalWeightKind kind) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(stats.NACOV.size());
  if (kind == OrdinalWeightKind::ULS) {
    for (const auto& G : stats.NACOV) {
      out.push_back(Eigen::MatrixXd::Identity(G.rows(), G.cols()));
    }
    return out;
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    if (Ws[b].size() == 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust score tests: weight matrix unavailable in block " +
              std::to_string(b)));
    }
    out.push_back(Ws[b]);
  }
  return out;
}

post_expected<std::vector<Eigen::MatrixXd>>
ordinal_sandwich_weights(const data::MixedOrdinalStats& stats,
                         OrdinalWeightKind kind) {
  if (kind == OrdinalWeightKind::ULS) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal robust score tests support DWLS/WLS weights only"));
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<Eigen::MatrixXd> out;
  out.reserve(Ws.size());
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    if (Ws[b].size() == 0) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed ordinal robust score tests: weight matrix unavailable in "
          "block " + std::to_string(b)));
    }
    out.push_back(Ws[b]);
  }
  return out;
}

template <class Stats>
post_expected<robust::ParamSpaceSandwich>
ordinal_param_space_sandwich(const Stats& stats,
                             const std::vector<Eigen::MatrixXd>& Ws,
                             const Eigen::MatrixXd& Delta_full) {
  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = ordinal_sandwich_block_rows(stats, b);
    blocks.push_back(WeightedMomentBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .gamma = stats.NACOV[b],
        .n_obs = stats.n_obs[b]});
    off += mb;
  }
  if (off != Delta_full.rows()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: moment Jacobian row count does not match "
        "the block layout"));
  }
  return weighted_param_space_sandwich(blocks);
}

template <class Stats, class ResidualFn, class JacobianFn,
          class MomentJacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_modification_indices_robust_impl(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const Stats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options,
    OrdinalParameterization parameterization,
    bool estimated_weight,
    ResidualFn residual_fn,
    JacobianFn jacobian_fn,
    MomentJacobianFn moment_jacobian_fn,
    PrepareFn prepare_fn) {
  if (auto v = validate_ordinal_nacov(stats); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto work = prepare_ordinal_modification_index_model(std::move(pt), rep,
                                                       options);
  if (!work.has_value()) return std::unexpected(work.error());

  if (auto v = validate_stats(stats, work->rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(work->pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != work->pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust modification indices: fitted theta length does not "
        "match delta partable"));
  }
  auto con0 = build_eq_constraints(work->pt);
  if (!con0.has_value()) return std::unexpected(con0.error());
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);
  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());

  // Estimated-weight (complete-sandwich) meat: precompute the per-block missing
  // pattern once (all-ordinal only); mixed-ordinal is not yet wired.
  std::vector<bool> block_has_missing;
  if constexpr (std::is_same_v<Stats, data::OrdinalStats>) {
    if (estimated_weight) {
      auto missing_or = ordinal_ij_block_missing(stats, weights);
      if (!missing_or.has_value()) return std::unexpected(missing_or.error());
      block_has_missing = std::move(*missing_or);
    }
  } else if (estimated_weight) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed-ordinal estimated-weight modification indices are not yet "
        "implemented; use estimated_weight = false"));
  }

  inference::ScoreTestTable table;
  for (std::size_t row = 0; row < work->pt.size(); ++row) {
    if (!ordinal_fixed_candidate(work->pt, work->rep, row)) continue;
    spec::LatentStructure aug = work->pt;
    const double fixed_value = aug.fixed_value[row];
    const std::int32_t old_n = aug.n_free();
    aug.free[row] = old_n + 1;
    aug.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
    ordinal_add_free_group(aug, old_n);

    Eigen::VectorXd theta(est.theta.size() + 1);
    if (est.theta.size() > 0) theta.head(est.theta.size()) = est.theta;
    theta(est.theta.size()) = fixed_value;

    auto layout = make_threshold_layout(aug, work->rep, stats);
    if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
    auto factors = weight_factors(stats, weights);
    if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
    auto ev = model::ModelEvaluator::build(aug, work->rep);
    if (!ev.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust modification indices: ModelEvaluator::build failed: " +
              ev.error().detail));
    }
    auto eval = ev->evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal robust modification indices: fitted evaluation failed: " +
              eval.error().detail));
    }
    auto r = residual_fn(stats, *layout, eval->moments, *factors, theta);
    if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
    auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                         eval->J_mu, *factors, theta);
    if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));

    const Eigen::MatrixXd Delta_full = moment_jacobian_fn(
        stats, *layout, eval->moments, eval->J_sigma, eval->J_mu, theta);
    auto sw = [&]() -> post_expected<robust::ParamSpaceSandwich> {
      if constexpr (std::is_same_v<Stats, data::OrdinalStats>) {
        if (estimated_weight)
          return ordinal_param_space_sandwich_ij(
              stats, *layout, eval->moments, theta, *Ws, Delta_full, weights,
              parameterization, block_has_missing);
      }
      return ordinal_param_space_sandwich(stats, *Ws, Delta_full);
    }();
    if (!sw.has_value()) return std::unexpected(sw.error());

    const bool generated_absent = row >= work->original_rows;
    // lavaan scores generated all-ordinal absent rows on the single-counted
    // moment scale; explicit fixed partable rows retain the fixed-row scale.
    const double moment_scale =
        (generated_absent && std::is_same_v<Stats, data::OrdinalStats>)
            ? 1.0
            : 2.0;
    const Eigen::VectorXd score =
        -moment_scale * n_total * (J->transpose() * *r);
    Eigen::MatrixXd info = moment_scale * n_total * (J->transpose() * *J);
    info = 0.5 * (info + info.transpose());

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(score.size());
    direction(score.size() - 1) = 1.0;
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::FixedParam;
    cand.row = row;
    cand.op = work->pt.op[row];
    cand.lhs_var = work->pt.lhs_var[row];
    cand.rhs_var = work->pt.rhs_var[row];
    cand.group = work->pt.group[row];
    Eigen::MatrixXd K_aug = Eigen::MatrixXd::Zero(score.size(), con0->K().cols());
    if (con0->K().rows() > 0) K_aug.topRows(con0->K().rows()) = con0->K();
    auto res = inference::frontier::score_for_direction_robust(
        cand, score, info, sw->A1, sw->B1, K_aug, direction);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}

template <class Stats, class ResidualFn, class JacobianFn,
          class MomentJacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_score_tests_robust_impl(spec::LatentStructure pt,
                                const model::MatrixRep& rep,
                                const Stats& stats,
                                const Estimates& est,
                                OrdinalWeightKind weights,
                                OrdinalParameterization parameterization,
                                bool estimated_weight,
                                ResidualFn residual_fn,
                                JacobianFn jacobian_fn,
                                MomentJacobianFn moment_jacobian_fn,
                                PrepareFn prepare_fn) {
  if (auto v = validate_ordinal_nacov(stats); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: fitted theta length does not match delta "
        "partable"));
  }
  auto con = build_eq_constraints(pt);
  if (!con.has_value()) return std::unexpected(con.error());
  inference::ScoreTestTable table;
  if (!con->active()) return table;
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);
  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());

  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: ModelEvaluator::build failed: " +
            ev.error().detail));
  }
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal robust score tests: fitted evaluation failed: " +
            eval.error().detail));
  }
  auto r = residual_fn(stats, *layout, eval->moments, *factors, est.theta);
  if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
  auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                       eval->J_mu, *factors, est.theta);
  if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));
  const Eigen::VectorXd score = -n_total * (J->transpose() * *r);
  Eigen::MatrixXd info = n_total * (J->transpose() * *J);
  info = 0.5 * (info + info.transpose());

  const Eigen::MatrixXd Delta_full = moment_jacobian_fn(
      stats, *layout, eval->moments, eval->J_sigma, eval->J_mu, est.theta);
  auto sw = [&]() -> post_expected<robust::ParamSpaceSandwich> {
    if constexpr (std::is_same_v<Stats, data::OrdinalStats>) {
      if (estimated_weight) {
        auto missing_or = ordinal_ij_block_missing(stats, weights);
        if (!missing_or.has_value())
          return std::unexpected(missing_or.error());
        return ordinal_param_space_sandwich_ij(
            stats, *layout, eval->moments, est.theta, *Ws, Delta_full, weights,
            parameterization, *missing_or);
      }
    } else if (estimated_weight) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed-ordinal estimated-weight score tests are not yet implemented; "
          "use estimated_weight = false"));
    }
    return ordinal_param_space_sandwich(stats, *Ws, Delta_full);
  }();
  if (!sw.has_value()) return std::unexpected(sw.error());

  for (Eigen::Index row = 0; row < con->A_eq.rows(); ++row) {
    auto d = ordinal_release_direction(*con, row);
    if (!d.has_value()) return std::unexpected(d.error());
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::EqualityRelease;
    cand.row = static_cast<std::size_t>(row);
    cand.op = parse::Op::EqConstraint;
    auto res = inference::frontier::score_for_direction_robust(
        cand, score, info, sw->A1, sw->B1, con->K(), *d);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}

post_expected<measures::BaselineFit>
ordinal_baseline_chi2(const data::OrdinalStats& stats,
                      OrdinalWeightKind weights) {
  const auto& Ws = weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls;

  measures::BaselineFit out;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(nth + ncorr);
    d.tail(ncorr) = -corr_lower(stats.R[b]);
    if (nth > 0 && ncorr > 0) {
      const Eigen::MatrixXd Wtt = Ws[b].topLeftCorner(nth, nth);
      const Eigen::MatrixXd Wtc = Ws[b].topRightCorner(nth, ncorr);
      Eigen::LDLT<Eigen::MatrixXd> ldlt(Wtt);
      if (ldlt.info() != Eigen::Success) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "ordinal baseline threshold block is not positive definite"));
      }
      d.head(nth) = -ldlt.solve(Wtc * d.tail(ncorr));
    }
    out.chi2 += static_cast<double>(stats.n_obs[b]) *
                d.dot(Ws[b] * d);
    out.df += static_cast<int>(ncorr);
  }
  return out;
}

post_expected<measures::BaselineFit>
mixed_ordinal_baseline_chi2(const data::MixedOrdinalStats& stats,
                            OrdinalWeightKind weights) {
  if (weights == OrdinalWeightKind::ULS) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal baseline supports DWLS/WLS weights only"));
  }
  const auto& Ws = weights == OrdinalWeightKind::DWLS ? stats.W_dwls
                                                       : stats.W_wls;

  measures::BaselineFit out;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    Eigen::Index n_cont = 0;
    for (const std::int32_t flag : stats.ordered[b]) {
      if (flag == 0) ++n_cont;
    }
    const Eigen::Index nmarg = stats.thresholds[b].size() + 2 * n_cont;
    const Eigen::Index nassoc = p * (p - 1) / 2;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(nmarg + nassoc);
    if (nassoc > 0) {
      d.tail(nassoc) = -stats.moments[b].tail(nassoc);
    }
    if (nmarg > 0 && nassoc > 0) {
      const Eigen::MatrixXd Wmm = Ws[b].topLeftCorner(nmarg, nmarg);
      const Eigen::MatrixXd Wma = Ws[b].topRightCorner(nmarg, nassoc);
      Eigen::LDLT<Eigen::MatrixXd> ldlt(Wmm);
      if (ldlt.info() != Eigen::Success) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "mixed ordinal baseline marginal block is not positive definite"));
      }
      d.head(nmarg) = -ldlt.solve(Wma * d.tail(nassoc));
    }
    out.chi2 += static_cast<double>(stats.n_obs[b]) * d.dot(Ws[b] * d);
    out.df += static_cast<int>(nassoc);
  }
  return out;
}

post_expected<double>
ordinal_srmr(const data::OrdinalStats& stats,
             const model::ImpliedMoments& moments,
             OrdinalParameterization parameterization) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(fit_to_post(N.error()));

  double out = 0.0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    if (ncorr <= 0) continue;
    const Eigen::VectorXd implied =
        parameterization == OrdinalParameterization::Theta
            ? std_corr_lower(moments.sigma[b])
            : corr_lower(moments.sigma[b]);
    const Eigen::VectorXd residual = implied - corr_lower(stats.R[b]);
    const double block =
        std::sqrt(residual.squaredNorm() / static_cast<double>(vech_len(p)));
    out += (static_cast<double>(stats.n_obs[b]) /
            static_cast<double>(*N)) *
           block;
  }
  return out;
}

// Correlation root-mean-square residual (lavaan `crmr`): identical to
// `ordinal_srmr` but averaging the squared polychoric-correlation residuals over
// the off-diagonal count p(p-1)/2 rather than the full vech length p(p+1)/2 (the
// diagonal residuals are identically zero, so only the denominator differs).
post_expected<double>
ordinal_crmr(const data::OrdinalStats& stats,
             const model::ImpliedMoments& moments,
             OrdinalParameterization parameterization) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(fit_to_post(N.error()));

  double out = 0.0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    if (ncorr <= 0) continue;
    const Eigen::VectorXd implied =
        parameterization == OrdinalParameterization::Theta
            ? std_corr_lower(moments.sigma[b])
            : corr_lower(moments.sigma[b]);
    const Eigen::VectorXd residual = implied - corr_lower(stats.R[b]);
    const double block =
        std::sqrt(residual.squaredNorm() / static_cast<double>(ncorr));
    out += (static_cast<double>(stats.n_obs[b]) /
            static_cast<double>(*N)) *
           block;
  }
  return out;
}

post_expected<double>
mixed_ordinal_srmr(const data::MixedOrdinalStats& stats,
                   const ThresholdLayout& layout,
                   const model::ImpliedMoments& moments,
                   const Eigen::VectorXd& theta,
                   OrdinalParameterization parameterization) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(fit_to_post(N.error()));

  double out = 0.0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nassoc = p * (p - 1) / 2;
    if (nassoc <= 0) continue;
    const Eigen::VectorXd implied =
        mixed_model_moments(stats, layout, moments, theta, b, parameterization)
            .tail(nassoc);
    const Eigen::VectorXd observed = stats.moments[b].tail(nassoc);
    double sum_sq = 0.0;
    Eigen::Index k = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = stats.ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = stats.ordered[b][static_cast<std::size_t>(j)] != 0;
        double scale = 1.0;
        if (!oi && !oj) {
          scale = std::sqrt(stats.R[b](i, i) * stats.R[b](j, j));
        } else if (oi != oj) {
          const Eigen::Index c = oi ? j : i;
          scale = std::sqrt(stats.R[b](c, c));
        }
        const double r = (implied(k) - observed(k)) / scale;
        sum_sq += r * r;
        ++k;
      }
    }
    const double block =
        std::sqrt(sum_sq / static_cast<double>(vech_len(p)));
    out += (static_cast<double>(stats.n_obs[b]) /
            static_cast<double>(*N)) *
           block;
  }
  return out;
}

}  // namespace

post_expected<robust::LRSatorra2000Result>
lr_test_satorra2000_ordinal(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const data::OrdinalStats& stats,
    const Estimates& est_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const Estimates& est_H0,
    OrdinalWeightKind weights,
    double T_H0,
    double T_H1,
    int df_H0,
    int df_H1,
    robust::SatorraAMethod a_method,
    OrdinalParameterization parameterization) {
  (void)df_H0;
  (void)df_H1;
  if (auto v = validate_stats(stats, rep_H1, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto v = validate_stats(stats, rep_H0, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_ordinal_delta_partable(pt_H1, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (auto p = prepare_ordinal_delta_partable(pt_H0, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est_H1.theta.size() != pt_H1.n_free() ||
      est_H0.theta.size() != pt_H0.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_ordinal: fitted theta length does not match "
        "prepared ordinal partable"));
  }

  auto con1 = build_eq_constraints(pt_H1);
  if (!con1.has_value()) return std::unexpected(con1.error());
  auto con0 = build_eq_constraints(pt_H0);
  if (!con0.has_value()) return std::unexpected(con0.error());
  const int df_H1_ordinal =
      static_cast<int>(ordinal_moment_rows(stats) - con1->n_alpha);
  const int df_H0_ordinal =
      static_cast<int>(ordinal_moment_rows(stats) - con0->n_alpha);
  const int df_diff = df_H0_ordinal - df_H1_ordinal;
  if (df_diff < 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_ordinal: ordinal df_H0 - df_H1 is negative; "
        "H1 must be the less-restricted model"));
  }

  auto delta_alpha = [&](const spec::LatentStructure& pt,
                         const model::MatrixRep& rep,
                         const Estimates& est,
                         const EqConstraints& con,
                         const char* who) -> post_expected<Eigen::MatrixXd> {
    auto layout_or = make_threshold_layout(pt, rep, stats);
    if (!layout_or.has_value()) {
      return std::unexpected(fit_to_post(layout_or.error()));
    }
    auto ev_or = model::ModelEvaluator::build(pt, rep);
    if (!ev_or.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": ModelEvaluator::build failed: " +
          ev_or.error().detail));
    }
    auto eval = ev_or->evaluate(est.theta, true, true);  // J_mu for released μ
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": fitted evaluation failed: " +
          eval.error().detail));
    }
    const Eigen::MatrixXd Delta_full =
        ordinal_moment_jacobian(stats, *layout_or, eval->moments,
                                eval->J_sigma, est.theta, parameterization,
                                eval->J_mu);
    if (con.Kmat.rows() != Delta_full.cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": constraint reparameterization has "
          "incompatible shape"));
    }
    return Eigen::MatrixXd(Delta_full * con.Kmat);
  };

  auto D1_or = delta_alpha(pt_H1, rep_H1, est_H1, *con1,
                           "lr_test_satorra2000_ordinal H1");
  if (!D1_or.has_value()) return std::unexpected(D1_or.error());

  Eigen::MatrixXd A_alpha;
  if (a_method == robust::SatorraAMethod::Exact) {
    auto restr_or = robust::restriction_alpha_from_K(*con1, *con0);
    if (!restr_or.has_value()) return std::unexpected(restr_or.error());
    A_alpha = std::move(restr_or->A);
  } else {
    auto D0_or = delta_alpha(pt_H0, rep_H0, est_H0, *con0,
                             "lr_test_satorra2000_ordinal H0");
    if (!D0_or.has_value()) return std::unexpected(D0_or.error());
    auto A_or = robust::restriction_alpha_delta_from_jacobians(
        *D1_or, *D0_or, df_diff);
    if (!A_or.has_value()) return std::unexpected(A_or.error());
    A_alpha = std::move(*A_or);
  }

  const int df_diff_from_A = static_cast<int>(A_alpha.rows());
  if (df_diff_from_A != df_diff) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_ordinal: df_diff mismatch -- derived A has m = " +
        std::to_string(df_diff_from_A) + " but ordinal df_H0 - df_H1 = " +
        std::to_string(df_diff)));
  }

  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());
  auto sandwich = ordinal_param_space_sandwich(stats, *Ws, *D1_or);
  if (!sandwich.has_value()) return std::unexpected(sandwich.error());
  auto sd_or = robust::compute_satorra2000_from_sandwich(
      sandwich->A1, sandwich->B1, A_alpha);
  if (!sd_or.has_value()) return std::unexpected(sd_or.error());
  return robust::lr_test_satorra2000(T_H0 - T_H1, *sd_or);
}

post_expected<robust::LRSatorra2000Result>
lr_test_satorra2000_mixed_ordinal(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const data::MixedOrdinalStats& stats,
    const Estimates& est_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const Estimates& est_H0,
    OrdinalWeightKind weights,
    double T_H0,
    double T_H1,
    int df_H0,
    int df_H1,
    robust::SatorraAMethod a_method,
    OrdinalParameterization parameterization) {
  (void)df_H0;
  (void)df_H1;
  if (weights == OrdinalWeightKind::ULS) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: mixed ordinal nested tests support "
        "DWLS/WLS weights only"));
  }
  if (auto v = validate_stats(stats, rep_H1, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto v = validate_stats(stats, rep_H0, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt_H1, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt_H0, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est_H1.theta.size() != pt_H1.n_free() ||
      est_H0.theta.size() != pt_H0.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: fitted theta length does not "
        "match prepared mixed ordinal partable"));
  }

  auto con1 = build_eq_constraints(pt_H1);
  if (!con1.has_value()) return std::unexpected(con1.error());
  auto con0 = build_eq_constraints(pt_H0);
  if (!con0.has_value()) return std::unexpected(con0.error());
  const int df_H1_mixed =
      static_cast<int>(mixed_moment_rows(stats) - con1->n_alpha);
  const int df_H0_mixed =
      static_cast<int>(mixed_moment_rows(stats) - con0->n_alpha);
  const int df_diff = df_H0_mixed - df_H1_mixed;
  if (df_diff < 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: mixed ordinal df_H0 - df_H1 is "
        "negative; H1 must be the less-restricted model"));
  }

  auto delta_alpha = [&](const spec::LatentStructure& pt,
                         const model::MatrixRep& rep,
                         const Estimates& est,
                         const EqConstraints& con,
                         const char* who) -> post_expected<Eigen::MatrixXd> {
    auto layout_or = make_threshold_layout(pt, rep, stats);
    if (!layout_or.has_value()) {
      return std::unexpected(fit_to_post(layout_or.error()));
    }
    auto ev_or = model::ModelEvaluator::build(pt, rep);
    if (!ev_or.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": ModelEvaluator::build failed: " +
          ev_or.error().detail));
    }
    auto eval = ev_or->evaluate(est.theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": fitted evaluation failed: " +
          eval.error().detail));
    }
    const Eigen::MatrixXd Delta_full =
        mixed_moment_jacobian(stats, *layout_or, eval->moments,
                              eval->J_sigma, eval->J_mu, est.theta,
                              parameterization);
    if (con.Kmat.rows() != Delta_full.cols()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(who) + ": constraint reparameterization has "
          "incompatible shape"));
    }
    return Eigen::MatrixXd(Delta_full * con.Kmat);
  };

  auto D1_or = delta_alpha(pt_H1, rep_H1, est_H1, *con1,
                           "lr_test_satorra2000_mixed_ordinal H1");
  if (!D1_or.has_value()) return std::unexpected(D1_or.error());

  Eigen::MatrixXd A_alpha;
  if (a_method == robust::SatorraAMethod::Exact) {
    auto restr_or = robust::restriction_alpha_from_K(*con1, *con0);
    if (!restr_or.has_value()) return std::unexpected(restr_or.error());
    A_alpha = std::move(restr_or->A);
  } else {
    auto D0_or = delta_alpha(pt_H0, rep_H0, est_H0, *con0,
                             "lr_test_satorra2000_mixed_ordinal H0");
    if (!D0_or.has_value()) return std::unexpected(D0_or.error());
    auto A_or = robust::restriction_alpha_delta_from_jacobians(
        *D1_or, *D0_or, df_diff);
    if (!A_or.has_value()) return std::unexpected(A_or.error());
    A_alpha = std::move(*A_or);
  }

  const int df_diff_from_A = static_cast<int>(A_alpha.rows());
  if (df_diff_from_A != df_diff) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "lr_test_satorra2000_mixed_ordinal: df_diff mismatch -- derived A "
        "has m = " +
        std::to_string(df_diff_from_A) +
        " but mixed ordinal df_H0 - df_H1 = " + std::to_string(df_diff)));
  }

  auto Ws = ordinal_sandwich_weights(stats, weights);
  if (!Ws.has_value()) return std::unexpected(Ws.error());
  auto sandwich = ordinal_param_space_sandwich(stats, *Ws, *D1_or);
  if (!sandwich.has_value()) return std::unexpected(sandwich.error());
  auto sd_or = robust::compute_satorra2000_from_sandwich(
      sandwich->A1, sandwich->B1, A_alpha);
  if (!sd_or.has_value()) return std::unexpected(sd_or.error());
  return robust::lr_test_satorra2000(T_H0 - T_H1, *sd_or);
}

post_expected<OrdinalFitMeasures>
fit_measures_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     const Estimates& est,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fit_measures_ordinal: fitted theta length does not match ordinal delta partable"));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fit_measures_ordinal: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, false, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fit_measures_ordinal: fitted evaluation failed: " +
            eval.error().detail));
  }

  auto baseline = ordinal_baseline_chi2(stats, weights);
  if (!baseline.has_value()) return std::unexpected(baseline.error());
  auto sr = ordinal_srmr(stats, eval->moments, parameterization);
  if (!sr.has_value()) return std::unexpected(sr.error());
  auto cr = ordinal_crmr(stats, eval->moments, parameterization);
  if (!cr.has_value()) return std::unexpected(cr.error());

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const int df = static_cast<int>(ordinal_moment_rows(stats) - con_or->n_alpha);
  const double chi2 = 2.0 * static_cast<double>(*N_or) * est.fmin;
  const measures::FitMeasures indices =
      measures::fit_measures(chi2, df, *baseline, *N_or, stats.R.size());

  return OrdinalFitMeasures{*baseline, indices, *sr, *cr};
}

post_expected<OrdinalCatmlDwlsRmsea>
catml_dwls_rmsea_ordinal(spec::LatentStructure pt,
                         const model::MatrixRep& rep,
                         const data::OrdinalStats& stats,
                         const Estimates& est,
                         OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, OrdinalWeightKind::DWLS);
      !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "catml_dwls_rmsea_ordinal: fitted theta length does not match "
        "ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  if (parameterization == OrdinalParameterization::Delta) {
    for (const auto& block : layout_or->scale_free) {
      if (std::any_of(block.begin(), block.end(), [](char c) { return c != 0; })) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "catml_dwls_rmsea_ordinal: delta fits with released response "
            "scales are not yet supported"));
      }
    }
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "catml_dwls_rmsea_ordinal: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "catml_dwls_rmsea_ordinal: fitted evaluation failed: " +
            eval.error().detail));
  }

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "catml_dwls_rmsea_ordinal: constraint reparameterization has "
        "incompatible shape"));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization, eval->J_mu);

  Eigen::Index total_full = 0;
  Eigen::Index total_assoc = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    total_full += nth + ncorr;
    total_assoc += ncorr;
  }
  if (Delta_full.rows() != total_full) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "catml_dwls_rmsea_ordinal: moment Jacobian row count mismatch"));
  }
  if (total_assoc == 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "catml_dwls_rmsea_ordinal: no association moments"));
  }

  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double N_total = static_cast<double>(*N_or);

  Eigen::MatrixXd Delta_assoc(total_assoc, Delta_full.cols());
  Eigen::MatrixXd Wg = Eigen::MatrixXd::Zero(total_assoc, total_assoc);
  Eigen::MatrixXd Vg = Eigen::MatrixXd::Zero(total_assoc, total_assoc);
  Eigen::MatrixXd Gg = Eigen::MatrixXd::Zero(total_assoc, total_assoc);
  Eigen::MatrixXd A_alpha =
      Eigen::MatrixXd::Zero(K.cols(), K.cols());

  double xx3 = 0.0;
  Eigen::Index full_off = 0;
  Eigen::Index assoc_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;
    const double fg = static_cast<double>(stats.n_obs[b]) / N_total;

    auto implied_R_or = ordinal_catml_correlation_matrix(
        eval->moments.sigma[b],
        parameterization,
        "catml_dwls_rmsea_ordinal: implied covariance block " +
            std::to_string(b));
    if (!implied_R_or.has_value()) {
      return std::unexpected(implied_R_or.error());
    }
    auto Fb_or = catml_correlation_discrepancy(stats.R[b], *implied_R_or);
    if (!Fb_or.has_value()) return std::unexpected(Fb_or.error());
    xx3 += static_cast<double>(stats.n_obs[b]) * *Fb_or;

    const Eigen::MatrixXd Dfull_b =
        Delta_full.block(full_off, 0, mdim, Delta_full.cols());
    A_alpha.noalias() +=
        fg * (K.transpose() * Dfull_b.transpose() * stats.W_dwls[b] *
              Dfull_b * K);

    const Eigen::MatrixXd Dassoc_b =
        Dfull_b.bottomRows(ncorr);
    Delta_assoc.block(assoc_off, 0, ncorr, Delta_full.cols()) = Dassoc_b;
    Wg.block(assoc_off, assoc_off, ncorr, ncorr) =
        fg * stats.W_dwls[b].bottomRightCorner(ncorr, ncorr);
    Gg.block(assoc_off, assoc_off, ncorr, ncorr) =
        (1.0 / fg) * stats.NACOV[b].bottomRightCorner(ncorr, ncorr);

    auto Vb_or = catml_correlation_information(*implied_R_or);
    if (!Vb_or.has_value()) return std::unexpected(Vb_or.error());
    Vg.block(assoc_off, assoc_off, ncorr, ncorr) = fg * *Vb_or;

    full_off += mdim;
    assoc_off += ncorr;
  }
  A_alpha = 0.5 * (A_alpha + A_alpha.transpose()).eval();
  auto Ainv_or = sym_inverse_pd_post(
      A_alpha, "catml_dwls_rmsea_ordinal categorical information");
  if (!Ainv_or.has_value()) return std::unexpected(Ainv_or.error());
  const Eigen::MatrixXd e_inv = K * (*Ainv_or) * K.transpose();

  const Eigen::Index rank_assoc = numerical_rank(Delta_assoc * K);
  const int df3 = static_cast<int>(total_assoc - rank_assoc);
  if (df3 < 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "catml_dwls_rmsea_ordinal: negative CATML df"));
  }

  OrdinalCatmlDwlsRmsea out;
  out.xx3 = std::max(0.0, xx3);
  out.df3 = df3;
  if (df3 > 0) {
    Eigen::MatrixXd wi_u =
        Eigen::MatrixXd::Identity(total_assoc, total_assoc) -
        Delta_assoc * e_inv * Delta_assoc.transpose() * Wg;
    const double ks =
        (wi_u.transpose() * Vg * wi_u * Gg).trace();
    out.c_hat3 = ks / static_cast<double>(df3);
    out.xx3_scaled = out.xx3 / out.c_hat3;
    const double num =
        std::max(out.xx3 - static_cast<double>(df3) * out.c_hat3, 0.0);
    out.rmsea_robust =
        std::sqrt(num / (static_cast<double>(df3) * N_total)) *
        std::sqrt(static_cast<double>(std::max<std::size_t>(1, stats.R.size())));
  } else {
    out.c_hat3 = std::numeric_limits<double>::quiet_NaN();
    out.xx3_scaled = std::numeric_limits<double>::quiet_NaN();
    out.rmsea_robust = 0.0;
  }
  return out;
}

post_expected<WeightedProfileRMSEAResult>
ordinal_dwls_profile_rmsea(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::OrdinalStats& stats,
                           const Estimates& est,
                           OrdinalParameterization parameterization,
                           double eig_tol) {
  if (auto v = validate_stats(stats, rep, OrdinalWeightKind::DWLS);
      !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size() ||
      stats.int_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_dwls_profile_rmsea: per-case influence functions and integer "
        "data are required (recompute ordinal stats with moment_influence and "
        "int_data)"));
  }

  // Integer-code validation and missing-pattern detection, mirroring the
  // estimated-weight influence requirements of robust_ordinal_ij.
  std::vector<bool> block_has_missing(stats.R.size(), false);
  const bool allow_pairwise_missing = stats.pairwise_gamma == "overlap";
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::MatrixXi& Xcat = stats.int_data[b];
    if (stats.n_levels[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_dwls_profile_rmsea: n_levels length mismatch in block " +
              std::to_string(b)));
    }
    if (Xcat.rows() != stats.n_obs[b] || Xcat.cols() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_dwls_profile_rmsea: int_data shape mismatch in block " +
              std::to_string(b)));
    }
    for (Eigen::Index r = 0; r < Xcat.rows(); ++r) {
      for (Eigen::Index j = 0; j < Xcat.cols(); ++j) {
        const int c = Xcat(r, j);
        if (c < 0) {
          if (!allow_pairwise_missing) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "ordinal_dwls_profile_rmsea: missing ordinal int_data requires "
                "pairwise overlap Gamma in block " + std::to_string(b)));
          }
          block_has_missing[b] = true;
          continue;
        }
        if (c >= stats.n_levels[b][static_cast<std::size_t>(j)]) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "ordinal_dwls_profile_rmsea: int_data must contain 0-based "
              "category codes in block " + std::to_string(b)));
        }
      }
    }
  }

  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_dwls_profile_rmsea: fitted theta length does not match "
        "ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  if (parameterization == OrdinalParameterization::Delta) {
    for (const auto& block : layout_or->scale_free) {
      if (std::any_of(block.begin(), block.end(),
                      [](char c) { return c != 0; })) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "ordinal_dwls_profile_rmsea: delta fits with released response "
            "scales are not yet supported"));
      }
    }
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_dwls_profile_rmsea: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_dwls_profile_rmsea: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                              est.theta, parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_dwls_profile_rmsea: constraint reparameterization has "
        "incompatible shape"));
  }

  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double N_total = static_cast<double>(*N_or);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, stats.W_dwls, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  const Eigen::MatrixXd B = 0.5 * (*ob + ob->transpose()).eval();

  std::vector<WeightedEstimatedWeightProfileBlock> blocks;
  blocks.reserve(stats.R.size());
  double weighted_F = 0.0;  // Σ_b n_b · d_bᵀ W_b d_b
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    const Eigen::MatrixXd& G = stats.moment_influence[b];  // n_b × mb
    if (G.cols() != mb || G.rows() != stats.n_obs[b]) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_dwls_profile_rmsea: moment_influence shape mismatch in "
          "block " + std::to_string(b)));
    }

    const Eigen::MatrixXd D_b = Delta_full.block(off, 0, mb, Delta_full.cols());
    const Eigen::VectorXd gamma_diag = stats.NACOV[b].diagonal();
    const Eigen::VectorXd d_b = ordinal_block_residual(
        stats, *layout_or, eval->moments, est.theta, parameterization, b);
    weighted_F += static_cast<double>(stats.n_obs[b]) *
                  (d_b.transpose() * stats.W_dwls[b] * d_b).value();

    // Per-case influence of γ̂ = diag(Γ̂): data-direct sandwich at fixed κ plus
    // the κ-movement channel Σ_l (∂Γ̂_kk/∂κ_l) g_{i,l}. Stack [g_i | IF_i(γ)]
    // so the cross-product is the joint NACOV Γ_x of (u, γ).
    auto ifg_or = block_has_missing[b]
        ? data::ordinal_observed_gamma_diag_data_influence(
              stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
              stats.R[b])
        : data::ordinal_gamma_diag_data_influence(
              stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
              stats.R[b]);
    if (!ifg_or.has_value()) return std::unexpected(ifg_or.error());
    auto jac_or = block_has_missing[b]
        ? data::ordinal_observed_gamma_diag_jacobian_fd(
              stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
              stats.R[b])
        : data::ordinal_gamma_diag_jacobian_fd(
              stats.int_data[b], stats.n_levels[b], stats.thresholds[b],
              stats.R[b]);
    if (!jac_or.has_value()) return std::unexpected(jac_or.error());
    if (ifg_or->rows() != G.rows() || ifg_or->cols() != mb ||
        jac_or->rows() != mb || jac_or->cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_dwls_profile_rmsea: gamma influence shape mismatch in "
          "block " + std::to_string(b)));
    }
    Eigen::MatrixXd if_gamma = *ifg_or;            // n_b × mb
    if_gamma.noalias() += G * jac_or->transpose();  // + κ-movement channel

    Eigen::MatrixXd H(G.rows(), 2 * mb);
    H.leftCols(mb) = G;
    H.rightCols(mb) = if_gamma;
    Eigen::MatrixXd gamma_x =
        (H.transpose() * H) / static_cast<double>(stats.n_obs[b]);
    gamma_x = 0.5 * (gamma_x + gamma_x.transpose()).eval();

    blocks.push_back(WeightedEstimatedWeightProfileBlock{
        .jacobian = D_b,
        .weight_diag = gamma_diag,
        .residual = d_b,
        .gamma = std::move(gamma_x),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  const double fmin = weighted_F / N_total;
  return weighted_moment_profile_rmsea_estimated_weight(
      blocks, K, fmin, B, stats.R.size(), eig_tol);
}

post_expected<WeightedProfileLRTResult>
ordinal_dwls_profile_lrt(spec::LatentStructure pt_H1,
                         const model::MatrixRep& rep_H1,
                         const data::OrdinalStats& stats,
                         const Estimates& est_H1,
                         spec::LatentStructure pt_H0,
                         const model::MatrixRep& rep_H0,
                         const Estimates& est_H0,
                         OrdinalParameterization parameterization,
                         double eig_tol) {
  auto h1 = ordinal_dwls_profile_rmsea(std::move(pt_H1), rep_H1, stats, est_H1,
                                       parameterization, eig_tol);
  if (!h1.has_value()) return std::unexpected(h1.error());
  auto h0 = ordinal_dwls_profile_rmsea(std::move(pt_H0), rep_H0, stats, est_H0,
                                       parameterization, eig_tol);
  if (!h0.has_value()) return std::unexpected(h0.error());
  return weighted_moment_profile_lrt(*h1, *h0, eig_tol);
}

namespace {
// Inverse standard-normal CDF (Acklam rational approximation), for the
// two-sided normal-theory CI quantile. Mirrors `normal_quantile` used elsewhere.
double inv_normal_cdf(double p) noexcept {
  p = std::clamp(p, 1e-12, 1.0 - 1e-12);
  static constexpr double a[] = {
      -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static constexpr double b[] = {
      -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01};
  static constexpr double c[] = {
      -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static constexpr double d[] = {
      7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00};
  constexpr double plow = 0.02425;
  constexpr double phigh = 1.0 - plow;
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
            c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
             c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
         q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

// Zero the γ coordinates of a block-diagonal extended (u,γ) NACOV — the
// fixed-weight comparator that drops the estimated-weight channel. Each group b
// occupies a 2mb diagonal block laid out [u (mb); γ (mb)]; `m_blocks` gives the
// per-group u-moment count mb (so the single-group case zeros the trailing band).
void zero_extended_gamma_channel(Eigen::MatrixXd& Gamma,
                                 const std::vector<Eigen::Index>& m_blocks) {
  Eigen::Index off = 0;
  for (Eigen::Index mb : m_blocks) {
    Gamma.middleRows(off + mb, mb).setZero();
    Gamma.middleCols(off + mb, mb).setZero();
    off += 2 * mb;
  }
}

post_expected<Eigen::MatrixXd>
mixed_diag_gamma_influence_block(const data::MixedOrdinalStats& stats,
                                 std::size_t b,
                                 bool has_diag_gamma_if,
                                 const Eigen::MatrixXd& G,
                                 Eigen::Index mb,
                                 const std::string& context) {
  if (has_diag_gamma_if) {
    Eigen::MatrixXd if_gamma = stats.gamma_diag_influence[b];
    if (if_gamma.rows() != G.rows() || if_gamma.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          context + ": precomputed Gamma diagonal influence shape mismatch "
          "in block " + std::to_string(b)));
    }
    return if_gamma;
  }

  const bool observed_raw = !stats.raw_data[b].allFinite();
  auto ifg_or = observed_raw
      ? data::mixed_observed_gamma_diag_data_influence(
            stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
            stats.thresholds[b], stats.mean[b], stats.R[b])
      : data::mixed_gamma_diag_data_influence(
            stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
            stats.thresholds[b], stats.mean[b], stats.R[b]);
  if (!ifg_or.has_value()) return std::unexpected(ifg_or.error());
  auto jac_or = observed_raw
      ? data::mixed_observed_gamma_diag_jacobian_fd(
            stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
            stats.thresholds[b], stats.mean[b], stats.R[b])
      : data::mixed_gamma_diag_jacobian_fd(
            stats.raw_data[b], stats.ordered[b], stats.n_levels[b],
            stats.thresholds[b], stats.mean[b], stats.R[b]);
  if (!jac_or.has_value()) return std::unexpected(jac_or.error());
  if (ifg_or->rows() != G.rows() || ifg_or->cols() != mb ||
      jac_or->rows() != mb || jac_or->cols() != mb) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        context + ": mixed Gamma diagonal influence shape mismatch in block " +
            std::to_string(b)));
  }
  return (*ifg_or + G * jac_or->transpose()).eval();
}
}  // namespace

post_expected<OrdinalCrmrInference>
ordinal_crmr_misspec_inference(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const data::OrdinalStats& stats,
                               const Estimates& est,
                               OrdinalParameterization parameterization,
                               bool estimated_weight,
                               bool srmr_denominator,
                               double conf_level,
                               double eig_tol) {
  if (auto v = validate_stats(stats, rep, OrdinalWeightKind::DWLS);
      !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size() ||
      stats.int_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: per-case influence functions and "
        "integer data are required (recompute ordinal stats with "
        "moment_influence and int_data)"));
  }
  if (!(conf_level > 0.0 && conf_level < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: conf_level must lie in (0,1)"));
  }

  // Per-group u-moment counts and totals. CRMR/SRMR pool a per-correlation mean
  // square, so the denominator (k = ncorr / vech) must be common across groups.
  const Eigen::Index p = stats.R[0].rows();
  const Eigen::Index ncorr = p * (p - 1) / 2;
  if (ncorr <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: no association moments"));
  }
  std::vector<Eigen::Index> m_blocks(stats.R.size());
  Eigen::Index M = 0;
  for (std::size_t bb = 0; bb < stats.R.size(); ++bb) {
    const Eigen::Index pb = stats.R[bb].rows();
    if (pb != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_crmr_misspec_inference: all groups must have the same number "
          "of variables (common CRMR/SRMR denominator)"));
    }
    m_blocks[bb] = stats.thresholds[bb].size() + pb * (pb - 1) / 2;
    M += m_blocks[bb];
  }

  // Integer-code validation + missing-pattern detection per group (mirror the
  // profile path).
  std::vector<bool> block_has_missing(stats.R.size(), false);
  const bool allow_pairwise_missing = stats.pairwise_gamma == "overlap";
  for (std::size_t bb = 0; bb < stats.R.size(); ++bb) {
    const Eigen::MatrixXi& Xcat = stats.int_data[bb];
    if (stats.n_levels[bb].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_crmr_misspec_inference: n_levels length mismatch"));
    }
    if (Xcat.rows() != stats.n_obs[bb] || Xcat.cols() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_crmr_misspec_inference: int_data shape mismatch"));
    }
    for (Eigen::Index r = 0; r < Xcat.rows(); ++r) {
      for (Eigen::Index j = 0; j < Xcat.cols(); ++j) {
        const int c = Xcat(r, j);
        if (c < 0) {
          if (!allow_pairwise_missing) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "ordinal_crmr_misspec_inference: missing int_data requires "
                "pairwise overlap Gamma"));
          }
          block_has_missing[bb] = true;
          continue;
        }
        if (c >= stats.n_levels[bb][static_cast<std::size_t>(j)]) {
          return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
              "ordinal_crmr_misspec_inference: int_data must be 0-based codes"));
        }
      }
    }
  }

  if (auto pr = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !pr.has_value()) {
    return std::unexpected(fit_to_post(pr.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: fitted theta length does not match "
        "ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  if (parameterization == OrdinalParameterization::Delta) {
    for (const auto& blk : layout_or->scale_free) {
      if (std::any_of(blk.begin(), blk.end(),
                      [](char c) { return c != 0; })) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "ordinal_crmr_misspec_inference: delta fits with released response "
            "scales are not yet supported"));
      }
    }
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd D = ordinal_moment_jacobian(
      stats, *layout_or, eval->moments, eval->J_sigma, est.theta,
      parameterization);  // M × q (all blocks stacked)
  if (D.rows() != M) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: moment Jacobian row count mismatch"));
  }

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != D.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_crmr_misspec_inference: constraint reparameterization has "
        "incompatible shape"));
  }

  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double N_total = static_cast<double>(*N_or);

  auto ob = ordinal_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, stats.W_dwls, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  const Eigen::MatrixXd Balpha = 0.5 * (*ob + ob->transpose()).eval();
  auto Binv_or = sym_inverse_pd_post(
      Balpha, "ordinal_crmr_misspec_inference observed bread");
  if (!Binv_or.has_value()) return std::unexpected(Binv_or.error());
  const Eigen::MatrixXd e_inv = K * (*Binv_or) * K.transpose();  // q × q

  // Per-group correlation-selector sandwich Q_{G,b} = Dφ_bᵀ V0_b Dφ_b and
  // gradient g_{G,b} = 2 Dφ_bᵀ V0_b d_b, stacked block-diagonally; the statistic
  // is the pooled Σ_b n_b G_b, G_b = ‖corr residual_b‖². The joint NACOV Γ_x is
  // the block-diagonal of the per-group stacked-influence cross-products, and
  // g_G carries the √(n_b/N) scaling so grad_var = g_Gᵀ Γ_x g_G and
  // Var(N·G) = N·grad_var pool the independent groups (note's Multi-group sec).
  Eigen::MatrixXd Q_G = Eigen::MatrixXd::Zero(2 * M, 2 * M);
  Eigen::MatrixXd Gamma_x = Eigen::MatrixXd::Zero(2 * M, 2 * M);
  Eigen::VectorXd g_G = Eigen::VectorXd::Zero(2 * M);
  double weighted_G = 0.0;  // Σ_b n_b G_b = N·G_pooled
  Eigen::Index moff = 0, xoff = 0;
  for (std::size_t bb = 0; bb < stats.R.size(); ++bb) {
    const Eigen::Index mb = m_blocks[bb];
    const Eigen::Index ncb = stats.R[bb].rows() * (stats.R[bb].rows() - 1) / 2;
    const Eigen::MatrixXd D_b = D.block(moff, 0, mb, D.cols());
    const Eigen::MatrixXd& W_b = stats.W_dwls[bb];
    const Eigen::VectorXd d_b = ordinal_block_residual(
        stats, *layout_or, eval->moments, est.theta, parameterization, bb);
    const Eigen::VectorXd gamma_b = stats.NACOV[bb].diagonal();

    // Extended residual Jacobian Dφ_b = [ −(I − P_b) | D_b e_inv D_bᵀ diag(d/γ²) ].
    const Eigen::MatrixXd DeinvDt = D_b * e_inv * D_b.transpose();  // mb × mb
    Eigen::MatrixXd Dphi(mb, 2 * mb);
    Dphi.leftCols(mb) = DeinvDt * W_b - Eigen::MatrixXd::Identity(mb, mb);
    const Eigen::VectorXd dg2 =
        (d_b.array() / gamma_b.array().square()).matrix();
    Dphi.rightCols(mb) = DeinvDt * dg2.asDiagonal();

    Eigen::VectorXd v0diag = Eigen::VectorXd::Zero(mb);
    v0diag.tail(ncb).setOnes();  // correlation-selector
    const Eigen::MatrixXd V0Dphi = v0diag.asDiagonal() * Dphi;
    Eigen::MatrixXd Q_Gb = Dphi.transpose() * V0Dphi;  // 2mb × 2mb
    Q_Gb = 0.5 * (Q_Gb + Q_Gb.transpose()).eval();
    Q_G.block(xoff, xoff, 2 * mb, 2 * mb) = Q_Gb;
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[bb]) / N_total);
    g_G.segment(xoff, 2 * mb) =
        sw * (2.0 * (Dphi.transpose() * (v0diag.asDiagonal() * d_b)));
    weighted_G +=
        static_cast<double>(stats.n_obs[bb]) * d_b.tail(ncb).squaredNorm();

    // Per-group joint NACOV Γ_{x,b} from stacked influence [Gmat_b | if_gamma_b].
    const Eigen::MatrixXd& Gmat = stats.moment_influence[bb];
    if (Gmat.cols() != mb || Gmat.rows() != stats.n_obs[bb]) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_crmr_misspec_inference: moment_influence shape mismatch"));
    }
    auto ifg_or = block_has_missing[bb]
        ? data::ordinal_observed_gamma_diag_data_influence(
              stats.int_data[bb], stats.n_levels[bb], stats.thresholds[bb],
              stats.R[bb])
        : data::ordinal_gamma_diag_data_influence(
              stats.int_data[bb], stats.n_levels[bb], stats.thresholds[bb],
              stats.R[bb]);
    if (!ifg_or.has_value()) return std::unexpected(ifg_or.error());
    auto jac_or = block_has_missing[bb]
        ? data::ordinal_observed_gamma_diag_jacobian_fd(
              stats.int_data[bb], stats.n_levels[bb], stats.thresholds[bb],
              stats.R[bb])
        : data::ordinal_gamma_diag_jacobian_fd(
              stats.int_data[bb], stats.n_levels[bb], stats.thresholds[bb],
              stats.R[bb]);
    if (!jac_or.has_value()) return std::unexpected(jac_or.error());
    if (ifg_or->rows() != Gmat.rows() || ifg_or->cols() != mb ||
        jac_or->rows() != mb || jac_or->cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_crmr_misspec_inference: gamma influence shape mismatch"));
    }
    Eigen::MatrixXd if_gamma = *ifg_or;
    if_gamma.noalias() += Gmat * jac_or->transpose();
    Eigen::MatrixXd H(Gmat.rows(), 2 * mb);
    H.leftCols(mb) = Gmat;
    H.rightCols(mb) = if_gamma;
    Eigen::MatrixXd gx =
        (H.transpose() * H) / static_cast<double>(stats.n_obs[bb]);
    Gamma_x.block(xoff, xoff, 2 * mb, 2 * mb) =
        0.5 * (gx + gx.transpose()).eval();
    moff += mb;
    xoff += 2 * mb;
  }
  if (!estimated_weight)  // fixed-weight comparator: drop the γ channel per group
    zero_extended_gamma_channel(Gamma_x, m_blocks);

  auto spec_or =
      robust::compute_profile_contrast_spectrum(Q_G, Gamma_x, eig_tol);
  if (!spec_or.has_value()) return std::unexpected(spec_or.error());

  OrdinalCrmrInference out;
  out.fixed_weight = !estimated_weight;
  out.srmr_denominator = srmr_denominator;
  out.eigvals = spec_or->eigenvalues;
  out.spectrum_size = static_cast<int>(spec_or->eigenvalues.size());
  out.bias_trace = spec_or->trace_signed;
  out.grad_var = std::max(0.0, g_G.dot(Gamma_x * g_G));
  out.k = static_cast<int>(srmr_denominator ? vech_len(p) : ncorr);
  out.stat = weighted_G;  // = N·G_pooled, G_pooled = Σ_b (n_b/N) G_b
  const double Nk = N_total * static_cast<double>(out.k);
  out.point = std::sqrt(std::max(weighted_G / N_total, 0.0) /
                        static_cast<double>(out.k));
  out.point_bias_corrected =
      std::sqrt(std::max(out.stat - out.bias_trace, 0.0) / Nk);
  out.exact_fit_pvalue =
      out.eigvals.size() > 0
          ? robust::weighted_chisq_upper(out.eigvals, std::max(0.0, out.stat))
          : std::numeric_limits<double>::quiet_NaN();
  out.warnings = std::move(spec_or->warnings);

  // Misspecification normal-theory CI on the bias-removed statistic N·G₀.
  const double z = inv_normal_cdf(0.5 * (1.0 + conf_level));
  const double sd = std::sqrt(std::max(N_total * out.grad_var, 0.0));
  const double center = out.stat - out.bias_trace;
  out.ci_lower = std::sqrt(std::max(center - z * sd, 0.0) / Nk);
  out.ci_upper = std::sqrt(std::max(center + z * sd, 0.0) / Nk);
  return out;
}

post_expected<OrdinalCrmrInference>
mixed_ordinal_crmr_misspec_inference(spec::LatentStructure pt,
                                     const model::MatrixRep& rep,
                                     const data::MixedOrdinalStats& stats,
                                     const Estimates& est,
                                     OrdinalParameterization parameterization,
                                     bool estimated_weight,
                                     bool srmr_denominator,
                                     double conf_level,
                                     double eig_tol) {
  if (auto v = validate_stats(stats, rep, OrdinalWeightKind::DWLS);
      !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: per-case influence functions "
        "are required"));
  }
  const bool has_diag_gamma_if =
      stats.gamma_diag_influence.size() == stats.R.size();
  if (!has_diag_gamma_if && stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: estimated-weight influence "
        "unavailable; recompute mixed ordinal stats with gamma_diag_influence "
        "or raw_data"));
  }
  if (!(conf_level > 0.0 && conf_level < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: conf_level must lie in (0,1)"));
  }

  const Eigen::Index p = stats.R[0].rows();
  const Eigen::Index nassoc = p * (p - 1) / 2;
  if (nassoc <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: no association moments"));
  }
  std::vector<Eigen::Index> m_blocks(stats.R.size());
  Eigen::Index M = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    if (stats.R[b].rows() != p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_crmr_misspec_inference: all groups must have the same "
          "number of variables (common CRMR/SRMR denominator)"));
    }
    m_blocks[b] = stats.moments[b].size();
    M += m_blocks[b];
  }

  if (auto pr = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !pr.has_value()) {
    return std::unexpected(fit_to_post(pr.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: fitted theta length does not "
        "match mixed delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd D =
      mixed_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma,
                            eval->J_mu, est.theta, parameterization);
  if (D.rows() != M) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: moment Jacobian row count "
        "mismatch"));
  }

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != D.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_crmr_misspec_inference: constraint reparameterization "
        "has incompatible shape"));
  }

  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double N_total = static_cast<double>(*N_or);

  auto ob = mixed_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, stats.W_dwls, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  const Eigen::MatrixXd Balpha = 0.5 * (*ob + ob->transpose()).eval();
  auto Binv_or = sym_inverse_pd_post(
      Balpha, "mixed_ordinal_crmr_misspec_inference observed bread");
  if (!Binv_or.has_value()) return std::unexpected(Binv_or.error());
  const Eigen::MatrixXd e_inv = K * (*Binv_or) * K.transpose();

  Eigen::MatrixXd Q_G = Eigen::MatrixXd::Zero(2 * M, 2 * M);
  Eigen::MatrixXd Gamma_x = Eigen::MatrixXd::Zero(2 * M, 2 * M);
  Eigen::VectorXd g_G = Eigen::VectorXd::Zero(2 * M);
  double weighted_G = 0.0;
  Eigen::Index moff = 0, xoff = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = m_blocks[b];
    const Eigen::Index pb = stats.R[b].rows();
    const Eigen::Index n_assoc_b = pb * (pb - 1) / 2;
    const Eigen::Index nth = stats.thresholds[b].size();

    std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(pb), -1);
    Eigen::Index n_cont = 0;
    for (Eigen::Index j = 0; j < pb; ++j) {
      if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
        cont_pos[static_cast<std::size_t>(j)] = n_cont++;
      }
    }
    const Eigen::Index assoc_off = nth + 2 * n_cont;
    if (assoc_off + n_assoc_b != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_crmr_misspec_inference: mixed moment layout "
          "mismatch"));
    }
    std::vector<Eigen::Index> var_row(static_cast<std::size_t>(pb), -1);
    for (Eigen::Index j = 0; j < pb; ++j) {
      const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
      if (cp >= 0) var_row[static_cast<std::size_t>(j)] = nth + n_cont + cp;
    }

    const Eigen::MatrixXd D_b = D.block(moff, 0, mb, D.cols());
    const Eigen::VectorXd d_b =
        mixed_model_moments(stats, *layout_or, eval->moments, est.theta, b,
                            parameterization) -
        stats.moments[b];
    const Eigen::VectorXd gamma_b = stats.NACOV[b].diagonal();
    const Eigen::MatrixXd& W_b = stats.W_dwls[b];
    const Eigen::MatrixXd DeinvDt = D_b * e_inv * D_b.transpose();
    Eigen::MatrixXd Dphi(mb, 2 * mb);
    Dphi.leftCols(mb) = DeinvDt * W_b - Eigen::MatrixXd::Identity(mb, mb);
    const Eigen::VectorXd dg2 =
        (d_b.array() / gamma_b.array().square()).matrix();
    Dphi.rightCols(mb) = DeinvDt * dg2.asDiagonal();

    Eigen::MatrixXd Dstd(n_assoc_b, 2 * mb);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n_assoc_b);
    Eigen::Index k = 0;
    for (Eigen::Index j = 0; j < pb; ++j) {
      for (Eigen::Index i = j + 1; i < pb; ++i) {
        const Eigen::Index assoc_row = assoc_off + k;
        const bool oi = stats.ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = stats.ordered[b][static_cast<std::size_t>(j)] != 0;
        double inv_scale = 1.0;
        std::vector<Eigen::Index> scale_vars;
        if (!oi && !oj) {
          scale_vars = {i, j};
          const double vi = stats.R[b](i, i);
          const double vj = stats.R[b](j, j);
          if (!(vi > 0.0) || !(vj > 0.0)) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_crmr_misspec_inference: non-positive "
                "continuous variance in standardization"));
          }
          inv_scale = 1.0 / std::sqrt(vi * vj);
        } else if (oi != oj) {
          const Eigen::Index c = oi ? j : i;
          scale_vars = {c};
          const double vc = stats.R[b](c, c);
          if (!(vc > 0.0)) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_crmr_misspec_inference: non-positive "
                "continuous variance in standardization"));
          }
          inv_scale = 1.0 / std::sqrt(vc);
        }

        z(k) = inv_scale * d_b(assoc_row);
        Dstd.row(k) = inv_scale * Dphi.row(assoc_row);
        for (const Eigen::Index c : scale_vars) {
          const Eigen::Index vr = var_row[static_cast<std::size_t>(c)];
          if (vr < 0) {
            return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_crmr_misspec_inference: missing continuous "
                "variance row in standardization"));
          }
          Dstd(k, vr) += d_b(assoc_row) *
                         (-0.5 * inv_scale / stats.R[b](c, c));
        }
        ++k;
      }
    }

    Eigen::MatrixXd Q_Gb = Dstd.transpose() * Dstd;
    Q_Gb = 0.5 * (Q_Gb + Q_Gb.transpose()).eval();
    Q_G.block(xoff, xoff, 2 * mb, 2 * mb) = Q_Gb;
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) / N_total);
    g_G.segment(xoff, 2 * mb) = sw * (2.0 * (Dstd.transpose() * z));
    weighted_G += static_cast<double>(stats.n_obs[b]) * z.squaredNorm();

    const Eigen::MatrixXd& Gmat = stats.moment_influence[b];
    if (Gmat.cols() != mb || Gmat.rows() != stats.n_obs[b]) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_crmr_misspec_inference: moment_influence shape "
          "mismatch in block " + std::to_string(b)));
    }
    auto ifg_or = mixed_diag_gamma_influence_block(
        stats, b, has_diag_gamma_if, Gmat, mb,
        "mixed_ordinal_crmr_misspec_inference");
    if (!ifg_or.has_value()) return std::unexpected(ifg_or.error());
    Eigen::MatrixXd H(Gmat.rows(), 2 * mb);
    H.leftCols(mb) = Gmat;
    H.rightCols(mb) = *ifg_or;
    Eigen::MatrixXd gx =
        (H.transpose() * H) / static_cast<double>(stats.n_obs[b]);
    Gamma_x.block(xoff, xoff, 2 * mb, 2 * mb) =
        0.5 * (gx + gx.transpose()).eval();

    moff += mb;
    xoff += 2 * mb;
  }
  if (!estimated_weight) zero_extended_gamma_channel(Gamma_x, m_blocks);

  auto spec_or =
      robust::compute_profile_contrast_spectrum(Q_G, Gamma_x, eig_tol);
  if (!spec_or.has_value()) return std::unexpected(spec_or.error());

  OrdinalCrmrInference out;
  out.fixed_weight = !estimated_weight;
  out.srmr_denominator = srmr_denominator;
  out.eigvals = spec_or->eigenvalues;
  out.spectrum_size = static_cast<int>(spec_or->eigenvalues.size());
  out.bias_trace = spec_or->trace_signed;
  out.grad_var = std::max(0.0, g_G.dot(Gamma_x * g_G));
  out.k = static_cast<int>(srmr_denominator ? vech_len(p) : nassoc);
  out.stat = weighted_G;
  const double Nk = N_total * static_cast<double>(out.k);
  out.point = std::sqrt(std::max(weighted_G / N_total, 0.0) /
                        static_cast<double>(out.k));
  out.point_bias_corrected =
      std::sqrt(std::max(out.stat - out.bias_trace, 0.0) / Nk);
  out.exact_fit_pvalue =
      out.eigvals.size() > 0
          ? robust::weighted_chisq_upper(out.eigvals, std::max(0.0, out.stat))
          : std::numeric_limits<double>::quiet_NaN();
  out.warnings = std::move(spec_or->warnings);

  const double zcrit = inv_normal_cdf(0.5 * (1.0 + conf_level));
  const double sd = std::sqrt(std::max(N_total * out.grad_var, 0.0));
  const double center = out.stat - out.bias_trace;
  out.ci_lower = std::sqrt(std::max(center - zcrit * sd, 0.0) / Nk);
  out.ci_upper = std::sqrt(std::max(center + zcrit * sd, 0.0) / Nk);
  return out;
}

post_expected<OrdinalRmseaInference>
ordinal_rmsea_misspec_inference(spec::LatentStructure pt,
                                const model::MatrixRep& rep,
                                const data::OrdinalStats& stats,
                                const Estimates& est,
                                OrdinalParameterization parameterization,
                                bool estimated_weight,
                                double conf_level,
                                double eig_tol) {
  if (!(conf_level > 0.0 && conf_level < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rmsea_misspec_inference: conf_level must lie in (0,1)"));
  }

  // The profile path validates the fit and supplies Q (profile_hessian), the
  // joint NACOV Γ_x (gamma), the signed bias trace, the QΓ spectrum, F (fmin),
  // N·F (chisq_standard), df, and N. RMSEA's criterion is the discrepancy F
  // itself, so the bias/Hessian objects are exactly the profile ones.
  auto prof_or = ordinal_dwls_profile_rmsea(pt, rep, stats, est,
                                            parameterization, eig_tol);
  if (!prof_or.has_value()) return std::unexpected(prof_or.error());
  const WeightedProfileRMSEAResult& prof = *prof_or;

  // Per-group u-moment counts and the total extended dimension.
  std::vector<Eigen::Index> m_blocks(stats.R.size());
  Eigen::Index M = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index pb = stats.R[b].rows();
    m_blocks[b] = stats.thresholds[b].size() + pb * (pb - 1) / 2;
    M += m_blocks[b];
  }
  if (prof.profile_hessian.rows() != 2 * M || prof.gamma.rows() != 2 * M) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rmsea_misspec_inference: unexpected profile dimension"));
  }

  // Recompute the fitted residuals d_b = σ_b(θ̂) − u_b for the envelope-score
  // gradient. The criterion is the pooled F = Σ_b (n_b/N) F_b, so over the
  // stacked x = (x_1,…,x_G) its gradient is g_F = stack_b √(n_b/N)·(−2 W_b d_b,
  // −d_b²/γ_b²); then grad_var = g_Fᵀ Γ_x g_F and Var(N·F) = N·grad_var pool the
  // independent groups (Section "Multi-group" of the note).
  spec::LatentStructure pt2 = std::move(pt);
  if (auto pr = prepare_ordinal_delta_partable(pt2, stats, nullptr);
      !pr.has_value()) {
    return std::unexpected(fit_to_post(pr.error()));
  }
  auto layout_or = make_threshold_layout(pt2, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  auto ev_or = model::ModelEvaluator::build(pt2, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rmsea_misspec_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_rmsea_misspec_inference: fitted evaluation failed: " +
            eval.error().detail));
  }

  const double N = static_cast<double>(prof.ntotal);
  Eigen::VectorXd g_F = Eigen::VectorXd::Zero(2 * M);
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = m_blocks[b];
    const Eigen::VectorXd d_b = ordinal_block_residual(
        stats, *layout_or, eval->moments, est.theta, parameterization, b);
    if (d_b.size() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_rmsea_misspec_inference: residual length mismatch"));
    }
    const Eigen::VectorXd gamma_b = stats.NACOV[b].diagonal();
    const Eigen::MatrixXd& W_b = stats.W_dwls[b];
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) / N);
    g_F.segment(off, mb) = sw * (-2.0 * (W_b * d_b));
    g_F.segment(off + mb, mb) =
        sw * (-(d_b.array().square() / gamma_b.array().square()).matrix());
    off += 2 * mb;
  }

  Eigen::MatrixXd Gamma_used = prof.gamma;
  double bias = prof.trace_signed;
  Eigen::VectorXd eigvals = prof.eigvals;
  int spectrum_size = prof.spectrum_size;
  if (!estimated_weight) {  // fixed-weight comparator: drop the γ channel
    zero_extended_gamma_channel(Gamma_used, m_blocks);
    auto spec = robust::compute_profile_contrast_spectrum(
        prof.profile_hessian, Gamma_used, eig_tol);
    if (!spec.has_value()) return std::unexpected(spec.error());
    bias = spec->trace_signed;
    eigvals = spec->eigenvalues;
    spectrum_size = static_cast<int>(spec->eigenvalues.size());
  }
  const double grad_var = std::max(0.0, g_F.dot(Gamma_used * g_F));

  const double F = prof.fmin;
  const double Gn = static_cast<double>(prof.n_groups);
  const int df = prof.df;

  OrdinalRmseaInference out;
  out.fixed_weight = !estimated_weight;
  out.eigvals = std::move(eigvals);
  out.spectrum_size = spectrum_size;
  out.bias_trace = bias;
  out.grad_var = grad_var;
  out.fmin = F;
  out.stat = prof.chisq_standard;  // N · F
  out.df = df;
  out.warnings = prof.warnings;
  if (df > 0 && N > 0.0) {
    const double denom = static_cast<double>(df);
    const double center = F - bias / N;            // estimate of F₀
    const double sd = std::sqrt(std::max(grad_var / N, 0.0));
    const double z = inv_normal_cdf(0.5 * (1.0 + conf_level));
    out.point = std::sqrt(std::max(center, 0.0) * Gn / denom);
    out.ci_lower = std::sqrt(std::max(center - z * sd, 0.0) * Gn / denom);
    out.ci_upper = std::sqrt(std::max(center + z * sd, 0.0) * Gn / denom);
  }
  out.exact_fit_pvalue =
      out.eigvals.size() > 0
          ? robust::weighted_chisq_upper(out.eigvals, std::max(0.0, out.stat))
          : std::numeric_limits<double>::quiet_NaN();
  return out;
}

post_expected<OrdinalRmseaInference>
mixed_ordinal_rmsea_misspec_inference(spec::LatentStructure pt,
                                      const model::MatrixRep& rep,
                                      const data::MixedOrdinalStats& stats,
                                      const Estimates& est,
                                      OrdinalParameterization parameterization,
                                      bool estimated_weight,
                                      double conf_level,
                                      double eig_tol) {
  if (!(conf_level > 0.0 && conf_level < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rmsea_misspec_inference: conf_level must lie in (0,1)"));
  }

  auto prof_or = mixed_ordinal_dwls_profile_rmsea(pt, rep, stats, est,
                                                  parameterization, eig_tol);
  if (!prof_or.has_value()) return std::unexpected(prof_or.error());
  const WeightedProfileRMSEAResult& prof = *prof_or;

  std::vector<Eigen::Index> m_blocks(stats.R.size());
  Eigen::Index M = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    m_blocks[b] = stats.moments[b].size();
    M += m_blocks[b];
  }
  if (prof.profile_hessian.rows() != 2 * M || prof.gamma.rows() != 2 * M) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rmsea_misspec_inference: unexpected profile dimension"));
  }

  if (auto pr = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !pr.has_value()) {
    return std::unexpected(fit_to_post(pr.error()));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rmsea_misspec_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_rmsea_misspec_inference: fitted evaluation failed: " +
            eval.error().detail));
  }

  const double N = static_cast<double>(prof.ntotal);
  Eigen::VectorXd g_F = Eigen::VectorXd::Zero(2 * M);
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = m_blocks[b];
    const Eigen::VectorXd d_b =
        mixed_model_moments(stats, *layout_or, eval->moments, est.theta, b,
                            parameterization) -
        stats.moments[b];
    if (d_b.size() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_rmsea_misspec_inference: residual length mismatch"));
    }
    const Eigen::VectorXd gamma_b = stats.NACOV[b].diagonal();
    const Eigen::MatrixXd& W_b = stats.W_dwls[b];
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) / N);
    g_F.segment(off, mb) = sw * (-2.0 * (W_b * d_b));
    g_F.segment(off + mb, mb) =
        sw * (-(d_b.array().square() / gamma_b.array().square()).matrix());
    off += 2 * mb;
  }

  Eigen::MatrixXd Gamma_used = prof.gamma;
  double bias = prof.trace_signed;
  Eigen::VectorXd eigvals = prof.eigvals;
  int spectrum_size = prof.spectrum_size;
  if (!estimated_weight) {
    zero_extended_gamma_channel(Gamma_used, m_blocks);
    auto spec = robust::compute_profile_contrast_spectrum(
        prof.profile_hessian, Gamma_used, eig_tol);
    if (!spec.has_value()) return std::unexpected(spec.error());
    bias = spec->trace_signed;
    eigvals = spec->eigenvalues;
    spectrum_size = static_cast<int>(spec->eigenvalues.size());
  }
  const double grad_var = std::max(0.0, g_F.dot(Gamma_used * g_F));

  const double F = prof.fmin;
  const double Gn = static_cast<double>(prof.n_groups);
  const int df = prof.df;

  OrdinalRmseaInference out;
  out.fixed_weight = !estimated_weight;
  out.eigvals = std::move(eigvals);
  out.spectrum_size = spectrum_size;
  out.bias_trace = bias;
  out.grad_var = grad_var;
  out.fmin = F;
  out.stat = prof.chisq_standard;
  out.df = df;
  out.warnings = prof.warnings;
  if (df > 0 && N > 0.0) {
    const double denom = static_cast<double>(df);
    const double center = F - bias / N;
    const double sd = std::sqrt(std::max(grad_var / N, 0.0));
    const double z = inv_normal_cdf(0.5 * (1.0 + conf_level));
    out.point = std::sqrt(std::max(center, 0.0) * Gn / denom);
    out.ci_lower = std::sqrt(std::max(center - z * sd, 0.0) * Gn / denom);
    out.ci_upper = std::sqrt(std::max(center + z * sd, 0.0) * Gn / denom);
  }
  out.exact_fit_pvalue =
      out.eigvals.size() > 0
          ? robust::weighted_chisq_upper(out.eigvals, std::max(0.0, out.stat))
          : std::numeric_limits<double>::quiet_NaN();
  return out;
}

post_expected<OrdinalIncrementalFitInference>
ordinal_cfi_tli_misspec_inference(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const data::OrdinalStats& stats,
                                  const Estimates& est,
                                  OrdinalParameterization parameterization,
                                  bool estimated_weight,
                                  double conf_level,
                                  double eig_tol) {
  if (!(conf_level > 0.0 && conf_level < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_cfi_tli_misspec_inference: conf_level must lie in (0,1)"));
  }

  // User model: profile Hessian Q_u, the shared joint NACOV Γ_x, T_u = N·F_u,
  // signed bias trace Q̄_u, df, N. The profile path also validates the fit and
  // the moment_influence/int_data requirements.
  spec::LatentStructure pt_user = pt;  // profile consumes its argument
  auto prof_or = ordinal_dwls_profile_rmsea(std::move(pt_user), rep, stats, est,
                                            parameterization, eig_tol);
  if (!prof_or.has_value()) return std::unexpected(prof_or.error());
  const WeightedProfileRMSEAResult& prof = *prof_or;

  // Per-group threshold / correlation / u-moment counts and the totals.
  std::vector<Eigen::Index> m_blocks(stats.R.size());
  std::vector<Eigen::Index> nth_blocks(stats.R.size());
  std::vector<Eigen::Index> ncorr_blocks(stats.R.size());
  Eigen::Index M = 0, nth_total = 0, ncorr_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index pb = stats.R[b].rows();
    nth_blocks[b] = stats.thresholds[b].size();
    ncorr_blocks[b] = pb * (pb - 1) / 2;
    m_blocks[b] = nth_blocks[b] + ncorr_blocks[b];
    M += m_blocks[b];
    nth_total += nth_blocks[b];
    ncorr_total += ncorr_blocks[b];
  }
  if (ncorr_total <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_cfi_tli_misspec_inference: no association moments"));
  }
  if (prof.profile_hessian.rows() != 2 * M || prof.gamma.rows() != 2 * M) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_cfi_tli_misspec_inference: unexpected profile dimension"));
  }

  // User residuals d_{u,b} = σ_b(θ̂) − u_b (recompute as the RMSEA path does).
  if (auto pr = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !pr.has_value()) {
    return std::unexpected(fit_to_post(pr.error()));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_cfi_tli_misspec_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_cfi_tli_misspec_inference: fitted evaluation failed: " +
            eval.error().detail));
  }
  const double N = static_cast<double>(prof.ntotal);

  // Stacked envelope gradients g = stack_b √(n_b/N)·(−2W_b d_b, −d_b²/γ_b²) for
  // the user and (per-group independence) baseline criteria, and the analytic
  // multi-group baseline through the same primitive with the SAME per-group
  // Γ_{x,b} as the user profile. The baseline model is the direct sum of the
  // groups' independence models (free thresholds, zero correlations; σ_b linear,
  // so its observed bread is the Gauss-Newton W_tt). See the note's Multi-group
  // section.
  Eigen::VectorXd g_u = Eigen::VectorXd::Zero(2 * M);
  Eigen::VectorXd g_b = Eigen::VectorXd::Zero(2 * M);
  std::vector<WeightedEstimatedWeightProfileBlock> bblocks;
  bblocks.reserve(stats.R.size());
  const Eigen::MatrixXd K_b = Eigen::MatrixXd::Identity(nth_total, nth_total);
  Eigen::MatrixXd B_b = Eigen::MatrixXd::Zero(nth_total, nth_total);
  double weighted_F_b = 0.0;
  Eigen::Index off = 0, thoff = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = m_blocks[b], nthb = nth_blocks[b], ncb = ncorr_blocks[b];
    const Eigen::VectorXd gamma_b = stats.NACOV[b].diagonal();
    const Eigen::MatrixXd& W_b = stats.W_dwls[b];
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) / N);

    const Eigen::VectorXd d_ub = ordinal_block_residual(
        stats, *layout_or, eval->moments, est.theta, parameterization, b);
    if (d_ub.size() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_cfi_tli_misspec_inference: user residual length mismatch"));
    }
    g_u.segment(off, mb) = sw * (-2.0 * (W_b * d_ub));
    g_u.segment(off + mb, mb) =
        sw * (-(d_ub.array().square() / gamma_b.array().square()).matrix());

    Eigen::VectorXd d_bb = Eigen::VectorXd::Zero(mb);
    d_bb.tail(ncb) = -corr_lower(stats.R[b]);
    g_b.segment(off, mb) = sw * (-2.0 * (W_b * d_bb));
    g_b.segment(off + mb, mb) =
        sw * (-(d_bb.array().square() / gamma_b.array().square()).matrix());

    Eigen::MatrixXd D_bb = Eigen::MatrixXd::Zero(mb, nth_total);
    D_bb.block(0, thoff, nthb, nthb).setIdentity();  // implied thresholds = params
    // Bread of the POOLED discrepancy: the threshold block is (n_b/N)·W_tt, so
    // its B⁻¹ scaling cancels the two-metric √(n_b/N) Jacobian weighting (as
    // ordinal_observed_bread_analytic does for the user model). Without the
    // weight the baseline Q_b/bias would not pool correctly across groups.
    B_b.block(thoff, thoff, nthb, nthb).diagonal() =
        (sw * sw * gamma_b.head(nthb).array().inverse()).matrix();
    weighted_F_b += static_cast<double>(stats.n_obs[b]) *
                    (d_bb.array().square() / gamma_b.array()).sum();
    bblocks.push_back(WeightedEstimatedWeightProfileBlock{
        .jacobian = D_bb,
        .weight_diag = gamma_b,
        .residual = d_bb,
        .gamma = prof.gamma.block(off, off, 2 * mb, 2 * mb),
        .n_obs = stats.n_obs[b]});
    off += 2 * mb;
    thoff += nthb;
  }
  auto prof_b_or = weighted_moment_profile_rmsea_estimated_weight(
      bblocks, K_b, weighted_F_b / N, B_b, stats.R.size(), eig_tol);
  if (!prof_b_or.has_value()) return std::unexpected(prof_b_or.error());
  const WeightedProfileRMSEAResult& prof_b = *prof_b_or;

  // Estimated-weight metric, or its u-block only per group (fixed-weight
  // comparator). The bias traces revert to fixed-weight values when the γ
  // channel is dropped.
  Eigen::MatrixXd Gamma_x = prof.gamma;
  double biasU = prof.trace_signed;
  double biasB = prof_b.trace_signed;
  if (!estimated_weight) {
    zero_extended_gamma_channel(Gamma_x, m_blocks);
    auto su = robust::compute_profile_contrast_spectrum(prof.profile_hessian,
                                                        Gamma_x, eig_tol);
    if (!su.has_value()) return std::unexpected(su.error());
    auto sb = robust::compute_profile_contrast_spectrum(prof_b.profile_hessian,
                                                        Gamma_x, eig_tol);
    if (!sb.has_value()) return std::unexpected(sb.error());
    biasU = su->trace_signed;
    biasB = sb->trace_signed;
  }

  OrdinalIncrementalFitInference out;
  out.fixed_weight = !estimated_weight;
  out.stat_user = prof.chisq_standard;
  out.stat_baseline = prof_b.chisq_standard;
  out.gendf_user = biasU;
  out.gendf_baseline = biasB;
  out.delta_user = out.stat_user - biasU;
  out.delta_baseline = out.stat_baseline - biasB;
  out.df_user = prof.df;
  out.df_baseline = prof_b.df;
  out.warnings = prof.warnings;
  for (const auto& w : prof_b.warnings) out.warnings.push_back(w);

  // Joint covariance of (T_u, T_b): one bilinear form in the shared Γ_x.
  const Eigen::VectorXd Gx_gu = Gamma_x * g_u;
  const Eigen::VectorXd Gx_gb = Gamma_x * g_b;
  out.var_user = std::max(0.0, N * g_u.dot(Gx_gu));
  out.var_baseline = std::max(0.0, N * g_b.dot(Gx_gb));
  out.cov_user_baseline = N * g_u.dot(Gx_gb);

  const double z = inv_normal_cdf(0.5 * (1.0 + conf_level));
  const double db = out.delta_baseline;
  if (!(db > 0.0)) {
    out.cfi = 1.0;
    out.tli = std::numeric_limits<double>::quiet_NaN();
    out.cfi_ci_lower = out.cfi_ci_upper =
        std::numeric_limits<double>::quiet_NaN();
    out.tli_ci_lower = out.tli_ci_upper =
        std::numeric_limits<double>::quiet_NaN();
    out.warnings.emplace_back(
        "ordinal_cfi_tli_misspec_inference: baseline noncentrality is "
        "non-positive (independence model fits); CFI set to 1, TLI/intervals "
        "undefined");
    return out;
  }

  const double r = out.delta_user / db;  // 1 − CFI
  out.var_cfi = std::max(0.0,
      (out.var_user - 2.0 * r * out.cov_user_baseline +
       r * r * out.var_baseline) / (db * db));
  out.cfi = std::clamp(1.0 - r, 0.0, 1.0);
  const double sd_cfi = std::sqrt(out.var_cfi);
  out.cfi_ci_lower = std::clamp(1.0 - r - z * sd_cfi, 0.0, 1.0);
  out.cfi_ci_upper = std::clamp(1.0 - r + z * sd_cfi, 0.0, 1.0);
  if (out.delta_user <= 0.0) {
    out.warnings.emplace_back(
        "ordinal_cfi_tli_misspec_inference: user noncentrality is non-positive "
        "(near-exact fit); CFI on the boundary, normal interval approximate");
  }

  // TLI is CFI's ratio rescaled by the generalized-df ratio c = Q̄_b/Q̄_u, and
  // its interval is the CFI interval scaled by c (Var(TLI) = c²·Var(CFI)).
  if (biasU > 0.0) {
    const double c = biasB / biasU;
    out.tli = 1.0 - c * r;
    out.var_tli = c * c * out.var_cfi;
    const double sd_tli = std::sqrt(out.var_tli);
    out.tli_ci_lower = out.tli - z * sd_tli;
    out.tli_ci_upper = out.tli + z * sd_tli;
  } else {
    out.tli = std::numeric_limits<double>::quiet_NaN();
    out.var_tli = std::numeric_limits<double>::quiet_NaN();
    out.tli_ci_lower = out.tli_ci_upper =
        std::numeric_limits<double>::quiet_NaN();
    out.warnings.emplace_back(
        "ordinal_cfi_tli_misspec_inference: user generalized df is "
        "non-positive; TLI undefined");
  }
  return out;
}

post_expected<OrdinalIncrementalFitInference>
mixed_ordinal_cfi_tli_misspec_inference(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalParameterization parameterization,
    bool estimated_weight,
    double conf_level,
    double eig_tol) {
  if (!(conf_level > 0.0 && conf_level < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_cfi_tli_misspec_inference: conf_level must lie in (0,1)"));
  }

  spec::LatentStructure pt_user = pt;
  auto prof_or = mixed_ordinal_dwls_profile_rmsea(
      std::move(pt_user), rep, stats, est, parameterization, eig_tol);
  if (!prof_or.has_value()) return std::unexpected(prof_or.error());
  const WeightedProfileRMSEAResult& prof = *prof_or;

  std::vector<Eigen::Index> m_blocks(stats.R.size());
  std::vector<Eigen::Index> nmarg_blocks(stats.R.size());
  std::vector<Eigen::Index> nassoc_blocks(stats.R.size());
  Eigen::Index M = 0, nmarg_total = 0, nassoc_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index pb = stats.R[b].rows();
    Eigen::Index n_cont = 0;
    for (const std::int32_t flag : stats.ordered[b])
      if (flag == 0) ++n_cont;
    nmarg_blocks[b] = stats.thresholds[b].size() + 2 * n_cont;
    nassoc_blocks[b] = pb * (pb - 1) / 2;
    m_blocks[b] = stats.moments[b].size();
    if (nmarg_blocks[b] + nassoc_blocks[b] != m_blocks[b]) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_cfi_tli_misspec_inference: mixed moment layout "
          "mismatch"));
    }
    M += m_blocks[b];
    nmarg_total += nmarg_blocks[b];
    nassoc_total += nassoc_blocks[b];
  }
  if (nassoc_total <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_cfi_tli_misspec_inference: no association moments"));
  }
  if (prof.profile_hessian.rows() != 2 * M || prof.gamma.rows() != 2 * M) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_cfi_tli_misspec_inference: unexpected profile dimension"));
  }

  if (auto pr = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !pr.has_value()) {
    return std::unexpected(fit_to_post(pr.error()));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_cfi_tli_misspec_inference: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_cfi_tli_misspec_inference: fitted evaluation failed: " +
            eval.error().detail));
  }

  const double N = static_cast<double>(prof.ntotal);
  Eigen::VectorXd g_u = Eigen::VectorXd::Zero(2 * M);
  Eigen::VectorXd g_b = Eigen::VectorXd::Zero(2 * M);
  std::vector<WeightedEstimatedWeightProfileBlock> bblocks;
  bblocks.reserve(stats.R.size());
  const Eigen::MatrixXd K_b =
      Eigen::MatrixXd::Identity(nmarg_total, nmarg_total);
  Eigen::MatrixXd B_b = Eigen::MatrixXd::Zero(nmarg_total, nmarg_total);
  double weighted_F_b = 0.0;
  Eigen::Index xoff = 0, margoff = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = m_blocks[b];
    const Eigen::Index nmarg = nmarg_blocks[b];
    const Eigen::Index nassoc = nassoc_blocks[b];
    const Eigen::VectorXd gamma_b = stats.NACOV[b].diagonal();
    const Eigen::MatrixXd& W_b = stats.W_dwls[b];
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) / N);

    const Eigen::VectorXd d_ub =
        mixed_model_moments(stats, *layout_or, eval->moments, est.theta, b,
                            parameterization) -
        stats.moments[b];
    if (d_ub.size() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_cfi_tli_misspec_inference: user residual length "
          "mismatch"));
    }
    g_u.segment(xoff, mb) = sw * (-2.0 * (W_b * d_ub));
    g_u.segment(xoff + mb, mb) =
        sw * (-(d_ub.array().square() / gamma_b.array().square()).matrix());

    Eigen::VectorXd d_bb = Eigen::VectorXd::Zero(mb);
    d_bb.tail(nassoc) = -stats.moments[b].tail(nassoc);
    g_b.segment(xoff, mb) = sw * (-2.0 * (W_b * d_bb));
    g_b.segment(xoff + mb, mb) =
        sw * (-(d_bb.array().square() / gamma_b.array().square()).matrix());

    Eigen::MatrixXd D_bb = Eigen::MatrixXd::Zero(mb, nmarg_total);
    D_bb.block(0, margoff, nmarg, nmarg).setIdentity();
    B_b.block(margoff, margoff, nmarg, nmarg).diagonal() =
        (sw * sw * gamma_b.head(nmarg).array().inverse()).matrix();
    weighted_F_b += static_cast<double>(stats.n_obs[b]) *
                    (d_bb.transpose() * W_b * d_bb).value();
    bblocks.push_back(WeightedEstimatedWeightProfileBlock{
        .jacobian = D_bb,
        .weight_diag = gamma_b,
        .residual = d_bb,
        .gamma = prof.gamma.block(xoff, xoff, 2 * mb, 2 * mb),
        .n_obs = stats.n_obs[b]});
    xoff += 2 * mb;
    margoff += nmarg;
  }
  auto prof_b_or = weighted_moment_profile_rmsea_estimated_weight(
      bblocks, K_b, weighted_F_b / N, B_b, stats.R.size(), eig_tol);
  if (!prof_b_or.has_value()) return std::unexpected(prof_b_or.error());
  const WeightedProfileRMSEAResult& prof_b = *prof_b_or;

  Eigen::MatrixXd Gamma_x = prof.gamma;
  double biasU = prof.trace_signed;
  double biasB = prof_b.trace_signed;
  if (!estimated_weight) {
    zero_extended_gamma_channel(Gamma_x, m_blocks);
    auto su = robust::compute_profile_contrast_spectrum(prof.profile_hessian,
                                                        Gamma_x, eig_tol);
    if (!su.has_value()) return std::unexpected(su.error());
    auto sb = robust::compute_profile_contrast_spectrum(prof_b.profile_hessian,
                                                        Gamma_x, eig_tol);
    if (!sb.has_value()) return std::unexpected(sb.error());
    biasU = su->trace_signed;
    biasB = sb->trace_signed;
  }

  OrdinalIncrementalFitInference out;
  out.fixed_weight = !estimated_weight;
  out.stat_user = prof.chisq_standard;
  out.stat_baseline = prof_b.chisq_standard;
  out.gendf_user = biasU;
  out.gendf_baseline = biasB;
  out.delta_user = out.stat_user - biasU;
  out.delta_baseline = out.stat_baseline - biasB;
  out.df_user = prof.df;
  out.df_baseline = prof_b.df;
  out.warnings = prof.warnings;
  for (const auto& w : prof_b.warnings) out.warnings.push_back(w);

  const Eigen::VectorXd Gx_gu = Gamma_x * g_u;
  const Eigen::VectorXd Gx_gb = Gamma_x * g_b;
  out.var_user = std::max(0.0, N * g_u.dot(Gx_gu));
  out.var_baseline = std::max(0.0, N * g_b.dot(Gx_gb));
  out.cov_user_baseline = N * g_u.dot(Gx_gb);

  const double z = inv_normal_cdf(0.5 * (1.0 + conf_level));
  const double db = out.delta_baseline;
  if (!(db > 0.0)) {
    out.cfi = 1.0;
    out.tli = std::numeric_limits<double>::quiet_NaN();
    out.cfi_ci_lower = out.cfi_ci_upper =
        std::numeric_limits<double>::quiet_NaN();
    out.tli_ci_lower = out.tli_ci_upper =
        std::numeric_limits<double>::quiet_NaN();
    out.warnings.emplace_back(
        "mixed_ordinal_cfi_tli_misspec_inference: baseline noncentrality is "
        "non-positive (independence model fits); CFI set to 1, TLI/intervals "
        "undefined");
    return out;
  }

  const double r = out.delta_user / db;
  out.var_cfi = std::max(0.0,
      (out.var_user - 2.0 * r * out.cov_user_baseline +
       r * r * out.var_baseline) / (db * db));
  out.cfi = std::clamp(1.0 - r, 0.0, 1.0);
  const double sd_cfi = std::sqrt(out.var_cfi);
  out.cfi_ci_lower = std::clamp(1.0 - r - z * sd_cfi, 0.0, 1.0);
  out.cfi_ci_upper = std::clamp(1.0 - r + z * sd_cfi, 0.0, 1.0);
  if (out.delta_user <= 0.0) {
    out.warnings.emplace_back(
        "mixed_ordinal_cfi_tli_misspec_inference: user noncentrality is "
        "non-positive (near-exact fit); CFI on the boundary, normal interval "
        "approximate");
  }

  if (biasU > 0.0) {
    const double c = biasB / biasU;
    out.tli = 1.0 - c * r;
    out.var_tli = c * c * out.var_cfi;
    const double sd_tli = std::sqrt(out.var_tli);
    out.tli_ci_lower = out.tli - z * sd_tli;
    out.tli_ci_upper = out.tli + z * sd_tli;
  } else {
    out.tli = std::numeric_limits<double>::quiet_NaN();
    out.var_tli = std::numeric_limits<double>::quiet_NaN();
    out.tli_ci_lower = out.tli_ci_upper =
        std::numeric_limits<double>::quiet_NaN();
    out.warnings.emplace_back(
        "mixed_ordinal_cfi_tli_misspec_inference: user generalized df is "
        "non-positive; TLI undefined");
  }
  return out;
}

post_expected<OrdinalMisspecFitMeasures>
ordinal_fit_measures_misspec_inference(spec::LatentStructure pt,
                                       const model::MatrixRep& rep,
                                       const data::OrdinalStats& stats,
                                       const Estimates& est,
                                       OrdinalParameterization parameterization,
                                       bool estimated_weight,
                                       double conf_level,
                                       double eig_tol) {
  OrdinalMisspecFitMeasures out;
  out.conf_level = conf_level;
  out.fixed_weight = !estimated_weight;

  auto rm = ordinal_rmsea_misspec_inference(pt, rep, stats, est, parameterization,
                                            estimated_weight, conf_level, eig_tol);
  if (!rm.has_value()) return std::unexpected(rm.error());
  out.rmsea = rm->point;
  out.rmsea_ci_lower = rm->ci_lower;
  out.rmsea_ci_upper = rm->ci_upper;
  out.rmsea_pvalue = rm->exact_fit_pvalue;

  auto cr = ordinal_crmr_misspec_inference(pt, rep, stats, est, parameterization,
                                           estimated_weight,
                                           /*srmr_denominator=*/false, conf_level,
                                           eig_tol);
  if (!cr.has_value()) return std::unexpected(cr.error());
  out.crmr = cr->point;
  out.crmr_ci_lower = cr->ci_lower;
  out.crmr_ci_upper = cr->ci_upper;
  out.crmr_pvalue = cr->exact_fit_pvalue;
  // SRMR shares the statistic; only the denominator differs (vech vs off-diag),
  // so every reported quantity rescales by sqrt(ncorr/vech) = sqrt(k/(k+p)).
  const double pdim = static_cast<double>(stats.R[0].rows());
  const double kc = static_cast<double>(cr->k);
  const double srmr_scale = (kc + pdim) > 0.0 ? std::sqrt(kc / (kc + pdim)) : 0.0;
  out.srmr = cr->point * srmr_scale;
  out.srmr_ci_lower = cr->ci_lower * srmr_scale;
  out.srmr_ci_upper = cr->ci_upper * srmr_scale;

  auto ct = ordinal_cfi_tli_misspec_inference(pt, rep, stats, est, parameterization,
                                              estimated_weight, conf_level, eig_tol);
  if (!ct.has_value()) return std::unexpected(ct.error());
  out.cfi = ct->cfi;
  out.cfi_ci_lower = ct->cfi_ci_lower;
  out.cfi_ci_upper = ct->cfi_ci_upper;
  out.tli = ct->tli;
  out.tli_ci_lower = ct->tli_ci_lower;
  out.tli_ci_upper = ct->tli_ci_upper;
  out.stat_user = ct->stat_user;
  out.stat_baseline = ct->stat_baseline;
  out.df_user = ct->df_user;
  out.df_baseline = ct->df_baseline;

  // Bundle the per-index warnings, de-duplicated.
  auto add_warnings = [&out](const std::vector<std::string>& ws) {
    for (const auto& w : ws)
      if (std::find(out.warnings.begin(), out.warnings.end(), w) ==
          out.warnings.end())
        out.warnings.push_back(w);
  };
  add_warnings(rm->warnings);
  add_warnings(cr->warnings);
  add_warnings(ct->warnings);
  return out;
}

post_expected<OrdinalMisspecFitMeasures>
mixed_ordinal_fit_measures_misspec_inference(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalParameterization parameterization,
    bool estimated_weight,
    double conf_level,
    double eig_tol) {
  OrdinalMisspecFitMeasures out;
  out.conf_level = conf_level;
  out.fixed_weight = !estimated_weight;

  auto rm = mixed_ordinal_rmsea_misspec_inference(
      pt, rep, stats, est, parameterization, estimated_weight, conf_level,
      eig_tol);
  if (!rm.has_value()) return std::unexpected(rm.error());
  out.rmsea = rm->point;
  out.rmsea_ci_lower = rm->ci_lower;
  out.rmsea_ci_upper = rm->ci_upper;
  out.rmsea_pvalue = rm->exact_fit_pvalue;

  auto cr = mixed_ordinal_crmr_misspec_inference(
      pt, rep, stats, est, parameterization, estimated_weight,
      /*srmr_denominator=*/false, conf_level, eig_tol);
  if (!cr.has_value()) return std::unexpected(cr.error());
  out.crmr = cr->point;
  out.crmr_ci_lower = cr->ci_lower;
  out.crmr_ci_upper = cr->ci_upper;
  out.crmr_pvalue = cr->exact_fit_pvalue;

  auto sr = mixed_ordinal_crmr_misspec_inference(
      pt, rep, stats, est, parameterization, estimated_weight,
      /*srmr_denominator=*/true, conf_level, eig_tol);
  if (!sr.has_value()) return std::unexpected(sr.error());
  out.srmr = sr->point;
  out.srmr_ci_lower = sr->ci_lower;
  out.srmr_ci_upper = sr->ci_upper;

  auto ct = mixed_ordinal_cfi_tli_misspec_inference(
      pt, rep, stats, est, parameterization, estimated_weight, conf_level,
      eig_tol);
  if (!ct.has_value()) return std::unexpected(ct.error());
  out.cfi = ct->cfi;
  out.cfi_ci_lower = ct->cfi_ci_lower;
  out.cfi_ci_upper = ct->cfi_ci_upper;
  out.tli = ct->tli;
  out.tli_ci_lower = ct->tli_ci_lower;
  out.tli_ci_upper = ct->tli_ci_upper;
  out.stat_user = ct->stat_user;
  out.stat_baseline = ct->stat_baseline;
  out.df_user = ct->df_user;
  out.df_baseline = ct->df_baseline;

  auto add_warnings = [&out](const std::vector<std::string>& ws) {
    for (const auto& w : ws)
      if (std::find(out.warnings.begin(), out.warnings.end(), w) ==
          out.warnings.end())
        out.warnings.push_back(w);
  };
  add_warnings(rm->warnings);
  add_warnings(cr->warnings);
  add_warnings(sr->warnings);
  add_warnings(ct->warnings);
  return out;
}

post_expected<WeightedProfileRMSEAResult>
mixed_ordinal_dwls_profile_rmsea(spec::LatentStructure pt,
                                 const model::MatrixRep& rep,
                                 const data::MixedOrdinalStats& stats,
                                 const Estimates& est,
                                 OrdinalParameterization parameterization,
                                 double eig_tol) {
  if (auto v = validate_stats(stats, rep, OrdinalWeightKind::DWLS);
      !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size() ||
      stats.moment_influence.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_dwls_profile_rmsea: per-case influence functions are "
        "required"));
  }
  const bool has_diag_gamma_if =
      stats.gamma_diag_influence.size() == stats.R.size();
  if (!has_diag_gamma_if && stats.raw_data.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_dwls_profile_rmsea: estimated-weight influence "
        "unavailable; recompute mixed ordinal stats with gamma_diag_influence "
        "or raw_data"));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_dwls_profile_rmsea: fitted theta length does not match "
        "mixed delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) {
    return std::unexpected(fit_to_post(layout_or.error()));
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_dwls_profile_rmsea: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_dwls_profile_rmsea: fitted evaluation failed: " +
            eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu, est.theta,
                            parameterization);

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_dwls_profile_rmsea: constraint reparameterization has "
        "incompatible shape"));
  }

  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double N_total = static_cast<double>(*N_or);

  auto ob = mixed_observed_bread_analytic(
      pt, rep, stats, est, *layout_or, stats.W_dwls, K, parameterization);
  if (!ob.has_value()) return std::unexpected(ob.error());
  const Eigen::MatrixXd B = 0.5 * (*ob + ob->transpose()).eval();

  std::vector<WeightedEstimatedWeightProfileBlock> blocks;
  blocks.reserve(stats.R.size());
  double weighted_F = 0.0;  // sum_b n_b * d_b' W_b d_b
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const Eigen::MatrixXd& G = stats.moment_influence[b];
    if (G.rows() != stats.n_obs[b] || G.cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_dwls_profile_rmsea: moment_influence shape mismatch "
          "in block " + std::to_string(b)));
    }

    const Eigen::MatrixXd D_b = Delta_full.block(off, 0, mb, Delta_full.cols());
    const Eigen::VectorXd gamma_diag = stats.NACOV[b].diagonal();
    const Eigen::VectorXd d_b =
        mixed_model_moments(stats, *layout_or, eval->moments, est.theta, b,
                            parameterization) -
        stats.moments[b];
    weighted_F += static_cast<double>(stats.n_obs[b]) *
                  (d_b.transpose() * stats.W_dwls[b] * d_b).value();

    auto ifg_or = mixed_diag_gamma_influence_block(
        stats, b, has_diag_gamma_if, G, mb,
        "mixed_ordinal_dwls_profile_rmsea");
    if (!ifg_or.has_value()) return std::unexpected(ifg_or.error());

    Eigen::MatrixXd H(G.rows(), 2 * mb);
    H.leftCols(mb) = G;
    H.rightCols(mb) = *ifg_or;
    Eigen::MatrixXd gamma_x =
        (H.transpose() * H) / static_cast<double>(stats.n_obs[b]);
    gamma_x = 0.5 * (gamma_x + gamma_x.transpose()).eval();

    blocks.push_back(WeightedEstimatedWeightProfileBlock{
        .jacobian = D_b,
        .weight_diag = gamma_diag,
        .residual = d_b,
        .gamma = std::move(gamma_x),
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  const double fmin = weighted_F / N_total;
  return weighted_moment_profile_rmsea_estimated_weight(
      blocks, K, fmin, B, stats.R.size(), eig_tol);
}

post_expected<WeightedProfileLRTResult>
mixed_ordinal_dwls_profile_lrt(spec::LatentStructure pt_H1,
                               const model::MatrixRep& rep_H1,
                               const data::MixedOrdinalStats& stats,
                               const Estimates& est_H1,
                               spec::LatentStructure pt_H0,
                               const model::MatrixRep& rep_H0,
                               const Estimates& est_H0,
                               OrdinalParameterization parameterization,
                               double eig_tol) {
  auto h1 = mixed_ordinal_dwls_profile_rmsea(
      std::move(pt_H1), rep_H1, stats, est_H1, parameterization, eig_tol);
  if (!h1.has_value()) return std::unexpected(h1.error());
  auto h0 = mixed_ordinal_dwls_profile_rmsea(
      std::move(pt_H0), rep_H0, stats, est_H0, parameterization, eig_tol);
  if (!h0.has_value()) return std::unexpected(h0.error());
  return weighted_moment_profile_lrt(*h1, *h0, eig_tol);
}

post_expected<OrdinalFitMeasures>
fit_measures_mixed_ordinal(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           const Estimates& est,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fit_measures_mixed_ordinal: fitted theta length does not match "
        "mixed ordinal delta partable"));
  }

  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fit_measures_mixed_ordinal: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, false, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "fit_measures_mixed_ordinal: fitted evaluation failed: " +
            eval.error().detail));
  }

  auto baseline = mixed_ordinal_baseline_chi2(stats, weights);
  if (!baseline.has_value()) return std::unexpected(baseline.error());
  auto sr = mixed_ordinal_srmr(stats, *layout, eval->moments, est.theta,
                               parameterization);
  if (!sr.has_value()) return std::unexpected(sr.error());

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const int df = static_cast<int>(mixed_moment_rows(stats) - con_or->n_alpha);
  const double chi2 = 2.0 * static_cast<double>(*N_or) * est.fmin;
  const measures::FitMeasures indices =
      measures::fit_measures(chi2, df, *baseline, *N_or, stats.R.size());

  return OrdinalFitMeasures{*baseline, indices, *sr};
}

post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights,
                             OrdinalParameterization parameterization) {
  inference::ModificationIndexOptions options;
  return modification_indices_ordinal(std::move(pt), rep, stats, est, weights,
                                      options, parameterization);
}

post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights,
                             const inference::ModificationIndexOptions& options,
                             OrdinalParameterization parameterization) {
  auto residual_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             parameterization);
  };
  auto jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd&,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors,
                            theta, parameterization);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_modification_indices_impl(std::move(pt), rep, stats, est,
                                           weights, options, residual_fn,
                                           jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
score_tests_ordinal(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    const Estimates& est,
                    OrdinalWeightKind weights,
                    OrdinalParameterization parameterization) {
  auto residual_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             parameterization);
  };
  auto jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd&,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors,
                            theta, parameterization);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_score_tests_impl(std::move(pt), rep, stats, est, weights,
                                  residual_fn, jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::MixedOrdinalStats& stats,
                                   const Estimates& est,
                                   OrdinalWeightKind weights,
                                   OrdinalParameterization parameterization) {
  inference::ModificationIndexOptions options;
  return modification_indices_mixed_ordinal(std::move(pt), rep, stats, est,
                                            weights, options, parameterization);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options,
    OrdinalParameterization parameterization) {
  auto residual_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta,
                                   parameterization);
  };
  auto jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd& J_mu,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors,
                                  theta, parameterization);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_modification_indices_impl(std::move(pt), rep, stats, est,
                                           weights, options, residual_fn,
                                           jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
score_tests_mixed_ordinal(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          const Estimates& est,
                          OrdinalWeightKind weights,
                          OrdinalParameterization parameterization) {
  auto residual_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta,
                                   parameterization);
  };
  auto jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma,
                                        const Eigen::MatrixXd& J_mu,
                                        const std::vector<Eigen::MatrixXd>& factors,
                                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors,
                                  theta, parameterization);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_score_tests_impl(std::move(pt), rep, stats, est, weights,
                                  residual_fn, jacobian_fn, prepare_fn);
}

namespace frontier {

namespace {

struct OrdinalObjectiveState {
  data::OrdinalStats stats;
  ThresholdLayout layout;
  model::ModelEvaluator ev;
  std::vector<Eigen::MatrixXd> factors;
  OrdinalParameterization parameterization = OrdinalParameterization::Delta;
};

struct MixedOrdinalObjectiveState {
  data::MixedOrdinalStats stats;
  ThresholdLayout layout;
  model::ModelEvaluator ev;
  std::vector<Eigen::MatrixXd> factors;
  OrdinalParameterization parameterization = OrdinalParameterization::Delta;
};

}  // namespace

fit_expected<OrdinalLsObjective>
ordinal_ls_objective(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     const Estimates& at,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  if (at.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ordinal_ls_objective: theta length does not match ordinal delta "
        "partable"));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ordinal_ls_objective: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval0 = ev_or->evaluate(at.theta, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ordinal_ls_objective: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, *layout_or, eval0->moments, *factors_or,
                              at.theta, parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  auto state = std::make_shared<OrdinalObjectiveState>(OrdinalObjectiveState{
      stats, std::move(*layout_or), std::move(*ev_or), std::move(*factors_or),
      parameterization});

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = at.theta.size();
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = state->ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ordinal_ls_objective: evaluate failed: " + eval.error().detail));
    }
    return ordinal_residuals(state->stats, state->layout, eval->moments,
                             state->factors, x, state->parameterization);
  };
  prob.J = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ordinal_ls_objective: evaluate failed: " + eval.error().detail));
    }
    return ordinal_jacobian(state->stats, state->layout, eval->moments,
                            eval->J_sigma, state->factors, x,
                            state->parameterization, eval->J_mu);
  };
  prob.eval =
      [state](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ordinal_ls_objective: evaluate failed: " + eval.error().detail));
    }
    auto r = ordinal_residuals(state->stats, state->layout, eval->moments,
                               state->factors, x, state->parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = ordinal_jacobian(state->stats, state->layout, eval->moments,
                              eval->J_sigma, state->factors, x,
                              state->parameterization, eval->J_mu);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  return OrdinalLsObjective{std::move(pt), std::move(prob)};
}

fit_expected<OrdinalLsObjective>
mixed_ordinal_ls_objective(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           const Estimates& at,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  if (at.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "mixed_ordinal_ls_objective: theta length does not match mixed delta "
        "partable"));
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "mixed_ordinal_ls_objective: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto eval0 = ev_or->evaluate(at.theta, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "mixed_ordinal_ls_objective: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = mixed_ordinal_residuals(stats, *layout_or, eval0->moments,
                                    *factors_or, at.theta, parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  auto state =
      std::make_shared<MixedOrdinalObjectiveState>(MixedOrdinalObjectiveState{
          stats, std::move(*layout_or), std::move(*ev_or),
          std::move(*factors_or), parameterization});

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = at.theta.size();
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = state->ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "mixed_ordinal_ls_objective: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_residuals(state->stats, state->layout, eval->moments,
                                   state->factors, x,
                                   state->parameterization);
  };
  prob.J = [state](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "mixed_ordinal_ls_objective: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_jacobian(state->stats, state->layout, eval->moments,
                                  eval->J_sigma, eval->J_mu, state->factors,
                                  x, state->parameterization);
  };
  prob.eval =
      [state](const Eigen::VectorXd& x) -> fit_expected<optim::LsEvaluation> {
    auto eval = state->ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "mixed_ordinal_ls_objective: evaluate failed: " +
              eval.error().detail));
    }
    auto r = mixed_ordinal_residuals(state->stats, state->layout,
                                     eval->moments, state->factors, x,
                                     state->parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = mixed_ordinal_jacobian(state->stats, state->layout, eval->moments,
                                    eval->J_sigma, eval->J_mu,
                                    state->factors, x,
                                    state->parameterization);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  return OrdinalLsObjective{std::move(pt), std::move(prob)};
}

post_expected<std::vector<Eigen::MatrixXd>>
ordinal_stage2_weight_blocks(const data::OrdinalStats& stats,
                             OrdinalStage2Weight kind,
                             OrdinalStage2DlsOptions dls) {
  if (kind == OrdinalStage2Weight::Dls &&
      !(dls.a >= 0.0 && dls.a <= 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_stage2_weight: DLS mixing scalar a must lie in [0, 1]"));
  }
  const std::size_t nb = stats.R.size();
  if (stats.thresholds.size() != nb || stats.NACOV.size() != nb) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal_stage2_weight: incomplete OrdinalStats blocks"));
  }

  std::vector<Eigen::MatrixXd> out;
  out.reserve(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mdim = stats.thresholds[b].size() + p * (p - 1) / 2;
    if (stats.R[b].cols() != p || stats.NACOV[b].rows() != mdim ||
        stats.NACOV[b].cols() != mdim || !matrix_all_finite(stats.R[b]) ||
        !matrix_all_finite(stats.NACOV[b])) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal_stage2_weight: malformed block " + std::to_string(b)));
    }

    switch (kind) {
      case OrdinalStage2Weight::Uls:
        out.push_back(Eigen::MatrixXd::Identity(mdim, mdim));
        break;
      case OrdinalStage2Weight::Dwls: {
        if (stats.W_dwls.size() == nb && stats.W_dwls[b].rows() == mdim &&
            stats.W_dwls[b].cols() == mdim && matrix_all_finite(stats.W_dwls[b])) {
          out.push_back(stats.W_dwls[b]);
        } else {
          auto W_or = dwls_weight_from_gamma(
              stats.NACOV[b],
              "ordinal_stage2_weight: observed Gamma block " +
                  std::to_string(b));
          if (!W_or.has_value()) return std::unexpected(W_or.error());
          out.push_back(std::move(*W_or));
        }
        break;
      }
      case OrdinalStage2Weight::Wls: {
        if (stats.W_wls.size() == nb && stats.W_wls[b].rows() == mdim &&
            stats.W_wls[b].cols() == mdim && matrix_all_finite(stats.W_wls[b])) {
          out.push_back(stats.W_wls[b]);
        } else {
          auto W_or = sym_inverse_pd_post(
              stats.NACOV[b],
              "ordinal_stage2_weight: observed Gamma block " +
                  std::to_string(b));
          if (!W_or.has_value()) return std::unexpected(W_or.error());
          out.push_back(std::move(*W_or));
        }
        break;
      }
      case OrdinalStage2Weight::Nt: {
        auto Gnt_or = ordinal_nt_gamma_block(stats, b);
        if (!Gnt_or.has_value()) return std::unexpected(Gnt_or.error());
        auto W_or = sym_inverse_pd_post(
            *Gnt_or, "ordinal_stage2_weight: NT Gamma block " +
                         std::to_string(b));
        if (!W_or.has_value()) return std::unexpected(W_or.error());
        out.push_back(std::move(*W_or));
        break;
      }
      case OrdinalStage2Weight::Dls: {
        auto Gnt_or = ordinal_nt_gamma_block(stats, b);
        if (!Gnt_or.has_value()) return std::unexpected(Gnt_or.error());
        Eigen::MatrixXd Gmix =
            (1.0 - dls.a) * (*Gnt_or) + dls.a * stats.NACOV[b];
        Gmix = 0.5 * (Gmix + Gmix.transpose()).eval();
        auto W_or = sym_inverse_pd_post(
            Gmix, "ordinal_stage2_weight: DLS Gamma block " +
                      std::to_string(b));
        if (!W_or.has_value()) return std::unexpected(W_or.error());
        out.push_back(std::move(*W_or));
        break;
      }
    }
  }
  return out;
}

post_expected<data::OrdinalStats>
ordinal_stats_with_stage2_weight(const data::OrdinalStats& stats,
                                 OrdinalStage2Weight kind,
                                 OrdinalStage2DlsOptions dls) {
  auto W_or = ordinal_stage2_weight_blocks(stats, kind, dls);
  if (!W_or.has_value()) return std::unexpected(W_or.error());
  data::OrdinalStats out = stats;
  out.W_wls = std::move(*W_or);
  return out;
}

namespace {

// The (residual, jacobian, moment-jacobian, prepare) lambdas for the robust
// sweeps, binding the parameterization exactly like the non-robust wrappers
// above. The moment-jacobian lambda produces the UNWHITENED Δ the sandwich
// blocks contract against. Returned as a struct of lambdas so the MI and
// score-test wrappers share one definition per stats type.
auto ordinal_robust_handles(OrdinalParameterization parameterization) {
  auto residual_fn = [parameterization](const data::OrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const std::vector<Eigen::MatrixXd>& factors,
                                     const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             parameterization);
  };
  auto jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const Eigen::MatrixXd& J_sigma,
                                     const Eigen::MatrixXd&,
                                     const std::vector<Eigen::MatrixXd>& factors,
                                     const Eigen::VectorXd& theta) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors, theta,
                            parameterization);
  };
  auto moment_jacobian_fn = [parameterization](const data::OrdinalStats& s,
                                            const ThresholdLayout& layout,
                                            const model::ImpliedMoments& moments,
                                            const Eigen::MatrixXd& J_sigma,
                                            const Eigen::MatrixXd&,
                                            const Eigen::VectorXd& theta) {
    return ordinal_moment_jacobian(s, layout, moments, J_sigma, theta,
                                   parameterization);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_delta_partable(p, s, nullptr);
  };
  struct Handles {
    decltype(residual_fn) residual;
    decltype(jacobian_fn) jacobian;
    decltype(moment_jacobian_fn) moment_jacobian;
    decltype(prepare_fn) prepare;
  };
  return Handles{std::move(residual_fn), std::move(jacobian_fn),
                 std::move(moment_jacobian_fn), std::move(prepare_fn)};
}

auto mixed_ordinal_robust_handles(OrdinalParameterization parameterization) {
  auto residual_fn = [parameterization](const data::MixedOrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const std::vector<Eigen::MatrixXd>& factors,
                                     const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta,
                                   parameterization);
  };
  auto jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                     const ThresholdLayout& layout,
                                     const model::ImpliedMoments& moments,
                                     const Eigen::MatrixXd& J_sigma,
                                     const Eigen::MatrixXd& J_mu,
                                     const std::vector<Eigen::MatrixXd>& factors,
                                     const Eigen::VectorXd& theta) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors,
                                  theta, parameterization);
  };
  auto moment_jacobian_fn = [parameterization](const data::MixedOrdinalStats& s,
                                            const ThresholdLayout& layout,
                                            const model::ImpliedMoments& moments,
                                            const Eigen::MatrixXd& J_sigma,
                                            const Eigen::MatrixXd& J_mu,
                                            const Eigen::VectorXd& theta) {
    return mixed_moment_jacobian(s, layout, moments, J_sigma, J_mu, theta,
                                 parameterization);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr);
  };
  struct Handles {
    decltype(residual_fn) residual;
    decltype(jacobian_fn) jacobian;
    decltype(moment_jacobian_fn) moment_jacobian;
    decltype(prepare_fn) prepare;
  };
  return Handles{std::move(residual_fn), std::move(jacobian_fn),
                 std::move(moment_jacobian_fn), std::move(prepare_fn)};
}

}  // namespace

post_expected<inference::ScoreTestTable>
modification_indices_ordinal_robust(spec::LatentStructure pt,
                                    const model::MatrixRep& rep,
                                    const data::OrdinalStats& stats,
                                    const Estimates& est,
                                    OrdinalWeightKind weights,
                                    const inference::ModificationIndexOptions&
                                        options,
                                    OrdinalParameterization parameterization,
                                    bool estimated_weight) {
  auto h = ordinal_robust_handles(parameterization);
  return ordinal_modification_indices_robust_impl(
      std::move(pt), rep, stats, est, weights, options, parameterization,
      estimated_weight, h.residual, h.jacobian, h.moment_jacobian, h.prepare);
}

post_expected<inference::ScoreTestTable>
score_tests_ordinal_robust(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::OrdinalStats& stats,
                           const Estimates& est,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization,
                           bool estimated_weight) {
  auto h = ordinal_robust_handles(parameterization);
  return ordinal_score_tests_robust_impl(
      std::move(pt), rep, stats, est, weights, parameterization,
      estimated_weight, h.residual, h.jacobian, h.moment_jacobian, h.prepare);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal_robust(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options,
    OrdinalParameterization parameterization,
    bool estimated_weight) {
  auto h = mixed_ordinal_robust_handles(parameterization);
  return ordinal_modification_indices_robust_impl(
      std::move(pt), rep, stats, est, weights, options, parameterization,
      estimated_weight, h.residual, h.jacobian, h.moment_jacobian, h.prepare);
}

post_expected<inference::ScoreTestTable>
score_tests_mixed_ordinal_robust(spec::LatentStructure pt,
                                 const model::MatrixRep& rep,
                                 const data::MixedOrdinalStats& stats,
                                 const Estimates& est,
                                 OrdinalWeightKind weights,
                                 OrdinalParameterization parameterization,
                                 bool estimated_weight) {
  auto h = mixed_ordinal_robust_handles(parameterization);
  return ordinal_score_tests_robust_impl(
      std::move(pt), rep, stats, est, weights, parameterization,
      estimated_weight, h.residual, h.jacobian, h.moment_jacobian, h.prepare);
}

}  // namespace frontier

namespace {

// Dispatch a bounded least-squares problem (residual + Jacobian closures) to
// the chosen backend. The problem is driven directly — any equality
// reparameterization has already been folded into the closures by the caller.
fit_expected<optim::OptimResult>
run_ordinal_ls(const optim::GmmProblem& prob, const Eigen::VectorXd& x0,
               const Bounds& bounds, Backend backend, OptimOptions opts) {
  if (backend == Backend::Ceres) {
#ifdef MAGMAAN_WITH_CERES
    optim::CeresOptions copts;
    copts.max_iter = opts.max_iter;
    copts.ftol     = opts.ftol;
    copts.gtol     = opts.gtol;
    return optim::ceres_lm(prob, x0, bounds, copts);
#else
    (void)opts;
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: Ceres backend requested but MAGMAAN_WITH_CERES "
        "is off"));
#endif
  }
  if (backend == Backend::PortNls) {
#ifdef MAGMAAN_WITH_PORT
    // NL2SOL sees the multi-residual structure directly, matching the Ceres
    // LM dispatch above; no scalarisation.
    return optim::port_nls(prob, x0, bounds, opts);
#else
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: PortNls backend requested but MAGMAAN_WITH_PORT "
        "is off"));
#endif
  }
  if (backend == Backend::Ipopt) {
#ifdef MAGMAAN_WITH_IPOPT
    return optim::ipopt(optim::scalarize(prob), x0, bounds, opts);
#else
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: IPOPT backend requested but MAGMAAN_WITH_IPOPT "
        "is off"));
#endif
  }
  return optim::nlopt_lbfgs(optim::scalarize(prob), x0, bounds, opts);
}

// Solve a data-only ordinal LS problem `prob` (residual size `prob.n_resid`,
// `prob.expand` the identity in full θ) under linear equality constraints.
//
// Equality constraints are eliminated by the affine reparameterization
// θ = θ₀ + K·α (same as the ML / GMM path in `fit.cpp`), *not* by a quadratic
// penalty: a large penalty makes the constrained directions O(μ)-stiff, so the
// optimizer stalls on function-value stagnation at an FP-path-dependent iterate
// well short of the true optimum. The reduced α-problem carries no such
// conditioning and converges on the gradient stop.
fit_expected<Estimates>
solve_ordinal_ls(const optim::GmmProblem& prob, const Eigen::VectorXd& x0,
                 const Bounds& bounds, const EqConstraints& con,
                 Backend backend, OptimOptions opts, const char* who) {
  if (!con.active()) {
    auto out = run_ordinal_ls(prob, x0, bounds, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    return Estimates{prob.expand(out->x), out->fmin, out->iterations};
  }

  const optim::GmmProblem prob_a = optim::reparameterize(prob, con);
  if (con.n_alpha == 0) {  // every parameter pinned by the linear system
    auto r = prob_a.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    return Estimates{prob_a.expand(Eigen::VectorXd(0)), 0.5 * r->squaredNorm(), 0};
  }

  Eigen::VectorXd alpha0 = con.contract(x0);
  Bounds abounds;
  const bool pure_merge = !con.group.empty();
  if (!bounds.empty() && pure_merge) {
    abounds = optim::fold_alpha_bounds(con, bounds);
    alpha0  = alpha0.cwiseMax(abounds.lower).cwiseMin(abounds.upper);
  }
  auto out = run_ordinal_ls(prob_a, alpha0, abounds, backend, opts);
  if (!out.has_value()) return std::unexpected(out.error());
  Eigen::VectorXd theta_hat = prob_a.expand(out->x);
  if (!bounds.empty() && !pure_merge) {
    // General-linear α was optimized unbounded — verify θ̂ honors the box.
    constexpr double tol = 1e-6;
    for (Eigen::Index k = 0; k < theta_hat.size(); ++k) {
      if (theta_hat(k) < bounds.lower(k) - tol ||
          theta_hat(k) > bounds.upper(k) + tol) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            std::string(who) +
                ": general-linear equality drove parameter " +
                std::to_string(k) + " past its bound"));
      }
    }
  }
  return Estimates{std::move(theta_hat), out->fmin, out->iterations};
}

}  // namespace

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    const Eigen::VectorXd& x0,
                    Backend backend,
                    OptimOptions opts,
                    OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: bounds size mismatch"));
  }

  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  const EqConstraints& con = *con_or;

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                              parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = x0.size();
  prob.expand  = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return ordinal_residuals(stats, layout, eval->moments, factors, x,
                             parameterization);
  };
  prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(x, true, true);  // J_mu: released-intercept gradient
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                            factors, x, parameterization, eval->J_mu);
  };

  return solve_ordinal_ls(prob, x0, bounds, con, backend, opts,
                          "fit_ordinal_bounded");
}

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalMoments& moments,
                    data::OrdinalGammaCache* gamma_cache,
                    Bounds bounds,
                    data::OrdinalWeightPlan plan,
                    const Eigen::VectorXd& x0,
                    Backend backend,
                    OptimOptions opts) {
  const OrdinalParameterization parameterization =
      to_estimate_parameterization(plan.parameterization);
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  data::OrdinalStats stats = stats_adapter(moments);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }
  if (parameterization == OrdinalParameterization::Theta) {
    if (bounds.empty()) {
      auto b_or = bounds_from_partable(pt);
      if (!b_or.has_value()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "fit_ordinal_bounded: bounds_from_partable failed: " +
                b_or.error().detail));
      }
      bounds = std::move(*b_or);
    }
    if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: bounds size mismatch"));
    }
    auto factors_or = full_weight_factors(moments, gamma_cache, plan);
    if (!factors_or.has_value()) return std::unexpected(factors_or.error());
    const auto& factors = *factors_or;

    auto con_or = build_eq_constraints(pt);
    if (!con_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "constraint: " + con_or.error().detail));
    }
    const EqConstraints& con = *con_or;

    auto eval0 = ev.evaluate(x0, false, false);
    if (!eval0.has_value()) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "fit_ordinal_bounded: start evaluation failed: " +
              eval0.error().detail));
    }
    auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                                parameterization);
    if (!r0.has_value()) return std::unexpected(r0.error());

    optim::GmmProblem prob;
    prob.n_resid = r0->size();
    prob.n_param = x0.size();
    prob.expand = [](const Eigen::VectorXd& x) { return x; };
    prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
      auto eval = ev.evaluate(x, false, false);
      if (!eval.has_value()) {
        return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
            "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
      }
      return ordinal_residuals(stats, layout, eval->moments, factors, x,
                               parameterization);
    };
    prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
      auto eval = ev.evaluate(x, true, true);  // J_mu: released-intercept gradient
      if (!eval.has_value()) {
        return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
            "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
      }
      return ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                              factors, x, parameterization, eval->J_mu);
    };

    return solve_ordinal_ls(prob, x0, bounds, con, backend, opts,
                            "fit_ordinal_bounded");
  }
  auto profile_or = make_threshold_design(pt, layout, stats, x0);
  if (!profile_or.has_value()) return std::unexpected(profile_or.error());
  ThresholdDesign profile = std::move(*profile_or);

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: bounds size mismatch"));
  }
  Bounds active_bounds = profile_bounds(profile, bounds);

  auto weights_or =
      profiled_weight_workspace(moments, layout, profile, gamma_cache, plan);
  if (!weights_or.has_value()) return std::unexpected(weights_or.error());
  const auto& profiled_weights = *weights_or;

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  if (con_or->active()) {
    auto constraint_check = ensure_no_unprofiled_equality_constraints(pt, profile);
    if (!constraint_check.has_value()) {
      return std::unexpected(constraint_check.error());
    }
  }

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = profiled_ordinal_residuals(moments, eval0->moments,
                                       profiled_weights);
  if (!r0.has_value()) return std::unexpected(r0.error());

  Eigen::VectorXd active_x0 = profile_contract(profile, x0);
  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = active_x0.size();
  prob.expand  = [profile](const Eigen::VectorXd& x) {
    return profile_expand(profile, x);
  };
  prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    const Eigen::VectorXd theta = profile_expand(profile, x);
    auto eval = ev.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return profiled_ordinal_residuals(moments, eval->moments,
                                      profiled_weights);
  };
  prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    const Eigen::VectorXd theta = profile_expand(profile, x);
    auto eval = ev.evaluate(theta, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto J_full = profiled_ordinal_jacobian(moments, eval->moments,
                                            eval->J_sigma, profiled_weights);
    if (!J_full.has_value()) return std::unexpected(J_full.error());
    return profile_jacobian(profile, *J_full);
  };

  auto est = solve_ordinal_ls(prob, active_x0, active_bounds, EqConstraints{},
                              backend, opts,
                              "fit_ordinal_bounded");
  if (!est.has_value()) return std::unexpected(est.error());
  auto eval_hat = ev.evaluate(est->theta, false, false);
  if (!eval_hat.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "fit_ordinal_bounded: final reconstruction evaluate failed: " +
            eval_hat.error().detail));
  }
  auto theta_or = reconstruct_profiled_thresholds(
      stats, layout, eval_hat->moments, profiled_weights, est->theta);
  if (!theta_or.has_value()) return std::unexpected(theta_or.error());
  est->theta = std::move(*theta_or);
  return est;
}

fit_expected<Estimates>
fit_ordinal_snlls(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalMoments& moments,
                  data::OrdinalGammaCache* gamma_cache,
                  data::OrdinalWeightPlan plan,
                  const Eigen::VectorXd& x0,
                  Backend backend,
                  OptimOptions opts) {
  const OrdinalParameterization parameterization =
      to_estimate_parameterization(plan.parameterization);
  if (parameterization == OrdinalParameterization::Theta) {
    return fit_ordinal_snlls_full_thresholds(
        std::move(pt), rep, moments, gamma_cache, plan, x0, backend, opts);
  }
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }

  data::OrdinalStats stats = stats_adapter(moments);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  auto threshold_fixed_or = fix_thresholds_for_snlls(pt, layout, stats, x0);
  if (!threshold_fixed_or.has_value()) {
    return std::unexpected(threshold_fixed_or.error());
  }
  ThresholdFixedProfile threshold_fixed = std::move(*threshold_fixed_or);

  auto profiled_weights_or = profiled_weight_workspace(
      moments, layout, threshold_fixed.design, gamma_cache, plan);
  if (!profiled_weights_or.has_value()) {
    return std::unexpected(profiled_weights_or.error());
  }
  const ProfiledWeightWorkspace& profiled_weights = *profiled_weights_or;

  auto ev_reduced_or = model::ModelEvaluator::build(threshold_fixed.pt, rep);
  if (!ev_reduced_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: ModelEvaluator::build failed: " +
            ev_reduced_or.error().detail));
  }
  auto ev_full_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_full_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: full ModelEvaluator::build failed: " +
            ev_full_or.error().detail));
  }
  auto ev_reduced = std::move(*ev_reduced_or);
  auto ev_full = std::move(*ev_full_or);

  auto eval0 = ev_reduced.evaluate(threshold_fixed.x0, true, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = profiled_ordinal_residuals(moments, eval0->moments,
                                       profiled_weights);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem base;
  base.n_resid = r0->size();
  base.n_param = threshold_fixed.x0.size();
  base.expand = [](const Eigen::VectorXd& x) { return x; };
  base.r = [&](const Eigen::VectorXd& theta)
      -> fit_expected<Eigen::VectorXd> {
    auto eval = ev_reduced.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls: evaluate failed: " + eval.error().detail));
    }
    return profiled_ordinal_residuals(moments, eval->moments,
                                      profiled_weights);
  };
  base.J = [&](const Eigen::VectorXd& theta)
      -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev_reduced.evaluate(theta, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls: evaluate failed: " + eval.error().detail));
    }
    return profiled_ordinal_jacobian(moments, eval->moments, eval->J_sigma,
                                     profiled_weights);
  };
  base.eval = [&](const Eigen::VectorXd& theta)
      -> fit_expected<optim::LsEvaluation> {
    auto eval = ev_reduced.evaluate(theta, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls: evaluate failed: " + eval.error().detail));
    }
    auto r = profiled_ordinal_residuals(moments, eval->moments,
                                        profiled_weights);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = profiled_ordinal_jacobian(moments, eval->moments, eval->J_sigma,
                                       profiled_weights);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto gp_or = gmm::gp(base, threshold_fixed.pt, ev_reduced,
                      threshold_fixed.x0);
  if (!gp_or.has_value()) return std::unexpected(gp_or.error());
  const optim::GmmProblem& prob = gp_or->problem;

  Eigen::VectorXd theta_reduced;
  double fmin = 0.0;
  int iterations = 0;
  int f_evals = 0;
  int g_evals = 0;
  optim::OptimStatus status = optim::OptimStatus::Converged;
  double grad_inf_norm = -1.0;
  optim::TerminalAudit audit;

  if (prob.n_param == 0) {
    auto r = prob.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    theta_reduced = prob.expand(Eigen::VectorXd(0));
    fmin = 0.5 * r->squaredNorm();
  } else {
    auto out = run_ordinal_ls(prob, gp_or->beta0, Bounds{}, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    theta_reduced = prob.expand(out->x);
    fmin = out->fmin;
    iterations = out->iterations;
    f_evals = out->f_evals;
    g_evals = out->g_evals;
    status = out->status;
    grad_inf_norm = out->grad_inf_norm;
    audit = std::move(out->audit);
  }

  auto theta_full_or =
      expand_threshold_fixed_theta(threshold_fixed, theta_reduced);
  if (!theta_full_or.has_value()) return std::unexpected(theta_full_or.error());
  auto eval_hat = ev_full.evaluate(*theta_full_or, false, false);
  if (!eval_hat.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "fit_ordinal_snlls: final reconstruction evaluate failed: " +
            eval_hat.error().detail));
  }
  auto theta_final_or = reconstruct_profiled_thresholds(
      stats, layout, eval_hat->moments, profiled_weights, *theta_full_or);
  if (!theta_final_or.has_value()) {
    return std::unexpected(theta_final_or.error());
  }
  return Estimates{std::move(*theta_final_or), fmin, iterations, f_evals,
                   g_evals, status, grad_inf_norm, std::move(audit), {},
                   gp_or->n_nonlinear, gp_or->n_linear};
}

fit_expected<Estimates>
fit_ordinal_snlls_full_thresholds(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const data::OrdinalMoments& moments,
                                  data::OrdinalGammaCache* gamma_cache,
                                  data::OrdinalWeightPlan plan,
                                  const Eigen::VectorXd& x0,
                                  Backend backend,
                                  OptimOptions opts) {
  const OrdinalParameterization parameterization =
      to_estimate_parameterization(plan.parameterization);
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }

  data::OrdinalStats stats = stats_adapter(moments);
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls_full_thresholds: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls_full_thresholds: x0 size (" +
            std::to_string(x0.size()) + ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  auto factors_or = full_weight_factors(moments, gamma_cache, plan);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto block_kinds_or =
      ordinal_gp_block_kinds(pt, layout, ev, parameterization);
  if (!block_kinds_or.has_value()) return std::unexpected(block_kinds_or.error());
  const auto& block_kinds = *block_kinds_or;

  auto eval0 = ev.evaluate(x0, true, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_snlls_full_thresholds: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                              parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem base;
  base.n_resid = r0->size();
  base.n_param = x0.size();
  base.expand = [](const Eigen::VectorXd& x) { return x; };
  base.r = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return ordinal_residuals(stats, layout, eval->moments, factors, theta,
                             parameterization);
  };
  base.J = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(theta, true, true);  // J_mu: released-intercept gradient
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                            factors, theta, parameterization, eval->J_mu);
  };
  base.eval = [&](const Eigen::VectorXd& theta)
      -> fit_expected<optim::LsEvaluation> {
    auto eval = ev.evaluate(theta, true, true);  // J_mu: released-intercept gradient
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    auto r = ordinal_residuals(stats, layout, eval->moments, factors, theta,
                               parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J = ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                              factors, theta, parameterization, eval->J_mu);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto gp_or = gmm::gp(base, pt, ev, x0, block_kinds);
  if (!gp_or.has_value()) return std::unexpected(gp_or.error());
  const optim::GmmProblem& prob = gp_or->problem;

  Eigen::VectorXd theta_hat;
  double fmin = 0.0;
  int iterations = 0;
  int f_evals = 0;
  int g_evals = 0;
  optim::OptimStatus status = optim::OptimStatus::Converged;
  double grad_inf_norm = -1.0;
  optim::TerminalAudit audit;

  if (prob.n_param == 0) {
    auto r = prob.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    theta_hat = prob.expand(Eigen::VectorXd(0));
    fmin = 0.5 * r->squaredNorm();
  } else {
    auto out = run_ordinal_ls(prob, gp_or->beta0, Bounds{}, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    theta_hat = prob.expand(out->x);
    fmin = out->fmin;
    iterations = out->iterations;
    f_evals = out->f_evals;
    g_evals = out->g_evals;
    status = out->status;
    grad_inf_norm = out->grad_inf_norm;
    audit = std::move(out->audit);
  }

  return Estimates{std::move(theta_hat), fmin, iterations, f_evals, g_evals,
                   status, grad_inf_norm, std::move(audit), {},
                   gp_or->n_nonlinear, gp_or->n_linear};
}

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          const Eigen::VectorXd& x0,
                          Backend backend,
                          OptimOptions opts,
                          OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_mixed_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_mixed_ordinal_bounded: bounds size mismatch"));
  }

  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  const EqConstraints& con = *con_or;

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = mixed_ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                                    parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem prob;
  prob.n_resid = r0->size();
  prob.n_param = x0.size();
  prob.expand  = [](const Eigen::VectorXd& x) { return x; };
  prob.r = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return mixed_ordinal_residuals(stats, layout, eval->moments, factors, x,
                                   parameterization);
  };
  prob.J = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    return mixed_ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                                  eval->J_mu, factors, x, parameterization);
  };

  return solve_ordinal_ls(prob, x0, bounds, con, backend, opts,
                          "fit_mixed_ordinal_bounded");
}

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalMoments& moments,
                          data::OrdinalGammaCache* gamma_cache,
                          Bounds bounds,
                          data::OrdinalWeightPlan plan,
                          const Eigen::VectorXd& x0,
                          Backend backend,
                          OptimOptions opts) {
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto stats_or = mixed_stats_from_moments_cache(
      moments, gamma_cache, plan, "fit_mixed_ordinal_bounded");
  if (!stats_or.has_value()) return std::unexpected(stats_or.error());
  return fit_mixed_ordinal_bounded(
      std::move(pt), rep, stats_or->first, std::move(bounds),
      stats_or->second, x0, backend, opts,
      to_estimate_parameterization(plan.parameterization));
}

fit_expected<Estimates>
fit_mixed_ordinal_snlls_full_thresholds(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    OrdinalWeightKind weights,
    const Eigen::VectorXd& x0,
    Backend backend,
    OptimOptions opts,
    OrdinalParameterization parameterization) {
  // Theta is supported through the same full-threshold moment stack: the
  // standardized covariance moments make the non-threshold covariance block
  // nonlinear, so ordinal_gp_block_kinds marks only thresholds as
  // Golub-Pereyra linear coordinates under theta (the implementation answer
  // to the response-scale separability boundary).
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_snlls_full_thresholds: ModelEvaluator::build "
        "failed: " +
            ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_snlls_full_thresholds: x0 size (" +
            std::to_string(x0.size()) + ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto block_kinds_or =
      ordinal_gp_block_kinds(pt, layout, ev, parameterization);
  if (!block_kinds_or.has_value()) {
    return std::unexpected(block_kinds_or.error());
  }
  const auto& block_kinds = *block_kinds_or;

  auto eval0 = ev.evaluate(x0, true, true);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(
        FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_snlls_full_thresholds: start evaluation failed: " +
            eval0.error().detail));
  }
  auto r0 = mixed_ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                                    parameterization);
  if (!r0.has_value()) return std::unexpected(r0.error());

  optim::GmmProblem base;
  base.n_resid = r0->size();
  base.n_param = x0.size();
  base.expand = [](const Eigen::VectorXd& x) { return x; };
  base.r = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(
          FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_residuals(stats, layout, eval->moments, factors, theta,
                                   parameterization);
  };
  base.J = [&](const Eigen::VectorXd& theta) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(
          FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    return mixed_ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                                  eval->J_mu, factors, theta, parameterization);
  };
  base.eval =
      [&](const Eigen::VectorXd& theta) -> fit_expected<optim::LsEvaluation> {
    auto eval = ev.evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(
          FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_snlls_full_thresholds: evaluate failed: " +
              eval.error().detail));
    }
    auto r = mixed_ordinal_residuals(stats, layout, eval->moments, factors,
                                     theta, parameterization);
    if (!r.has_value()) return std::unexpected(r.error());
    auto J =
        mixed_ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                               eval->J_mu, factors, theta, parameterization);
    if (!J.has_value()) return std::unexpected(J.error());
    return optim::LsEvaluation{std::move(*r), std::move(*J)};
  };

  auto gp_or = gmm::gp(base, pt, ev, x0, block_kinds);
  if (!gp_or.has_value()) return std::unexpected(gp_or.error());
  const optim::GmmProblem& prob = gp_or->problem;

  Eigen::VectorXd theta_hat;
  double fmin = 0.0;
  int iterations = 0;
  int f_evals = 0;
  int g_evals = 0;
  optim::OptimStatus status = optim::OptimStatus::Converged;
  double grad_inf_norm = -1.0;
  optim::TerminalAudit audit;

  if (prob.n_param == 0) {
    auto r = prob.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    theta_hat = prob.expand(Eigen::VectorXd(0));
    fmin = 0.5 * r->squaredNorm();
  } else {
    auto out = run_ordinal_ls(prob, gp_or->beta0, Bounds{}, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    theta_hat = prob.expand(out->x);
    fmin = out->fmin;
    iterations = out->iterations;
    f_evals = out->f_evals;
    g_evals = out->g_evals;
    status = out->status;
    grad_inf_norm = out->grad_inf_norm;
    audit = std::move(out->audit);
  }

  return Estimates{std::move(theta_hat), fmin, iterations, f_evals, g_evals,
                   status, grad_inf_norm, std::move(audit), {},
                   gp_or->n_nonlinear, gp_or->n_linear};
}

fit_expected<Estimates>
fit_mixed_ordinal_snlls_full_thresholds(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalMoments& moments,
    data::OrdinalGammaCache* gamma_cache,
    data::OrdinalWeightPlan plan,
    const Eigen::VectorXd& x0,
    Backend backend,
    OptimOptions opts) {
  if (auto v = validate_moments(moments, rep); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto stats_or = mixed_stats_from_moments_cache(
      moments, gamma_cache, plan, "fit_mixed_ordinal_snlls_full_thresholds");
  if (!stats_or.has_value()) return std::unexpected(stats_or.error());
  return fit_mixed_ordinal_snlls_full_thresholds(
      std::move(pt), rep, stats_or->first, stats_or->second, x0, backend, opts,
      to_estimate_parameterization(plan.parameterization));
}

}  // namespace magmaan::estimate
