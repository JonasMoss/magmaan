#include "magmaan/data/ordinal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/data/pairwise_mixed.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/optim/nlopt_optimizer.hpp"

#include "detail_linalg.hpp"

namespace magmaan::data {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kProbFloor = 1.4901161193847656e-8;

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

double normal_cdf(double x) noexcept {
  if (x == kInf) return 1.0;
  if (x == -kInf) return 0.0;
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normal_pdf(double x) noexcept {
  if (!std::isfinite(x)) return 0.0;
  constexpr double inv_sqrt_2pi = 0.39894228040143267794;
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// Acklam's rational approximation. Accuracy is ample for threshold starts and
// ACOV scaling; inputs are clamped before this is called.
double normal_quantile(double p) noexcept {
  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 =  2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 =  1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 =  2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 =  1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 =  6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 =  4.374664141464968e+00;
  constexpr double c6 =  2.938163982698783e+00;
  constexpr double d1 =  7.784695709041462e-03;
  constexpr double d2 =  3.224671290700398e-01;
  constexpr double d3 =  2.445134137142996e+00;
  constexpr double d4 =  3.754408661907416e+00;
  constexpr double plow = 0.02425;
  constexpr double phigh = 1.0 - plow;

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
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

post_expected<Eigen::MatrixXd> symmetric_inverse_pd(const Eigen::MatrixXd& A,
                                                    std::string what) {
  detail::SymInverseResult r = detail::symmetric_inverse_pd_gated(A);
  if (r.ok) return std::move(r.inverse);
  if (!r.finite) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not a finite square matrix"));
  }
  if (!r.decomposed) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " eigendecomposition failed"));
  }
  return std::unexpected(make_err(PostError::Kind::NumericIssue,
      std::move(what) + " is not positive definite"));
}

Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>
observed_moment_overlap_counts_slow(
    const Eigen::MatrixXi& Xcat,
    const std::vector<std::int32_t>& support_i,
    const std::vector<std::int32_t>& support_j) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index mdim = static_cast<Eigen::Index>(support_i.size());
  Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> overlap(mdim, mdim);
  overlap.setZero();
  for (Eigen::Index a = 0; a < mdim; ++a) {
    for (Eigen::Index bb = a; bb < mdim; ++bb) {
      const int ai = support_i[static_cast<std::size_t>(a)];
      const int aj = support_j[static_cast<std::size_t>(a)];
      const int bi = support_i[static_cast<std::size_t>(bb)];
      const int bj = support_j[static_cast<std::size_t>(bb)];
      std::int64_t nab = 0;
      for (Eigen::Index r = 0; r < n; ++r) {
        bool ok = ai >= 0 && Xcat(r, ai) >= 0 && bi >= 0 && Xcat(r, bi) >= 0;
        if (ok && aj >= 0) ok = Xcat(r, aj) >= 0;
        if (ok && bj >= 0) ok = Xcat(r, bj) >= 0;
        if (ok) ++nab;
      }
      overlap(a, bb) = overlap(bb, a) = nab;
    }
  }
  return overlap;
}

Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>
observed_moment_overlap_counts(
    const Eigen::MatrixXi& Xcat,
    const std::vector<std::int32_t>& support_i,
    const std::vector<std::int32_t>& support_j) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  const Eigen::Index mdim = static_cast<Eigen::Index>(support_i.size());
  if (p > 64 || support_j.size() != support_i.size()) {
    return observed_moment_overlap_counts_slow(Xcat, support_i, support_j);
  }

  std::vector<std::uint64_t> support_masks(static_cast<std::size_t>(mdim), 0);
  for (Eigen::Index m = 0; m < mdim; ++m) {
    const int i = support_i[static_cast<std::size_t>(m)];
    const int j = support_j[static_cast<std::size_t>(m)];
    if (i < 0 || i >= p || j >= p) {
      return observed_moment_overlap_counts_slow(Xcat, support_i, support_j);
    }
    support_masks[static_cast<std::size_t>(m)] |= (std::uint64_t{1} << i);
    if (j >= 0) {
      support_masks[static_cast<std::size_t>(m)] |= (std::uint64_t{1} << j);
    }
  }

  std::unordered_map<std::uint64_t, std::int64_t> count_cache;
  count_cache.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(1, mdim)) *
                      4U);
  auto count_for_mask = [&](std::uint64_t mask) -> std::int64_t {
    if (auto it = count_cache.find(mask); it != count_cache.end()) {
      return it->second;
    }
    std::array<Eigen::Index, 64> cols{};
    Eigen::Index n_cols = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if ((mask & (std::uint64_t{1} << j)) != 0U) {
        cols[static_cast<std::size_t>(n_cols)] = j;
        ++n_cols;
      }
    }
    std::int64_t total = 0;
    for (Eigen::Index r = 0; r < n; ++r) {
      bool ok = true;
      for (Eigen::Index k = 0; k < n_cols; ++k) {
        if (Xcat(r, cols[static_cast<std::size_t>(k)]) < 0) {
          ok = false;
          break;
        }
      }
      if (ok) ++total;
    }
    count_cache.emplace(mask, total);
    return total;
  };

  Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> overlap(mdim, mdim);
  overlap.setZero();
  for (Eigen::Index a = 0; a < mdim; ++a) {
    for (Eigen::Index bb = a; bb < mdim; ++bb) {
      const std::uint64_t mask = support_masks[static_cast<std::size_t>(a)] |
                                 support_masks[static_cast<std::size_t>(bb)];
      const std::int64_t nab = count_for_mask(mask);
      overlap(a, bb) = overlap(bb, a) = nab;
    }
  }
  return overlap;
}

struct CorrelationRepairResult {
  Eigen::MatrixXd R;
  double raw_min_eigen = 0.0;
  double min_eigen = 0.0;
  bool   repaired = false;
  double ridge = 0.0;
  double shrinkage = 0.0;
};

enum class SharedRobustOrdinalKind {
  HWeighted,
  Dpd,
  HuberResidual,
};

struct SharedRobustOrdinalOptions {
  SharedRobustOrdinalKind kind = SharedRobustOrdinalKind::HWeighted;
  PolychoricHScoreOptions h_score;
  HuberResidualClipOptions clip;
  double alpha = 0.3;
  double rho_lower = -0.999;
  double rho_upper = 0.999;
  int max_iter = 90;
  double ftol = 1e-10;
  double gtol = 2e-5;
  double fd_step = 1e-5;
  double min_threshold_spacing = 1e-6;
  bool lavaan_adjust_2x2 = true;
  PairwiseOrdinalCorrelationRepairOptions correlation_repair;
};

double pearson_cell_residual(double f, double p, double total) noexcept {
  return std::sqrt(std::max(1.0, total)) * (f - p) /
         std::sqrt(std::max(kProbFloor, p));
}

double clipped_cell_frequency(double f,
                              double p,
                              double total,
                              const HuberResidualClipOptions& clip) {
  const double r = pearson_cell_residual(f, p, total);
  auto c = eval_huber_residual_clip(r, clip);
  if (!c.has_value()) return std::numeric_limits<double>::quiet_NaN();
  return p + c->psi * std::sqrt(std::max(kProbFloor, p) /
                                std::max(1.0, total));
}

double clipped_cell_derivative(double f,
                               double p,
                               double total,
                               const HuberResidualClipOptions& clip) {
  const double r = pearson_cell_residual(f, p, total);
  auto c = eval_huber_residual_clip(r, clip);
  return c.has_value() ? c->dpsi : std::numeric_limits<double>::quiet_NaN();
}

struct SharedOrdinalPairTable {
  Eigen::Index i = 0;
  Eigen::Index j = 0;
  Eigen::Index corr_index = 0;
  Eigen::MatrixXd counts;
  Eigen::MatrixXd adjusted_counts;
};

bool h_score_is_ml_like_local(const PolychoricHScoreOptions& h_options) noexcept {
  return h_options.kind == PolychoricHScoreKind::ML ||
         (h_options.kind == PolychoricHScoreKind::WmaHardCap &&
          h_options.k == std::numeric_limits<double>::infinity());
}

Eigen::MatrixXd adjusted_polychoric_table_local(const Eigen::MatrixXd& counts) {
  Eigen::MatrixXd out = counts;
  if (out.rows() == 2 && out.cols() == 2) {
    std::vector<Eigen::Index> zeros;
    for (Eigen::Index r = 0; r < 2; ++r) {
      for (Eigen::Index c = 0; c < 2; ++c) {
        if (out(r, c) == 0.0) zeros.push_back(r * 2 + c);
      }
    }
    if (zeros.size() == 1) {
      const Eigen::Index z = zeros[0];
      if (z == 0 || z == 3) {
        out(0, 0) += 0.5;
        out(1, 1) += 0.5;
        out(1, 0) -= 0.5;
        out(0, 1) -= 0.5;
      } else {
        out(0, 0) -= 0.5;
        out(1, 1) -= 0.5;
        out(1, 0) += 0.5;
        out(0, 1) += 0.5;
      }
    }
  }
  return out.cwiseMax(0.0);
}

double encode_rho_local(double rho, double lower, double upper) noexcept {
  const double mid = 0.5 * (lower + upper);
  const double half = 0.5 * (upper - lower);
  const double z = std::clamp((rho - mid) / half, -0.999999, 0.999999);
  return 0.5 * std::log((1.0 + z) / (1.0 - z));
}

double decode_rho_local(double z, double lower, double upper) noexcept {
  const double mid = 0.5 * (lower + upper);
  const double half = 0.5 * (upper - lower);
  return mid + half * std::tanh(z);
}

Eigen::VectorXd encode_shared_ordinal_params(
    const Eigen::VectorXd& thresholds,
    const Eigen::MatrixXd& R,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    double rho_lower,
    double rho_upper,
    double min_spacing) {
  const Eigen::Index p = R.rows();
  const Eigen::Index nth = thresholds.size();
  const Eigen::Index ncorr = p * (p - 1) / 2;
  Eigen::VectorXd z(nth + ncorr);
  for (Eigen::Index j = 0; j < p; ++j) {
    const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index len =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    if (len <= 0) continue;
    z(start) = thresholds(start);
    for (Eigen::Index k = 1; k < len; ++k) {
      z(start + k) = std::log(std::max(
          min_spacing, thresholds(start + k) - thresholds(start + k - 1) -
                           min_spacing));
    }
  }
  Eigen::Index cidx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      z(nth + cidx++) = encode_rho_local(R(i, j), rho_lower, rho_upper);
    }
  }
  return z;
}

void decode_shared_ordinal_params(
    const Eigen::VectorXd& z,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    double rho_lower,
    double rho_upper,
    double min_spacing,
    Eigen::VectorXd& thresholds,
    std::vector<Eigen::VectorXd>& th_by_var,
    Eigen::MatrixXd& R) {
  const Eigen::Index p = static_cast<Eigen::Index>(levels.size());
  const Eigen::Index nth = th_start.empty() ? 0 :
      th_start.back() + static_cast<Eigen::Index>(levels.back() - 1);
  thresholds = Eigen::VectorXd::Zero(nth);
  th_by_var.assign(static_cast<std::size_t>(p), Eigen::VectorXd{});
  for (Eigen::Index j = 0; j < p; ++j) {
    const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index len =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    th_by_var[static_cast<std::size_t>(j)].resize(len);
    if (len <= 0) continue;
    thresholds(start) = z(start);
    th_by_var[static_cast<std::size_t>(j)](0) = thresholds(start);
    for (Eigen::Index k = 1; k < len; ++k) {
      const double gap = min_spacing + std::exp(std::clamp(
          z(start + k), -30.0, 30.0));
      thresholds(start + k) = thresholds(start + k - 1) + gap;
      th_by_var[static_cast<std::size_t>(j)](k) = thresholds(start + k);
    }
  }

  R = Eigen::MatrixXd::Identity(p, p);
  Eigen::Index cidx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      R(i, j) = R(j, i) = decode_rho_local(
          z(nth + cidx++), rho_lower, rho_upper);
    }
  }
}

post_expected<CorrelationRepairResult>
repair_correlation_if_requested(
    const Eigen::MatrixXd& R,
    PairwiseOrdinalCorrelationRepairOptions options,
    std::string what) {
  if (R.rows() != R.cols() || !R.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not a finite square matrix"));
  }
  if (!std::isfinite(options.min_eigenvalue) ||
      options.min_eigenvalue < 0.0 ||
      options.min_eigenvalue >= 1.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " repair target must be finite and in [0, 1)"));
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      0.5 * (R + R.transpose()));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " eigendecomposition failed"));
  }

  CorrelationRepairResult out;
  out.R = 0.5 * (R + R.transpose());
  out.raw_min_eigen = es.eigenvalues().minCoeff();
  out.min_eigen = out.raw_min_eigen;

  if (out.raw_min_eigen >= options.min_eigenvalue ||
      options.kind == PairwiseOrdinalCorrelationRepairKind::None) {
    return out;
  }

  if (options.kind == PairwiseOrdinalCorrelationRepairKind::Error) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " is below the requested minimum eigenvalue"));
  }

  const double target = options.min_eigenvalue;
  const double denom = 1.0 - out.raw_min_eigen;
  if (!(denom > 0.0) || !std::isfinite(denom)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " cannot be repaired toward the identity"));
  }
  const double shrinkage = (target - out.raw_min_eigen) / denom;
  if (!(shrinkage > 0.0) || !(shrinkage < 1.0) ||
      !std::isfinite(shrinkage)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " produced invalid repair intensity"));
  }

  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(R.rows(), R.cols());
  if (options.kind == PairwiseOrdinalCorrelationRepairKind::Ridge) {
    const double ridge = shrinkage / (1.0 - shrinkage);
    if (!(ridge > 0.0) || !std::isfinite(ridge)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::move(what) + " produced invalid ridge"));
    }
    out.R = (out.R + ridge * I) / (1.0 + ridge);
    out.repaired = true;
    out.ridge = ridge;
  } else if (options.kind == PairwiseOrdinalCorrelationRepairKind::Shrinkage) {
    out.R = (1.0 - shrinkage) * out.R + shrinkage * I;
    out.repaired = true;
    out.shrinkage = shrinkage;
  } else {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " has an unknown repair kind"));
  }

  out.R.diagonal().setOnes();
  out.R = 0.5 * (out.R + out.R.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> repaired_es(out.R);
  if (repaired_es.info() != Eigen::Success ||
      !repaired_es.eigenvalues().allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " repaired eigendecomposition failed"));
  }
  out.min_eigen = repaired_es.eigenvalues().minCoeff();
  return out;
}

double corner_rect_value(const Eigen::MatrixXd& grid,
                         Eigen::Index a,
                         Eigen::Index b) noexcept {
  return grid(a + 1, b + 1) - grid(a, b + 1) - grid(a + 1, b) + grid(a, b);
}

Eigen::MatrixXd expected_pair_counts(double total,
                                     const Eigen::VectorXd& th_i,
                                     const Eigen::VectorXd& th_j,
                                     double rho) {
  Eigen::MatrixXd cdf;
  ordinal_bvn_corner_cdf(th_i, th_j, rho, cdf);
  Eigen::MatrixXd out(th_i.size() + 1, th_j.size() + 1);
  for (Eigen::Index a = 0; a < out.rows(); ++a) {
    for (Eigen::Index c = 0; c < out.cols(); ++c) {
      out(a, c) = std::clamp(corner_rect_value(cdf, a, c), 0.0, 1.0);
    }
  }
  const double prob_sum = out.sum();
  if (std::isfinite(prob_sum) && prob_sum > 0.0) {
    out *= total / prob_sum;
  } else {
    out.setZero();
  }
  return out;
}

Eigen::VectorXd ordinal_pair_cell_log_score_local(
    Eigen::Index ci,
    Eigen::Index cj,
    const Eigen::Ref<const Eigen::VectorXd>& th_i,
    const Eigen::Ref<const Eigen::VectorXd>& th_j,
    double rho) {
  const Eigen::Index nth_i = th_i.size();
  const Eigen::Index nth_j = th_j.size();
  Eigen::VectorXd out = Eigen::VectorXd::Zero(nth_i + nth_j + 1);
  const Eigen::Index n_levels_i = nth_i + 1;
  const Eigen::Index n_levels_j = nth_j + 1;
  const double lo_i = (ci == 0) ? -kInf : th_i(ci - 1);
  const double hi_i = (ci + 1 == n_levels_i) ? kInf : th_i(ci);
  const double lo_j = (cj == 0) ? -kInf : th_j(cj - 1);
  const double hi_j = (cj + 1 == n_levels_j) ? kInf : th_j(cj);
  const double lik = std::max(
      kProbFloor, ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));

  for (Eigen::Index a = 0; a < nth_i; ++a) {
    const double t = th_i(a);
    const double z = normal_pdf(t) *
        (normal_cdf((hi_j - rho * t) / sd) -
         normal_cdf((lo_j - rho * t) / sd));
    if (ci == a) out(a) += z / lik;
    if (ci == a + 1) out(a) -= z / lik;
  }
  for (Eigen::Index a = 0; a < nth_j; ++a) {
    const double t = th_j(a);
    const double z = normal_pdf(t) *
        (normal_cdf((hi_i - rho * t) / sd) -
         normal_cdf((lo_i - rho * t) / sd));
    if (cj == a) out(nth_i + a) += z / lik;
    if (cj == a + 1) out(nth_i + a) -= z / lik;
  }
  out(out.size() - 1) =
      ordinal_bvn_rect_drho(lo_i, hi_i, lo_j, hi_j, rho) / lik;
  return out;
}

void add_pair_score_to_global(Eigen::Ref<Eigen::VectorXd> global,
                              const Eigen::VectorXd& local,
                              Eigen::Index nth,
                              Eigen::Index start_i,
                              Eigen::Index start_j,
                              Eigen::Index nth_i,
                              Eigen::Index nth_j,
                              Eigen::Index corr_index,
                              double scale) {
  if (nth_i > 0) {
    global.segment(start_i, nth_i).noalias() +=
        scale * local.head(nth_i);
  }
  if (nth_j > 0) {
    global.segment(start_j, nth_j).noalias() +=
        scale * local.segment(nth_i, nth_j);
  }
  global(nth + corr_index) += scale * local(local.size() - 1);
}

double shared_pair_objective(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho,
    const SharedRobustOrdinalOptions& options) {
  const double total = counts.sum();
  Eigen::MatrixXd cdf;
  ordinal_bvn_corner_cdf(th_i, th_j, rho, cdf);
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double p = std::max(
          kProbFloor, std::clamp(corner_rect_value(cdf, a, b), 0.0, 1.0));
      const double f = counts(a, b) / total;
      if (options.kind == SharedRobustOrdinalKind::HWeighted) {
        auto h = eval_polychoric_h_score(f / p, options.h_score);
        if (!h.has_value()) return std::numeric_limits<double>::infinity();
        out += p * h->phi;
      } else if (options.kind == SharedRobustOrdinalKind::Dpd) {
        const double alpha = options.alpha;
        const double p_alpha = std::pow(p, alpha);
        out += p * p_alpha - ((1.0 + alpha) / alpha) * f * p_alpha;
      } else {
        const double r = pearson_cell_residual(f, p, total);
        auto c = eval_huber_residual_clip(r, options.clip);
        if (!c.has_value()) return std::numeric_limits<double>::infinity();
        out += p * c->loss;
      }
    }
  }
  return out;
}

double shared_ordinal_objective(
    const Eigen::VectorXd& z,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    const std::vector<SharedOrdinalPairTable>& pairs,
    const SharedRobustOrdinalOptions& options) {
  Eigen::VectorXd thresholds;
  std::vector<Eigen::VectorXd> th_by_var;
  Eigen::MatrixXd R;
  decode_shared_ordinal_params(
      z, levels, th_start, options.rho_lower, options.rho_upper,
      options.min_threshold_spacing, thresholds, th_by_var, R);
  double out = 0.0;
  for (const auto& pair : pairs) {
    out += shared_pair_objective(
        pair.adjusted_counts, th_by_var[static_cast<std::size_t>(pair.i)],
        th_by_var[static_cast<std::size_t>(pair.j)], R(pair.i, pair.j),
        options);
  }
  return out;
}

double shared_ordinal_objective_subset(
    const Eigen::VectorXd& z,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    const std::vector<SharedOrdinalPairTable>& pairs,
    const std::vector<std::size_t>& subset,
    const SharedRobustOrdinalOptions& options) {
  Eigen::VectorXd thresholds;
  std::vector<Eigen::VectorXd> th_by_var;
  Eigen::MatrixXd R;
  decode_shared_ordinal_params(
      z, levels, th_start, options.rho_lower, options.rho_upper,
      options.min_threshold_spacing, thresholds, th_by_var, R);
  double out = 0.0;
  for (const std::size_t idx : subset) {
    const auto& pair = pairs[idx];
    out += shared_pair_objective(
        pair.adjusted_counts, th_by_var[static_cast<std::size_t>(pair.i)],
        th_by_var[static_cast<std::size_t>(pair.j)], R(pair.i, pair.j),
        options);
  }
  return out;
}

// Coordinate k moves either one variable's thresholds (the encoding is a
// per-variable cumulative gap chain) or one correlation, so finite
// differences only need the pairs whose contribution actually changes:
// touched[k] lists those pair indices. Valid for both the z encoding and
// the raw (thresholds, rho) layout, which share the coordinate map.
std::vector<std::vector<std::size_t>> shared_ordinal_touched_pairs(
    Eigen::Index nth,
    Eigen::Index mdim,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    const std::vector<SharedOrdinalPairTable>& pairs) {
  std::vector<std::vector<std::size_t>> out(static_cast<std::size_t>(mdim));
  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    const auto& pair = pairs[idx];
    for (const Eigen::Index v : {pair.i, pair.j}) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(v)];
      const Eigen::Index len =
          static_cast<Eigen::Index>(levels[static_cast<std::size_t>(v)] - 1);
      for (Eigen::Index k = 0; k < len; ++k) {
        out[static_cast<std::size_t>(start + k)].push_back(idx);
      }
    }
    out[static_cast<std::size_t>(nth + pair.corr_index)].push_back(idx);
  }
  return out;
}

Eigen::VectorXd shared_ordinal_gradient(
    const Eigen::VectorXd& z,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    const std::vector<SharedOrdinalPairTable>& pairs,
    const std::vector<std::vector<std::size_t>>& touched,
    const SharedRobustOrdinalOptions& options) {
  Eigen::VectorXd grad(z.size());
  for (Eigen::Index k = 0; k < z.size(); ++k) {
    const double h = options.fd_step * std::max(1.0, std::abs(z(k)));
    Eigen::VectorXd zp = z;
    Eigen::VectorXd zm = z;
    zp(k) += h;
    zm(k) -= h;
    const auto& subset = touched[static_cast<std::size_t>(k)];
    grad(k) = (shared_ordinal_objective_subset(
                   zp, levels, th_start, pairs, subset, options) -
               shared_ordinal_objective_subset(
                   zm, levels, th_start, pairs, subset, options)) /
              (2.0 * h);
  }
  return grad;
}

post_expected<void>
threshold_layout_for_block(const OrdinalStats& stats,
                           std::size_t b,
                           Eigen::Index p,
                           std::vector<Eigen::VectorXd>& th_by_var,
                           std::vector<Eigen::Index>& th_start) {
  if (b >= stats.thresholds.size() || b >= stats.threshold_ov.size() ||
      b >= stats.threshold_level.size() || b >= stats.n_levels.size() ||
      stats.n_levels[b].size() != static_cast<std::size_t>(p)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal threshold layout: metadata block mismatch"));
  }
  th_by_var.clear();
  th_by_var.reserve(static_cast<std::size_t>(p));
  th_start.assign(static_cast<std::size_t>(p), -1);
  for (Eigen::Index j = 0; j < p; ++j) {
    const auto levels = stats.n_levels[b][static_cast<std::size_t>(j)];
    if (levels < 2) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal threshold layout: variable has fewer than two levels"));
    }
    th_by_var.emplace_back(Eigen::VectorXd::Constant(
        levels - 1, std::numeric_limits<double>::quiet_NaN()));
  }
  if (stats.threshold_ov[b].size() !=
          static_cast<std::size_t>(stats.thresholds[b].size()) ||
      stats.threshold_level[b].size() !=
          static_cast<std::size_t>(stats.thresholds[b].size())) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal threshold layout: threshold metadata length mismatch"));
  }

  for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
    const auto ov = stats.threshold_ov[b][static_cast<std::size_t>(k)];
    const auto level = stats.threshold_level[b][static_cast<std::size_t>(k)];
    if (ov < 0 || ov >= p || level <= 0 ||
        level > th_by_var[static_cast<std::size_t>(ov)].size()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal threshold layout: invalid threshold metadata"));
    }
    if (level == 1) th_start[static_cast<std::size_t>(ov)] = k;
    th_by_var[static_cast<std::size_t>(ov)](level - 1) =
        stats.thresholds[b](k);
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (th_start[static_cast<std::size_t>(j)] < 0 ||
        !th_by_var[static_cast<std::size_t>(j)].allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal threshold layout: missing thresholds for variable"));
    }
  }
  return {};
}

Eigen::VectorXd shared_ordinal_estimating_equation(
    const Eigen::VectorXd& thresholds,
    const std::vector<Eigen::VectorXd>& th_by_var,
    const Eigen::MatrixXd& R,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    const std::vector<SharedOrdinalPairTable>& pairs,
    const std::vector<std::size_t>& subset,
    const SharedRobustOrdinalOptions& options) {
  const Eigen::Index nth = thresholds.size();
  const Eigen::Index p = R.rows();
  const Eigen::Index mdim = nth + p * (p - 1) / 2;
  Eigen::VectorXd out = Eigen::VectorXd::Zero(mdim);
  for (const std::size_t pair_idx : subset) {
    const auto& pair = pairs[pair_idx];
    const auto& th_i = th_by_var[static_cast<std::size_t>(pair.i)];
    const auto& th_j = th_by_var[static_cast<std::size_t>(pair.j)];
    const Eigen::Index nth_i = th_i.size();
    const Eigen::Index nth_j = th_j.size();
    const Eigen::Index si = th_start[static_cast<std::size_t>(pair.i)];
    const Eigen::Index sj = th_start[static_cast<std::size_t>(pair.j)];
    const double rho = R(pair.i, pair.j);
    const double total = pair.adjusted_counts.sum();
    for (Eigen::Index a = 0; a < levels[static_cast<std::size_t>(pair.i)]; ++a) {
      const double lo_i = (a == 0) ? -kInf : th_i(a - 1);
      const double hi_i = (a + 1 == levels[static_cast<std::size_t>(pair.i)])
          ? kInf : th_i(a);
      for (Eigen::Index b = 0; b < levels[static_cast<std::size_t>(pair.j)]; ++b) {
        const double lo_j = (b == 0) ? -kInf : th_j(b - 1);
        const double hi_j = (b + 1 == levels[static_cast<std::size_t>(pair.j)])
            ? kInf : th_j(b);
        const double pcell = std::max(
            kProbFloor, ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
        const double f = pair.adjusted_counts(a, b) / total;
        const Eigen::VectorXd score =
            ordinal_pair_cell_log_score_local(a, b, th_i, th_j, rho);
        double scale = 0.0;
        if (options.kind == SharedRobustOrdinalKind::HWeighted) {
          auto h = eval_polychoric_h_score(f / pcell, options.h_score);
          if (!h.has_value()) return Eigen::VectorXd::Constant(
              mdim, std::numeric_limits<double>::quiet_NaN());
          scale = h->h * pcell;
        } else if (options.kind == SharedRobustOrdinalKind::Dpd) {
          scale = (pcell - f) * std::pow(pcell, options.alpha);
        } else {
          scale = clipped_cell_frequency(f, pcell, total, options.clip);
        }
        add_pair_score_to_global(out, score, nth, si, sj, nth_i, nth_j,
                                 pair.corr_index, scale);
      }
    }
  }
  return out;
}

Eigen::MatrixXd shared_ordinal_casewise_psi(
    const Eigen::MatrixXi& Xcat,
    const Eigen::VectorXd& thresholds,
    const std::vector<Eigen::VectorXd>& th_by_var,
    const Eigen::MatrixXd& R,
    const std::vector<Eigen::Index>& th_start,
    const std::vector<SharedOrdinalPairTable>& pairs,
    const SharedRobustOrdinalOptions& options) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index nth = thresholds.size();
  const Eigen::Index p = R.rows();
  const Eigen::Index mdim = nth + p * (p - 1) / 2;
  // Accumulate transposed (cases as contiguous columns) so the per-row
  // scatter binds segments directly instead of copying whole rows.
  Eigen::MatrixXd psi_t = Eigen::MatrixXd::Zero(mdim, n);
  std::vector<Eigen::VectorXd> cell_score;
  for (const auto& pair : pairs) {
    const auto& th_i = th_by_var[static_cast<std::size_t>(pair.i)];
    const auto& th_j = th_by_var[static_cast<std::size_t>(pair.j)];
    const Eigen::Index nth_i = th_i.size();
    const Eigen::Index nth_j = th_j.size();
    const Eigen::Index si = th_start[static_cast<std::size_t>(pair.i)];
    const Eigen::Index sj = th_start[static_cast<std::size_t>(pair.j)];
    const double rho = R(pair.i, pair.j);
    const double total = pair.adjusted_counts.sum();
    const Eigen::Index n_levels_i = nth_i + 1;
    const Eigen::Index n_levels_j = nth_j + 1;
    // A row's contribution depends on the row only through its cell, so
    // evaluate the scaled score once per cell and scatter to rows.
    cell_score.assign(
        static_cast<std::size_t>(n_levels_i * n_levels_j), Eigen::VectorXd{});
    for (Eigen::Index ci = 0; ci < n_levels_i; ++ci) {
      const double lo_i = (ci == 0) ? -kInf : th_i(ci - 1);
      const double hi_i = (ci + 1 == n_levels_i) ? kInf : th_i(ci);
      for (Eigen::Index cj = 0; cj < n_levels_j; ++cj) {
        const double lo_j = (cj == 0) ? -kInf : th_j(cj - 1);
        const double hi_j = (cj + 1 == n_levels_j) ? kInf : th_j(cj);
        const double pcell = std::max(
            kProbFloor, ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
        double scale = 1.0;
        if (options.kind == SharedRobustOrdinalKind::HWeighted) {
          const double t = pair.adjusted_counts(ci, cj) / (total * pcell);
          auto h = eval_polychoric_h_score(t, options.h_score);
          scale = h.has_value() ? h->dh
                                : std::numeric_limits<double>::quiet_NaN();
        } else if (options.kind == SharedRobustOrdinalKind::Dpd) {
          scale = std::pow(pcell, options.alpha);
        } else {
          const double f = pair.adjusted_counts(ci, cj) / total;
          scale = clipped_cell_derivative(f, pcell, total, options.clip);
        }
        cell_score[static_cast<std::size_t>(ci * n_levels_j + cj)] =
            scale * ordinal_pair_cell_log_score_local(ci, cj, th_i, th_j, rho);
      }
    }
    for (Eigen::Index row = 0; row < n; ++row) {
      const Eigen::Index cell =
          Xcat(row, pair.i) * n_levels_j + Xcat(row, pair.j);
      add_pair_score_to_global(
          psi_t.col(row), cell_score[static_cast<std::size_t>(cell)], nth, si,
          sj, nth_i, nth_j, pair.corr_index, 1.0);
    }
  }
  Eigen::MatrixXd psi = psi_t.transpose();
  psi.rowwise() -= psi.colwise().mean();
  return psi;
}

Eigen::MatrixXd shared_ordinal_bread_numeric(
    const Eigen::VectorXd& thresholds,
    const Eigen::MatrixXd& R,
    const std::vector<std::int32_t>& levels,
    const std::vector<Eigen::Index>& th_start,
    const std::vector<SharedOrdinalPairTable>& pairs,
    const SharedRobustOrdinalOptions& options) {
  const Eigen::Index nth = thresholds.size();
  const Eigen::Index p = R.rows();
  const Eigen::Index mdim = nth + p * (p - 1) / 2;
  Eigen::VectorXd theta(mdim);
  theta.head(nth) = thresholds;
  Eigen::Index cidx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      theta(nth + cidx++) = R(i, j);
    }
  }

  auto unpack_g = [&](const Eigen::VectorXd& x,
                      const std::vector<std::size_t>& subset) {
    Eigen::VectorXd th = x.head(nth);
    std::vector<Eigen::VectorXd> by_var(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len =
          static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
      by_var[static_cast<std::size_t>(j)] = th.segment(start, len);
    }
    Eigen::MatrixXd r = Eigen::MatrixXd::Identity(p, p);
    Eigen::Index rc = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        r(i, j) = r(j, i) = std::clamp(
            x(nth + rc++), options.rho_lower, options.rho_upper);
      }
    }
    return shared_ordinal_estimating_equation(
        th, by_var, r, levels, th_start, pairs, subset, options);
  };

  // Pairs not touching coordinate k contribute identically at x+h and x-h,
  // so their finite difference is exactly zero; skip them.
  const auto touched =
      shared_ordinal_touched_pairs(nth, mdim, levels, th_start, pairs);
  Eigen::MatrixXd bread(mdim, mdim);
  for (Eigen::Index k = 0; k < mdim; ++k) {
    double h = options.fd_step * std::max(1.0, std::abs(theta(k)));
    if (k < nth) {
      for (Eigen::Index j = 0; j < p; ++j) {
        const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
        const Eigen::Index len =
            static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
        if (k < start || k >= start + len) continue;
        if (k > start) {
          h = std::min(h, 0.25 * (theta(k) - theta(k - 1)));
        }
        if (k + 1 < start + len) {
          h = std::min(h, 0.25 * (theta(k + 1) - theta(k)));
        }
      }
      h = std::max(h, 1e-7);
    }
    Eigen::VectorXd xp = theta;
    Eigen::VectorXd xm = theta;
    xp(k) += h;
    xm(k) -= h;
    if (k >= nth) {
      xp(k) = std::min(options.rho_upper, xp(k));
      xm(k) = std::max(options.rho_lower, xm(k));
    }
    const auto& subset = touched[static_cast<std::size_t>(k)];
    bread.col(k) = (unpack_g(xp, subset) - unpack_g(xm, subset)) /
                   (xp(k) - xm(k));
  }
  return 0.5 * (bread + bread.transpose());
}

}  // namespace

OrdinalMoments ordinal_moments_from_stats(const OrdinalStats& stats) {
  return OrdinalMoments{
      .R = stats.R,
      .thresholds = stats.thresholds,
      .threshold_ov = stats.threshold_ov,
      .threshold_level = stats.threshold_level,
      .n_obs = stats.n_obs,
      .n_levels = stats.n_levels,
      .ov_names = stats.ov_names};
}

MixedOrdinalMoments
mixed_ordinal_moments_from_stats(const MixedOrdinalStats& stats) {
  return MixedOrdinalMoments{
      .R = stats.R,
      .mean = stats.mean,
      .ordered = stats.ordered,
      .thresholds = stats.thresholds,
      .threshold_ov = stats.threshold_ov,
      .threshold_level = stats.threshold_level,
      .moments = stats.moments,
      .n_obs = stats.n_obs,
      .n_levels = stats.n_levels,
      .ov_names = stats.ov_names};
}

OrdinalGammaCache ordinal_gamma_cache_from_stats(const OrdinalStats& stats) {
  OrdinalGammaCache cache;
  cache.blocks.resize(stats.NACOV.size());
  for (std::size_t b = 0; b < stats.NACOV.size(); ++b) {
    auto& block = cache.blocks[b];
    block.gamma = stats.NACOV[b];
    block.has_full = true;
    block.diagonal = stats.NACOV[b].diagonal();
    block.has_diagonal = true;
    if (b < stats.W_dwls.size()) {
      block.w_dwls = stats.W_dwls[b];
      block.has_dwls_weight = true;
    }
    if (b < stats.W_wls.size()) {
      block.w_wls = stats.W_wls[b];
      block.has_wls_weight = true;
    }
  }
  return cache;
}

OrdinalGammaCache
ordinal_gamma_cache_from_stats(const MixedOrdinalStats& stats) {
  OrdinalGammaCache cache;
  cache.blocks.resize(stats.NACOV.size());
  for (std::size_t b = 0; b < stats.NACOV.size(); ++b) {
    auto& block = cache.blocks[b];
    block.gamma = stats.NACOV[b];
    block.has_full = true;
    block.diagonal = stats.NACOV[b].diagonal();
    block.has_diagonal = true;
    // Stats built without the eager WLS inverse (full_wls_weight = false)
    // carry empty W_wls placeholders; leave the provenance flag unset so the
    // cache ensure helpers build the weight on demand.
    if (b < stats.W_dwls.size() && stats.W_dwls[b].size() > 0) {
      block.w_dwls = stats.W_dwls[b];
      block.has_dwls_weight = true;
    }
    if (b < stats.W_wls.size() && stats.W_wls[b].size() > 0) {
      block.w_wls = stats.W_wls[b];
      block.has_wls_weight = true;
    }
  }
  return cache;
}

OrdinalGammaCache
ordinal_gamma_cache_from_diagonal(const std::vector<Eigen::VectorXd>& diagonal) {
  OrdinalGammaCache cache;
  cache.blocks.resize(diagonal.size());
  for (std::size_t b = 0; b < diagonal.size(); ++b) {
    cache.blocks[b].diagonal = diagonal[b];
    cache.blocks[b].has_diagonal = true;
  }
  return cache;
}

OrdinalWeightPlan ordinal_weight_plan(
    OrdinalWorkspacePurpose purpose,
    OrdinalEstimatorKind estimator,
    OrdinalMomentParameterization parameterization,
    OrdinalThresholdMode threshold_mode) {
  OrdinalGammaMaterialization materialization =
      OrdinalGammaMaterialization::None;
  if (purpose == OrdinalWorkspacePurpose::FitOnly) {
    if (estimator == OrdinalEstimatorKind::DWLS) {
      materialization = OrdinalGammaMaterialization::Diagonal;
    } else if (estimator == OrdinalEstimatorKind::WLS) {
      materialization = OrdinalGammaMaterialization::Full;
    }
  } else {
    materialization = OrdinalGammaMaterialization::Full;
  }
  return OrdinalWeightPlan{
      .purpose = purpose,
      .estimator = estimator,
      .parameterization = parameterization,
      .threshold_mode = threshold_mode,
      .materialization = materialization};
}

post_expected<void> ordinal_gamma_cache_ensure_diagonal(
    OrdinalGammaCache& cache) {
  for (std::size_t b = 0; b < cache.blocks.size(); ++b) {
    auto& block = cache.blocks[b];
    if (block.has_diagonal) continue;
    if (!block.has_full) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "OrdinalGammaCache block " + std::to_string(b) +
              " has no full Gamma to derive a diagonal from"));
    }
    if (block.gamma.rows() != block.gamma.cols() || !block.gamma.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "OrdinalGammaCache block " + std::to_string(b) +
              " full Gamma is not a finite square matrix"));
    }
    block.diagonal = block.gamma.diagonal();
    block.has_diagonal = true;
  }
  return {};
}

post_expected<void> ordinal_gamma_cache_ensure_dwls_weights(
    OrdinalGammaCache& cache) {
  auto diag_ok = ordinal_gamma_cache_ensure_diagonal(cache);
  if (!diag_ok.has_value()) return std::unexpected(diag_ok.error());
  for (std::size_t b = 0; b < cache.blocks.size(); ++b) {
    auto& block = cache.blocks[b];
    if (block.has_dwls_weight) continue;
    const Eigen::Index n = block.diagonal.size();
    block.w_dwls = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index k = 0; k < n; ++k) {
      const double v = block.diagonal(k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "OrdinalGammaCache block " + std::to_string(b) +
                " has a non-positive Gamma diagonal"));
      }
      block.w_dwls(k, k) = 1.0 / v;
    }
    block.has_dwls_weight = true;
  }
  return {};
}

post_expected<void> ordinal_gamma_cache_ensure_wls_weights(
    OrdinalGammaCache& cache) {
  for (std::size_t b = 0; b < cache.blocks.size(); ++b) {
    auto& block = cache.blocks[b];
    if (block.has_wls_weight) continue;
    if (!block.has_full) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "OrdinalGammaCache block " + std::to_string(b) +
              " has no full Gamma for WLS weight construction"));
    }
    auto inv_or = symmetric_inverse_pd(
        block.gamma, "OrdinalGammaCache block " + std::to_string(b) +
                         " full Gamma");
    if (!inv_or.has_value()) return std::unexpected(inv_or.error());
    block.w_wls = std::move(*inv_or);
    block.has_wls_weight = true;
  }
  return {};
}

post_expected<OrdinalWorkspace> ordinal_workspace_from_integer_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    OrdinalWeightPlan plan) {
  const bool lazy_fit_only_uls =
      plan.purpose == OrdinalWorkspacePurpose::FitOnly &&
      plan.estimator == OrdinalEstimatorKind::ULS &&
      plan.materialization == OrdinalGammaMaterialization::None;
  const bool lazy_fit_only_dwls =
      plan.purpose == OrdinalWorkspacePurpose::FitOnly &&
      plan.estimator == OrdinalEstimatorKind::DWLS &&
      plan.materialization == OrdinalGammaMaterialization::Diagonal;
  if (!lazy_fit_only_uls && !lazy_fit_only_dwls) {
    auto stats_or = ordinal_stats_from_integer_data(Xs);
    if (!stats_or.has_value()) return std::unexpected(stats_or.error());
    return OrdinalWorkspace{
        .moments = ordinal_moments_from_stats(*stats_or),
        .gamma_cache = ordinal_gamma_cache_from_stats(*stats_or)};
  }
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_workspace_from_integer_data: no data blocks"));
  }

  const bool need_diagonal = lazy_fit_only_dwls;
  OrdinalWorkspace out;
  auto& moments = out.moments;
  moments.R.reserve(Xs.size());
  moments.thresholds.reserve(Xs.size());
  moments.threshold_ov.reserve(Xs.size());
  moments.threshold_level.reserve(Xs.size());
  moments.n_obs.reserve(Xs.size());
  moments.n_levels.reserve(Xs.size());
  if (need_diagonal) out.gamma_cache.blocks.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2 || p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_workspace_from_integer_data: block " + std::to_string(b) +
              " must have at least 2 rows and 1 column"));
    }

    Eigen::MatrixXi Xcat(n, p);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
      int max_level = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        const double v = X(i, j);
        if (!std::isfinite(v) || std::floor(v) != v || v < 1.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_workspace_from_integer_data: block " +
                  std::to_string(b) +
                  " has non-integer/non-positive category values"));
        }
        max_level = std::max(max_level, static_cast<int>(v));
      }
      if (max_level < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_workspace_from_integer_data: block " + std::to_string(b) +
                " variable " + std::to_string(j) + " has fewer than 2 levels"));
      }
      counts[static_cast<std::size_t>(j)].assign(
          static_cast<std::size_t>(max_level), 0);
      for (Eigen::Index i = 0; i < n; ++i) {
        const int lvl = static_cast<int>(X(i, j));
        counts[static_cast<std::size_t>(j)]
              [static_cast<std::size_t>(lvl - 1)] += 1;
        Xcat(i, j) = lvl - 1;
      }
      for (int c = 0; c < max_level; ++c) {
        if (counts[static_cast<std::size_t>(j)]
                  [static_cast<std::size_t>(c)] == 0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_workspace_from_integer_data: block " +
                  std::to_string(b) + " variable " + std::to_string(j) +
                  " has an empty category"));
        }
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::Index nth = 0;
    for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;
    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    th_ov.reserve(static_cast<std::size_t>(nth));
    th_level.reserve(static_cast<std::size_t>(nth));

    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      th_start[static_cast<std::size_t>(j)] = off;
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
        if (cum <= 0 || cum >= n) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_workspace_from_integer_data: block " +
                  std::to_string(b) + " variable " + std::to_string(j) +
                  " has an empty boundary category"));
        }
        const double prob = std::clamp(
            static_cast<double>(cum) / static_cast<double>(n), 1e-12,
            1.0 - 1e-12);
        const double z = normal_quantile(prob);
        th(off) = z;
        th_by_var[static_cast<std::size_t>(j)](c) = z;
        th_ov.push_back(static_cast<std::int32_t>(j));
        th_level.push_back(static_cast<std::int32_t>(c + 1));
        ++off;
      }
    }

    Eigen::MatrixXd SC_TH;
    if (need_diagonal) {
      SC_TH = Eigen::MatrixXd::Zero(n, nth);
      for (Eigen::Index r = 0; r < n; ++r) {
        for (Eigen::Index j = 0; j < p; ++j) {
          const int c = Xcat(r, j);
          const auto& thj = th_by_var[static_cast<std::size_t>(j)];
          const double lo = (c == 0) ? -kInf : thj(c - 1);
          const double hi = (c == thj.size()) ? kInf : thj(c);
          const double pr =
              std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
          const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
          if (c < thj.size()) {
            SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
          }
          if (c > 0) {
            SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
          }
        }
      }
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    Eigen::MatrixXd SC_COR;
    Eigen::MatrixXd A21;
    if (need_diagonal) {
      SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
      A21 = Eigen::MatrixXd::Zero(ncorr, nth);
    }
    Eigen::Index corr_idx = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const Eigen::VectorXi xi = Xcat.col(i);
        const Eigen::VectorXi xj = Xcat.col(j);
        auto tab_or = ordinal_pair_table(
            xi, xj, levels[static_cast<std::size_t>(i)],
            levels[static_cast<std::size_t>(j)]);
        if (!tab_or.has_value()) return std::unexpected(tab_or.error());
        auto rho_or = fit_ordinal_pair_rho_ml(
            *tab_or, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!rho_or.has_value()) return std::unexpected(rho_or.error());
        const double rho = rho_or->rho;
        R(i, j) = R(j, i) = rho;
        if (need_diagonal) {
          auto ps_or = ordinal_pair_scores(
              xi, xj, rho, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          if (!ps_or.has_value()) return std::unexpected(ps_or.error());
          const auto& ps = *ps_or;
          SC_COR.col(corr_idx) = ps.rho;
          const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
          const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
          A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
              ps.rho.transpose() * ps.threshold_i;
          A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
              ps.rho.transpose() * ps.threshold_j;
        }
        ++corr_idx;
      }
    }
    auto r_diag_or = repair_correlation_if_requested(
        R, {}, "ordinal_workspace_from_integer_data: block " +
                   std::to_string(b) + " polychoric R");
    if (!r_diag_or.has_value()) return std::unexpected(r_diag_or.error());

    moments.R.push_back(std::move(R));
    moments.thresholds.push_back(std::move(th));
    moments.threshold_ov.push_back(std::move(th_ov));
    moments.threshold_level.push_back(std::move(th_level));
    moments.n_obs.push_back(static_cast<std::int64_t>(n));
    moments.n_levels.push_back(std::move(levels));

    if (!need_diagonal) continue;

    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
    const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
    for (Eigen::Index j = 0; j < p; ++j) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len = static_cast<Eigen::Index>(
          moments.n_levels.back()[static_cast<std::size_t>(j)] - 1);
      A11.block(start, start, len, len) =
          INNER_TH.block(start, start, len, len);
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "ordinal threshold information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::VectorXd A22_diag(ncorr);
    for (Eigen::Index k = 0; k < ncorr; ++k) {
      A22_diag(k) = SC_COR.col(k).squaredNorm();
      if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_workspace_from_integer_data: block " + std::to_string(b) +
                " has a singular polychoric score block"));
      }
    }

    Eigen::VectorXd diagonal(mdim);
    for (Eigen::Index k = 0; k < nth; ++k) {
      const Eigen::VectorXd coeff = A11_inv_or->row(k).transpose();
      const Eigen::VectorXd infl = SC_TH * coeff;
      diagonal(k) = static_cast<double>(n) * infl.squaredNorm();
    }
    for (Eigen::Index k = 0; k < ncorr; ++k) {
      const double a22_inv = 1.0 / A22_diag(k);
      const Eigen::RowVectorXd coeff_th =
          -a22_inv * (A21.row(k) * (*A11_inv_or));
      const Eigen::VectorXd infl =
          SC_TH * coeff_th.transpose() + a22_inv * SC_COR.col(k);
      diagonal(nth + k) = static_cast<double>(n) * infl.squaredNorm();
    }
    for (Eigen::Index k = 0; k < mdim; ++k) {
      if (!std::isfinite(diagonal(k)) || diagonal(k) <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_workspace_from_integer_data: block " +
                std::to_string(b) + " has a non-positive Gamma diagonal"));
      }
    }

    OrdinalGammaCacheBlock block;
    block.diagonal = std::move(diagonal);
    block.has_diagonal = true;
    out.gamma_cache.blocks.push_back(std::move(block));
  }
  return out;
}

post_expected<MixedOrdinalWorkspace> mixed_ordinal_workspace_from_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    const std::vector<std::vector<std::int32_t>>& ordered,
    OrdinalWeightPlan plan) {
  const bool lazy_fit_only_uls =
      plan.purpose == OrdinalWorkspacePurpose::FitOnly &&
      plan.estimator == OrdinalEstimatorKind::ULS &&
      plan.materialization == OrdinalGammaMaterialization::None;
  const bool lazy_fit_only_dwls =
      plan.purpose == OrdinalWorkspacePurpose::FitOnly &&
      plan.estimator == OrdinalEstimatorKind::DWLS &&
      plan.materialization == OrdinalGammaMaterialization::Diagonal;
  if (!lazy_fit_only_uls && !lazy_fit_only_dwls) {
    // WLS and fit-plus-inference plans carry the full Gamma in the cache but
    // defer every weight: the O(m^3) WLS inverse and the DWLS diagonal weight
    // are built on demand by the ordinal_gamma_cache_ensure_* helpers when a
    // fit or robust call actually needs them.
    auto stats_or = mixed_ordinal_stats_from_data(Xs, ordered,
                                                  /*full_wls_weight=*/false);
    if (!stats_or.has_value()) return std::unexpected(stats_or.error());
    return MixedOrdinalWorkspace{
        .moments = mixed_ordinal_moments_from_stats(*stats_or),
        .gamma_cache = ordinal_gamma_cache_from_stats(*stats_or)};
  }
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_workspace_from_data: no data blocks"));
  }
  if (Xs.size() != ordered.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_workspace_from_data: ordered mask block count mismatch"));
  }

  const bool need_diagonal = lazy_fit_only_dwls;
  MixedOrdinalWorkspace out;
  auto& moments = out.moments;
  moments.R.reserve(Xs.size());
  moments.mean.reserve(Xs.size());
  moments.ordered.reserve(Xs.size());
  moments.thresholds.reserve(Xs.size());
  moments.threshold_ov.reserve(Xs.size());
  moments.threshold_level.reserve(Xs.size());
  moments.moments.reserve(Xs.size());
  moments.n_obs.reserve(Xs.size());
  moments.n_levels.reserve(Xs.size());
  if (need_diagonal) out.gamma_cache.blocks.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2 || p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_workspace_from_data: block " + std::to_string(b) +
              " must have at least 2 rows and 1 column"));
    }
    if (ordered[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_workspace_from_data: block " + std::to_string(b) +
              " ordered mask length does not match column count"));
    }
    bool have_ordered = false;
    bool have_cont = false;
    for (auto z : ordered[b]) {
      have_ordered = have_ordered || z != 0;
      have_cont = have_cont || z == 0;
    }
    if (!have_ordered || !have_cont) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_workspace_from_data: block " + std::to_string(b) +
              " must contain both ordered and continuous variables"));
    }

    Eigen::MatrixXi Xcat = Eigen::MatrixXi::Zero(n, p);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        for (Eigen::Index i = 0; i < n; ++i) {
          if (!std::isfinite(X(i, j))) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_workspace_from_data: block " +
                    std::to_string(b) +
                    " has non-finite continuous values"));
          }
        }
        continue;
      }
      int max_level = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        const double v = X(i, j);
        if (!std::isfinite(v) || std::floor(v) != v || v < 1.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_workspace_from_data: block " +
                  std::to_string(b) +
                  " has non-integer/non-positive category values"));
        }
        max_level = std::max(max_level, static_cast<int>(v));
      }
      if (max_level < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_workspace_from_data: block " + std::to_string(b) +
                " variable " + std::to_string(j) + " has fewer than 2 levels"));
      }
      counts[static_cast<std::size_t>(j)].assign(
          static_cast<std::size_t>(max_level), 0);
      for (Eigen::Index i = 0; i < n; ++i) {
        const int lvl = static_cast<int>(X(i, j));
        counts[static_cast<std::size_t>(j)]
              [static_cast<std::size_t>(lvl - 1)] += 1;
        Xcat(i, j) = lvl - 1;
      }
      for (int c = 0; c < max_level; ++c) {
        if (counts[static_cast<std::size_t>(j)]
                  [static_cast<std::size_t>(c)] == 0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_workspace_from_data: block " +
                  std::to_string(b) + " variable " + std::to_string(j) +
                  " has an empty category"));
        }
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd var = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
      mean(j) = X.col(j).mean();
      for (Eigen::Index i = 0; i < n; ++i) {
        const double c = X(i, j) - mean(j);
        var(j) += c * c;
      }
      var(j) /= static_cast<double>(n);
      if (var(j) <= 0.0 || !std::isfinite(var(j))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_workspace_from_data: block " + std::to_string(b) +
                " has a non-positive continuous variance"));
      }
      U.col(j) = (X.col(j).array() - mean(j)) / std::sqrt(var(j));
    }

    Eigen::Index nth = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) {
        nth += static_cast<Eigen::Index>(
            levels[static_cast<std::size_t>(j)] - 1);
      }
    }
    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    th_ov.reserve(static_cast<std::size_t>(nth));
    th_level.reserve(static_cast<std::size_t>(nth));
    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
      th_start[static_cast<std::size_t>(j)] = off;
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
        const double prob = std::clamp(
            static_cast<double>(cum) / static_cast<double>(n), 1e-12,
            1.0 - 1e-12);
        const double z = normal_quantile(prob);
        th(off) = z;
        th_by_var[static_cast<std::size_t>(j)](c) = z;
        th_ov.push_back(static_cast<std::int32_t>(j));
        th_level.push_back(static_cast<std::int32_t>(c + 1));
        ++off;
      }
    }

    // The lazy diagonal mirrors the eager muthen1984-style sandwich in
    // mixed_ordinal_stats_from_data_impl (stage-1 stack [th | mu | var],
    // per-variable A11 blocks, pair-ML association scores, delta-rule
    // covariance transform) but only ever materializes column norms.
    Eigen::Index n_cont = 0;
    std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        cont_pos[static_cast<std::size_t>(j)] = n_cont++;
      }
    }
    const Eigen::Index s1 = nth + 2 * n_cont;

    Eigen::MatrixXd SC1;
    Eigen::MatrixXd A11_inv;
    if (need_diagonal) {
      SC1 = Eigen::MatrixXd::Zero(n, s1);
      for (Eigen::Index r = 0; r < n; ++r) {
        for (Eigen::Index j = 0; j < p; ++j) {
          if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
          const int c = Xcat(r, j);
          const auto& thj = th_by_var[static_cast<std::size_t>(j)];
          const double lo = (c == 0) ? -kInf : thj(c - 1);
          const double hi = (c == thj.size()) ? kInf : thj(c);
          const double pr =
              std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
          const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
          if (c < thj.size()) {
            SC1(r, base + c) += normal_pdf(thj(c)) / pr;
          }
          if (c > 0) {
            SC1(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
          }
        }
      }
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
        const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
        const double v = var(j);
        SC1.col(nth + cp) = (X.col(j).array() - mean(j)) / v;
        SC1.col(nth + n_cont + cp) =
            ((X.col(j).array() - mean(j)).square() - v) / (2.0 * v * v);
      }
      const Eigen::MatrixXd INNER1 = SC1.transpose() * SC1;
      Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(s1, s1);
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] != 0) {
          const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
          const Eigen::Index len = static_cast<Eigen::Index>(
              levels[static_cast<std::size_t>(j)] - 1);
          A11.block(start, start, len, len) =
              INNER1.block(start, start, len, len);
        } else {
          const Eigen::Index a = nth + cont_pos[static_cast<std::size_t>(j)];
          const Eigen::Index v =
              nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
          A11(a, a) = INNER1(a, a);
          A11(v, v) = INNER1(v, v);
          A11(a, v) = A11(v, a) = INNER1(a, v);
        }
      }
      auto A11_inv_or = symmetric_inverse_pd(
          A11, "mixed ordinal stage-1 information matrix");
      if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());
      A11_inv = std::move(*A11_inv_or);
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) R(j, j) = var(j);
    }

    const Eigen::Index n_assoc = p * (p - 1) / 2;
    std::vector<Eigen::VectorXd> assoc_scores;
    assoc_scores.reserve(static_cast<std::size_t>(n_assoc));
    Eigen::MatrixXd A21 =
        need_diagonal ? Eigen::MatrixXd::Zero(n_assoc, s1)
                      : Eigen::MatrixXd();
    Eigen::VectorXd A22_diag =
        need_diagonal ? Eigen::VectorXd::Zero(n_assoc) : Eigen::VectorXd();
    Eigen::Index assoc_count = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        if (oi && oj) {
          const Eigen::VectorXi xi = Xcat.col(i);
          const Eigen::VectorXi xj = Xcat.col(j);
          auto tab_or = ordinal_pair_table(
              xi, xj, levels[static_cast<std::size_t>(i)],
              levels[static_cast<std::size_t>(j)]);
          if (!tab_or.has_value()) return std::unexpected(tab_or.error());
          auto rho_or = fit_ordinal_pair_rho_ml(
              *tab_or, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          if (!rho_or.has_value()) return std::unexpected(rho_or.error());
          const double rho = rho_or->rho;
          R(i, j) = R(j, i) = rho;
          if (need_diagonal) {
            auto ps_or = ordinal_pair_scores(
                xi, xj, rho, th_by_var[static_cast<std::size_t>(i)],
                th_by_var[static_cast<std::size_t>(j)]);
            if (!ps_or.has_value()) return std::unexpected(ps_or.error());
            const auto& ps = *ps_or;
            assoc_scores.push_back(ps.rho);
            const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
            const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
            A21.block(assoc_count, si, 1, ps.threshold_i.cols()) =
                ps.rho.transpose() * ps.threshold_i;
            A21.block(assoc_count, sj, 1, ps.threshold_j.cols()) =
                ps.rho.transpose() * ps.threshold_j;
          }
        } else if (oi || oj) {
          const Eigen::Index o = oi ? i : j;
          const Eigen::Index c = oi ? j : i;
          Eigen::VectorXi cat(n);
          for (Eigen::Index r = 0; r < n; ++r) cat(r) = Xcat(r, o);
          const double sd = std::sqrt(var(c));
          auto rho_or = fit_polyserial_pair_rho_ml(
              cat, U.col(c), th_by_var[static_cast<std::size_t>(o)]);
          if (!rho_or.has_value()) return std::unexpected(rho_or.error());
          const double rho = rho_or->rho;
          R(i, j) = R(j, i) = rho * sd;
          if (need_diagonal) {
            const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
            auto ps_or = polyserial_pair_scores(
                cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
            if (!ps_or.has_value()) return std::unexpected(ps_or.error());
            const auto& ps = *ps_or;
            assoc_scores.push_back(ps.rho);
            A21.block(assoc_count, so, 1, ps.thresholds.cols()) =
                ps.rho.transpose() * ps.thresholds;
            A21(assoc_count, nth + cont_pos[static_cast<std::size_t>(c)]) =
                ps.rho.dot(ps.mu_unit) / sd;
            A21(assoc_count,
                nth + n_cont + cont_pos[static_cast<std::size_t>(c)]) =
                ps.rho.dot(ps.var_unit) / var(c);
          }
        } else {
          double cov = 0.0;
          for (Eigen::Index r = 0; r < n; ++r) {
            cov += (X(r, i) - mean(i)) * (X(r, j) - mean(j));
          }
          cov /= static_cast<double>(n);
          R(i, j) = R(j, i) = cov;
          if (need_diagonal) {
            auto sc_or = continuous_pair_normal_scores(
                X.col(i), X.col(j), mean(i), mean(j), var(i), var(j), cov);
            if (!sc_or.has_value()) return std::unexpected(sc_or.error());
            const Eigen::MatrixXd& S = sc_or->score_contributions;
            const double sdi = std::sqrt(var(i));
            const double sdj = std::sqrt(var(j));
            const double rho = cov / (sdi * sdj);
            const Eigen::VectorXd s_rho = sdi * sdj * S.col(4);
            assoc_scores.push_back(s_rho);
            const Eigen::VectorXd ch_var_i =
                S.col(2) + (rho * sdj / (2.0 * sdi)) * S.col(4);
            const Eigen::VectorXd ch_var_j =
                S.col(3) + (rho * sdi / (2.0 * sdj)) * S.col(4);
            A21(assoc_count, nth + cont_pos[static_cast<std::size_t>(i)]) =
                s_rho.dot(S.col(0));
            A21(assoc_count, nth + cont_pos[static_cast<std::size_t>(j)]) =
                s_rho.dot(S.col(1));
            A21(assoc_count,
                nth + n_cont + cont_pos[static_cast<std::size_t>(i)]) =
                s_rho.dot(ch_var_i);
            A21(assoc_count,
                nth + n_cont + cont_pos[static_cast<std::size_t>(j)]) =
                s_rho.dot(ch_var_j);
          }
        }
        if (need_diagonal) {
          A22_diag(assoc_count) =
              assoc_scores[static_cast<std::size_t>(assoc_count)].squaredNorm();
          if (A22_diag(assoc_count) <= 1e-12 ||
              !std::isfinite(A22_diag(assoc_count))) {
            A22_diag(assoc_count) = 1.0;
          }
        }
        ++assoc_count;
      }
    }

    const Eigen::VectorXd moment = mixed_moment_vector(R, mean, ordered[b], th);
    if (!need_diagonal) {
      moments.R.push_back(std::move(R));
      moments.mean.push_back(std::move(mean));
      moments.ordered.push_back(ordered[b]);
      moments.thresholds.push_back(std::move(th));
      moments.threshold_ov.push_back(std::move(th_ov));
      moments.threshold_level.push_back(std::move(th_level));
      moments.moments.push_back(moment);
      moments.n_obs.push_back(static_cast<std::int64_t>(n));
      moments.n_levels.push_back(levels);
      continue;
    }

    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(s1 + n_assoc,
                                                  s1 + n_assoc);
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
    B_inv.block(0, 0, s1, s1) = A11_inv;
    B_inv.block(s1, 0, n_assoc, s1).noalias() =
        -A22_inv * A21.topRows(n_assoc) * A11_inv;
    B_inv.block(s1, s1, n_assoc, n_assoc) = A22_inv;
    Eigen::MatrixXd SC(n, s1 + n_assoc);
    SC.leftCols(s1) = SC1;
    for (Eigen::Index k = 0; k < n_assoc; ++k) {
      SC.col(s1 + k) = assoc_scores[static_cast<std::size_t>(k)];
    }
    Eigen::MatrixXd IF_est = static_cast<double>(n) * SC * B_inv.transpose();

    Eigen::VectorXd diagonal(moment.size());
    Eigen::Index pos = 0;
    for (Eigen::Index k = 0; k < nth; ++k) {
      diagonal(pos++) = IF_est.col(k).squaredNorm() / static_cast<double>(n);
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        diagonal(pos++) =
            IF_est.col(nth + cont_pos[static_cast<std::size_t>(j)])
                .squaredNorm() /
            static_cast<double>(n);
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        diagonal(pos++) =
            IF_est
                .col(nth + n_cont + cont_pos[static_cast<std::size_t>(j)])
                .squaredNorm() /
            static_cast<double>(n);
      }
    }
    Eigen::Index assoc_pos = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        const Eigen::VectorXd rho_col = IF_est.col(s1 + assoc_pos);
        if (oi && oj) {
          diagonal(pos++) = rho_col.squaredNorm() / static_cast<double>(n);
        } else if (oi || oj) {
          const Eigen::Index c = oi ? j : i;
          const double sd = std::sqrt(var(c));
          const double rho = R(i, j) / sd;
          const Eigen::VectorXd col = sd * rho_col +
              (rho / (2.0 * sd)) *
                  IF_est.col(nth + n_cont +
                             cont_pos[static_cast<std::size_t>(c)]);
          diagonal(pos++) = col.squaredNorm() / static_cast<double>(n);
        } else {
          const double sdi = std::sqrt(var(i));
          const double sdj = std::sqrt(var(j));
          const double rho = R(i, j) / (sdi * sdj);
          const Eigen::VectorXd col = sdi * sdj * rho_col +
              (rho * sdj / (2.0 * sdi)) *
                  IF_est.col(nth + n_cont +
                             cont_pos[static_cast<std::size_t>(i)]) +
              (rho * sdi / (2.0 * sdj)) *
                  IF_est.col(nth + n_cont +
                             cont_pos[static_cast<std::size_t>(j)]);
          diagonal(pos++) = col.squaredNorm() / static_cast<double>(n);
        }
        ++assoc_pos;
      }
    }
    for (Eigen::Index k = 0; k < diagonal.size(); ++k) {
      if (!std::isfinite(diagonal(k)) || diagonal(k) <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_workspace_from_data: block " + std::to_string(b) +
                " has a non-positive Gamma diagonal"));
      }
    }

    OrdinalGammaCacheBlock block;
    block.diagonal = std::move(diagonal);
    block.has_diagonal = true;
    out.gamma_cache.blocks.push_back(std::move(block));
    moments.R.push_back(std::move(R));
    moments.mean.push_back(std::move(mean));
    moments.ordered.push_back(ordered[b]);
    moments.thresholds.push_back(std::move(th));
    moments.threshold_ov.push_back(std::move(th_ov));
    moments.threshold_level.push_back(std::move(th_level));
    moments.moments.push_back(moment);
    moments.n_obs.push_back(static_cast<std::int64_t>(n));
    moments.n_levels.push_back(levels);
  }
  return out;
}

// Recompute the polychoric NACOV diagonal Γ̂_kk = n·diag(B_inv·INNER·B_inv') at a
// GIVEN κ = (thresholds, R), holding the integer data fixed. Mirrors the
// estimation path's score/B_inv/NACOV assembly but takes κ as input (no
// re-estimation), so finite-differencing it over κ gives ∂Γ̂/∂κ.
static post_expected<Eigen::VectorXd> gamma_diag_at_kappa(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  Eigen::Index off = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    th_start[static_cast<std::size_t>(j)] = off;
    const Eigen::Index kj =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    th_by_var[static_cast<std::size_t>(j)] = thresholds.segment(off, kj);
    off += kj;
  }

  Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const int c = Xcat(r, j);
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }

  Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
  Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
  Eigen::Index corr_idx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const Eigen::VectorXi xi = Xcat.col(i);
      const Eigen::VectorXi xj = Xcat.col(j);
      auto ps_or = ordinal_pair_scores(xi, xj, R(i, j),
          th_by_var[static_cast<std::size_t>(i)],
          th_by_var[static_cast<std::size_t>(j)]);
      if (!ps_or.has_value()) return std::unexpected(ps_or.error());
      const auto& ps = *ps_or;
      SC_COR.col(corr_idx) = ps.rho;
      const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
      const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
      A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
          ps.rho.transpose() * ps.threshold_i;
      A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
          ps.rho.transpose() * ps.threshold_j;
      ++corr_idx;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(nth) = SC_TH;
  SC.rightCols(ncorr) = SC_COR;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
  const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
  for (Eigen::Index j = 0; j < p; ++j) {
    const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index len =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    A11.block(start, start, len, len) = INNER_TH.block(start, start, len, len);
  }
  auto A11_inv_or = symmetric_inverse_pd(A11, "gamma_diag_at_kappa A11");
  if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());
  Eigen::VectorXd A22_diag(ncorr);
  for (Eigen::Index k = 0; k < ncorr; ++k) {
    A22_diag(k) = SC_COR.col(k).squaredNorm();
    if (!(A22_diag(k) > 0.0))
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "gamma_diag_at_kappa: singular polychoric score block"));
  }
  const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

  Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
  B_inv.block(0, 0, nth, nth) = *A11_inv_or;
  B_inv.block(nth, 0, ncorr, nth).noalias() = -A22_inv * A21 * (*A11_inv_or);
  B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

  const Eigen::MatrixXd BI_INNER = B_inv * INNER;
  Eigen::VectorXd gdiag(mdim);
  for (Eigen::Index k = 0; k < mdim; ++k)
    gdiag(k) = static_cast<double>(n) * BI_INNER.row(k).dot(B_inv.row(k));
  return gdiag;
}

static post_expected<Eigen::MatrixXd> gamma_at_kappa(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  Eigen::Index off = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    th_start[static_cast<std::size_t>(j)] = off;
    const Eigen::Index kj =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    th_by_var[static_cast<std::size_t>(j)] = thresholds.segment(off, kj);
    off += kj;
  }

  Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const int c = Xcat(r, j);
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }

  Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
  Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
  Eigen::Index corr_idx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const Eigen::VectorXi xi = Xcat.col(i);
      const Eigen::VectorXi xj = Xcat.col(j);
      auto ps_or = ordinal_pair_scores(xi, xj, R(i, j),
          th_by_var[static_cast<std::size_t>(i)],
          th_by_var[static_cast<std::size_t>(j)]);
      if (!ps_or.has_value()) return std::unexpected(ps_or.error());
      const auto& ps = *ps_or;
      SC_COR.col(corr_idx) = ps.rho;
      const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
      const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
      A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
          ps.rho.transpose() * ps.threshold_i;
      A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
          ps.rho.transpose() * ps.threshold_j;
      ++corr_idx;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(nth) = SC_TH;
  SC.rightCols(ncorr) = SC_COR;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
  const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
  for (Eigen::Index j = 0; j < p; ++j) {
    const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index len =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    A11.block(start, start, len, len) = INNER_TH.block(start, start, len, len);
  }
  auto A11_inv_or = symmetric_inverse_pd(A11, "gamma_at_kappa A11");
  if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());
  Eigen::VectorXd A22_diag(ncorr);
  for (Eigen::Index k = 0; k < ncorr; ++k) {
    A22_diag(k) = SC_COR.col(k).squaredNorm();
    if (!(A22_diag(k) > 0.0))
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "gamma_at_kappa: singular polychoric score block"));
  }
  const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

  Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
  B_inv.block(0, 0, nth, nth) = *A11_inv_or;
  B_inv.block(nth, 0, ncorr, nth).noalias() = -A22_inv * A21 * (*A11_inv_or);
  B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

  Eigen::MatrixXd Gamma =
      static_cast<double>(n) * B_inv * INNER * B_inv.transpose();
  return 0.5 * (Gamma + Gamma.transpose()).eval();
}

namespace {

struct OrdinalObservedPairBread {
  Eigen::Index vi = 0;
  Eigen::Index vj = 0;
  Eigen::MatrixXd a21i;
  Eigen::MatrixXd a21j;
};

struct OrdinalObservedGammaAssembly {
  Eigen::MatrixXd scores;
  Eigen::MatrixXd bread_inv;
  Eigen::MatrixXd gamma;
  std::vector<Eigen::Index> th_start;
  std::vector<Eigen::Index> th_len;
  std::vector<OrdinalObservedPairBread> pairs;
  Eigen::Index nth = 0;
  Eigen::Index ncorr = 0;
  Eigen::Index mdim = 0;
};

post_expected<OrdinalObservedGammaAssembly>
ordinal_observed_gamma_assembly_at_kappa(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R,
    const std::string& context) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  if (n < 2 || p == 0 || R.rows() != p || R.cols() != p ||
      levels.size() != static_cast<std::size_t>(p)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        context + ": dimension mismatch"));
  }

  Eigen::Index nth = 0;
  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::Index> th_len(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  for (Eigen::Index j = 0; j < p; ++j) {
    const auto lev = levels[static_cast<std::size_t>(j)];
    if (lev < 2) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          context + ": variable has fewer than two levels"));
    }
    th_start[static_cast<std::size_t>(j)] = nth;
    th_len[static_cast<std::size_t>(j)] =
        static_cast<Eigen::Index>(lev - 1);
    nth += th_len[static_cast<std::size_t>(j)];
  }
  if (thresholds.size() != nth) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        context + ": threshold length mismatch"));
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    th_by_var[static_cast<std::size_t>(j)] =
        thresholds.segment(th_start[static_cast<std::size_t>(j)],
                           th_len[static_cast<std::size_t>(j)]);
  }

  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const int c = Xcat(r, j);
      if (c < 0) continue;
      if (c >= levels[static_cast<std::size_t>(j)]) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            context + ": observed category is out of range"));
      }
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }

  Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
  Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
  std::vector<OrdinalObservedPairBread> pairs;
  pairs.reserve(static_cast<std::size_t>(ncorr));
  Eigen::Index corr_idx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      std::vector<Eigen::Index> obs_rows;
      obs_rows.reserve(static_cast<std::size_t>(n));
      for (Eigen::Index r = 0; r < n; ++r) {
        if (Xcat(r, i) >= 0 && Xcat(r, j) >= 0) obs_rows.push_back(r);
      }
      if (obs_rows.size() < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            context + ": pair has fewer than two observed rows"));
      }

      Eigen::VectorXi xi(static_cast<Eigen::Index>(obs_rows.size()));
      Eigen::VectorXi xj(static_cast<Eigen::Index>(obs_rows.size()));
      for (Eigen::Index r = 0; r < xi.size(); ++r) {
        const Eigen::Index src = obs_rows[static_cast<std::size_t>(r)];
        xi(r) = Xcat(src, i);
        xj(r) = Xcat(src, j);
      }
      auto ps_or = ordinal_pair_scores(
          xi, xj, R(i, j), th_by_var[static_cast<std::size_t>(i)],
          th_by_var[static_cast<std::size_t>(j)]);
      if (!ps_or.has_value()) return std::unexpected(ps_or.error());
      const auto& ps = *ps_or;

      Eigen::MatrixXd a21i = Eigen::MatrixXd::Zero(n, ps.threshold_i.cols());
      Eigen::MatrixXd a21j = Eigen::MatrixXd::Zero(n, ps.threshold_j.cols());
      for (Eigen::Index r = 0; r < xi.size(); ++r) {
        const Eigen::Index dst = obs_rows[static_cast<std::size_t>(r)];
        SC_COR(dst, corr_idx) = ps.rho(r);
        a21i.row(dst) = ps.rho(r) * ps.threshold_i.row(r);
        a21j.row(dst) = ps.rho(r) * ps.threshold_j.row(r);
      }

      const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
      const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
      A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
          ps.rho.transpose() * ps.threshold_i;
      A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
          ps.rho.transpose() * ps.threshold_j;

      pairs.push_back(OrdinalObservedPairBread{
          .vi = i,
          .vj = j,
          .a21i = std::move(a21i),
          .a21j = std::move(a21j)});
      ++corr_idx;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(nth) = SC_TH;
  SC.rightCols(ncorr) = SC_COR;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
  const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
  for (Eigen::Index j = 0; j < p; ++j) {
    const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index len = th_len[static_cast<std::size_t>(j)];
    A11.block(start, start, len, len) =
        INNER_TH.block(start, start, len, len);
  }
  auto A11_inv_or = symmetric_inverse_pd(A11, context + " A11");
  if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

  Eigen::VectorXd A22_diag(ncorr);
  for (Eigen::Index k = 0; k < ncorr; ++k) {
    A22_diag(k) = SC_COR.col(k).squaredNorm();
    if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          context + ": singular polychoric score block"));
    }
  }
  const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

  Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
  B_inv.block(0, 0, nth, nth) = *A11_inv_or;
  B_inv.block(nth, 0, ncorr, nth).noalias() =
      -A22_inv * A21 * (*A11_inv_or);
  B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

  Eigen::MatrixXd Gamma =
      static_cast<double>(n) * B_inv * INNER * B_inv.transpose();
  Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();

  return OrdinalObservedGammaAssembly{
      .scores = std::move(SC),
      .bread_inv = std::move(B_inv),
      .gamma = std::move(Gamma),
      .th_start = std::move(th_start),
      .th_len = std::move(th_len),
      .pairs = std::move(pairs),
      .nth = nth,
      .ncorr = ncorr,
      .mdim = mdim};
}

static post_expected<Eigen::MatrixXd> observed_gamma_at_kappa(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R) {
  auto assm_or = ordinal_observed_gamma_assembly_at_kappa(
      Xcat, levels, thresholds, R, "observed_gamma_at_kappa");
  if (!assm_or.has_value()) return std::unexpected(assm_or.error());
  return std::move(assm_or->gamma);
}

}  // namespace

post_expected<Eigen::MatrixXd>
ordinal_gamma_diag_jacobian_fd(const Eigen::MatrixXi& Xcat,
                               const std::vector<std::int32_t>& levels,
                               const Eigen::VectorXd& thresholds,
                               const Eigen::MatrixXd& R, double h_rel) {
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<std::pair<Eigen::Index, Eigen::Index>> pair_of(
      static_cast<std::size_t>(ncorr));
  {
    Eigen::Index ci = 0;
    for (Eigen::Index j = 0; j < p; ++j)
      for (Eigen::Index i = j + 1; i < p; ++i)
        pair_of[static_cast<std::size_t>(ci++)] = {i, j};
  }

  Eigen::MatrixXd D(mdim, mdim);  // D(k,l) = ∂Γ̂_kk/∂κ_l
  for (Eigen::Index l = 0; l < mdim; ++l) {
    Eigen::VectorXd th_p = thresholds, th_m = thresholds;
    Eigen::MatrixXd R_p = R, R_m = R;
    double h;
    if (l < nth) {
      h = h_rel * std::max(1.0, std::abs(thresholds(l)));
      th_p(l) += h;
      th_m(l) -= h;
    } else {
      const auto [ii, jj] = pair_of[static_cast<std::size_t>(l - nth)];
      const double base = R(ii, jj);
      h = h_rel * std::max(1.0, std::abs(base));
      R_p(ii, jj) = R_p(jj, ii) = base + h;
      R_m(ii, jj) = R_m(jj, ii) = base - h;
    }
    auto gp = gamma_diag_at_kappa(Xcat, levels, th_p, R_p);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = gamma_diag_at_kappa(Xcat, levels, th_m, R_m);
    if (!gm.has_value()) return std::unexpected(gm.error());
    D.col(l) = (*gp - *gm) / (2.0 * h);
  }
  return D;
}

post_expected<Eigen::MatrixXd>
ordinal_gamma_jacobian_fd(const Eigen::MatrixXi& Xcat,
                          const std::vector<std::int32_t>& levels,
                          const Eigen::VectorXd& thresholds,
                          const Eigen::MatrixXd& R,
                          double h_rel) {
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<std::pair<Eigen::Index, Eigen::Index>> pair_of(
      static_cast<std::size_t>(ncorr));
  {
    Eigen::Index ci = 0;
    for (Eigen::Index j = 0; j < p; ++j)
      for (Eigen::Index i = j + 1; i < p; ++i)
        pair_of[static_cast<std::size_t>(ci++)] = {i, j};
  }

  Eigen::MatrixXd D(mdim * mdim, mdim);
  for (Eigen::Index l = 0; l < mdim; ++l) {
    Eigen::VectorXd th_p = thresholds, th_m = thresholds;
    Eigen::MatrixXd R_p = R, R_m = R;
    double h;
    if (l < nth) {
      h = h_rel * std::max(1.0, std::abs(thresholds(l)));
      th_p(l) += h;
      th_m(l) -= h;
    } else {
      const auto [ii, jj] = pair_of[static_cast<std::size_t>(l - nth)];
      const double base = R(ii, jj);
      h = h_rel * std::max(1.0, std::abs(base));
      R_p(ii, jj) = R_p(jj, ii) = base + h;
      R_m(ii, jj) = R_m(jj, ii) = base - h;
    }
    auto gp = gamma_at_kappa(Xcat, levels, th_p, R_p);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = gamma_at_kappa(Xcat, levels, th_m, R_m);
    if (!gm.has_value()) return std::unexpected(gm.error());
    Eigen::MatrixXd dG = (*gp - *gm) / (2.0 * h);
    D.col(l) = Eigen::Map<const Eigen::VectorXd>(dG.data(), dG.size());
  }
  return D;
}

post_expected<Eigen::MatrixXd>
ordinal_observed_gamma_diag_jacobian_fd(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R,
    double h_rel) {
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<std::pair<Eigen::Index, Eigen::Index>> pair_of(
      static_cast<std::size_t>(ncorr));
  {
    Eigen::Index ci = 0;
    for (Eigen::Index j = 0; j < p; ++j)
      for (Eigen::Index i = j + 1; i < p; ++i)
        pair_of[static_cast<std::size_t>(ci++)] = {i, j};
  }

  Eigen::MatrixXd D(mdim, mdim);
  for (Eigen::Index l = 0; l < mdim; ++l) {
    Eigen::VectorXd th_p = thresholds, th_m = thresholds;
    Eigen::MatrixXd R_p = R, R_m = R;
    double h;
    if (l < nth) {
      h = h_rel * std::max(1.0, std::abs(thresholds(l)));
      th_p(l) += h;
      th_m(l) -= h;
    } else {
      const auto [ii, jj] = pair_of[static_cast<std::size_t>(l - nth)];
      const double base = R(ii, jj);
      h = h_rel * std::max(1.0, std::abs(base));
      R_p(ii, jj) = R_p(jj, ii) = base + h;
      R_m(ii, jj) = R_m(jj, ii) = base - h;
    }
    auto gp = observed_gamma_at_kappa(Xcat, levels, th_p, R_p);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = observed_gamma_at_kappa(Xcat, levels, th_m, R_m);
    if (!gm.has_value()) return std::unexpected(gm.error());
    D.col(l) = (gp->diagonal() - gm->diagonal()) / (2.0 * h);
  }
  return D;
}

post_expected<Eigen::MatrixXd>
ordinal_observed_gamma_jacobian_fd(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R,
    double h_rel) {
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<std::pair<Eigen::Index, Eigen::Index>> pair_of(
      static_cast<std::size_t>(ncorr));
  {
    Eigen::Index ci = 0;
    for (Eigen::Index j = 0; j < p; ++j)
      for (Eigen::Index i = j + 1; i < p; ++i)
        pair_of[static_cast<std::size_t>(ci++)] = {i, j};
  }

  Eigen::MatrixXd D(mdim * mdim, mdim);
  for (Eigen::Index l = 0; l < mdim; ++l) {
    Eigen::VectorXd th_p = thresholds, th_m = thresholds;
    Eigen::MatrixXd R_p = R, R_m = R;
    double h;
    if (l < nth) {
      h = h_rel * std::max(1.0, std::abs(thresholds(l)));
      th_p(l) += h;
      th_m(l) -= h;
    } else {
      const auto [ii, jj] = pair_of[static_cast<std::size_t>(l - nth)];
      const double base = R(ii, jj);
      h = h_rel * std::max(1.0, std::abs(base));
      R_p(ii, jj) = R_p(jj, ii) = base + h;
      R_m(ii, jj) = R_m(jj, ii) = base - h;
    }
    auto gp = observed_gamma_at_kappa(Xcat, levels, th_p, R_p);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = observed_gamma_at_kappa(Xcat, levels, th_m, R_m);
    if (!gm.has_value()) return std::unexpected(gm.error());
    Eigen::MatrixXd dG = (*gp - *gm) / (2.0 * h);
    D.col(l) = Eigen::Map<const Eigen::VectorXd>(dG.data(), dG.size());
  }
  return D;
}

post_expected<Eigen::MatrixXd>
ordinal_gamma_diag_data_influence(const Eigen::MatrixXi& Xcat,
                                  const std::vector<std::int32_t>& levels,
                                  const Eigen::VectorXd& thresholds,
                                  const Eigen::MatrixXd& R) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::Index> th_len(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  Eigen::Index off = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    const std::size_t jz = static_cast<std::size_t>(j);
    th_start[jz] = off;
    const Eigen::Index kj = static_cast<Eigen::Index>(levels[jz] - 1);
    th_len[jz] = kj;
    th_by_var[jz] = thresholds.segment(off, kj);
    off += kj;
  }

  Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const int c = Xcat(r, j);
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }

  // Pair loop: correlation scores, A21, and per-case A21 contributions
  // (rho_score · bivariate threshold score) for each pair's two variables.
  Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
  Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
  struct PairBiv { Eigen::Index vi, vj; Eigen::MatrixXd a21i, a21j; };
  std::vector<PairBiv> pairs;
  pairs.reserve(static_cast<std::size_t>(ncorr));
  Eigen::Index corr_idx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      auto ps_or = ordinal_pair_scores(Xcat.col(i), Xcat.col(j), R(i, j),
          th_by_var[static_cast<std::size_t>(i)],
          th_by_var[static_cast<std::size_t>(j)]);
      if (!ps_or.has_value()) return std::unexpected(ps_or.error());
      const auto& ps = *ps_or;
      SC_COR.col(corr_idx) = ps.rho;
      const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
      const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
      A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
          ps.rho.transpose() * ps.threshold_i;
      A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
          ps.rho.transpose() * ps.threshold_j;
      PairBiv pb;
      pb.vi = i;
      pb.vj = j;
      pb.a21i = ps.rho.asDiagonal() * ps.threshold_i;  // n × th_len[i]
      pb.a21j = ps.rho.asDiagonal() * ps.threshold_j;
      pairs.push_back(std::move(pb));
      ++corr_idx;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(nth) = SC_TH;
  SC.rightCols(ncorr) = SC_COR;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
  const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
  for (Eigen::Index j = 0; j < p; ++j) {
    const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index len = th_len[static_cast<std::size_t>(j)];
    A11.block(start, start, len, len) = INNER_TH.block(start, start, len, len);
  }
  auto A11_inv_or = symmetric_inverse_pd(A11, "gamma_data_influence A11");
  if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());
  Eigen::VectorXd A22_diag(ncorr);
  for (Eigen::Index k = 0; k < ncorr; ++k) {
    A22_diag(k) = SC_COR.col(k).squaredNorm();
    if (!(A22_diag(k) > 0.0))
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "gamma_data_influence: singular polychoric score block"));
  }
  const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

  Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
  B_inv.block(0, 0, nth, nth) = *A11_inv_or;
  B_inv.block(nth, 0, ncorr, nth).noalias() = -A22_inv * A21 * (*A11_inv_or);
  B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

  Eigen::MatrixXd Gam = static_cast<double>(n) * B_inv * INNER * B_inv.transpose();
  Gam = 0.5 * (Gam + Gam.transpose()).eval();
  const Eigen::MatrixXd G = static_cast<double>(n) * SC * B_inv.transpose();  // g_i rows

  Eigen::MatrixXd IFG(n, mdim);
  for (Eigen::Index ii = 0; ii < n; ++ii) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index v = 0; v < p; ++v) {
      const Eigen::Index s = th_start[static_cast<std::size_t>(v)];
      const Eigen::Index l = th_len[static_cast<std::size_t>(v)];
      const Eigen::VectorXd sv = SC_TH.row(ii).segment(s, l).transpose();
      bi.block(s, s, l, l) = sv * sv.transpose();
    }
    for (Eigen::Index a = 0; a < ncorr; ++a)
      bi(nth + a, nth + a) = SC_COR(ii, a) * SC_COR(ii, a);
    for (Eigen::Index a = 0; a < ncorr; ++a) {
      const PairBiv& pb = pairs[static_cast<std::size_t>(a)];
      bi.block(nth + a, th_start[static_cast<std::size_t>(pb.vi)], 1,
               th_len[static_cast<std::size_t>(pb.vi)]) = pb.a21i.row(ii);
      bi.block(nth + a, th_start[static_cast<std::size_t>(pb.vj)], 1,
               th_len[static_cast<std::size_t>(pb.vj)]) = pb.a21j.row(ii);
    }
    const Eigen::MatrixXd M1 = B_inv * bi * Gam;
    const Eigen::MatrixXd M2 = Gam * bi.transpose() * B_inv.transpose();
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double gik = G(ii, k);
      IFG(ii, k) = gik * gik + Gam(k, k) -
                   static_cast<double>(n) * (M1(k, k) + M2(k, k));
    }
  }
  return IFG;
}

post_expected<Eigen::MatrixXd>
ordinal_gamma_data_influence(const Eigen::MatrixXi& Xcat,
                             const std::vector<std::int32_t>& levels,
                             const Eigen::VectorXd& thresholds,
                             const Eigen::MatrixXd& R) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  Eigen::Index nth = 0;
  for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
  const Eigen::Index ncorr = p * (p - 1) / 2;
  const Eigen::Index mdim = nth + ncorr;

  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::Index> th_len(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  Eigen::Index off = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    const std::size_t jz = static_cast<std::size_t>(j);
    th_start[jz] = off;
    const Eigen::Index kj = static_cast<Eigen::Index>(levels[jz] - 1);
    th_len[jz] = kj;
    th_by_var[jz] = thresholds.segment(off, kj);
    off += kj;
  }

  Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const int c = Xcat(r, j);
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }

  Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
  Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
  struct PairBiv { Eigen::Index vi, vj; Eigen::MatrixXd a21i, a21j; };
  std::vector<PairBiv> pairs;
  pairs.reserve(static_cast<std::size_t>(ncorr));
  Eigen::Index corr_idx = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      auto ps_or = ordinal_pair_scores(Xcat.col(i), Xcat.col(j), R(i, j),
          th_by_var[static_cast<std::size_t>(i)],
          th_by_var[static_cast<std::size_t>(j)]);
      if (!ps_or.has_value()) return std::unexpected(ps_or.error());
      const auto& ps = *ps_or;
      SC_COR.col(corr_idx) = ps.rho;
      const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
      const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
      A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
          ps.rho.transpose() * ps.threshold_i;
      A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
          ps.rho.transpose() * ps.threshold_j;
      PairBiv pb;
      pb.vi = i;
      pb.vj = j;
      pb.a21i = ps.rho.asDiagonal() * ps.threshold_i;
      pb.a21j = ps.rho.asDiagonal() * ps.threshold_j;
      pairs.push_back(std::move(pb));
      ++corr_idx;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(nth) = SC_TH;
  SC.rightCols(ncorr) = SC_COR;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
  const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
  for (Eigen::Index j = 0; j < p; ++j) {
    const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index len = th_len[static_cast<std::size_t>(j)];
    A11.block(start, start, len, len) = INNER_TH.block(start, start, len, len);
  }
  auto A11_inv_or = symmetric_inverse_pd(A11, "gamma_data_influence A11");
  if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());
  Eigen::VectorXd A22_diag(ncorr);
  for (Eigen::Index k = 0; k < ncorr; ++k) {
    A22_diag(k) = SC_COR.col(k).squaredNorm();
    if (!(A22_diag(k) > 0.0))
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "gamma_data_influence: singular polychoric score block"));
  }
  const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

  Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
  B_inv.block(0, 0, nth, nth) = *A11_inv_or;
  B_inv.block(nth, 0, ncorr, nth).noalias() = -A22_inv * A21 * (*A11_inv_or);
  B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

  Eigen::MatrixXd Gam =
      static_cast<double>(n) * B_inv * INNER * B_inv.transpose();
  Gam = 0.5 * (Gam + Gam.transpose()).eval();
  const Eigen::MatrixXd G = static_cast<double>(n) * SC * B_inv.transpose();

  Eigen::MatrixXd IFG(n, mdim * mdim);
  for (Eigen::Index ii = 0; ii < n; ++ii) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index v = 0; v < p; ++v) {
      const Eigen::Index s = th_start[static_cast<std::size_t>(v)];
      const Eigen::Index l = th_len[static_cast<std::size_t>(v)];
      const Eigen::VectorXd sv = SC_TH.row(ii).segment(s, l).transpose();
      bi.block(s, s, l, l) = sv * sv.transpose();
    }
    for (Eigen::Index a = 0; a < ncorr; ++a) {
      bi(nth + a, nth + a) = SC_COR(ii, a) * SC_COR(ii, a);
    }
    for (Eigen::Index a = 0; a < ncorr; ++a) {
      const PairBiv& pb = pairs[static_cast<std::size_t>(a)];
      bi.block(nth + a, th_start[static_cast<std::size_t>(pb.vi)], 1,
               th_len[static_cast<std::size_t>(pb.vi)]) = pb.a21i.row(ii);
      bi.block(nth + a, th_start[static_cast<std::size_t>(pb.vj)], 1,
               th_len[static_cast<std::size_t>(pb.vj)]) = pb.a21j.row(ii);
    }
    const Eigen::MatrixXd M1 = B_inv * bi * Gam;
    const Eigen::MatrixXd M2 = Gam * bi.transpose() * B_inv.transpose();
    Eigen::MatrixXd Mi =
        G.row(ii).transpose() * G.row(ii) + Gam -
        static_cast<double>(n) * (M1 + M2);
    Mi = 0.5 * (Mi + Mi.transpose()).eval();
    const Eigen::Map<const Eigen::VectorXd> vec(Mi.data(), Mi.size());
    IFG.row(ii) = vec.transpose();
  }
  return IFG;
}

post_expected<Eigen::MatrixXd>
ordinal_observed_gamma_diag_data_influence(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R) {
  auto assm_or = ordinal_observed_gamma_assembly_at_kappa(
      Xcat, levels, thresholds, R,
      "ordinal_observed_gamma_diag_data_influence");
  if (!assm_or.has_value()) return std::unexpected(assm_or.error());
  const auto& assm = *assm_or;
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  const Eigen::MatrixXd G =
      static_cast<double>(n) * assm.scores * assm.bread_inv.transpose();

  Eigen::MatrixXd IFG(n, assm.mdim);
  for (Eigen::Index ii = 0; ii < n; ++ii) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(assm.mdim, assm.mdim);
    for (Eigen::Index v = 0; v < p; ++v) {
      const Eigen::Index s = assm.th_start[static_cast<std::size_t>(v)];
      const Eigen::Index l = assm.th_len[static_cast<std::size_t>(v)];
      const Eigen::VectorXd sv = assm.scores.row(ii).segment(s, l).transpose();
      bi.block(s, s, l, l) = sv * sv.transpose();
    }
    for (Eigen::Index a = 0; a < assm.ncorr; ++a) {
      bi(assm.nth + a, assm.nth + a) =
          assm.scores(ii, assm.nth + a) * assm.scores(ii, assm.nth + a);
      const auto& pb = assm.pairs[static_cast<std::size_t>(a)];
      bi.block(assm.nth + a,
               assm.th_start[static_cast<std::size_t>(pb.vi)], 1,
               assm.th_len[static_cast<std::size_t>(pb.vi)]) =
          pb.a21i.row(ii);
      bi.block(assm.nth + a,
               assm.th_start[static_cast<std::size_t>(pb.vj)], 1,
               assm.th_len[static_cast<std::size_t>(pb.vj)]) =
          pb.a21j.row(ii);
    }
    const Eigen::MatrixXd M1 = assm.bread_inv * bi * assm.gamma;
    const Eigen::MatrixXd M2 =
        assm.gamma * bi.transpose() * assm.bread_inv.transpose();
    for (Eigen::Index k = 0; k < assm.mdim; ++k) {
      const double gik = G(ii, k);
      IFG(ii, k) = gik * gik + assm.gamma(k, k) -
                   static_cast<double>(n) * (M1(k, k) + M2(k, k));
    }
  }
  return IFG;
}

post_expected<Eigen::MatrixXd>
ordinal_observed_gamma_data_influence(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R) {
  auto assm_or = ordinal_observed_gamma_assembly_at_kappa(
      Xcat, levels, thresholds, R,
      "ordinal_observed_gamma_data_influence");
  if (!assm_or.has_value()) return std::unexpected(assm_or.error());
  const auto& assm = *assm_or;
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index p = Xcat.cols();
  const Eigen::MatrixXd G =
      static_cast<double>(n) * assm.scores * assm.bread_inv.transpose();

  Eigen::MatrixXd IFG(n, assm.mdim * assm.mdim);
  for (Eigen::Index ii = 0; ii < n; ++ii) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(assm.mdim, assm.mdim);
    for (Eigen::Index v = 0; v < p; ++v) {
      const Eigen::Index s = assm.th_start[static_cast<std::size_t>(v)];
      const Eigen::Index l = assm.th_len[static_cast<std::size_t>(v)];
      const Eigen::VectorXd sv = assm.scores.row(ii).segment(s, l).transpose();
      bi.block(s, s, l, l) = sv * sv.transpose();
    }
    for (Eigen::Index a = 0; a < assm.ncorr; ++a) {
      bi(assm.nth + a, assm.nth + a) =
          assm.scores(ii, assm.nth + a) * assm.scores(ii, assm.nth + a);
      const auto& pb = assm.pairs[static_cast<std::size_t>(a)];
      bi.block(assm.nth + a,
               assm.th_start[static_cast<std::size_t>(pb.vi)], 1,
               assm.th_len[static_cast<std::size_t>(pb.vi)]) =
          pb.a21i.row(ii);
      bi.block(assm.nth + a,
               assm.th_start[static_cast<std::size_t>(pb.vj)], 1,
               assm.th_len[static_cast<std::size_t>(pb.vj)]) =
          pb.a21j.row(ii);
    }
    const Eigen::MatrixXd M1 = assm.bread_inv * bi * assm.gamma;
    const Eigen::MatrixXd M2 =
        assm.gamma * bi.transpose() * assm.bread_inv.transpose();
    Eigen::MatrixXd Mi =
        G.row(ii).transpose() * G.row(ii) + assm.gamma -
        static_cast<double>(n) * (M1 + M2);
    Mi = 0.5 * (Mi + Mi.transpose()).eval();
    const Eigen::Map<const Eigen::VectorXd> vec(Mi.data(), Mi.size());
    IFG.row(ii) = vec.transpose();
  }
  return IFG;
}

namespace {

struct MixedGammaAssembly {
  Eigen::MatrixXd scores;
  Eigen::MatrixXd bread_inv;
  Eigen::MatrixXd gamma_theta;
  Eigen::MatrixXd transform;
  Eigen::MatrixXd gamma;
  std::vector<Eigen::MatrixXd> pair_a21_case;
  Eigen::Index nth = 0;
  Eigen::Index n_cont = 0;
  Eigen::Index s1 = 0;
  Eigen::Index n_assoc = 0;
  Eigen::Index mdim = 0;
};

post_expected<MixedGammaAssembly> mixed_gamma_assembly_at_kappa(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R,
    bool keep_case_bread) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 2 || p == 0 || R.rows() != p || R.cols() != p ||
      mean.size() != p || ordered.size() != static_cast<std::size_t>(p) ||
      levels.size() != static_cast<std::size_t>(p)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_gamma_assembly_at_kappa: dimension mismatch"));
  }

  Eigen::Index nth = 0;
  Eigen::Index n_cont = 0;
  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
  std::vector<Eigen::Index> th_len(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
  for (Eigen::Index j = 0; j < p; ++j) {
    const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
    if (oj) {
      const auto lev = levels[static_cast<std::size_t>(j)];
      if (lev < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_gamma_assembly_at_kappa: ordered variable has fewer than "
            "two levels"));
      }
      th_start[static_cast<std::size_t>(j)] = nth;
      th_len[static_cast<std::size_t>(j)] =
          static_cast<Eigen::Index>(lev - 1);
      nth += th_len[static_cast<std::size_t>(j)];
    } else {
      cont_pos[static_cast<std::size_t>(j)] = n_cont++;
      if (!(R(j, j) > 0.0) || !std::isfinite(R(j, j)) ||
          !std::isfinite(mean(j))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_gamma_assembly_at_kappa: continuous variable has invalid "
            "mean/variance"));
      }
    }
  }
  if (thresholds.size() != nth) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_gamma_assembly_at_kappa: threshold length mismatch"));
  }

  Eigen::MatrixXi Xcat = Eigen::MatrixXi::Constant(n, p, -1);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);
  for (Eigen::Index j = 0; j < p; ++j) {
    const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
    if (oj) {
      const int lev = levels[static_cast<std::size_t>(j)];
      for (Eigen::Index r = 0; r < n; ++r) {
        const double v = X(r, j);
        if (!std::isfinite(v) || std::floor(v) != v || v < 1.0 ||
            v > static_cast<double>(lev)) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_gamma_assembly_at_kappa: ordinal data must be complete "
              "1-based categories"));
        }
        Xcat(r, j) = static_cast<int>(v) - 1;
      }
    } else {
      const double sd = std::sqrt(R(j, j));
      for (Eigen::Index r = 0; r < n; ++r) {
        if (!std::isfinite(X(r, j))) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_gamma_assembly_at_kappa: continuous data must be finite"));
        }
      }
      U.col(j) = (X.col(j).array() - mean(j)) / sd;
    }
  }

  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) continue;
    th_by_var[static_cast<std::size_t>(j)] =
        thresholds.segment(th_start[static_cast<std::size_t>(j)],
                           th_len[static_cast<std::size_t>(j)]);
  }

  const Eigen::Index s1 = nth + 2 * n_cont;
  const Eigen::Index n_assoc = p * (p - 1) / 2;
  const Eigen::Index mdim = s1 + n_assoc;

  Eigen::MatrixXd SC1 = Eigen::MatrixXd::Zero(n, s1);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[static_cast<std::size_t>(j)] == 0) continue;
      const int c = Xcat(r, j);
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC1(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) SC1(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) continue;
    const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
    const double v = R(j, j);
    SC1.col(nth + cp) = (X.col(j).array() - mean(j)) / v;
    SC1.col(nth + n_cont + cp) =
        ((X.col(j).array() - mean(j)).square() - v) / (2.0 * v * v);
  }

  const Eigen::MatrixXd INNER1 = SC1.transpose() * SC1;
  Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(s1, s1);
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len = th_len[static_cast<std::size_t>(j)];
      A11.block(start, start, len, len) =
          INNER1.block(start, start, len, len);
    } else {
      const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(j)];
      const Eigen::Index va =
          nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
      A11(mu, mu) = INNER1(mu, mu);
      A11(va, va) = INNER1(va, va);
      A11(mu, va) = A11(va, mu) = INNER1(mu, va);
    }
  }
  auto A11_inv_or = symmetric_inverse_pd(
      A11, "mixed_gamma_assembly_at_kappa A11");
  if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

  Eigen::MatrixXd SC_ASSOC = Eigen::MatrixXd::Zero(n, n_assoc);
  Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(n_assoc, s1);
  Eigen::VectorXd A22_diag = Eigen::VectorXd::Ones(n_assoc);
  std::vector<Eigen::MatrixXd> pair_a21_case;
  if (keep_case_bread) {
    pair_a21_case.reserve(static_cast<std::size_t>(n_assoc));
  }

  Eigen::Index assoc_count = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      Eigen::MatrixXd a21_case;
      if (keep_case_bread) a21_case = Eigen::MatrixXd::Zero(n, s1);

      if (oi && oj) {
        auto ps_or = ordinal_pair_scores(
            Xcat.col(i), Xcat.col(j), R(i, j),
            th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!ps_or.has_value()) return std::unexpected(ps_or.error());
        const auto& ps = *ps_or;
        SC_ASSOC.col(assoc_count) = ps.rho;
        const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
        const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
        A21.block(assoc_count, si, 1, ps.threshold_i.cols()) =
            ps.rho.transpose() * ps.threshold_i;
        A21.block(assoc_count, sj, 1, ps.threshold_j.cols()) =
            ps.rho.transpose() * ps.threshold_j;
        if (keep_case_bread) {
          a21_case.block(0, si, n, ps.threshold_i.cols()) =
              ps.rho.asDiagonal() * ps.threshold_i;
          a21_case.block(0, sj, n, ps.threshold_j.cols()) =
              ps.rho.asDiagonal() * ps.threshold_j;
        }
      } else if (oi || oj) {
        const Eigen::Index o = oi ? i : j;
        const Eigen::Index c = oi ? j : i;
        Eigen::VectorXi cat(n);
        for (Eigen::Index r = 0; r < n; ++r) cat(r) = Xcat(r, o);
        const double sd = std::sqrt(R(c, c));
        const double rho = R(i, j) / sd;
        auto ps_or = polyserial_pair_scores(
            cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
        if (!ps_or.has_value()) return std::unexpected(ps_or.error());
        const auto& ps = *ps_or;
        SC_ASSOC.col(assoc_count) = ps.rho;
        const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
        const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(c)];
        const Eigen::Index va =
            nth + n_cont + cont_pos[static_cast<std::size_t>(c)];
        A21.block(assoc_count, so, 1, ps.thresholds.cols()) =
            ps.rho.transpose() * ps.thresholds;
        A21(assoc_count, mu) = ps.rho.dot(ps.mu_unit) / sd;
        A21(assoc_count, va) = ps.rho.dot(ps.var_unit) / R(c, c);
        if (keep_case_bread) {
          a21_case.block(0, so, n, ps.thresholds.cols()) =
              ps.rho.asDiagonal() * ps.thresholds;
          a21_case.col(mu) =
              (ps.rho.array() * ps.mu_unit.array() / sd).matrix();
          a21_case.col(va) =
              (ps.rho.array() * ps.var_unit.array() / R(c, c)).matrix();
        }
      } else {
        auto sc_or = continuous_pair_normal_scores(
            X.col(i), X.col(j), mean(i), mean(j), R(i, i), R(j, j), R(i, j));
        if (!sc_or.has_value()) return std::unexpected(sc_or.error());
        const Eigen::MatrixXd& S = sc_or->score_contributions;
        const double sdi = std::sqrt(R(i, i));
        const double sdj = std::sqrt(R(j, j));
        const double rho = R(i, j) / (sdi * sdj);
        const Eigen::VectorXd s_rho = sdi * sdj * S.col(4);
        SC_ASSOC.col(assoc_count) = s_rho;
        const Eigen::Index mui = nth + cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index muj = nth + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::Index vai =
            nth + n_cont + cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index vaj =
            nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::VectorXd ch_var_i =
            S.col(2) + (rho * sdj / (2.0 * sdi)) * S.col(4);
        const Eigen::VectorXd ch_var_j =
            S.col(3) + (rho * sdi / (2.0 * sdj)) * S.col(4);
        A21(assoc_count, mui) = s_rho.dot(S.col(0));
        A21(assoc_count, muj) = s_rho.dot(S.col(1));
        A21(assoc_count, vai) = s_rho.dot(ch_var_i);
        A21(assoc_count, vaj) = s_rho.dot(ch_var_j);
        if (keep_case_bread) {
          a21_case.col(mui) = (s_rho.array() * S.col(0).array()).matrix();
          a21_case.col(muj) = (s_rho.array() * S.col(1).array()).matrix();
          a21_case.col(vai) = (s_rho.array() * ch_var_i.array()).matrix();
          a21_case.col(vaj) = (s_rho.array() * ch_var_j.array()).matrix();
        }
      }

      A22_diag(assoc_count) = SC_ASSOC.col(assoc_count).squaredNorm();
      if (A22_diag(assoc_count) <= 1e-12 ||
          !std::isfinite(A22_diag(assoc_count))) {
        A22_diag(assoc_count) = 1.0;
      }
      if (keep_case_bread) pair_a21_case.push_back(std::move(a21_case));
      ++assoc_count;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(s1) = SC1;
  SC.rightCols(n_assoc) = SC_ASSOC;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
  Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
  B_inv.block(0, 0, s1, s1) = *A11_inv_or;
  B_inv.block(s1, 0, n_assoc, s1).noalias() =
      -A22_inv * A21 * (*A11_inv_or);
  B_inv.block(s1, s1, n_assoc, n_assoc) = A22_inv;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(mdim, mdim);
  Eigen::Index row = 0;
  for (Eigen::Index k = 0; k < nth; ++k) H(row++, k) = 1.0;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) {
      H(row++, nth + cont_pos[static_cast<std::size_t>(j)]) = -1.0;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) {
      H(row++, nth + n_cont + cont_pos[static_cast<std::size_t>(j)]) = 1.0;
    }
  }
  assoc_count = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      if (oi && oj) {
        H(row, s1 + assoc_count) = 1.0;
      } else if (oi || oj) {
        const Eigen::Index c = oi ? j : i;
        const double sd = std::sqrt(R(c, c));
        const double rho = R(i, j) / sd;
        H(row, s1 + assoc_count) = sd;
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(c)]) =
            rho / (2.0 * sd);
      } else {
        const double sdi = std::sqrt(R(i, i));
        const double sdj = std::sqrt(R(j, j));
        const double rho = R(i, j) / (sdi * sdj);
        H(row, s1 + assoc_count) = sdi * sdj;
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(i)]) =
            rho * sdj / (2.0 * sdi);
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(j)]) =
            rho * sdi / (2.0 * sdj);
      }
      ++row;
      ++assoc_count;
    }
  }

  MixedGammaAssembly out;
  out.scores = std::move(SC);
  out.bread_inv = std::move(B_inv);
  out.gamma_theta =
      static_cast<double>(n) * out.bread_inv * INNER * out.bread_inv.transpose();
  out.gamma_theta =
      0.5 * (out.gamma_theta + out.gamma_theta.transpose()).eval();
  out.transform = std::move(H);
  out.gamma = out.transform * out.gamma_theta * out.transform.transpose();
  out.gamma = 0.5 * (out.gamma + out.gamma.transpose()).eval();
  out.pair_a21_case = std::move(pair_a21_case);
  out.nth = nth;
  out.n_cont = n_cont;
  out.s1 = s1;
  out.n_assoc = n_assoc;
  out.mdim = mdim;
  return out;
}

post_expected<MixedGammaAssembly> mixed_observed_gamma_assembly_at_kappa(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R,
    bool keep_case_bread) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 2 || p == 0 || R.rows() != p || R.cols() != p ||
      mean.size() != p || ordered.size() != static_cast<std::size_t>(p) ||
      levels.size() != static_cast<std::size_t>(p)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_observed_gamma_assembly_at_kappa: dimension mismatch"));
  }

  Eigen::Index nth = 0;
  Eigen::Index n_cont = 0;
  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
  std::vector<Eigen::Index> th_len(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
  for (Eigen::Index j = 0; j < p; ++j) {
    const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
    if (oj) {
      const auto lev = levels[static_cast<std::size_t>(j)];
      if (lev < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_observed_gamma_assembly_at_kappa: ordered variable has "
            "fewer than two levels"));
      }
      th_start[static_cast<std::size_t>(j)] = nth;
      th_len[static_cast<std::size_t>(j)] =
          static_cast<Eigen::Index>(lev - 1);
      nth += th_len[static_cast<std::size_t>(j)];
    } else {
      cont_pos[static_cast<std::size_t>(j)] = n_cont++;
      if (!(R(j, j) > 0.0) || !std::isfinite(R(j, j)) ||
          !std::isfinite(mean(j))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_observed_gamma_assembly_at_kappa: continuous variable has "
            "invalid mean/variance"));
      }
    }
  }
  if (thresholds.size() != nth) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_observed_gamma_assembly_at_kappa: threshold length mismatch"));
  }

  Eigen::MatrixXi Xcat = Eigen::MatrixXi::Constant(n, p, -1);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);
  for (Eigen::Index j = 0; j < p; ++j) {
    const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
    if (oj) {
      const int lev = levels[static_cast<std::size_t>(j)];
      for (Eigen::Index r = 0; r < n; ++r) {
        const double v = X(r, j);
        if (!std::isfinite(v)) continue;
        if (std::floor(v) != v || v < 1.0 ||
            v > static_cast<double>(lev)) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_observed_gamma_assembly_at_kappa: observed ordinal data "
              "must be 1-based categories within the fitted levels"));
        }
        Xcat(r, j) = static_cast<int>(v) - 1;
      }
    } else {
      const double sd = std::sqrt(R(j, j));
      for (Eigen::Index r = 0; r < n; ++r) {
        if (std::isfinite(X(r, j))) U(r, j) = (X(r, j) - mean(j)) / sd;
      }
    }
  }

  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) continue;
    th_by_var[static_cast<std::size_t>(j)] =
        thresholds.segment(th_start[static_cast<std::size_t>(j)],
                           th_len[static_cast<std::size_t>(j)]);
  }

  const Eigen::Index s1 = nth + 2 * n_cont;
  const Eigen::Index n_assoc = p * (p - 1) / 2;
  const Eigen::Index mdim = s1 + n_assoc;

  Eigen::MatrixXd SC1 = Eigen::MatrixXd::Zero(n, s1);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[static_cast<std::size_t>(j)] == 0) continue;
      const int c = Xcat(r, j);
      if (c < 0) continue;
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC1(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) SC1(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) continue;
    const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
    const double v = R(j, j);
    for (Eigen::Index r = 0; r < n; ++r) {
      if (!std::isfinite(X(r, j))) continue;
      const double c = X(r, j) - mean(j);
      SC1(r, nth + cp) = c / v;
      SC1(r, nth + n_cont + cp) = (c * c - v) / (2.0 * v * v);
    }
  }

  const Eigen::MatrixXd INNER1 = SC1.transpose() * SC1;
  Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(s1, s1);
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len = th_len[static_cast<std::size_t>(j)];
      A11.block(start, start, len, len) =
          INNER1.block(start, start, len, len);
    } else {
      const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(j)];
      const Eigen::Index va =
          nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
      A11(mu, mu) = INNER1(mu, mu);
      A11(va, va) = INNER1(va, va);
      A11(mu, va) = A11(va, mu) = INNER1(mu, va);
    }
  }
  auto A11_inv_or = symmetric_inverse_pd(
      A11, "mixed_observed_gamma_assembly_at_kappa A11");
  if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

  Eigen::MatrixXd SC_ASSOC = Eigen::MatrixXd::Zero(n, n_assoc);
  Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(n_assoc, s1);
  Eigen::VectorXd A22_diag = Eigen::VectorXd::Ones(n_assoc);
  std::vector<Eigen::MatrixXd> pair_a21_case;
  if (keep_case_bread) {
    pair_a21_case.reserve(static_cast<std::size_t>(n_assoc));
  }

  Eigen::Index assoc_count = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      std::vector<Eigen::Index> rows;
      rows.reserve(static_cast<std::size_t>(n));
      for (Eigen::Index r = 0; r < n; ++r) {
        const bool i_obs = oi ? Xcat(r, i) >= 0 : std::isfinite(X(r, i));
        const bool j_obs = oj ? Xcat(r, j) >= 0 : std::isfinite(X(r, j));
        if (i_obs && j_obs) rows.push_back(r);
      }
      if (rows.size() < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_observed_gamma_assembly_at_kappa: pair has fewer than two "
            "observed rows"));
      }

      const Eigen::Index np = static_cast<Eigen::Index>(rows.size());
      Eigen::MatrixXd a21_case;
      if (keep_case_bread) a21_case = Eigen::MatrixXd::Zero(n, s1);

      if (oi && oj) {
        Eigen::VectorXi xi(np);
        Eigen::VectorXi xj(np);
        for (Eigen::Index r = 0; r < np; ++r) {
          const Eigen::Index src = rows[static_cast<std::size_t>(r)];
          xi(r) = Xcat(src, i);
          xj(r) = Xcat(src, j);
        }
        auto ps_or = ordinal_pair_scores(
            xi, xj, R(i, j), th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!ps_or.has_value()) return std::unexpected(ps_or.error());
        const auto& ps = *ps_or;
        for (Eigen::Index r = 0; r < np; ++r) {
          SC_ASSOC(rows[static_cast<std::size_t>(r)], assoc_count) =
              ps.rho(r);
        }
        const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
        const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
        A21.block(assoc_count, si, 1, ps.threshold_i.cols()) =
            ps.rho.transpose() * ps.threshold_i;
        A21.block(assoc_count, sj, 1, ps.threshold_j.cols()) =
            ps.rho.transpose() * ps.threshold_j;
        if (keep_case_bread) {
          for (Eigen::Index r = 0; r < np; ++r) {
            const Eigen::Index dst = rows[static_cast<std::size_t>(r)];
            a21_case.block(dst, si, 1, ps.threshold_i.cols()) =
                ps.rho(r) * ps.threshold_i.row(r);
            a21_case.block(dst, sj, 1, ps.threshold_j.cols()) =
                ps.rho(r) * ps.threshold_j.row(r);
          }
        }
      } else if (oi || oj) {
        const Eigen::Index o = oi ? i : j;
        const Eigen::Index c = oi ? j : i;
        Eigen::VectorXi cat(np);
        Eigen::VectorXd u(np);
        for (Eigen::Index r = 0; r < np; ++r) {
          const Eigen::Index src = rows[static_cast<std::size_t>(r)];
          cat(r) = Xcat(src, o);
          u(r) = U(src, c);
        }
        const double sd = std::sqrt(R(c, c));
        const double rho = R(i, j) / sd;
        auto ps_or = polyserial_pair_scores(
            cat, u, rho, th_by_var[static_cast<std::size_t>(o)]);
        if (!ps_or.has_value()) return std::unexpected(ps_or.error());
        const auto& ps = *ps_or;
        for (Eigen::Index r = 0; r < np; ++r) {
          SC_ASSOC(rows[static_cast<std::size_t>(r)], assoc_count) =
              ps.rho(r);
        }
        const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
        const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(c)];
        const Eigen::Index va =
            nth + n_cont + cont_pos[static_cast<std::size_t>(c)];
        A21.block(assoc_count, so, 1, ps.thresholds.cols()) =
            ps.rho.transpose() * ps.thresholds;
        A21(assoc_count, mu) = ps.rho.dot(ps.mu_unit) / sd;
        A21(assoc_count, va) = ps.rho.dot(ps.var_unit) / R(c, c);
        if (keep_case_bread) {
          for (Eigen::Index r = 0; r < np; ++r) {
            const Eigen::Index dst = rows[static_cast<std::size_t>(r)];
            a21_case.block(dst, so, 1, ps.thresholds.cols()) =
                ps.rho(r) * ps.thresholds.row(r);
            a21_case(dst, mu) = ps.rho(r) * ps.mu_unit(r) / sd;
            a21_case(dst, va) = ps.rho(r) * ps.var_unit(r) / R(c, c);
          }
        }
      } else {
        Eigen::VectorXd xi(np);
        Eigen::VectorXd xj(np);
        for (Eigen::Index r = 0; r < np; ++r) {
          const Eigen::Index src = rows[static_cast<std::size_t>(r)];
          xi(r) = X(src, i);
          xj(r) = X(src, j);
        }
        auto sc_or = continuous_pair_normal_scores(
            xi, xj, mean(i), mean(j), R(i, i), R(j, j), R(i, j));
        if (!sc_or.has_value()) return std::unexpected(sc_or.error());
        const Eigen::MatrixXd& S = sc_or->score_contributions;
        const double sdi = std::sqrt(R(i, i));
        const double sdj = std::sqrt(R(j, j));
        const double rho = R(i, j) / (sdi * sdj);
        const Eigen::VectorXd s_rho = sdi * sdj * S.col(4);
        for (Eigen::Index r = 0; r < np; ++r) {
          SC_ASSOC(rows[static_cast<std::size_t>(r)], assoc_count) =
              s_rho(r);
        }
        const Eigen::Index mui = nth + cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index muj = nth + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::Index vai =
            nth + n_cont + cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index vaj =
            nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::VectorXd ch_var_i =
            S.col(2) + (rho * sdj / (2.0 * sdi)) * S.col(4);
        const Eigen::VectorXd ch_var_j =
            S.col(3) + (rho * sdi / (2.0 * sdj)) * S.col(4);
        A21(assoc_count, mui) = s_rho.dot(S.col(0));
        A21(assoc_count, muj) = s_rho.dot(S.col(1));
        A21(assoc_count, vai) = s_rho.dot(ch_var_i);
        A21(assoc_count, vaj) = s_rho.dot(ch_var_j);
        if (keep_case_bread) {
          for (Eigen::Index r = 0; r < np; ++r) {
            const Eigen::Index dst = rows[static_cast<std::size_t>(r)];
            a21_case(dst, mui) = s_rho(r) * S(r, 0);
            a21_case(dst, muj) = s_rho(r) * S(r, 1);
            a21_case(dst, vai) = s_rho(r) * ch_var_i(r);
            a21_case(dst, vaj) = s_rho(r) * ch_var_j(r);
          }
        }
      }

      A22_diag(assoc_count) = SC_ASSOC.col(assoc_count).squaredNorm();
      if (A22_diag(assoc_count) <= 1e-12 ||
          !std::isfinite(A22_diag(assoc_count))) {
        A22_diag(assoc_count) = 1.0;
      }
      if (keep_case_bread) pair_a21_case.push_back(std::move(a21_case));
      ++assoc_count;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(s1) = SC1;
  SC.rightCols(n_assoc) = SC_ASSOC;
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
  Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
  B_inv.block(0, 0, s1, s1) = *A11_inv_or;
  B_inv.block(s1, 0, n_assoc, s1).noalias() =
      -A22_inv * A21 * (*A11_inv_or);
  B_inv.block(s1, s1, n_assoc, n_assoc) = A22_inv;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(mdim, mdim);
  Eigen::Index row = 0;
  for (Eigen::Index k = 0; k < nth; ++k) H(row++, k) = 1.0;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) {
      H(row++, nth + cont_pos[static_cast<std::size_t>(j)]) = -1.0;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) {
      H(row++, nth + n_cont + cont_pos[static_cast<std::size_t>(j)]) = 1.0;
    }
  }
  assoc_count = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      if (oi && oj) {
        H(row, s1 + assoc_count) = 1.0;
      } else if (oi || oj) {
        const Eigen::Index c = oi ? j : i;
        const double sd = std::sqrt(R(c, c));
        const double rho = R(i, j) / sd;
        H(row, s1 + assoc_count) = sd;
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(c)]) =
            rho / (2.0 * sd);
      } else {
        const double sdi = std::sqrt(R(i, i));
        const double sdj = std::sqrt(R(j, j));
        const double rho = R(i, j) / (sdi * sdj);
        H(row, s1 + assoc_count) = sdi * sdj;
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(i)]) =
            rho * sdj / (2.0 * sdi);
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(j)]) =
            rho * sdi / (2.0 * sdj);
      }
      ++row;
      ++assoc_count;
    }
  }

  MixedGammaAssembly out;
  out.scores = std::move(SC);
  out.bread_inv = std::move(B_inv);
  out.gamma_theta =
      static_cast<double>(n) * out.bread_inv * INNER * out.bread_inv.transpose();
  out.gamma_theta =
      0.5 * (out.gamma_theta + out.gamma_theta.transpose()).eval();
  out.transform = std::move(H);
  out.gamma = out.transform * out.gamma_theta * out.transform.transpose();
  out.gamma = 0.5 * (out.gamma + out.gamma.transpose()).eval();
  out.pair_a21_case = std::move(pair_a21_case);
  out.nth = nth;
  out.n_cont = n_cont;
  out.s1 = s1;
  out.n_assoc = n_assoc;
  out.mdim = mdim;
  return out;
}

}  // namespace

namespace {

post_expected<Eigen::MatrixXd> mixed_gamma_data_influence_impl(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R,
    bool diagonal_only,
    bool observed) {
  auto asm_or = observed
      ? mixed_observed_gamma_assembly_at_kappa(
            X, ordered, levels, thresholds, mean, R, true)
      : mixed_gamma_assembly_at_kappa(
            X, ordered, levels, thresholds, mean, R, true);
  if (!asm_or.has_value()) return std::unexpected(asm_or.error());
  const MixedGammaAssembly& a = *asm_or;
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::MatrixXd Gtheta =
      static_cast<double>(n) * a.scores * a.bread_inv.transpose();
  const Eigen::MatrixXd G = Gtheta * a.transform.transpose();

  Eigen::MatrixXd IFG(
      n, diagonal_only ? a.mdim : static_cast<Eigen::Index>(a.mdim * a.mdim));
  for (Eigen::Index r = 0; r < n; ++r) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(a.mdim, a.mdim);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[static_cast<std::size_t>(j)] != 0) {
        Eigen::Index start = 0;
        for (Eigen::Index v = 0; v < j; ++v) {
          if (ordered[static_cast<std::size_t>(v)] != 0) {
            start += static_cast<Eigen::Index>(
                levels[static_cast<std::size_t>(v)] - 1);
          }
        }
        const Eigen::Index len =
            static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
        const Eigen::VectorXd sv =
            a.scores.row(r).segment(start, len).transpose();
        bi.block(start, start, len, len) = sv * sv.transpose();
      } else {
        Eigen::Index cp = 0;
        for (Eigen::Index v = 0; v < j; ++v) {
          if (ordered[static_cast<std::size_t>(v)] == 0) ++cp;
        }
        const Eigen::Index mu = a.nth + cp;
        const Eigen::Index va = a.nth + a.n_cont + cp;
        bi(mu, mu) = a.scores(r, mu) * a.scores(r, mu);
        bi(mu, va) = a.scores(r, mu) * a.scores(r, va);
        bi(va, mu) = a.scores(r, va) * a.scores(r, mu);
        bi(va, va) = a.scores(r, va) * a.scores(r, va);
      }
    }
    for (Eigen::Index k = 0; k < a.n_assoc; ++k) {
      bi(a.s1 + k, a.s1 + k) =
          a.scores(r, a.s1 + k) * a.scores(r, a.s1 + k);
      bi.block(a.s1 + k, 0, 1, a.s1) =
          a.pair_a21_case[static_cast<std::size_t>(k)].row(r);
    }

    const Eigen::MatrixXd M =
        a.bread_inv * bi * a.gamma_theta +
        a.gamma_theta * bi.transpose() * a.bread_inv.transpose();
    const Eigen::MatrixXd HMHT = a.transform * M * a.transform.transpose();
    Eigen::MatrixXd Mi =
        G.row(r).transpose() * G.row(r) + a.gamma -
        static_cast<double>(n) * HMHT;
    Mi = 0.5 * (Mi + Mi.transpose()).eval();
    if (diagonal_only) {
      IFG.row(r) = Mi.diagonal().transpose();
    } else {
      const Eigen::Map<const Eigen::VectorXd> vec(Mi.data(), Mi.size());
      IFG.row(r) = vec.transpose();
    }
  }
  return IFG;
}

}  // namespace

post_expected<Eigen::MatrixXd>
mixed_gamma_diag_data_influence(const Eigen::MatrixXd& X,
                                const std::vector<std::int32_t>& ordered,
                                const std::vector<std::int32_t>& levels,
                                const Eigen::VectorXd& thresholds,
                                const Eigen::VectorXd& mean,
                                const Eigen::MatrixXd& R) {
  return mixed_gamma_data_influence_impl(
      X, ordered, levels, thresholds, mean, R, true, false);
}

post_expected<Eigen::MatrixXd>
mixed_gamma_data_influence(const Eigen::MatrixXd& X,
                           const std::vector<std::int32_t>& ordered,
                           const std::vector<std::int32_t>& levels,
                           const Eigen::VectorXd& thresholds,
                           const Eigen::VectorXd& mean,
                           const Eigen::MatrixXd& R) {
  return mixed_gamma_data_influence_impl(
      X, ordered, levels, thresholds, mean, R, false, false);
}

post_expected<Eigen::MatrixXd>
mixed_observed_gamma_diag_data_influence(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R) {
  return mixed_gamma_data_influence_impl(
      X, ordered, levels, thresholds, mean, R, true, true);
}

post_expected<Eigen::MatrixXd>
mixed_observed_gamma_data_influence(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R) {
  return mixed_gamma_data_influence_impl(
      X, ordered, levels, thresholds, mean, R, false, true);
}

namespace {

post_expected<Eigen::MatrixXd> mixed_gamma_jacobian_fd_impl(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R,
    double h_rel,
    bool diagonal_only,
    bool observed) {
  auto base_or = observed
      ? mixed_observed_gamma_assembly_at_kappa(
            X, ordered, levels, thresholds, mean, R, false)
      : mixed_gamma_assembly_at_kappa(
            X, ordered, levels, thresholds, mean, R, false);
  if (!base_or.has_value()) return std::unexpected(base_or.error());
  const Eigen::Index p = X.cols();
  const Eigen::Index mdim = base_or->mdim;
  const Eigen::VectorXd kappa = mixed_moment_vector(R, mean, ordered, thresholds);
  if (kappa.size() != mdim) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_gamma_jacobian_fd: mixed moment length mismatch"));
  }

  std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
  Eigen::Index n_cont = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) {
      cont_pos[static_cast<std::size_t>(j)] = n_cont++;
    }
  }

  std::vector<std::pair<Eigen::Index, Eigen::Index>> pairs;
  pairs.reserve(static_cast<std::size_t>(p * (p - 1) / 2));
  for (Eigen::Index j = 0; j < p; ++j)
    for (Eigen::Index i = j + 1; i < p; ++i) pairs.push_back({i, j});

  Eigen::MatrixXd D(
      diagonal_only ? mdim : static_cast<Eigen::Index>(mdim * mdim), mdim);
  for (Eigen::Index l = 0; l < mdim; ++l) {
    Eigen::VectorXd th_p = thresholds, th_m = thresholds;
    Eigen::VectorXd mean_p = mean, mean_m = mean;
    Eigen::MatrixXd R_p = R, R_m = R;
    double h = h_rel * std::max(1.0, std::abs(kappa(l)));
    if (l < base_or->nth) {
      Eigen::Index start = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[static_cast<std::size_t>(j)] == 0) continue;
        const Eigen::Index len =
            static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
        if (l >= start && l < start + len) {
          if (l > start) h = std::min(h, 0.25 * (thresholds(l) - thresholds(l - 1)));
          if (l + 1 < start + len)
            h = std::min(h, 0.25 * (thresholds(l + 1) - thresholds(l)));
          break;
        }
        start += len;
      }
      h = std::max(h, 1e-7);
      th_p(l) += h;
      th_m(l) -= h;
    } else if (l < base_or->nth + n_cont) {
      const Eigen::Index cp = l - base_or->nth;
      Eigen::Index var = -1;
      for (Eigen::Index j = 0; j < p; ++j) {
        if (cont_pos[static_cast<std::size_t>(j)] == cp) {
          var = j;
          break;
        }
      }
      mean_p(var) -= h;
      mean_m(var) += h;
    } else if (l < base_or->nth + 2 * n_cont) {
      const Eigen::Index cp = l - base_or->nth - n_cont;
      Eigen::Index var = -1;
      for (Eigen::Index j = 0; j < p; ++j) {
        if (cont_pos[static_cast<std::size_t>(j)] == cp) {
          var = j;
          break;
        }
      }
      h = std::min(h, 0.25 * R(var, var));
      h = std::max(h, 1e-7);
      R_p(var, var) += h;
      R_m(var, var) -= h;
    } else {
      const auto [i, j] =
          pairs[static_cast<std::size_t>(l - base_or->nth - 2 * n_cont)];
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      double limit = 0.999;
      if (oi != oj) {
        const Eigen::Index c = oi ? j : i;
        limit = 0.999 * std::sqrt(R(c, c));
      } else if (!oi && !oj) {
        limit = 0.999 * std::sqrt(R(i, i) * R(j, j));
      }
      const double margin = limit - std::abs(R(i, j));
      if (margin > 1e-8) h = std::min(h, 0.25 * margin);
      h = std::max(h, 1e-7);
      if (margin > 0.0 && h >= margin) h = 0.5 * margin;
      R_p(i, j) = R_p(j, i) = R(i, j) + h;
      R_m(i, j) = R_m(j, i) = R(i, j) - h;
    }

    auto gp_or = observed
        ? mixed_observed_gamma_assembly_at_kappa(
              X, ordered, levels, th_p, mean_p, R_p, false)
        : mixed_gamma_assembly_at_kappa(
              X, ordered, levels, th_p, mean_p, R_p, false);
    if (!gp_or.has_value()) return std::unexpected(gp_or.error());
    auto gm_or = observed
        ? mixed_observed_gamma_assembly_at_kappa(
              X, ordered, levels, th_m, mean_m, R_m, false)
        : mixed_gamma_assembly_at_kappa(
              X, ordered, levels, th_m, mean_m, R_m, false);
    if (!gm_or.has_value()) return std::unexpected(gm_or.error());
    if (diagonal_only) {
      D.col(l) = (gp_or->gamma.diagonal() - gm_or->gamma.diagonal()) /
                 (2.0 * h);
    } else {
      const Eigen::MatrixXd dG = (gp_or->gamma - gm_or->gamma) / (2.0 * h);
      const Eigen::Map<const Eigen::VectorXd> vec(dG.data(), dG.size());
      D.col(l) = vec;
    }
  }
  return D;
}

}  // namespace

post_expected<Eigen::MatrixXd>
mixed_gamma_diag_jacobian_fd(const Eigen::MatrixXd& X,
                             const std::vector<std::int32_t>& ordered,
                             const std::vector<std::int32_t>& levels,
                             const Eigen::VectorXd& thresholds,
                             const Eigen::VectorXd& mean,
                             const Eigen::MatrixXd& R,
                             double h_rel) {
  return mixed_gamma_jacobian_fd_impl(
      X, ordered, levels, thresholds, mean, R, h_rel, true, false);
}

post_expected<Eigen::MatrixXd>
mixed_gamma_jacobian_fd(const Eigen::MatrixXd& X,
                        const std::vector<std::int32_t>& ordered,
                        const std::vector<std::int32_t>& levels,
                        const Eigen::VectorXd& thresholds,
                        const Eigen::VectorXd& mean,
                        const Eigen::MatrixXd& R,
                        double h_rel) {
  return mixed_gamma_jacobian_fd_impl(
      X, ordered, levels, thresholds, mean, R, h_rel, false, false);
}

post_expected<Eigen::MatrixXd>
mixed_observed_gamma_diag_jacobian_fd(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R,
    double h_rel) {
  return mixed_gamma_jacobian_fd_impl(
      X, ordered, levels, thresholds, mean, R, h_rel, true, true);
}

post_expected<Eigen::MatrixXd>
mixed_observed_gamma_jacobian_fd(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R,
    double h_rel) {
  return mixed_gamma_jacobian_fd_impl(
      X, ordered, levels, thresholds, mean, R, h_rel, false, true);
}

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& Xs,
                                         bool full_wls_weight) {
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_stats_from_integer_data: no data blocks"));
  }

  PairwiseOrdinalStats out;
  out.stats.R.reserve(Xs.size());
  out.stats.thresholds.reserve(Xs.size());
  out.stats.threshold_ov.reserve(Xs.size());
  out.stats.threshold_level.reserve(Xs.size());
  out.stats.NACOV.reserve(Xs.size());
  out.stats.W_dwls.reserve(Xs.size());
  out.stats.W_wls.reserve(Xs.size());
  out.stats.n_obs.reserve(Xs.size());
  out.stats.n_levels.reserve(Xs.size());
  out.block_diagnostics.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2 || p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_stats_from_integer_data: block " + std::to_string(b) +
              " must have at least 2 rows and 1 column"));
    }
    Eigen::MatrixXi Xcat(n, p);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
      int max_level = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        const double v = X(i, j);
        if (!std::isfinite(v) || std::floor(v) != v || v < 1.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                  " has non-integer/non-positive category values"));
        }
        max_level = std::max(max_level, static_cast<int>(v));
      }
      if (max_level < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                " variable " + std::to_string(j) + " has fewer than 2 levels"));
      }
      counts[static_cast<std::size_t>(j)].assign(static_cast<std::size_t>(max_level), 0);
      for (Eigen::Index i = 0; i < n; ++i) {
        const int lvl = static_cast<int>(X(i, j));
        counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(lvl - 1)] += 1;
        Xcat(i, j) = lvl - 1;
      }
      for (int c = 0; c < max_level; ++c) {
        if (counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)] == 0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                  " variable " + std::to_string(j) + " has an empty category"));
        }
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::Index nth = 0;
    for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;
    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    th_ov.reserve(static_cast<std::size_t>(nth));
    th_level.reserve(static_cast<std::size_t>(nth));

    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      th_start[static_cast<std::size_t>(j)] = off;
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
        if (cum <= 0 || cum >= n) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                  " variable " + std::to_string(j) +
                  " has an empty boundary category"));
        }
        const double prob = std::clamp(static_cast<double>(cum) / static_cast<double>(n),
                                       1e-12, 1.0 - 1e-12);
        const double z = normal_quantile(prob);
        th(off) = z;
        th_by_var[static_cast<std::size_t>(j)](c) = z;
        th_ov.push_back(static_cast<std::int32_t>(j));
        th_level.push_back(static_cast<std::int32_t>(c + 1));
        ++off;
      }
    }

    Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index j = 0; j < p; ++j) {
        const int c = Xcat(r, j);
        const auto& thj = th_by_var[static_cast<std::size_t>(j)];
        const double lo = (c == 0) ? -kInf : thj(c - 1);
        const double hi = (c == thj.size()) ? kInf : thj(c);
        const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
        const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
        if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
        if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
      }
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
    Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
    PairwiseOrdinalBlockDiagnostics block_diag;
    block_diag.pair_diagnostics.reserve(static_cast<std::size_t>(ncorr));
    Eigen::Index corr_idx = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const Eigen::VectorXi xi = Xcat.col(i);
        const Eigen::VectorXi xj = Xcat.col(j);
        auto tab_or = ordinal_pair_table(
            xi, xj, levels[static_cast<std::size_t>(i)],
            levels[static_cast<std::size_t>(j)]);
        if (!tab_or.has_value()) return std::unexpected(tab_or.error());
        auto rho_or = fit_ordinal_pair_rho_ml(
            *tab_or, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!rho_or.has_value()) return std::unexpected(rho_or.error());
        const double rho = rho_or->rho;
        Eigen::MatrixXd expected = expected_pair_counts(
            rho_or->adjusted_counts.sum(),
            th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)],
            rho);
        Eigen::MatrixXd residual = rho_or->adjusted_counts - expected;
        Eigen::MatrixXd pearson = Eigen::MatrixXd::Zero(
            residual.rows(), residual.cols());
        for (Eigen::Index pr = 0; pr < residual.rows(); ++pr) {
          for (Eigen::Index pc = 0; pc < residual.cols(); ++pc) {
            pearson(pr, pc) = residual(pr, pc) /
                std::sqrt(std::max(kProbFloor, expected(pr, pc)));
          }
        }
        block_diag.pair_diagnostics.push_back(OrdinalPairDiagnostics{
            .label = OrdinalPairLabel{
                .block = static_cast<std::int32_t>(b),
                .i = static_cast<std::int32_t>(i),
                .j = static_cast<std::int32_t>(j),
                .n_levels_i = levels[static_cast<std::size_t>(i)],
                .n_levels_j = levels[static_cast<std::size_t>(j)]},
            .rho = rho,
            .negloglik = rho_or->negloglik,
            .objective = rho_or->negloglik,
            .score = 0.0,
            .iterations = rho_or->iterations,
            .h_weighted = false,
            .converged = true,
            .hit_lower = rho_or->hit_lower,
            .hit_upper = rho_or->hit_upper,
            .n_obs = static_cast<std::int64_t>(tab_or->sum()),
            .n_missing = 0,
            .ridge_applied = false,
            .ridge = 0.0,
            .shrinkage_applied = false,
            .shrinkage_intensity = 0.0,
            .counts = *tab_or,
            .adjusted_counts = rho_or->adjusted_counts,
            .expected_counts = std::move(expected),
            .residual_counts = std::move(residual),
            .pearson_residuals = std::move(pearson),
            .weights = Eigen::MatrixXd::Ones(tab_or->rows(), tab_or->cols())});
        R(i, j) = R(j, i) = rho;
        auto ps_or = ordinal_pair_scores(
            xi, xj, rho, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!ps_or.has_value()) return std::unexpected(ps_or.error());
        const auto& ps = *ps_or;
        SC_COR.col(corr_idx) = ps.rho;
        const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
        const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
        A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
            ps.rho.transpose() * ps.threshold_i;
        A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
            ps.rho.transpose() * ps.threshold_j;
        ++corr_idx;
      }
    }
    auto r_diag_or = repair_correlation_if_requested(
        R, {}, "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                   " polychoric R");
    if (!r_diag_or.has_value()) return std::unexpected(r_diag_or.error());
    block_diag.min_eigen_r = r_diag_or->min_eigen;
    block_diag.raw_min_eigen_r = r_diag_or->raw_min_eigen;

    Eigen::MatrixXd SC(n, mdim);
    SC.leftCols(nth) = SC_TH;
    SC.rightCols(ncorr) = SC_COR;
    const Eigen::MatrixXd INNER = SC.transpose() * SC;

    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
    const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
    for (Eigen::Index j = 0; j < p; ++j) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len = static_cast<Eigen::Index>(
          levels[static_cast<std::size_t>(j)] - 1);
      A11.block(start, start, len, len) =
          INNER_TH.block(start, start, len, len);
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "ordinal threshold information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::VectorXd A22_diag(ncorr);
    for (Eigen::Index k = 0; k < ncorr; ++k) {
      A22_diag(k) = SC_COR.col(k).squaredNorm();
      if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                " has a singular polychoric score block"));
      }
    }
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
    B_inv.block(0, 0, nth, nth) = *A11_inv_or;
    B_inv.block(nth, 0, ncorr, nth).noalias() =
        -A22_inv * A21 * (*A11_inv_or);
    B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

    Eigen::MatrixXd IF = static_cast<double>(n) * SC * B_inv.transpose();
    Eigen::MatrixXd NACOV = static_cast<double>(n) *
        (B_inv * INNER * B_inv.transpose());
    NACOV = 0.5 * (NACOV + NACOV.transpose());

    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                " has a non-positive NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    // The full-WLS weight is the NACOV inverse; it is needed only for full WLS
    // fitting, not for DWLS (which uses the diagonal W_dwls) or the robust
    // sandwich (which uses NACOV itself). DWLS-only callers pass
    // `full_wls_weight = false` to skip the O(m³) inverse entirely. Even when
    // requested, the inverse is non-fatal: at small N with many indicators the
    // ordinal NACOV is often singular, so a failed inverse leaves W_wls empty
    // and an explicit WLS request reports it. This matches lavaan WLSMV, which
    // fits the diagonal weight regardless of NACOV rank.
    Eigen::MatrixXd W_wls;
    if (full_wls_weight) {
      if (auto W_wls_or = symmetric_inverse_pd(NACOV, "ordinal NACOV matrix");
          W_wls_or.has_value()) {
        W_wls = std::move(*W_wls_or);
      }
    }
    block_diag.moment_influence = IF;
    block_diag.gamma = NACOV;

    out.stats.R.push_back(std::move(R));
    out.stats.thresholds.push_back(std::move(th));
    out.stats.threshold_ov.push_back(std::move(th_ov));
    out.stats.threshold_level.push_back(std::move(th_level));
    out.stats.NACOV.push_back(std::move(NACOV));
    out.stats.moment_influence.push_back(std::move(IF));
    out.stats.int_data.push_back(Xcat);
    out.stats.moment_bread.push_back(B_inv);
    out.stats.W_dwls.push_back(std::move(W_dwls));
    out.stats.W_wls.push_back(std::move(W_wls));
    out.stats.n_obs.push_back(static_cast<std::int64_t>(n));
    out.stats.n_levels.push_back(std::move(levels));
    out.block_diagnostics.push_back(std::move(block_diag));
  }
  return out;
}

post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& Xs,
                                bool full_wls_weight) {
  auto out = pairwise_ordinal_stats_from_integer_data(Xs, full_wls_weight);
  if (!out.has_value()) return std::unexpected(out.error());
  return std::move(out->stats);
}

post_expected<OrdinalStats>
ordinal_stats_from_observed_integer_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    OrdinalPairwiseGammaKind gamma_kind,
    bool full_wls_weight) {
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_stats_from_observed_integer_data: no data blocks"));
  }

  OrdinalStats out;
  out.pairwise_gamma =
      gamma_kind == OrdinalPairwiseGammaKind::Overlap ? "overlap" : "nominal";
  out.R.reserve(Xs.size());
  out.thresholds.reserve(Xs.size());
  out.threshold_ov.reserve(Xs.size());
  out.threshold_level.reserve(Xs.size());
  out.NACOV.reserve(Xs.size());
  out.W_dwls.reserve(Xs.size());
  out.W_wls.reserve(Xs.size());
  out.n_obs.reserve(Xs.size());
  out.n_levels.reserve(Xs.size());
  if (gamma_kind == OrdinalPairwiseGammaKind::Overlap) {
    out.moment_influence.reserve(Xs.size());
    out.int_data.reserve(Xs.size());
  }
  out.moment_support_i.reserve(Xs.size());
  out.moment_support_j.reserve(Xs.size());
  out.moment_n_obs.reserve(Xs.size());
  out.moment_overlap_n_obs.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2 || p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_stats_from_observed_integer_data: block " +
              std::to_string(b) + " must have at least 2 rows and 1 column"));
    }

    Eigen::MatrixXi Xcat = Eigen::MatrixXi::Constant(n, p, -1);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    std::vector<std::int64_t> item_n(static_cast<std::size_t>(p), 0);
    for (Eigen::Index j = 0; j < p; ++j) {
      int max_level = 0;
      for (Eigen::Index r = 0; r < n; ++r) {
        const double v = X(r, j);
        if (!std::isfinite(v)) continue;
        if (std::floor(v) != v || v < 1.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_observed_integer_data: block " +
                  std::to_string(b) +
                  " has non-integer/non-positive observed category values"));
        }
        max_level = std::max(max_level, static_cast<int>(v));
        ++item_n[static_cast<std::size_t>(j)];
      }
      if (item_n[static_cast<std::size_t>(j)] < 2 || max_level < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_observed_integer_data: block " +
                std::to_string(b) + " variable " + std::to_string(j) +
                " has fewer than 2 observed levels"));
      }
      counts[static_cast<std::size_t>(j)].assign(
          static_cast<std::size_t>(max_level), 0);
      for (Eigen::Index r = 0; r < n; ++r) {
        const double v = X(r, j);
        if (!std::isfinite(v)) continue;
        const int lvl = static_cast<int>(v);
        counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(lvl - 1)] += 1;
        Xcat(r, j) = lvl - 1;
      }
      for (int c = 0; c < max_level; ++c) {
        if (counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)] == 0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_observed_integer_data: block " +
                  std::to_string(b) + " variable " + std::to_string(j) +
                  " has an empty observed category"));
        }
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::Index nth = 0;
    for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;

    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    th_ov.reserve(static_cast<std::size_t>(nth));
    th_level.reserve(static_cast<std::size_t>(nth));
    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    std::vector<std::int32_t> support_i(static_cast<std::size_t>(mdim), -1);
    std::vector<std::int32_t> support_j(static_cast<std::size_t>(mdim), -1);
    std::vector<std::int64_t> moment_n(static_cast<std::size_t>(mdim), 0);

    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      th_start[static_cast<std::size_t>(j)] = off;
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      const auto nj = item_n[static_cast<std::size_t>(j)];
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
        if (cum <= 0 || cum >= nj) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_observed_integer_data: block " +
                  std::to_string(b) + " variable " + std::to_string(j) +
                  " has an empty observed boundary category"));
        }
        const double prob = std::clamp(static_cast<double>(cum) /
                                           static_cast<double>(nj),
                                       1e-12, 1.0 - 1e-12);
        const double z = normal_quantile(prob);
        th(off) = z;
        th_by_var[static_cast<std::size_t>(j)](c) = z;
        th_ov.push_back(static_cast<std::int32_t>(j));
        th_level.push_back(static_cast<std::int32_t>(c + 1));
        support_i[static_cast<std::size_t>(off)] = static_cast<std::int32_t>(j);
        moment_n[static_cast<std::size_t>(off)] = nj;
        ++off;
      }
    }

    Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index j = 0; j < p; ++j) {
        const int c = Xcat(r, j);
        if (c < 0) continue;
        const auto& thj = th_by_var[static_cast<std::size_t>(j)];
        const double lo = (c == 0) ? -kInf : thj(c - 1);
        const double hi = (c == thj.size()) ? kInf : thj(c);
        const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
        const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
        if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
        if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
      }
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
    Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
    Eigen::Index corr_idx = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        auto tab_or = ordinal_pair_observed_table(
            X.col(i), X.col(j), levels[static_cast<std::size_t>(i)],
            levels[static_cast<std::size_t>(j)]);
        if (!tab_or.has_value()) return std::unexpected(tab_or.error());
        auto rho_or = fit_ordinal_pair_rho_ml(
            tab_or->counts, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!rho_or.has_value()) return std::unexpected(rho_or.error());
        const double rho = rho_or->rho;
        R(i, j) = R(j, i) = rho;

        Eigen::VectorXi xi(tab_or->n_obs);
        Eigen::VectorXi xj(tab_or->n_obs);
        Eigen::Index obs = 0;
        for (Eigen::Index r = 0; r < n; ++r) {
          if (Xcat(r, i) < 0 || Xcat(r, j) < 0) continue;
          xi(obs) = Xcat(r, i);
          xj(obs) = Xcat(r, j);
          ++obs;
        }
        auto ps_or = ordinal_pair_scores(
            xi, xj, rho, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!ps_or.has_value()) return std::unexpected(ps_or.error());
        const auto& ps = *ps_or;
        Eigen::Index obs_row = 0;
        for (Eigen::Index r = 0; r < n; ++r) {
          if (Xcat(r, i) < 0 || Xcat(r, j) < 0) continue;
          SC_COR(r, corr_idx) = ps.rho(obs_row);
          ++obs_row;
        }
        const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
        const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
        A21.block(corr_idx, si, 1, ps.threshold_i.cols()) =
            ps.rho.transpose() * ps.threshold_i;
        A21.block(corr_idx, sj, 1, ps.threshold_j.cols()) =
            ps.rho.transpose() * ps.threshold_j;

        const Eigen::Index m = nth + corr_idx;
        support_i[static_cast<std::size_t>(m)] = static_cast<std::int32_t>(i);
        support_j[static_cast<std::size_t>(m)] = static_cast<std::int32_t>(j);
        moment_n[static_cast<std::size_t>(m)] = tab_or->n_obs;
        ++corr_idx;
      }
    }

    auto r_diag_or = repair_correlation_if_requested(
        R, {}, "ordinal_stats_from_observed_integer_data: block " +
                   std::to_string(b) + " polychoric R");
    if (!r_diag_or.has_value()) return std::unexpected(r_diag_or.error());

    Eigen::MatrixXd SC(n, mdim);
    SC.leftCols(nth) = SC_TH;
    SC.rightCols(ncorr) = SC_COR;
    const Eigen::MatrixXd INNER = SC.transpose() * SC;

    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
    for (Eigen::Index j = 0; j < p; ++j) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len =
          static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
      A11.block(start, start, len, len) =
          INNER.block(start, start, len, len);
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "ordinal observed threshold information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::VectorXd A22_diag(ncorr);
    for (Eigen::Index k = 0; k < ncorr; ++k) {
      A22_diag(k) = INNER(nth + k, nth + k);
      if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_observed_integer_data: block " +
                std::to_string(b) + " has a singular polychoric score block"));
      }
    }
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
    B_inv.block(0, 0, nth, nth) = *A11_inv_or;
    B_inv.block(nth, 0, ncorr, nth).noalias() =
        -A22_inv * A21 * (*A11_inv_or);
    B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

    Eigen::MatrixXd NACOV = B_inv * INNER * B_inv.transpose();
    NACOV *= static_cast<double>(n);
    NACOV = 0.5 * (NACOV + NACOV.transpose());
    Eigen::MatrixXd IF;
    if (gamma_kind == OrdinalPairwiseGammaKind::Overlap) {
      IF.noalias() = static_cast<double>(n) * SC * B_inv.transpose();
    }

    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> overlap =
        observed_moment_overlap_counts(Xcat, support_i, support_j);

    if (gamma_kind == OrdinalPairwiseGammaKind::Nominal) {
      for (Eigen::Index a = 0; a < mdim; ++a) {
        for (Eigen::Index bb = a; bb < mdim; ++bb) {
          const double na = static_cast<double>(moment_n[static_cast<std::size_t>(a)]);
          const double nb = static_cast<double>(moment_n[static_cast<std::size_t>(bb)]);
          const double nab = static_cast<double>(overlap(a, bb));
          if (na > 0.0 && nb > 0.0 && nab > 0.0) {
            NACOV(a, bb) /= (static_cast<double>(n) * nab) / (na * nb);
            if (bb != a) NACOV(bb, a) = NACOV(a, bb);
          }
        }
      }
      NACOV = 0.5 * (NACOV + NACOV.transpose());
    }

    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_observed_integer_data: block " +
                std::to_string(b) + " has a non-positive NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    Eigen::MatrixXd W_wls;
    if (full_wls_weight) {
      if (auto W_wls_or = symmetric_inverse_pd(NACOV, "ordinal observed NACOV matrix");
          W_wls_or.has_value()) {
        W_wls = std::move(*W_wls_or);
      }
    }

    out.R.push_back(std::move(R));
    out.thresholds.push_back(std::move(th));
    out.threshold_ov.push_back(std::move(th_ov));
    out.threshold_level.push_back(std::move(th_level));
    out.NACOV.push_back(std::move(NACOV));
    out.W_dwls.push_back(std::move(W_dwls));
    out.W_wls.push_back(std::move(W_wls));
    out.n_obs.push_back(static_cast<std::int64_t>(n));
    out.n_levels.push_back(std::move(levels));
    if (gamma_kind == OrdinalPairwiseGammaKind::Overlap) {
      out.moment_influence.push_back(std::move(IF));
      out.int_data.push_back(std::move(Xcat));
    }
    out.moment_support_i.push_back(std::move(support_i));
    out.moment_support_j.push_back(std::move(support_j));
    out.moment_n_obs.push_back(std::move(moment_n));
    out.moment_overlap_n_obs.push_back(std::move(overlap));
  }

  return out;
}

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_shared_robust_from_integer_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    SharedRobustOrdinalOptions options) {
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1 ||
      !(std::isfinite(options.ftol) && options.ftol > 0.0) ||
      !(std::isfinite(options.gtol) && options.gtol > 0.0) ||
      !(std::isfinite(options.fd_step) && options.fd_step > 0.0) ||
      !(std::isfinite(options.min_threshold_spacing) &&
        options.min_threshold_spacing > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise_ordinal_stats_shared_robust_from_integer_data: invalid options"));
  }
  if (options.kind == SharedRobustOrdinalKind::HWeighted) {
    auto h_ok = eval_polychoric_h_score(1.0, options.h_score);
    if (!h_ok.has_value()) return std::unexpected(h_ok.error());
    if (h_score_is_ml_like_local(options.h_score)) {
      return pairwise_ordinal_stats_from_integer_data(Xs);
    }
  } else if (options.kind == SharedRobustOrdinalKind::Dpd &&
             !(std::isfinite(options.alpha) && options.alpha > 0.0)) {
    if (options.alpha == 0.0) return pairwise_ordinal_stats_from_integer_data(Xs);
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise_ordinal_stats_shared_robust_from_integer_data: invalid DPD alpha"));
  } else if (options.kind == SharedRobustOrdinalKind::HuberResidual) {
    auto clip_ok = eval_huber_residual_clip(0.0, options.clip);
    if (!clip_ok.has_value()) return std::unexpected(clip_ok.error());
    if (options.clip.kind == HuberResidualClipKind::None) {
      return pairwise_ordinal_stats_from_integer_data(Xs);
    }
  }

  auto base_or = pairwise_ordinal_stats_from_integer_data(Xs);
  if (!base_or.has_value()) return std::unexpected(base_or.error());
  PairwiseOrdinalStats out = std::move(*base_or);

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    const auto& levels = out.stats.n_levels[b];
    const Eigen::Index nth = out.stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;
    if (ncorr == 0) continue;

    std::vector<Eigen::VectorXd> th_by_var;
    std::vector<Eigen::Index> th_start;
    auto layout_ok = threshold_layout_for_block(
        out.stats, b, p, th_by_var, th_start);
    if (!layout_ok.has_value()) return std::unexpected(layout_ok.error());

    Eigen::MatrixXi Xcat(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = 0; i < n; ++i) {
        Xcat(i, j) = static_cast<int>(X(i, j)) - 1;
      }
    }

    std::vector<SharedOrdinalPairTable> pairs;
    pairs.reserve(static_cast<std::size_t>(ncorr));
    Eigen::Index corr_idx = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const Eigen::VectorXi xi = Xcat.col(i);
        const Eigen::VectorXi xj = Xcat.col(j);
        auto tab_or = ordinal_pair_table(
            xi, xj, levels[static_cast<std::size_t>(i)],
            levels[static_cast<std::size_t>(j)]);
        if (!tab_or.has_value()) return std::unexpected(tab_or.error());
        Eigen::MatrixXd adjusted = options.lavaan_adjust_2x2
            ? adjusted_polychoric_table_local(*tab_or)
            : Eigen::MatrixXd(*tab_or);
        pairs.push_back(SharedOrdinalPairTable{
            .i = i,
            .j = j,
            .corr_index = corr_idx++,
            .counts = std::move(*tab_or),
            .adjusted_counts = std::move(adjusted)});
      }
    }

    Eigen::VectorXd z = encode_shared_ordinal_params(
        out.stats.thresholds[b], out.stats.R[b], levels, th_start,
        options.rho_lower, options.rho_upper, options.min_threshold_spacing);
    double f = shared_ordinal_objective(z, levels, th_start, pairs, options);
    if (!std::isfinite(f)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise_ordinal_stats_shared_robust_from_integer_data: non-finite starting objective"));
    }

    const auto touched =
        shared_ordinal_touched_pairs(nth, mdim, levels, th_start, pairs);
    // The encoding already removes the bounds (cumulative-gap thresholds,
    // squashed rho), so this is an unconstrained smooth problem: delegate to
    // the vendored NLopt L-BFGS instead of hand-rolled steepest descent.
    // Non-finite objective or gradient at a probe point returns +inf, which
    // the line search treats as a barrier — matching the old step-halving
    // rejection. A hard optimizer failure keeps the ML starting values, like
    // the old loop's bail-out on an exhausted line search.
    int iterations = 0;
    bool converged = false;
    {
      const optim::NloptOptimizer lbfgs(
          optim::OptimOptions{.max_iter = options.max_iter,
                              .ftol = options.ftol,
                              .gtol = options.gtol},
          optim::NloptAlgorithm::Lbfgs);
      auto sol = lbfgs.minimize(
          [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad_out) {
            const double fx = shared_ordinal_objective(
                x, levels, th_start, pairs, options);
            if (!std::isfinite(fx)) {
              grad_out.setZero();
              return std::numeric_limits<double>::infinity();
            }
            grad_out = shared_ordinal_gradient(
                x, levels, th_start, pairs, touched, options);
            if (!grad_out.allFinite()) {
              grad_out.setZero();
              return std::numeric_limits<double>::infinity();
            }
            return fx;
          },
          z);
      if (sol.has_value() && sol->theta_hat.allFinite() &&
          std::isfinite(sol->fmin) && sol->fmin <= f) {
        z = sol->theta_hat;
        f = sol->fmin;
        iterations = sol->iterations;
        converged = sol->status == optim::OptimStatus::Converged ||
                    sol->status == optim::OptimStatus::LineSearchSalvaged;
      }
    }

    Eigen::VectorXd thresholds;
    Eigen::MatrixXd R;
    decode_shared_ordinal_params(
        z, levels, th_start, options.rho_lower, options.rho_upper,
        options.min_threshold_spacing, thresholds, th_by_var, R);

    Eigen::MatrixXd psi = shared_ordinal_casewise_psi(
        Xcat, thresholds, th_by_var, R, th_start, pairs, options);
    Eigen::MatrixXd bread = shared_ordinal_bread_numeric(
        thresholds, R, levels, th_start, pairs, options);
    if (!psi.allFinite() || !bread.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise_ordinal_stats_shared_robust_from_integer_data: non-finite sandwich components"));
    }
    Eigen::FullPivLU<Eigen::MatrixXd> lu(bread);
    lu.setThreshold(1e-10);
    if (lu.rank() < bread.rows()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise_ordinal_stats_shared_robust_from_integer_data: singular bread"));
    }
    const Eigen::MatrixXd bread_inv =
        lu.solve(Eigen::MatrixXd::Identity(mdim, mdim));
    Eigen::MatrixXd IF = -psi * bread_inv.transpose();

    std::vector<OrdinalPairDiagnostics> pair_diag;
    pair_diag.reserve(static_cast<std::size_t>(ncorr));
    for (const auto& pair : pairs) {
      const auto& th_i = th_by_var[static_cast<std::size_t>(pair.i)];
      const auto& th_j = th_by_var[static_cast<std::size_t>(pair.j)];
      const double rho = R(pair.i, pair.j);
      const double total = pair.adjusted_counts.sum();
      Eigen::MatrixXd probs = Eigen::MatrixXd::Zero(
          pair.adjusted_counts.rows(), pair.adjusted_counts.cols());
      Eigen::MatrixXd expected = probs;
      Eigen::MatrixXd residual = probs;
      Eigen::MatrixXd pearson = probs;
      Eigen::MatrixXd weights = Eigen::MatrixXd::Ones(
          pair.adjusted_counts.rows(), pair.adjusted_counts.cols());
      double objective = 0.0;
      double score = 0.0;
      for (Eigen::Index a = 0; a < pair.adjusted_counts.rows(); ++a) {
        const double lo_i = (a == 0) ? -kInf : th_i(a - 1);
        const double hi_i = (a + 1 == pair.adjusted_counts.rows())
            ? kInf : th_i(a);
        for (Eigen::Index c = 0; c < pair.adjusted_counts.cols(); ++c) {
          const double lo_j = (c == 0) ? -kInf : th_j(c - 1);
          const double hi_j = (c + 1 == pair.adjusted_counts.cols())
              ? kInf : th_j(c);
          const double raw_p = ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
          const double pcell = std::max(kProbFloor, raw_p);
          const double fhat = pair.adjusted_counts(a, c) / total;
          probs(a, c) = pcell;
          expected(a, c) = total * pcell;
          residual(a, c) = pair.adjusted_counts(a, c) - expected(a, c);
          pearson(a, c) = residual(a, c) /
              std::sqrt(std::max(kProbFloor, expected(a, c)));
          if (options.kind == SharedRobustOrdinalKind::HWeighted) {
            auto h = eval_polychoric_h_score(fhat / pcell, options.h_score);
            if (!h.has_value()) return std::unexpected(h.error());
            weights(a, c) = fhat > 0.0 ? h->h / (fhat / pcell) : 1.0;
            objective += pcell * h->phi;
            if (raw_p > kProbFloor) {
              score += h->h * ordinal_bvn_rect_drho(lo_i, hi_i, lo_j, hi_j, rho);
            }
          } else if (options.kind == SharedRobustOrdinalKind::Dpd) {
            weights(a, c) = std::pow(pcell, options.alpha);
            objective += pcell * weights(a, c) -
                         ((1.0 + options.alpha) / options.alpha) *
                             fhat * weights(a, c);
          } else {
            const double r = pearson_cell_residual(fhat, pcell, total);
            auto h = eval_huber_residual_clip(r, options.clip);
            if (!h.has_value()) return std::unexpected(h.error());
            weights(a, c) = h->weight;
            objective += pcell * h->loss;
            if (raw_p > kProbFloor) {
              score += clipped_cell_frequency(fhat, pcell, total, options.clip) *
                       ordinal_bvn_rect_drho(lo_i, hi_i, lo_j, hi_j, rho) /
                       pcell;
            }
          }
        }
      }
      const double width = options.rho_upper - options.rho_lower;
      pair_diag.push_back(OrdinalPairDiagnostics{
          .label = OrdinalPairLabel{
              .block = static_cast<std::int32_t>(b),
              .i = static_cast<std::int32_t>(pair.i),
              .j = static_cast<std::int32_t>(pair.j),
              .n_levels_i = levels[static_cast<std::size_t>(pair.i)],
              .n_levels_j = levels[static_cast<std::size_t>(pair.j)]},
          .rho = rho,
          .negloglik = objective,
          .objective = objective,
          .score = score,
          .iterations = iterations,
          .h_weighted = options.kind == SharedRobustOrdinalKind::HWeighted,
          .converged = converged,
          .hit_lower = std::abs(rho - options.rho_lower) <= 1e-8 * width,
          .hit_upper = std::abs(rho - options.rho_upper) <= 1e-8 * width,
          .n_obs = static_cast<std::int64_t>(pair.counts.sum()),
          .n_missing = 0,
          .ridge_applied = false,
          .ridge = 0.0,
          .shrinkage_applied = false,
          .shrinkage_intensity = 0.0,
          .counts = pair.counts,
          .adjusted_counts = pair.adjusted_counts,
          .expected_counts = std::move(expected),
          .residual_counts = std::move(residual),
          .pearson_residuals = std::move(pearson),
          .weights = std::move(weights)});
    }

    auto r_repair_or = repair_correlation_if_requested(
        R, options.correlation_repair,
        "pairwise_ordinal_stats_shared_robust_from_integer_data: block " +
            std::to_string(b) + " robust R");
    if (!r_repair_or.has_value()) return std::unexpected(r_repair_or.error());
    R = std::move(r_repair_or->R);
    if (r_repair_or->repaired) {
      const double corr_scale = r_repair_or->ridge > 0.0
          ? 1.0 / (1.0 + r_repair_or->ridge)
          : 1.0 - r_repair_or->shrinkage;
      IF.rightCols(ncorr) *= corr_scale;
      for (auto& pd : pair_diag) {
        pd.ridge_applied = r_repair_or->ridge > 0.0;
        pd.ridge = r_repair_or->ridge;
        pd.shrinkage_applied = r_repair_or->shrinkage > 0.0;
        pd.shrinkage_intensity = r_repair_or->shrinkage;
      }
    }

    Eigen::MatrixXd gamma = (IF.transpose() * IF) / static_cast<double>(n);
    gamma = 0.5 * (gamma + gamma.transpose());
    if (!gamma.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise_ordinal_stats_shared_robust_from_integer_data: non-finite Gamma"));
    }
    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = gamma(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "pairwise_ordinal_stats_shared_robust_from_integer_data: non-positive Gamma diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    auto W_wls_or = symmetric_inverse_pd(gamma, "shared robust ordinal Gamma");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());

    out.stats.thresholds[b] = std::move(thresholds);
    out.stats.R[b] = std::move(R);
    out.stats.NACOV[b] = gamma;
    out.stats.W_dwls[b] = std::move(W_dwls);
    out.stats.W_wls[b] = std::move(*W_wls_or);
    out.block_diagnostics[b].pair_diagnostics = std::move(pair_diag);
    out.block_diagnostics[b].moment_influence = std::move(IF);
    out.block_diagnostics[b].gamma = out.stats.NACOV[b];
    out.block_diagnostics[b].min_eigen_r = r_repair_or->min_eigen;
    out.block_diagnostics[b].raw_min_eigen_r = r_repair_or->raw_min_eigen;
    out.block_diagnostics[b].r_repair_applied = r_repair_or->repaired;
    out.block_diagnostics[b].r_ridge = r_repair_or->ridge;
    out.block_diagnostics[b].r_shrinkage_intensity = r_repair_or->shrinkage;
  }
  return out;
}

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_h_weighted_from_integer_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    PairwiseOrdinalHWeightedStatsOptions options) {
  return pairwise_ordinal_stats_shared_robust_from_integer_data(
      Xs, SharedRobustOrdinalOptions{
              .kind = SharedRobustOrdinalKind::HWeighted,
              .h_score = options.rho.h_score,
              .clip = {},
              .rho_lower = options.rho.rho_lower,
              .rho_upper = options.rho.rho_upper,
              .max_iter = options.max_iter,
              .ftol = options.ftol,
              .gtol = options.gtol,
              .fd_step = options.fd_step,
              .min_threshold_spacing = options.min_threshold_spacing,
              .lavaan_adjust_2x2 = options.rho.lavaan_adjust_2x2,
              .correlation_repair = options.correlation_repair});
}

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_dpd_from_integer_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    PairwiseOrdinalDpdStatsOptions options) {
  return pairwise_ordinal_stats_shared_robust_from_integer_data(
      Xs, SharedRobustOrdinalOptions{
              .kind = SharedRobustOrdinalKind::Dpd,
              .h_score = {},
              .clip = {},
              .alpha = options.alpha,
              .rho_lower = options.rho_lower,
              .rho_upper = options.rho_upper,
              .max_iter = options.max_iter,
              .ftol = options.ftol,
              .gtol = options.gtol,
              .fd_step = options.fd_step,
              .min_threshold_spacing = options.min_threshold_spacing,
              .lavaan_adjust_2x2 = options.lavaan_adjust_2x2,
              .correlation_repair = options.correlation_repair});
}

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_huber_residual_from_integer_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    PairwiseOrdinalHuberResidualStatsOptions options) {
  return pairwise_ordinal_stats_shared_robust_from_integer_data(
      Xs, SharedRobustOrdinalOptions{
              .kind = SharedRobustOrdinalKind::HuberResidual,
              .h_score = {},
              .clip = options.clip,
              .alpha = 0.0,
              .rho_lower = options.rho_lower,
              .rho_upper = options.rho_upper,
              .max_iter = options.max_iter,
              .ftol = options.ftol,
              .gtol = options.gtol,
              .fd_step = options.fd_step,
              .min_threshold_spacing = options.min_threshold_spacing,
              .lavaan_adjust_2x2 = options.lavaan_adjust_2x2,
              .correlation_repair = options.correlation_repair});
}

post_expected<MixedOrdinalStats>
mixed_ordinal_stats_from_observed_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    const std::vector<std::vector<std::int32_t>>& ordered,
    bool full_wls_weight) {
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_observed_data: no data blocks"));
  }
  if (Xs.size() != ordered.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_observed_data: ordered mask block count "
        "mismatch"));
  }

  MixedOrdinalStats stats;
  stats.R.reserve(Xs.size());
  stats.mean.reserve(Xs.size());
  stats.ordered.reserve(Xs.size());
  stats.thresholds.reserve(Xs.size());
  stats.threshold_ov.reserve(Xs.size());
  stats.threshold_level.reserve(Xs.size());
  stats.moments.reserve(Xs.size());
  stats.NACOV.reserve(Xs.size());
  stats.W_dwls.reserve(Xs.size());
  stats.W_wls.reserve(Xs.size());
  stats.n_obs.reserve(Xs.size());
  stats.n_levels.reserve(Xs.size());
  stats.moment_influence.reserve(Xs.size());
  stats.raw_data.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2 || p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_observed_data: block " +
              std::to_string(b) + " must have at least 2 rows and 1 column"));
    }
    if (ordered[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_observed_data: block " +
              std::to_string(b) +
              " ordered mask length does not match column count"));
    }
    bool have_ordered = false;
    bool have_cont = false;
    for (auto z : ordered[b]) {
      if (z != 0 && z != 1) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_observed_data: block " +
                std::to_string(b) +
                " ordered mask must contain only 0/1 values"));
      }
      have_ordered = have_ordered || z != 0;
      have_cont = have_cont || z == 0;
    }
    if (!have_ordered || !have_cont) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_observed_data: block " +
              std::to_string(b) +
              " must contain both ordered and continuous variables"));
    }

    Eigen::MatrixXi Xcat = Eigen::MatrixXi::Constant(n, p, -1);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    std::vector<std::int64_t> obs_n(static_cast<std::size_t>(p), 0);
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd var = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);

    for (Eigen::Index j = 0; j < p; ++j) {
      const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
      if (oj) {
        int max_level = 0;
        for (Eigen::Index r = 0; r < n; ++r) {
          const double v = X(r, j);
          if (!std::isfinite(v)) continue;
          if (std::floor(v) != v || v < 1.0) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_stats_from_observed_data: block " +
                    std::to_string(b) +
                    " has non-integer/non-positive observed category values"));
          }
          max_level = std::max(max_level, static_cast<int>(v));
          ++obs_n[static_cast<std::size_t>(j)];
        }
        if (obs_n[static_cast<std::size_t>(j)] < 2 || max_level < 2) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_observed_data: block " +
                  std::to_string(b) + " variable " + std::to_string(j) +
                  " has fewer than 2 observed levels"));
        }
        counts[static_cast<std::size_t>(j)].assign(
            static_cast<std::size_t>(max_level), 0);
        for (Eigen::Index r = 0; r < n; ++r) {
          const double v = X(r, j);
          if (!std::isfinite(v)) continue;
          const int lvl = static_cast<int>(v);
          counts[static_cast<std::size_t>(j)]
                [static_cast<std::size_t>(lvl - 1)] += 1;
          Xcat(r, j) = lvl - 1;
        }
        for (int c = 0; c < max_level; ++c) {
          if (counts[static_cast<std::size_t>(j)]
                    [static_cast<std::size_t>(c)] == 0) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_stats_from_observed_data: block " +
                    std::to_string(b) + " variable " + std::to_string(j) +
                    " has an empty observed category"));
          }
        }
        levels[static_cast<std::size_t>(j)] = max_level;
      } else {
        double sum = 0.0;
        for (Eigen::Index r = 0; r < n; ++r) {
          const double v = X(r, j);
          if (!std::isfinite(v)) continue;
          sum += v;
          ++obs_n[static_cast<std::size_t>(j)];
        }
        if (obs_n[static_cast<std::size_t>(j)] < 2) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_observed_data: block " +
                  std::to_string(b) + " continuous variable " +
                  std::to_string(j) + " has fewer than 2 observed values"));
        }
        mean(j) = sum / static_cast<double>(obs_n[static_cast<std::size_t>(j)]);
        for (Eigen::Index r = 0; r < n; ++r) {
          const double v = X(r, j);
          if (!std::isfinite(v)) continue;
          const double c = v - mean(j);
          var(j) += c * c;
        }
        var(j) /= static_cast<double>(obs_n[static_cast<std::size_t>(j)]);
        if (var(j) <= 0.0 || !std::isfinite(var(j))) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_observed_data: block " +
                  std::to_string(b) +
                  " has a non-positive observed continuous variance"));
        }
        const double sd = std::sqrt(var(j));
        for (Eigen::Index r = 0; r < n; ++r) {
          if (std::isfinite(X(r, j))) U(r, j) = (X(r, j) - mean(j)) / sd;
        }
      }
    }

    Eigen::Index nth = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) {
        nth += static_cast<Eigen::Index>(
            levels[static_cast<std::size_t>(j)] - 1);
      }
    }
    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    th_ov.reserve(static_cast<std::size_t>(nth));
    th_level.reserve(static_cast<std::size_t>(nth));

    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
      th_start[static_cast<std::size_t>(j)] = off;
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      const auto nj = obs_n[static_cast<std::size_t>(j)];
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
        if (cum <= 0 || cum >= nj) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_observed_data: block " +
                  std::to_string(b) + " variable " + std::to_string(j) +
                  " has an empty observed boundary category"));
        }
        const double prob = std::clamp(
            static_cast<double>(cum) / static_cast<double>(nj), 1e-12,
            1.0 - 1e-12);
        const double z = normal_quantile(prob);
        th(off) = z;
        th_by_var[static_cast<std::size_t>(j)](c) = z;
        th_ov.push_back(static_cast<std::int32_t>(j));
        th_level.push_back(static_cast<std::int32_t>(c + 1));
        ++off;
      }
    }

    Eigen::Index n_cont = 0;
    std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        cont_pos[static_cast<std::size_t>(j)] = n_cont++;
      }
    }
    const Eigen::Index s1 = nth + 2 * n_cont;
    Eigen::MatrixXd SC1 = Eigen::MatrixXd::Zero(n, s1);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
        const int c = Xcat(r, j);
        if (c < 0) continue;
        const auto& thj = th_by_var[static_cast<std::size_t>(j)];
        const double lo = (c == 0) ? -kInf : thj(c - 1);
        const double hi = (c == thj.size()) ? kInf : thj(c);
        const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
        const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
        if (c < thj.size()) SC1(r, base + c) += normal_pdf(thj(c)) / pr;
        if (c > 0) SC1(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
      const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
      const double v = var(j);
      for (Eigen::Index r = 0; r < n; ++r) {
        if (!std::isfinite(X(r, j))) continue;
        const double c = X(r, j) - mean(j);
        SC1(r, nth + cp) = c / v;
        SC1(r, nth + n_cont + cp) = (c * c - v) / (2.0 * v * v);
      }
    }

    const Eigen::MatrixXd INNER1 = SC1.transpose() * SC1;
    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(s1, s1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) {
        const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
        const Eigen::Index len = static_cast<Eigen::Index>(
            levels[static_cast<std::size_t>(j)] - 1);
        A11.block(start, start, len, len) =
            INNER1.block(start, start, len, len);
      } else {
        const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::Index va =
            nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
        A11(mu, mu) = INNER1(mu, mu);
        A11(va, va) = INNER1(va, va);
        A11(mu, va) = A11(va, mu) = INNER1(mu, va);
      }
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "mixed observed ordinal stage-1 information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) R(j, j) = var(j);
    }

    const Eigen::Index n_assoc = p * (p - 1) / 2;
    Eigen::MatrixXd SC_ASSOC = Eigen::MatrixXd::Zero(n, n_assoc);
    Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(n_assoc, s1);
    Eigen::VectorXd A22_diag = Eigen::VectorXd::Ones(n_assoc);
    Eigen::Index assoc_count = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        std::vector<Eigen::Index> rows;
        rows.reserve(static_cast<std::size_t>(n));
        for (Eigen::Index r = 0; r < n; ++r) {
          const bool i_obs = oi ? Xcat(r, i) >= 0 : std::isfinite(X(r, i));
          const bool j_obs = oj ? Xcat(r, j) >= 0 : std::isfinite(X(r, j));
          if (i_obs && j_obs) rows.push_back(r);
        }
        if (rows.size() < 2) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_observed_data: block " +
                  std::to_string(b) + " pair (" + std::to_string(i) + ", " +
                  std::to_string(j) +
                  ") has fewer than 2 observed rows"));
        }

        const Eigen::Index np = static_cast<Eigen::Index>(rows.size());
        if (oi && oj) {
          Eigen::VectorXi xi(np);
          Eigen::VectorXi xj(np);
          for (Eigen::Index r = 0; r < np; ++r) {
            const Eigen::Index src = rows[static_cast<std::size_t>(r)];
            xi(r) = Xcat(src, i);
            xj(r) = Xcat(src, j);
          }
          auto tab_or = ordinal_pair_table(
              xi, xj, levels[static_cast<std::size_t>(i)],
              levels[static_cast<std::size_t>(j)]);
          if (!tab_or.has_value()) return std::unexpected(tab_or.error());
          auto rho_or = fit_ordinal_pair_rho_ml(
              *tab_or, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          if (!rho_or.has_value()) return std::unexpected(rho_or.error());
          const double rho = rho_or->rho;
          R(i, j) = R(j, i) = rho;
          auto ps_or = ordinal_pair_scores(
              xi, xj, rho, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          if (!ps_or.has_value()) return std::unexpected(ps_or.error());
          const auto& ps = *ps_or;
          for (Eigen::Index r = 0; r < np; ++r) {
            SC_ASSOC(rows[static_cast<std::size_t>(r)], assoc_count) =
                ps.rho(r);
          }
          const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
          const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
          A21.block(assoc_count, si, 1, ps.threshold_i.cols()) =
              ps.rho.transpose() * ps.threshold_i;
          A21.block(assoc_count, sj, 1, ps.threshold_j.cols()) =
              ps.rho.transpose() * ps.threshold_j;
        } else if (oi || oj) {
          const Eigen::Index o = oi ? i : j;
          const Eigen::Index c = oi ? j : i;
          Eigen::VectorXi cat(np);
          Eigen::VectorXd u(np);
          for (Eigen::Index r = 0; r < np; ++r) {
            const Eigen::Index src = rows[static_cast<std::size_t>(r)];
            cat(r) = Xcat(src, o);
            u(r) = U(src, c);
          }
          const double sd = std::sqrt(var(c));
          auto rho_or = fit_polyserial_pair_rho_ml(
              cat, u, th_by_var[static_cast<std::size_t>(o)]);
          if (!rho_or.has_value()) return std::unexpected(rho_or.error());
          const double rho = rho_or->rho;
          R(i, j) = R(j, i) = rho * sd;
          auto ps_or = polyserial_pair_scores(
              cat, u, rho, th_by_var[static_cast<std::size_t>(o)]);
          if (!ps_or.has_value()) return std::unexpected(ps_or.error());
          const auto& ps = *ps_or;
          for (Eigen::Index r = 0; r < np; ++r) {
            SC_ASSOC(rows[static_cast<std::size_t>(r)], assoc_count) =
                ps.rho(r);
          }
          const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
          const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(c)];
          const Eigen::Index va =
              nth + n_cont + cont_pos[static_cast<std::size_t>(c)];
          A21.block(assoc_count, so, 1, ps.thresholds.cols()) =
              ps.rho.transpose() * ps.thresholds;
          A21(assoc_count, mu) = ps.rho.dot(ps.mu_unit) / sd;
          A21(assoc_count, va) = ps.rho.dot(ps.var_unit) / var(c);
        } else {
          Eigen::VectorXd xi(np);
          Eigen::VectorXd xj(np);
          double cov = 0.0;
          for (Eigen::Index r = 0; r < np; ++r) {
            const Eigen::Index src = rows[static_cast<std::size_t>(r)];
            xi(r) = X(src, i);
            xj(r) = X(src, j);
            cov += (xi(r) - mean(i)) * (xj(r) - mean(j));
          }
          cov /= static_cast<double>(np);
          R(i, j) = R(j, i) = cov;
          auto sc_or = continuous_pair_normal_scores(
              xi, xj, mean(i), mean(j), var(i), var(j), cov);
          if (!sc_or.has_value()) return std::unexpected(sc_or.error());
          const Eigen::MatrixXd& S = sc_or->score_contributions;
          const double sdi = std::sqrt(var(i));
          const double sdj = std::sqrt(var(j));
          const double rho = cov / (sdi * sdj);
          const Eigen::VectorXd s_rho = sdi * sdj * S.col(4);
          for (Eigen::Index r = 0; r < np; ++r) {
            SC_ASSOC(rows[static_cast<std::size_t>(r)], assoc_count) =
                s_rho(r);
          }
          const Eigen::Index mui = nth + cont_pos[static_cast<std::size_t>(i)];
          const Eigen::Index muj = nth + cont_pos[static_cast<std::size_t>(j)];
          const Eigen::Index vai =
              nth + n_cont + cont_pos[static_cast<std::size_t>(i)];
          const Eigen::Index vaj =
              nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
          const Eigen::VectorXd ch_var_i =
              S.col(2) + (rho * sdj / (2.0 * sdi)) * S.col(4);
          const Eigen::VectorXd ch_var_j =
              S.col(3) + (rho * sdi / (2.0 * sdj)) * S.col(4);
          A21(assoc_count, mui) = s_rho.dot(S.col(0));
          A21(assoc_count, muj) = s_rho.dot(S.col(1));
          A21(assoc_count, vai) = s_rho.dot(ch_var_i);
          A21(assoc_count, vaj) = s_rho.dot(ch_var_j);
        }
        A22_diag(assoc_count) = SC_ASSOC.col(assoc_count).squaredNorm();
        if (A22_diag(assoc_count) <= 1e-12 ||
            !std::isfinite(A22_diag(assoc_count))) {
          A22_diag(assoc_count) = 1.0;
        }
        ++assoc_count;
      }
    }

    const Eigen::VectorXd moment = mixed_moment_vector(R, mean, ordered[b], th);
    const Eigen::Index mdim = moment.size();
    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(s1 + n_assoc, s1 + n_assoc);
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
    B_inv.block(0, 0, s1, s1) = *A11_inv_or;
    B_inv.block(s1, 0, n_assoc, s1).noalias() =
        -A22_inv * A21.topRows(n_assoc) * (*A11_inv_or);
    B_inv.block(s1, s1, n_assoc, n_assoc) = A22_inv;
    Eigen::MatrixXd SC(n, s1 + n_assoc);
    SC.leftCols(s1) = SC1;
    SC.rightCols(n_assoc) = SC_ASSOC;
    Eigen::MatrixXd IF_est = static_cast<double>(n) * SC * B_inv.transpose();

    Eigen::MatrixXd IF(n, mdim);
    Eigen::Index pos = 0;
    IF.middleCols(pos, nth) = IF_est.leftCols(nth);
    pos += nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) =
            -IF_est.col(nth + cont_pos[static_cast<std::size_t>(j)]);
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) =
            IF_est.col(nth + n_cont + cont_pos[static_cast<std::size_t>(j)]);
      }
    }
    Eigen::Index assoc_pos = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        const Eigen::VectorXd rho_col = IF_est.col(s1 + assoc_pos);
        if (oi && oj) {
          IF.col(pos++) = rho_col;
        } else if (oi || oj) {
          const Eigen::Index c = oi ? j : i;
          const double sd = std::sqrt(var(c));
          const double rho = R(i, j) / sd;
          const Eigen::Index va =
              nth + n_cont + cont_pos[static_cast<std::size_t>(c)];
          IF.col(pos++) =
              sd * rho_col + (rho / (2.0 * sd)) * IF_est.col(va);
        } else {
          const double sdi = std::sqrt(var(i));
          const double sdj = std::sqrt(var(j));
          const double rho = R(i, j) / (sdi * sdj);
          const Eigen::Index vai =
              nth + n_cont + cont_pos[static_cast<std::size_t>(i)];
          const Eigen::Index vaj =
              nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
          IF.col(pos++) = sdi * sdj * rho_col +
              (rho * sdj / (2.0 * sdi)) * IF_est.col(vai) +
              (rho * sdi / (2.0 * sdj)) * IF_est.col(vaj);
        }
        ++assoc_pos;
      }
    }

    Eigen::MatrixXd NACOV = (IF.transpose() * IF) / static_cast<double>(n);
    NACOV = 0.5 * (NACOV + NACOV.transpose());
    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_observed_data: block " +
                std::to_string(b) + " has a non-positive NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    Eigen::MatrixXd W_wls;
    if (full_wls_weight) {
      if (auto W_wls_or =
              symmetric_inverse_pd(NACOV, "mixed observed ordinal NACOV matrix");
          W_wls_or.has_value()) {
        W_wls = std::move(*W_wls_or);
      }
    }

    stats.R.push_back(std::move(R));
    stats.mean.push_back(std::move(mean));
    stats.ordered.push_back(ordered[b]);
    stats.thresholds.push_back(std::move(th));
    stats.threshold_ov.push_back(std::move(th_ov));
    stats.threshold_level.push_back(std::move(th_level));
    stats.moments.push_back(std::move(moment));
    stats.NACOV.push_back(std::move(NACOV));
    stats.moment_influence.push_back(std::move(IF));
    stats.W_dwls.push_back(std::move(W_dwls));
    stats.W_wls.push_back(std::move(W_wls));
    stats.n_obs.push_back(static_cast<std::int64_t>(n));
    stats.n_levels.push_back(std::move(levels));
    stats.raw_data.push_back(X);
  }
  return stats;
}

post_expected<MixedOrdinalPolyserialDpdStats>
mixed_ordinal_stats_from_data_impl(
    const std::vector<Eigen::MatrixXd>& Xs,
    const std::vector<std::vector<std::int32_t>>& ordered,
    bool use_polyserial_dpd,
    PolyserialPairDpdOptions dpd_options,
    bool full_wls_weight = true) {
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_data: no data blocks"));
  }
  if (Xs.size() != ordered.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_data: ordered mask block count mismatch"));
  }

  MixedOrdinalPolyserialDpdStats out;
  MixedOrdinalStats& stats = out.stats;
  stats.R.reserve(Xs.size());
  stats.mean.reserve(Xs.size());
  stats.ordered.reserve(Xs.size());
  stats.thresholds.reserve(Xs.size());
  stats.threshold_ov.reserve(Xs.size());
  stats.threshold_level.reserve(Xs.size());
  stats.moments.reserve(Xs.size());
  stats.NACOV.reserve(Xs.size());
  stats.W_dwls.reserve(Xs.size());
  stats.W_wls.reserve(Xs.size());
  stats.n_obs.reserve(Xs.size());
  stats.n_levels.reserve(Xs.size());
  stats.moment_influence.reserve(Xs.size());
  if (!use_polyserial_dpd) stats.raw_data.reserve(Xs.size());
  if (use_polyserial_dpd) out.block_diagnostics.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2 || p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
              " must have at least 2 rows and 1 column"));
    }
    if (ordered[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
              " ordered mask length does not match column count"));
    }
    bool have_ordered = false;
    bool have_cont = false;
    for (auto z : ordered[b]) {
      have_ordered = have_ordered || z != 0;
      have_cont = have_cont || z == 0;
    }
    if (!have_ordered || !have_cont) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
              " must contain both ordered and continuous variables"));
    }

    Eigen::MatrixXi Xcat = Eigen::MatrixXi::Zero(n, p);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        for (Eigen::Index i = 0; i < n; ++i) {
          if (!std::isfinite(X(i, j))) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                    " has non-finite continuous values"));
          }
        }
        continue;
      }
      int max_level = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        const double v = X(i, j);
        if (!std::isfinite(v) || std::floor(v) != v || v < 1.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                  " has non-integer/non-positive category values"));
        }
        max_level = std::max(max_level, static_cast<int>(v));
      }
      if (max_level < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                " variable " + std::to_string(j) + " has fewer than 2 levels"));
      }
      counts[static_cast<std::size_t>(j)].assign(static_cast<std::size_t>(max_level), 0);
      for (Eigen::Index i = 0; i < n; ++i) {
        const int lvl = static_cast<int>(X(i, j));
        counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(lvl - 1)] += 1;
        Xcat(i, j) = lvl - 1;
      }
      for (int c = 0; c < max_level; ++c) {
        if (counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)] == 0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                  " variable " + std::to_string(j) + " has an empty category"));
        }
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd var = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
      mean(j) = X.col(j).mean();
      for (Eigen::Index i = 0; i < n; ++i) {
        const double c = X(i, j) - mean(j);
        var(j) += c * c;
      }
      var(j) /= static_cast<double>(n);
      if (var(j) <= 0.0 || !std::isfinite(var(j))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                " has a non-positive continuous variance"));
      }
      U.col(j) = (X.col(j).array() - mean(j)) / std::sqrt(var(j));
    }

    Eigen::Index nth = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) {
        nth += static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
      }
    }
    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
      th_start[static_cast<std::size_t>(j)] = off;
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
        const double prob = std::clamp(static_cast<double>(cum) / static_cast<double>(n),
                                       1e-12, 1.0 - 1e-12);
        const double z = normal_quantile(prob);
        th(off) = z;
        th_by_var[static_cast<std::size_t>(j)](c) = z;
        th_ov.push_back(static_cast<std::int32_t>(j));
        th_level.push_back(static_cast<std::int32_t>(c + 1));
        ++off;
      }
    }

    // Stage-1 estimating-equation stack mirrors lavaan's muthen1984:
    // per-variable univariate ML scores — thresholds for ordinal variables,
    // (mu, sigma^2) for continuous ones — with a block-diagonal-per-variable
    // bread A11 taken from the score OPG, a full-OPG meat, and pair scores
    // for every association type. Stack layout: [th | mu | var].
    Eigen::Index n_cont = 0;
    std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        cont_pos[static_cast<std::size_t>(j)] = n_cont++;
      }
    }
    const Eigen::Index s1 = nth + 2 * n_cont;

    Eigen::MatrixXd SC1 = Eigen::MatrixXd::Zero(n, s1);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
        const int c = Xcat(r, j);
        const auto& thj = th_by_var[static_cast<std::size_t>(j)];
        const double lo = (c == 0) ? -kInf : thj(c - 1);
        const double hi = (c == thj.size()) ? kInf : thj(c);
        const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
        const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
        if (c < thj.size()) SC1(r, base + c) += normal_pdf(thj(c)) / pr;
        if (c > 0) SC1(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
      const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
      const double v = var(j);
      SC1.col(nth + cp) = (X.col(j).array() - mean(j)) / v;
      SC1.col(nth + n_cont + cp) =
          ((X.col(j).array() - mean(j)).square() - v) / (2.0 * v * v);
    }
    const Eigen::MatrixXd INNER1 = SC1.transpose() * SC1;
    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(s1, s1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) {
        const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
        const Eigen::Index len = static_cast<Eigen::Index>(
            levels[static_cast<std::size_t>(j)] - 1);
        A11.block(start, start, len, len) =
            INNER1.block(start, start, len, len);
      } else {
        const Eigen::Index a = nth + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::Index v = nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
        A11(a, a) = INNER1(a, a);
        A11(v, v) = INNER1(v, v);
        A11(a, v) = A11(v, a) = INNER1(a, v);
      }
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "mixed ordinal stage-1 information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) R(j, j) = var(j);
    }

    std::vector<Eigen::VectorXd> assoc_scores;
    Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(p * (p - 1) / 2, s1);
    Eigen::VectorXd A22_override =
        Eigen::VectorXd::Constant(p * (p - 1) / 2,
                                  std::numeric_limits<double>::quiet_NaN());
    MixedOrdinalPolyserialDpdBlockDiagnostics block_diag;
    Eigen::Index assoc_count = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        if (oi && oj) {
          const Eigen::VectorXi xi = Xcat.col(i);
          const Eigen::VectorXi xj = Xcat.col(j);
          auto tab_or = ordinal_pair_table(
              xi, xj, levels[static_cast<std::size_t>(i)],
              levels[static_cast<std::size_t>(j)]);
          if (!tab_or.has_value()) return std::unexpected(tab_or.error());
          auto rho_or = fit_ordinal_pair_rho_ml(
              *tab_or, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          if (!rho_or.has_value()) return std::unexpected(rho_or.error());
          const double rho = rho_or->rho;
          R(i, j) = R(j, i) = rho;
          auto ps_or = ordinal_pair_scores(
              xi, xj, rho, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          if (!ps_or.has_value()) return std::unexpected(ps_or.error());
          const auto& ps = *ps_or;
          assoc_scores.push_back(ps.rho);
          const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
          const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
          A21.block(assoc_count, si, 1, ps.threshold_i.cols()) =
              ps.rho.transpose() * ps.threshold_i;
          A21.block(assoc_count, sj, 1, ps.threshold_j.cols()) =
              ps.rho.transpose() * ps.threshold_j;
        } else if (oi || oj) {
          const Eigen::Index o = oi ? i : j;
          const Eigen::Index c = oi ? j : i;
          Eigen::VectorXi cat(n);
          for (Eigen::Index r = 0; r < n; ++r) cat(r) = Xcat(r, o);
          const double sd = std::sqrt(var(c));
          const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
          const Eigen::Index mu_col = nth + cont_pos[static_cast<std::size_t>(c)];
          const Eigen::Index var_col =
              nth + n_cont + cont_pos[static_cast<std::size_t>(c)];
          if (use_polyserial_dpd) {
            auto rho_or = fit_polyserial_pair_rho_dpd(
                cat, U.col(c), th_by_var[static_cast<std::size_t>(o)],
                dpd_options);
            if (!rho_or.has_value()) return std::unexpected(rho_or.error());
            const double rho = rho_or->rho;
            R(i, j) = R(j, i) = rho * sd;
            auto ps_or = polyserial_pair_dpd_scores(
                cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)],
                dpd_options);
            if (!ps_or.has_value()) return std::unexpected(ps_or.error());
            const auto& ps = *ps_or;
            assoc_scores.push_back(ps.rho);
            A21.block(assoc_count, so, 1, ps.thresholds.cols()) =
                static_cast<double>(n) *
                ps.bread.block(ps.bread.rows() - 1, 0, 1, ps.thresholds.cols());
            A22_override(assoc_count) =
                static_cast<double>(n) *
                ps.bread(ps.bread.rows() - 1, ps.bread.cols() - 1);
            // The DPD bread covers the (thresholds, rho) block; the mu/var
            // coupling channels use the ML score-cross identity at the DPD
            // rho. The DPD Gamma is a research surface and is allowed to
            // differ from lavaan here.
            auto mlch_or = polyserial_pair_scores(
                cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
            if (!mlch_or.has_value()) return std::unexpected(mlch_or.error());
            A21(assoc_count, mu_col) = ps.rho.dot(mlch_or->mu_unit) / sd;
            A21(assoc_count, var_col) = ps.rho.dot(mlch_or->var_unit) / var(c);
            block_diag.dpd_pairs.push_back(MixedPairLabel{
                .i = static_cast<std::int32_t>(i),
                .j = static_cast<std::int32_t>(j),
                .moment_index = static_cast<std::int32_t>(
                    nth + 2 * std::count(ordered[b].begin(), ordered[b].end(), 0) +
                    assoc_count),
                .kind = MixedPairKind::continuous_ordinal});
            block_diag.dpd_fits.push_back(std::move(*rho_or));
          } else {
            auto rho_or = fit_polyserial_pair_rho_ml(
                cat, U.col(c), th_by_var[static_cast<std::size_t>(o)]);
            if (!rho_or.has_value()) return std::unexpected(rho_or.error());
            const double rho = rho_or->rho;
            R(i, j) = R(j, i) = rho * sd;
            auto ps_or = polyserial_pair_scores(
                cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
            if (!ps_or.has_value()) return std::unexpected(ps_or.error());
            const auto& ps = *ps_or;
            assoc_scores.push_back(ps.rho);
            A21.block(assoc_count, so, 1, ps.thresholds.cols()) =
                assoc_scores.back().transpose() * ps.thresholds;
            A21(assoc_count, mu_col) = ps.rho.dot(ps.mu_unit) / sd;
            A21(assoc_count, var_col) = ps.rho.dot(ps.var_unit) / var(c);
          }
        } else {
          double cov = 0.0;
          for (Eigen::Index r = 0; r < n; ++r) {
            cov += (X(r, i) - mean(i)) * (X(r, j) - mean(j));
          }
          cov /= static_cast<double>(n);
          R(i, j) = R(j, i) = cov;
          // Bivariate normal correlation ML scores: take the covariance-
          // metric pair scores and chain-rule into the (mu, sigma^2, rho)
          // parameterization used by the stage-2 sandwich.
          auto sc_or = continuous_pair_normal_scores(
              X.col(i), X.col(j), mean(i), mean(j), var(i), var(j), cov);
          if (!sc_or.has_value()) return std::unexpected(sc_or.error());
          const Eigen::MatrixXd& S = sc_or->score_contributions;
          const double sdi = std::sqrt(var(i));
          const double sdj = std::sqrt(var(j));
          const double rho = cov / (sdi * sdj);
          const Eigen::VectorXd s_rho = sdi * sdj * S.col(4);
          assoc_scores.push_back(s_rho);
          const Eigen::Index mui = nth + cont_pos[static_cast<std::size_t>(i)];
          const Eigen::Index muj = nth + cont_pos[static_cast<std::size_t>(j)];
          const Eigen::Index vari =
              nth + n_cont + cont_pos[static_cast<std::size_t>(i)];
          const Eigen::Index varj =
              nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
          const Eigen::VectorXd ch_var_i =
              S.col(2) + (rho * sdj / (2.0 * sdi)) * S.col(4);
          const Eigen::VectorXd ch_var_j =
              S.col(3) + (rho * sdi / (2.0 * sdj)) * S.col(4);
          A21(assoc_count, mui) = s_rho.dot(S.col(0));
          A21(assoc_count, muj) = s_rho.dot(S.col(1));
          A21(assoc_count, vari) = s_rho.dot(ch_var_i);
          A21(assoc_count, varj) = s_rho.dot(ch_var_j);
        }
        ++assoc_count;
      }
    }

    const Eigen::Index n_assoc = p * (p - 1) / 2;
    Eigen::MatrixXd SC_ASSOC(n, n_assoc);
    Eigen::VectorXd A22_diag(n_assoc);
    for (Eigen::Index k = 0; k < n_assoc; ++k) {
      SC_ASSOC.col(k) = assoc_scores[static_cast<std::size_t>(k)];
      A22_diag(k) = std::isfinite(A22_override(k))
          ? A22_override(k)
          : SC_ASSOC.col(k).squaredNorm();
      if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
        A22_diag(k) = 1.0;
      }
    }
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(s1 + n_assoc, s1 + n_assoc);
    B_inv.block(0, 0, s1, s1) = *A11_inv_or;
    B_inv.block(s1, 0, n_assoc, s1).noalias() =
        -A22_inv * A21.topRows(n_assoc) * (*A11_inv_or);
    B_inv.block(s1, s1, n_assoc, n_assoc) = A22_inv;
    Eigen::MatrixXd SC(n, s1 + n_assoc);
    SC.leftCols(s1) = SC1;
    SC.rightCols(n_assoc) = SC_ASSOC;
    Eigen::MatrixXd IF_est = static_cast<double>(n) * SC * B_inv.transpose();

    // Map the stage-1/stage-2 parameter influence onto the moment vector
    // [th | -mu | var | assoc], applying the delta rule cov = rho*sd_i*sd_j
    // for the association rows (lavaan's H matrix): the covariance-metric
    // influence picks up the post-sandwich variance influence columns, not
    // raw moment residuals.
    const Eigen::VectorXd moment = mixed_moment_vector(R, mean, ordered[b], th);
    Eigen::MatrixXd IF(n, moment.size());
    Eigen::Index pos = 0;
    IF.middleCols(pos, nth) = IF_est.leftCols(nth);
    pos += nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) =
            -IF_est.col(nth + cont_pos[static_cast<std::size_t>(j)]);
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) =
            IF_est.col(nth + n_cont + cont_pos[static_cast<std::size_t>(j)]);
      }
    }
    Eigen::Index assoc_pos = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        const Eigen::VectorXd rho_col = IF_est.col(s1 + assoc_pos);
        if (oi && oj) {
          IF.col(pos++) = rho_col;
        } else if (oi || oj) {
          const Eigen::Index c = oi ? j : i;
          const double sd = std::sqrt(var(c));
          const double rho = R(i, j) / sd;
          const Eigen::Index var_col =
              nth + n_cont + cont_pos[static_cast<std::size_t>(c)];
          IF.col(pos++) =
              sd * rho_col + (rho / (2.0 * sd)) * IF_est.col(var_col);
        } else {
          const double sdi = std::sqrt(var(i));
          const double sdj = std::sqrt(var(j));
          const double rho = R(i, j) / (sdi * sdj);
          const Eigen::Index vari =
              nth + n_cont + cont_pos[static_cast<std::size_t>(i)];
          const Eigen::Index varj =
              nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
          IF.col(pos++) = sdi * sdj * rho_col +
              (rho * sdj / (2.0 * sdi)) * IF_est.col(vari) +
              (rho * sdi / (2.0 * sdj)) * IF_est.col(varj);
        }
        ++assoc_pos;
      }
    }
    Eigen::MatrixXd NACOV = (IF.transpose() * IF) / static_cast<double>(n);
    NACOV = 0.5 * (NACOV + NACOV.transpose());

    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(NACOV.rows(), NACOV.cols());
    for (Eigen::Index k = 0; k < NACOV.rows(); ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                " has a non-positive NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    // The full-WLS weight is the NACOV inverse; it is needed only for full WLS
    // fitting, not for DWLS (which uses the diagonal W_dwls) or the robust
    // sandwich (which uses NACOV itself). DWLS-only callers pass
    // `full_wls_weight = false` to skip the O(m³) inverse entirely. Even when
    // requested, the inverse is non-fatal: at small N with many indicators the
    // mixed NACOV is often singular, so a failed inverse leaves W_wls empty and
    // an explicit WLS request reports it. This matches lavaan WLSMV, which fits
    // the diagonal weight regardless of NACOV rank.
    Eigen::MatrixXd W_wls;
    if (full_wls_weight) {
      if (auto W_wls_or =
              symmetric_inverse_pd(NACOV, "mixed ordinal NACOV matrix");
          W_wls_or.has_value()) {
        W_wls = std::move(*W_wls_or);
      }
    }

    if (use_polyserial_dpd) {
      block_diag.moment_influence = IF;
      block_diag.gamma = NACOV;
      out.block_diagnostics.push_back(std::move(block_diag));
    }

    stats.R.push_back(std::move(R));
    stats.mean.push_back(std::move(mean));
    stats.ordered.push_back(ordered[b]);
    stats.thresholds.push_back(std::move(th));
    stats.threshold_ov.push_back(std::move(th_ov));
    stats.threshold_level.push_back(std::move(th_level));
    stats.moments.push_back(std::move(moment));
    stats.NACOV.push_back(std::move(NACOV));
    stats.moment_influence.push_back(std::move(IF));
    stats.W_dwls.push_back(std::move(W_dwls));
    stats.W_wls.push_back(std::move(W_wls));
    stats.n_obs.push_back(static_cast<std::int64_t>(n));
    stats.n_levels.push_back(std::move(levels));
    if (!use_polyserial_dpd) stats.raw_data.push_back(X);
  }
  return out;
}

post_expected<MixedOrdinalStats>
mixed_ordinal_stats_from_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    const std::vector<std::vector<std::int32_t>>& ordered,
    bool full_wls_weight) {
  auto out = mixed_ordinal_stats_from_data_impl(Xs, ordered, false, {},
                                                full_wls_weight);
  if (!out.has_value()) return std::unexpected(out.error());
  return std::move(out->stats);
}

post_expected<MixedOrdinalPolyserialDpdStats>
mixed_ordinal_stats_polyserial_dpd_from_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    const std::vector<std::vector<std::int32_t>>& ordered,
    PolyserialPairDpdOptions options) {
  return mixed_ordinal_stats_from_data_impl(Xs, ordered, true, options);
}

namespace {

double mixed_polyserial_prob_unchecked(int cat,
                                       double u,
                                       double rho,
                                       const Eigen::VectorXd& th) noexcept {
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));
  const double lo = (cat == 0) ? -kInf : th(cat - 1);
  const double hi = (cat == th.size()) ? kInf : th(cat);
  return std::max(kProbFloor,
                  normal_cdf((hi - rho * u) / sd) -
                      normal_cdf((lo - rho * u) / sd));
}

double polyserial_huber_objective_unchecked(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    double rho,
    const HuberResidualClipOptions& clip) {
  if (clip.kind == HuberResidualClipKind::None) {
    double nll = 0.0;
    for (Eigen::Index r = 0; r < categories.size(); ++r) {
      nll -= std::log(mixed_polyserial_prob_unchecked(
          categories(r), u(r), rho, thresholds));
    }
    return nll / static_cast<double>(categories.size());
  }
  double out = 0.0;
  const Eigen::Index nlev = thresholds.size() + 1;
  for (Eigen::Index r = 0; r < categories.size(); ++r) {
    for (Eigen::Index c = 0; c < nlev; ++c) {
      const double p = mixed_polyserial_prob_unchecked(
          static_cast<int>(c), u(r), rho, thresholds);
      const double f = (categories(r) == c) ? 1.0 : 0.0;
      const double e = (f - p) / std::sqrt(std::max(kProbFloor, p));
      auto clipped = eval_huber_residual_clip(e, clip);
      if (!clipped.has_value()) return std::numeric_limits<double>::infinity();
      out += p * clipped->loss;
    }
  }
  return out / static_cast<double>(categories.size());
}

post_expected<PolyserialPairDpdResult>
fit_polyserial_pair_rho_huber_residual_local(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    const PolyserialPairHuberResidualOptions& options) {
  auto clip_ok = eval_huber_residual_clip(0.0, options.clip);
  if (!clip_ok.has_value()) return std::unexpected(clip_ok.error());
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_polyserial_pair_rho_huber_residual: invalid options"));
  }
  if (options.clip.kind == HuberResidualClipKind::None) {
    auto ml = fit_polyserial_pair_rho_ml(
        categories, u, thresholds,
        PolyserialPairMlOptions{.rho_lower = options.rho_lower,
                                .rho_upper = options.rho_upper,
                                .max_iter = options.max_iter});
    if (!ml.has_value()) return std::unexpected(ml.error());
    PolyserialPairDpdResult out;
    out.rho = ml->rho;
    out.objective = ml->negloglik / static_cast<double>(categories.size());
    out.iterations = ml->iterations;
    out.hit_lower = ml->hit_lower;
    out.hit_upper = ml->hit_upper;
    return out;
  }

  PolyserialPairDpdResult out;
  double lo = options.rho_lower;
  double hi = options.rho_upper;
  constexpr double invphi = 0.6180339887498948482;
  double c = hi - invphi * (hi - lo);
  double d = lo + invphi * (hi - lo);
  double fc = polyserial_huber_objective_unchecked(
      categories, u, thresholds, c, options.clip);
  double fd = polyserial_huber_objective_unchecked(
      categories, u, thresholds, d, options.clip);
  for (int it = 0; it < options.max_iter; ++it) {
    out.iterations = it + 1;
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - invphi * (hi - lo);
      fc = polyserial_huber_objective_unchecked(
          categories, u, thresholds, c, options.clip);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + invphi * (hi - lo);
      fd = polyserial_huber_objective_unchecked(
          categories, u, thresholds, d, options.clip);
    }
  }
  out.rho = 0.5 * (lo + hi);
  out.objective = polyserial_huber_objective_unchecked(
      categories, u, thresholds, out.rho, options.clip);
  const double width = options.rho_upper - options.rho_lower;
  out.hit_lower = std::abs(out.rho - options.rho_lower) <= 1e-8 * width;
  out.hit_upper = std::abs(out.rho - options.rho_upper) <= 1e-8 * width;
  return out;
}

PairwiseOrdinalCorrelationRepairOptions
pairwise_repair_from_mixed(MixedOrdinalCorrelationRepairOptions options) {
  PairwiseOrdinalCorrelationRepairKind kind =
      PairwiseOrdinalCorrelationRepairKind::None;
  switch (options.kind) {
    case MixedOrdinalCorrelationRepairKind::None:
      kind = PairwiseOrdinalCorrelationRepairKind::None;
      break;
    case MixedOrdinalCorrelationRepairKind::Error:
      kind = PairwiseOrdinalCorrelationRepairKind::Error;
      break;
    case MixedOrdinalCorrelationRepairKind::Ridge:
      kind = PairwiseOrdinalCorrelationRepairKind::Ridge;
      break;
    case MixedOrdinalCorrelationRepairKind::Shrinkage:
      kind = PairwiseOrdinalCorrelationRepairKind::Shrinkage;
      break;
  }
  return PairwiseOrdinalCorrelationRepairOptions{
      .kind = kind,
      .min_eigenvalue = options.min_eigenvalue};
}

Eigen::VectorXd polyserial_huber_scaled_rho_scores(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    double rho,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    const HuberResidualClipOptions& clip) {
  auto scores_or = polyserial_pair_scores(categories, u, rho, thresholds);
  if (!scores_or.has_value()) {
    return Eigen::VectorXd::Constant(categories.size(),
        std::numeric_limits<double>::quiet_NaN());
  }
  if (clip.kind == HuberResidualClipKind::None) return scores_or->rho;
  Eigen::VectorXd out = scores_or->rho;
  for (Eigen::Index r = 0; r < categories.size(); ++r) {
    const double p = mixed_polyserial_prob_unchecked(
        categories(r), u(r), rho, thresholds);
    const double e = (1.0 - p) / std::sqrt(std::max(kProbFloor, p));
    auto clipped = eval_huber_residual_clip(e, clip);
    if (!clipped.has_value()) {
      out(r) = std::numeric_limits<double>::quiet_NaN();
    } else {
      out(r) *= clipped->dpsi;
    }
  }
  out.array() -= out.mean();
  return out;
}

struct PolyserialHuberScoreBlock {
  Eigen::VectorXd rho;
  Eigen::MatrixXd thresholds;
};

PolyserialHuberScoreBlock polyserial_huber_scaled_scores(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    double rho,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    const HuberResidualClipOptions& clip) {
  auto scores_or = polyserial_pair_scores(categories, u, rho, thresholds);
  if (!scores_or.has_value()) {
    return PolyserialHuberScoreBlock{
        .rho = Eigen::VectorXd::Constant(categories.size(),
            std::numeric_limits<double>::quiet_NaN()),
        .thresholds = Eigen::MatrixXd::Constant(
            categories.size(), thresholds.size(),
            std::numeric_limits<double>::quiet_NaN())};
  }
  if (clip.kind == HuberResidualClipKind::None) {
    return PolyserialHuberScoreBlock{
        .rho = scores_or->rho,
        .thresholds = scores_or->thresholds};
  }

  PolyserialHuberScoreBlock out{
      .rho = scores_or->rho,
      .thresholds = scores_or->thresholds};
  for (Eigen::Index r = 0; r < categories.size(); ++r) {
    const double p = mixed_polyserial_prob_unchecked(
        categories(r), u(r), rho, thresholds);
    const double e = (1.0 - p) / std::sqrt(std::max(kProbFloor, p));
    auto clipped = eval_huber_residual_clip(e, clip);
    if (!clipped.has_value()) {
      out.rho(r) = std::numeric_limits<double>::quiet_NaN();
      out.thresholds.row(r).array() =
          std::numeric_limits<double>::quiet_NaN();
    } else {
      out.rho(r) *= clipped->dpsi;
      out.thresholds.row(r) *= clipped->dpsi;
    }
  }
  out.rho.array() -= out.rho.mean();
  out.thresholds.rowwise() -= out.thresholds.colwise().mean();
  return out;
}

post_expected<Eigen::MatrixXd> mixed_marginal_threshold_scores(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<Eigen::VectorXd>& th_by_var,
    const std::vector<Eigen::Index>& th_start,
    Eigen::Index nth) {
  Eigen::MatrixXd scores = Eigen::MatrixXd::Zero(X.rows(), nth);
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      if (ordered[static_cast<std::size_t>(j)] == 0) continue;
      const int c = static_cast<int>(X(r, j)) - 1;
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      if (c < 0 || c > thj.size()) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed Huber residual threshold scores: category outside range"));
      }
      const double lo = (c == 0) ? -kInf : thj(c - 1);
      const double hi = (c == thj.size()) ? kInf : thj(c);
      const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) scores(r, base + c) += normal_pdf(thj(c)) / pr;
      if (c > 0) scores(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
    }
  }
  return scores;
}

}  // namespace

post_expected<MixedOrdinalHuberResidualStats>
mixed_ordinal_stats_huber_residual_from_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    const std::vector<std::vector<std::int32_t>>& ordered,
    MixedOrdinalHuberResidualOptions options) {
  auto clip_ok = eval_huber_residual_clip(0.0, options.clip);
  if (!clip_ok.has_value()) return std::unexpected(clip_ok.error());
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1 ||
      !(std::isfinite(options.fd_step) && options.fd_step > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_huber_residual_from_data: invalid options"));
  }

  auto base_or = mixed_ordinal_stats_from_data_impl(
      Xs, ordered, true, PolyserialPairDpdOptions{.alpha = 0.0});
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  MixedOrdinalHuberResidualStats out;
  out.stats = std::move(base_or->stats);
  out.block_diagnostics.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    MixedOrdinalStats& stats = out.stats;
    Eigen::MatrixXd IF = base_or->block_diagnostics[b].moment_influence;
    Eigen::MatrixXd R = stats.R[b];
    Eigen::VectorXd thresholds = stats.thresholds[b];

    std::vector<Eigen::Index> ordinal_cols;
    std::vector<int> ordinal_pos(static_cast<std::size_t>(p), -1);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) {
        ordinal_pos[static_cast<std::size_t>(j)] =
            static_cast<int>(ordinal_cols.size());
        ordinal_cols.push_back(j);
      }
    }

    Eigen::MatrixXd ordinal_robust_if;
    bool have_ordinal_robust_if = false;
    double ordinal_raw_min_eigen = 0.0;
    double ordinal_min_eigen = 0.0;
    bool ordinal_repair_applied = false;
    double ordinal_ridge = 0.0;
    double ordinal_shrinkage = 0.0;
    if (ordinal_cols.size() >= 2) {
      Eigen::MatrixXd Xord(n, static_cast<Eigen::Index>(ordinal_cols.size()));
      for (Eigen::Index k = 0; k < Xord.cols(); ++k) {
        Xord.col(k) = X.col(ordinal_cols[static_cast<std::size_t>(k)]);
      }
      auto ord_or = pairwise_ordinal_stats_shared_robust_from_integer_data(
          {Xord}, SharedRobustOrdinalOptions{
                      .kind = SharedRobustOrdinalKind::HuberResidual,
                      .h_score = {},
                      .clip = options.clip,
                      .alpha = 0.0,
                      .rho_lower = options.rho_lower,
                      .rho_upper = options.rho_upper,
                      .max_iter = options.max_iter,
                      .ftol = options.ftol,
                      .gtol = options.gtol,
                      .fd_step = options.fd_step,
                      .min_threshold_spacing = options.min_threshold_spacing,
                      .lavaan_adjust_2x2 = options.lavaan_adjust_2x2,
                      .correlation_repair =
                          pairwise_repair_from_mixed(options.correlation_repair)});
      if (!ord_or.has_value()) return std::unexpected(ord_or.error());
      thresholds = ord_or->stats.thresholds[0];
      ordinal_robust_if = ord_or->block_diagnostics[0].moment_influence;
      have_ordinal_robust_if = true;
      ordinal_raw_min_eigen = ord_or->block_diagnostics[0].raw_min_eigen_r;
      ordinal_min_eigen = ord_or->block_diagnostics[0].min_eigen_r;
      ordinal_repair_applied = ord_or->block_diagnostics[0].r_repair_applied;
      ordinal_ridge = ord_or->block_diagnostics[0].r_ridge;
      ordinal_shrinkage =
          ord_or->block_diagnostics[0].r_shrinkage_intensity;
      IF.leftCols(thresholds.size()) =
          ordinal_robust_if.leftCols(thresholds.size());
      for (Eigen::Index oj = 0; oj < static_cast<Eigen::Index>(ordinal_cols.size()); ++oj) {
        for (Eigen::Index oi = oj + 1; oi < static_cast<Eigen::Index>(ordinal_cols.size()); ++oi) {
          const Eigen::Index i = ordinal_cols[static_cast<std::size_t>(oi)];
          const Eigen::Index j = ordinal_cols[static_cast<std::size_t>(oj)];
          R(i, j) = R(j, i) = ord_or->stats.R[0](oi, oj);
        }
      }
    }

    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
    for (Eigen::Index k = 0; k < thresholds.size(); ++k) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(k)];
      if (stats.threshold_level[b][static_cast<std::size_t>(k)] == 1) {
        th_start[static_cast<std::size_t>(ov)] = k;
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len =
          static_cast<Eigen::Index>(stats.n_levels[b][static_cast<std::size_t>(j)] - 1);
      th_by_var[static_cast<std::size_t>(j)] = thresholds.segment(start, len);
    }

    Eigen::VectorXd mean = stats.mean[b];
    Eigen::VectorXd var = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
      var(j) = R(j, j);
      const double sd = std::sqrt(var(j));
      U.col(j) = (X.col(j).array() - mean(j)) / sd;
    }

    MixedOrdinalHuberResidualBlockDiagnostics diag;
    diag.raw_min_eigen_r = ordinal_raw_min_eigen;
    diag.min_eigen_r = ordinal_min_eigen;
    diag.r_repair_applied = ordinal_repair_applied;
    diag.r_ridge = ordinal_ridge;
    diag.r_shrinkage_intensity = ordinal_shrinkage;
    std::vector<double> rho_diag;
    std::vector<double> obj_diag;
    Eigen::Index assoc_count = 0;
    const Eigen::Index nth = thresholds.size();
    const Eigen::Index n_cont = std::count(ordered[b].begin(), ordered[b].end(), 0);
    const Eigen::Index n_assoc = p * (p - 1) / 2;
    const bool rebuild_polyserial_bread = ordinal_cols.size() == 1;
    // The rebuilt stage-1 stack mirrors the base muthen1984-style sandwich
    // ([th | mu | var] with per-variable A11 blocks); at no-clip the Huber
    // scores equal the ML scores, so the rebuilt Gamma reproduces the base
    // construction exactly.
    std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
    {
      Eigen::Index cp = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] == 0) {
          cont_pos[static_cast<std::size_t>(j)] = cp++;
        }
      }
    }
    const Eigen::Index s1h = nth + 2 * n_cont;
    Eigen::MatrixXd SC1;
    Eigen::MatrixXd A11_inv;
    std::vector<Eigen::VectorXd> assoc_scores;
    Eigen::MatrixXd A21;
    Eigen::VectorXd A22_diag;
    if (rebuild_polyserial_bread) {
      auto sc_th_or = mixed_marginal_threshold_scores(
          X, ordered[b], th_by_var, th_start, nth);
      if (!sc_th_or.has_value()) return std::unexpected(sc_th_or.error());
      SC1 = Eigen::MatrixXd::Zero(n, s1h);
      SC1.leftCols(nth) = std::move(*sc_th_or);
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
        const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
        const double v = var(j);
        SC1.col(nth + cp) = (X.col(j).array() - mean(j)) / v;
        SC1.col(nth + n_cont + cp) =
            ((X.col(j).array() - mean(j)).square() - v) / (2.0 * v * v);
      }
      const Eigen::MatrixXd inner1 = SC1.transpose() * SC1;
      Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(s1h, s1h);
      for (Eigen::Index col : ordinal_cols) {
        const Eigen::Index start = th_start[static_cast<std::size_t>(col)];
        const Eigen::Index len = static_cast<Eigen::Index>(
            stats.n_levels[b][static_cast<std::size_t>(col)] - 1);
        A11.block(start, start, len, len) =
            inner1.block(start, start, len, len);
      }
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
        const Eigen::Index a = nth + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::Index v =
            nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
        A11(a, a) = inner1(a, a);
        A11(v, v) = inner1(v, v);
        A11(a, v) = A11(v, a) = inner1(a, v);
      }
      auto A11_inv_or = symmetric_inverse_pd(
          A11, "mixed Huber residual stage-1 information matrix");
      if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());
      A11_inv = std::move(*A11_inv_or);
      assoc_scores.assign(static_cast<std::size_t>(n_assoc),
                          Eigen::VectorXd::Zero(n));
      A21 = Eigen::MatrixXd::Zero(n_assoc, s1h);
      A22_diag = Eigen::VectorXd::Ones(n_assoc);
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        const Eigen::Index moment_index = nth + 2 * n_cont + assoc_count;
        if (oi && oj) {
          const int oi_pos = ordinal_pos[static_cast<std::size_t>(i)];
          const int oj_pos = ordinal_pos[static_cast<std::size_t>(j)];
          Eigen::Index ord_corr = 0;
          for (int cj = 0; cj < oj_pos; ++cj) {
            ord_corr += static_cast<Eigen::Index>(ordinal_cols.size()) - cj - 1;
          }
          ord_corr += oi_pos - oj_pos - 1;
          if (have_ordinal_robust_if) {
            IF.col(moment_index) =
                ordinal_robust_if.col(nth + ord_corr);
          }
          diag.robust_pairs.push_back(MixedPairLabel{
              .i = static_cast<std::int32_t>(i),
              .j = static_cast<std::int32_t>(j),
              .moment_index = static_cast<std::int32_t>(moment_index),
              .kind = MixedPairKind::ordinal_ordinal});
          rho_diag.push_back(R(i, j));
          obj_diag.push_back(0.0);
        } else if (oi || oj) {
          const Eigen::Index o = oi ? i : j;
          const Eigen::Index c = oi ? j : i;
          Eigen::VectorXi cat(n);
          for (Eigen::Index r = 0; r < n; ++r) {
            cat(r) = static_cast<int>(X(r, o)) - 1;
          }
          const double sd = std::sqrt(var(c));
          auto fit_or = fit_polyserial_pair_rho_huber_residual_local(
              cat, U.col(c), th_by_var[static_cast<std::size_t>(o)],
              PolyserialPairHuberResidualOptions{
                  .rho_lower = options.rho_lower,
                  .rho_upper = options.rho_upper,
                  .max_iter = options.max_iter,
                  .fd_step = options.fd_step,
                  .clip = options.clip});
          if (!fit_or.has_value()) return std::unexpected(fit_or.error());
          const double rho = fit_or->rho;
          R(i, j) = R(j, i) = rho * sd;
          Eigen::VectorXd sc = polyserial_huber_scaled_rho_scores(
              cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)],
              options.clip);
          if (!sc.allFinite()) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_stats_huber_residual_from_data: non-finite polyserial scores"));
          }
          double bread = sc.squaredNorm();
          if (!(bread > 1e-12) || !std::isfinite(bread)) bread = 1.0;
          Eigen::VectorXd col = static_cast<double>(n) * sc / bread * sd;
          col.array() += (rho / (2.0 * sd)) *
              ((X.col(c).array() - mean(c)).square() - var(c));
          IF.col(moment_index) = std::move(col);
          if (rebuild_polyserial_bread) {
            auto score_block = polyserial_huber_scaled_scores(
                cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)],
                options.clip);
            if (!score_block.rho.allFinite() ||
                !score_block.thresholds.allFinite()) {
              return std::unexpected(make_err(PostError::Kind::NumericIssue,
                  "mixed_ordinal_stats_huber_residual_from_data: non-finite polyserial sandwich scores"));
            }
            const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
            assoc_scores[static_cast<std::size_t>(assoc_count)] =
                score_block.rho;
            A21.block(assoc_count, so, 1, score_block.thresholds.cols()) =
                score_block.rho.transpose() * score_block.thresholds;
            // mu/var coupling channels use the ML pair-score units at the
            // Huber rho (exact at no-clip; an estimating-equation
            // approximation under clipping).
            auto mlch_or = polyserial_pair_scores(
                cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
            if (!mlch_or.has_value()) return std::unexpected(mlch_or.error());
            A21(assoc_count, nth + cont_pos[static_cast<std::size_t>(c)]) =
                score_block.rho.dot(mlch_or->mu_unit) / sd;
            A21(assoc_count,
                nth + n_cont + cont_pos[static_cast<std::size_t>(c)]) =
                score_block.rho.dot(mlch_or->var_unit) / var(c);
            const double robust_bread = score_block.rho.squaredNorm();
            A22_diag(assoc_count) =
                (robust_bread > 1e-12 && std::isfinite(robust_bread))
                    ? robust_bread
                    : 1.0;
          }
          diag.robust_pairs.push_back(MixedPairLabel{
              .i = static_cast<std::int32_t>(i),
              .j = static_cast<std::int32_t>(j),
              .moment_index = static_cast<std::int32_t>(moment_index),
              .kind = MixedPairKind::continuous_ordinal});
          rho_diag.push_back(rho);
          obj_diag.push_back(fit_or->objective);
        }
        ++assoc_count;
      }
    }

    if (rebuild_polyserial_bread) {
      Eigen::MatrixXd SC = Eigen::MatrixXd::Zero(n, s1h + n_assoc);
      SC.leftCols(s1h) = SC1;
      for (Eigen::Index k = 0; k < n_assoc; ++k) {
        SC.col(s1h + k) = assoc_scores[static_cast<std::size_t>(k)];
      }
      Eigen::MatrixXd B_inv =
          Eigen::MatrixXd::Zero(s1h + n_assoc, s1h + n_assoc);
      const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
      B_inv.block(0, 0, s1h, s1h) = A11_inv;
      B_inv.block(s1h, 0, n_assoc, s1h).noalias() =
          -A22_inv * A21 * A11_inv;
      B_inv.block(s1h, s1h, n_assoc, n_assoc) = A22_inv;
      Eigen::MatrixXd IF_est =
          static_cast<double>(n) * SC * B_inv.transpose();
      IF.leftCols(nth) = IF_est.leftCols(nth);
      Eigen::Index assoc_pos = 0;
      for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = j + 1; i < p; ++i) {
          const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
          const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
          if (oi != oj) {
            const Eigen::Index moment_index = nth + 2 * n_cont + assoc_pos;
            const Eigen::Index c = oi ? j : i;
            const double sd = std::sqrt(var(c));
            const double rho = R(i, j) / sd;
            IF.col(moment_index) = sd * IF_est.col(s1h + assoc_pos) +
                (rho / (2.0 * sd)) *
                    IF_est.col(nth + n_cont +
                               cont_pos[static_cast<std::size_t>(c)]);
          }
          ++assoc_pos;
        }
      }
    }

    Eigen::MatrixXd NACOV = (IF.transpose() * IF) / static_cast<double>(n);
    NACOV = 0.5 * (NACOV + NACOV.transpose());
    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(NACOV.rows(), NACOV.cols());
    for (Eigen::Index k = 0; k < NACOV.rows(); ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_huber_residual_from_data: non-positive Gamma diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    auto W_wls_or = symmetric_inverse_pd(NACOV, "mixed Huber residual Gamma");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());

    stats.thresholds[b] = std::move(thresholds);
    stats.R[b] = std::move(R);
    stats.moments[b] = mixed_moment_vector(
        stats.R[b], stats.mean[b], ordered[b], stats.thresholds[b]);
    stats.NACOV[b] = NACOV;
    stats.W_dwls[b] = std::move(W_dwls);
    stats.W_wls[b] = std::move(*W_wls_or);
    diag.rho = Eigen::Map<Eigen::VectorXd>(
        rho_diag.data(), static_cast<Eigen::Index>(rho_diag.size()));
    diag.objective = Eigen::Map<Eigen::VectorXd>(
        obj_diag.data(), static_cast<Eigen::Index>(obj_diag.size()));
    diag.moment_influence = std::move(IF);
    diag.gamma = stats.NACOV[b];
    out.block_diagnostics.push_back(std::move(diag));
  }

  return out;
}

}  // namespace magmaan::data
