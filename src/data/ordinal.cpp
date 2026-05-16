#include "magmaan/data/ordinal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/data/pairwise_mixed.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"

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
  if (A.rows() != A.cols() || !A.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not a finite square matrix"));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      0.5 * (A + A.transpose()));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " eigendecomposition failed"));
  }
  const double max_eval = es.eigenvalues().maxCoeff();
  const double tol = 1e-10 * std::max(1.0, max_eval);
  if (es.eigenvalues().minCoeff() <= tol) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not positive definite"));
  }
  Eigen::VectorXd inv = es.eigenvalues().cwiseInverse();
  return es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
}

Eigen::MatrixXd expected_pair_counts(double total,
                                     const Eigen::VectorXd& th_i,
                                     const Eigen::VectorXd& th_j,
                                     double rho) {
  Eigen::MatrixXd out(th_i.size() + 1, th_j.size() + 1);
  for (Eigen::Index a = 0; a < out.rows(); ++a) {
    const double lo_i = (a == 0) ? -kInf : th_i(a - 1);
    const double hi_i = (a + 1 == out.rows()) ? kInf : th_i(a);
    for (Eigen::Index c = 0; c < out.cols(); ++c) {
      const double lo_j = (c == 0) ? -kInf : th_j(c - 1);
      const double hi_j = (c + 1 == out.cols()) ? kInf : th_j(c);
      out(a, c) = ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
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

}  // namespace

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& Xs) {
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
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> r_es(
        0.5 * (R + R.transpose()));
    if (r_es.info() != Eigen::Success || !r_es.eigenvalues().allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_stats_from_integer_data: block " + std::to_string(b) +
              " polychoric R eigendecomposition failed"));
    }
    block_diag.min_eigen_r = r_es.eigenvalues().minCoeff();

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
    auto W_wls_or = symmetric_inverse_pd(
        NACOV, "ordinal NACOV matrix");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());
    block_diag.moment_influence = std::move(IF);
    block_diag.gamma = NACOV;

    out.stats.R.push_back(std::move(R));
    out.stats.thresholds.push_back(std::move(th));
    out.stats.threshold_ov.push_back(std::move(th_ov));
    out.stats.threshold_level.push_back(std::move(th_level));
    out.stats.NACOV.push_back(std::move(NACOV));
    out.stats.W_dwls.push_back(std::move(W_dwls));
    out.stats.W_wls.push_back(std::move(*W_wls_or));
    out.stats.n_obs.push_back(static_cast<std::int64_t>(n));
    out.stats.n_levels.push_back(std::move(levels));
    out.block_diagnostics.push_back(std::move(block_diag));
  }
  return out;
}

post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& Xs) {
  auto out = pairwise_ordinal_stats_from_integer_data(Xs);
  if (!out.has_value()) return std::unexpected(out.error());
  return std::move(out->stats);
}

post_expected<PairwiseOrdinalStats>
pairwise_ordinal_stats_h_weighted_from_integer_data(
    const std::vector<Eigen::MatrixXd>& Xs,
    PairwiseOrdinalHWeightedStatsOptions options) {
  if (!(std::isfinite(options.influence_fd_step) &&
        options.influence_fd_step > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise_ordinal_stats_h_weighted_from_integer_data: invalid options"));
  }
  auto h_ok = eval_polychoric_h_score(1.0, options.rho.h_score);
  if (!h_ok.has_value()) return std::unexpected(h_ok.error());

  auto base_or = pairwise_ordinal_stats_from_integer_data(Xs);
  if (!base_or.has_value()) return std::unexpected(base_or.error());
  PairwiseOrdinalStats out = std::move(*base_or);

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    const Eigen::Index nth = out.stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;

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

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    Eigen::MatrixXd IF = Eigen::MatrixXd::Zero(n, mdim);
    if (nth > 0) {
      IF.leftCols(nth) =
          out.block_diagnostics[b].moment_influence.leftCols(nth);
    }

    std::vector<OrdinalPairDiagnostics> pair_diag;
    pair_diag.reserve(static_cast<std::size_t>(ncorr));
    Eigen::Index corr_idx = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const Eigen::VectorXi xi = Xcat.col(i);
        const Eigen::VectorXi xj = Xcat.col(j);
        auto tab_or = ordinal_pair_table(
            xi, xj, out.stats.n_levels[b][static_cast<std::size_t>(i)],
            out.stats.n_levels[b][static_cast<std::size_t>(j)]);
        if (!tab_or.has_value()) return std::unexpected(tab_or.error());
        auto rho_or = fit_ordinal_pair_rho_h_weighted(
            *tab_or, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)], options.rho);
        if (!rho_or.has_value()) return std::unexpected(rho_or.error());

        const double rho = rho_or->rho;
        R(i, j) = R(j, i) = rho;
        auto influence_or = ordinal_pair_h_weighted_influence(
            *tab_or, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)], rho,
            OrdinalPairHWeightedInfluenceOptions{
                .fd_step = options.influence_fd_step,
                .h_score = options.rho.h_score});
        if (!influence_or.has_value()) {
          return std::unexpected(influence_or.error());
        }
        auto scores_or = ordinal_pair_scores(
            xi, xj, rho, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        if (!scores_or.has_value()) return std::unexpected(scores_or.error());

        const Eigen::Index nth_i =
            th_by_var[static_cast<std::size_t>(i)].size();
        const Eigen::Index nth_j =
            th_by_var[static_cast<std::size_t>(j)].size();
        const Eigen::Index pair_nth = nth_i + nth_j;
        const Eigen::Index last = pair_nth;
        const double bread_rr = influence_or->bread(last, last);
        if (!std::isfinite(bread_rr) || std::abs(bread_rr) <= 1e-12) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "pairwise_ordinal_stats_h_weighted_from_integer_data: singular rho bread"));
        }

        Eigen::VectorXd psi_rho(n);
        for (Eigen::Index row = 0; row < n; ++row) {
          const int ci = xi(row);
          const int cj = xj(row);
          psi_rho(row) =
              influence_or->weights(ci, cj) * scores_or->rho(row);
        }
        Eigen::VectorXd threshold_adjust = Eigen::VectorXd::Zero(n);
        if (pair_nth > 0) {
          Eigen::MatrixXd pair_if = Eigen::MatrixXd::Zero(n, pair_nth);
          pair_if.leftCols(nth_i) =
              IF.block(0, th_start[static_cast<std::size_t>(i)], n, nth_i);
          pair_if.rightCols(nth_j) =
              IF.block(0, th_start[static_cast<std::size_t>(j)], n, nth_j);
          threshold_adjust =
              pair_if * influence_or->bread.row(last).head(pair_nth).transpose();
        }
        IF.col(nth + corr_idx) =
            -(psi_rho - threshold_adjust) / bread_rr;

        pair_diag.push_back(OrdinalPairDiagnostics{
            .label = OrdinalPairLabel{
                .block = static_cast<std::int32_t>(b),
                .i = static_cast<std::int32_t>(i),
                .j = static_cast<std::int32_t>(j),
                .n_levels_i = out.stats.n_levels[b][static_cast<std::size_t>(i)],
                .n_levels_j = out.stats.n_levels[b][static_cast<std::size_t>(j)]},
            .rho = rho,
            .negloglik = rho_or->objective,
            .objective = rho_or->objective,
            .score = rho_or->score,
            .iterations = rho_or->iterations,
            .h_weighted = true,
            .converged = rho_or->converged,
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
            .expected_counts = rho_or->expected_counts,
            .residual_counts = rho_or->residual_counts,
            .pearson_residuals = rho_or->pearson_residuals,
            .weights = rho_or->weights});
        ++corr_idx;
      }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> r_es(
        0.5 * (R + R.transpose()));
    if (r_es.info() != Eigen::Success || !r_es.eigenvalues().allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise_ordinal_stats_h_weighted_from_integer_data: block " +
              std::to_string(b) + " robust R eigendecomposition failed"));
    }

    Eigen::MatrixXd gamma =
        (IF.transpose() * IF) / static_cast<double>(n);
    gamma = 0.5 * (gamma + gamma.transpose());
    if (!gamma.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise_ordinal_stats_h_weighted_from_integer_data: non-finite Gamma"));
    }
    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = gamma(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "pairwise_ordinal_stats_h_weighted_from_integer_data: non-positive Gamma diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    auto W_wls_or = symmetric_inverse_pd(gamma, "h-weighted ordinal Gamma");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());

    out.stats.R[b] = std::move(R);
    out.stats.NACOV[b] = gamma;
    out.stats.W_dwls[b] = std::move(W_dwls);
    out.stats.W_wls[b] = std::move(*W_wls_or);
    out.block_diagnostics[b].pair_diagnostics = std::move(pair_diag);
    out.block_diagnostics[b].moment_influence = std::move(IF);
    out.block_diagnostics[b].gamma = out.stats.NACOV[b];
    out.block_diagnostics[b].min_eigen_r = r_es.eigenvalues().minCoeff();
  }

  return out;
}

post_expected<MixedOrdinalStats>
mixed_ordinal_stats_from_data(const std::vector<Eigen::MatrixXd>& Xs,
                              const std::vector<std::vector<std::int32_t>>& ordered) {
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_data: no data blocks"));
  }
  if (Xs.size() != ordered.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_data: ordered mask block count mismatch"));
  }

  MixedOrdinalStats out;
  out.R.reserve(Xs.size());
  out.mean.reserve(Xs.size());
  out.ordered.reserve(Xs.size());
  out.thresholds.reserve(Xs.size());
  out.threshold_ov.reserve(Xs.size());
  out.threshold_level.reserve(Xs.size());
  out.moments.reserve(Xs.size());
  out.NACOV.reserve(Xs.size());
  out.W_dwls.reserve(Xs.size());
  out.W_wls.reserve(Xs.size());
  out.n_obs.reserve(Xs.size());
  out.n_levels.reserve(Xs.size());

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

    Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
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
    const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len = static_cast<Eigen::Index>(
          levels[static_cast<std::size_t>(j)] - 1);
      A11.block(start, start, len, len) =
          INNER_TH.block(start, start, len, len);
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "mixed ordinal threshold information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) R(j, j) = var(j);
    }

    std::vector<Eigen::VectorXd> assoc_scores;
    std::vector<Eigen::VectorXd> assoc_if_scale;
    Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(p * (p - 1) / 2, nth);
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
          assoc_if_scale.push_back(Eigen::VectorXd::Ones(n));
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
          auto rho_or = fit_polyserial_pair_rho_ml(
              cat, U.col(c), th_by_var[static_cast<std::size_t>(o)]);
          if (!rho_or.has_value()) return std::unexpected(rho_or.error());
          const double rho = rho_or->rho;
          const double sd = std::sqrt(var(c));
          R(i, j) = R(j, i) = rho * sd;
          auto ps_or = polyserial_pair_scores(
              cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
          if (!ps_or.has_value()) return std::unexpected(ps_or.error());
          const auto& ps = *ps_or;
          assoc_scores.push_back(ps.rho);
          assoc_if_scale.push_back(Eigen::VectorXd::Constant(n, sd));
          const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
          A21.block(assoc_count, so, 1, ps.thresholds.cols()) =
              ps.rho.transpose() * ps.thresholds;
        } else {
          double cov = 0.0;
          for (Eigen::Index r = 0; r < n; ++r) {
            cov += (X(r, i) - mean(i)) * (X(r, j) - mean(j));
          }
          cov /= static_cast<double>(n);
          R(i, j) = R(j, i) = cov;
          assoc_scores.push_back(Eigen::VectorXd::Zero(n));
          assoc_if_scale.push_back(Eigen::VectorXd::Zero(n));
        }
        ++assoc_count;
      }
    }

    const Eigen::Index n_assoc = p * (p - 1) / 2;
    Eigen::MatrixXd SC_ASSOC(n, n_assoc);
    Eigen::VectorXd A22_diag(n_assoc);
    for (Eigen::Index k = 0; k < n_assoc; ++k) {
      SC_ASSOC.col(k) = assoc_scores[static_cast<std::size_t>(k)];
      A22_diag(k) = SC_ASSOC.col(k).squaredNorm();
      if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
        A22_diag(k) = 1.0;
      }
    }
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(nth + n_assoc, nth + n_assoc);
    B_inv.block(0, 0, nth, nth) = *A11_inv_or;
    B_inv.block(nth, 0, n_assoc, nth).noalias() =
        -A22_inv * A21.topRows(n_assoc) * (*A11_inv_or);
    B_inv.block(nth, nth, n_assoc, n_assoc) = A22_inv;
    Eigen::MatrixXd SC(n, nth + n_assoc);
    SC.leftCols(nth) = SC_TH;
    SC.rightCols(n_assoc) = SC_ASSOC;
    Eigen::MatrixXd IF_est = static_cast<double>(n) * SC * B_inv.transpose();
    for (Eigen::Index k = 0; k < n_assoc; ++k) {
      const Eigen::Index row = nth + k;
      IF_est.col(row).array() *= assoc_if_scale[static_cast<std::size_t>(k)].array();
    }

    const Eigen::VectorXd moment = mixed_moment_vector(R, mean, ordered[b], th);
    Eigen::MatrixXd IF(n, moment.size());
    Eigen::Index pos = 0;
    IF.middleCols(pos, nth) = IF_est.leftCols(nth);
    pos += nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) = -(X.col(j).array() - mean(j)).matrix();
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) = (X.col(j).array() - mean(j)).square().matrix().array() - var(j);
      }
    }
    Eigen::Index assoc_pos = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        if (oi || oj) {
          IF.col(pos++) = IF_est.col(nth + assoc_pos);
        } else {
          IF.col(pos++) =
              ((X.col(i).array() - mean(i)) * (X.col(j).array() - mean(j))).matrix().array() -
              R(i, j);
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
    auto W_wls_or = symmetric_inverse_pd(NACOV, "mixed ordinal NACOV matrix");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());

    out.R.push_back(std::move(R));
    out.mean.push_back(std::move(mean));
    out.ordered.push_back(ordered[b]);
    out.thresholds.push_back(std::move(th));
    out.threshold_ov.push_back(std::move(th_ov));
    out.threshold_level.push_back(std::move(th_level));
    out.moments.push_back(std::move(moment));
    out.NACOV.push_back(std::move(NACOV));
    out.W_dwls.push_back(std::move(W_dwls));
    out.W_wls.push_back(std::move(*W_wls_or));
    out.n_obs.push_back(static_cast<std::int64_t>(n));
    out.n_levels.push_back(std::move(levels));
  }
  return out;
}

}  // namespace magmaan::data
