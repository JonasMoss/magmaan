#include "magmaan/estimate/frontier/communality.hpp"

#include <algorithm>
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

namespace magmaan::estimate::frontier {

namespace {

constexpr double kZeroTol = 1e-12;

fit_expected<Eigen::VectorXd> num_error(std::string detail) {
  return std::unexpected(FitError{FitError::Kind::NumericIssue, std::move(detail)});
}

fit_expected<Eigen::MatrixXd> mat_error(std::string detail) {
  return std::unexpected(FitError{FitError::Kind::NumericIssue, std::move(detail)});
}

struct Pair {
  Eigen::Index i = 0;
  Eigen::Index j = 0;
};

struct ValidatedInput {
  Eigen::MatrixXd R;
  std::vector<std::vector<Eigen::Index>> blocks;
  std::vector<Eigen::Index> block_of;
};

std::int64_t pair_key(Eigen::Index a, Eigen::Index b, Eigen::Index p) {
  if (a > b) std::swap(a, b);
  return static_cast<std::int64_t>(a) * static_cast<std::int64_t>(p) +
         static_cast<std::int64_t>(b);
}

fit_expected<ValidatedInput>
validate_input(const Eigen::MatrixXd& S,
               const std::vector<std::int32_t>& block_of_indicator) {
  if (S.rows() != S.cols())
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality estimator: covariance matrix is not square"});
  if (!S.allFinite())
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality estimator: covariance matrix has non-finite values"});

  const Eigen::Index p = S.rows();
  if (p == 0)
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality estimator: empty covariance matrix"});
  if (static_cast<Eigen::Index>(block_of_indicator.size()) != p)
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality estimator: block vector length does not match S"});

  const double scale = std::max(1.0, S.cwiseAbs().maxCoeff());
  if ((S - S.transpose()).cwiseAbs().maxCoeff() > 1e-8 * scale)
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality estimator: covariance matrix is not symmetric"});

  std::vector<std::int32_t> labels;
  labels.reserve(block_of_indicator.size());
  std::vector<Eigen::Index> block_of(static_cast<std::size_t>(p), -1);
  for (Eigen::Index i = 0; i < p; ++i) {
    const std::int32_t label = block_of_indicator[static_cast<std::size_t>(i)];
    if (label < 0)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality estimator: block labels must be non-negative"});
    auto it = std::find(labels.begin(), labels.end(), label);
    if (it == labels.end()) {
      labels.push_back(label);
      block_of[static_cast<std::size_t>(i)] =
          static_cast<Eigen::Index>(labels.size() - 1);
    } else {
      block_of[static_cast<std::size_t>(i)] =
          static_cast<Eigen::Index>(std::distance(labels.begin(), it));
    }
  }

  std::vector<std::vector<Eigen::Index>> blocks(labels.size());
  for (Eigen::Index i = 0; i < p; ++i) {
    blocks[static_cast<std::size_t>(block_of[static_cast<std::size_t>(i)])]
        .push_back(i);
  }
  for (const auto& b : blocks) {
    if (b.size() < 3)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality estimator: every factor block needs at least 3 "
          "indicators"});
  }

  Eigen::VectorXd sd(p);
  for (Eigen::Index i = 0; i < p; ++i) {
    if (!(S(i, i) > 0.0) || !std::isfinite(S(i, i)))
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality estimator: covariance diagonal must be positive"});
    sd(i) = std::sqrt(S(i, i));
  }

  Eigen::MatrixXd R(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) R(i, j) = S(i, j) / (sd(i) * sd(j));
  }
  R.diagonal().setOnes();
  return ValidatedInput{std::move(R), std::move(blocks), std::move(block_of)};
}

fit_expected<Eigen::MatrixXd> correlation_direction(const Eigen::MatrixXd& S,
                                                    const Eigen::MatrixXd& dS,
                                                    const Eigen::MatrixXd& R) {
  if (dS.rows() != S.rows() || dS.cols() != S.cols())
    return mat_error("communality derivative: direction dimension mismatch");
  if (!dS.allFinite())
    return mat_error("communality derivative: direction has non-finite values");
  const double scale = std::max(1.0, dS.cwiseAbs().maxCoeff());
  if ((dS - dS.transpose()).cwiseAbs().maxCoeff() > 1e-8 * scale)
    return mat_error("communality derivative: direction is not symmetric");

  const Eigen::Index p = S.rows();
  Eigen::VectorXd sd(p);
  for (Eigen::Index i = 0; i < p; ++i) sd(i) = std::sqrt(S(i, i));

  Eigen::MatrixXd dR(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) {
      dR(i, j) = dS(i, j) / (sd(i) * sd(j)) -
                 0.5 * R(i, j) *
                     (dS(i, i) / S(i, i) + dS(j, j) / S(j, j));
    }
  }
  dR.diagonal().setZero();
  return dR;
}

Eigen::Index vech_size(Eigen::Index p) { return p * (p + 1) / 2; }

Eigen::Index vech_index(Eigen::Index r, Eigen::Index c, Eigen::Index p) {
  if (r < c) std::swap(r, c);
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

struct CorrelationJacobian {
  Eigen::MatrixXd dR_off;  // one row per unordered off-diagonal correlation
  std::vector<Eigen::Index> row_of_pair;
  Eigen::Index p = 0;
  Eigen::Index pstar = 0;
};

fit_expected<CorrelationJacobian>
correlation_jacobian(const Eigen::MatrixXd& S, const Eigen::MatrixXd& R) {
  const Eigen::Index p = S.rows();
  if (S.cols() != p || R.rows() != p || R.cols() != p)
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality Jacobian: correlation input dimension mismatch"});

  const Eigen::Index pstar = vech_size(p);
  const Eigen::Index n_off = p * (p - 1) / 2;
  CorrelationJacobian out;
  out.dR_off = Eigen::MatrixXd::Zero(n_off, pstar);
  out.row_of_pair.assign(static_cast<std::size_t>(p * p), -1);
  out.p = p;
  out.pstar = pstar;

  Eigen::VectorXd sd(p);
  for (Eigen::Index i = 0; i < p; ++i) {
    if (!std::isfinite(S(i, i)) || S(i, i) <= 0.0)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality Jacobian: non-positive indicator variance"});
    sd(i) = std::sqrt(S(i, i));
  }

  Eigen::Index row = 0;
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      out.row_of_pair[static_cast<std::size_t>(i * p + j)] = row;
      out.row_of_pair[static_cast<std::size_t>(j * p + i)] = row;
      out.dR_off(row, vech_index(j, i, p)) = 1.0 / (sd(i) * sd(j));
      out.dR_off(row, vech_index(i, i, p)) = -0.5 * R(i, j) / S(i, i);
      out.dR_off(row, vech_index(j, j, p)) = -0.5 * R(i, j) / S(j, j);
      ++row;
    }
  }
  return out;
}

Eigen::Index corr_row_index(const CorrelationJacobian& J,
                            Eigen::Index a,
                            Eigen::Index b) {
  if (a == b || a < 0 || b < 0 || a >= J.p || b >= J.p) return -1;
  return J.row_of_pair[static_cast<std::size_t>(a * J.p + b)];
}

void add_corr_row(Eigen::Ref<Eigen::RowVectorXd> out,
                  const CorrelationJacobian& J,
                  Eigen::Index a,
                  Eigen::Index b,
                  double scale) {
  const Eigen::Index row = corr_row_index(J, a, b);
  if (row >= 0 && scale != 0.0) out.noalias() += scale * J.dR_off.row(row);
}

void add_corr_active_row(Eigen::RowVectorXd& out,
                         const CorrelationJacobian& J,
                         const std::vector<Eigen::Index>& active_cols,
                         Eigen::Index a,
                         Eigen::Index b,
                         double scale) {
  const Eigen::Index row = corr_row_index(J, a, b);
  if (row < 0 || scale == 0.0) return;
  for (std::size_t k = 0; k < active_cols.size(); ++k)
    out(static_cast<Eigen::Index>(k)) +=
        scale * J.dR_off(row, active_cols[k]);
}

Eigen::RowVectorXd corr_active_row(
    const CorrelationJacobian& J,
    const std::vector<Eigen::Index>& active_cols,
    Eigen::Index a,
    Eigen::Index b) {
  Eigen::RowVectorXd out =
      Eigen::RowVectorXd::Zero(static_cast<Eigen::Index>(active_cols.size()));
  add_corr_active_row(out, J, active_cols, a, b, 1.0);
  return out;
}

fit_expected<Eigen::MatrixXd>
itemwise_h2_jacobian(const Eigen::MatrixXd& R,
                     const std::vector<std::vector<Eigen::Index>>& blocks,
                     const std::vector<Eigen::Index>& block_of,
                     CommunalityMethod method,
                     const CorrelationJacobian& J) {
  const Eigen::Index p = R.rows();
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(p, J.pstar);

  for (Eigen::Index i = 0; i < p; ++i) {
    const auto& idx = blocks[static_cast<std::size_t>(
        block_of[static_cast<std::size_t>(i)])];
    double ar_sum = 0.0;
    double rs_num = 0.0;
    double rs_den = 0.0;
    double ilm_num = 0.0;
    double ilm_den = 0.0;
    Eigen::RowVectorXd dar_sum = Eigen::RowVectorXd::Zero(J.pstar);
    Eigen::RowVectorXd drs_num = Eigen::RowVectorXd::Zero(J.pstar);
    Eigen::RowVectorXd drs_den = Eigen::RowVectorXd::Zero(J.pstar);
    Eigen::RowVectorXd dilm_num = Eigen::RowVectorXd::Zero(J.pstar);
    Eigen::RowVectorXd dilm_den = Eigen::RowVectorXd::Zero(J.pstar);
    Eigen::Index count = 0;

    for (std::size_t a_pos = 0; a_pos < idx.size(); ++a_pos) {
      const Eigen::Index j = idx[a_pos];
      if (j == i) continue;
      for (std::size_t b_pos = a_pos + 1; b_pos < idx.size(); ++b_pos) {
        const Eigen::Index k = idx[b_pos];
        if (k == i) continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        if (method == CommunalityMethod::AverageRatio) {
          if (std::abs(rjk) <= kZeroTol)
            return mat_error("communality AR Jacobian: zero triad denominator");
          const double ratio = rij * rik / rjk;
          ar_sum += ratio;
          add_corr_row(dar_sum, J, i, j, rik / rjk);
          add_corr_row(dar_sum, J, i, k, rij / rjk);
          add_corr_row(dar_sum, J, j, k, -ratio / rjk);
        }
        rs_num += rij * rik;
        rs_den += rjk;
        add_corr_row(drs_num, J, i, j, rik);
        add_corr_row(drs_num, J, i, k, rij);
        add_corr_row(drs_den, J, j, k, 1.0);

        ilm_num += rjk * rij * rik;
        ilm_den += rjk * rjk;
        add_corr_row(dilm_num, J, j, k, rij * rik);
        add_corr_row(dilm_num, J, i, j, rjk * rik);
        add_corr_row(dilm_num, J, i, k, rjk * rij);
        add_corr_row(dilm_den, J, j, k, 2.0 * rjk);
        ++count;
      }
    }

    if (count == 0)
      return mat_error("communality Jacobian: no triads for an indicator");
    switch (method) {
      case CommunalityMethod::AverageRatio:
        (void)ar_sum;
        out.row(i) = dar_sum / static_cast<double>(count);
        break;
      case CommunalityMethod::RatioOfSums:
        if (std::abs(rs_den) <= kZeroTol)
          return mat_error("communality RS Jacobian: zero triad denominator");
        out.row(i) =
            (drs_num * rs_den - rs_num * drs_den) / (rs_den * rs_den);
        break;
      case CommunalityMethod::TriadLeastSquares:
        if (ilm_den <= kZeroTol)
          return mat_error(
              "communality triad LS Jacobian: zero triad denominator");
        out.row(i) =
            (dilm_num * ilm_den - ilm_num * dilm_den) / (ilm_den * ilm_den);
        break;
      default:
        return mat_error("communality itemwise Jacobian: unsupported method");
    }
  }
  if (!out.allFinite())
    return mat_error("communality Jacobian: non-finite output");
  return out;
}

fit_expected<Eigen::MatrixXd>
anchor_triad_ls_h2_jacobian(
    const Eigen::MatrixXd& R,
    const std::vector<std::vector<Eigen::Index>>& blocks,
    const std::vector<Eigen::Index>& block_of,
    const CorrelationJacobian& J) {
  const Eigen::Index p = R.rows();
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(p, J.pstar);

  for (Eigen::Index i = 0; i < p; ++i) {
    const auto& same = blocks[static_cast<std::size_t>(
        block_of[static_cast<std::size_t>(i)])];
    double num = 0.0;
    double den = 0.0;
    Eigen::RowVectorXd dnum = Eigen::RowVectorXd::Zero(J.pstar);
    Eigen::RowVectorXd dden = Eigen::RowVectorXd::Zero(J.pstar);
    Eigen::Index count = 0;

    for (std::size_t a_pos = 0; a_pos < same.size(); ++a_pos) {
      const Eigen::Index j = same[a_pos];
      if (j == i) continue;

      for (std::size_t b_pos = a_pos + 1; b_pos < same.size(); ++b_pos) {
        const Eigen::Index k = same[b_pos];
        if (k == i) continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        num += rjk * rij * rik;
        den += rjk * rjk;
        add_corr_row(dnum, J, j, k, rij * rik);
        add_corr_row(dnum, J, i, j, rjk * rik);
        add_corr_row(dnum, J, i, k, rjk * rij);
        add_corr_row(dden, J, j, k, 2.0 * rjk);
        ++count;
      }

      for (Eigen::Index k = 0; k < p; ++k) {
        if (k == i || k == j) continue;
        if (block_of[static_cast<std::size_t>(k)] ==
            block_of[static_cast<std::size_t>(i)])
          continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        num += rjk * rij * rik;
        den += rjk * rjk;
        add_corr_row(dnum, J, j, k, rij * rik);
        add_corr_row(dnum, J, i, j, rjk * rik);
        add_corr_row(dnum, J, i, k, rjk * rij);
        add_corr_row(dden, J, j, k, 2.0 * rjk);
        ++count;
      }
    }

    if (count == 0)
      return mat_error("communality anchor triad LS Jacobian: no anchor rows");
    if (den <= kZeroTol)
      return mat_error(
          "communality anchor triad LS Jacobian: zero anchor denominator");
    out.row(i) = (dnum * den - num * dden) / (den * den);
  }
  if (!out.allFinite())
    return mat_error("communality anchor triad LS Jacobian: non-finite output");
  return out;
}

fit_expected<Eigen::MatrixXd> symmetric_pinv(const Eigen::MatrixXd& A) {
  if (A.rows() != A.cols())
    return mat_error("communality GMM: pseudo-inverse input is not square");
  if (!A.allFinite())
    return mat_error("communality GMM: pseudo-inverse input has non-finite values");
  if (A.size() == 0) return Eigen::MatrixXd(A.rows(), A.cols());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A + A.transpose()));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite())
    return mat_error("communality GMM: pseudo-inverse eigendecomposition failed");

  const Eigen::VectorXd& evals = es.eigenvalues();
  const double max_abs = evals.cwiseAbs().maxCoeff();
  const double cutoff =
      std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, max_abs);

  Eigen::VectorXd inv = Eigen::VectorXd::Zero(evals.size());
  for (Eigen::Index i = 0; i < evals.size(); ++i) {
    if (evals(i) > cutoff) inv(i) = 1.0 / evals(i);
  }
  return es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
}

fit_expected<Eigen::MatrixXd> full_rank_inverse(const Eigen::MatrixXd& A) {
  if (A.rows() != A.cols())
    return mat_error("communality GMM: inverse input is not square");
  if (!A.allFinite())
    return mat_error("communality GMM: inverse input has non-finite values");

  Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
  lu.setThreshold(1e-12);
  if (lu.rank() != A.cols())
    return mat_error("communality GMM: inverse input is rank deficient");
  Eigen::MatrixXd inv =
      lu.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
  if (!inv.allFinite())
    return mat_error("communality GMM: non-finite inverse");
  return inv;
}

fit_expected<Eigen::MatrixXd>
row_covariance_pinv(const Eigen::MatrixXd& M,
                    const Eigen::MatrixXd& gamma,
                    const Eigen::MatrixXd& omega) {
  if (M.rows() != omega.rows() || omega.rows() != omega.cols() ||
      M.cols() != gamma.rows() || gamma.rows() != gamma.cols())
    return mat_error("communality GMM: row covariance dimension mismatch");
  if (!M.allFinite() || !gamma.allFinite() || !omega.allFinite())
    return mat_error("communality GMM: non-finite row covariance input");
  if (M.cols() == 0) return symmetric_pinv(omega);

  const Eigen::MatrixXd gram = M.transpose() * M;
  auto gram_inv = full_rank_inverse(gram);
  auto gamma_inv = full_rank_inverse(gamma);
  if (!gram_inv.has_value() || !gamma_inv.has_value())
    return symmetric_pinv(omega);

  Eigen::MatrixXd W =
      M * (*gram_inv) * (*gamma_inv) * (*gram_inv) * M.transpose();
  W = 0.5 * (W + W.transpose());
  if (!W.allFinite()) return symmetric_pinv(omega);
  return W;
}

struct PinvDirection {
  Eigen::MatrixXd value;
  Eigen::MatrixXd deriv;
};

fit_expected<PinvDirection>
symmetric_pinv_directional(const Eigen::MatrixXd& A, const Eigen::MatrixXd& dA) {
  auto W = symmetric_pinv(A);
  if (!W.has_value()) return std::unexpected(W.error());
  const Eigen::Index n = A.rows();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  Eigen::MatrixXd dW =
      -(*W) * dA * (*W) +
      (*W) * (*W) * dA * (I - A * (*W)) +
      (I - (*W) * A) * dA * (*W) * (*W);
  dW = 0.5 * (dW + dW.transpose());
  if (!dW.allFinite())
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality GMM derivative: non-finite pseudo-inverse derivative"});
  return PinvDirection{std::move(*W), std::move(dW)};
}

fit_expected<Eigen::VectorXd>
itemwise_h2(const Eigen::MatrixXd& R,
            const std::vector<std::vector<Eigen::Index>>& blocks,
            const std::vector<Eigen::Index>& block_of,
            CommunalityMethod method) {
  const Eigen::Index p = R.rows();
  Eigen::VectorXd h2(p);

  for (Eigen::Index i = 0; i < p; ++i) {
    const auto& idx = blocks[static_cast<std::size_t>(
        block_of[static_cast<std::size_t>(i)])];
    double ar_sum = 0.0;
    double rs_num = 0.0;
    double rs_den = 0.0;
    double ilm_num = 0.0;
    double ilm_den = 0.0;
    Eigen::Index count = 0;

    for (std::size_t a_pos = 0; a_pos < idx.size(); ++a_pos) {
      const Eigen::Index j = idx[a_pos];
      if (j == i) continue;
      for (std::size_t b_pos = a_pos + 1; b_pos < idx.size(); ++b_pos) {
        const Eigen::Index k = idx[b_pos];
        if (k == i) continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        if (!std::isfinite(rij) || !std::isfinite(rik) || !std::isfinite(rjk))
          return num_error("communality estimator: non-finite correlation");
        if (method == CommunalityMethod::AverageRatio) {
          if (std::abs(rjk) <= kZeroTol)
            return num_error("communality AR: zero triad denominator");
          ar_sum += rij * rik / rjk;
        }
        rs_num += rij * rik;
        rs_den += rjk;
        ilm_num += rjk * rij * rik;
        ilm_den += rjk * rjk;
        ++count;
      }
    }

    if (count == 0)
      return num_error("communality estimator: no triads for an indicator");
    switch (method) {
      case CommunalityMethod::AverageRatio:
        h2(i) = ar_sum / static_cast<double>(count);
        break;
      case CommunalityMethod::RatioOfSums:
        if (std::abs(rs_den) <= kZeroTol)
          return num_error("communality RS: zero triad denominator");
        h2(i) = rs_num / rs_den;
        break;
      case CommunalityMethod::TriadLeastSquares:
        if (ilm_den <= kZeroTol)
          return num_error("communality triad LS: zero triad denominator");
        h2(i) = ilm_num / ilm_den;
        break;
      default:
        return num_error("communality itemwise: unsupported method");
    }
  }
  if (!h2.allFinite())
    return num_error("communality estimator: non-finite output");
  return h2;
}

fit_expected<Eigen::VectorXd>
itemwise_h2_directional(const Eigen::MatrixXd& R,
                        const Eigen::MatrixXd& dR,
                        const std::vector<std::vector<Eigen::Index>>& blocks,
                        const std::vector<Eigen::Index>& block_of,
                        CommunalityMethod method) {
  const Eigen::Index p = R.rows();
  Eigen::VectorXd dh2 = Eigen::VectorXd::Zero(p);

  for (Eigen::Index i = 0; i < p; ++i) {
    const auto& idx = blocks[static_cast<std::size_t>(
        block_of[static_cast<std::size_t>(i)])];
    double ar_sum = 0.0;
    double rs_num = 0.0;
    double rs_den = 0.0;
    double ilm_num = 0.0;
    double ilm_den = 0.0;
    double dar_sum = 0.0;
    double drs_num = 0.0;
    double drs_den = 0.0;
    double dilm_num = 0.0;
    double dilm_den = 0.0;
    Eigen::Index count = 0;

    for (std::size_t a_pos = 0; a_pos < idx.size(); ++a_pos) {
      const Eigen::Index j = idx[a_pos];
      if (j == i) continue;
      for (std::size_t b_pos = a_pos + 1; b_pos < idx.size(); ++b_pos) {
        const Eigen::Index k = idx[b_pos];
        if (k == i) continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        const double drij = dR(i, j);
        const double drik = dR(i, k);
        const double drjk = dR(j, k);
        if (method == CommunalityMethod::AverageRatio) {
          if (std::abs(rjk) <= kZeroTol)
            return num_error("communality AR derivative: zero triad denominator");
          const double ratio = rij * rik / rjk;
          ar_sum += ratio;
          dar_sum += (drij * rik + rij * drik) / rjk -
                     ratio * drjk / rjk;
        }
        rs_num += rij * rik;
        rs_den += rjk;
        drs_num += drij * rik + rij * drik;
        drs_den += drjk;
        ilm_num += rjk * rij * rik;
        ilm_den += rjk * rjk;
        dilm_num += drjk * rij * rik + rjk * drij * rik + rjk * rij * drik;
        dilm_den += 2.0 * rjk * drjk;
        ++count;
      }
    }

    if (count == 0)
      return num_error("communality derivative: no triads for an indicator");
    switch (method) {
      case CommunalityMethod::AverageRatio:
        (void)ar_sum;
        dh2(i) = dar_sum / static_cast<double>(count);
        break;
      case CommunalityMethod::RatioOfSums:
        if (std::abs(rs_den) <= kZeroTol)
          return num_error("communality RS derivative: zero triad denominator");
        dh2(i) = (drs_num * rs_den - rs_num * drs_den) / (rs_den * rs_den);
        break;
      case CommunalityMethod::TriadLeastSquares:
        if (ilm_den <= kZeroTol)
          return num_error(
              "communality triad LS derivative: zero triad denominator");
        dh2(i) =
            (dilm_num * ilm_den - ilm_num * dilm_den) / (ilm_den * ilm_den);
        break;
      default:
        return num_error("communality itemwise derivative: unsupported method");
    }
  }
  if (!dh2.allFinite())
    return num_error("communality derivative: non-finite output");
  return dh2;
}

fit_expected<Eigen::VectorXd>
anchor_triad_ls_h2(const Eigen::MatrixXd& R,
                   const std::vector<std::vector<Eigen::Index>>& blocks,
                   const std::vector<Eigen::Index>& block_of) {
  const Eigen::Index p = R.rows();
  Eigen::VectorXd h2(p);

  for (Eigen::Index i = 0; i < p; ++i) {
    const auto& same = blocks[static_cast<std::size_t>(
        block_of[static_cast<std::size_t>(i)])];
    double num = 0.0;
    double den = 0.0;
    Eigen::Index count = 0;

    for (std::size_t a_pos = 0; a_pos < same.size(); ++a_pos) {
      const Eigen::Index j = same[a_pos];
      if (j == i) continue;

      for (std::size_t b_pos = a_pos + 1; b_pos < same.size(); ++b_pos) {
        const Eigen::Index k = same[b_pos];
        if (k == i) continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        if (!std::isfinite(rij) || !std::isfinite(rik) || !std::isfinite(rjk))
          return num_error("communality anchor triad LS: non-finite correlation");
        num += rjk * rij * rik;
        den += rjk * rjk;
        ++count;
      }

      for (Eigen::Index k = 0; k < p; ++k) {
        if (k == i || k == j) continue;
        if (block_of[static_cast<std::size_t>(k)] ==
            block_of[static_cast<std::size_t>(i)])
          continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        if (!std::isfinite(rij) || !std::isfinite(rik) || !std::isfinite(rjk))
          return num_error("communality anchor triad LS: non-finite correlation");
        num += rjk * rij * rik;
        den += rjk * rjk;
        ++count;
      }
    }

    if (count == 0)
      return num_error(
          "communality anchor triad LS: no anchor rows for an indicator");
    if (den <= kZeroTol)
      return num_error("communality anchor triad LS: zero anchor denominator");
    h2(i) = num / den;
  }
  if (!h2.allFinite())
    return num_error("communality anchor triad LS: non-finite output");
  return h2;
}

fit_expected<Eigen::VectorXd>
anchor_triad_ls_h2_directional(
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& dR,
    const std::vector<std::vector<Eigen::Index>>& blocks,
    const std::vector<Eigen::Index>& block_of) {
  const Eigen::Index p = R.rows();
  Eigen::VectorXd dh2(p);

  for (Eigen::Index i = 0; i < p; ++i) {
    const auto& same = blocks[static_cast<std::size_t>(
        block_of[static_cast<std::size_t>(i)])];
    double num = 0.0;
    double den = 0.0;
    double dnum = 0.0;
    double dden = 0.0;
    Eigen::Index count = 0;

    for (std::size_t a_pos = 0; a_pos < same.size(); ++a_pos) {
      const Eigen::Index j = same[a_pos];
      if (j == i) continue;

      for (std::size_t b_pos = a_pos + 1; b_pos < same.size(); ++b_pos) {
        const Eigen::Index k = same[b_pos];
        if (k == i) continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        const double drij = dR(i, j);
        const double drik = dR(i, k);
        const double drjk = dR(j, k);
        num += rjk * rij * rik;
        den += rjk * rjk;
        dnum += drjk * rij * rik + rjk * drij * rik + rjk * rij * drik;
        dden += 2.0 * rjk * drjk;
        ++count;
      }

      for (Eigen::Index k = 0; k < p; ++k) {
        if (k == i || k == j) continue;
        if (block_of[static_cast<std::size_t>(k)] ==
            block_of[static_cast<std::size_t>(i)])
          continue;
        const double rij = R(i, j);
        const double rik = R(i, k);
        const double rjk = R(j, k);
        const double drij = dR(i, j);
        const double drik = dR(i, k);
        const double drjk = dR(j, k);
        num += rjk * rij * rik;
        den += rjk * rjk;
        dnum += drjk * rij * rik + rjk * drij * rik + rjk * rij * drik;
        dden += 2.0 * rjk * drjk;
        ++count;
      }
    }

    if (count == 0)
      return num_error(
          "communality anchor triad LS derivative: no anchor rows");
    if (den <= kZeroTol)
      return num_error(
          "communality anchor triad LS derivative: zero anchor denominator");
    dh2(i) = (dnum * den - num * dden) / (den * den);
  }
  if (!dh2.allFinite())
    return num_error("communality anchor triad LS derivative: non-finite output");
  return dh2;
}

std::vector<Pair>
within_off_pairs(const std::vector<std::vector<Eigen::Index>>& blocks) {
  std::vector<Pair> out;
  for (const auto& idx : blocks) {
    for (std::size_t a = 0; a < idx.size(); ++a) {
      for (std::size_t b = a + 1; b < idx.size(); ++b) {
        out.push_back(Pair{idx[a], idx[b]});
      }
    }
  }
  return out;
}

Eigen::Index pair_index(const std::unordered_map<std::int64_t, Eigen::Index>& map,
                        Eigen::Index a, Eigen::Index b, Eigen::Index p) {
  const auto it = map.find(pair_key(a, b, p));
  return it == map.end() ? -1 : it->second;
}

struct TriadSystem {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::MatrixXd M;
};

struct TriadSystemDirection {
  TriadSystem sys;
  Eigen::MatrixXd dA;
  Eigen::VectorXd db;
  Eigen::MatrixXd dM;
};

fit_expected<TriadSystem>
triad_system(const Eigen::MatrixXd& R,
             const std::vector<std::vector<Eigen::Index>>& blocks,
             const std::vector<Pair>& selected_off,
             const Eigen::VectorXd* h2_for_jac,
             bool include_anchor_rows = false,
             const std::vector<Eigen::Index>* block_of_ptr = nullptr) {
  const Eigen::Index p = R.rows();
  Eigen::Index n_rows = 0;
  for (const auto& idx : blocks) {
    const auto m = static_cast<Eigen::Index>(idx.size());
    n_rows += m * (m - 1) * (m - 2) / 2;
    if (include_anchor_rows) n_rows += m * (m - 1) * (p - m);
  }
  if (include_anchor_rows &&
      (block_of_ptr == nullptr ||
       static_cast<Eigen::Index>(block_of_ptr->size()) != p)) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality anchor system: block map is missing or mis-sized"});
  }

  std::unordered_map<std::int64_t, Eigen::Index> off_index;
  off_index.reserve(selected_off.size());
  for (std::size_t k = 0; k < selected_off.size(); ++k) {
    off_index.emplace(pair_key(selected_off[k].i, selected_off[k].j, p),
                      static_cast<Eigen::Index>(k));
  }

  TriadSystem sys;
  sys.A = Eigen::MatrixXd::Zero(n_rows, p);
  sys.b = Eigen::VectorXd::Zero(n_rows);
  sys.M = Eigen::MatrixXd::Zero(n_rows,
                                static_cast<Eigen::Index>(selected_off.size()));

  Eigen::Index row = 0;
  for (const auto& idx : blocks) {
    for (const Eigen::Index i : idx) {
      for (std::size_t a = 0; a < idx.size(); ++a) {
        const Eigen::Index j = idx[a];
        if (j == i) continue;
        for (std::size_t c = a + 1; c < idx.size(); ++c) {
          const Eigen::Index k = idx[c];
          if (k == i) continue;
          sys.A(row, i) = R(j, k);
          sys.b(row) = R(i, j) * R(i, k);

          if (h2_for_jac != nullptr) {
            const Eigen::Index jk = pair_index(off_index, j, k, p);
            const Eigen::Index ij = pair_index(off_index, i, j, p);
            const Eigen::Index ik = pair_index(off_index, i, k, p);
            if (jk < 0 || ij < 0 || ik < 0)
              return std::unexpected(FitError{FitError::Kind::NumericIssue,
                  "communality GMM: selected pair set is incomplete"});
            sys.M(row, jk) += (*h2_for_jac)(i);
            sys.M(row, ij) -= R(i, k);
            sys.M(row, ik) -= R(i, j);
          }
          ++row;
        }

        if (!include_anchor_rows) continue;
        const Eigen::Index bi = (*block_of_ptr)[static_cast<std::size_t>(i)];
        for (Eigen::Index k = 0; k < p; ++k) {
          if (k == i || k == j) continue;
          if ((*block_of_ptr)[static_cast<std::size_t>(k)] == bi) continue;

          sys.A(row, i) = R(j, k);
          sys.b(row) = R(i, j) * R(i, k);

          if (h2_for_jac != nullptr) {
            const Eigen::Index jk = pair_index(off_index, j, k, p);
            const Eigen::Index ij = pair_index(off_index, i, j, p);
            const Eigen::Index ik = pair_index(off_index, i, k, p);
            if (jk < 0 || ij < 0 || ik < 0)
              return std::unexpected(FitError{FitError::Kind::NumericIssue,
                  "communality GMM: selected pair set is incomplete"});
            sys.M(row, jk) += (*h2_for_jac)(i);
            sys.M(row, ij) -= R(i, k);
            sys.M(row, ik) -= R(i, j);
          }
          ++row;
        }
      }
    }
  }
  return sys;
}

fit_expected<TriadSystemDirection>
triad_system_directional(const Eigen::MatrixXd& R,
                         const Eigen::MatrixXd& dR,
                         const std::vector<std::vector<Eigen::Index>>& blocks,
                         const std::vector<Pair>& selected_off,
                         const Eigen::VectorXd* h2_for_jac,
                         const Eigen::VectorXd* dh2_for_jac,
                         bool include_anchor_rows = false,
                         const std::vector<Eigen::Index>* block_of_ptr = nullptr) {
  auto sys = triad_system(R, blocks, selected_off, h2_for_jac,
                          include_anchor_rows, block_of_ptr);
  if (!sys.has_value()) return std::unexpected(sys.error());

  const Eigen::Index p = R.rows();
  if ((h2_for_jac == nullptr) != (dh2_for_jac == nullptr))
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "communality GMM derivative: h2 derivative is incomplete"});
  Eigen::Index n_rows = sys->A.rows();

  std::unordered_map<std::int64_t, Eigen::Index> off_index;
  off_index.reserve(selected_off.size());
  for (std::size_t k = 0; k < selected_off.size(); ++k) {
    off_index.emplace(pair_key(selected_off[k].i, selected_off[k].j, p),
                      static_cast<Eigen::Index>(k));
  }

  Eigen::MatrixXd dA = Eigen::MatrixXd::Zero(n_rows, p);
  Eigen::VectorXd db = Eigen::VectorXd::Zero(n_rows);
  Eigen::MatrixXd dM = Eigen::MatrixXd::Zero(n_rows,
      static_cast<Eigen::Index>(selected_off.size()));

  Eigen::Index row = 0;
  for (const auto& idx : blocks) {
    for (const Eigen::Index i : idx) {
      for (std::size_t a = 0; a < idx.size(); ++a) {
        const Eigen::Index j = idx[a];
        if (j == i) continue;
        for (std::size_t c = a + 1; c < idx.size(); ++c) {
          const Eigen::Index k = idx[c];
          if (k == i) continue;
          dA(row, i) = dR(j, k);
          db(row) = dR(i, j) * R(i, k) + R(i, j) * dR(i, k);

          if (h2_for_jac != nullptr) {
            const Eigen::Index jk = pair_index(off_index, j, k, p);
            const Eigen::Index ij = pair_index(off_index, i, j, p);
            const Eigen::Index ik = pair_index(off_index, i, k, p);
            if (jk < 0 || ij < 0 || ik < 0)
              return std::unexpected(FitError{FitError::Kind::NumericIssue,
                  "communality GMM derivative: selected pair set is incomplete"});
            dM(row, jk) += (*dh2_for_jac)(i);
            dM(row, ij) -= dR(i, k);
            dM(row, ik) -= dR(i, j);
          }
          ++row;
        }

        if (!include_anchor_rows) continue;
        const Eigen::Index bi = (*block_of_ptr)[static_cast<std::size_t>(i)];
        for (Eigen::Index k = 0; k < p; ++k) {
          if (k == i || k == j) continue;
          if ((*block_of_ptr)[static_cast<std::size_t>(k)] == bi) continue;

          dA(row, i) = dR(j, k);
          db(row) = dR(i, j) * R(i, k) + R(i, j) * dR(i, k);

          if (h2_for_jac != nullptr) {
            const Eigen::Index jk = pair_index(off_index, j, k, p);
            const Eigen::Index ij = pair_index(off_index, i, j, p);
            const Eigen::Index ik = pair_index(off_index, i, k, p);
            if (jk < 0 || ij < 0 || ik < 0)
              return std::unexpected(FitError{FitError::Kind::NumericIssue,
                  "communality GMM derivative: selected pair set is incomplete"});
            dM(row, jk) += (*dh2_for_jac)(i);
            dM(row, ij) -= dR(i, k);
            dM(row, ik) -= dR(i, j);
          }
          ++row;
        }
      }
    }
  }
  return TriadSystemDirection{std::move(*sys), std::move(dA), std::move(db),
                              std::move(dM)};
}

Eigen::MatrixXd normal_cor_gamma_pairs(const Eigen::MatrixXd& R,
                                       const std::vector<Pair>& off) {
  const auto n_pairs = static_cast<Eigen::Index>(off.size());
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_pairs, n_pairs);

  for (Eigen::Index a = 0; a < n_pairs; ++a) {
    const Eigen::Index i = off[static_cast<std::size_t>(a)].i;
    const Eigen::Index j = off[static_cast<std::size_t>(a)].j;
    const double rho_a = R(i, j);
    const Eigen::Index aa[3] = {i, i, j};
    const Eigen::Index ab[3] = {j, i, j};
    const double ad[3] = {1.0, -0.5 * rho_a, -0.5 * rho_a};

    for (Eigen::Index b = 0; b <= a; ++b) {
      const Eigen::Index k = off[static_cast<std::size_t>(b)].i;
      const Eigen::Index l = off[static_cast<std::size_t>(b)].j;
      const double rho_b = R(k, l);
      const Eigen::Index ba[3] = {k, k, l};
      const Eigen::Index bb[3] = {l, k, l};
      const double bd[3] = {1.0, -0.5 * rho_b, -0.5 * rho_b};

      double val = 0.0;
      for (int u = 0; u < 3; ++u) {
        for (int v = 0; v < 3; ++v) {
          const double cov_s = R(aa[u], ba[v]) * R(ab[u], bb[v]) +
                               R(aa[u], bb[v]) * R(ab[u], ba[v]);
          val += ad[u] * bd[v] * cov_s;
        }
      }
      out(a, b) = val;
      out(b, a) = val;
    }
  }
  return out;
}

Eigen::MatrixXd normal_cor_gamma_pairs_directional(const Eigen::MatrixXd& R,
                                                   const Eigen::MatrixXd& dR,
                                                   const std::vector<Pair>& off) {
  const auto n_pairs = static_cast<Eigen::Index>(off.size());
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_pairs, n_pairs);

  for (Eigen::Index a = 0; a < n_pairs; ++a) {
    const Eigen::Index i = off[static_cast<std::size_t>(a)].i;
    const Eigen::Index j = off[static_cast<std::size_t>(a)].j;
    const double rho_a = R(i, j);
    const double drho_a = dR(i, j);
    const Eigen::Index aa[3] = {i, i, j};
    const Eigen::Index ab[3] = {j, i, j};
    const double ad[3] = {1.0, -0.5 * rho_a, -0.5 * rho_a};
    const double dad[3] = {0.0, -0.5 * drho_a, -0.5 * drho_a};

    for (Eigen::Index b = 0; b <= a; ++b) {
      const Eigen::Index k = off[static_cast<std::size_t>(b)].i;
      const Eigen::Index l = off[static_cast<std::size_t>(b)].j;
      const double rho_b = R(k, l);
      const double drho_b = dR(k, l);
      const Eigen::Index ba[3] = {k, k, l};
      const Eigen::Index bb[3] = {l, k, l};
      const double bd[3] = {1.0, -0.5 * rho_b, -0.5 * rho_b};
      const double dbd[3] = {0.0, -0.5 * drho_b, -0.5 * drho_b};

      double val = 0.0;
      for (int u = 0; u < 3; ++u) {
        for (int v = 0; v < 3; ++v) {
          const double cov_s = R(aa[u], ba[v]) * R(ab[u], bb[v]) +
                               R(aa[u], bb[v]) * R(ab[u], ba[v]);
          const double dcov_s =
              dR(aa[u], ba[v]) * R(ab[u], bb[v]) +
              R(aa[u], ba[v]) * dR(ab[u], bb[v]) +
              dR(aa[u], bb[v]) * R(ab[u], ba[v]) +
              R(aa[u], bb[v]) * dR(ab[u], ba[v]);
          val += (dad[u] * bd[v] + ad[u] * dbd[v]) * cov_s +
                 ad[u] * bd[v] * dcov_s;
        }
      }
      out(a, b) = val;
      out(b, a) = val;
    }
  }
  return out;
}

std::vector<Eigen::MatrixXd>
normal_cor_gamma_pairs_jacobian(
    const Eigen::MatrixXd& R,
    const std::vector<Pair>& off,
    const std::vector<Eigen::Index>& global_index,
    const CorrelationJacobian& J,
    const std::vector<Eigen::Index>& active_cols) {
  const auto n_pairs = static_cast<Eigen::Index>(off.size());
  std::vector<Eigen::MatrixXd> out(
      active_cols.size(),
      Eigen::MatrixXd::Zero(n_pairs, n_pairs));

  for (Eigen::Index a = 0; a < n_pairs; ++a) {
    const Eigen::Index i = off[static_cast<std::size_t>(a)].i;
    const Eigen::Index j = off[static_cast<std::size_t>(a)].j;
    const double rho_a = R(i, j);
    const Eigen::Index aa[3] = {i, i, j};
    const Eigen::Index ab[3] = {j, i, j};
    const double ad[3] = {1.0, -0.5 * rho_a, -0.5 * rho_a};

    for (Eigen::Index b = 0; b <= a; ++b) {
      const Eigen::Index k = off[static_cast<std::size_t>(b)].i;
      const Eigen::Index l = off[static_cast<std::size_t>(b)].j;
      const double rho_b = R(k, l);
      const Eigen::Index ba[3] = {k, k, l};
      const Eigen::Index bb[3] = {l, k, l};
      const double bd[3] = {1.0, -0.5 * rho_b, -0.5 * rho_b};

      Eigen::RowVectorXd dval =
          Eigen::RowVectorXd::Zero(
              static_cast<Eigen::Index>(active_cols.size()));
      for (int u = 0; u < 3; ++u) {
        for (int v = 0; v < 3; ++v) {
          const double cov_s = R(aa[u], ba[v]) * R(ab[u], bb[v]) +
                               R(aa[u], bb[v]) * R(ab[u], ba[v]);
          if (u > 0)
            add_corr_active_row(
                dval, J, active_cols,
                global_index[static_cast<std::size_t>(i)],
                global_index[static_cast<std::size_t>(j)],
                -0.5 * bd[v] * cov_s);
          if (v > 0)
            add_corr_active_row(
                dval, J, active_cols,
                global_index[static_cast<std::size_t>(k)],
                global_index[static_cast<std::size_t>(l)],
                -0.5 * ad[u] * cov_s);

          const double scale = ad[u] * bd[v];
          add_corr_active_row(
              dval, J, active_cols,
              global_index[static_cast<std::size_t>(aa[u])],
              global_index[static_cast<std::size_t>(ba[v])],
              scale * R(ab[u], bb[v]));
          add_corr_active_row(
              dval, J, active_cols,
              global_index[static_cast<std::size_t>(ab[u])],
              global_index[static_cast<std::size_t>(bb[v])],
              scale * R(aa[u], ba[v]));
          add_corr_active_row(
              dval, J, active_cols,
              global_index[static_cast<std::size_t>(aa[u])],
              global_index[static_cast<std::size_t>(bb[v])],
              scale * R(ab[u], ba[v]));
          add_corr_active_row(
              dval, J, active_cols,
              global_index[static_cast<std::size_t>(ab[u])],
              global_index[static_cast<std::size_t>(ba[v])],
              scale * R(aa[u], bb[v]));
        }
      }

      for (std::size_t active = 0; active < active_cols.size(); ++active) {
        out[active](a, b) = dval(static_cast<Eigen::Index>(active));
        out[active](b, a) = dval(static_cast<Eigen::Index>(active));
      }
    }
  }
  return out;
}

fit_expected<Eigen::VectorXd> solve_gmm(const Eigen::MatrixXd& A,
                                        const Eigen::VectorXd& b,
                                        const Eigen::MatrixXd& W) {
  if (A.rows() == A.cols()) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    lu.setThreshold(1e-12);
    if (lu.rank() == A.cols()) {
      const Eigen::VectorXd direct = lu.solve(b);
      if (direct.allFinite()) return direct;
    }
  }

  const Eigen::MatrixXd lhs = A.transpose() * W * A;
  const Eigen::VectorXd rhs = A.transpose() * W * b;
  auto lhs_pinv = symmetric_pinv(lhs);
  if (!lhs_pinv.has_value()) return std::unexpected(lhs_pinv.error());
  Eigen::VectorXd out = *lhs_pinv * rhs;
  if (!out.allFinite())
    return num_error("communality GMM: non-finite solution");
  return out;
}

fit_expected<Eigen::VectorXd>
solve_gmm_directional(const CommunalitySystem& system,
                      const Eigen::MatrixXd& dA,
                      const Eigen::VectorXd& db,
                      const Eigen::MatrixXd& dW) {
  const Eigen::MatrixXd& A = system.A;
  const Eigen::VectorXd& b = system.b;
  const Eigen::MatrixXd& W = system.W;
  if (dA.rows() != A.rows() || dA.cols() != A.cols() ||
      db.size() != b.size() || dW.rows() != W.rows() ||
      dW.cols() != W.cols())
    return num_error("communality GMM derivative: dimension mismatch");

  if (A.rows() == A.cols()) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    lu.setThreshold(1e-12);
    if (lu.rank() == A.cols()) {
      const Eigen::VectorXd h = lu.solve(b);
      const Eigen::VectorXd dh = lu.solve(db - dA * h);
      if (dh.allFinite()) return dh;
    }
  }

  const Eigen::MatrixXd G = A.transpose() * W * A;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(G);
  lu.setThreshold(1e-12);
  if (lu.rank() != G.cols())
    return num_error("communality GMM derivative: normal equations are singular");
  const Eigen::VectorXd rhs = A.transpose() * W * b;
  const Eigen::VectorXd h = lu.solve(rhs);
  const Eigen::MatrixXd dG =
      dA.transpose() * W * A + A.transpose() * dW * A +
      A.transpose() * W * dA;
  const Eigen::VectorXd drhs =
      dA.transpose() * W * b + A.transpose() * dW * b +
      A.transpose() * W * db;
  const Eigen::VectorXd dh = lu.solve(drhs - dG * h);
  if (!dh.allFinite())
    return num_error("communality GMM derivative: non-finite solution");
  return dh;
}

fit_expected<Eigen::MatrixXd>
solve_gmm_jacobian_weight_actions(
    const CommunalitySystem& system,
    const Eigen::MatrixXd& omega,
    const Eigen::MatrixXd& M,
    const Eigen::MatrixXd& gamma,
    const std::vector<Eigen::MatrixXd>& dA_cols,
    const Eigen::MatrixXd& db_cols,
    const std::vector<Eigen::MatrixXd>& dM_cols,
    const std::vector<Eigen::MatrixXd>& dgamma_cols) {
  const Eigen::MatrixXd& A = system.A;
  const Eigen::VectorXd& b = system.b;
  const Eigen::MatrixXd& W = system.W;
  const auto pstar = static_cast<Eigen::Index>(dA_cols.size());
  if (A.rows() != b.size() || W.rows() != A.rows() ||
      W.cols() != A.rows() || omega.rows() != A.rows() ||
      omega.cols() != A.rows() || M.rows() != A.rows() ||
      gamma.rows() != M.cols() || gamma.cols() != M.cols() ||
      db_cols.rows() != b.size() || db_cols.cols() != pstar ||
      static_cast<Eigen::Index>(dM_cols.size()) != pstar ||
      static_cast<Eigen::Index>(dgamma_cols.size()) != pstar)
    return mat_error("communality GMM Jacobian: action dimension mismatch");
  for (Eigen::Index col = 0; col < pstar; ++col) {
    const auto& dA = dA_cols[static_cast<std::size_t>(col)];
    const auto& dM = dM_cols[static_cast<std::size_t>(col)];
    const auto& dgamma = dgamma_cols[static_cast<std::size_t>(col)];
    if (dA.rows() != A.rows() || dA.cols() != A.cols() ||
        dM.rows() != M.rows() || dM.cols() != M.cols() ||
        dgamma.rows() != gamma.rows() || dgamma.cols() != gamma.cols())
      return mat_error("communality GMM Jacobian: action derivative dimension "
                       "mismatch");
  }

  if (A.rows() == A.cols()) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    lu.setThreshold(1e-12);
    if (lu.rank() == A.cols()) {
      const Eigen::VectorXd h = lu.solve(b);
      Eigen::MatrixXd rhs(A.rows(), pstar);
      for (Eigen::Index col = 0; col < pstar; ++col) {
        rhs.col(col) = db_cols.col(col) -
                       dA_cols[static_cast<std::size_t>(col)] * h;
      }
      Eigen::MatrixXd dh = lu.solve(rhs);
      if (dh.allFinite()) return dh;
    }
  }

  const Eigen::MatrixXd G = A.transpose() * W * A;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(G);
  lu.setThreshold(1e-12);
  if (lu.rank() != G.cols())
    return mat_error("communality GMM Jacobian: normal equations are singular");
  const Eigen::VectorXd c = A.transpose() * W * b;
  const Eigen::VectorXd h = lu.solve(c);
  const Eigen::VectorXd e = b - A * h;
  const Eigen::VectorXd We = W * e;
  const Eigen::MatrixXd AtW = A.transpose() * W;
  const Eigen::MatrixXd W2 = W * W;
  const Eigen::MatrixXd left_null =
      Eigen::MatrixXd::Identity(W.rows(), W.cols()) - W * omega;
  const Eigen::VectorXd right_null_e = e - omega * We;
  const Eigen::VectorXd W2e = W2 * e;

  Eigen::MatrixXd dh(A.cols(), pstar);
  for (Eigen::Index col = 0; col < pstar; ++col) {
    const auto& dA = dA_cols[static_cast<std::size_t>(col)];
    const auto& dM = dM_cols[static_cast<std::size_t>(col)];
    const auto& dgamma = dgamma_cols[static_cast<std::size_t>(col)];
    const auto domega_times = [&](const Eigen::VectorXd& x) {
      const Eigen::VectorXd Mtx = M.transpose() * x;
      Eigen::VectorXd out = dM * (gamma * Mtx);
      out.noalias() += M * (dgamma * Mtx);
      out.noalias() += M * (gamma * (dM.transpose() * x));
      return out;
    };

    const Eigen::VectorXd dWe =
        -W * domega_times(We) +
        W2 * domega_times(right_null_e) +
        left_null * domega_times(W2e);
    const Eigen::VectorXd rhs =
        dA.transpose() * We +
        AtW * (db_cols.col(col) - dA * h) +
        A.transpose() * dWe;
    dh.col(col) = lu.solve(rhs);
  }
  if (!dh.allFinite())
    return mat_error("communality GMM Jacobian: non-finite action solution");
  return dh;
}

fit_expected<Eigen::MatrixXd>
gmm_jacobian_rhs_weight_actions(
    const CommunalitySystem& system,
    const Eigen::MatrixXd& omega,
    const Eigen::MatrixXd& M,
    const Eigen::MatrixXd& gamma,
    const std::vector<Eigen::MatrixXd>& dA_cols,
    const Eigen::MatrixXd& db_cols,
    const std::vector<Eigen::MatrixXd>& dM_cols,
    const std::vector<Eigen::MatrixXd>& dgamma_cols,
    const Eigen::VectorXd& h) {
  const Eigen::MatrixXd& A = system.A;
  const Eigen::VectorXd& b = system.b;
  const Eigen::MatrixXd& W = system.W;
  const auto pstar = static_cast<Eigen::Index>(dA_cols.size());
  if (A.rows() != b.size() || W.rows() != A.rows() ||
      W.cols() != A.rows() || omega.rows() != A.rows() ||
      omega.cols() != A.rows() || M.rows() != A.rows() ||
      gamma.rows() != M.cols() || gamma.cols() != M.cols() ||
      db_cols.rows() != b.size() || db_cols.cols() != pstar ||
      h.size() != A.cols() ||
      static_cast<Eigen::Index>(dM_cols.size()) != pstar ||
      static_cast<Eigen::Index>(dgamma_cols.size()) != pstar)
    return mat_error("communality GMM Jacobian RHS: action dimension mismatch");
  for (Eigen::Index col = 0; col < pstar; ++col) {
    const auto& dA = dA_cols[static_cast<std::size_t>(col)];
    const auto& dM = dM_cols[static_cast<std::size_t>(col)];
    const auto& dgamma = dgamma_cols[static_cast<std::size_t>(col)];
    if (dA.rows() != A.rows() || dA.cols() != A.cols() ||
        dM.rows() != M.rows() || dM.cols() != M.cols() ||
        dgamma.rows() != gamma.rows() || dgamma.cols() != gamma.cols())
      return mat_error("communality GMM Jacobian RHS: derivative dimension "
                       "mismatch");
  }

  const Eigen::VectorXd e = b - A * h;
  const Eigen::VectorXd We = W * e;
  const Eigen::MatrixXd AtW = A.transpose() * W;
  const Eigen::MatrixXd W2 = W * W;
  const Eigen::MatrixXd left_null =
      Eigen::MatrixXd::Identity(W.rows(), W.cols()) - W * omega;
  const Eigen::VectorXd right_null_e = e - omega * We;
  const Eigen::VectorXd W2e = W2 * e;

  Eigen::MatrixXd rhs(A.cols(), pstar);
  for (Eigen::Index col = 0; col < pstar; ++col) {
    const auto& dA = dA_cols[static_cast<std::size_t>(col)];
    const auto& dM = dM_cols[static_cast<std::size_t>(col)];
    const auto& dgamma = dgamma_cols[static_cast<std::size_t>(col)];
    const auto domega_times = [&](const Eigen::VectorXd& x) {
      const Eigen::VectorXd Mtx = M.transpose() * x;
      Eigen::VectorXd out = dM * (gamma * Mtx);
      out.noalias() += M * (dgamma * Mtx);
      out.noalias() += M * (gamma * (dM.transpose() * x));
      return out;
    };

    const Eigen::VectorXd dWe =
        -W * domega_times(We) +
        W2 * domega_times(right_null_e) +
        left_null * domega_times(W2e);
    rhs.col(col) =
        dA.transpose() * We +
        AtW * (db_cols.col(col) - dA * h) +
        A.transpose() * dWe;
  }
  if (!rhs.allFinite())
    return mat_error("communality GMM Jacobian RHS: non-finite action output");
  return rhs;
}

fit_expected<Eigen::MatrixXd>
solve_constrained_h2_jacobian_from_rhs(const CommunalitySystem& system,
                                       const Eigen::MatrixXd& R_h2,
                                       const Eigen::VectorXd& r_h2,
                                       const Eigen::MatrixXd& rhs_head,
                                       const Eigen::MatrixXd& S,
                                       const Eigen::VectorXd& h2) {
  const Eigen::MatrixXd& A = system.A;
  const Eigen::VectorXd& b = system.b;
  const Eigen::MatrixXd& W = system.W;
  const Eigen::Index n = A.cols();
  const Eigen::Index m = R_h2.rows();
  const Eigen::Index pstar = rhs_head.cols();
  if (A.rows() != b.size() || W.rows() != A.rows() ||
      W.cols() != A.rows() || R_h2.rows() != r_h2.size() ||
      (m > 0 && R_h2.cols() != n) || rhs_head.rows() != n ||
      h2.size() != n || S.rows() != n || S.cols() != n)
    return mat_error("communality constrained Jacobian: dimension mismatch");

  const Eigen::MatrixXd G = A.transpose() * W * A;
  const Eigen::VectorXd c = A.transpose() * W * b;
  if (m == 0) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(G);
    lu.setThreshold(1e-12);
    if (lu.rank() != n)
      return mat_error("communality constrained Jacobian: normal equations are "
                       "singular");
    const Eigen::VectorXd h_check = lu.solve(c);
    const double scale = std::max(1.0, h2.cwiseAbs().maxCoeff());
    if ((h_check - h2).cwiseAbs().maxCoeff() > 1e-7 * scale)
      return mat_error("communality constrained Jacobian: base solution "
                       "mismatch");
    Eigen::MatrixXd out = lu.solve(rhs_head);
    if (!out.allFinite())
      return mat_error("communality constrained Jacobian: non-finite solution");
    return out;
  }

  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n + m, n + m);
  K.block(0, 0, n, n) = G;
  K.block(0, n, n, m) = R_h2.transpose();
  K.block(n, 0, m, n) = R_h2;
  Eigen::VectorXd krhs(n + m);
  krhs.head(n) = c;
  krhs.tail(m) = r_h2;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
  lu.setThreshold(1e-12);
  if (lu.rank() != n + m)
    return mat_error("communality constrained Jacobian: KKT system is singular");
  const Eigen::VectorXd x = lu.solve(krhs);
  if (!x.allFinite())
    return mat_error("communality constrained Jacobian: non-finite KKT "
                     "solution");
  const double scale = std::max(1.0, h2.cwiseAbs().maxCoeff());
  if ((x.head(n) - h2).cwiseAbs().maxCoeff() > 1e-7 * scale)
    return mat_error("communality constrained Jacobian: KKT base solution "
                     "mismatch");
  const Eigen::VectorXd lambda = x.tail(m);

  Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n + m, pstar);
  rhs.topRows(n) = rhs_head;
  for (Eigen::Index row = 0; row < m; ++row) {
    for (Eigen::Index i = 0; i < n; ++i) {
      const double coeff = R_h2(row, i);
      if (std::abs(coeff) <= kZeroTol) continue;
      const double sii = S(i, i);
      if (!std::isfinite(sii) || sii <= 0.0)
        return mat_error("communality constrained Jacobian: non-positive "
                         "variance in constraint derivative");
      const double dcoeff = coeff / sii;
      const Eigen::Index col = vech_index(i, i, n);
      rhs(i, col) -= dcoeff * lambda(row);
      rhs(n + row, col) += dcoeff * (1.0 - h2(i));
    }
  }

  Eigen::MatrixXd dx = lu.solve(rhs);
  if (!dx.allFinite())
    return mat_error("communality constrained Jacobian: non-finite KKT "
                     "derivative");
  return dx.topRows(n);
}

fit_expected<CommunalitySystem>
triad_ls_system(const Eigen::MatrixXd& R,
                const std::vector<std::vector<Eigen::Index>>& blocks,
                const std::vector<Eigen::Index>& block_of,
                bool include_anchor_rows) {
  const std::vector<Pair> off = include_anchor_rows
      ? std::vector<Pair>{}
      : within_off_pairs(blocks);
  auto sys = triad_system(R, blocks, off, nullptr, include_anchor_rows,
                          include_anchor_rows ? &block_of : nullptr);
  if (!sys.has_value()) return std::unexpected(sys.error());
  CommunalitySystem out;
  out.A = std::move(sys->A);
  out.b = std::move(sys->b);
  out.W = Eigen::MatrixXd::Identity(out.A.rows(), out.A.rows());
  return out;
}

fit_expected<CommunalitySystem>
gmm_block_system(const Eigen::MatrixXd& R,
                 const std::vector<std::vector<Eigen::Index>>& blocks) {
  const Eigen::Index p = R.rows();
  std::vector<CommunalitySystem> local;
  local.reserve(blocks.size());
  Eigen::Index n_rows = 0;

  for (const auto& idx : blocks) {
    const Eigen::Index m = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Rb(m, m);
    for (Eigen::Index a = 0; a < m; ++a) {
      for (Eigen::Index b = 0; b < m; ++b) {
        Rb(a, b) = R(idx[static_cast<std::size_t>(a)],
                     idx[static_cast<std::size_t>(b)]);
      }
    }
    std::vector<std::vector<Eigen::Index>> local_blocks(1);
    local_blocks[0].resize(static_cast<std::size_t>(m));
    for (Eigen::Index i = 0; i < m; ++i)
      local_blocks[0][static_cast<std::size_t>(i)] = i;
    std::vector<Eigen::Index> local_block_of(static_cast<std::size_t>(m), 0);

    auto h20 =
        itemwise_h2(Rb, local_blocks, local_block_of,
                    CommunalityMethod::TriadLeastSquares);
    if (!h20.has_value()) return std::unexpected(h20.error());

    const std::vector<Pair> off = within_off_pairs(local_blocks);
    auto sys = triad_system(Rb, local_blocks, off, nullptr);
    if (!sys.has_value()) return std::unexpected(sys.error());
    auto sys_w = triad_system(Rb, local_blocks, off, &*h20);
    if (!sys_w.has_value()) return std::unexpected(sys_w.error());
    const Eigen::MatrixXd gamma = normal_cor_gamma_pairs(Rb, off);
    const Eigen::MatrixXd omega = sys_w->M * gamma * sys_w->M.transpose();
    auto W = row_covariance_pinv(sys_w->M, gamma, omega);
    if (!W.has_value()) return std::unexpected(W.error());

    CommunalitySystem cs;
    cs.A = Eigen::MatrixXd::Zero(sys->A.rows(), p);
    for (Eigen::Index c = 0; c < m; ++c) {
      const Eigen::Index global = idx[static_cast<std::size_t>(c)];
      cs.A.col(global) = sys->A.col(c);
    }
    cs.b = std::move(sys->b);
    cs.W = std::move(*W);
    n_rows += cs.A.rows();
    local.push_back(std::move(cs));
  }

  CommunalitySystem out;
  out.A = Eigen::MatrixXd::Zero(n_rows, p);
  out.b = Eigen::VectorXd::Zero(n_rows);
  out.W = Eigen::MatrixXd::Zero(n_rows, n_rows);
  Eigen::Index off = 0;
  for (const auto& cs : local) {
    const Eigen::Index m = cs.A.rows();
    out.A.block(off, 0, m, p) = cs.A;
    out.b.segment(off, m) = cs.b;
    out.W.block(off, off, m, m) = cs.W;
    off += m;
  }
  return out;
}

fit_expected<CommunalitySystemDirection>
gmm_block_system_directional(const Eigen::MatrixXd& R,
                             const Eigen::MatrixXd& dR,
                             const std::vector<std::vector<Eigen::Index>>& blocks) {
  const Eigen::Index p = R.rows();
  std::vector<CommunalitySystemDirection> local;
  local.reserve(blocks.size());
  Eigen::Index n_rows = 0;

  for (const auto& idx : blocks) {
    const Eigen::Index m = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Rb(m, m);
    Eigen::MatrixXd dRb(m, m);
    for (Eigen::Index a = 0; a < m; ++a) {
      for (Eigen::Index b = 0; b < m; ++b) {
        Rb(a, b) = R(idx[static_cast<std::size_t>(a)],
                     idx[static_cast<std::size_t>(b)]);
        dRb(a, b) = dR(idx[static_cast<std::size_t>(a)],
                       idx[static_cast<std::size_t>(b)]);
      }
    }
    std::vector<std::vector<Eigen::Index>> local_blocks(1);
    local_blocks[0].resize(static_cast<std::size_t>(m));
    for (Eigen::Index i = 0; i < m; ++i)
      local_blocks[0][static_cast<std::size_t>(i)] = i;
    std::vector<Eigen::Index> local_block_of(static_cast<std::size_t>(m), 0);

    auto h20 =
        itemwise_h2(Rb, local_blocks, local_block_of,
                    CommunalityMethod::TriadLeastSquares);
    if (!h20.has_value()) return std::unexpected(h20.error());
    auto dh20 =
        itemwise_h2_directional(Rb, dRb, local_blocks, local_block_of,
                                CommunalityMethod::TriadLeastSquares);
    if (!dh20.has_value()) return std::unexpected(dh20.error());

    const std::vector<Pair> off = within_off_pairs(local_blocks);
    auto sys = triad_system_directional(Rb, dRb, local_blocks, off, nullptr,
                                        nullptr);
    if (!sys.has_value()) return std::unexpected(sys.error());
    auto sys_w = triad_system_directional(Rb, dRb, local_blocks, off, &*h20,
                                          &*dh20);
    if (!sys_w.has_value()) return std::unexpected(sys_w.error());
    const Eigen::MatrixXd gamma = normal_cor_gamma_pairs(Rb, off);
    const Eigen::MatrixXd dgamma = normal_cor_gamma_pairs_directional(Rb, dRb, off);
    const Eigen::MatrixXd omega =
        sys_w->sys.M * gamma * sys_w->sys.M.transpose();
    const Eigen::MatrixXd domega =
        sys_w->dM * gamma * sys_w->sys.M.transpose() +
        sys_w->sys.M * dgamma * sys_w->sys.M.transpose() +
        sys_w->sys.M * gamma * sys_w->dM.transpose();
    auto W = symmetric_pinv_directional(omega, domega);
    if (!W.has_value()) return std::unexpected(W.error());

    CommunalitySystemDirection cs;
    cs.system.A = Eigen::MatrixXd::Zero(sys->sys.A.rows(), p);
    cs.dA = Eigen::MatrixXd::Zero(sys->sys.A.rows(), p);
    for (Eigen::Index c = 0; c < m; ++c) {
      const Eigen::Index global = idx[static_cast<std::size_t>(c)];
      cs.system.A.col(global) = sys->sys.A.col(c);
      cs.dA.col(global) = sys->dA.col(c);
    }
    cs.system.b = std::move(sys->sys.b);
    cs.db = std::move(sys->db);
    cs.system.W = std::move(W->value);
    cs.dW = std::move(W->deriv);
    n_rows += cs.system.A.rows();
    local.push_back(std::move(cs));
  }

  CommunalitySystemDirection out;
  out.system.A = Eigen::MatrixXd::Zero(n_rows, p);
  out.system.b = Eigen::VectorXd::Zero(n_rows);
  out.system.W = Eigen::MatrixXd::Zero(n_rows, n_rows);
  out.dA = Eigen::MatrixXd::Zero(n_rows, p);
  out.db = Eigen::VectorXd::Zero(n_rows);
  out.dW = Eigen::MatrixXd::Zero(n_rows, n_rows);
  Eigen::Index off = 0;
  for (const auto& cs : local) {
    const Eigen::Index m = cs.system.A.rows();
    out.system.A.block(off, 0, m, p) = cs.system.A;
    out.system.b.segment(off, m) = cs.system.b;
    out.system.W.block(off, off, m, m) = cs.system.W;
    out.dA.block(off, 0, m, p) = cs.dA;
    out.db.segment(off, m) = cs.db;
    out.dW.block(off, off, m, m) = cs.dW;
    off += m;
  }
  return out;
}

fit_expected<CommunalitySystem>
gmm_full_system(const Eigen::MatrixXd& R,
                const std::vector<std::vector<Eigen::Index>>& blocks,
                const std::vector<Eigen::Index>& block_of) {
  auto h20 =
      itemwise_h2(R, blocks, block_of, CommunalityMethod::TriadLeastSquares);
  if (!h20.has_value()) return std::unexpected(h20.error());

  const std::vector<Pair> off = within_off_pairs(blocks);
  auto sys = triad_system(R, blocks, off, nullptr);
  if (!sys.has_value()) return std::unexpected(sys.error());
  auto sys_w = triad_system(R, blocks, off, &*h20);
  if (!sys_w.has_value()) return std::unexpected(sys_w.error());
  const Eigen::MatrixXd gamma = normal_cor_gamma_pairs(R, off);
  const Eigen::MatrixXd omega = sys_w->M * gamma * sys_w->M.transpose();
  auto W = row_covariance_pinv(sys_w->M, gamma, omega);
  if (!W.has_value()) return std::unexpected(W.error());
  return CommunalitySystem{std::move(sys->A), std::move(sys->b), std::move(*W)};
}

fit_expected<CommunalitySystemDirection>
gmm_full_system_directional(const Eigen::MatrixXd& R,
                            const Eigen::MatrixXd& dR,
                            const std::vector<std::vector<Eigen::Index>>& blocks,
                            const std::vector<Eigen::Index>& block_of) {
  auto h20 =
      itemwise_h2(R, blocks, block_of, CommunalityMethod::TriadLeastSquares);
  if (!h20.has_value()) return std::unexpected(h20.error());
  auto dh20 =
      itemwise_h2_directional(R, dR, blocks, block_of,
                              CommunalityMethod::TriadLeastSquares);
  if (!dh20.has_value()) return std::unexpected(dh20.error());

  const std::vector<Pair> off = within_off_pairs(blocks);
  auto sys = triad_system_directional(R, dR, blocks, off, nullptr, nullptr);
  if (!sys.has_value()) return std::unexpected(sys.error());
  auto sys_w = triad_system_directional(R, dR, blocks, off, &*h20, &*dh20);
  if (!sys_w.has_value()) return std::unexpected(sys_w.error());
  const Eigen::MatrixXd gamma = normal_cor_gamma_pairs(R, off);
  const Eigen::MatrixXd dgamma = normal_cor_gamma_pairs_directional(R, dR, off);
  const Eigen::MatrixXd omega =
      sys_w->sys.M * gamma * sys_w->sys.M.transpose();
  const Eigen::MatrixXd domega =
      sys_w->dM * gamma * sys_w->sys.M.transpose() +
      sys_w->sys.M * dgamma * sys_w->sys.M.transpose() +
      sys_w->sys.M * gamma * sys_w->dM.transpose();
  auto W = symmetric_pinv_directional(omega, domega);
  if (!W.has_value()) return std::unexpected(W.error());

  CommunalitySystemDirection out;
  out.system.A = std::move(sys->sys.A);
  out.system.b = std::move(sys->sys.b);
  out.system.W = std::move(W->value);
  out.dA = std::move(sys->dA);
  out.db = std::move(sys->db);
  out.dW = std::move(W->deriv);
  return out;
}

fit_expected<Eigen::MatrixXd>
gmm_block_h2_jacobian(const Eigen::MatrixXd& R,
                      const std::vector<std::vector<Eigen::Index>>& blocks,
                      const std::vector<Eigen::Index>& block_of,
                      const CorrelationJacobian& J) {
  auto h20_global =
      itemwise_h2(R, blocks, block_of, CommunalityMethod::TriadLeastSquares);
  if (!h20_global.has_value()) return std::unexpected(h20_global.error());
  auto Jh20_global =
      itemwise_h2_jacobian(R, blocks, block_of,
                           CommunalityMethod::TriadLeastSquares, J);
  if (!Jh20_global.has_value()) return std::unexpected(Jh20_global.error());

  const Eigen::Index p = R.rows();
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(p, J.pstar);

  for (const auto& idx : blocks) {
    const Eigen::Index m = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Rb(m, m);
    std::vector<Eigen::Index> global_index(static_cast<std::size_t>(m));
    for (Eigen::Index a = 0; a < m; ++a) {
      global_index[static_cast<std::size_t>(a)] =
          idx[static_cast<std::size_t>(a)];
      for (Eigen::Index b = 0; b < m; ++b) {
        Rb(a, b) = R(idx[static_cast<std::size_t>(a)],
                     idx[static_cast<std::size_t>(b)]);
      }
    }
    std::vector<Eigen::Index> active_cols;
    active_cols.reserve(static_cast<std::size_t>(m * (m + 1) / 2));
    for (Eigen::Index c = 0; c < m; ++c) {
      const Eigen::Index gc = global_index[static_cast<std::size_t>(c)];
      for (Eigen::Index r = c; r < m; ++r) {
        const Eigen::Index gr = global_index[static_cast<std::size_t>(r)];
        active_cols.push_back(vech_index(gr, gc, J.p));
      }
    }
    const auto n_active = static_cast<Eigen::Index>(active_cols.size());

    std::vector<std::vector<Eigen::Index>> local_blocks(1);
    local_blocks[0].resize(static_cast<std::size_t>(m));
    for (Eigen::Index i = 0; i < m; ++i)
      local_blocks[0][static_cast<std::size_t>(i)] = i;

    Eigen::VectorXd h20(m);
    Eigen::MatrixXd Jh20(m, J.pstar);
    for (Eigen::Index i = 0; i < m; ++i) {
      const Eigen::Index gi = idx[static_cast<std::size_t>(i)];
      h20(i) = (*h20_global)(gi);
      Jh20.row(i) = Jh20_global->row(gi);
    }

    const std::vector<Pair> off = within_off_pairs(local_blocks);
    auto sys = triad_system(Rb, local_blocks, off, nullptr);
    if (!sys.has_value()) return std::unexpected(sys.error());
    auto sys_w = triad_system(Rb, local_blocks, off, &h20);
    if (!sys_w.has_value()) return std::unexpected(sys_w.error());

    const Eigen::MatrixXd gamma = normal_cor_gamma_pairs(Rb, off);
    const auto dgamma =
        normal_cor_gamma_pairs_jacobian(Rb, off, global_index, J, active_cols);

    const Eigen::MatrixXd omega =
        sys_w->M * gamma * sys_w->M.transpose();
    auto W = row_covariance_pinv(sys_w->M, gamma, omega);
    if (!W.has_value()) return std::unexpected(W.error());

    const Eigen::Index n_rows = sys->A.rows();
    const Eigen::Index n_pairs = static_cast<Eigen::Index>(off.size());
    std::vector<Eigen::MatrixXd> dA_cols(
        active_cols.size(),
        Eigen::MatrixXd::Zero(n_rows, m));
    std::vector<Eigen::MatrixXd> dM_cols(
        active_cols.size(),
        Eigen::MatrixXd::Zero(n_rows, n_pairs));
    Eigen::MatrixXd db_cols = Eigen::MatrixXd::Zero(n_rows, n_active);

    std::unordered_map<std::int64_t, Eigen::Index> off_index;
    off_index.reserve(off.size());
    for (std::size_t k = 0; k < off.size(); ++k) {
      off_index.emplace(pair_key(off[k].i, off[k].j, m),
                        static_cast<Eigen::Index>(k));
    }

    Eigen::Index row = 0;
    for (const Eigen::Index i : local_blocks[0]) {
      for (std::size_t a = 0; a < local_blocks[0].size(); ++a) {
        const Eigen::Index j = local_blocks[0][a];
        if (j == i) continue;
        for (std::size_t c = a + 1; c < local_blocks[0].size(); ++c) {
          const Eigen::Index k = local_blocks[0][c];
          if (k == i) continue;

          const Eigen::Index gi = global_index[static_cast<std::size_t>(i)];
          const Eigen::Index gj = global_index[static_cast<std::size_t>(j)];
          const Eigen::Index gk = global_index[static_cast<std::size_t>(k)];
          const Eigen::RowVectorXd d_rjk =
              corr_active_row(J, active_cols, gj, gk);
          const Eigen::RowVectorXd d_rij =
              corr_active_row(J, active_cols, gi, gj);
          const Eigen::RowVectorXd d_rik =
              corr_active_row(J, active_cols, gi, gk);

          const Eigen::RowVectorXd db_active =
              d_rij * Rb(i, k) + Rb(i, j) * d_rik;
          const Eigen::Index jk = pair_index(off_index, j, k, m);
          const Eigen::Index ij = pair_index(off_index, i, j, m);
          const Eigen::Index ik = pair_index(off_index, i, k, m);
          if (jk < 0 || ij < 0 || ik < 0)
            return mat_error(
                "communality GMM Jacobian: selected pair set is incomplete");

          for (std::size_t active = 0; active < active_cols.size(); ++active) {
            const Eigen::Index col = active_cols[active];
            const auto ac = static_cast<Eigen::Index>(active);
            db_cols(row, ac) = db_active(ac);
            dA_cols[active](row, i) = d_rjk(ac);
            dM_cols[active](row, jk) += Jh20(i, col);
            dM_cols[active](row, ij) -= d_rik(ac);
            dM_cols[active](row, ik) -= d_rij(ac);
          }
          ++row;
        }
      }
    }

    CommunalitySystem system{std::move(sys->A), std::move(sys->b), std::move(*W)};
    auto dh = solve_gmm_jacobian_weight_actions(system, omega, sys_w->M, gamma,
                                                dA_cols, db_cols, dM_cols,
                                                dgamma);
    if (!dh.has_value()) return std::unexpected(dh.error());
    for (Eigen::Index i = 0; i < m; ++i) {
      const Eigen::Index gi = idx[static_cast<std::size_t>(i)];
      for (std::size_t active = 0; active < active_cols.size(); ++active)
        out(gi, active_cols[active]) =
            (*dh)(i, static_cast<Eigen::Index>(active));
    }
  }

  if (!out.allFinite())
    return mat_error("communality GMM block Jacobian: non-finite output");
  return out;
}

fit_expected<Eigen::MatrixXd>
gmm_block_h2_constrained_jacobian(
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& R,
    const std::vector<std::vector<Eigen::Index>>& blocks,
    const std::vector<Eigen::Index>& block_of,
    const CorrelationJacobian& J,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2) {
  const Eigen::Index p = R.rows();
  auto system = gmm_block_system(R, blocks);
  if (!system.has_value()) return std::unexpected(system.error());
  auto h2 = solve_communality_system(*system, R_h2, r_h2);
  if (!h2.has_value()) return std::unexpected(h2.error());
  if (h2->size() != p)
    return mat_error("communality constrained GMM block Jacobian: h2 dimension "
                     "mismatch");

  auto h20_global =
      itemwise_h2(R, blocks, block_of, CommunalityMethod::TriadLeastSquares);
  if (!h20_global.has_value()) return std::unexpected(h20_global.error());
  auto Jh20_global =
      itemwise_h2_jacobian(R, blocks, block_of,
                           CommunalityMethod::TriadLeastSquares, J);
  if (!Jh20_global.has_value()) return std::unexpected(Jh20_global.error());

  Eigen::MatrixXd rhs_head = Eigen::MatrixXd::Zero(p, J.pstar);
  for (const auto& idx : blocks) {
    const Eigen::Index m = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Rb(m, m);
    std::vector<Eigen::Index> global_index(static_cast<std::size_t>(m));
    for (Eigen::Index a = 0; a < m; ++a) {
      global_index[static_cast<std::size_t>(a)] =
          idx[static_cast<std::size_t>(a)];
      for (Eigen::Index b = 0; b < m; ++b) {
        Rb(a, b) = R(idx[static_cast<std::size_t>(a)],
                     idx[static_cast<std::size_t>(b)]);
      }
    }

    std::vector<Eigen::Index> active_cols;
    active_cols.reserve(static_cast<std::size_t>(m * (m + 1) / 2));
    for (Eigen::Index c = 0; c < m; ++c) {
      const Eigen::Index gc = global_index[static_cast<std::size_t>(c)];
      for (Eigen::Index r = c; r < m; ++r) {
        const Eigen::Index gr = global_index[static_cast<std::size_t>(r)];
        active_cols.push_back(vech_index(gr, gc, J.p));
      }
    }
    const auto n_active = static_cast<Eigen::Index>(active_cols.size());

    std::vector<std::vector<Eigen::Index>> local_blocks(1);
    local_blocks[0].resize(static_cast<std::size_t>(m));
    for (Eigen::Index i = 0; i < m; ++i)
      local_blocks[0][static_cast<std::size_t>(i)] = i;

    Eigen::VectorXd h20(m);
    Eigen::VectorXd h_local(m);
    Eigen::MatrixXd Jh20(m, J.pstar);
    for (Eigen::Index i = 0; i < m; ++i) {
      const Eigen::Index gi = idx[static_cast<std::size_t>(i)];
      h20(i) = (*h20_global)(gi);
      h_local(i) = (*h2)(gi);
      Jh20.row(i) = Jh20_global->row(gi);
    }

    const std::vector<Pair> off = within_off_pairs(local_blocks);
    auto sys = triad_system(Rb, local_blocks, off, nullptr);
    if (!sys.has_value()) return std::unexpected(sys.error());
    auto sys_w = triad_system(Rb, local_blocks, off, &h20);
    if (!sys_w.has_value()) return std::unexpected(sys_w.error());

    const Eigen::MatrixXd gamma = normal_cor_gamma_pairs(Rb, off);
    const auto dgamma =
        normal_cor_gamma_pairs_jacobian(Rb, off, global_index, J, active_cols);

    const Eigen::MatrixXd omega =
        sys_w->M * gamma * sys_w->M.transpose();
    auto W = row_covariance_pinv(sys_w->M, gamma, omega);
    if (!W.has_value()) return std::unexpected(W.error());

    const Eigen::Index n_rows = sys->A.rows();
    const Eigen::Index n_pairs = static_cast<Eigen::Index>(off.size());
    std::vector<Eigen::MatrixXd> dA_cols(
        active_cols.size(),
        Eigen::MatrixXd::Zero(n_rows, m));
    std::vector<Eigen::MatrixXd> dM_cols(
        active_cols.size(),
        Eigen::MatrixXd::Zero(n_rows, n_pairs));
    Eigen::MatrixXd db_cols = Eigen::MatrixXd::Zero(n_rows, n_active);

    std::unordered_map<std::int64_t, Eigen::Index> off_index;
    off_index.reserve(off.size());
    for (std::size_t k = 0; k < off.size(); ++k) {
      off_index.emplace(pair_key(off[k].i, off[k].j, m),
                        static_cast<Eigen::Index>(k));
    }

    Eigen::Index row = 0;
    for (const Eigen::Index i : local_blocks[0]) {
      for (std::size_t a = 0; a < local_blocks[0].size(); ++a) {
        const Eigen::Index j = local_blocks[0][a];
        if (j == i) continue;
        for (std::size_t c = a + 1; c < local_blocks[0].size(); ++c) {
          const Eigen::Index k = local_blocks[0][c];
          if (k == i) continue;

          const Eigen::Index gi = global_index[static_cast<std::size_t>(i)];
          const Eigen::Index gj = global_index[static_cast<std::size_t>(j)];
          const Eigen::Index gk = global_index[static_cast<std::size_t>(k)];
          const Eigen::RowVectorXd d_rjk =
              corr_active_row(J, active_cols, gj, gk);
          const Eigen::RowVectorXd d_rij =
              corr_active_row(J, active_cols, gi, gj);
          const Eigen::RowVectorXd d_rik =
              corr_active_row(J, active_cols, gi, gk);

          const Eigen::RowVectorXd db_active =
              d_rij * Rb(i, k) + Rb(i, j) * d_rik;
          const Eigen::Index jk = pair_index(off_index, j, k, m);
          const Eigen::Index ij = pair_index(off_index, i, j, m);
          const Eigen::Index ik = pair_index(off_index, i, k, m);
          if (jk < 0 || ij < 0 || ik < 0)
            return mat_error("communality constrained GMM block Jacobian: "
                             "selected pair set is incomplete");

          for (std::size_t active = 0; active < active_cols.size(); ++active) {
            const Eigen::Index col = active_cols[active];
            const auto ac = static_cast<Eigen::Index>(active);
            db_cols(row, ac) = db_active(ac);
            dA_cols[active](row, i) = d_rjk(ac);
            dM_cols[active](row, jk) += Jh20(i, col);
            dM_cols[active](row, ij) -= d_rik(ac);
            dM_cols[active](row, ik) -= d_rij(ac);
          }
          ++row;
        }
      }
    }

    CommunalitySystem local_system{
        std::move(sys->A), std::move(sys->b), std::move(*W)};
    auto rhs_local = gmm_jacobian_rhs_weight_actions(
        local_system, omega, sys_w->M, gamma, dA_cols, db_cols, dM_cols,
        dgamma, h_local);
    if (!rhs_local.has_value()) return std::unexpected(rhs_local.error());
    for (Eigen::Index i = 0; i < m; ++i) {
      const Eigen::Index gi = idx[static_cast<std::size_t>(i)];
      for (std::size_t active = 0; active < active_cols.size(); ++active) {
        rhs_head(gi, active_cols[active]) =
            (*rhs_local)(i, static_cast<Eigen::Index>(active));
      }
    }
  }

  return solve_constrained_h2_jacobian_from_rhs(
      *system, R_h2, r_h2, rhs_head, S, *h2);
}

fit_expected<Eigen::VectorXd>
gmm_block_h2(const Eigen::MatrixXd& R,
             const std::vector<std::vector<Eigen::Index>>& blocks) {
  auto sys = gmm_block_system(R, blocks);
  if (!sys.has_value()) return std::unexpected(sys.error());
  auto out = solve_gmm(sys->A, sys->b, sys->W);
  if (!out.has_value()) return std::unexpected(out.error());
  if (!out->allFinite())
    return num_error("communality GMM block: non-finite output");
  return out;
}

fit_expected<Eigen::VectorXd>
gmm_full_h2(const Eigen::MatrixXd& R,
            const std::vector<std::vector<Eigen::Index>>& blocks,
            const std::vector<Eigen::Index>& block_of) {
  auto sys = gmm_full_system(R, blocks, block_of);
  if (!sys.has_value()) return std::unexpected(sys.error());
  return solve_gmm(sys->A, sys->b, sys->W);
}

}  // namespace

const char* communality_method_name(CommunalityMethod method) {
  switch (method) {
    case CommunalityMethod::AverageRatio:
      return "ar";
    case CommunalityMethod::RatioOfSums:
      return "rs";
    case CommunalityMethod::TriadLeastSquares:
      return "triad_ls";
    case CommunalityMethod::AnchorTriadLeastSquares:
      return "anchor_triad_ls";
    case CommunalityMethod::GmmBlock:
      return "gmm_block";
    case CommunalityMethod::GmmFull:
      return "gmm_full";
  }
  return "unknown";
}

fit_expected<CommunalitySystem>
communality_system(const Eigen::MatrixXd& S,
                   const std::vector<std::int32_t>& block_of_indicator,
                   CommunalityMethod method) {
  auto vin = validate_input(S, block_of_indicator);
  if (!vin.has_value()) return std::unexpected(vin.error());

  switch (method) {
    case CommunalityMethod::TriadLeastSquares:
      return triad_ls_system(vin->R, vin->blocks, vin->block_of,
                             /*include_anchor_rows=*/false);
    case CommunalityMethod::AnchorTriadLeastSquares:
      return triad_ls_system(vin->R, vin->blocks, vin->block_of,
                             /*include_anchor_rows=*/true);
    case CommunalityMethod::GmmBlock:
      return gmm_block_system(vin->R, vin->blocks);
    case CommunalityMethod::GmmFull:
      return gmm_full_system(vin->R, vin->blocks, vin->block_of);
    case CommunalityMethod::AverageRatio:
    case CommunalityMethod::RatioOfSums:
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality constrained system: method is not a least-squares/GMM "
          "triad system"});
  }
  return std::unexpected(FitError{FitError::Kind::NumericIssue,
      "communality constrained system: unknown method"});
}

fit_expected<CommunalitySystemDirection>
communality_system_directional(
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& dS,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method) {
  auto vin = validate_input(S, block_of_indicator);
  if (!vin.has_value()) return std::unexpected(vin.error());
  auto dR = correlation_direction(S, dS, vin->R);
  if (!dR.has_value()) return std::unexpected(dR.error());

  switch (method) {
    case CommunalityMethod::TriadLeastSquares: {
      const std::vector<Pair> off = within_off_pairs(vin->blocks);
      auto sys = triad_system_directional(vin->R, *dR, vin->blocks, off, nullptr,
                                          nullptr);
      if (!sys.has_value()) return std::unexpected(sys.error());
      CommunalitySystemDirection out;
      out.system.A = std::move(sys->sys.A);
      out.system.b = std::move(sys->sys.b);
      out.system.W = Eigen::MatrixXd::Identity(out.system.A.rows(),
                                               out.system.A.rows());
      out.dA = std::move(sys->dA);
      out.db = std::move(sys->db);
      out.dW = Eigen::MatrixXd::Zero(out.system.A.rows(), out.system.A.rows());
      return out;
    }
    case CommunalityMethod::AnchorTriadLeastSquares: {
      auto sys = triad_system_directional(
          vin->R, *dR, vin->blocks, std::vector<Pair>{}, nullptr, nullptr,
          /*include_anchor_rows=*/true, &vin->block_of);
      if (!sys.has_value()) return std::unexpected(sys.error());
      CommunalitySystemDirection out;
      out.system.A = std::move(sys->sys.A);
      out.system.b = std::move(sys->sys.b);
      out.system.W = Eigen::MatrixXd::Identity(out.system.A.rows(),
                                               out.system.A.rows());
      out.dA = std::move(sys->dA);
      out.db = std::move(sys->db);
      out.dW = Eigen::MatrixXd::Zero(out.system.A.rows(), out.system.A.rows());
      return out;
    }
    case CommunalityMethod::GmmBlock:
      return gmm_block_system_directional(vin->R, *dR, vin->blocks);
    case CommunalityMethod::GmmFull:
      return gmm_full_system_directional(vin->R, *dR, vin->blocks, vin->block_of);
    case CommunalityMethod::AverageRatio:
    case CommunalityMethod::RatioOfSums:
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality system derivative: method is not a least-squares/GMM "
          "triad system"});
  }
  return std::unexpected(FitError{FitError::Kind::NumericIssue,
      "communality system derivative: unknown method"});
}

fit_expected<Eigen::VectorXd>
solve_communality_system(const CommunalitySystem& system,
                         const Eigen::MatrixXd& R_h2,
                         const Eigen::VectorXd& r_h2) {
  if (system.A.rows() != system.b.size() ||
      system.W.rows() != system.A.rows() ||
      system.W.cols() != system.A.rows()) {
    return num_error("communality constrained GMM: system dimension mismatch");
  }
  if (R_h2.rows() != r_h2.size() ||
      (R_h2.rows() > 0 && R_h2.cols() != system.A.cols())) {
    return num_error("communality constrained GMM: constraint dimension mismatch");
  }
  if (!system.A.allFinite() || !system.b.allFinite() || !system.W.allFinite() ||
      !R_h2.allFinite() || !r_h2.allFinite()) {
    return num_error("communality constrained GMM: non-finite input");
  }
  if (R_h2.rows() == 0)
    return solve_gmm(system.A, system.b, system.W);

  const Eigen::MatrixXd G = system.A.transpose() * system.W * system.A;
  const Eigen::VectorXd rhs = system.A.transpose() * system.W * system.b;
  auto Ginv = symmetric_pinv(G);
  if (!Ginv.has_value()) return std::unexpected(Ginv.error());
  Eigen::VectorXd h = (*Ginv) * rhs;

  const Eigen::MatrixXd GRt = (*Ginv) * R_h2.transpose();
  const Eigen::MatrixXd Ssm = R_h2 * GRt;
  auto Ssm_inv = symmetric_pinv(Ssm);
  if (!Ssm_inv.has_value()) return std::unexpected(Ssm_inv.error());
  h -= GRt * ((*Ssm_inv) * (R_h2 * h - r_h2));
  if (!h.allFinite())
    return num_error("communality constrained GMM: non-finite solution");

  const Eigen::VectorXd residual = R_h2 * h - r_h2;
  const double scale = std::max(1.0, r_h2.cwiseAbs().maxCoeff());
  if (residual.cwiseAbs().maxCoeff() > 1e-8 * scale) {
    return num_error("communality constrained GMM: infeasible h2 constraints");
  }
  return h;
}

fit_expected<Eigen::VectorXd>
estimate_h2_communalities(const Eigen::MatrixXd& S,
                          const std::vector<std::int32_t>& block_of_indicator,
                          CommunalityMethod method) {
  auto vin = validate_input(S, block_of_indicator);
  if (!vin.has_value()) return std::unexpected(vin.error());

  switch (method) {
    case CommunalityMethod::AverageRatio:
    case CommunalityMethod::RatioOfSums:
    case CommunalityMethod::TriadLeastSquares:
      return itemwise_h2(vin->R, vin->blocks, vin->block_of, method);
    case CommunalityMethod::AnchorTriadLeastSquares:
      return anchor_triad_ls_h2(vin->R, vin->blocks, vin->block_of);
    case CommunalityMethod::GmmBlock:
      return gmm_block_h2(vin->R, vin->blocks);
    case CommunalityMethod::GmmFull:
      return gmm_full_h2(vin->R, vin->blocks, vin->block_of);
  }
  return num_error("communality estimator: unknown method");
}

fit_expected<Eigen::VectorXd>
estimate_h2_communalities_directional(
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& dS,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method) {
  auto vin = validate_input(S, block_of_indicator);
  if (!vin.has_value()) return std::unexpected(vin.error());
  auto dR = correlation_direction(S, dS, vin->R);
  if (!dR.has_value()) return std::unexpected(dR.error());

  switch (method) {
    case CommunalityMethod::AverageRatio:
    case CommunalityMethod::RatioOfSums:
    case CommunalityMethod::TriadLeastSquares:
      return itemwise_h2_directional(vin->R, *dR, vin->blocks, vin->block_of,
                                     method);
    case CommunalityMethod::AnchorTriadLeastSquares:
      return anchor_triad_ls_h2_directional(vin->R, *dR, vin->blocks,
                                            vin->block_of);
    case CommunalityMethod::GmmBlock: {
      auto sys = gmm_block_system_directional(vin->R, *dR, vin->blocks);
      if (!sys.has_value()) return std::unexpected(sys.error());
      return solve_gmm_directional(sys->system, sys->dA, sys->db, sys->dW);
    }
    case CommunalityMethod::GmmFull: {
      auto sys = gmm_full_system_directional(vin->R, *dR, vin->blocks,
                                             vin->block_of);
      if (!sys.has_value()) return std::unexpected(sys.error());
      return solve_gmm_directional(sys->system, sys->dA, sys->db, sys->dW);
    }
  }
  return num_error("communality derivative: unknown method");
}

fit_expected<Eigen::MatrixXd>
estimate_h2_communalities_jacobian(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method) {
  auto vin = validate_input(S, block_of_indicator);
  if (!vin.has_value()) return std::unexpected(vin.error());
  auto J = correlation_jacobian(S, vin->R);
  if (!J.has_value()) return std::unexpected(J.error());

  switch (method) {
    case CommunalityMethod::AverageRatio:
    case CommunalityMethod::RatioOfSums:
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality Jacobian: batched path is only enabled for LS/GMM "
          "methods"});
    case CommunalityMethod::TriadLeastSquares:
      return itemwise_h2_jacobian(vin->R, vin->blocks, vin->block_of, method,
                                  *J);
    case CommunalityMethod::AnchorTriadLeastSquares:
      return anchor_triad_ls_h2_jacobian(vin->R, vin->blocks, vin->block_of,
                                         *J);
    case CommunalityMethod::GmmBlock:
      return gmm_block_h2_jacobian(vin->R, vin->blocks, vin->block_of, *J);
    case CommunalityMethod::GmmFull:
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "communality Jacobian: gmm_full batched derivative is deferred"});
  }
  return mat_error("communality Jacobian: unknown method");
}

fit_expected<Eigen::MatrixXd>
estimate_h2_communalities_constrained_jacobian(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2) {
  auto vin = validate_input(S, block_of_indicator);
  if (!vin.has_value()) return std::unexpected(vin.error());
  if (R_h2.rows() != r_h2.size() ||
      (R_h2.rows() > 0 && R_h2.cols() != S.rows()))
    return mat_error("communality constrained Jacobian: constraint dimension "
                     "mismatch");
  if (!R_h2.allFinite() || !r_h2.allFinite())
    return mat_error("communality constrained Jacobian: non-finite constraints");

  auto J = correlation_jacobian(S, vin->R);
  if (!J.has_value()) return std::unexpected(J.error());

  switch (method) {
    case CommunalityMethod::GmmBlock:
      return gmm_block_h2_constrained_jacobian(
          S, vin->R, vin->blocks, vin->block_of, *J, R_h2, r_h2);
    case CommunalityMethod::TriadLeastSquares:
    case CommunalityMethod::AnchorTriadLeastSquares:
    case CommunalityMethod::GmmFull:
      return mat_error("communality constrained Jacobian: batched constrained "
                       "path is currently implemented for gmm_block");
    case CommunalityMethod::AverageRatio:
    case CommunalityMethod::RatioOfSums:
      return mat_error("communality constrained Jacobian: method is not a "
                       "least-squares/GMM constrained system");
  }
  return mat_error("communality constrained Jacobian: unknown method");
}

fit_expected<Eigen::VectorXd>
estimate_h2_communalities_constrained(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2) {
  auto sys = communality_system(S, block_of_indicator, method);
  if (!sys.has_value()) return std::unexpected(sys.error());
  return solve_communality_system(*sys, R_h2, r_h2);
}

fit_expected<HCommunalityResult>
estimate_h_communalities(const Eigen::MatrixXd& S,
                         const std::vector<std::int32_t>& block_of_indicator,
                         CommunalityMethod method) {
  auto h2 = estimate_h2_communalities(S, block_of_indicator, method);
  if (!h2.has_value()) return std::unexpected(h2.error());
  Eigen::VectorXd h_diag = S.diagonal().cwiseProduct(*h2);
  Eigen::MatrixXd H = S;
  H.diagonal() = h_diag;
  return HCommunalityResult{std::move(*h2), std::move(h_diag), std::move(H)};
}

fit_expected<HCommunalityResult>
estimate_h_communalities_constrained(
    const Eigen::MatrixXd& S,
    const std::vector<std::int32_t>& block_of_indicator,
    CommunalityMethod method,
    const Eigen::MatrixXd& R_h2,
    const Eigen::VectorXd& r_h2) {
  auto h2 = estimate_h2_communalities_constrained(S, block_of_indicator,
                                                  method, R_h2, r_h2);
  if (!h2.has_value()) return std::unexpected(h2.error());
  Eigen::VectorXd h_diag = S.diagonal().cwiseProduct(*h2);
  Eigen::MatrixXd H = S;
  H.diagonal() = h_diag;
  return HCommunalityResult{std::move(*h2), std::move(h_diag), std::move(H)};
}

}  // namespace magmaan::estimate::frontier
