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

fit_expected<TriadSystem>
triad_system(const Eigen::MatrixXd& R,
             const std::vector<std::vector<Eigen::Index>>& blocks,
             const std::vector<Pair>& selected_off,
             const Eigen::VectorXd* h2_for_jac) {
  const Eigen::Index p = R.rows();
  Eigen::Index n_rows = 0;
  for (const auto& idx : blocks) {
    const auto m = static_cast<Eigen::Index>(idx.size());
    n_rows += m * (m - 1) * (m - 2) / 2;
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
      }
    }
  }
  return sys;
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
gmm_block_h2(const Eigen::MatrixXd& R,
             const std::vector<std::vector<Eigen::Index>>& blocks) {
  const Eigen::Index p = R.rows();
  Eigen::VectorXd out(p);

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
    const Eigen::MatrixXd omega =
        sys_w->M * normal_cor_gamma_pairs(Rb, off) * sys_w->M.transpose();
    auto W = symmetric_pinv(omega);
    if (!W.has_value()) return std::unexpected(W.error());
    auto h2 = solve_gmm(sys->A, sys->b, *W);
    if (!h2.has_value()) return std::unexpected(h2.error());
    for (Eigen::Index i = 0; i < m; ++i) {
      out(idx[static_cast<std::size_t>(i)]) = (*h2)(i);
    }
  }
  if (!out.allFinite())
    return num_error("communality GMM block: non-finite output");
  return out;
}

fit_expected<Eigen::VectorXd>
gmm_full_h2(const Eigen::MatrixXd& R,
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
  const Eigen::MatrixXd omega =
      sys_w->M * normal_cor_gamma_pairs(R, off) * sys_w->M.transpose();
  auto W = symmetric_pinv(omega);
  if (!W.has_value()) return std::unexpected(W.error());
  return solve_gmm(sys->A, sys->b, *W);
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

}  // namespace magmaan::estimate::frontier
