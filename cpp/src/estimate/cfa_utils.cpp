#include "magmaan/estimate/cfa_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/Dense>

namespace magmaan::estimate {

Eigen::VectorXd theta_spearman(const Eigen::MatrixXd& S) {
  const Eigen::Index p = S.rows();
  Eigen::VectorXd out = Eigen::VectorXd::Zero(p);
  if (p < 3) {  // communality undefined for < 3 indicators
    for (Eigen::Index i = 0; i < p; ++i) out(i) = 0.5 * S(i, i);
    return out;
  }
  // Correlation matrix.
  Eigen::VectorXd sd(p);
  for (Eigen::Index i = 0; i < p; ++i) {
    sd(i) = std::sqrt(S(i, i) > 0.0 ? S(i, i) : 1.0);
  }
  Eigen::MatrixXd R(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) R(i, j) = S(i, j) / (sd(i) * sd(j));
  }
  for (Eigen::Index k = 0; k < p; ++k) {
    // h² = mean over pairs (a<b, a≠k, b≠k) of R_ak·R_bk / R_ab.
    double sum_ratio = 0.0;
    long count = 0;
    for (Eigen::Index a = 0; a < p; ++a) {
      if (a == k) continue;
      for (Eigen::Index b = a + 1; b < p; ++b) {
        if (b == k) continue;
        const double ss = R(a, b);
        if (ss != 0.0) {
          sum_ratio += (R(a, k) * R(b, k)) / ss;
          ++count;
        }
      }
    }
    double h2 = count > 0 ? sum_ratio / static_cast<double>(count) : 0.0;
    h2 = std::min(std::max(h2, -0.05), 1.20);  // "wide" bounds
    out(k) = (1.0 - h2) * S(k, k);
  }
  return out;
}

std::optional<double> smallest_gen_root(const Eigen::MatrixXd& A,
                                        const Eigen::MatrixXd& B) {
  Eigen::LLT<Eigen::MatrixXd> llt(B);
  if (llt.info() != Eigen::Success) return std::nullopt;
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(A, B);
  if (ges.info() != Eigen::Success) return std::nullopt;
  if (ges.eigenvalues().size() == 0) return std::nullopt;
  return ges.eigenvalues()(0);  // ascending → smallest
}

Eigen::MatrixXd force_pd(const Eigen::MatrixXd& M, double tol) {
  const Eigen::MatrixXd sym = 0.5 * (M + M.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sym);
  if (es.info() != Eigen::Success) return sym;
  Eigen::VectorXd ev = es.eigenvalues();
  bool clamped = false;
  for (Eigen::Index i = 0; i < ev.size(); ++i) {
    if (ev(i) < tol) {
      ev(i) = tol;
      clamped = true;
    }
  }
  if (!clamped) return sym;
  return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
}

std::vector<CfaBlockLayout>
cfa_block_layouts(const spec::LatentStructure& pt, const model::MatrixRep& rep) {
  std::vector<CfaBlockLayout> out(rep.dims.size());
  for (std::size_t b = 0; b < rep.dims.size(); ++b) {
    out[b].n_observed = rep.dims[b].n_observed;
  }

  // Map original latent column → compacted factor index, per block.
  std::vector<std::vector<std::int16_t>> col_to_factor(rep.dims.size());
  for (std::size_t b = 0; b < rep.dims.size(); ++b) {
    col_to_factor[b].assign(static_cast<std::size_t>(rep.dims[b].n_latent),
                            std::int16_t{-1});
  }

  // First pass: discover factors that carry observed Lambda indicators.
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto& c = rep.cell_for_row[i];
    if (!c.used || c.mat != model::MatId::Lambda) continue;
    const std::size_t b = static_cast<std::size_t>(c.block);
    if (b >= out.size()) continue;
    const std::size_t col = static_cast<std::size_t>(c.col);
    if (col >= col_to_factor[b].size()) continue;
    if (col_to_factor[b][col] < 0) {
      col_to_factor[b][col] =
          static_cast<std::int16_t>(out[b].factor_col.size());
      out[b].factor_col.push_back(c.col);
      out[b].marker_ov.push_back(-1);
    }
  }

  // Second pass: collect the loads and markers; flag crossloadings.
  std::vector<std::vector<int>> factors_per_ov(rep.dims.size());
  for (std::size_t b = 0; b < rep.dims.size(); ++b) {
    factors_per_ov[b].assign(static_cast<std::size_t>(rep.dims[b].n_observed),
                             0);
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto& c = rep.cell_for_row[i];
    if (!c.used || c.mat != model::MatId::Lambda) continue;
    const std::size_t b = static_cast<std::size_t>(c.block);
    if (b >= out.size()) continue;
    const std::size_t col = static_cast<std::size_t>(c.col);
    if (col >= col_to_factor[b].size() || col_to_factor[b][col] < 0) continue;

    CfaBlockLayout::Load load;
    load.ov_row = c.row;
    load.factor = col_to_factor[b][col];
    load.free_idx = pt.free[i] > 0 ? pt.free[i] - 1 : -1;
    out[b].loads.push_back(load);

    if (c.row >= 0 &&
        static_cast<std::size_t>(c.row) < factors_per_ov[b].size()) {
      ++factors_per_ov[b][static_cast<std::size_t>(c.row)];
    }
    if (pt.free[i] == 0) {  // a fixed loading marks the factor's scale
      out[b].marker_ov[static_cast<std::size_t>(load.factor)] = c.row;
    }
  }

  for (std::size_t b = 0; b < out.size(); ++b) {
    for (int n : factors_per_ov[b]) {
      if (n > 1) out[b].crossloadings = true;
    }
    for (std::int16_t m : out[b].marker_ov) {
      if (m < 0) out[b].all_have_marker = false;
    }
  }
  return out;
}

}  // namespace magmaan::estimate
