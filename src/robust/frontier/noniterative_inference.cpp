#include "magmaan/robust/frontier/noniterative_inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/error.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
#include "magmaan/robust/satorra2000.hpp"

namespace magmaan::robust::frontier {

namespace {

std::unexpected<PostError> perr(std::string detail) {
  return std::unexpected(PostError{PostError::Kind::NumericIssue, std::move(detail)});
}

PostError model_to_post(ModelError e) {
  return PostError{PostError::Kind::NumericIssue, std::move(e.detail)};
}

// vech, lower-triangle column-major (aligns with dsigma_dtheta / gamma_nt).
Eigen::VectorXd vech_lower(const Eigen::MatrixXd& M) {
  const Eigen::Index p = M.rows();
  Eigen::VectorXd v(p * (p + 1) / 2);
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c)
    for (Eigen::Index r = c; r < p; ++r) v(k++) = M(r, c);
  return v;
}

// ULS weight V_U = D'D over vech: 1 on variance coordinates (r==c), 2 on
// covariance coordinates. Diagonal. (Not magmaan's identity-vech ULS.)
Eigen::MatrixXd uls_weight(Eigen::Index p) {
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::VectorXd d(pstar);
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c)
    for (Eigen::Index r = c; r < p; ++r) d(k++) = (r == c) ? 1.0 : 2.0;
  return d.asDiagonal();
}

double total_n(const data::SampleStats& samp) {
  double n = 0.0;
  for (auto v : samp.n_obs) n += static_cast<double>(v);
  return n;
}

// True when the model carries free intercepts / latent means. In that case the
// moment vector is mean-augmented [m_b ; vech(S_b)] (mean-first), the estimator
// Jacobian gains an analytic mean block (∂ν/∂m = I), and Γ is the mean-augmented
// NACOV. The GOF spectrum is unchanged (the saturated mean part contributes zero
// eigenvalues); only Ω picks up the mean block.
bool has_mean_params(const model::ModelEvaluator& ev) {
  for (const auto& loc : ev.param_locations())
    if (loc.mat == model::MatId::Nu || loc.mat == model::MatId::Alpha)
      return true;
  return false;
}

// Pseudo-inverse quadratic form d' C⁺ d for a symmetric PSD (possibly singular)
// C, plus the numeric rank of C. Directions with eigenvalue ≤ rtol·λ_max are
// dropped (they carry the deterministic-zero components of a degenerate Wald
// statistic). Returns {W, rank}.
struct PinvQuad { double W; int rank; };
PinvQuad psd_pinv_quadform(const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
                           double rtol) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
  const Eigen::VectorXd& lam = es.eigenvalues();       // ascending
  const Eigen::MatrixXd& Vv = es.eigenvectors();
  const double lam_max = lam.size() ? std::max(lam(lam.size() - 1), 0.0) : 0.0;
  const double tol = rtol * lam_max;
  const Eigen::VectorXd proj = Vv.transpose() * d;
  double W = 0.0;
  int rank = 0;
  for (Eigen::Index i = 0; i < lam.size(); ++i) {
    if (lam(i) > tol && lam(i) > 0.0) {
      W += proj(i) * proj(i) / lam(i);
      ++rank;
    }
  }
  return {W, rank};
}

int numeric_rank(const Eigen::MatrixXd& A) {
  if (A.size() == 0) return 0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.singularValues().size() == 0) return 0;
  const double smax = svd.singularValues()(0);
  const double dim = static_cast<double>(
      std::max<Eigen::Index>(A.rows(), A.cols()));
  const double tol = std::sqrt(std::numeric_limits<double>::epsilon()) *
                     dim * std::max(1.0, smax);
  int rank = 0;
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i)
    if (svd.singularValues()(i) > tol) ++rank;
  return rank;
}

post_expected<Eigen::MatrixXd>
symmetric_pinv_psd(const Eigen::MatrixXd& A, const char* what) {
  if (A.rows() != A.cols()) {
    return perr(std::string(what) + " is not square");
  }
  if (!A.allFinite()) return perr(std::string(what) + " is non-finite");
  if (A.rows() == 0) return Eigen::MatrixXd(0, 0);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A + A.transpose()));
  if (es.info() != Eigen::Success) {
    return perr(std::string(what) + " eigensolve failed");
  }
  const Eigen::VectorXd& lam = es.eigenvalues();
  const double lam_max = std::max(lam(lam.size() - 1), 0.0);
  const double tol = std::sqrt(std::numeric_limits<double>::epsilon()) *
                     static_cast<double>(std::max<Eigen::Index>(A.rows(), A.cols())) *
                     std::max(1.0, lam_max);
  Eigen::VectorXd inv = Eigen::VectorXd::Zero(lam.size());
  for (Eigen::Index i = 0; i < lam.size(); ++i) {
    if (lam(i) > tol) inv(i) = 1.0 / lam(i);
  }
  return es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
}

enum class MapKind : std::uint8_t { Configural, Restricted };

Eigen::VectorXd se_from_omega(const Eigen::MatrixXd& Omega) {
  Eigen::VectorXd se(Omega.rows());
  for (Eigen::Index i = 0; i < Omega.rows(); ++i)
    se(i) = std::sqrt(std::max(Omega(i, i), 0.0));
  return se;
}

Eigen::MatrixXd casewise_moment_rows(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                     const Eigen::MatrixXd& S,
                                     const Eigen::VectorXd& mean,
                                     bool include_means) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::Index pstar = p * (p + 1) / 2;
  Eigen::MatrixXd Z(n, (include_means ? p : 0) + pstar);
  const Eigen::Index sig_off = include_means ? p : 0;
  for (Eigen::Index i = 0; i < n; ++i) {
    if (include_means) {
      for (Eigen::Index a = 0; a < p; ++a)
        Z(i, a) = X(i, a) - mean(a);
    }
    Eigen::Index k = sig_off;
    for (Eigen::Index c = 0; c < p; ++c) {
      const double xc = X(i, c) - mean(c);
      for (Eigen::Index r = c; r < p; ++r)
        Z(i, k++) = (X(i, r) - mean(r)) * xc - S(r, c);
    }
  }
  return Z;
}

struct SingleDerivative {
  Eigen::MatrixXd Sigma_hat;
  Eigen::MatrixXd Delta;
  Eigen::MatrixXd J;
  Eigen::Index pstar = 0;
  Eigen::Index q = 0;
  double N = 0.0;
};

post_expected<SingleDerivative>
build_single_derivative(const spec::LatentStructure& pt,
                        const model::MatrixRep& rep,
                        const data::SampleStats& samp,
                        const Eigen::VectorXd& theta,
                        estimate::frontier::NonIterativeEstimator which,
                        MapKind map_kind,
                        estimate::frontier::CommunalityMethod comm,
                        estimate::frontier::CompositeWeight composite) {
  if (samp.S.empty()) return perr("non-iterative SE: empty sample stats");
  const Eigen::MatrixXd& S = samp.S[0];
  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;

  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("non-iterative SE: evaluator build failed");
  const auto q = static_cast<Eigen::Index>(ev->n_free());
  if (theta.size() != q) return perr("non-iterative SE: theta size mismatch");
  if (has_mean_params(*ev))
    return perr("non-iterative SE: mean structure requires the grouped path");

  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("non-iterative SE: Sigma(theta) failed");
  auto dS = ev->dsigma_dtheta(theta);
  if (!dS.has_value()) return perr("non-iterative SE: dsigma_dtheta failed");

  auto Jf = (map_kind == MapKind::Restricted)
                ? estimate::frontier::estimator_map_jacobian_restricted(
                      pt, rep, *ev, samp, which, 1e-6, comm, composite)
                : estimate::frontier::estimator_map_jacobian(
                      pt, rep, *ev, samp, which, 1e-6, composite);
  if (!Jf.has_value()) return perr("non-iterative SE: estimator Jacobian failed");
  if (Jf->rows() != q || Jf->cols() != pstar)
    return perr("non-iterative SE: estimator Jacobian dimension mismatch");

  SingleDerivative out;
  out.Sigma_hat = sig->sigma[0];
  out.Delta = std::move(*dS);
  out.J = std::move(*Jf);
  out.pstar = pstar;
  out.q = q;
  out.N = total_n(samp);
  return out;
}

NonIterativeSE finish_single_se(const Eigen::VectorXd& theta,
                                const SingleDerivative& d,
                                Eigen::MatrixXd Omega) {
  NonIterativeSE out;
  out.theta_hat = theta;
  out.Omega = 0.5 * (Omega + Omega.transpose()).eval();
  out.se = se_from_omega(out.Omega);
  out.block_of_param.assign(static_cast<std::size_t>(d.q), -1);
  return out;
}

struct MiCandidateRow {
  std::size_t row = 0;
  double fixed_value = 0.0;
  inference::ScoreCandidate candidate;
};

bool var_is_latent(const spec::LatentStructure& pt, std::int32_t v) {
  return v >= 0 && static_cast<std::size_t>(v) < pt.var_role.size() &&
         pt.var_role[static_cast<std::size_t>(v)] == spec::VarRole::Latent;
}

bool var_is_indicator(const spec::LatentStructure& pt, std::int32_t v) {
  return v >= 0 && static_cast<std::size_t>(v) < pt.var_role.size() &&
         pt.var_role[static_cast<std::size_t>(v)] == spec::VarRole::Indicator;
}

bool fixed_zero_candidate_row(const spec::LatentStructure& pt,
                              const model::MatrixRep& rep,
                              std::size_t row) {
  if (row >= pt.size() || row >= rep.cell_for_row.size()) return false;
  if (pt.is_constraint_row(row)) return false;
  if (!rep.cell_for_row[row].used) return false;
  if (row >= pt.free.size() || pt.free[row] != 0) return false;
  if (row < pt.exo.size() && pt.exo[row] != 0) return false;
  if (row >= pt.fixed_value.size() || !std::isfinite(pt.fixed_value[row]))
    return false;
  return std::abs(pt.fixed_value[row]) <= 1e-14;
}

inference::ScoreCandidate make_mi_candidate(const spec::LatentStructure& pt,
                                            std::size_t row) {
  inference::ScoreCandidate cand;
  cand.kind = inference::ScoreCandidateKind::FixedParam;
  cand.row = row;
  cand.op = pt.op[row];
  cand.lhs_var = pt.lhs_var[row];
  cand.rhs_var = pt.rhs_var[row];
  cand.group = pt.group[row];
  return cand;
}

std::vector<MiCandidateRow>
collect_mi_candidates(const spec::LatentStructure& pt,
                      const model::MatrixRep& rep) {
  std::vector<MiCandidateRow> out;
  std::set<std::array<std::int32_t, 4>> seen;
  for (std::size_t row = 0; row < pt.size(); ++row) {
    if (!fixed_zero_candidate_row(pt, rep, row)) continue;
    const auto& c = rep.cell_for_row[row];
    const std::array<std::int32_t, 4> key{
        static_cast<std::int32_t>(c.block),
        static_cast<std::int32_t>(c.mat),
        static_cast<std::int32_t>(c.row),
        static_cast<std::int32_t>(c.col)};
    if (!seen.insert(key).second) continue;
    out.push_back(MiCandidateRow{row, pt.fixed_value[row],
                                 make_mi_candidate(pt, row)});
  }
  return out;
}

void append_absent_mi_rows(spec::LatentStructure& pt,
                           const inference::ModificationIndexOptions& opts) {
  if (opts.candidates != inference::ScoreCandidateSet::WithAbsentRows) return;

  std::vector<std::int32_t> latents;
  std::vector<std::int32_t> indicators;
  for (std::int32_t v = 0; v < pt.n_vars; ++v) {
    if (var_is_latent(pt, v)) latents.push_back(v);
    if (var_is_indicator(pt, v)) indicators.push_back(v);
  }

  using Key = std::array<std::int32_t, 3>;  // {op-tag, a, b}
  auto append_row = [&](parse::Op op, std::int32_t lhs, std::int32_t rhs,
                        std::int32_t group) {
    pt.op.push_back(op);
    pt.lhs_var.push_back(lhs);
    pt.rhs_var.push_back(rhs);
    pt.group.push_back(group);
    if (!pt.level.empty()) pt.level.push_back(1);
    pt.free.push_back(0);
    pt.exo.push_back(0);
    pt.fixed_value.push_back(0.0);
  };

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
      }
    }
    if (opts.include_loadings) {
      for (const std::int32_t f : latents) {
        for (const std::int32_t x : indicators) {
          if (!present.count({0, f, x})) {
            append_row(parse::Op::Measurement, f, x, g);
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
              append_row(parse::Op::Covariance, a, b, g);
            }
          }
        }
      };
      cov_pairs(indicators);
      cov_pairs(latents);
    }
  }
}

void free_mi_candidate_rows(spec::LatentStructure& pt,
                            const std::vector<MiCandidateRow>& rows) {
  const std::int32_t old_n = pt.n_free();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::size_t row = rows[i].row;
    pt.free[row] = old_n + static_cast<std::int32_t>(i) + 1;
    pt.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
  }
  if (static_cast<std::int32_t>(pt.eq_groups.size()) == old_n) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      pt.eq_groups.push_back(old_n + static_cast<std::int32_t>(i));
    }
  } else if (!pt.eq_groups.empty()) {
    pt.eq_groups.clear();
  }
}

Eigen::VectorXd append_mi_candidate_thetas(
    const Eigen::VectorXd& theta,
    const std::vector<MiCandidateRow>& rows) {
  Eigen::VectorXd out(theta.size() + static_cast<Eigen::Index>(rows.size()));
  if (theta.size() > 0) out.head(theta.size()) = theta;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    out(theta.size() + static_cast<Eigen::Index>(i)) = rows[i].fixed_value;
  }
  return out;
}

struct MiVariantStats {
  double score = std::numeric_limits<double>::quiet_NaN();
  double var = std::numeric_limits<double>::quiet_NaN();
  double z = std::numeric_limits<double>::quiet_NaN();
  double mi = std::numeric_limits<double>::quiet_NaN();
  double p = std::numeric_limits<double>::quiet_NaN();
  double epc = std::numeric_limits<double>::quiet_NaN();
  double drop = std::numeric_limits<double>::quiet_NaN();
  double p_drop = std::numeric_limits<double>::quiet_NaN();
};

MiVariantStats mi_variant(const Eigen::VectorXd& g,
                          const Eigen::VectorXd& r,
                          const Eigen::MatrixXd& V,
                          const Eigen::MatrixXd& Omega_r,
                          double N) {
  MiVariantStats out;
  const Eigen::VectorXd Vg = V * g;
  const double denom = g.dot(Vg);
  out.score = g.dot(V * r);
  out.var = Vg.dot(Omega_r * Vg);
  const double tol_denom = 1e-12 * std::max(1.0, std::abs(denom));
  if (denom > tol_denom && std::isfinite(out.score)) {
    out.epc = out.score / denom;
    out.drop = N * out.score * out.score / denom;
    out.p_drop = inference::chi2_pvalue(out.drop, 1);
  }
  const double tol_var = 1e-12 * std::max(1.0, std::abs(out.var));
  if (out.var > tol_var && std::isfinite(out.score)) {
    out.z = out.score / std::sqrt(out.var);
    out.mi = out.z * out.z;
    out.p = inference::chi2_pvalue(out.mi, 1);
  }
  return out;
}

struct GroupedDerivative {
  std::vector<model::ParamLocation> locs;
  std::vector<Eigen::Index> pstar_b;
  std::vector<Eigen::Index> offset;
  std::vector<Eigen::MatrixXd> J_blocks;
  Eigen::Index pstar_total = 0;
  Eigen::Index q = 0;
  bool has_means = false;
  std::vector<std::int32_t> block_of_param;
};

post_expected<GroupedDerivative>
build_grouped_derivative(const spec::LatentStructure& pt,
                         const model::MatrixRep& rep,
                         const data::SampleStats& samp,
                         const Eigen::VectorXd& theta,
                         estimate::frontier::NonIterativeEstimator which,
                         MapKind map_kind,
                         estimate::frontier::CommunalityMethod comm,
                         estimate::frontier::CompositeWeight composite) {
  if (samp.S.empty()) return perr("grouped SE: empty sample stats");
  const std::size_t nblk = samp.S.size();
  if (samp.n_obs.size() != nblk) return perr("grouped SE: n_obs count != block count");

  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("grouped SE: evaluator build failed");
  const auto q = static_cast<Eigen::Index>(ev->n_free());
  if (theta.size() != q) return perr("grouped SE: theta size mismatch");
  const bool has_means = has_mean_params(*ev);
  if (has_means && samp.mean.size() != nblk)
    return perr("grouped SE: mean structure requires per-block sample means");

  GroupedDerivative out;
  out.q = q;
  out.has_means = has_means;
  out.pstar_b.resize(nblk);
  out.offset.resize(nblk);
  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::Index p = samp.S[b].rows();
    out.pstar_b[b] = p * (p + 1) / 2;
    out.offset[b] = out.pstar_total;
    out.pstar_total += out.pstar_b[b];
  }

  out.locs = ev->param_locations();
  out.block_of_param.assign(static_cast<std::size_t>(q), -1);
  for (std::size_t k = 0; k < out.locs.size(); ++k) {
    const auto b = static_cast<std::size_t>(out.locs[k].block);
    if (out.locs[k].block < 0 || b >= nblk)
      return perr("grouped SE: parameter has out-of-range block");
    out.block_of_param[k] = static_cast<std::int32_t>(b);
  }

  out.J_blocks.reserve(nblk);
  for (std::size_t b = 0; b < nblk; ++b) {
    auto Jf = (map_kind == MapKind::Restricted)
                  ? estimate::frontier::estimator_map_jacobian_restricted_block(
                        pt, rep, *ev, samp, which, b, 1e-6, comm, composite)
                  : estimate::frontier::estimator_map_jacobian_block(
                        pt, rep, *ev, samp, which, b, 1e-6, composite);
    if (!Jf.has_value()) return perr("grouped SE: block Jacobian failed");
    if (Jf->rows() != q || Jf->cols() != out.pstar_b[b])
      return perr("grouped SE: block Jacobian dimension mismatch");
    out.J_blocks.push_back(std::move(*Jf));
  }
  return out;
}

post_expected<Eigen::MatrixXd>
block_j_aug(const GroupedDerivative& d, const data::SampleStats& samp,
            std::size_t b, const Eigen::Index maug) {
  const Eigen::Index p = samp.S[b].rows();
  const Eigen::Index ps = d.pstar_b[b];
  Eigen::MatrixXd Jaug = Eigen::MatrixXd::Zero(d.q, maug);
  Jaug.rightCols(ps) = d.J_blocks[b];
  if (!d.has_means) return Jaug;
  for (std::size_t k = 0; k < d.locs.size(); ++k) {
    const auto& loc = d.locs[k];
    if (loc.mat == model::MatId::Nu && static_cast<std::size_t>(loc.block) == b) {
      if (loc.row < 0 || loc.row >= p)
        return perr("grouped SE: intercept row out of range");
      Jaug(static_cast<Eigen::Index>(k), loc.row) = 1.0;
    }
  }
  return Jaug;
}

post_expected<NonIterativeSE>
finish_grouped_se(const Eigen::VectorXd& theta,
                  const data::SampleStats& samp,
                  const GroupedDerivative& d,
                  const std::vector<Eigen::MatrixXd>& gamma_per_block) {
  const std::size_t nblk = samp.S.size();
  if (gamma_per_block.size() != nblk) return perr("grouped SE: Gamma count != block count");
  NonIterativeSE out;
  out.theta_hat = theta;
  out.Omega = Eigen::MatrixXd::Zero(d.q, d.q);
  out.block_of_param = d.block_of_param;
  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::Index p = samp.S[b].rows();
    const Eigen::Index ps = d.pstar_b[b];
    const Eigen::Index pm = d.has_means ? p : 0;
    const Eigen::Index maug = pm + ps;
    if (gamma_per_block[b].rows() != maug || gamma_per_block[b].cols() != maug)
      return perr("grouped SE: Gamma[b] dimension mismatch");
    const double N_b = static_cast<double>(samp.n_obs[b]);
    if (!(N_b > 0.0)) return perr("grouped SE: non-positive n_obs");
    if (d.has_means) {
      auto Jaug = block_j_aug(d, samp, b, maug);
      if (!Jaug.has_value()) return std::unexpected(Jaug.error());
      out.Omega.noalias() +=
          (*Jaug * gamma_per_block[b] * Jaug->transpose()) / N_b;
    } else {
      out.Omega.noalias() +=
          (d.J_blocks[b] * gamma_per_block[b] * d.J_blocks[b].transpose()) / N_b;
    }
  }
  out.Omega = 0.5 * (out.Omega + out.Omega.transpose()).eval();
  out.se = se_from_omega(out.Omega);
  return out;
}

post_expected<NonIterativeSE>
finish_grouped_empirical_se(const Eigen::VectorXd& theta,
                            const data::SampleStats& samp,
                            const data::RawData& raw,
                            const GroupedDerivative& d) {
  const std::size_t nblk = samp.S.size();
  if (raw.X.size() != nblk) return perr("grouped SE: raw-data block count mismatch");
  NonIterativeSE out;
  out.theta_hat = theta;
  out.Omega = Eigen::MatrixXd::Zero(d.q, d.q);
  out.block_of_param = d.block_of_param;
  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::Index p = samp.S[b].rows();
    const Eigen::Index ps = d.pstar_b[b];
    const Eigen::Index pm = d.has_means ? p : 0;
    const Eigen::Index maug = pm + ps;
    const auto& Xb = raw.X[b];
    if (Xb.cols() != p) return perr("grouped SE: raw-data column mismatch");
    const double N_b = static_cast<double>(Xb.rows());
    if (!(N_b > 0.0)) return perr("grouped SE: non-positive raw-data rows");
    const Eigen::VectorXd mean =
        (b < samp.mean.size() && samp.mean[b].size() == p)
            ? samp.mean[b]
            : Xb.colwise().mean().transpose().eval();
    const Eigen::MatrixXd Z =
        casewise_moment_rows(Xb, samp.S[b], mean, d.has_means);
    Eigen::MatrixXd U;
    if (d.has_means) {
      auto Jaug = block_j_aug(d, samp, b, maug);
      if (!Jaug.has_value()) return std::unexpected(Jaug.error());
      U = Z * Jaug->transpose();  // n_b × q
    } else {
      U = Z * d.J_blocks[b].transpose();  // n_b × q
    }
    out.Omega.noalias() += (U.transpose() * U) / (N_b * N_b);
  }
  out.Omega = 0.5 * (out.Omega + out.Omega.transpose()).eval();
  out.se = se_from_omega(out.Omega);
  return out;
}

post_expected<NonIterativeDiffTest>
noniterative_difference_test_impl(const Eigen::MatrixXd& M0,
                                  const Eigen::MatrixXd& M1,
                                  const Eigen::MatrixXd& V1,
                                  const Eigen::MatrixXd& U1,
                                  const Eigen::MatrixXd& Gamma1,
                                  double T0,
                                  double T1,
                                  int df_d) {
  if (M0.rows() != M1.rows() || M0.cols() != M1.cols() ||
      V1.rows() != M1.rows() || V1.cols() != M1.rows() ||
      U1.rows() != M1.cols() || U1.cols() != M1.cols() ||
      Gamma1.rows() != M1.cols() || Gamma1.cols() != M1.cols()) {
    return perr("difference test: inconsistent moment dimensions between fits");
  }
  if (df_d < 1) return perr("difference test: df_d must be >= 1");

  // Anchor V and Gamma at H1; ULS makes this a no-op, NTML needs the H1
  // model-implied weight for both residual projectors.
  const Eigen::MatrixXd U0 = M0.transpose() * V1 * M0;
  auto sd = compute_diff_spectrum_2001(U0, U1, Gamma1, df_d);
  if (!sd.has_value()) return perr("difference test: diff spectrum failed");

  const double T_d = T0 - T1;
  auto lr = lr_test_satorra2000(T_d, *sd);
  if (!lr.has_value()) return perr("difference test: p-values failed");

  NonIterativeDiffTest out;
  out.T_d = T_d;
  out.df_d = df_d;
  out.eigenvalues = sd->eigenvalues;
  out.p_scaled = lr->p_scaled;
  out.p_adjusted = lr->p_adjusted;
  out.p_scaled_shifted = lr->p_scaled_shifted;
  out.p_mixture = lr->p_mixture;
  if (T_d < 0.0)
    out.warnings.push_back("difference statistic T_d < 0 (non-minimizing "
                           "estimator; the pseudo-LRT is not monotone)");
  for (const auto& w : sd->warnings) out.warnings.push_back(w);
  for (const auto& w : lr->warnings) out.warnings.push_back(w);
  return out;
}

}  // namespace

post_expected<NonIterativeInference>
noniterative_inference_impl(const spec::LatentStructure& pt,
                            const model::MatrixRep& rep,
                            const data::SampleStats& samp,
                            const Eigen::VectorXd& theta,
                            estimate::frontier::NonIterativeEstimator which,
                            Discrepancy disc,
                            const Eigen::MatrixXd& gamma,
                            MapKind map_kind,
                            estimate::frontier::CommunalityMethod comm,
                            estimate::frontier::CompositeWeight composite) {
  if (samp.S.empty()) return perr("non-iterative inference: empty sample stats");
  const Eigen::MatrixXd& S = samp.S[0];
  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  if (gamma.rows() != pstar || gamma.cols() != pstar)
    return perr("non-iterative inference: Gamma dimension mismatch");

  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("non-iterative inference: evaluator build failed");
  const auto q = static_cast<Eigen::Index>(ev->n_free());
  if (theta.size() != q) return perr("non-iterative inference: theta size mismatch");
  if (has_mean_params(*ev))
    return perr("non-iterative inference: mean structure requires the grouped "
                "path (noniterative_inference_grouped*); the single-block SEs "
                "omit the intercept block");

  // Σ(θ̂) — copy out before any call that overwrites the evaluator buffers.
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("non-iterative inference: Sigma(theta) failed");
  const Eigen::MatrixXd Sigma_hat = sig->sigma[0];

  auto dS = ev->dsigma_dtheta(theta);  // Δ, p* × q (materialized copy)
  if (!dS.has_value()) return perr("non-iterative inference: dsigma_dtheta failed");
  const Eigen::MatrixXd Delta = *dS;

  auto Jf = (map_kind == MapKind::Restricted)
                ? estimate::frontier::estimator_map_jacobian_restricted(
                      pt, rep, *ev, samp, which, 1e-6, comm, composite)
                : estimate::frontier::estimator_map_jacobian(
                      pt, rep, *ev, samp, which, 1e-6, composite);
  if (!Jf.has_value()) return perr("non-iterative inference: estimator Jacobian failed");
  const Eigen::MatrixXd J = *Jf;  // q × p*

  // Discrepancy weight V (p* × p*).
  Eigen::MatrixXd V;
  double rls_check = std::numeric_limits<double>::quiet_NaN();
  if (disc == Discrepancy::ULS) {
    V = uls_weight(p);
  } else {  // NTML: model-implied ½D'(Σ⁻¹⊗Σ⁻¹)D — NOT the sample GLS weight.
    auto w = estimate::gmm::expected_information_weight(*ev, samp, theta);
    if (!w.has_value() || w->empty())
      return perr("non-iterative inference: NTML weight build failed");
    const Eigen::MatrixXd& W0 = (*w)[0];
    if (W0.rows() == pstar) V = W0;
    else if (W0.rows() > pstar) V = W0.bottomRightCorner(pstar, pstar);
    else return perr("non-iterative inference: NTML weight smaller than p*");
    model::ImpliedMoments im;
    im.sigma = {Sigma_hat};
    if (auto rc = inference::rls_chi2(samp, im); rc.has_value()) rls_check = *rc;
  }

  const double N = total_n(samp);
  const Eigen::VectorXd r = vech_lower(S) - vech_lower(Sigma_hat);
  const Eigen::MatrixXd M =
      Eigen::MatrixXd::Identity(pstar, pstar) - Delta * J;   // I − ΔJ
  const Eigen::MatrixXd U = M.transpose() * V * M;           // M'VM
  const double T_gof = N * r.dot(V * r);
  const int df = (map_kind == MapKind::Restricted)
                     ? static_cast<int>(pstar - numeric_rank(Delta * J))
                     : static_cast<int>(pstar - q);

  NonIterativeInference out;
  out.warnings = {};

  // Reference spectrum: positive eigenvalues of U·Γ, trimmed to df.
  auto sd = compute_profile_contrast_spectrum(U, gamma, 1e-10);
  if (!sd.has_value()) return perr("non-iterative inference: GOF spectrum failed");
  SatorraDiffResult sd_df = *sd;
  const Eigen::Index navail = sd_df.eigenvalues.size();
  if (navail > df) {
    // .eval() breaks the self-assignment aliasing (the resize would free the
    // storage the .tail() block still points at).
    sd_df.eigenvalues = sd_df.eigenvalues.tail(df).eval();  // largest df (ascending)
  } else if (navail < df) {
    out.warnings.push_back("GOF spectrum rank " + std::to_string(navail) +
                           " < df " + std::to_string(df) +
                           " (ill-conditioned residual projector)");
  }
  sd_df.trace_CinvS = sd_df.eigenvalues.sum();
  sd_df.trace_CinvS_sq = sd_df.eigenvalues.squaredNorm();
  for (const auto& w : sd_df.warnings) out.warnings.push_back(w);

  auto lr = lr_test_satorra2000(T_gof, sd_df);
  if (!lr.has_value()) return perr("non-iterative inference: GOF p-values failed");

  // Delta-method SEs: Ω = J Γ Jᵀ / N.
  const Eigen::MatrixXd Omega = (J * gamma * J.transpose()) / N;
  Eigen::VectorXd se(q);
  for (Eigen::Index i = 0; i < q; ++i)
    se(i) = std::sqrt(std::max(Omega(i, i), 0.0));

  out.Omega = Omega;
  out.se = std::move(se);
  out.T_gof = T_gof;
  out.df = df;
  out.gof_eigenvalues = sd_df.eigenvalues;
  out.scale_c = lr->scale_c;
  out.p_scaled = lr->p_scaled;
  out.p_meanvar = lr->p_adjusted;
  out.p_scaled_shifted = lr->p_scaled_shifted;
  out.p_mixture = lr->p_mixture;
  out.rls_check = rls_check;
  out.J = J;
  out.Delta = Delta;
  out.M = M;
  out.V = std::move(V);
  out.Gamma = gamma;
  out.U = U;
  for (const auto& w : lr->warnings) out.warnings.push_back(w);
  return out;
}

post_expected<NonIterativeModificationIndexTable>
noniterative_modification_indices(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    const NonIterativeInference& inf,
    const inference::ModificationIndexOptions& options) {
  if (samp.S.size() != 1) {
    return perr("non-iterative modification indices: grouped fits are not yet "
                "wired; use a single-group fit");
  }
  const Eigen::MatrixXd& S = samp.S[0];
  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  if (inf.M.rows() != pstar || inf.M.cols() != pstar ||
      inf.V.rows() != pstar || inf.V.cols() != pstar ||
      inf.Gamma.rows() != pstar || inf.Gamma.cols() != pstar ||
      inf.Delta.rows() != pstar || inf.Delta.cols() != theta.size()) {
    return perr("non-iterative modification indices: incompatible inference "
                "dimensions");
  }

  auto ev0 = model::ModelEvaluator::build(pt, rep);
  if (!ev0.has_value()) {
    return std::unexpected(model_to_post(ev0.error()));
  }
  if (theta.size() != static_cast<Eigen::Index>(ev0->n_free())) {
    return perr("non-iterative modification indices: theta size mismatch");
  }
  if (has_mean_params(*ev0)) {
    return perr("non-iterative modification indices: mean-structure/grouped "
                "candidate diagnostics are deferred");
  }
  auto sig = ev0->sigma(theta);
  if (!sig.has_value()) {
    return perr("non-iterative modification indices: Sigma(theta) failed");
  }
  const Eigen::VectorXd r = vech_lower(S) - vech_lower(sig->sigma[0]);
  const double N = total_n(samp);
  const Eigen::MatrixXd Omega_r =
      (inf.M * inf.Gamma * inf.M.transpose()) / N;

  auto Aplus = symmetric_pinv_psd(
      inf.Delta.transpose() * inf.V * inf.Delta,
      "non-iterative modification-index tangent metric");
  if (!Aplus.has_value()) return std::unexpected(Aplus.error());
  const Eigen::MatrixXd tangent_coef =
      (*Aplus) * inf.Delta.transpose() * inf.V;

  spec::LatentStructure pt_candidates = pt;
  append_absent_mi_rows(pt_candidates, options);
  auto rep_candidates = model::build_matrix_rep(pt_candidates);
  if (!rep_candidates.has_value()) {
    return std::unexpected(model_to_post(rep_candidates.error()));
  }
  const std::vector<MiCandidateRow> candidates =
      collect_mi_candidates(pt_candidates, *rep_candidates);

  NonIterativeModificationIndexTable out;
  out.warnings = inf.warnings;
  if (candidates.empty()) return out;

  spec::LatentStructure pt_free = pt_candidates;
  free_mi_candidate_rows(pt_free, candidates);
  auto rep_free = model::build_matrix_rep(pt_free);
  if (!rep_free.has_value()) return std::unexpected(model_to_post(rep_free.error()));

  const Eigen::VectorXd theta_aug = append_mi_candidate_thetas(theta, candidates);
  auto ev_aug = model::ModelEvaluator::build(pt_free, *rep_free);
  if (!ev_aug.has_value()) return std::unexpected(model_to_post(ev_aug.error()));
  auto Delta_aug = ev_aug->dsigma_dtheta(theta_aug);
  if (!Delta_aug.has_value()) {
    return perr("non-iterative modification indices: augmented dsigma_dtheta "
                "failed");
  }
  if (Delta_aug->rows() != pstar ||
      Delta_aug->cols() < theta.size() + static_cast<Eigen::Index>(candidates.size())) {
    return perr("non-iterative modification indices: augmented Jacobian "
                "dimension mismatch");
  }

  out.rows.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const Eigen::Index col = theta.size() + static_cast<Eigen::Index>(i);
    const Eigen::VectorXd g = Delta_aug->col(col);
    const Eigen::VectorXd g_resid = g - inf.Delta * (tangent_coef * g);

    const MiVariantStats raw = mi_variant(g, r, inf.V, Omega_r, N);
    const MiVariantStats resid = mi_variant(g_resid, r, inf.V, Omega_r, N);

    NonIterativeModificationIndexResult row;
    row.candidate = candidates[i].candidate;
    row.score_raw = raw.score;
    row.var_raw = raw.var;
    row.z_raw = raw.z;
    row.mi_raw = raw.mi;
    row.p_raw = raw.p;
    row.epc_raw = raw.epc;
    row.drop_raw = raw.drop;
    row.p_drop_raw = raw.p_drop;
    row.score_resid = resid.score;
    row.var_resid = resid.var;
    row.z_resid = resid.z;
    row.mi_resid = resid.mi;
    row.p_resid = resid.p;
    row.epc_resid = resid.epc;
    row.drop_resid = resid.drop;
    row.p_drop_resid = resid.p_drop;
    row.signature_norm = std::sqrt(std::max(0.0, g.dot(inf.V * g)));
    row.residualized_norm =
        std::sqrt(std::max(0.0, g_resid.dot(inf.V * g_resid)));
    out.rows.push_back(std::move(row));
  }
  return out;
}

post_expected<NonIterativeInference>
noniterative_inference(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                       const data::SampleStats& samp, const Eigen::VectorXd& theta,
                       estimate::frontier::NonIterativeEstimator which,
                       Discrepancy disc, const Eigen::MatrixXd& gamma,
                       estimate::frontier::CompositeWeight composite) {
  return noniterative_inference_impl(
      pt, rep, samp, theta, which, disc, gamma, MapKind::Configural,
      estimate::frontier::CommunalityMethod::TriadWls, composite);
}

post_expected<NonIterativeInference>
noniterative_inference_restricted(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    const Eigen::MatrixXd& gamma,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  return noniterative_inference_impl(
      pt, rep, samp, theta, which, disc, gamma, MapKind::Restricted, comm,
      composite);
}

post_expected<NonIterativeInference>
noniterative_inference_nt(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                          const data::SampleStats& samp, const Eigen::VectorXd& theta,
                          estimate::frontier::NonIterativeEstimator which,
                          Discrepancy disc,
                          estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("non-iterative inference: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("non-iterative inference: Sigma(theta) failed");
  auto g = data::gamma_nt(sig->sigma[0]);
  if (!g.has_value()) return perr("non-iterative inference: gamma_nt failed");
  return noniterative_inference(pt, rep, samp, theta, which, disc, *g, composite);
}

post_expected<NonIterativeInference>
noniterative_inference_restricted_nt(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("non-iterative inference: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("non-iterative inference: Sigma(theta) failed");
  auto g = data::gamma_nt(sig->sigma[0]);
  if (!g.has_value()) return perr("non-iterative inference: gamma_nt failed");
  return noniterative_inference_restricted(
      pt, rep, samp, theta, which, disc, *g, comm, composite);
}

post_expected<NonIterativeInference>
noniterative_inference_empirical(const spec::LatentStructure& pt,
                                 const model::MatrixRep& rep, const data::SampleStats& samp,
                                 const data::RawData& raw, const Eigen::VectorXd& theta,
                                 estimate::frontier::NonIterativeEstimator which,
                                 Discrepancy disc,
                                 estimate::frontier::CompositeWeight composite) {
  if (raw.X.empty()) return perr("non-iterative inference: empty raw data");
  auto g = data::empirical_gamma(raw.X[0]);
  if (!g.has_value()) return perr("non-iterative inference: empirical_gamma failed");
  return noniterative_inference(pt, rep, samp, theta, which, disc, *g, composite);
}

post_expected<NonIterativeSE>
noniterative_se(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                const data::SampleStats& samp, const Eigen::VectorXd& theta,
                estimate::frontier::NonIterativeEstimator which,
                const Eigen::MatrixXd& gamma,
                estimate::frontier::CompositeWeight composite) {
  auto d = build_single_derivative(
      pt, rep, samp, theta, which, MapKind::Configural,
      estimate::frontier::CommunalityMethod::TriadWls, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  if (gamma.rows() != d->pstar || gamma.cols() != d->pstar)
    return perr("non-iterative SE: Gamma dimension mismatch");
  const Eigen::MatrixXd Omega = (d->J * gamma * d->J.transpose()) / d->N;
  return finish_single_se(theta, *d, Omega);
}

post_expected<NonIterativeSE>
noniterative_se_nt(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                   const data::SampleStats& samp, const Eigen::VectorXd& theta,
                   estimate::frontier::NonIterativeEstimator which,
                   estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("non-iterative SE: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("non-iterative SE: Sigma(theta) failed");
  auto g = data::gamma_nt(sig->sigma[0]);
  if (!g.has_value()) return perr("non-iterative SE: gamma_nt failed");
  return noniterative_se(pt, rep, samp, theta, which, *g, composite);
}

post_expected<NonIterativeSE>
noniterative_se_empirical(const spec::LatentStructure& pt,
                          const model::MatrixRep& rep,
                          const data::SampleStats& samp,
                          const data::RawData& raw,
                          const Eigen::VectorXd& theta,
                          estimate::frontier::NonIterativeEstimator which,
                          estimate::frontier::CompositeWeight composite) {
  if (raw.X.empty()) return perr("non-iterative SE: empty raw data");
  auto d = build_single_derivative(
      pt, rep, samp, theta, which, MapKind::Configural,
      estimate::frontier::CommunalityMethod::TriadWls, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  const Eigen::MatrixXd& X = raw.X[0];
  if (X.cols() != samp.S[0].rows()) return perr("non-iterative SE: raw-data column mismatch");
  const double N = static_cast<double>(X.rows());
  if (!(N > 0.0)) return perr("non-iterative SE: non-positive raw-data rows");
  const Eigen::VectorXd mean =
      (!samp.mean.empty() && samp.mean[0].size() == X.cols())
          ? samp.mean[0]
          : X.colwise().mean().transpose().eval();
  const Eigen::MatrixXd Z = casewise_moment_rows(X, samp.S[0], mean,
                                                 /*include_means=*/false);
  const Eigen::MatrixXd U = Z * d->J.transpose();
  return finish_single_se(theta, *d, (U.transpose() * U) / (N * N));
}

post_expected<NonIterativeInference>
noniterative_inference_restricted_empirical(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const data::RawData& raw,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  if (raw.X.empty()) return perr("non-iterative inference: empty raw data");
  auto g = data::empirical_gamma(raw.X[0]);
  if (!g.has_value()) return perr("non-iterative inference: empirical_gamma failed");
  return noniterative_inference_restricted(
      pt, rep, samp, theta, which, disc, *g, comm, composite);
}

post_expected<NonIterativeSE>
noniterative_se_restricted(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    const Eigen::MatrixXd& gamma,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  auto d = build_single_derivative(
      pt, rep, samp, theta, which, MapKind::Restricted, comm, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  if (gamma.rows() != d->pstar || gamma.cols() != d->pstar)
    return perr("restricted non-iterative SE: Gamma dimension mismatch");
  const Eigen::MatrixXd Omega = (d->J * gamma * d->J.transpose()) / d->N;
  return finish_single_se(theta, *d, Omega);
}

post_expected<NonIterativeSE>
noniterative_se_restricted_nt(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("restricted non-iterative SE: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("restricted non-iterative SE: Sigma(theta) failed");
  auto g = data::gamma_nt(sig->sigma[0]);
  if (!g.has_value()) return perr("restricted non-iterative SE: gamma_nt failed");
  return noniterative_se_restricted(pt, rep, samp, theta, which, *g, comm, composite);
}

post_expected<NonIterativeSE>
noniterative_se_restricted_empirical(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const data::RawData& raw,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  if (raw.X.empty()) return perr("restricted non-iterative SE: empty raw data");
  auto d = build_single_derivative(
      pt, rep, samp, theta, which, MapKind::Restricted, comm, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  const Eigen::MatrixXd& X = raw.X[0];
  if (X.cols() != samp.S[0].rows())
    return perr("restricted non-iterative SE: raw-data column mismatch");
  const double N = static_cast<double>(X.rows());
  if (!(N > 0.0)) return perr("restricted non-iterative SE: non-positive raw-data rows");
  const Eigen::VectorXd mean =
      (!samp.mean.empty() && samp.mean[0].size() == X.cols())
          ? samp.mean[0]
          : X.colwise().mean().transpose().eval();
  const Eigen::MatrixXd Z = casewise_moment_rows(X, samp.S[0], mean,
                                                 /*include_means=*/false);
  const Eigen::MatrixXd U = Z * d->J.transpose();
  return finish_single_se(theta, *d, (U.transpose() * U) / (N * N));
}

post_expected<inference::WaldTestResult>
noniterative_wald(const Eigen::VectorXd& theta, const NonIterativeInference& inf,
                  const Eigen::MatrixXd& R, const Eigen::VectorXd& q) {
  estimate::Estimates est;
  est.theta = theta;
  return inference::wald_test(R, q, est, inf.Omega);
}

post_expected<inference::WaldTestResult>
noniterative_wald(const Eigen::VectorXd& theta, const NonIterativeSE& se,
                  const Eigen::MatrixXd& R, const Eigen::VectorXd& q) {
  estimate::Estimates est;
  est.theta = theta;
  return inference::wald_test(R, q, est, se.Omega);
}

post_expected<NonIterativeDiffTest>
noniterative_difference_test(const NonIterativeInference& inf0,
                             const NonIterativeInference& inf1, int df_d) {
  return noniterative_difference_test_impl(
      inf0.M, inf1.M, inf1.V, inf1.U, inf1.Gamma,
      inf0.T_gof, inf1.T_gof, df_d);
}

// ---------------------------------------------------------------------------
// Grouped inference + linearly-constrained fit.
// ---------------------------------------------------------------------------

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                               const data::SampleStats& samp, const Eigen::VectorXd& theta,
                               estimate::frontier::NonIterativeEstimator which,
                               Discrepancy disc,
                               const std::vector<Eigen::MatrixXd>& gamma_per_block,
                               estimate::frontier::CompositeWeight composite) {
  if (samp.S.empty()) return perr("grouped inference: empty sample stats");
  const std::size_t nblk = samp.S.size();
  if (gamma_per_block.size() != nblk)
    return perr("grouped inference: Gamma count != block count");
  if (samp.n_obs.size() != nblk)
    return perr("grouped inference: n_obs count != block count");

  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("grouped inference: evaluator build failed");
  const auto q = static_cast<Eigen::Index>(ev->n_free());
  if (theta.size() != q) return perr("grouped inference: theta size mismatch");

  // Mean structure: the moment vector per block is [m_b ; vech(S_b)] (mean-first)
  // and `gamma_per_block[b]` is the (p_b + p*_b)-square mean-augmented NACOV. The
  // estimator Jacobian gains an analytic mean block (∂ν/∂m = I) so Ω picks up the
  // intercept variances; the GOF stays covariance-only (the saturated mean part
  // contributes exactly zero, since ν_g = m_g).
  const bool has_means = has_mean_params(*ev);
  if (has_means && samp.mean.size() != nblk)
    return perr("grouped inference: mean structure requires per-block sample "
                "means");

  // Σ(θ̂) and the stacked Δ — copy out before any buffer-overwriting call.
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("grouped inference: Sigma(theta) failed");
  auto dS = ev->dsigma_dtheta(theta);
  if (!dS.has_value()) return perr("grouped inference: dsigma_dtheta failed");
  const Eigen::MatrixXd Delta_full = *dS;  // (Σ p*_b) × q, covariance-only

  // NTML weights (model-implied, per block), computed once.
  std::vector<Eigen::MatrixXd> Wntml;
  if (disc == Discrepancy::NTML) {
    auto w = estimate::gmm::expected_information_weight(*ev, samp, theta);
    if (!w.has_value() || w->size() != nblk)
      return perr("grouped inference: NTML weight build failed");
    Wntml = std::move(*w);
  }

  // Per-block moment offsets in the stacked vech vector.
  std::vector<Eigen::Index> pstar_b(nblk), offset(nblk);
  Eigen::Index pstar_total = 0;
  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::Index p_b = samp.S[b].rows();
    pstar_b[b] = p_b * (p_b + 1) / 2;
    offset[b] = pstar_total;
    pstar_total += pstar_b[b];
  }
  if (Delta_full.rows() != pstar_total || Delta_full.cols() != q)
    return perr("grouped inference: Delta dimension mismatch");

  // Per-block free-parameter counts (from the param layout). `q_cov_of_block`
  // excludes intercept / latent-mean params: the mean moments and the ν params
  // cancel one-for-one in the GOF df (saturated ν_g = m_g), so the covariance-only
  // GOF df is Σ_b (p*_b − q_cov_of_block[b]).
  const std::vector<model::ParamLocation> locs = ev->param_locations();
  GroupedNonIterativeInference out;
  out.block_of_param.assign(static_cast<std::size_t>(q), -1);
  std::vector<int> q_of_block(nblk, 0), q_cov_of_block(nblk, 0);
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto b = static_cast<std::size_t>(locs[k].block);
    if (locs[k].block < 0 || b >= nblk)
      return perr("grouped inference: parameter has out-of-range block");
    out.block_of_param[k] = static_cast<std::int32_t>(b);
    q_of_block[b]++;
    if (locs[k].mat != model::MatId::Nu && locs[k].mat != model::MatId::Alpha)
      q_cov_of_block[b]++;
  }

  out.Omega = Eigen::MatrixXd::Zero(q, q);
  Eigen::MatrixXd M_big = Eigen::MatrixXd::Zero(pstar_total, pstar_total);
  Eigen::MatrixXd V_big = Eigen::MatrixXd::Zero(pstar_total, pstar_total);
  Eigen::MatrixXd U_big = Eigen::MatrixXd::Zero(pstar_total, pstar_total);
  Eigen::MatrixXd G_big = Eigen::MatrixXd::Zero(pstar_total, pstar_total);
  double T = 0.0;
  int df = 0;
  double rls = (disc == Discrepancy::NTML)
                   ? 0.0 : std::numeric_limits<double>::quiet_NaN();

  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::Index p_b = samp.S[b].rows();
    const Eigen::Index ps = pstar_b[b];
    // Augmented moment count for this block: mean-first [m_b ; vech(S_b)] when
    // the model has mean structure, else vech(S_b) alone.
    const Eigen::Index pm = has_means ? p_b : 0;
    const Eigen::Index maug = pm + ps;
    if (gamma_per_block[b].rows() != maug || gamma_per_block[b].cols() != maug)
      return perr("grouped inference: Gamma[b] dimension mismatch");

    const Eigen::MatrixXd Sigma_hat_b = sig->sigma[b];              // copy (P4)
    const Eigen::MatrixXd Delta_b = Delta_full.middleRows(offset[b], ps);  // ps × q

    auto Jf = estimate::frontier::estimator_map_jacobian_block(
        pt, rep, *ev, samp, which, b, 1e-6, composite);
    if (!Jf.has_value()) return perr("grouped inference: block Jacobian failed");
    const Eigen::MatrixXd J_b = *Jf;  // q × ps (rows outside block b are exactly 0)

    Eigen::MatrixXd V_b;
    if (disc == Discrepancy::ULS) {
      V_b = uls_weight(p_b);
    } else {
      const Eigen::MatrixXd& W0 = Wntml[b];
      if (W0.rows() == ps) V_b = W0;
      else if (W0.rows() > ps) V_b = W0.bottomRightCorner(ps, ps);
      else return perr("grouped inference: NTML weight smaller than p*");
    }

    // Covariance-only GOF: r_b, M_b = I − Δ_b J_b, U_b = M_b'V_b M_b, and the
    // vech-block of Γ. Δ_b and J_b already carry zero ν rows/cols, so this is
    // identical whether or not the model has means (P5).
    const Eigen::MatrixXd Gamma_cov_b =
        has_means ? gamma_per_block[b].bottomRightCorner(ps, ps).eval()
                  : gamma_per_block[b];
    const Eigen::VectorXd r_b = vech_lower(samp.S[b]) - vech_lower(Sigma_hat_b);
    const Eigen::MatrixXd M_b = Eigen::MatrixXd::Identity(ps, ps) - Delta_b * J_b;
    const Eigen::MatrixXd U_b = M_b.transpose() * V_b * M_b;
    const double N_b = static_cast<double>(samp.n_obs[b]);  // per-block N (P2)

    T += N_b * r_b.dot(V_b * r_b);
    // Ω = J_aug Γ_aug J_augᵀ / N. Without means J_aug = J_b (q × p*). With means
    // the analytic mean columns select ∂ν_i/∂m_i = 1; J_b (cov FD) has zero ν
    // rows, so the two column groups don't overlap. J_aug has zero rows outside
    // block b either way, so Ω accumulates block-diagonal across groups.
    if (has_means) {
      Eigen::MatrixXd Jaug = Eigen::MatrixXd::Zero(q, maug);
      Jaug.rightCols(ps) = J_b;                                 // vech columns
      for (std::size_t k = 0; k < locs.size(); ++k) {
        const auto& loc = locs[k];
        if (loc.mat == model::MatId::Nu &&
            static_cast<std::size_t>(loc.block) == b) {
          if (loc.row < 0 || loc.row >= p_b)
            return perr("grouped inference: intercept row out of range");
          Jaug(static_cast<Eigen::Index>(k), loc.row) = 1.0;    // ∂ν_i/∂m_i = 1
        }
      }
      out.Omega.noalias() += (Jaug * gamma_per_block[b] * Jaug.transpose()) / N_b;
    } else {
      out.Omega.noalias() += (J_b * gamma_per_block[b] * J_b.transpose()) / N_b;
    }
    M_big.block(offset[b], offset[b], ps, ps) = M_b;
    V_big.block(offset[b], offset[b], ps, ps) = V_b;
    U_big.block(offset[b], offset[b], ps, ps) = U_b;
    G_big.block(offset[b], offset[b], ps, ps) = Gamma_cov_b;
    df += static_cast<int>(ps) - q_cov_of_block[b];

    if (disc == Discrepancy::NTML) {
      model::ImpliedMoments im;
      im.sigma = {Sigma_hat_b};
      data::SampleStats one;
      one.S = {samp.S[b]};
      one.n_obs = {samp.n_obs[b]};
      // Covariance-only cross-check: ν_g = m_g saturates the mean, so drop the
      // mean from the RLS χ² comparator (its ImpliedMoments carries no μ).
      if (!has_means && !samp.mean.empty()) one.mean = {samp.mean[b]};
      if (auto rc = inference::rls_chi2(one, im); rc.has_value()) rls += *rc;
    }
  }

  out.theta_hat = theta;
  out.se = Eigen::VectorXd(q);
  for (Eigen::Index i = 0; i < q; ++i)
    out.se(i) = std::sqrt(std::max(out.Omega(i, i), 0.0));
  out.T_gof = T;
  out.df = df;
  out.rls_check = rls;

  if (df >= 1) {
    auto sd = compute_profile_contrast_spectrum(U_big, G_big, 1e-10);
    if (!sd.has_value()) return perr("grouped inference: GOF spectrum failed");
    SatorraDiffResult sd_df = *sd;
    const Eigen::Index navail = sd_df.eigenvalues.size();
    if (navail > df) {
      sd_df.eigenvalues = sd_df.eigenvalues.tail(df).eval();  // largest df (ascending)
    } else if (navail < df) {
      out.warnings.push_back("GOF spectrum rank " + std::to_string(navail) +
                             " < df " + std::to_string(df) +
                             " (ill-conditioned residual projector)");
    }
    sd_df.trace_CinvS = sd_df.eigenvalues.sum();
    sd_df.trace_CinvS_sq = sd_df.eigenvalues.squaredNorm();
    for (const auto& w : sd_df.warnings) out.warnings.push_back(w);

    auto lr = lr_test_satorra2000(T, sd_df);
    if (!lr.has_value()) return perr("grouped inference: GOF p-values failed");
    out.gof_eigenvalues = sd_df.eigenvalues;
    out.scale_c = lr->scale_c;
    out.p_scaled = lr->p_scaled;
    out.p_meanvar = lr->p_adjusted;
    out.p_scaled_shifted = lr->p_scaled_shifted;
    out.p_mixture = lr->p_mixture;
    for (const auto& w : lr->warnings) out.warnings.push_back(w);
  } else {
    out.warnings.push_back("joint df <= 0 (saturated blocks); GOF not reported");
  }

  out.M = std::move(M_big);
  out.V = std::move(V_big);
  out.U = std::move(U_big);
  out.Gamma = std::move(G_big);
  return out;
}

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_nt(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                                  const data::SampleStats& samp, const Eigen::VectorXd& theta,
                                  estimate::frontier::NonIterativeEstimator which,
                                  Discrepancy disc,
                                  estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("grouped inference: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("grouped inference: Sigma(theta) failed");
  const bool has_means = has_mean_params(*ev);
  std::vector<Eigen::MatrixXd> gammas;
  gammas.reserve(sig->sigma.size());
  for (const auto& Sig : sig->sigma) {
    auto g = has_means ? data::gamma_nt_with_means(Sig) : data::gamma_nt(Sig);
    if (!g.has_value()) return perr("grouped inference: gamma_nt failed");
    gammas.push_back(std::move(*g));
  }
  return noniterative_inference_grouped(
      pt, rep, samp, theta, which, disc, gammas, composite);
}

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_empirical(const spec::LatentStructure& pt,
                                         const model::MatrixRep& rep,
                                         const data::SampleStats& samp,
                                         const data::RawData& raw, const Eigen::VectorXd& theta,
                                         estimate::frontier::NonIterativeEstimator which,
                                         Discrepancy disc,
                                         estimate::frontier::CompositeWeight composite) {
  if (raw.X.size() != samp.S.size())
    return perr("grouped inference: raw-data block count != sample-stats blocks");
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("grouped inference: evaluator build failed");
  const bool has_means = has_mean_params(*ev);
  std::vector<Eigen::MatrixXd> gammas;
  gammas.reserve(raw.X.size());
  for (const auto& Xb : raw.X) {
    auto g = has_means ? data::empirical_gamma_with_means(Xb)
                       : data::empirical_gamma(Xb);
    if (!g.has_value()) return perr("grouped inference: empirical_gamma failed");
    gammas.push_back(std::move(*g));
  }
  return noniterative_inference_grouped(
      pt, rep, samp, theta, which, disc, gammas, composite);
}

post_expected<NonIterativeSE>
noniterative_se_grouped(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                        const data::SampleStats& samp, const Eigen::VectorXd& theta,
                        estimate::frontier::NonIterativeEstimator which,
                        const std::vector<Eigen::MatrixXd>& gamma_per_block,
                        estimate::frontier::CompositeWeight composite) {
  auto d = build_grouped_derivative(
      pt, rep, samp, theta, which, MapKind::Configural,
      estimate::frontier::CommunalityMethod::TriadWls, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  return finish_grouped_se(theta, samp, *d, gamma_per_block);
}

post_expected<NonIterativeSE>
noniterative_se_grouped_nt(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                           const data::SampleStats& samp, const Eigen::VectorXd& theta,
                           estimate::frontier::NonIterativeEstimator which,
                           estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("grouped SE: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("grouped SE: Sigma(theta) failed");
  const bool has_means = has_mean_params(*ev);
  std::vector<Eigen::MatrixXd> gammas;
  gammas.reserve(sig->sigma.size());
  for (const auto& Sig : sig->sigma) {
    auto g = has_means ? data::gamma_nt_with_means(Sig) : data::gamma_nt(Sig);
    if (!g.has_value()) return perr("grouped SE: gamma_nt failed");
    gammas.push_back(std::move(*g));
  }
  return noniterative_se_grouped(pt, rep, samp, theta, which, gammas, composite);
}

post_expected<NonIterativeSE>
noniterative_se_grouped_empirical(const spec::LatentStructure& pt,
                                  const model::MatrixRep& rep,
                                  const data::SampleStats& samp,
                                  const data::RawData& raw,
                                  const Eigen::VectorXd& theta,
                                  estimate::frontier::NonIterativeEstimator which,
                                  estimate::frontier::CompositeWeight composite) {
  auto d = build_grouped_derivative(
      pt, rep, samp, theta, which, MapKind::Configural,
      estimate::frontier::CommunalityMethod::TriadWls, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  return finish_grouped_empirical_se(theta, samp, raw, *d);
}

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_restricted(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    const std::vector<Eigen::MatrixXd>& gamma_per_block,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  if (samp.S.empty()) return perr("restricted grouped inference: empty sample stats");
  const std::size_t nblk = samp.S.size();
  if (gamma_per_block.size() != nblk)
    return perr("restricted grouped inference: Gamma count != block count");
  if (samp.n_obs.size() != nblk)
    return perr("restricted grouped inference: n_obs count != block count");

  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("restricted grouped inference: evaluator build failed");
  const auto q = static_cast<Eigen::Index>(ev->n_free());
  if (theta.size() != q)
    return perr("restricted grouped inference: theta size mismatch");

  const bool has_means = has_mean_params(*ev);
  if (has_means && samp.mean.size() != nblk)
    return perr("restricted grouped inference: mean structure requires per-block "
                "sample means");

  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("restricted grouped inference: Sigma(theta) failed");
  auto dS = ev->dsigma_dtheta(theta);
  if (!dS.has_value()) return perr("restricted grouped inference: dsigma_dtheta failed");
  const Eigen::MatrixXd Delta_full = *dS;

  std::vector<Eigen::MatrixXd> Wntml;
  if (disc == Discrepancy::NTML) {
    auto w = estimate::gmm::expected_information_weight(*ev, samp, theta);
    if (!w.has_value() || w->size() != nblk)
      return perr("restricted grouped inference: NTML weight build failed");
    Wntml = std::move(*w);
  }

  std::vector<Eigen::Index> pstar_b(nblk), offset(nblk);
  Eigen::Index pstar_total = 0;
  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::Index p_b = samp.S[b].rows();
    pstar_b[b] = p_b * (p_b + 1) / 2;
    offset[b] = pstar_total;
    pstar_total += pstar_b[b];
  }
  if (Delta_full.rows() != pstar_total || Delta_full.cols() != q)
    return perr("restricted grouped inference: Delta dimension mismatch");

  const std::vector<model::ParamLocation> locs = ev->param_locations();
  GroupedNonIterativeInference out;
  out.block_of_param.assign(static_cast<std::size_t>(q), -1);
  for (std::size_t k = 0; k < locs.size(); ++k) {
    const auto b = static_cast<std::size_t>(locs[k].block);
    if (locs[k].block < 0 || b >= nblk)
      return perr("restricted grouped inference: parameter has out-of-range block");
    out.block_of_param[k] = static_cast<std::int32_t>(b);
  }

  std::vector<Eigen::MatrixXd> J_blocks;
  J_blocks.reserve(nblk);
  for (std::size_t b = 0; b < nblk; ++b) {
    auto Jf = estimate::frontier::estimator_map_jacobian_restricted_block(
        pt, rep, *ev, samp, which, b, 1e-6, comm, composite);
    if (!Jf.has_value())
      return perr("restricted grouped inference: block Jacobian failed");
    if (Jf->rows() != q || Jf->cols() != pstar_b[b])
      return perr("restricted grouped inference: block Jacobian dimension mismatch");
    J_blocks.push_back(std::move(*Jf));
  }

  out.Omega = Eigen::MatrixXd::Zero(q, q);
  Eigen::MatrixXd V_big = Eigen::MatrixXd::Zero(pstar_total, pstar_total);
  Eigen::MatrixXd G_big = Eigen::MatrixXd::Zero(pstar_total, pstar_total);
  Eigen::MatrixXd DJ = Eigen::MatrixXd::Zero(pstar_total, pstar_total);
  double T = 0.0;
  double rls = (disc == Discrepancy::NTML)
                   ? 0.0 : std::numeric_limits<double>::quiet_NaN();

  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::Index p_b = samp.S[b].rows();
    const Eigen::Index ps = pstar_b[b];
    const Eigen::Index pm = has_means ? p_b : 0;
    const Eigen::Index maug = pm + ps;
    if (gamma_per_block[b].rows() != maug || gamma_per_block[b].cols() != maug)
      return perr("restricted grouped inference: Gamma[b] dimension mismatch");

    const Eigen::MatrixXd Sigma_hat_b = sig->sigma[b];
    Eigen::MatrixXd V_b;
    if (disc == Discrepancy::ULS) {
      V_b = uls_weight(p_b);
    } else {
      const Eigen::MatrixXd& W0 = Wntml[b];
      if (W0.rows() == ps) V_b = W0;
      else if (W0.rows() > ps) V_b = W0.bottomRightCorner(ps, ps);
      else return perr("restricted grouped inference: NTML weight smaller than p*");
    }

    const Eigen::MatrixXd Gamma_cov_b =
        has_means ? gamma_per_block[b].bottomRightCorner(ps, ps).eval()
                  : gamma_per_block[b];
    const Eigen::VectorXd r_b =
        vech_lower(samp.S[b]) - vech_lower(Sigma_hat_b);
    const double N_b = static_cast<double>(samp.n_obs[b]);
    if (!(N_b > 0.0))
      return perr("restricted grouped inference: non-positive n_obs");
    T += N_b * r_b.dot(V_b * r_b);

    if (has_means) {
      Eigen::MatrixXd Jaug = Eigen::MatrixXd::Zero(q, maug);
      Jaug.rightCols(ps) = J_blocks[b];
      for (std::size_t k = 0; k < locs.size(); ++k) {
        const auto& loc = locs[k];
        if (loc.mat == model::MatId::Nu &&
            static_cast<std::size_t>(loc.block) == b) {
          if (loc.row < 0 || loc.row >= p_b)
            return perr("restricted grouped inference: intercept row out of range");
          Jaug(static_cast<Eigen::Index>(k), loc.row) = 1.0;
        }
      }
      out.Omega.noalias() += (Jaug * gamma_per_block[b] * Jaug.transpose()) / N_b;
    } else {
      out.Omega.noalias() +=
          (J_blocks[b] * gamma_per_block[b] * J_blocks[b].transpose()) / N_b;
    }

    V_big.block(offset[b], offset[b], ps, ps) = V_b;
    G_big.block(offset[b], offset[b], ps, ps) = Gamma_cov_b;

    if (disc == Discrepancy::NTML) {
      model::ImpliedMoments im;
      im.sigma = {Sigma_hat_b};
      data::SampleStats one;
      one.S = {samp.S[b]};
      one.n_obs = {samp.n_obs[b]};
      if (!has_means && !samp.mean.empty()) one.mean = {samp.mean[b]};
      if (auto rc = inference::rls_chi2(one, im); rc.has_value()) rls += *rc;
    }
  }

  for (std::size_t b = 0; b < nblk; ++b) {
    const Eigen::MatrixXd Delta_b =
        Delta_full.middleRows(offset[b], pstar_b[b]);
    const double N_b = static_cast<double>(samp.n_obs[b]);
    for (std::size_t c = 0; c < nblk; ++c) {
      const double N_c = static_cast<double>(samp.n_obs[c]);
      const double scale = std::sqrt(N_b / N_c);
      DJ.block(offset[b], offset[c], pstar_b[b], pstar_b[c]).noalias() =
          scale * Delta_b * J_blocks[c];
    }
  }

  const Eigen::MatrixXd M =
      Eigen::MatrixXd::Identity(pstar_total, pstar_total) - DJ;
  const Eigen::MatrixXd U_big = M.transpose() * V_big * M;
  const int df = std::max(0, static_cast<int>(pstar_total - numeric_rank(DJ)));

  out.theta_hat = theta;
  out.se = Eigen::VectorXd(q);
  for (Eigen::Index i = 0; i < q; ++i)
    out.se(i) = std::sqrt(std::max(out.Omega(i, i), 0.0));
  out.T_gof = T;
  out.df = df;
  out.rls_check = rls;

  if (df >= 1) {
    auto sd = compute_profile_contrast_spectrum(U_big, G_big, 1e-10);
    if (!sd.has_value())
      return perr("restricted grouped inference: GOF spectrum failed");
    SatorraDiffResult sd_df = *sd;
    const Eigen::Index navail = sd_df.eigenvalues.size();
    if (navail > df) {
      sd_df.eigenvalues = sd_df.eigenvalues.tail(df).eval();
    } else if (navail < df) {
      out.warnings.push_back("GOF spectrum rank " + std::to_string(navail) +
                             " < df " + std::to_string(df) +
                             " (ill-conditioned residual projector)");
    }
    sd_df.trace_CinvS = sd_df.eigenvalues.sum();
    sd_df.trace_CinvS_sq = sd_df.eigenvalues.squaredNorm();
    for (const auto& w : sd_df.warnings) out.warnings.push_back(w);

    auto lr = lr_test_satorra2000(T, sd_df);
    if (!lr.has_value())
      return perr("restricted grouped inference: GOF p-values failed");
    out.gof_eigenvalues = sd_df.eigenvalues;
    out.scale_c = lr->scale_c;
    out.p_scaled = lr->p_scaled;
    out.p_meanvar = lr->p_adjusted;
    out.p_scaled_shifted = lr->p_scaled_shifted;
    out.p_mixture = lr->p_mixture;
    for (const auto& w : lr->warnings) out.warnings.push_back(w);
  } else {
    out.warnings.push_back("joint df <= 0 (saturated blocks); GOF not reported");
  }

  out.U = U_big;
  out.M = M;
  out.V = V_big;
  out.Gamma = G_big;
  return out;
}

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_restricted_nt(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("restricted grouped inference: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("restricted grouped inference: Sigma(theta) failed");
  const bool has_means = has_mean_params(*ev);
  std::vector<Eigen::MatrixXd> gammas;
  gammas.reserve(sig->sigma.size());
  for (const auto& Sig : sig->sigma) {
    auto g = has_means ? data::gamma_nt_with_means(Sig) : data::gamma_nt(Sig);
    if (!g.has_value()) return perr("restricted grouped inference: gamma_nt failed");
    gammas.push_back(std::move(*g));
  }
  return noniterative_inference_grouped_restricted(
      pt, rep, samp, theta, which, disc, gammas, comm, composite);
}

post_expected<GroupedNonIterativeInference>
noniterative_inference_grouped_restricted_empirical(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const data::RawData& raw,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    Discrepancy disc,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  if (raw.X.size() != samp.S.size())
    return perr("restricted grouped inference: raw-data block count != sample-stats blocks");
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("restricted grouped inference: evaluator build failed");
  const bool has_means = has_mean_params(*ev);
  std::vector<Eigen::MatrixXd> gammas;
  gammas.reserve(raw.X.size());
  for (const auto& Xb : raw.X) {
    auto g = has_means ? data::empirical_gamma_with_means(Xb)
                       : data::empirical_gamma(Xb);
    if (!g.has_value())
      return perr("restricted grouped inference: empirical_gamma failed");
    gammas.push_back(std::move(*g));
  }
  return noniterative_inference_grouped_restricted(
      pt, rep, samp, theta, which, disc, gammas, comm, composite);
}

post_expected<NonIterativeSE>
noniterative_se_grouped_restricted(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    const std::vector<Eigen::MatrixXd>& gamma_per_block,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  auto d = build_grouped_derivative(
      pt, rep, samp, theta, which, MapKind::Restricted, comm, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  return finish_grouped_se(theta, samp, *d, gamma_per_block);
}

post_expected<NonIterativeSE>
noniterative_se_grouped_restricted_nt(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("restricted grouped SE: evaluator build failed");
  auto sig = ev->sigma(theta);
  if (!sig.has_value()) return perr("restricted grouped SE: Sigma(theta) failed");
  const bool has_means = has_mean_params(*ev);
  std::vector<Eigen::MatrixXd> gammas;
  gammas.reserve(sig->sigma.size());
  for (const auto& Sig : sig->sigma) {
    auto g = has_means ? data::gamma_nt_with_means(Sig) : data::gamma_nt(Sig);
    if (!g.has_value()) return perr("restricted grouped SE: gamma_nt failed");
    gammas.push_back(std::move(*g));
  }
  return noniterative_se_grouped_restricted(
      pt, rep, samp, theta, which, gammas, comm, composite);
}

post_expected<NonIterativeSE>
noniterative_se_grouped_restricted_empirical(
    const spec::LatentStructure& pt,
    const model::MatrixRep& rep,
    const data::SampleStats& samp,
    const data::RawData& raw,
    const Eigen::VectorXd& theta,
    estimate::frontier::NonIterativeEstimator which,
    estimate::frontier::CommunalityMethod comm,
    estimate::frontier::CompositeWeight composite) {
  auto d = build_grouped_derivative(
      pt, rep, samp, theta, which, MapKind::Restricted, comm, composite);
  if (!d.has_value()) return std::unexpected(d.error());
  return finish_grouped_empirical_se(theta, samp, raw, *d);
}

post_expected<NonIterativeDiffTest>
noniterative_difference_test(const GroupedNonIterativeInference& inf0,
                             const GroupedNonIterativeInference& inf1,
                             int df_d) {
  return noniterative_difference_test_impl(
      inf0.M, inf1.M, inf1.V, inf1.U, inf1.Gamma,
      inf0.T_gof, inf1.T_gof, df_d);
}

post_expected<ConstrainedNonIterativeFit>
noniterative_constrained_fit(const spec::LatentStructure& pt,
                             const GroupedNonIterativeInference& inf) {
  auto eqc = estimate::build_eq_constraints(pt);  // errors on inequality / nonlinear
  if (!eqc.has_value()) return std::unexpected(eqc.error());

  const Eigen::Index q = inf.theta_hat.size();
  if (eqc->npar != static_cast<std::int32_t>(q))
    return perr("constrained fit: constraint npar != theta size (a merged "
                "parameterization was passed; use the configural partable)");

  ConstrainedNonIterativeFit out;
  out.theta_hat = inf.theta_hat;
  out.Omega = inf.Omega;

  if (eqc->rank == 0) {  // no constraints — configural fit is returned unchanged.
    out.theta_tilde = inf.theta_hat;
    out.Omega_tilde = inf.Omega;
    out.se_constrained = inf.se;
    out.k = 0;
    out.W = 0.0;
    out.p_wald = 1.0;
    return out;
  }

  const Eigen::MatrixXd& R = eqc->A_eq;   // k × q
  const Eigen::VectorXd& c = eqc->b_eq;   // k (0 for pure merges, != 0 general)
  const int k = eqc->rank;

  const Eigen::VectorXd g = R * inf.theta_hat - c;
  const Eigen::MatrixXd OmRt = inf.Omega * R.transpose();  // q × k
  const Eigen::MatrixXd RORt = R * OmRt;                   // k × k
  Eigen::LDLT<Eigen::MatrixXd> ldlt(RORt);
  if (ldlt.info() != Eigen::Success)
    return perr("constrained fit: R Omega R' is not positive definite");

  const Eigen::VectorXd sol = ldlt.solve(g);               // (RΩR')⁻¹(Rθ̂−c)
  out.theta_tilde = inf.theta_hat - OmRt * sol;            // projection
  out.W = g.dot(sol);                                      // exact χ²_k
  out.k = k;
  out.p_wald = inference::chi2_pvalue(out.W, k);
  // Ω̃ = Ω − Ω R'(RΩR')⁻¹ R Ω  (rank q − k).
  out.Omega_tilde = inf.Omega - OmRt * ldlt.solve(OmRt.transpose());
  out.se_constrained = Eigen::VectorXd(q);
  for (Eigen::Index i = 0; i < q; ++i)
    out.se_constrained(i) = std::sqrt(std::max(out.Omega_tilde(i, i), 0.0));
  return out;
}

post_expected<ConstrainedNonIterativeFit>
noniterative_constrained_fit(const spec::LatentStructure& pt,
                             const NonIterativeSE& se) {
  GroupedNonIterativeInference inf;
  inf.theta_hat = se.theta_hat;
  inf.Omega = se.Omega;
  inf.se = se.se;
  inf.block_of_param = se.block_of_param;
  inf.warnings = se.warnings;
  return noniterative_constrained_fit(pt, inf);
}

post_expected<ScalarInvarianceFit>
noniterative_scalar_invariance(const spec::LatentStructure& pt,
                               const model::MatrixRep& rep,
                               const GroupedNonIterativeInference& inf,
                               std::size_t ref_group) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return perr("scalar invariance: evaluator build failed");
  if (!has_mean_params(*ev))
    return perr("scalar invariance: the model has no mean structure (fit a "
                "configural meanstructure model first)");
  const auto q = static_cast<Eigen::Index>(ev->n_free());
  if (inf.theta_hat.size() != q || inf.Omega.rows() != q || inf.Omega.cols() != q)
    return perr("scalar invariance: inference bundle does not match the model");

  auto asm_ = ev->assembled(inf.theta_hat);
  if (!asm_.has_value()) return perr("scalar invariance: assemble failed");
  const std::size_t nblk = asm_->blocks.size();
  if (nblk < 2) return perr("scalar invariance: need >= 2 groups");
  if (ref_group >= nblk) return perr("scalar invariance: ref_group out of range");

  // Reference-group metric: ν = m_r, projector P onto col(Λ_r).
  const Eigen::MatrixXd Lam  = asm_->blocks[ref_group].Lambda;  // p × m
  const Eigen::VectorXd m_ref = asm_->blocks[ref_group].Nu;     // p (= sample mean_r)
  const Eigen::Index p = Lam.rows();
  const Eigen::Index m = Lam.cols();
  if (m_ref.size() != p) return perr("scalar invariance: reference intercepts missing");
  if (p <= m) return perr("scalar invariance: p must exceed the number of factors");

  const Eigen::MatrixXd LtL = Lam.transpose() * Lam;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(LtL);
  if (ldlt.info() != Eigen::Success)
    return perr("scalar invariance: reference loadings are rank-deficient (Λ'Λ)");
  const Eigen::MatrixXd Lpinv = ldlt.solve(Lam.transpose());     // m × p, (Λ'Λ)⁻¹Λ'
  const Eigen::MatrixXd P     = Lam * Lpinv;                     // p × p
  const Eigen::MatrixXd Mperp = Eigen::MatrixXd::Identity(p, p) - P;

  std::vector<std::size_t> gs;
  for (std::size_t b = 0; b < nblk; ++b)
    if (b != ref_group) gs.push_back(b);
  const Eigen::Index G1 = static_cast<Eigen::Index>(gs.size());

  const std::vector<model::ParamLocation> locs = ev->param_locations();
  ScalarInvarianceFit out;
  out.ref_group = ref_group;
  out.groups = gs;
  out.nu = m_ref;
  out.d_stacked = Eigen::VectorXd::Zero(G1 * p);
  Eigen::MatrixXd Gstack = Eigen::MatrixXd::Zero(G1 * p, q);

  for (Eigen::Index gi = 0; gi < G1; ++gi) {
    const std::size_t b = gs[static_cast<std::size_t>(gi)];
    const auto& blk = asm_->blocks[b];
    if (blk.Lambda.rows() != p || blk.Lambda.cols() != m || blk.Nu.size() != p)
      return perr("scalar invariance: groups do not share a common CFA layout");

    const Eigen::VectorXd delta = blk.Nu - m_ref;   // m_g − m_r
    const Eigen::VectorXd alpha = Lpinv * delta;     // α_g = (Λ'Λ)⁻¹Λ'δ
    const Eigen::VectorXd d_g   = Mperp * delta;     // (I − P)δ
    out.d_stacked.segment(gi * p, p) = d_g;

    // Leading-order Jacobians (valid to O(d_g) under H0):
    //   ∂d_g/∂m_g[i]   =  (I−P) e_i      ∂d_g/∂m_r[i]   = −(I−P) e_i
    //   ∂d_g/∂Λ_r[i,j] = −α_j (I−P) e_i
    //   ∂α_g/∂m_g[i]   =  Λ⁺ e_i         ∂α_g/∂m_r[i]   = −Λ⁺ e_i
    //   ∂α_g/∂Λ_r[i,j] = −α_j Λ⁺ e_i
    Eigen::MatrixXd Gg = Eigen::MatrixXd::Zero(p, q);   // ∂d_g/∂θ
    Eigen::MatrixXd Ag = Eigen::MatrixXd::Zero(m, q);   // ∂α_g/∂θ
    for (std::size_t k = 0; k < locs.size(); ++k) {
      const auto& loc = locs[k];
      const auto kk = static_cast<Eigen::Index>(k);
      const auto lb = static_cast<std::size_t>(loc.block);
      if (loc.mat == model::MatId::Nu && lb == b) {
        if (loc.row < 0 || loc.row >= p) continue;
        Gg.col(kk) =  Mperp.col(loc.row);
        Ag.col(kk) =  Lpinv.col(loc.row);
      } else if (loc.mat == model::MatId::Nu && lb == ref_group) {
        if (loc.row < 0 || loc.row >= p) continue;
        Gg.col(kk) = -Mperp.col(loc.row);
        Ag.col(kk) = -Lpinv.col(loc.row);
      } else if (loc.mat == model::MatId::Lambda && lb == ref_group) {
        if (loc.row < 0 || loc.row >= p || loc.col < 0 || loc.col >= m) continue;
        const double a_j = alpha(loc.col);
        Gg.col(kk) = -a_j * Mperp.col(loc.row);
        Ag.col(kk) = -a_j * Lpinv.col(loc.row);
      }
    }
    Gstack.middleRows(gi * p, p) = Gg;

    const Eigen::MatrixXd acov = Ag * inf.Omega * Ag.transpose();
    Eigen::VectorXd ase(m);
    for (Eigen::Index i = 0; i < m; ++i) ase(i) = std::sqrt(std::max(acov(i, i), 0.0));
    out.alpha.push_back(alpha);
    out.alpha_cov.push_back(acov);
    out.alpha_se.push_back(std::move(ase));
  }

  const Eigen::MatrixXd Cov = Gstack * inf.Omega * Gstack.transpose();
  auto pq = psd_pinv_quadform(Cov, out.d_stacked, 1e-9);
  out.W = pq.W;
  out.rank = pq.rank;
  out.df = static_cast<int>(G1 * (p - m));
  if (pq.rank != out.df)
    out.warnings.push_back("Cov(d) numeric rank " + std::to_string(pq.rank) +
                           " != expected df " + std::to_string(out.df) +
                           " (near-degenerate reference metric)");
  out.p_value = inference::chi2_pvalue(out.W, out.df);
  return out;
}

post_expected<ScalarInvarianceFit>
noniterative_scalar_invariance(const spec::LatentStructure& pt,
                               const model::MatrixRep& rep,
                               const NonIterativeSE& se,
                               std::size_t ref_group) {
  GroupedNonIterativeInference inf;
  inf.theta_hat = se.theta_hat;
  inf.Omega = se.Omega;
  inf.se = se.se;
  inf.block_of_param = se.block_of_param;
  inf.warnings = se.warnings;
  return noniterative_scalar_invariance(pt, rep, inf, ref_group);
}

}  // namespace magmaan::robust::frontier
