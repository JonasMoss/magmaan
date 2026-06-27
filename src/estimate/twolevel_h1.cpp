#include "magmaan/estimate/twolevel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Cholesky>

// Two-level saturated (H1) model: maximize the full-information likelihood over
// unstructured (Sigma_W, Sigma_B, mu). No closed form under unbalanced cluster
// sizes; solved by ECM (the analogue of fiml_h1_moments). Math in
// docs/research/notes/twolevel_ml_derivation.tex, section "Saturated (H1)".
//
// Per iteration, per group:
//   A. mu  <- (sum_d d Jd Md)^-1 (sum_d d Md T1_d)        exact GLS (g_mu = 0)
//   B. E-step (Kalman-gain form, needs only Md = (Sw + d Sb)^-1):
//        Bd = d Sb Md            (posterior gain, m_j = Bd (ybar_j - mu))
//        Vd = (I - Bd) Sb        (posterior cov of the cluster effect)
//   C. M-step:
//        Sb <- (1/J)  sum_d [ Bd Ad Bd' + Jd Vd ]
//        Sw <- (1/N) [ SSW + sum_d d ( (I-Bd) Ad (I-Bd)' + Jd Vd ) ]
// monotone in the observed deviance (exact CM in mu, EM in the covariances).

namespace magmaan::estimate::twolevel {

namespace {

FitError h1_err(std::string detail) {
  return FitError{FitError::Kind::NumericIssue, std::move(detail)};
}
FitError h1_npd(const std::string& what) {
  return FitError{FitError::Kind::NonPositiveDefiniteSigma,
                  what + " is not positive definite (twolevel H1)"};
}

double logdet_chol(const Eigen::LLT<Eigen::MatrixXd>& llt) {
  const auto L = llt.matrixL();
  double s = 0.0;
  for (Eigen::Index i = 0; i < L.rows(); ++i) s += std::log(L(i, i));
  return 2.0 * s;
}
Eigen::MatrixXd symmetrize(const Eigen::MatrixXd& M) {
  return (0.5 * (M + M.transpose())).eval();
}

// A_d(mu) = T2 - mu T1' - T1 mu' + Jd mu mu'.
Eigen::MatrixXd a_of(const data::ClusterSizePattern& sp,
                     const Eigen::VectorXd& mu) {
  const double Jd = static_cast<double>(sp.n_clusters);
  return sp.sum_cluster_mean_cp - mu * sp.sum_cluster_mean.transpose() -
         sp.sum_cluster_mean * mu.transpose() + Jd * (mu * mu.transpose());
}

// Saturated fit for one group. Returns the converged (Sw, Sb, mu), the deviance
// F (Thm 1), and the iteration count.
struct GroupH1 {
  Eigen::MatrixXd Sw, Sb;
  Eigen::VectorXd mu;
  double value = 0.0;
  int iterations = 0;
};

fit_expected<GroupH1> solve_group(const data::ClusterGroupStats& cg,
                                  const TwoLevelH1Options& opts) {
  const Eigen::Index p = cg.within_scatter.rows();
  if (p <= 0 || cg.n_clusters <= 0 || cg.size_patterns.empty()) {
    return std::unexpected(h1_err("twolevel H1: empty or degenerate group"));
  }
  const double N = static_cast<double>(cg.n_within);
  const double J = static_cast<double>(cg.n_clusters);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(p, p);

  // --- initial values -------------------------------------------------------
  Eigen::MatrixXd Sw = symmetrize(cg.within_scatter / std::max(1.0, N - J));
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(p);
  for (const auto& sp : cg.size_patterns) mu += sp.sum_cluster_mean;
  mu /= J;
  // Method-of-moments Sb = (cluster-mean scatter)/J - Sw/dbar, floored to PD.
  Eigen::MatrixXd between_scatter = Eigen::MatrixXd::Zero(p, p);
  double dbar = 0.0;
  for (const auto& sp : cg.size_patterns) {
    between_scatter += a_of(sp, mu);
    dbar += static_cast<double>(sp.cluster_size) *
            static_cast<double>(sp.n_clusters);
  }
  dbar /= J;
  Eigen::MatrixXd Sb = symmetrize(between_scatter / J - Sw / dbar);
  if (Eigen::LLT<Eigen::MatrixXd>(Sb).info() != Eigen::Success) {
    Sb = symmetrize(0.1 * Sw);  // PD fallback (Sw is PD)
  }

  // --- ECM iterations -------------------------------------------------------
  int it = 0;
  for (; it < opts.max_iter; ++it) {
    // M_d from the current (Sw, Sb), reused by the mu solve and the E-step.
    std::vector<Eigen::MatrixXd> Md(cg.size_patterns.size());
    Eigen::MatrixXd mu_lhs = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd mu_rhs = Eigen::VectorXd::Zero(p);
    for (std::size_t s = 0; s < cg.size_patterns.size(); ++s) {
      const auto& sp = cg.size_patterns[s];
      const double d = static_cast<double>(sp.cluster_size);
      const double Jd = static_cast<double>(sp.n_clusters);
      Eigen::LLT<Eigen::MatrixXd> llt(Sw + d * Sb);
      if (llt.info() != Eigen::Success)
        return std::unexpected(h1_npd("Sigma_W + d*Sigma_B"));
      Md[s] = llt.solve(I);
      mu_lhs += (d * Jd) * Md[s];
      mu_rhs += d * (Md[s] * sp.sum_cluster_mean);
    }
    // A. exact conditional max for mu.
    mu = mu_lhs.ldlt().solve(mu_rhs);

    // B/C. E-step + covariance M-step with the updated mu.
    Eigen::MatrixXd accB = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd accW = cg.within_scatter;
    for (std::size_t s = 0; s < cg.size_patterns.size(); ++s) {
      const auto& sp = cg.size_patterns[s];
      const double d = static_cast<double>(sp.cluster_size);
      const double Jd = static_cast<double>(sp.n_clusters);
      const Eigen::MatrixXd Bd = d * Sb * Md[s];          // posterior gain
      const Eigen::MatrixXd Vd = symmetrize(Sb - Bd * Sb);  // posterior cov
      const Eigen::MatrixXd Ad = a_of(sp, mu);
      const Eigen::MatrixXd ImBd = I - Bd;
      accB += Bd * Ad * Bd.transpose() + Jd * Vd;
      accW += d * (ImBd * Ad * ImBd.transpose() + Jd * Vd);
    }
    const Eigen::MatrixXd Sb_new = symmetrize(accB / J);
    const Eigen::MatrixXd Sw_new = symmetrize(accW / N);

    const double change =
        std::max((Sw_new - Sw).cwiseAbs().maxCoeff(),
                 (Sb_new - Sb).cwiseAbs().maxCoeff());
    Sw = Sw_new;
    Sb = Sb_new;
    if (change < opts.tol) {
      ++it;
      break;
    }
  }

  // --- converged deviance F (Thm 1) -----------------------------------------
  Eigen::LLT<Eigen::MatrixXd> llt_w(Sw);
  if (llt_w.info() != Eigen::Success)
    return std::unexpected(h1_npd("Sigma_W (final)"));
  double F = (N - J) * logdet_chol(llt_w) + llt_w.solve(cg.within_scatter).trace();
  for (const auto& sp : cg.size_patterns) {
    const double d = static_cast<double>(sp.cluster_size);
    const double Jd = static_cast<double>(sp.n_clusters);
    Eigen::LLT<Eigen::MatrixXd> llt_b(Sw + d * Sb);
    if (llt_b.info() != Eigen::Success)
      return std::unexpected(h1_npd("Sigma_W + d*Sigma_B (final)"));
    F += Jd * logdet_chol(llt_b) + d * llt_b.solve(a_of(sp, mu)).trace();
  }
  if (!std::isfinite(F))
    return std::unexpected(h1_err("twolevel H1: deviance non-finite"));

  return GroupH1{Sw, Sb, mu, F, it};
}

}  // namespace

fit_expected<TwoLevelH1>
twolevel_h1_moments(const ClusterSampleStats& cs, TwoLevelH1Options opts) {
  TwoLevelH1 h1;
  const std::size_t ng = cs.groups.size();
  h1.sigma_w.resize(ng);
  h1.sigma_b.resize(ng);
  h1.mu_b.resize(ng);
  double value = 0.0;
  int iters = 0;
  for (std::size_t g = 0; g < ng; ++g) {
    auto gh = solve_group(cs.groups[g], opts);
    if (!gh) return std::unexpected(gh.error());
    h1.sigma_w[g] = gh->Sw;
    h1.sigma_b[g] = gh->Sb;
    h1.mu_b[g] = gh->mu;
    value += gh->value;
    iters = std::max(iters, gh->iterations);
  }
  h1.value = value;
  h1.iterations = iters;
  return h1;
}

fit_expected<data::SampleStats>
twolevel_h1_sample_stats(const ClusterSampleStats& cs, const model::MatrixRep& rep,
                         TwoLevelH1Options opts) {
  auto h1 = twolevel_h1_moments(cs, opts);
  if (!h1) return std::unexpected(h1.error());

  const auto pairs = model::level_block_pairs(rep);
  const std::size_t nb = rep.dims.size();
  data::SampleStats samp;
  samp.S.assign(nb, Eigen::MatrixXd());
  samp.mean.assign(nb, Eigen::VectorXd());
  samp.n_obs.assign(nb, 0);
  for (std::size_t g = 0; g < cs.groups.size(); ++g) {
    if (g >= pairs.size() || pairs[g].within < 0 || pairs[g].between < 0) {
      return std::unexpected(h1_err(
          "twolevel_h1_sample_stats: group lacks a within/between block pair"));
    }
    const auto w = static_cast<std::size_t>(pairs[g].within);
    const auto b = static_cast<std::size_t>(pairs[g].between);
    samp.S[w] = h1->sigma_w[g];
    samp.S[b] = h1->sigma_b[g];
    samp.mean[b] = h1->mu_b[g];  // within level carries no mean structure
    samp.n_obs[w] = cs.groups[g].n_within;
    samp.n_obs[b] = cs.groups[g].n_clusters;
  }
  return samp;
}

}  // namespace magmaan::estimate::twolevel
