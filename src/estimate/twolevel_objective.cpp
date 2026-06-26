#include "magmaan/estimate/twolevel.hpp"

#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Cholesky>

#include "detail_vech.hpp"

// Two-level full-information normal-theory ML: the deviance value and its
// analytic gradient. The math is derived in
// docs/research/notes/twolevel_ml_derivation.tex; theorem references below are
// to that note.
//
//   F = (N-J) log|Sw| + tr(Sw^-1 SSW)
//       + sum_d [ Jd log|Sw + d Sb| + d tr(Md Ad(mu)) ]                  (Thm 1)
//
// with Md = (Sw + d Sb)^-1 and Ad(mu) = T2_d - mu T1_d' - T1_d mu' + Jd mu mu'.
// The gradient (Thm 2 / Cor 1) is the ordinary per-block discrepancy-gradient
// contraction g = J_sigma' w + J_mu' u: the within block carries the
// vech-doubled G_W, the between block the vech-doubled G_B, the between mean
// carries g_mu.

namespace magmaan::estimate::twolevel {

using detail::vech_index;
using detail::vech_len;

namespace {

FitError err(std::string detail) {
  return FitError{FitError::Kind::NumericIssue, std::move(detail)};
}
FitError npd(const std::string& what) {
  return FitError{FitError::Kind::NonPositiveDefiniteSigma,
                  what + " is not positive definite"};
}

double log_det_chol(const Eigen::LLT<Eigen::MatrixXd>& llt) {
  const auto L = llt.matrixL();
  double s = 0.0;
  for (Eigen::Index i = 0; i < L.rows(); ++i) s += std::log(L(i, i));
  return 2.0 * s;
}

// Matrix gradients of F for one group (symmetric G_W, G_B and the mean
// gradient g_mu), filled only when want_grad is set.
struct GroupGrad {
  Eigen::MatrixXd GW;
  Eigen::MatrixXd GB;
  Eigen::VectorXd gmu;
};

// Accumulate one group's contribution to F and (optionally) its matrix
// gradients. The algebra lives here so twolevel_value and
// twolevel_value_gradient cannot drift apart.
fit_expected<double>
accumulate_group(const data::ClusterGroupStats& cg, const Eigen::MatrixXd& Sw,
                 const Eigen::MatrixXd& Sb, const Eigen::VectorXd& mu,
                 bool want_grad, GroupGrad* out) {
  const Eigen::Index p = Sw.rows();
  if (Sw.cols() != p || Sb.rows() != p || Sb.cols() != p || mu.size() != p) {
    return std::unexpected(err("twolevel: Sigma_W/Sigma_B/mu shape mismatch"));
  }
  if (cg.within_scatter.rows() != p || cg.within_scatter.cols() != p) {
    return std::unexpected(
        err("twolevel: within_scatter shape does not match Sigma_W"));
  }

  Eigen::LLT<Eigen::MatrixXd> llt_w(Sw);
  if (llt_w.info() != Eigen::Success)
    return std::unexpected(npd("Sigma_W"));
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(p, p);
  const Eigen::MatrixXd W0 = llt_w.solve(I);            // Sigma_W^-1
  const double NmJ =
      static_cast<double>(cg.n_within - cg.n_clusters);  // N_g - J_g

  double F =
      NmJ * log_det_chol(llt_w) + llt_w.solve(cg.within_scatter).trace();

  if (want_grad) {
    out->GW = NmJ * W0 - W0 * cg.within_scatter * W0;
    out->GB = Eigen::MatrixXd::Zero(p, p);
    out->gmu = Eigen::VectorXd::Zero(p);
  }

  for (const auto& sp : cg.size_patterns) {
    const double d = static_cast<double>(sp.cluster_size);
    const double Jd = static_cast<double>(sp.n_clusters);
    if (sp.sum_cluster_mean.size() != p ||
        sp.sum_cluster_mean_cp.rows() != p ||
        sp.sum_cluster_mean_cp.cols() != p) {
      return std::unexpected(
          err("twolevel: cluster-size-pattern statistic shape mismatch"));
    }
    const Eigen::MatrixXd SwdSb = Sw + d * Sb;           // Sigma_W + d Sigma_B
    Eigen::LLT<Eigen::MatrixXd> llt_b(SwdSb);
    if (llt_b.info() != Eigen::Success)
      return std::unexpected(npd("Sigma_W + d*Sigma_B"));
    // A_d(mu) = T2 - mu T1' - T1 mu' + Jd mu mu'
    const Eigen::MatrixXd Ad =
        sp.sum_cluster_mean_cp - mu * sp.sum_cluster_mean.transpose() -
        sp.sum_cluster_mean * mu.transpose() + Jd * (mu * mu.transpose());

    F += Jd * log_det_chol(llt_b) + d * llt_b.solve(Ad).trace();

    if (want_grad) {
      const Eigen::MatrixXd Md = llt_b.solve(I);         // (Sw + d Sb)^-1
      const Eigen::MatrixXd Rd = Jd * Md - d * (Md * Ad * Md);
      out->GW += Rd;
      out->GB += d * Rd;
      out->gmu.noalias() +=
          (-2.0 * d) * (Md * (sp.sum_cluster_mean - Jd * mu));
    }
  }

  if (want_grad) {
    out->GW = (0.5 * (out->GW + out->GW.transpose())).eval();
    out->GB = (0.5 * (out->GB + out->GB.transpose())).eval();
  }
  return F;
}

struct GroupBlocks {
  std::size_t bW = 0, bB = 0;
};
fit_expected<GroupBlocks>
group_blocks(std::size_t g, const std::vector<model::LevelBlockPair>& pairs,
             const model::ImpliedMoments& m) {
  if (g >= pairs.size() || pairs[g].within < 0 || pairs[g].between < 0) {
    return std::unexpected(err("twolevel: group " + std::to_string(g) +
                               " lacks a within/between block pair"));
  }
  GroupBlocks gb{static_cast<std::size_t>(pairs[g].within),
                 static_cast<std::size_t>(pairs[g].between)};
  if (gb.bW >= m.sigma.size() || gb.bB >= m.sigma.size() ||
      gb.bB >= m.mu.size()) {
    return std::unexpected(err("twolevel: block index out of range for group " +
                               std::to_string(g)));
  }
  return gb;
}

}  // namespace

fit_expected<TwoLevelCache> twolevel_prepare(const ClusterSampleStats& cs) {
  TwoLevelCache cache;
  for (const auto& g : cs.groups) cache.n_total += g.n_within;
  return cache;
}

fit_expected<double>
twolevel_value(const ClusterSampleStats& cs, const TwoLevelCache& /*cache*/,
               const model::ImpliedMoments& moments,
               const model::MatrixRep& rep) {
  const auto pairs = model::level_block_pairs(rep);
  double F = 0.0;
  for (std::size_t g = 0; g < cs.groups.size(); ++g) {
    auto gb = group_blocks(g, pairs, moments);
    if (!gb) return std::unexpected(gb.error());
    auto Fg = accumulate_group(cs.groups[g], moments.sigma[gb->bW],
                               moments.sigma[gb->bB], moments.mu[gb->bB],
                               /*want_grad=*/false, nullptr);
    if (!Fg) return std::unexpected(Fg.error());
    F += *Fg;
  }
  if (!std::isfinite(F)) {
    return std::unexpected(FitError{FitError::Kind::NonFiniteObjective,
                                    "twolevel: F evaluated to non-finite"});
  }
  return F;
}

fit_expected<TwoLevelValueGradient>
twolevel_value_gradient(const ClusterSampleStats& cs,
                        const TwoLevelCache& /*cache*/,
                        const model::ImpliedMoments& moments,
                        const Eigen::MatrixXd& J_sigma,
                        const Eigen::MatrixXd& J_mu,
                        const model::MatrixRep& rep) {
  const auto pairs = model::level_block_pairs(rep);

  // Block offsets into the stacked vech / mean vectors (block order == sigma).
  const std::size_t nb = moments.sigma.size();
  std::vector<Eigen::Index> vech_off(nb), mu_off(nb);
  Eigen::Index total_vech = 0, total_p = 0;
  for (std::size_t b = 0; b < nb; ++b) {
    vech_off[b] = total_vech;
    mu_off[b] = total_p;
    total_vech += vech_len(moments.sigma[b].rows());
    total_p += moments.sigma[b].rows();
  }

  Eigen::VectorXd w = Eigen::VectorXd::Zero(total_vech);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(total_p);
  double F = 0.0;

  for (std::size_t g = 0; g < cs.groups.size(); ++g) {
    auto gb = group_blocks(g, pairs, moments);
    if (!gb) return std::unexpected(gb.error());
    GroupGrad gg;
    auto Fg = accumulate_group(cs.groups[g], moments.sigma[gb->bW],
                               moments.sigma[gb->bB], moments.mu[gb->bB],
                               /*want_grad=*/true, &gg);
    if (!Fg) return std::unexpected(Fg.error());
    F += *Fg;

    const Eigen::Index p = moments.sigma[gb->bW].rows();
    for (Eigen::Index c = 0; c < p; ++c) {
      for (Eigen::Index r = c; r < p; ++r) {
        const Eigen::Index k = vech_index(p, r, c);
        const double dbl = (r == c) ? 1.0 : 2.0;
        w(vech_off[gb->bW] + k) += dbl * gg.GW(r, c);
        w(vech_off[gb->bB] + k) += dbl * gg.GB(r, c);
      }
    }
    u.segment(mu_off[gb->bB], p) += gg.gmu;
  }

  if (!std::isfinite(F)) {
    return std::unexpected(FitError{FitError::Kind::NonFiniteObjective,
                                    "twolevel: F evaluated to non-finite"});
  }
  if (J_sigma.rows() != total_vech) {
    return std::unexpected(err(
        "twolevel: J_sigma row count " + std::to_string(J_sigma.rows()) +
        " != stacked vech length " + std::to_string(total_vech)));
  }
  const bool has_means = J_mu.size() > 0;
  if (has_means && J_mu.rows() != total_p) {
    return std::unexpected(
        err("twolevel: J_mu row count " + std::to_string(J_mu.rows()) +
            " != stacked p " + std::to_string(total_p)));
  }

  TwoLevelValueGradient out;
  out.value = F;
  out.gradient = J_sigma.transpose() * w;
  if (has_means) out.gradient.noalias() += J_mu.transpose() * u;
  return out;
}

fit_expected<optim::ScalarProblem>
twolevel_ml_objective(const model::ModelEvaluator& ev,
                      const ClusterSampleStats& cs) {
  auto cache_or = twolevel_prepare(cs);
  if (!cache_or) return std::unexpected(cache_or.error());
  optim::ScalarProblem prob;
  prob.n_param = static_cast<Eigen::Index>(ev.n_free());
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  // `ev` and `cs` must outlive the returned problem (see header). f returns the
  // optimiser's objective ½F and writes ½∇F (magmaan's fmin = ½·deviance).
  prob.f = [&ev, &cs, cache = *cache_or](const Eigen::VectorXd& theta,
                                         Eigen::VectorXd& grad) -> double {
    auto e = ev.evaluate(theta, /*with_sigma_jac=*/true, /*with_mu_jac=*/true);
    if (!e) return std::numeric_limits<double>::infinity();
    auto vg = twolevel_value_gradient(cs, cache, e->moments, e->J_sigma,
                                      e->J_mu, ev.matrix_rep());
    if (!vg) return std::numeric_limits<double>::infinity();
    grad = 0.5 * vg->gradient;
    return 0.5 * vg->value;
  };
  return prob;
}

fit_expected<Eigen::MatrixXd>
twolevel_information(const model::ModelEvaluator& ev,
                     const ClusterSampleStats& cs, const Eigen::VectorXd& theta,
                     bool /*expected*/) {
  // v1: observed information by central finite differences of the analytic
  // full-deviance gradient. info = ½·∂²F/∂θ² is the Fisher information on the
  // F = -2logL scale; the caller inverts it for vcov. (`expected` is accepted
  // for the contract but the expected-information path is deferred.)
  auto cache_or = twolevel_prepare(cs);
  if (!cache_or) return std::unexpected(cache_or.error());
  const TwoLevelCache cache = *cache_or;
  const model::MatrixRep& rep = ev.matrix_rep();
  const Eigen::Index n = theta.size();

  auto grad_at = [&](const Eigen::VectorXd& th) -> fit_expected<Eigen::VectorXd> {
    auto e = ev.evaluate(th, true, true);
    if (!e) {
      return std::unexpected(err("twolevel information: evaluator failed at a "
                                 "finite-difference point"));
    }
    auto vg = twolevel_value_gradient(cs, cache, e->moments, e->J_sigma,
                                      e->J_mu, rep);
    if (!vg) return std::unexpected(vg.error());
    return vg->gradient;  // full-deviance gradient ∂F/∂θ
  };

  const double h = 1e-5;
  Eigen::MatrixXd Hess(n, n);
  for (Eigen::Index j = 0; j < n; ++j) {
    Eigen::VectorXd tp = theta, tm = theta;
    tp(j) += h;
    tm(j) -= h;
    auto gp = grad_at(tp);
    if (!gp) return std::unexpected(gp.error());
    auto gm = grad_at(tm);
    if (!gm) return std::unexpected(gm.error());
    Hess.col(j) = (*gp - *gm) / (2.0 * h);
  }
  Hess = (0.5 * (Hess + Hess.transpose())).eval();  // symmetrize
  return Eigen::MatrixXd(0.5 * Hess);               // ½·∂²F/∂θ²
}

}  // namespace magmaan::estimate::twolevel
