#include "fiml_internal.hpp"

namespace magmaan::estimate::fiml {
namespace internal {
post_expected<Eigen::MatrixXd>
fiml_casewise_scores(const RawData& raw,
                     const FIMLCache& cache,
                     const model::ImpliedMoments& moments,
                     const Eigen::MatrixXd& J_sigma,
                     const Eigen::MatrixXd& J_mu) {
  if (raw.X.size() != moments.sigma.size() ||
      (!moments.mu.empty() && moments.mu.size() != moments.sigma.size())) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust: raw and implied moment block count mismatch"));
  }
  Eigen::Index n_total = 0;
  for (const auto& X : raw.X) n_total += X.rows();
  Eigen::MatrixXd scores(n_total, J_sigma.cols());
  const bool has_means = !moments.mu.empty() && J_mu.rows() > 0;

  Eigen::VectorXd w(J_sigma.rows());
  Eigen::VectorXd u(J_mu.rows());
  Eigen::Index row_out = 0;
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    const Eigen::Index p = X.cols();
    const Eigen::Index sigma_off = cache.sigma_offsets[b];
    const Eigen::Index mu_off = cache.mu_offsets[b];
    PatternMomentMap bypat;
    std::vector<Eigen::Index> obs;
    for (Eigen::Index r = 0; r < X.rows(); ++r) {
      obs.clear();
      obs.reserve(static_cast<std::size_t>(p));
      if (raw.mask.empty()) {
        for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
      } else {
        for (Eigen::Index c = 0; c < p; ++c)
          if (raw.mask[b](r, c) != 0) obs.push_back(c);
      }
      if (obs.empty()) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML robust: row has no observed values"));
      }
      if (!finite_observed_row(X, obs, r)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "FIML robust: non-finite observed value"));
      }
      const Eigen::VectorXd zero_mu = Eigen::VectorXd::Zero(p);
      const Eigen::VectorXd& mu_b = has_means ? moments.mu[b] : zero_mu;
      auto pm_or = pattern_moments_for(
          bypat, obs, mu_b, moments.sigma[b],
          "FIML robust: implied observed-pattern Sigma");
      if (!pm_or.has_value()) return std::unexpected(pm_or.error());
      const PatternMoments& pm = **pm_or;

      const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
      Eigen::VectorXd d(q);
      for (Eigen::Index j = 0; j < q; ++j) {
        d(j) = X(r, obs[static_cast<std::size_t>(j)]) - pm.Mu_o(j);
      }
      const Eigen::VectorXd z = pm.SigmaInv * d;
      const Eigen::MatrixXd G = pm.SigmaInv - z * z.transpose();

      w.setZero();
      u.setZero();
      for (Eigen::Index cj = 0; cj < q; ++cj) {
        const Eigen::Index c = obs[static_cast<std::size_t>(cj)];
        for (Eigen::Index ri = cj; ri < q; ++ri) {
          const Eigen::Index rr0 = obs[static_cast<std::size_t>(ri)];
          const Eigen::Index rr = std::max(rr0, c);
          const Eigen::Index cc = std::min(rr0, c);
          const Eigen::Index idx = sigma_off + vech_index(p, rr, cc);
          w(idx) += (ri == cj) ? G(ri, cj) : 2.0 * G(ri, cj);
        }
      }
      if (has_means) {
        for (Eigen::Index i = 0; i < q; ++i) {
          const Eigen::Index rr = obs[static_cast<std::size_t>(i)];
          u(mu_off + rr) += -2.0 * z(i);
        }
      }
      scores.row(row_out).noalias() = w.transpose() * J_sigma;
      if (has_means) scores.row(row_out).noalias() += u.transpose() * J_mu;
      ++row_out;
    }
  }
  return scores;
}

post_expected<Eigen::MatrixXd>
fiml_observed_hessian_fd(spec::LatentStructure pt,
                         const model::MatrixRep& rep,
                         const RawData& raw,
                         const FIMLCache& cache,
                         const SampleStats& start_samp,
                         const Estimates& est,
                         FIML discrepancy,
                         double h_step) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error(), "resolve_fixed_x_from_sample"));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  const Eigen::Index q = static_cast<Eigen::Index>(ev.n_free());
  auto grad_at = [&](const Eigen::VectorXd& theta)
      -> post_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ModelEvaluator::evaluate failed: " + eval.error().detail));
    }
    auto g_or = discrepancy.gradient(raw, cache, eval->moments,
                                     eval->J_sigma, eval->J_mu);
    if (!g_or.has_value()) {
      return std::unexpected(fit_to_post(g_or.error(), "FIML gradient"));
    }
    return *g_or;
  };

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  for (Eigen::Index k = 0; k < q; ++k) {
    Eigen::VectorXd tp = est.theta;
    Eigen::VectorXd tm = est.theta;
    tp(k) += h_step;
    tm(k) -= h_step;
    auto gp = grad_at(tp);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = grad_at(tm);
    if (!gm.has_value()) return std::unexpected(gm.error());
    H.col(k) = (*gp - *gm) / (2.0 * h_step);
  }
  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

Eigen::Index vech_row(Eigen::Index p, Eigen::Index target) {
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      if (k == target) return r;
      ++k;
    }
  }
  return 0;
}

Eigen::Index vech_col(Eigen::Index p, Eigen::Index target) {
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      if (k == target) return c;
      ++k;
    }
  }
  return 0;
}



// trace(D_l · X · D_k · Y) where D is the symmetric vech basis matrix of the
// (a, b) entry (≤2 nonzeros), expanded entry-wise in O(1).
double basis_trace_xy(const Eigen::MatrixXd& X,
                      const Eigen::MatrixXd& Y,
                      const SigmaBasis& l,
                      const SigmaBasis& k) {
  const Eigen::Index lu[2] = {l.a, l.b};
  const Eigen::Index lv[2] = {l.b, l.a};
  const Eigen::Index ku[2] = {k.a, k.b};
  const Eigen::Index kv[2] = {k.b, k.a};
  const int ln = (l.a == l.b) ? 1 : 2;
  const int kn = (k.a == k.b) ? 1 : 2;

  double out = 0.0;
  for (int i = 0; i < ln; ++i) {
    for (int j = 0; j < kn; ++j) {
      out += X(lv[i], ku[j]) * Y(kv[j], lu[i]);
    }
  }
  return out;
}

post_expected<Eigen::MatrixXd>
fiml_saturated_scores_block(const RawData& raw,
                            std::size_t block,
                            const Eigen::VectorXd& mu,
                            const Eigen::MatrixXd& Sigma) {
  if (block >= raw.X.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: block index out of range"));
  }
  if (!raw.mask.empty() && block >= raw.mask.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: mask block index out of range"));
  }
  const Eigen::MatrixXd& X = raw.X[block];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (mu.size() != p || Sigma.rows() != p || Sigma.cols() != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1: saturated moments do not match raw block"));
  }
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd scores = Eigen::MatrixXd::Zero(n, p + pstar);
  PatternMomentMap bypat;
  std::vector<Eigen::Index> obs;
  for (Eigen::Index r = 0; r < n; ++r) {
    obs.clear();
    obs.reserve(static_cast<std::size_t>(p));
    if (raw.mask.empty()) {
      for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
    } else {
      for (Eigen::Index c = 0; c < p; ++c)
        if (raw.mask[block](r, c) != 0) obs.push_back(c);
    }
    if (obs.empty() || !finite_observed_row(X, obs, r)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML robust H1: invalid observed row"));
    }
    auto pm_or = pattern_moments_for(bypat, obs, mu, Sigma,
                                     "FIML robust H1: Sigma_oo");
    if (!pm_or.has_value()) return std::unexpected(pm_or.error());
    const PatternMoments& pm = **pm_or;

    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    Eigen::VectorXd d(q);
    for (Eigen::Index j = 0; j < q; ++j) {
      d(j) = X(r, obs[static_cast<std::size_t>(j)]) - pm.Mu_o(j);
    }
    const Eigen::VectorXd z = pm.SigmaInv * d;
    const Eigen::MatrixXd G = pm.SigmaInv - z * z.transpose();
    for (Eigen::Index i = 0; i < q; ++i) {
      const Eigen::Index rr = obs[static_cast<std::size_t>(i)];
      scores(r, rr) = -2.0 * z(i);
    }
    for (Eigen::Index cj = 0; cj < q; ++cj) {
      const Eigen::Index c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = cj; ri < q; ++ri) {
        const Eigen::Index rr0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index rr = std::max(rr0, c);
        const Eigen::Index cc = std::min(rr0, c);
        const Eigen::Index idx = p + vech_index(p, rr, cc);
        scores(r, idx) += (ri == cj) ? G(ri, cj) : 2.0 * G(ri, cj);
      }
    }
  }
  return scores;
}

// Analytic observed Hessian of the mean saturated deviance, aggregated per
// observed-value pattern instead of per row: with z_r = Σ_oo⁻¹(x_r − μ_o),
// every row contribution is at most quadratic in z_r, so a pattern enters
// only through n_r, Σ_r z_r = A·Σ_r d_r and Σ_r z_r z_rᵀ = A(Σ_r d_r d_rᵀ)A —
// all closed-form in the pattern's stored mean and covariance.
post_expected<Eigen::MatrixXd>
fiml_saturated_hessian_analytic_block(const FIMLCache& cache,
                                      std::size_t block,
                                      const Eigen::VectorXd& mu,
                                      const Eigen::MatrixXd& Sigma) {
  if (block >= cache.block_p.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1 analytic: block index out of range"));
  }
  const Eigen::Index p = cache.block_p[block];
  if (mu.size() != p || Sigma.rows() != p || Sigma.cols() != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1 analytic: saturated moments do not match raw block"));
  }

  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index qfull = p + pstar;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(qfull, qfull);
  std::int64_t n_block = 0;

  for (const FIMLPattern& pat : cache.patterns) {
    if (pat.block != block) continue;
    n_block += pat.n_obs;
    const auto& obs = pat.observed;
    const Eigen::Index qobs = static_cast<Eigen::Index>(obs.size());
    const double nr = static_cast<double>(pat.n_obs);

    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "FIML robust H1 analytic: Sigma_oo is not positive definite"));
    }
    Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(qobs, qobs));
    A = 0.5 * (A + A.transpose()).eval();
    const Eigen::VectorXd Mu_o = select_vector(mu, obs);
    const Eigen::VectorXd dbar = pat.mean - Mu_o;

    const Eigen::VectorXd Zsum = nr * (A * dbar);
    const Eigen::MatrixXd sum_ddT = nr * (pat.cov + dbar * dbar.transpose());
    const Eigen::MatrixXd M = A * sum_ddT * A;

    for (Eigen::Index j = 0; j < qobs; ++j) {
      const Eigen::Index fj = obs[static_cast<std::size_t>(j)];
      for (Eigen::Index i = 0; i < qobs; ++i) {
        const Eigen::Index fi = obs[static_cast<std::size_t>(i)];
        H(fi, fj) += 2.0 * nr * A(i, j);
      }
    }

    std::vector<SigmaBasis> basis;
    basis.reserve(static_cast<std::size_t>(qobs * (qobs + 1) / 2));
    for (Eigen::Index cj = 0; cj < qobs; ++cj) {
      const Eigen::Index full_c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = cj; ri < qobs; ++ri) {
        const Eigen::Index full_r0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index full_r = std::max(full_r0, full_c);
        const Eigen::Index full_l = std::min(full_r0, full_c);

        SigmaBasis e;
        e.full_index = p + vech_index(p, full_r, full_l);
        e.a = ri;
        e.b = cj;

        // Σ_r A·D_e·z_r = A·D_e·Zsum, with D_e·Zsum having ≤2 nonzeros.
        Eigen::VectorXd ADz;
        if (ri == cj) {
          ADz = A.col(ri) * Zsum(ri);
        } else {
          ADz = A.col(ri) * Zsum(cj) + A.col(cj) * Zsum(ri);
        }
        for (Eigen::Index i = 0; i < qobs; ++i) {
          const Eigen::Index full_i = obs[static_cast<std::size_t>(i)];
          const double h = 2.0 * ADz(i);
          H(full_i, e.full_index) += h;
          H(e.full_index, full_i) += h;
        }
        basis.push_back(e);
      }
    }

    for (std::size_t kk = 0; kk < basis.size(); ++kk) {
      const SigmaBasis& k = basis[kk];
      for (std::size_t ll = 0; ll <= kk; ++ll) {
        const SigmaBasis& l = basis[ll];
        // Σ_r [−tr(D_l A D_k A) + z_rᵀD_l A D_k z_r + z_rᵀD_k A D_l z_r]
        const double h = -nr * basis_trace_xy(A, A, l, k) +
                         basis_trace_xy(A, M, l, k) +
                         basis_trace_xy(A, M, k, l);
        H(k.full_index, l.full_index) += h;
        if (kk != ll) H(l.full_index, k.full_index) += h;
      }
    }
  }

  if (n_block <= 0) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "FIML robust H1 analytic: block has no observations"));
  }
  H /= static_cast<double>(n_block);
  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

post_expected<WeightedPatternBlock>
weighted_pattern_block(const RawData& raw,
                       std::size_t block,
                       Eigen::Index perturb_row,
                       double perturb_delta) {
  if (block >= raw.X.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: block index out of range"));
  }
  if (!raw.mask.empty() && block >= raw.mask.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: mask block index out of range"));
  }
  if (!(perturb_delta > -1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: case perturbation gives non-positive weight"));
  }

  const Eigen::MatrixXd& X = raw.X[block];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (perturb_row < 0 || perturb_row >= n) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: row index out of range"));
  }

  struct Accum {
    double w = 0.0;
    Eigen::VectorXd sum;
    Eigen::MatrixXd sumsq;
  };
  std::unordered_map<std::vector<Eigen::Index>, Accum, IndexVectorHash> acc;
  Eigen::VectorXd row_weight(n);
  std::vector<Eigen::Index> obs;
  for (Eigen::Index r = 0; r < n; ++r) {
    obs.clear();
    obs.reserve(static_cast<std::size_t>(p));
    if (raw.mask.empty()) {
      for (Eigen::Index c = 0; c < p; ++c) obs.push_back(c);
    } else {
      for (Eigen::Index c = 0; c < p; ++c) {
        if (raw.mask[block](r, c) != 0) obs.push_back(c);
      }
    }
    if (obs.empty() || !finite_observed_row(X, obs, r)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: invalid observed row"));
    }

    const double w = 1.0 + ((r == perturb_row) ? perturb_delta : 0.0);
    if (!(w > 0.0) || !std::isfinite(w)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: non-positive row weight"));
    }
    row_weight(r) = w;

    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    auto [it, inserted] = acc.try_emplace(obs);
    if (inserted) {
      it->second.sum = Eigen::VectorXd::Zero(q);
      it->second.sumsq = Eigen::MatrixXd::Zero(q, q);
    }
    Eigen::VectorXd x(q);
    for (Eigen::Index j = 0; j < q; ++j) {
      x(j) = X(r, obs[static_cast<std::size_t>(j)]);
    }
    it->second.w += w;
    it->second.sum.noalias() += w * x;
    it->second.sumsq.noalias() += w * (x * x.transpose());
  }

  WeightedPatternBlock out;
  out.block = block;
  out.p = p;
  out.row_weight = std::move(row_weight);
  out.n_weight = out.row_weight.sum();
  if (!(out.n_weight > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: empty weighted block"));
  }
  out.patterns.reserve(acc.size());
  for (auto& [key, a] : acc) {
    if (!(a.w > 0.0)) continue;
    Eigen::VectorXd mean = a.sum / a.w;
    Eigen::MatrixXd cov = a.sumsq / a.w - mean * mean.transpose();
    cov = 0.5 * (cov + cov.transpose()).eval();
    out.patterns.push_back(WeightedFIMLPattern{
        block, std::move(key), a.w, std::move(mean), std::move(cov)});
  }
  return out;
}

post_expected<SampleStats>
weighted_start_sample_stats_block(const RawData& raw,
                                  const WeightedPatternBlock& wp) {
  const Eigen::MatrixXd& X = raw.X[wp.block];
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (wp.row_weight.size() != n || wp.p != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: weighted start shape mismatch"));
  }

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd count = Eigen::VectorXd::Zero(p);
  for (Eigen::Index r = 0; r < n; ++r) {
    const double w = wp.row_weight(r);
    for (Eigen::Index c = 0; c < p; ++c) {
      const bool observed = raw.mask.empty() || raw.mask[wp.block](r, c) != 0;
      if (!observed) continue;
      mean(c) += w * X(r, c);
      count(c) += w;
    }
  }
  for (Eigen::Index c = 0; c < p; ++c) {
    if (!(count(c) > 0.0)) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: column has no weighted observed values"));
    }
    mean(c) /= count(c);
  }

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(p, p);
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j; i < p; ++i) {
      double acc = 0.0;
      double nij = 0.0;
      for (Eigen::Index r = 0; r < n; ++r) {
        const bool oi = raw.mask.empty() || raw.mask[wp.block](r, i) != 0;
        const bool oj = raw.mask.empty() || raw.mask[wp.block](r, j) != 0;
        if (!oi || !oj) continue;
        const double w = wp.row_weight(r);
        acc += w * (X(r, i) - mean(i)) * (X(r, j) - mean(j));
        nij += w;
      }
      if (!(nij > 0.0)) {
        return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
            "ML2S Gamma influence: variable pair has no weighted joint "
            "observations"));
      }
      S(i, j) = acc / nij;
      S(j, i) = S(i, j);
    }
  }

  SampleStats out;
  out.mean.push_back(std::move(mean));
  out.S.push_back(std::move(S));
  out.n_obs.push_back(static_cast<std::int64_t>(n));
  return out;
}

fit_expected<double>
weighted_h1_em_step_block(const WeightedPatternBlock& wp,
                          const Eigen::VectorXd& mu,
                          const Eigen::MatrixXd& Sigma,
                          Eigen::VectorXd& mu_next,
                          Eigen::MatrixXd& Sigma_next) {
  const Eigen::Index p = wp.p;
  Eigen::VectorXd sum_x = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd sum_xx = Eigen::MatrixXd::Zero(p, p);
  double n_block = 0.0;
  double f = 0.0;

  for (const WeightedFIMLPattern& pat : wp.patterns) {
    n_block += pat.n_weight;
    const auto& obs = pat.observed;
    const std::vector<Eigen::Index> miss = missing_indices(p, obs);
    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    const Eigen::Index m = static_cast<Eigen::Index>(miss.size());

    Eigen::VectorXd avg_x = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd avg_xx = Eigen::MatrixXd::Zero(p, p);
    for (Eigen::Index i = 0; i < q; ++i) {
      const Eigen::Index ri = obs[static_cast<std::size_t>(i)];
      avg_x(ri) = pat.mean(i);
      for (Eigen::Index j = 0; j < q; ++j) {
        const Eigen::Index cj = obs[static_cast<std::size_t>(j)];
        avg_xx(ri, cj) = pat.cov(i, j) + pat.mean(i) * pat.mean(j);
      }
    }

    const Eigen::MatrixXd Sigma_oo = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_oo);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "ML2S Gamma influence: saturated observed-pattern covariance is "
          "not positive definite"));
    }
    const Eigen::VectorXd mu_o = select_vector(mu, obs);
    const Eigen::VectorXd d = pat.mean - mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();
    const double scale = pat.n_weight / wp.n_weight;

    if (m > 0) {
      const Eigen::MatrixXd Sigma_mo = select_rect(Sigma, miss, obs);
      const Eigen::MatrixXd Sigma_om = Sigma_mo.transpose();
      const Eigen::MatrixXd Sigma_mm = select_square(Sigma, miss);
      const Eigen::MatrixXd Sigma_oo_inv =
          llt.solve(Eigen::MatrixXd::Identity(q, q));
      f += scale * (log_det_from_llt(llt) + Sigma_oo_inv.cwiseProduct(A).sum());
      const Eigen::MatrixXd B = Sigma_mo * Sigma_oo_inv;
      const Eigen::MatrixXd C = Sigma_mm - B * Sigma_om;

      const Eigen::VectorXd mu_m = select_vector(mu, miss);
      const Eigen::VectorXd mbar = mu_m + B * d;
      const Eigen::MatrixXd mcov = B * pat.cov * B.transpose();
      const Eigen::MatrixXd m2 = C + mcov + mbar * mbar.transpose();
      const Eigen::MatrixXd cross =
          pat.mean * mbar.transpose() + pat.cov * B.transpose();

      for (Eigen::Index i = 0; i < m; ++i) {
        const Eigen::Index ri = miss[static_cast<std::size_t>(i)];
        avg_x(ri) = mbar(i);
        for (Eigen::Index j = 0; j < m; ++j) {
          const Eigen::Index cj = miss[static_cast<std::size_t>(j)];
          avg_xx(ri, cj) = m2(i, j);
        }
      }
      for (Eigen::Index i = 0; i < q; ++i) {
        const Eigen::Index oi = obs[static_cast<std::size_t>(i)];
        for (Eigen::Index j = 0; j < m; ++j) {
          const Eigen::Index mj = miss[static_cast<std::size_t>(j)];
          avg_xx(oi, mj) = cross(i, j);
          avg_xx(mj, oi) = cross(i, j);
        }
      }
    } else {
      f += scale * (log_det_from_llt(llt) + llt.solve(A).trace());
    }

    sum_x.noalias() += pat.n_weight * avg_x;
    sum_xx.noalias() += pat.n_weight * avg_xx;
  }

  if (!(n_block > 0.0) || !(wp.n_weight > 0.0)) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "ML2S Gamma influence: weighted block has no observations"));
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_fit_err(FitError::Kind::NonFiniteObjective,
        "ML2S Gamma influence: H1 objective evaluated to non-finite"));
  }
  mu_next = sum_x / n_block;
  Sigma_next = sum_xx / n_block - mu_next * mu_next.transpose();
  Sigma_next = 0.5 * (Sigma_next + Sigma_next.transpose());
  return f;
}

fit_expected<void>
weighted_h1_em_iterate_block(const WeightedPatternBlock& wp,
                             Eigen::VectorXd& mu,
                             Eigen::MatrixXd& Sigma) {
  Eigen::VectorXd mu_next;
  Eigen::MatrixXd Sigma_next;
  double prev = std::numeric_limits<double>::infinity();
  for (int iter = 0; iter < 10000; ++iter) {
    auto step = weighted_h1_em_step_block(wp, mu, Sigma, mu_next, Sigma_next);
    if (!step.has_value()) return std::unexpected(step.error());
    const double cur = *step;
    if (std::isfinite(prev) &&
        std::abs(prev - cur) <= 1e-11 * (1.0 + std::abs(cur))) {
      return {};
    }
    prev = cur;
    mu.swap(mu_next);
    Sigma.swap(Sigma_next);
  }
  return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
      "ML2S Gamma influence: weighted H1 EM did not converge"));
}

post_expected<Eigen::MatrixXd>
weighted_saturated_hessian_analytic_block(const WeightedPatternBlock& wp,
                                          const Eigen::VectorXd& mu,
                                          const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = wp.p;
  if (mu.size() != p || Sigma.rows() != p || Sigma.cols() != p) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: saturated moments do not match raw block"));
  }

  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index qfull = p + pstar;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(qfull, qfull);
  double n_block = 0.0;

  for (const WeightedFIMLPattern& pat : wp.patterns) {
    n_block += pat.n_weight;
    const auto& obs = pat.observed;
    const Eigen::Index qobs = static_cast<Eigen::Index>(obs.size());
    const double nr = pat.n_weight;

    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ML2S Gamma influence: Sigma_oo is not positive definite"));
    }
    Eigen::MatrixXd A = llt.solve(Eigen::MatrixXd::Identity(qobs, qobs));
    A = 0.5 * (A + A.transpose()).eval();
    const Eigen::VectorXd Mu_o = select_vector(mu, obs);
    const Eigen::VectorXd dbar = pat.mean - Mu_o;

    const Eigen::VectorXd Zsum = nr * (A * dbar);
    const Eigen::MatrixXd sum_ddT = nr * (pat.cov + dbar * dbar.transpose());
    const Eigen::MatrixXd M = A * sum_ddT * A;

    for (Eigen::Index j = 0; j < qobs; ++j) {
      const Eigen::Index fj = obs[static_cast<std::size_t>(j)];
      for (Eigen::Index i = 0; i < qobs; ++i) {
        const Eigen::Index fi = obs[static_cast<std::size_t>(i)];
        H(fi, fj) += 2.0 * nr * A(i, j);
      }
    }

    std::vector<SigmaBasis> basis;
    basis.reserve(static_cast<std::size_t>(qobs * (qobs + 1) / 2));
    for (Eigen::Index cj = 0; cj < qobs; ++cj) {
      const Eigen::Index full_c = obs[static_cast<std::size_t>(cj)];
      for (Eigen::Index ri = cj; ri < qobs; ++ri) {
        const Eigen::Index full_r0 = obs[static_cast<std::size_t>(ri)];
        const Eigen::Index full_r = std::max(full_r0, full_c);
        const Eigen::Index full_l = std::min(full_r0, full_c);

        SigmaBasis e;
        e.full_index = p + vech_index(p, full_r, full_l);
        e.a = ri;
        e.b = cj;

        Eigen::VectorXd ADz;
        if (ri == cj) {
          ADz = A.col(ri) * Zsum(ri);
        } else {
          ADz = A.col(ri) * Zsum(cj) + A.col(cj) * Zsum(ri);
        }
        for (Eigen::Index i = 0; i < qobs; ++i) {
          const Eigen::Index full_i = obs[static_cast<std::size_t>(i)];
          const double h = 2.0 * ADz(i);
          H(full_i, e.full_index) += h;
          H(e.full_index, full_i) += h;
        }
        basis.push_back(e);
      }
    }

    for (std::size_t kk = 0; kk < basis.size(); ++kk) {
      const SigmaBasis& k = basis[kk];
      for (std::size_t ll = 0; ll <= kk; ++ll) {
        const SigmaBasis& l = basis[ll];
        const double h = -nr * basis_trace_xy(A, A, l, k) +
                         basis_trace_xy(A, M, l, k) +
                         basis_trace_xy(A, M, k, l);
        H(k.full_index, l.full_index) += h;
        if (kk != ll) H(l.full_index, k.full_index) += h;
      }
    }
  }

  if (!(n_block > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: weighted block has no observations"));
  }
  H /= n_block;
  return Eigen::MatrixXd(0.5 * (H + H.transpose()));
}

post_expected<Eigen::MatrixXd>
weighted_two_stage_gamma_block(const RawData& raw,
                               std::size_t block,
                               Eigen::Index perturb_row,
                               double perturb_delta) {
  auto wp_or = weighted_pattern_block(raw, block, perturb_row, perturb_delta);
  if (!wp_or.has_value()) return std::unexpected(wp_or.error());
  const WeightedPatternBlock& wp = *wp_or;

  auto start_or = weighted_start_sample_stats_block(raw, wp);
  if (!start_or.has_value()) return std::unexpected(start_or.error());
  const Eigen::VectorXd x0 = h1_start_for_block(*start_or, 0);
  Eigen::VectorXd mu;
  Eigen::MatrixXd L;
  Eigen::MatrixXd Sigma;
  h1_decode(x0, wp.p, mu, L, Sigma);

  auto em_or = weighted_h1_em_iterate_block(wp, mu, Sigma);
  if (!em_or.has_value()) {
    return std::unexpected(fit_to_post(em_or.error(),
        "ML2S Gamma influence: weighted H1 EM"));
  }

  auto scores_or = fiml_saturated_scores_block(raw, block, mu, Sigma);
  if (!scores_or.has_value()) return std::unexpected(scores_or.error());
  auto Hdev_or = weighted_saturated_hessian_analytic_block(wp, mu, Sigma);
  if (!Hdev_or.has_value()) return std::unexpected(Hdev_or.error());

  const Eigen::Index q = wp.p + vech_len(wp.p);
  const Eigen::MatrixXd H = (wp.n_weight / 2.0) * (*Hdev_or);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(q, q);
  for (Eigen::Index i = 0; i < scores_or->rows(); ++i) {
    const Eigen::VectorXd s = scores_or->row(i).transpose();
    J.noalias() += wp.row_weight(i) * (s * s.transpose());
  }
  J *= 0.25;

  auto Hinv_or = invert_symmetric(
      H, "ML2S Gamma influence: weighted saturated information");
  if (!Hinv_or.has_value()) return std::unexpected(Hinv_or.error());
  Eigen::MatrixXd acov = (*Hinv_or) * J * (*Hinv_or);
  acov = 0.5 * (acov + acov.transpose()).eval();
  Eigen::MatrixXd gamma = wp.n_weight * acov;
  gamma = 0.5 * (gamma + gamma.transpose()).eval();
  if (!gamma.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: non-finite weighted Gamma"));
  }
  return gamma;
}

post_expected<Eigen::MatrixXd>
weighted_two_stage_gamma_influence_fd_block(const RawData& raw,
                                            std::size_t block,
                                            Eigen::Index row,
                                            double eps) {
  if (!(eps > 0.0 && eps < 1.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ML2S Gamma influence: finite-difference step must lie in (0, 1)"));
  }
  auto plus_or = weighted_two_stage_gamma_block(raw, block, row, eps);
  if (!plus_or.has_value()) return std::unexpected(plus_or.error());
  auto minus_or = weighted_two_stage_gamma_block(raw, block, row, -eps);
  if (!minus_or.has_value()) return std::unexpected(minus_or.error());

  const double n = static_cast<double>(raw.X[block].rows());
  Eigen::MatrixXd out = (n / (2.0 * eps)) * (*plus_or - *minus_or);
  out = 0.5 * (out + out.transpose()).eval();
  return out;
}


}  // namespace internal

}  // namespace magmaan::estimate::fiml
