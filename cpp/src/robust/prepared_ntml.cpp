#include "magmaan/robust/prepared_ntml.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/inference/score.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <Eigen/Cholesky>

#include "../detail_linalg.hpp"

namespace magmaan::robust::frontier {
namespace {
PostError invalid(const char* why) { return {PostError::Kind::NumericIssue, why}; }
Eigen::Index start(const UFactor::Block& b, bool means) { return means ? b.mu_off : b.row_offset; }
Eigen::Index width(const UFactor::Block& b, bool means) { return b.pstar + (means ? b.p : 0); }

post_expected<Eigen::MatrixXd> project(NTMLData& data, const UFactor& base,
                                      const Eigen::MatrixXd& directions) {
  ++data.projection_passes;
  if (data.storage == ContributionStorage::Tiled) {
    UFactor factor = base;
    factor.B = directions; factor.df = directions.cols();
    factor.kind = UFactor::Kind::ProjectionExpected;
    return casewise_projected_rows_tiled(factor, data.raw, data.sample);
  }
  if (!data.contributions) {
    auto rows = casewise_contributions(data.raw, data.sample, data.has_means);
    if (!rows) return std::unexpected(rows.error());
    data.contributions = std::move(*rows); ++data.contribution_builds;
  }
  return Eigen::MatrixXd(*data.contributions * directions);
}

// Convert centered sample-moment projections into likelihood projections at
// the fitted point. The mean shift adds linear and constant terms only; the
// fourth-moment contributions are reused, never reconstructed.
void likelihood_rows(const NTMLFit& fit, const UFactor& base,
                     const Eigen::MatrixXd& directions, Eigen::MatrixXd& rows,
                     const RawData* subset = nullptr) {
  const auto& raw = subset ? *subset : fit.data->raw;
  const auto map = likelihood_projection(base, fit.data->sample,
                                         fit.geometry->mean_hat, directions);
  Eigen::Index offset = 0;
  for (std::size_t b = 0; b < base.blocks.size(); ++b) {
    const auto n = raw.X[b].rows();
    if (base.has_means) {
      const Eigen::MatrixXd centered = raw.X[b].rowwise() - fit.data->sample.mean[b].transpose();
      rows.middleRows(offset,n).noalias() += centered *
          map.mean_corrections[b];
    }
    rows.middleRows(offset,n).rowwise() += map.group_means.row(static_cast<Eigen::Index>(b));
    offset += n;
  }
}

post_expected<const Eigen::MatrixXd*> weighted_delta(NTMLFit& fit) {
  if (fit.weighted_delta) return &*fit.weighted_delta;
  auto g = ntml_geometry(fit); if (!g) return std::unexpected(g.error());
  Eigen::MatrixXd out = (*g)->Delta;
  for (const auto& b : (*g)->base.blocks) {
    out.middleRows(b.row_offset,b.pstar) = b.llt_gamma_nt.solve((*g)->Delta.middleRows(b.row_offset,b.pstar));
    if ((*g)->base.has_means)
      out.middleRows(b.mu_off,b.p) = b.llt_M.solve((*g)->Delta.middleRows(b.mu_off,b.p));
  }
  fit.weighted_delta = std::move(out);
  return &*fit.weighted_delta;
}
void scale_blocks(Eigen::MatrixXd& rows, const NTMLData& data, bool inverse) {
  Eigen::Index offset = 0;
  for (const auto& x : data.raw.X) {
    const double scale = std::sqrt(static_cast<double>(x.rows()));
    rows.middleRows(offset,x.rows()) *= inverse ? 1.0 / scale : scale;
    offset += x.rows();
  }
}
post_expected<std::shared_ptr<NTMLQuadratic>> from_rows(
    Eigen::MatrixXd rows, double statistic, int df) {
  if (df <= 0 || !std::isfinite(statistic) || statistic < -1e-7 || !rows.allFinite())
    return std::unexpected(invalid("NTML quadratic: invalid statistic, degrees of freedom or rows"));
  auto q = std::make_shared<NTMLQuadratic>();
  q->statistic = std::max(0.0,statistic); q->df = df; q->rows = std::move(rows);
  return q;
}
}

LikelihoodProjection likelihood_projection(
    const UFactor& base, const SampleStats& sample,
    const std::vector<Eigen::VectorXd>& mean_hat,
    const Eigen::MatrixXd& directions) {
  LikelihoodProjection out;
  out.directions = directions;
  out.group_means = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(base.blocks.size()), directions.cols());
  out.mean_corrections.reserve(base.blocks.size());
  for (std::size_t b = 0; b < base.blocks.size(); ++b) {
    const auto& block = base.blocks[b];
    Eigen::VectorXd shift = Eigen::VectorXd::Zero(block.p);
    if (base.has_means) shift = sample.mean[b] - mean_hat[b];
    Eigen::MatrixXd linear = Eigen::MatrixXd::Zero(block.p,directions.cols());
    auto constant = out.group_means.row(static_cast<Eigen::Index>(b));
    if (base.has_means)
      constant += shift.transpose() * directions.middleRows(block.mu_off,block.p);
    Eigen::Index v = block.row_offset;
    for (Eigen::Index j = 0; j < block.p; ++j) {
      for (Eigen::Index i = j; i < block.p; ++i, ++v) {
        constant += (block.S(i,j) - block.Sigma_hat(i,j) + shift(i)*shift(j)) * directions.row(v);
        if (base.has_means) {
          linear.row(i) += shift(j)*directions.row(v);
          linear.row(j) += shift(i)*directions.row(v);
        }
      }
    }
    if (base.has_means) out.directions.middleRows(block.mu_off,block.p) += linear;
    out.mean_corrections.push_back(std::move(linear));
  }
  return out;
}

post_expected<std::shared_ptr<NTMLData>> prepare_ntml_data(
    RawData raw, bool has_means, ContributionStorage storage) {
  if (!raw.mask.empty()) return std::unexpected(invalid("NTML data: complete observations required"));
  auto pack = estimate::fiml::fiml_pack(raw);
  if (!pack) return std::unexpected(PostError{PostError::Kind::NumericIssue,pack.error().detail});
  auto out = std::make_shared<NTMLData>();
  out->raw = std::move(raw); out->pack = std::move(*pack); out->sample = out->pack.start_stats; out->has_means = has_means;
  double n = 0, m = 0;
  for (const auto& x : out->raw.X) { n += static_cast<double>(x.rows()); const double p = static_cast<double>(x.cols()); m += p*(p+1)/2 + (has_means ? p : 0); }
  out->storage = storage == ContributionStorage::Auto
      ? (n*m*sizeof(double) <= 64*1024*1024 ? ContributionStorage::Casewise : ContributionStorage::Tiled) : storage;
  return out;
}
post_expected<std::shared_ptr<NTMLFit>> prepare_ntml_fit(
    std::shared_ptr<NTMLData> data, spec::LatentStructure pt,
    model::MatrixRep rep, Estimates estimates) {
  if (!data || pt.has_inequality_constraints || !pt.nonlinear_eq_rows.empty() ||
      estimates.diagnostics.active_bounds_full.any_active() ||
      std::any_of(pt.exo.begin(),pt.exo.end(),[](auto x){return x != 0;}))
    return std::unexpected(invalid("prepared NTML: requires interior complete-data fits with affine constraints and random X"));
  auto out = std::make_shared<NTMLFit>();
  out->data = std::move(data); out->pt = std::move(pt); out->rep = std::move(rep); out->estimates = std::move(estimates);
  return out;
}
post_expected<const NTMLGeometry*> ntml_geometry(NTMLFit& fit) {
  if (!fit.geometry) {
    auto g = prepare_ntml_geometry(fit.pt,fit.rep,fit.data->sample,fit.estimates);
    if (!g) return std::unexpected(g.error());
    if (g->base.has_means != fit.data->has_means)
      return std::unexpected(invalid("prepared NTML: data and fit mean layouts differ"));
    fit.geometry = std::move(*g); ++fit.geometry_builds;
  }
  return &*fit.geometry;
}
post_expected<const UFactor*> ntml_factor(NTMLFit& fit, Information bread) {
  auto& slot = bread == Information::Expected ? fit.expected : fit.observed;
  if (!slot) {
    auto g = ntml_geometry(fit); if (!g) return std::unexpected(g.error());
    auto u = bread == Information::Expected ? ntml_u_factor(**g)
        : ntml_u_factor_observed(**g,fit.pt,fit.rep,fit.data->sample,fit.estimates);
    if (!u) return std::unexpected(u.error());
    slot = std::move(*u); ++fit.u_builds;
  }
  return &*slot;
}
namespace {
// One tiled projection supplies both the centered GOF and likelihood-score
// reductions. No N-by-df output is retained in the large-N path.
post_expected<void> global_reduced(NTMLFit& fit, const UFactor& u) {
  auto lr=std::make_shared<NTMLQuadratic>(), score=std::make_shared<NTMLQuadratic>();
  lr->df=score->df=static_cast<int>(u.df);
  lr->reduced=Eigen::MatrixXd::Zero(u.df,u.df);
  score->reduced=Eigen::MatrixXd::Zero(u.df,u.df);
  Eigen::RowVectorXd total=Eigen::RowVectorXd::Zero(u.df);
  RawData tile;
  for (const auto& x : fit.data->raw.X) tile.X.emplace_back(0,x.cols());
  ++fit.data->projection_passes;
  for (std::size_t b=0;b<tile.X.size();++b) {
    const auto& x=fit.data->raw.X[b];
    const double n=static_cast<double>(x.rows());
    for (Eigen::Index at=0;at<x.rows();at+=128) {
      const auto count=std::min<Eigen::Index>(128,x.rows()-at);
      tile.X[b]=x.middleRows(at,count);
      auto rows=casewise_projected_rows_tiled(u,tile,fit.data->sample);
      if (!rows) return std::unexpected(rows.error());
      lr->reduced->noalias() += rows->transpose()* *rows/n;
      likelihood_rows(fit,u,u.B,*rows,&tile);
      score->reduced->noalias() += rows->transpose()* *rows/n;
      total += rows->colwise().sum()/std::sqrt(n);
    }
    tile.X[b].resize(0,x.cols());
  }
  lr->statistic=inference::chi2_stat(fit.data->sample,fit.estimates);
  score->statistic=total.squaredNorm();
  fit.lr=std::move(lr); fit.score=std::move(score);
  return {};
}
}
post_expected<std::shared_ptr<NTMLQuadratic>> ntml_quadratic(NTMLFit& fit, bool score) {
  auto& slot = score ? fit.score : fit.lr;
  if (slot) return slot;
  auto u = ntml_factor(fit); if (!u) return std::unexpected(u.error());
  if (fit.data->storage == ContributionStorage::Tiled && fit.geometry->N_total >= static_cast<double>((*u)->df)) {
    auto ok=global_reduced(fit,**u); if (!ok) return std::unexpected(ok.error());
    return score ? fit.score : fit.lr;
  }
  if (!fit.projected_rows) {
    auto y = project(*fit.data,**u,(*u)->B); if (!y) return std::unexpected(y.error());
    fit.projected_rows = std::move(*y);
  }
  Eigen::MatrixXd rows = *fit.projected_rows;
  double statistic = inference::chi2_stat(fit.data->sample,fit.estimates);
  if (score) {
    likelihood_rows(fit,**u,(*u)->B,rows);
    scale_blocks(rows,*fit.data,true);
    statistic = rows.colwise().sum().squaredNorm();
  } else scale_blocks(rows,*fit.data,true);
  auto q = from_rows(std::move(rows),statistic,static_cast<int>((*u)->df));
  if (!q) return std::unexpected(q.error());
  slot = *q; return slot;
}
post_expected<const Eigen::VectorXd*> ntml_unbiased_spectrum(NTMLFit& fit) {
  if (!fit.unbiased_spectrum) {
    auto u = ntml_factor(fit); if (!u) return std::unexpected(u.error());
    auto& d = *fit.data;
    if (!d.contributions) {
      auto z = casewise_contributions(d.raw,d.sample,d.has_means);
      if (!z) return std::unexpected(z.error());
      d.contributions = std::move(*z); ++d.contribution_builds;
    }
    Eigen::VectorXd denom(static_cast<Eigen::Index>(d.sample.n_obs.size()));
    for (Eigen::Index i=0;i<denom.size();++i) denom(i)=static_cast<double>(d.sample.n_obs[static_cast<std::size_t>(i)]);
    auto m = reduced_gamma_unbiased_casewise(**u,d.sample,*d.contributions,denom);
    if (!m) return std::unexpected(m.error());
    auto e = ugamma_eigenvalues(*m); if (!e) return std::unexpected(e.error());
    fit.unbiased_spectrum = std::move(*e);
  }
  return &*fit.unbiased_spectrum;
}
post_expected<const Eigen::VectorXd*> ntml_spectrum(NTMLQuadratic& q) {
  if (!q.eigenvalues) {
    q.row_space = q.rows.rows() < q.rows.cols();
    Eigen::MatrixXd M;
    if (q.row_space) M = q.rows*q.rows.transpose();
    else { if (!q.reduced) q.reduced = q.rows.transpose()*q.rows; M = *q.reduced; }
    auto eigen = ugamma_eigenvalues(M); if (!eigen) return std::unexpected(eigen.error());
    Eigen::VectorXd values = Eigen::VectorXd::Zero(q.df);
    values.tail(eigen->size()) = *eigen;
    std::sort(values.data(),values.data()+values.size());
    q.eigenvalues = std::move(values); ++q.spectrum_builds;
  }
  return &*q.eigenvalues;
}
post_expected<const Eigen::MatrixXd*> ntml_information(NTMLFit& fit) {
  if (!fit.information) {
    auto wd = weighted_delta(fit); if (!wd) return std::unexpected(wd.error());
    Eigen::MatrixXd I = Eigen::MatrixXd::Zero((*wd)->cols(),(*wd)->cols());
    for (const auto& b : fit.geometry->base.blocks) {
      const auto off = start(b,fit.geometry->base.has_means), size = width(b,fit.geometry->base.has_means);
      I.noalias() += static_cast<double>(b.n_obs) * fit.geometry->Delta.middleRows(off,size).transpose() * (*wd)->middleRows(off,size);
    }
    fit.information = 0.5*(I+I.transpose()).eval(); ++fit.information_builds;
  }
  return &*fit.information;
}
post_expected<const Eigen::MatrixXd*> ntml_covariance(NTMLFit& fit, bool robust) {
  if (!fit.covariance) {
    auto info = ntml_information(fit); if (!info) return std::unexpected(info.error());
    auto cov = inference::vcov(**info,fit.pt,fit.estimates.theta);
    if (!cov) return std::unexpected(cov.error());
    fit.covariance = std::move(*cov);
  }
  if (!robust) return &*fit.covariance;
  if (!fit.robust_covariance) {
    auto wd = weighted_delta(fit); if (!wd) return std::unexpected(wd.error());
    auto y = project(*fit.data,fit.geometry->base,**wd * *fit.covariance);
    if (!y) return std::unexpected(y.error());
    fit.robust_covariance = y->transpose()* *y;
  }
  return &*fit.robust_covariance;
}
post_expected<const Eigen::MatrixXd*> ntml_observed_information(NTMLFit& fit) {
  if (!fit.observed_information) {
    auto H = inference::information_observed_analytic(fit.pt,fit.rep,fit.data->sample,fit.estimates);
    if (!H) return std::unexpected(H.error());
    if (!H->allFinite()) return std::unexpected(invalid("NTML observed information: non-finite Hessian"));
    fit.observed_information = 0.5*(*H+H->transpose()).eval();
  }
  return &*fit.observed_information;
}
post_expected<const Eigen::MatrixXd*> ntml_observed_covariance(NTMLFit& fit) {
  if (!fit.observed_covariance) {
    auto H = ntml_observed_information(fit);
    if (!H) return std::unexpected(H.error());
    auto V = inference::vcov(**H,fit.pt,fit.estimates.theta);
    if (!V) return std::unexpected(V.error());
    fit.observed_covariance = std::move(*V);
  }
  return &*fit.observed_covariance;
}
post_expected<const Eigen::MatrixXd*> ntml_score_sandwich(NTMLFit& fit, Information bread) {
  auto& slot = bread == Information::Expected ? fit.score_sandwich_expected : fit.score_sandwich_observed;
  if (slot) return &*slot;
  auto g = ntml_geometry(fit); if (!g) return std::unexpected(g.error());
  auto wd = weighted_delta(fit); if (!wd) return std::unexpected(wd.error());
  auto V = bread == Information::Expected ? ntml_covariance(fit) : ntml_observed_covariance(fit);
  if (!V) return std::unexpected(V.error());
  const Eigen::MatrixXd directions = **wd * **V;
  auto rows = project(*fit.data,(*g)->base,directions);
  if (!rows) return std::unexpected(rows.error());
  likelihood_rows(fit,(*g)->base,directions,*rows);
  if (!rows->allFinite()) return std::unexpected(invalid("NTML score sandwich: non-finite casewise scores"));
  slot = rows->transpose()* *rows;
  return &*slot;
}
post_expected<std::shared_ptr<NTMLHypothesis>> prepare_ntml_hypothesis(
    std::shared_ptr<NTMLFit> null_fit, std::shared_ptr<NTMLFit> alternative) {
  if (!null_fit || !alternative || null_fit->data != alternative->data)
    return std::unexpected(PostError{PostError::Kind::NotNested,
        "NTML hypothesis: fits must share one prepared inference dataset"});
  auto c0 = build_eq_constraints(null_fit->pt), c1 = build_eq_constraints(alternative->pt);
  if (!c0) return std::unexpected(c0.error());
  if (!c1) return std::unexpected(c1.error());
  auto embedding = embed_nested_null(alternative->pt,alternative->rep,
      null_fit->pt,null_fit->rep,null_fit->estimates.theta,*c1,*c0,true,
      &alternative->estimates.theta);
  if (!embedding) return std::unexpected(embedding.error());
  if (!embedding->restriction.A.rows()) return std::unexpected(PostError{PostError::Kind::NotNested,
        "NTML hypothesis: no restrictions released"});
  auto out = std::make_shared<NTMLHypothesis>();
  out->null_fit = std::move(null_fit); out->alternative = std::move(alternative);
  out->restriction = std::move(embedding->restriction);
  if (!embedding->same_ambient) {
    auto estimates=out->null_fit->estimates;
    estimates.theta=std::move(embedding->theta);
    auto evaluated=prepare_ntml_fit(out->null_fit->data,
        embedded_null_structure(out->alternative->pt,embedding->null_constraints),
        out->alternative->rep,std::move(estimates));
    if (!evaluated) return std::unexpected(evaluated.error());
    out->embedded_null=std::move(*evaluated);
  }
  return out;
}
post_expected<std::shared_ptr<NTMLQuadratic>> ntml_quadratic(
    NTMLHypothesis& h, bool score, Information geometry) {
  const bool observed = geometry == Information::Observed;
  auto& slot = observed ? (score ? h.score_observed : h.lr_observed)
                        : (score ? h.score : h.lr);
  if (slot) return slot;
  // The per-null LR cache is keyed by the alternative only, so it holds the
  // expected geometry alone.
  if (!score && !observed) for (const auto& entry : h.null_fit->nested_lr)
    if (entry.first == h.alternative) { slot=entry.second; return slot; }
  NTMLFit& fit = score ? *(h.embedded_null ? h.embedded_null : h.null_fit) : *h.alternative;
  auto info = ntml_information(fit); if (!info) return std::unexpected(info.error());
  auto wd = weighted_delta(fit); if (!wd) return std::unexpected(wd.error());
  auto c1 = build_eq_constraints(h.alternative->pt); if (!c1) return std::unexpected(c1.error());
  const Eigen::MatrixXd K = c1->K();
  // Sensitivity used to project (score) or reduce (LR): the expected or the
  // observed information of the alternative's model at `fit`.
  const Eigen::MatrixXd* sensitivity = *info;
  if (observed) {
    auto H = ntml_observed_information(fit); if (!H) return std::unexpected(H.error());
    sensitivity = *H;
  }
  Eigen::MatrixXd directions, metric;
  if (score) {
    // Efficient-score directions: remove the null tangent K0 with the
    // sensitivity S, D - K0 (K0' S K0)^-1 K0' S D, where the constrained
    // inverse supplies K0 (K0' S K0)^-1 K0'. The metric stays expected.
    auto covariance = observed ? ntml_observed_covariance(fit) : ntml_covariance(fit);
    if (!covariance) return std::unexpected(covariance.error());
    Eigen::MatrixXd D = K*h.restriction.A.transpose();
    D -= **covariance * *sensitivity * D;
    metric = D.transpose()* **info * D;
    directions = **wd * D;
  } else {
    const Eigen::MatrixXd P = K.transpose()* *sensitivity * K;
    const auto normalized_P = detail::equilibrate_symmetric(P);
    Eigen::LLT<Eigen::MatrixXd> factor(normalized_P.matrix);
    if (factor.info() != Eigen::Success || factor.rcond() < 1e-12)
      return std::unexpected(invalid(observed
          ? "NTML LR: alternative observed information is not positive definite on its free directions"
          : "NTML LR: singular alternative information"));
    const Eigen::MatrixXd Y = normalized_P.scale.asDiagonal() *
        factor.solve(normalized_P.scale.asDiagonal() * h.restriction.A.transpose());
    metric = h.restriction.A * Y;
    directions = **wd * K * Y;
  }
  metric = 0.5*(metric+metric.transpose()).eval();
  const auto normalized_metric = detail::equilibrate_symmetric(metric);
  Eigen::LLT<Eigen::MatrixXd> factor(normalized_metric.matrix);
  if (factor.info() != Eigen::Success || factor.rcond() < 1e-12)
    return std::unexpected(invalid("NTML hypothesis: singular restriction metric"));
  auto rows = project(*fit.data,fit.geometry->base,directions);
  if (!rows) return std::unexpected(rows.error());
  // LR uses likelihood scores at the alternative, including a restricted
  // fitted mean's shift; score uses the embedded null evaluation point.
  likelihood_rows(fit,fit.geometry->base,directions,*rows);
  Eigen::MatrixXd whitened = factor.matrixL().solve(
      normalized_metric.scale.asDiagonal() * rows->transpose()).transpose();
  const double statistic = score ? whitened.colwise().sum().squaredNorm()
      : inference::chi2_stat(fit.data->sample,h.null_fit->estimates) -
        inference::chi2_stat(fit.data->sample,h.alternative->estimates);
  auto q = from_rows(std::move(whitened),statistic,static_cast<int>(h.restriction.A.rows()));
  if (!q) return std::unexpected(q.error());
  slot = *q;
  if (!score && !observed) h.null_fit->nested_lr.emplace_back(h.alternative,slot);
  return slot;
}
} // namespace magmaan::robust::frontier
