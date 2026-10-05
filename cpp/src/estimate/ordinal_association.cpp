#include "ordinal_internal.hpp"
#include "magmaan/estimate/nt.hpp"
#include <Eigen/Eigenvalues>

namespace magmaan::estimate::frontier {
using namespace detail_ordinal;

post_expected<AssociationMlIJ>
association_ml_ij(spec::LatentStructure pt, const model::MatrixRep& rep,
                  const data::OrdinalStats& stats, const Estimates& est,
                  const std::vector<std::int8_t>* row_user, bool penalized) {
  auto refuse = [](const std::string& detail,
                   PostError::Kind kind = PostError::Kind::UnsupportedInference)
      -> post_expected<AssociationMlIJ> {
    return std::unexpected(make_post_err(kind,
        "association_ml_ij: " + detail));
  };
  if (penalized) return refuse("penalized inference is unsupported");
  if (!pt.nonlinear_eq_rows.empty() || pt.has_inequality_constraints)
    return refuse("nonlinear or inequality constraints are unsupported");
  if (stats.R.empty() || stats.int_data.size() != stats.R.size() ||
      stats.n_obs.size() != stats.R.size())
    return refuse("complete all-ordinal raw rows are required");
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    if (stats.int_data[b].rows() != stats.n_obs[b] ||
        stats.int_data[b].cols() != stats.R[b].cols() ||
        stats.int_data[b].size() == 0 || stats.int_data[b].minCoeff() < 0)
      return refuse("missing data or incompatible raw rows are unsupported");
  }
  for (std::size_t row = 0; row < pt.size(); ++row)
    if (pt.op[row] == parse::Op::Threshold && pt.free[row] == 0)
      return refuse("fixed thresholds change the Stage-1 estimand");
  auto valid = validate_ordinal_association_model(pt, row_user);
  if (!valid) return refuse(valid.error().detail);
  auto prepared = prepare_ordinal_partable(pt, stats, OrdinalParameterization::Delta,
                                          nullptr, row_user);
  if (!prepared) return std::unexpected(fit_to_post(prepared.error()));
  auto layout = ordinal_association_layout(pt, rep, stats, est.theta);
  if (!layout) return std::unexpected(fit_to_post(layout.error()));
  if (!layout->theta.isApprox(est.theta, 1e-10))
    return refuse("evaluation point does not contain the Stage-1 thresholds");
  const auto& constraints = layout->constraints;
  if (constraints.A_eq.rows() > 0 &&
      (constraints.A_eq*est.theta-constraints.b_eq).lpNorm<Eigen::Infinity>() > 1e-8)
    return refuse("evaluation point violates linear constraints");
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return refuse(ev.error().detail);
  // Reject primitive PSD faces even when the observed correlation is PD.
  auto matrices = ev->assembled(est.theta);
  if (!matrices) return refuse(matrices.error().detail);
  for (const auto& block : matrices->blocks) {
    for (const auto* covariance : {&block.Theta, &block.Psi}) {
      if (covariance->rows() == 0) continue;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(*covariance);
      if (eigen.info() != Eigen::Success || eigen.eigenvalues().minCoeff() <=
          1e-10 * std::max(1.0, eigen.eigenvalues().cwiseAbs().maxCoeff()))
        return refuse("primitive covariance PSD boundary or inadmissible covariance");
    }
  }
  auto info = ordinal_association_info(*ev, *layout, est.theta);
  if (!info) return refuse(info.error().detail, PostError::Kind::InfoMatrixSingular);
  data::SampleStats sample{stats.R, {}, stats.n_obs};
  auto cache = ml_prepare(sample, model::MomentTarget::Correlation);
  if (!cache) return std::unexpected(fit_to_post(cache.error()));
  AssociationMlIJ out;
  out.coordinates = layout->constraints.K();
  const auto& K = out.coordinates;
  const Eigen::Index a = K.cols();
  auto score = [&](const Eigen::VectorXd& theta) -> post_expected<MlValueGradient> {
    auto evaluation = ev->evaluate(theta, true, false);
    if (!evaluation) return std::unexpected(make_post_err(PostError::Kind::NumericIssue, evaluation.error().detail));
    auto corr = model::correlation_evaluation(std::move(*evaluation));
    if (!corr) return std::unexpected(make_post_err(PostError::Kind::NumericIssue, corr.error().detail));
    auto result = ml_value_gradient(sample, *cache, corr->moments, corr->J_sigma);
    if (!result) return std::unexpected(fit_to_post(result.error()));
    result->value *= 0.5;
    result->gradient = (0.5 * K.transpose() * result->gradient).eval();
    return *result;
  };
  auto center = score(est.theta);
  if (!center) return std::unexpected(center.error());
  out.value = center->value;
  out.score = center->gradient;
  out.sensitivity.resize(a, a);
  // Differentiate the analytic score, not an expected-information surrogate.
  // Thus all covariance standardization and LISREL curvature is retained.
  for (Eigen::Index j = 0; j < a; ++j) {
    const double h = 1e-5 * std::max(1.0, est.theta.norm()) /
                     std::max(1.0, K.col(j).norm());
    auto plus = score(est.theta + h*K.col(j));
    auto minus = score(est.theta - h*K.col(j));
    if (!plus) return std::unexpected(plus.error());
    if (!minus) return std::unexpected(minus.error());
    out.sensitivity.col(j) = (plus->gradient-minus->gradient)/(2*h);
  }
  out.sensitivity = (0.5*(out.sensitivity+out.sensitivity.transpose())).eval();
  Eigen::LDLT<Eigen::MatrixXd> solve(out.sensitivity);
  if (a && (solve.info() != Eigen::Success || !solve.isPositive() ||
      solve.vectorD().minCoeff() <= 1e-12*solve.vectorD().cwiseAbs().maxCoeff()))
    return refuse("observed sensitivity is singular or not positive definite",
                  PostError::Kind::InfoMatrixSingular);
  auto thresholds = make_threshold_layout(pt, rep, stats);
  if (!thresholds) return std::unexpected(fit_to_post(thresholds.error()));
  auto evaluation = ev->evaluate(est.theta, true, false);
  if (!evaluation) return refuse(evaluation.error().detail);
  auto corr = model::correlation_evaluation(std::move(*evaluation));
  if (!corr) return refuse(corr.error().detail);
  out.meat = Eigen::MatrixXd::Zero(a,a);
  out.vcov = Eigen::MatrixXd::Zero(K.rows(),K.rows());
  Eigen::Index offset = 0;
  const double N = static_cast<double>(cache->n_total);
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const auto& C = corr->moments.sigma[b];
    const Eigen::Index p = C.rows(), m = p*(p-1)/2;
    Eigen::LLT<Eigen::MatrixXd> chol(C);
    if (chol.info() != Eigen::Success) return refuse("non-positive-definite correlation target");
    Eigen::MatrixXd inverse = chol.solve(Eigen::MatrixXd::Identity(p,p));
    Eigen::MatrixXd J = corr->J_sigma.middleRows(offset,vech_len(p))*K;
    Eigen::MatrixXd D(a,m);
    Eigen::Index k = 0;
    for (Eigen::Index c = 0; c < p; ++c) for (Eigen::Index r = c+1; r < p; ++r) {
      Eigen::MatrixXd E = Eigen::MatrixXd::Zero(p,p);
      E(r,c) = E(c,r) = 1;
      Eigen::MatrixXd derivative = -0.5 * inverse * E * inverse;
      Eigen::VectorXd v(vech_len(p));
      for (Eigen::Index col = 0; col < p; ++col)
        for (Eigen::Index row = col; row < p; ++row)
          v(vech_index(p,row,col)) = derivative(row,col)*(row == col ? 1.0 : 2.0);
      D.col(k++) = J.transpose()*v;
    }
    auto sampling = data::ordinal_moment_sampling_influence(stats.int_data[b],
        stats.n_levels[b], stats.thresholds[b], stats.R[b]);
    if (!sampling) return std::unexpected(sampling.error());
    Eigen::MatrixXd g = sampling->rows.rightCols(m);
    out.meat.noalias() += cache->weight[b]*D*sampling->gamma.bottomRightCorner(m,m)*D.transpose();
    Eigen::MatrixXd transport = a ? -solve.solve(D).eval() : Eigen::MatrixXd(0,m);
    Eigen::MatrixXd active = g*transport.transpose();
    Eigen::MatrixXd full = active*K.transpose();
    // A within-group threshold mean has global influence g_i / w_b.
    // Active association influences already use the global H and unweighted D.
    for (std::size_t t = 0; t < thresholds->free[b].size(); ++t) {
      const auto fr = thresholds->free[b][t];
      if (fr > 0) full.col(fr-1) += sampling->rows.col(static_cast<Eigen::Index>(t))/cache->weight[b];
    }
    out.vcov.noalias() += full.transpose()*full/(N*N);
    out.target_derivative.push_back(std::move(D));
    out.influence_active.push_back(std::move(active));
    out.influence.push_back(std::move(full));
    offset += vech_len(p);
  }
  Eigen::MatrixXd H_inv = a ? solve.solve(Eigen::MatrixXd::Identity(a,a)).eval() : Eigen::MatrixXd(0,0);
  out.vcov_active = H_inv*out.meat*H_inv.transpose()/N;
  return out;
}

namespace {
void association_references(AssociationMlTest& out) {
  using namespace robust::frontier;
  out.all = fmg_test(out.statistic, out.df, out.spectrum, {FmgMethod::All, 0, true});
  out.sb = fmg_test(out.statistic, out.df, out.spectrum, {FmgMethod::SatorraBentler, 0, true});
  out.peba4 = fmg_test(out.statistic, out.df, out.spectrum, {FmgMethod::Peba, 4, true});
}
}

post_expected<AssociationMlTest>
association_ml_global_test(spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::OrdinalStats& stats, const Estimates& est,
    const std::vector<std::int8_t>* row_user, bool penalized) {
  if (!est.association) return std::unexpected(make_post_err(PostError::Kind::UnsupportedInference,
      "association_ml_global_test: association ML estimates required"));
  auto ij = association_ml_ij(pt, rep, stats, est, row_user, penalized);
  if (!ij) return std::unexpected(ij.error());
  auto prepared = prepare_ordinal_partable(pt, stats, OrdinalParameterization::Delta,
                                          nullptr, row_user);
  if (!prepared) return std::unexpected(fit_to_post(prepared.error()));
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return std::unexpected(model_to_post(ev.error()));
  auto eval = ev->evaluate(est.theta, true, false);
  if (!eval) return std::unexpected(model_to_post(eval.error()));
  auto corr = model::correlation_evaluation(std::move(*eval));
  if (!corr) return std::unexpected(model_to_post(corr.error()));
  AssociationMlTest out;
  double N = 0;
  Eigen::Index m = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    N += static_cast<double>(stats.n_obs[b]);
    const auto p = stats.R[b].rows();
    m += p*(p-1)/2;
  }
  out.statistic = 2*N*ij->value;
  if (out.statistic < -1e-8 || !std::isfinite(out.statistic))
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "association_ml_global_test: invalid discrepancy"));
  out.statistic = std::max(0.0, out.statistic);
  out.df = static_cast<int>(m-ij->coordinates.cols());
  if (out.df < 0) return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
      "association_ml_global_test: association tangent exceeds target dimension"));
  out.metric = Eigen::MatrixXd::Zero(m,m);
  out.gamma = Eigen::MatrixXd::Zero(m,m);
  out.tangent.resize(m,ij->coordinates.cols());
  Eigen::Index offset = 0, vech_offset = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const auto& C = corr->moments.sigma[b];
    const Eigen::Index p = C.rows(), mb = p*(p-1)/2;
    const double w = static_cast<double>(stats.n_obs[b])/N;
    Eigen::MatrixXd inverse = C.inverse();
    std::vector<Eigen::MatrixXd> basis;
    for (Eigen::Index c = 0; c < p; ++c) for (Eigen::Index r = c+1; r < p; ++r) {
      Eigen::MatrixXd E = Eigen::MatrixXd::Zero(p,p);
      E(r,c) = E(c,r) = 1;
      basis.push_back(E);
      out.tangent.row(offset+static_cast<Eigen::Index>(basis.size())-1) =
          corr->J_sigma.row(vech_offset+vech_index(p,r,c))*ij->coordinates;
    }
    // q = sum_b w_b/2 * F_b. Its null local Hessian is
    // V_b[j,k] = w_b/2 tr(C_b^-1 E_j C_b^-1 E_k).
    // Cov(sqrt(N) r_b) = Gamma_b/w_b for independent strata.
    // Thus 2Nq -> z' U z, with no additional 2 or group-N factor.
    for (Eigen::Index j = 0; j < mb; ++j) for (Eigen::Index k = 0; k < mb; ++k)
      out.metric(offset+j,offset+k) = 0.5*w*(inverse*basis[static_cast<std::size_t>(j)]*inverse*basis[static_cast<std::size_t>(k)]).trace();
    auto sampling = data::ordinal_moment_sampling_influence(stats.int_data[b],
        stats.n_levels[b],stats.thresholds[b],stats.R[b]);
    if (!sampling) return std::unexpected(sampling.error());
    out.gamma.block(offset,offset,mb,mb) = sampling->gamma.bottomRightCorner(mb,mb)/w;
    offset += mb;
    vech_offset += vech_len(p);
  }
  out.residual = out.metric;
  if (out.tangent.cols()) {
    Eigen::MatrixXd VD = out.metric*out.tangent;
    Eigen::MatrixXd information = out.tangent.transpose()*VD;
    Eigen::LDLT<Eigen::MatrixXd> solve(information);
    if (solve.info() != Eigen::Success || !solve.isPositive() ||
        solve.vectorD().minCoeff() <= 1e-12*solve.vectorD().cwiseAbs().maxCoeff())
      return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
          "association_ml_global_test: singular association tangent"));
    out.residual.noalias() -= VD*solve.solve(VD.transpose());
  }
  out.residual = (0.5*(out.residual+out.residual.transpose())).eval();
  if (out.df) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> gs(out.gamma);
    if (gs.info() != Eigen::Success || gs.eigenvalues().minCoeff() < -1e-10)
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "association_ml_global_test: invalid sampling covariance"));
    Eigen::MatrixXd root = gs.eigenvectors()*gs.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    Eigen::MatrixXd reduced = root.transpose()*out.residual*root;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(reduced);
    if (spectrum.info() != Eigen::Success)
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "association_ml_global_test: spectrum decomposition failed"));
    out.spectrum = spectrum.eigenvalues().tail(out.df);
  }
  association_references(out);
  return out;
}

post_expected<AssociationMlTest>
association_ml_nested_test(spec::LatentStructure pt_alt, const model::MatrixRep& rep_alt,
    const data::OrdinalStats& stats, const Estimates& est_alt,
    spec::LatentStructure pt_null, const model::MatrixRep& rep_null,
    const Estimates& est_null, const std::vector<std::int8_t>* row_user_alt,
    const std::vector<std::int8_t>* row_user_null, bool penalized) {
  if (!est_alt.association || !est_null.association)
    return std::unexpected(make_post_err(PostError::Kind::UnsupportedInference,
        "association_ml_nested_test: association ML estimates required"));
  auto ij = association_ml_ij(pt_alt,rep_alt,stats,est_alt,row_user_alt,penalized);
  if (!ij) return std::unexpected(ij.error());
  auto null_ij = association_ml_ij(pt_null,rep_null,stats,est_null,row_user_null,penalized);
  if (!null_ij) return std::unexpected(null_ij.error());
  auto p1 = prepare_ordinal_partable(pt_alt,stats,OrdinalParameterization::Delta,nullptr,row_user_alt);
  auto p0 = prepare_ordinal_partable(pt_null,stats,OrdinalParameterization::Delta,nullptr,row_user_null);
  if (!p1) return std::unexpected(fit_to_post(p1.error()));
  if (!p0) return std::unexpected(fit_to_post(p0.error()));
  auto c1 = ordinal_association_layout(pt_alt,rep_alt,stats,est_alt.theta);
  auto c0 = ordinal_association_layout(pt_null,rep_null,stats,est_null.theta);
  if (!c1) return std::unexpected(fit_to_post(c1.error()));
  if (!c0) return std::unexpected(fit_to_post(c0.error()));
  auto embedding = robust::embed_nested_null(pt_alt,rep_alt,pt_null,rep_null,
      est_null.theta,c1->constraints,c0->constraints);
  if (!embedding) return std::unexpected(embedding.error());
  AssociationMlTest out;
  out.restriction = embedding->restriction.A;
  out.df = static_cast<int>(out.restriction.rows());
  double N = 0;
  for (auto n : stats.n_obs) N += static_cast<double>(n);
  out.statistic = 2*N*(null_ij->value-ij->value);
  if (out.statistic < -1e-7 || !std::isfinite(out.statistic))
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "association_ml_nested_test: alternative discrepancy exceeds null; refit required"));
  out.statistic = std::max(0.0,out.statistic);
  auto law = robust::compute_satorra2000_from_sandwich(ij->sensitivity,ij->meat,out.restriction);
  if (!law) return std::unexpected(law.error());
  out.C = law->C;
  out.S = law->S;
  out.spectrum = law->eigenvalues;
  association_references(out);
  return out;
}
}  // namespace magmaan::estimate::frontier
