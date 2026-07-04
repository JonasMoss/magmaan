#include "magmaan/estimate/frontier/noniterative_cfa.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "magmaan/error.hpp"
#include "magmaan/estimate/cfa_utils.hpp"

namespace magmaan::estimate::frontier {

using data::SampleStats;

namespace {

fit_expected<Eigen::VectorXd> num_error(std::string detail) {
  return std::unexpected(FitError{FitError::Kind::NumericIssue, std::move(detail)});
}

// Clean per-block Guttman (1952) map: covariances → (Λ_G, Φ_G, ψ_G). No clamps,
// no floors; ψ_G = diag(S − Λ_G Φ_G Λ_Gᵀ). Errors out on the interior-point
// assumption failures rather than repairing them (start_guttman.cpp keeps the
// clamps because it is only a start-value producer). See
// guttman_cfa_asymptotics.tex:104-127.
struct BlockGuttman {
  Eigen::MatrixXd Lambda;  // nvar × nfac, marker-scaled
  Eigen::MatrixXd Phi;     // nfac × nfac, latent covariance = M P M
  Eigen::VectorXd psi;     // nvar, residual variances
};

std::expected<BlockGuttman, FitError>
guttman_block(const CfaBlockLayout& L, const Eigen::MatrixXd& S) {
  const Eigen::Index nvar = L.n_observed;
  const Eigen::Index nfac = L.n_factor();

  // Loading pattern X (nvar × nfac) and the per-factor indicator rows.
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(nvar, nfac);
  std::vector<std::vector<Eigen::Index>> rows_of(static_cast<std::size_t>(nfac));
  for (const auto& load : L.loads) {
    X(load.ov_row, load.factor) = 1.0;
    rows_of[static_cast<std::size_t>(load.factor)].push_back(load.ov_row);
  }
  for (const auto& rr : rows_of) {
    if (rr.size() < 3)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "Guttman map: a factor has < 3 indicators (Spearman communality "
          "undefined)"});
  }

  // Communalities on the diagonal: C = S − diag(residual), with per-factor
  // Spearman residual variances. No clamp / no floor (the clean analytic map).
  Eigen::VectorXd resid = Eigen::VectorXd::Zero(nvar);
  for (const auto& rr : rows_of) {
    const auto k = static_cast<Eigen::Index>(rr.size());
    Eigen::MatrixXd Sf(k, k);
    for (Eigen::Index a = 0; a < k; ++a)
      for (Eigen::Index d = 0; d < k; ++d)
        Sf(a, d) = S(rr[static_cast<std::size_t>(a)], rr[static_cast<std::size_t>(d)]);
    const Eigen::VectorXd tf = theta_spearman(Sf);
    for (Eigen::Index a = 0; a < k; ++a)
      resid(rr[static_cast<std::size_t>(a)]) = tf(a);
  }
  Eigen::MatrixXd C = S;
  C.diagonal() -= resid;

  const Eigen::MatrixXd A_mat = C * X;                  // nvar × nfac  (= CX)
  const Eigen::MatrixXd Bmat = X.transpose() * A_mat;   // nfac × nfac  (= XᵀCX)

  Eigen::VectorXd dinv(nfac);
  for (Eigen::Index f = 0; f < nfac; ++f) {
    if (Bmat(f, f) <= 0.0)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "Guttman map: non-positive latent variance on the score diagonal"});
    dinv(f) = 1.0 / std::sqrt(Bmat(f, f));
  }

  Eigen::MatrixXd P(nfac, nfac);  // factor correlation, unit diagonal
  for (Eigen::Index f = 0; f < nfac; ++f)
    for (Eigen::Index g = 0; g < nfac; ++g)
      P(f, g) = Bmat(f, g) * dinv(f) * dinv(g);

  Eigen::MatrixXd Lt(nfac, nvar);  // Lᵀ: row f = (A dinv_f)ᵀ
  for (Eigen::Index f = 0; f < nfac; ++f)
    for (Eigen::Index i = 0; i < nvar; ++i) Lt(f, i) = A_mat(i, f) * dinv(f);

  Eigen::LDLT<Eigen::MatrixXd> ldlt(P);
  if (ldlt.info() != Eigen::Success)
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "Guttman map: singular factor correlation P"});
  Eigen::MatrixXd Kstar = ldlt.solve(Lt).transpose();  // nvar × nfac = L P⁻¹

  // Marker rescaling: M = diag(K*_{marker_f, f}); Λ_G = K* M⁻¹, Φ_G = M P M.
  Eigen::VectorXd ml(nfac);
  Eigen::MatrixXd Lambda = Kstar;
  for (Eigen::Index f = 0; f < nfac; ++f) {
    const std::int16_t mk = L.marker_ov[static_cast<std::size_t>(f)];
    if (mk < 0 || mk >= nvar)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "Guttman map: factor without a marker indicator"});
    ml(f) = Kstar(mk, f);
    if (std::abs(ml(f)) < 1e-10)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "Guttman map: near-zero marker loading"});
    Lambda.col(f) /= ml(f);
  }
  Eigen::MatrixXd Phi(nfac, nfac);
  for (Eigen::Index f = 0; f < nfac; ++f)
    for (Eigen::Index g = 0; g < nfac; ++g) Phi(f, g) = ml(f) * P(f, g) * ml(g);

  const Eigen::MatrixXd implied = Lambda * Phi * Lambda.transpose();
  Eigen::VectorXd psi(nvar);
  for (Eigen::Index i = 0; i < nvar; ++i) psi(i) = S(i, i) - implied(i, i);

  return BlockGuttman{std::move(Lambda), std::move(Phi), std::move(psi)};
}

// Assemble the full free-parameter vector θ from the clean block matrices by
// reading each free cell's (mat, row, col) from ev.param_locations(). Latent
// (Ψ / Λ column) indices are mapped to the layout's factor ordering. Every free
// param must be a supported CFA cell; anything else is a v1-scope error.
fit_expected<Eigen::VectorXd>
assemble_theta(const model::ModelEvaluator& ev, const CfaBlockLayout& L,
               const BlockGuttman& g) {
  const Eigen::Index nvar = L.n_observed;
  const Eigen::Index nfac = L.n_factor();

  // latent column index → layout factor ordinal.
  std::int16_t max_lat = -1;
  for (auto c : L.factor_col) max_lat = std::max(max_lat, c);
  std::vector<std::int16_t> lat_to_f(static_cast<std::size_t>(max_lat + 1), -1);
  for (Eigen::Index f = 0; f < nfac; ++f)
    lat_to_f[static_cast<std::size_t>(L.factor_col[static_cast<std::size_t>(f)])] =
        static_cast<std::int16_t>(f);

  const auto lat_f = [&](std::int16_t lat) -> std::int16_t {
    if (lat < 0 || lat >= static_cast<std::int16_t>(lat_to_f.size())) return -1;
    return lat_to_f[static_cast<std::size_t>(lat)];
  };

  const std::vector<model::ParamLocation> locs = ev.param_locations();
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(ev.n_free()));

  for (std::size_t k = 0; k < locs.size(); ++k) {
    const model::ParamLocation& loc = locs[k];
    const Eigen::Index kk = static_cast<Eigen::Index>(k);
    switch (loc.mat) {
      case model::MatId::Lambda: {
        const std::int16_t f = lat_f(loc.col);
        if (f < 0 || loc.row < 0 || loc.row >= nvar)
          return num_error("Guttman map: loading on an untracked factor/row");
        // A free marker loading means std.lv (or a free marker) identification,
        // which the marker-scaled map does not produce.
        if (L.marker_ov[static_cast<std::size_t>(f)] == loc.row)
          return num_error("Guttman map: v1 requires marker (fixed unit "
                           "loading) identification, not a free marker loading");
        theta(kk) = g.Lambda(loc.row, f);
        break;
      }
      case model::MatId::Psi: {
        const std::int16_t fr = lat_f(loc.row);
        const std::int16_t fc = lat_f(loc.col);
        if (fr < 0 || fc < 0)
          return num_error("Guttman map: latent (co)variance on an untracked "
                           "factor");
        theta(kk) = g.Phi(fr, fc);
        break;
      }
      case model::MatId::Theta: {
        if (loc.row != loc.col)
          return num_error("Guttman map: residual covariances are out of v1 "
                           "scope");
        if (loc.row < 0 || loc.row >= nvar)
          return num_error("Guttman map: residual variance row out of range");
        theta(kk) = g.psi(loc.row);
        break;
      }
      default:
        return num_error("Guttman map: structural / mean-structure parameters "
                         "are out of v1 scope");
    }
  }
  return theta;
}

// Shared entry: validate scope, extract the single block layout, run the map.
fit_expected<Eigen::VectorXd>
map_from_S(const spec::LatentStructure& pt, const model::MatrixRep& rep,
           const model::ModelEvaluator& ev, const Eigen::MatrixXd& S,
           NonIterativeEstimator which) {
  if (rep.form != model::RepForm::PureCFA)
    return num_error("non-iterative CFA: pure-CFA models only (a structural "
                     "part is out of v1 scope)");
  if (rep.dims.size() != 1)
    return num_error("non-iterative CFA: single-block (single-group, "
                     "single-level) models only in v1");

  const std::vector<CfaBlockLayout> layouts = cfa_block_layouts(pt, rep);
  if (layouts.empty()) return num_error("non-iterative CFA: no CFA block found");
  const CfaBlockLayout& L = layouts[0];
  if (L.n_factor() == 0) return num_error("non-iterative CFA: no factors");
  if (L.crossloadings)
    return num_error("non-iterative CFA: cross-loadings are out of v1 scope");
  if (!L.all_have_marker)
    return num_error("non-iterative CFA: every factor needs a marker in v1");
  if (S.rows() != L.n_observed || S.cols() != L.n_observed)
    return num_error("non-iterative CFA: sample covariance dimension mismatch");

  switch (which) {
    case NonIterativeEstimator::Guttman: {
      auto gb = guttman_block(L, S);
      if (!gb.has_value()) return std::unexpected(gb.error());
      return assemble_theta(ev, L, *gb);
    }
  }
  return num_error("non-iterative CFA: unknown estimator");
}

}  // namespace

fit_expected<Eigen::VectorXd>
noniterative_cfa_theta(const spec::LatentStructure& pt,
                       const model::MatrixRep& rep,
                       const model::ModelEvaluator& ev,
                       const data::SampleStats& samp,
                       NonIterativeEstimator which) {
  if (samp.S.empty()) return num_error("non-iterative CFA: empty sample stats");
  return map_from_S(pt, rep, ev, samp.S[0], which);
}

fit_expected<NonIterativeFit>
fit_noniterative_cfa(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                     const data::SampleStats& samp, NonIterativeEstimator which) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value())
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "non-iterative CFA: model evaluator build failed"});
  if (samp.S.empty()) return std::unexpected(FitError{FitError::Kind::NumericIssue,
      "non-iterative CFA: empty sample stats"});

  auto theta = noniterative_cfa_theta(pt, rep, *ev, samp, which);
  if (!theta.has_value()) return std::unexpected(theta.error());

  // Recover the clean block matrices for reporting (single block, v1).
  const std::vector<CfaBlockLayout> layouts = cfa_block_layouts(pt, rep);
  auto gb = guttman_block(layouts[0], samp.S[0]);
  if (!gb.has_value()) return std::unexpected(gb.error());

  NonIterativeFit out;
  out.theta = std::move(*theta);
  out.Lambda = {std::move(gb->Lambda)};
  out.Phi = {std::move(gb->Phi)};
  out.psi = {std::move(gb->psi)};
  return out;
}

fit_expected<Eigen::MatrixXd>
estimator_map_jacobian(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                       const model::ModelEvaluator& ev, const data::SampleStats& samp,
                       NonIterativeEstimator which, double rel_step) {
  if (samp.S.empty()) return std::unexpected(FitError{FitError::Kind::NumericIssue,
      "non-iterative CFA: empty sample stats"});
  const Eigen::MatrixXd& S0 = samp.S[0];
  const Eigen::Index p = S0.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  const auto q = static_cast<Eigen::Index>(ev.n_free());

  Eigen::MatrixXd J(q, pstar);
  Eigen::Index col = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {  // lower-triangle, column-major vech
      const double h = rel_step * std::max(std::abs(S0(r, c)), 1.0);

      Eigen::MatrixXd Sp = S0, Sm = S0;
      Sp(r, c) += h;
      Sm(r, c) -= h;
      if (r != c) {  // keep the perturbed covariance symmetric (one vech coord)
        Sp(c, r) += h;
        Sm(c, r) -= h;
      }

      auto tp = map_from_S(pt, rep, ev, Sp, which);
      if (!tp.has_value()) return std::unexpected(tp.error());
      auto tm = map_from_S(pt, rep, ev, Sm, which);
      if (!tm.has_value()) return std::unexpected(tm.error());

      J.col(col) = (*tp - *tm) / (2.0 * h);
      ++col;
    }
  }
  return J;
}

}  // namespace magmaan::estimate::frontier
