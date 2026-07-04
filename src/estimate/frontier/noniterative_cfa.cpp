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

// latent column index → block layout's factor ordinal, precomputed per block.
std::vector<std::int16_t> lat_to_f_of(const CfaBlockLayout& L) {
  std::int16_t max_lat = -1;
  for (auto c : L.factor_col) max_lat = std::max(max_lat, c);
  std::vector<std::int16_t> lat_to_f(static_cast<std::size_t>(max_lat + 1), -1);
  for (Eigen::Index f = 0; f < L.n_factor(); ++f)
    lat_to_f[static_cast<std::size_t>(L.factor_col[static_cast<std::size_t>(f)])] =
        static_cast<std::int16_t>(f);
  return lat_to_f;
}

// Assemble the full free-parameter vector θ from the per-block clean matrices by
// reading each free cell's (mat, row, col, block) from ev.param_locations().
// Multi-group: every free parameter carries a `block` naming which group/level
// block's matrices to read. Latent (Ψ / Λ column) indices are mapped to that
// block's factor ordering. Free intercepts ν take the saturated configural
// value ν_g = m_g (the sample mean); free latent means α (scalar invariance) are
// Phase C, not v1. Any other cell (structural regression) is a v1-scope error.
fit_expected<Eigen::VectorXd>
assemble_theta(const model::ModelEvaluator& ev,
               const std::vector<CfaBlockLayout>& layouts,
               const std::vector<BlockGuttman>& blocks,
               const data::SampleStats& samp) {
  const std::size_t nblk = layouts.size();
  std::vector<std::vector<std::int16_t>> lat_to_f(nblk);
  for (std::size_t b = 0; b < nblk; ++b) lat_to_f[b] = lat_to_f_of(layouts[b]);

  const auto lat_f = [&](std::size_t b, std::int16_t lat) -> std::int16_t {
    if (lat < 0 || lat >= static_cast<std::int16_t>(lat_to_f[b].size())) return -1;
    return lat_to_f[b][static_cast<std::size_t>(lat)];
  };

  const std::vector<model::ParamLocation> locs = ev.param_locations();
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(ev.n_free()));

  for (std::size_t k = 0; k < locs.size(); ++k) {
    const model::ParamLocation& loc = locs[k];
    const Eigen::Index kk = static_cast<Eigen::Index>(k);
    const auto b = static_cast<std::size_t>(loc.block);
    if (loc.block < 0 || b >= nblk)
      return num_error("Guttman map: parameter location has out-of-range block");
    const CfaBlockLayout& L = layouts[b];
    const BlockGuttman& g = blocks[b];
    const Eigen::Index nvar = L.n_observed;
    switch (loc.mat) {
      case model::MatId::Lambda: {
        const std::int16_t f = lat_f(b, loc.col);
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
        const std::int16_t fr = lat_f(b, loc.row);
        const std::int16_t fc = lat_f(b, loc.col);
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
      case model::MatId::Nu: {
        // Saturated configural intercept ν_g = m_g. With latent means fixed at
        // 0 (the CFA meanstructure default), μ_g(θ) = ν_g = m_g exactly, so the
        // mean part of the residual vanishes and the mean moments / ν params
        // cancel in the GOF df; the mean structure only enters the SEs (Ω).
        if (loc.row < 0 || loc.row >= nvar)
          return num_error("Guttman map: intercept row out of range");
        if (b >= samp.mean.size() ||
            samp.mean[b].size() != nvar)
          return num_error("Guttman map: mean structure requested but "
                           "per-block sample means are missing / mis-sized");
        theta(kk) = samp.mean[b](loc.row);
        break;
      }
      case model::MatId::Alpha:
        return num_error("Guttman map: free latent means (true scalar "
                         "invariance) are Phase C, not v1 scope");
      default:
        return num_error("Guttman map: structural parameters are out of v1 "
                         "scope");
    }
  }
  return theta;
}

// Validate scope per block, run the Guttman map on each block's covariance, and
// assemble the stacked θ. Blocks are independent (multi-group / multi-level);
// group.equal constraints are NOT imposed here (they are handled downstream by
// the projection in robust::frontier). `samp.S[b]` pairs with `layouts[b]`.
fit_expected<Eigen::VectorXd>
map_multi(const spec::LatentStructure& pt, const model::MatrixRep& rep,
          const model::ModelEvaluator& ev, const data::SampleStats& samp,
          NonIterativeEstimator which) {
  if (rep.form != model::RepForm::PureCFA)
    return num_error("non-iterative CFA: pure-CFA models only (a structural "
                     "part is out of v1 scope)");

  const std::vector<CfaBlockLayout> layouts = cfa_block_layouts(pt, rep);
  if (layouts.empty()) return num_error("non-iterative CFA: no CFA block found");
  if (layouts.size() != samp.S.size())
    return num_error("non-iterative CFA: block / sample-stats count mismatch");

  std::vector<BlockGuttman> blocks;
  blocks.reserve(layouts.size());
  for (std::size_t b = 0; b < layouts.size(); ++b) {
    const CfaBlockLayout& L = layouts[b];
    if (L.n_factor() == 0) return num_error("non-iterative CFA: no factors");
    if (L.crossloadings)
      return num_error("non-iterative CFA: cross-loadings are out of v1 scope");
    if (!L.all_have_marker)
      return num_error("non-iterative CFA: every factor needs a marker in v1");
    if (samp.S[b].rows() != L.n_observed || samp.S[b].cols() != L.n_observed)
      return num_error("non-iterative CFA: sample covariance dimension mismatch");

    switch (which) {
      case NonIterativeEstimator::Guttman: {
        auto gb = guttman_block(L, samp.S[b]);
        if (!gb.has_value()) return std::unexpected(gb.error());
        blocks.push_back(std::move(*gb));
        break;
      }
      default:
        return num_error("non-iterative CFA: unknown estimator");
    }
  }
  return assemble_theta(ev, layouts, blocks, samp);
}

}  // namespace

fit_expected<Eigen::VectorXd>
noniterative_cfa_theta(const spec::LatentStructure& pt,
                       const model::MatrixRep& rep,
                       const model::ModelEvaluator& ev,
                       const data::SampleStats& samp,
                       NonIterativeEstimator which) {
  if (samp.S.empty()) return num_error("non-iterative CFA: empty sample stats");
  return map_multi(pt, rep, ev, samp, which);
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

  // Recover the clean per-block matrices for reporting.
  const std::vector<CfaBlockLayout> layouts = cfa_block_layouts(pt, rep);
  NonIterativeFit out;
  out.theta = std::move(*theta);
  for (std::size_t b = 0; b < layouts.size() && b < samp.S.size(); ++b) {
    auto gb = guttman_block(layouts[b], samp.S[b]);
    if (!gb.has_value()) return std::unexpected(gb.error());
    out.Lambda.push_back(std::move(gb->Lambda));
    out.Phi.push_back(std::move(gb->Phi));
    out.psi.push_back(std::move(gb->psi));
  }
  return out;
}

fit_expected<Eigen::MatrixXd>
estimator_map_jacobian_block(const spec::LatentStructure& pt,
                             const model::MatrixRep& rep,
                             const model::ModelEvaluator& ev,
                             const data::SampleStats& samp,
                             NonIterativeEstimator which, std::size_t block,
                             double rel_step) {
  if (block >= samp.S.size())
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "non-iterative CFA: Jacobian block index out of range"});
  const Eigen::MatrixXd S0 = samp.S[block];  // copy: `work` is perturbed in place
  const Eigen::Index p = S0.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  const auto q = static_cast<Eigen::Index>(ev.n_free());

  // Perturbing only block `block`'s covariance leaves every other block's map
  // output identical, so the rows of J for other blocks' parameters are exactly
  // zero. We return the full q × p*_block Jacobian; the inference layer slices
  // out the rows whose param_location.block == block.
  data::SampleStats work = samp;
  Eigen::MatrixXd J(q, pstar);
  Eigen::Index col = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {  // lower-triangle, column-major vech
      const double h = rel_step * std::max(std::abs(S0(r, c)), 1.0);

      work.S[block] = S0;
      work.S[block](r, c) += h;
      if (r != c) work.S[block](c, r) += h;
      auto tp = map_multi(pt, rep, ev, work, which);
      if (!tp.has_value()) return std::unexpected(tp.error());

      work.S[block] = S0;
      work.S[block](r, c) -= h;
      if (r != c) work.S[block](c, r) -= h;
      auto tm = map_multi(pt, rep, ev, work, which);
      if (!tm.has_value()) return std::unexpected(tm.error());

      J.col(col) = (*tp - *tm) / (2.0 * h);
      ++col;
    }
  }
  return J;
}

fit_expected<Eigen::MatrixXd>
estimator_map_jacobian(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                       const model::ModelEvaluator& ev, const data::SampleStats& samp,
                       NonIterativeEstimator which, double rel_step) {
  if (samp.S.empty()) return std::unexpected(FitError{FitError::Kind::NumericIssue,
      "non-iterative CFA: empty sample stats"});
  // Single-block convenience wrapper (block 0).
  return estimator_map_jacobian_block(pt, rep, ev, samp, which, 0, rel_step);
}

}  // namespace magmaan::estimate::frontier
