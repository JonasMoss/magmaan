#include "magmaan/estimate/frontier/sam.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/model_evaluator.hpp"

#include "detail_sam_partition.hpp"
#include "detail_sam_se.hpp"

namespace magmaan::estimate::frontier {

namespace {

FitError mapping_error(std::string detail) {
  return FitError{FitError::Kind::NumericIssue, std::move(detail), 0, 0.0};
}

// std::vector index/size boundary casts (Eigen indexing stays signed).
inline std::size_t sz(Eigen::Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(int i) { return static_cast<std::size_t>(i); }

// Symmetric PD-projection: floor eigenvalues of a symmetric matrix at `floor`.
Eigen::MatrixXd force_pd(const Eigen::MatrixXd& A, double floor = 1e-10) {
  Eigen::MatrixXd S = 0.5 * (A + A.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
  Eigen::VectorXd d = es.eigenvalues();
  bool clipped = false;
  for (Eigen::Index i = 0; i < d.size(); ++i)
    if (d(i) < floor) { d(i) = floor; clipped = true; }
  if (!clipped) return S;
  return es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
}

}  // namespace

fit_expected<Eigen::MatrixXd>
mapping_matrix(const Eigen::MatrixXd& Lambda, const Eigen::MatrixXd& Theta,
               const Eigen::MatrixXd& S, SamMapping method) {
  const Eigen::Index p = Lambda.rows();
  const Eigen::Index m = Lambda.cols();
  if (p == 0 || m == 0) {
    return std::unexpected(mapping_error("mapping_matrix: empty Λ"));
  }
  if (m > p) {
    return std::unexpected(mapping_error(
        "mapping_matrix: more latents (" + std::to_string(m) +
        ") than indicators (" + std::to_string(p) + ")"));
  }

  // WiL = W⁻¹ Λ, where the weight W is Θ (ML), S (GLS), or I (ULS). The mapping
  // is then M = (Λᵀ W⁻¹ Λ)⁻¹ Λᵀ W⁻¹ = (Λᵀ WiL)⁻¹ WiLᵀ, which satisfies M·Λ = I.
  Eigen::MatrixXd WiL(p, m);
  if (method == SamMapping::ULS) {
    WiL = Lambda;
  } else {
    const Eigen::MatrixXd& Wsrc = (method == SamMapping::ML) ? Theta : S;
    if (Wsrc.rows() != p || Wsrc.cols() != p) {
      return std::unexpected(mapping_error(
          "mapping_matrix: weight matrix is not p×p"));
    }
    const Eigen::MatrixXd W = 0.5 * (Wsrc + Wsrc.transpose());
    Eigen::LLT<Eigen::MatrixXd> wllt(W);
    if (wllt.info() != Eigen::Success) {
      return std::unexpected(mapping_error(
          method == SamMapping::ML
              ? "mapping_matrix: Θ is not positive definite (Wall-Amemiya "
                "zero-residual case is not yet supported)"
              : "mapping_matrix: S is not positive definite"));
    }
    WiL.noalias() = wllt.solve(Lambda);
  }

  Eigen::MatrixXd MM(m, m);
  MM.noalias() = Lambda.transpose() * WiL;   // ΛᵀW⁻¹Λ, symmetric PD
  Eigen::LLT<Eigen::MatrixXd> mllt(MM);
  if (mllt.info() != Eigen::Success) {
    return std::unexpected(mapping_error(
        "mapping_matrix: ΛᵀW⁻¹Λ is singular — mapping is undefined"));
  }
  Eigen::MatrixXd M(m, p);
  M.noalias() = mllt.solve(WiL.transpose());
  return M;
}

fit_expected<SamResult>
fit_sam(spec::LatentStructure pt, const model::MatrixRep& rep,
        const spec::LatentNames& names, const data::SampleStats& samp,
        SamOptions opts) {
  namespace lv = compat::lavaan;
  if (samp.S.size() != 1) {
    return std::unexpected(mapping_error(
        "fit_sam: single-group only (got " + std::to_string(samp.S.size()) +
        " blocks)"));
  }
  const Eigen::MatrixXd& S_full = samp.S[0];
  const std::int64_t N = samp.n_obs.empty() ? 0 : samp.n_obs[0];
  const bool has_means = opts.meanstructure && !samp.mean.empty() &&
                         samp.mean[0].size() == S_full.rows();
  const Eigen::VectorXd ybar_full =
      has_means ? samp.mean[0] : Eigen::VectorXd::Zero(S_full.rows());

  // Full observed-variable names, positional to samp.S[0] columns.
  const Eigen::Index p_full = S_full.rows();
  std::vector<std::string> full_ov_names(sz(p_full));
  std::unordered_map<std::string, int> full_ov_pos;
  for (Eigen::Index k = 0; k < p_full; ++k) {
    const std::int32_t vid = pt.ov_order[sz(k)];
    full_ov_names[sz(k)] = names.var_name[sz(vid)];
    full_ov_pos[full_ov_names[sz(k)]] = static_cast<int>(k);
  }

  const lv::LavaanParTable full = lv::to_lavaan_partable(pt, names);
  detail::SamPartition part = detail::partition_model(
      full, opts.method == SamMethod::Global, opts.meanstructure);

  const int M = static_cast<int>(part.latent_order.size());
  std::unordered_map<std::string, int> lat_pos;
  for (int j = 0; j < M; ++j) lat_pos[part.latent_order[sz(j)]] = j;

  // Block-diagonal full-model Λ, Θ, ν assembled from the per-block measurement
  // fits. One mapping matrix M is then formed over them: block-diagonal for
  // ML/ULS, but dense over the full sample S for GLS (matching lavaan's local
  // SAM, where the GLS weight couples the measurement blocks through S⁻¹).
  Eigen::MatrixXd Lambda_full = Eigen::MatrixXd::Zero(p_full, M);
  Eigen::MatrixXd Theta_full = Eigen::MatrixXd::Zero(p_full, p_full);
  Eigen::VectorXd Nu_full = Eigen::VectorXd::Zero(p_full);

  SamResult out;
  out.measurement.reserve(part.mm.size());

  // Standard-error bookkeeping: a (lhs,op,rhs) -> value map of all measurement
  // and structural estimates, plus per-block free-index -> key, used to compose
  // the joint θ̂ and place the block vcovs for the twostep partition.
  const bool want_se = opts.se != SamSe::None;
  auto keyof = [](const std::string& l, parse::Op o, const std::string& r) {
    return l + "\x1f" + std::string(parse::to_string(o)) + "\x1f" + r;
  };
  std::unordered_map<std::string, double> se_value;
  std::vector<std::vector<std::string>> block_free_keys;

  for (const detail::MeasurementBlockSpec& mb : part.mm) {
    lv::ParsedLavaanParTable parsed = lv::from_lavaan_partable(mb.pt);
    auto rep_or = model::build_matrix_rep(parsed.structure, &parsed.names);
    if (!rep_or.has_value()) {
      return std::unexpected(mapping_error(
          "fit_sam: measurement-block matrix_rep failed: " +
          rep_or.error().detail));
    }
    const model::MatrixRep rep_b = std::move(*rep_or);
    const std::vector<std::string>& ov_b = rep_b.ov_names[0];
    const Eigen::Index p_b = static_cast<Eigen::Index>(ov_b.size());

    // Column indices of this block's indicators in the full observed order.
    std::vector<int> idx_b(sz(p_b));
    for (Eigen::Index k = 0; k < p_b; ++k) {
      auto it = full_ov_pos.find(ov_b[sz(k)]);
      if (it == full_ov_pos.end()) {
        return std::unexpected(mapping_error(
            "fit_sam: block indicator '" + ov_b[sz(k)] +
            "' not found in sample stats"));
      }
      idx_b[sz(k)] = it->second;
    }

    Eigen::MatrixXd S_b(p_b, p_b);
    for (Eigen::Index a = 0; a < p_b; ++a)
      for (Eigen::Index c = 0; c < p_b; ++c)
        S_b(a, c) = S_full(idx_b[sz(a)], idx_b[sz(c)]);
    data::SampleStats samp_b;
    samp_b.S = {S_b};
    samp_b.n_obs = {N};
    if (has_means) {
      Eigen::VectorXd m_b(p_b);
      for (Eigen::Index a = 0; a < p_b; ++a) m_b(a) = ybar_full(idx_b[sz(a)]);
      samp_b.mean = {m_b};
    }

    auto x0_or =
        simple_start_values(parsed.structure, rep_b, samp_b, parsed.starts);
    if (!x0_or.has_value()) return std::unexpected(x0_or.error());
    auto bounds_or = variance_bounds(parsed.structure);
    Bounds bounds = bounds_or.has_value() ? std::move(*bounds_or) : Bounds{};
    auto fit_or =
        fit_ml(parsed.structure, rep_b, samp_b, *x0_or, bounds, opts.mm_backend);
    if (!fit_or.has_value()) return std::unexpected(fit_or.error());
    const Estimates est_b = std::move(*fit_or);

    auto ev_or = model::ModelEvaluator::build(parsed.structure, rep_b);
    if (!ev_or.has_value())
      return std::unexpected(mapping_error(
          "fit_sam: measurement evaluator build failed: " +
          ev_or.error().detail));
    auto asm_or = ev_or->assembled(est_b.theta);
    if (!asm_or.has_value())
      return std::unexpected(mapping_error(
          "fit_sam: measurement assembly failed: " + asm_or.error().detail));
    const model::BlockMatrices& B0 = asm_or->blocks[0];

    // Latent columns of this block, mapped to the global latent order by name.
    const std::vector<std::string>& lv_b = rep_b.lv_names[0];
    const Eigen::Index m_b = static_cast<Eigen::Index>(lv_b.size());
    std::vector<int> lcol(sz(m_b));
    for (Eigen::Index j = 0; j < m_b; ++j) {
      auto it = lat_pos.find(lv_b[sz(j)]);
      if (it == lat_pos.end()) {
        return std::unexpected(mapping_error(
            "fit_sam: block latent '" + lv_b[sz(j)] + "' is not a user latent"));
      }
      lcol[sz(j)] = it->second;
    }

    // Scatter Λ_b, Θ_b, Nu_b into the full block-diagonal matrices.
    for (Eigen::Index k = 0; k < p_b; ++k)
      for (Eigen::Index j = 0; j < m_b; ++j)
        Lambda_full(idx_b[sz(k)], lcol[sz(j)]) = B0.Lambda(k, j);
    for (Eigen::Index a = 0; a < p_b; ++a)
      for (Eigen::Index c = 0; c < p_b; ++c)
        Theta_full(idx_b[sz(a)], idx_b[sz(c)]) = B0.Theta(a, c);
    if (has_means && B0.Nu.size() == p_b)
      for (Eigen::Index a = 0; a < p_b; ++a) Nu_full(idx_b[sz(a)]) = B0.Nu(a);

    SamMeasurementBlock smb;
    for (int j : lcol) smb.latents.push_back(j);
    for (int i : idx_b) smb.indicators.push_back(i);
    smb.estimates = est_b;
    smb.Lambda = B0.Lambda;
    smb.Theta = B0.Theta;
    smb.Nu = B0.Nu;

    if (want_se) {
      auto bi_or = inference::information_expected(parsed.structure, rep_b,
                                                   samp_b, est_b);
      if (bi_or.has_value()) {
        auto bv_or = inference::vcov(*bi_or, parsed.structure);
        if (bv_or.has_value()) smb.vcov = std::move(*bv_or);
      }
      const int bnf = static_cast<int>(est_b.theta.size());
      std::vector<std::string> bfk(sz(bnf));
      for (std::size_t r = 0; r < parsed.structure.op.size(); ++r) {
        const parse::Op o = parsed.structure.op[r];
        if (o != parse::Op::Measurement && o != parse::Op::Regression &&
            o != parse::Op::Covariance && o != parse::Op::Intercept)
          continue;
        const std::int32_t lid = parsed.structure.lhs_var[r];
        const std::int32_t rid = parsed.structure.rhs_var[r];
        const std::string k = keyof(
            lid >= 0 ? parsed.names.var_name[sz(lid)] : std::string(), o,
            rid >= 0 ? parsed.names.var_name[sz(rid)] : std::string());
        const std::int32_t f = parsed.structure.free[r];
        se_value[k] =
            f > 0 ? est_b.theta[f - 1] : parsed.structure.fixed_value[r];
        if (f > 0) bfk[sz(f - 1)] = k;
      }
      block_free_keys.push_back(std::move(bfk));
    }
    out.measurement.push_back(std::move(smb));
  }

  // One mapping matrix over the assembled full-model Λ, Θ (and S for GLS).
  auto Mmap_or =
      mapping_matrix(Lambda_full, Theta_full, S_full, opts.local.mapping);
  if (!Mmap_or.has_value()) return std::unexpected(Mmap_or.error());
  const Eigen::MatrixXd Mmap = *Mmap_or;
  out.mapping = Mmap;
  for (SamMeasurementBlock& smb : out.measurement) {
    Eigen::MatrixXd Mb(static_cast<Eigen::Index>(smb.latents.size()),
                       static_cast<Eigen::Index>(smb.indicators.size()));
    for (std::size_t j = 0; j < smb.latents.size(); ++j)
      for (std::size_t k = 0; k < smb.indicators.size(); ++k)
        Mb(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k)) =
            Mmap(smb.latents[j], smb.indicators[k]);
    smb.M = std::move(Mb);
  }

  // MSM = M S Mᵀ, MTM = M Θ Mᵀ, then the Fuller-corrected VETA = MSM − λ*·MTM.
  Eigen::MatrixXd MSM = Mmap * S_full * Mmap.transpose();
  MSM = 0.5 * (MSM + MSM.transpose());
  Eigen::MatrixXd MTM = Mmap * Theta_full * Mmap.transpose();
  MTM = 0.5 * (MTM + MTM.transpose());
  if (opts.local.alpha_correction > 0 && N > 1) {
    double a = static_cast<double>(opts.local.alpha_correction) /
               static_cast<double>(N - 1);
    a = std::min(1.0, std::max(0.0, a));
    MTM *= (1.0 - a);
  }

  double lambda_star = 1.0;
  if (opts.local.lambda_correction && N > 1) {
    const Eigen::MatrixXd MTM_pd = force_pd(MTM);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(MSM, MTM_pd);
    if (ges.info() == Eigen::Success && ges.eigenvalues().size() > 0) {
      const double lambda = ges.eigenvalues()(0);   // smallest generalized root
      const double cutoff = 1.0 + 1.0 / static_cast<double>(N - 1);
      if (lambda < cutoff)
        lambda_star = lambda - 1.0 / static_cast<double>(N - 1);
    }
  }
  Eigen::MatrixXd VETA = MSM - lambda_star * MTM;
  VETA = 0.5 * (VETA + VETA.transpose());
  out.VETA = VETA;
  out.lambda_star = (opts.local.lambda_correction && lambda_star != 1.0)
                        ? lambda_star
                        : std::numeric_limits<double>::infinity();
  out.reliability.resize(sz(M));
  for (int j = 0; j < M; ++j)
    out.reliability[sz(j)] = MSM(j, j) > 0.0 ? VETA(j, j) / MSM(j, j) : 0.0;

  Eigen::VectorXd EETA;
  if (has_means) {
    EETA = Mmap * (ybar_full - Nu_full);
    out.EETA = EETA;
  }

  // Structural step: fit the promoted-to-observed sub-model against VETA/EETA.
  lv::ParsedLavaanParTable parsed_s = lv::from_lavaan_partable(part.structural);
  auto rep_s_or = model::build_matrix_rep(parsed_s.structure, &parsed_s.names);
  if (!rep_s_or.has_value())
    return std::unexpected(mapping_error(
        "fit_sam: structural matrix_rep failed: " + rep_s_or.error().detail));
  model::MatrixRep rep_s = std::move(*rep_s_or);

  // Permute VETA/EETA into the structural observed order (by latent name).
  const std::vector<std::string>& ov_s = rep_s.ov_names[0];
  const Eigen::Index ms = static_cast<Eigen::Index>(ov_s.size());
  std::vector<int> perm(sz(ms));
  for (Eigen::Index j = 0; j < ms; ++j) {
    auto it = lat_pos.find(ov_s[sz(j)]);
    if (it == lat_pos.end())
      return std::unexpected(mapping_error(
          "fit_sam: structural variable '" + ov_s[sz(j)] +
          "' is not a user latent"));
    perm[sz(j)] = it->second;
  }
  Eigen::MatrixXd VETA_s(ms, ms);
  for (Eigen::Index a = 0; a < ms; ++a)
    for (Eigen::Index c = 0; c < ms; ++c)
      VETA_s(a, c) = VETA(perm[sz(a)], perm[sz(c)]);
  data::SampleStats samp_s;
  samp_s.S = {VETA_s};
  samp_s.n_obs = {N};
  if (has_means) {
    Eigen::VectorXd EETA_s(ms);
    for (Eigen::Index a = 0; a < ms; ++a) EETA_s(a) = EETA(perm[sz(a)]);
    samp_s.mean = {EETA_s};
  }

  auto x0s_or =
      simple_start_values(parsed_s.structure, rep_s, samp_s, parsed_s.starts);
  if (!x0s_or.has_value()) return std::unexpected(x0s_or.error());
  auto fits_or = fit_ml(parsed_s.structure, rep_s, samp_s, *x0s_or, Bounds{},
                        opts.struc_backend);
  if (!fits_or.has_value()) return std::unexpected(fits_or.error());

  out.structural = std::move(*fits_or);

  // Twostep / standard standard errors: partition the joint information.
  if (opts.se == SamSe::Twostep || opts.se == SamSe::Standard) {
    std::unordered_map<std::string, char> step2;
    for (std::size_t r = 0; r < parsed_s.structure.op.size(); ++r) {
      const parse::Op o = parsed_s.structure.op[r];
      if (o != parse::Op::Regression && o != parse::Op::Covariance &&
          o != parse::Op::Intercept)
        continue;
      const std::int32_t lid = parsed_s.structure.lhs_var[r];
      const std::int32_t rid = parsed_s.structure.rhs_var[r];
      const std::string k = keyof(
          lid >= 0 ? parsed_s.names.var_name[sz(lid)] : std::string(), o,
          rid >= 0 ? parsed_s.names.var_name[sz(rid)] : std::string());
      const std::int32_t f = parsed_s.structure.free[r];
      se_value[k] = f > 0 ? out.structural.theta[f - 1]
                          : parsed_s.structure.fixed_value[r];
      if (f > 0) step2[k] = 1;
    }

    const std::int32_t nfree = pt.n_free();
    std::vector<std::string> key_of_free(sz(nfree));
    std::vector<char> seen(sz(nfree), 0);
    std::unordered_map<std::string, int> joint_free_of_key;
    for (std::size_t r = 0; r < full.op.size(); ++r) {
      const std::int32_t f = full.free[r];
      if (f > 0 && !seen[sz(f - 1)]) {
        const std::string k = keyof(full.lhs[r], full.op[r], full.rhs[r]);
        key_of_free[sz(f - 1)] = k;
        joint_free_of_key[k] = f;
        seen[sz(f - 1)] = 1;
      }
    }

    Eigen::VectorXd theta_joint(nfree);
    std::vector<char> is_step2(sz(nfree), 0);
    bool ok = true;
    for (std::int32_t f = 0; f < nfree && ok; ++f) {
      auto it = se_value.find(key_of_free[sz(f)]);
      if (it == se_value.end()) { ok = false; break; }
      theta_joint(f) = it->second;
      is_step2[sz(f)] = step2.count(key_of_free[sz(f)]) ? 1 : 0;
    }

    if (ok) {
      Eigen::MatrixXd Sigma11_full = Eigen::MatrixXd::Zero(nfree, nfree);
      for (std::size_t bi = 0; bi < part.mm.size(); ++bi) {
        const Eigen::MatrixXd& bv = out.measurement[bi].vcov;
        const std::vector<std::string>& bfk = block_free_keys[bi];
        if (bv.size() == 0) continue;
        const int bnf = static_cast<int>(bfk.size());
        std::vector<int> b2j(sz(bnf), 0);
        for (int a = 0; a < bnf; ++a) {
          auto it = joint_free_of_key.find(bfk[sz(a)]);
          b2j[sz(a)] = it != joint_free_of_key.end() ? it->second : 0;
        }
        for (int a = 0; a < bnf; ++a)
          for (int c = 0; c < bnf; ++c)
            if (b2j[sz(a)] > 0 && b2j[sz(c)] > 0)
              Sigma11_full(b2j[sz(a)] - 1, b2j[sz(c)] - 1) = bv(a, c);
      }
      auto se_or =
          detail::sam_twostep_se(pt, rep, samp, theta_joint, is_step2,
                                 Sigma11_full, opts.se == SamSe::Standard);
      if (se_or.has_value()) {
        out.vcov = std::move(se_or->vcov);
        out.se = std::move(se_or->se);
      } else {
        return std::unexpected(se_or.error());
      }
    }
  }

  out.structural_pt = std::move(parsed_s.structure);
  out.structural_rep = std::move(rep_s);
  out.latent_samp = std::move(samp_s);
  return out;
}

}  // namespace magmaan::estimate::frontier
