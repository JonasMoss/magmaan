#include "fiml_internal.hpp"

namespace magmaan::estimate::fiml {
namespace internal {



FitError make_fit_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

PostError make_post_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

PostError fit_to_post(const FitError& err, std::string prefix) {
  return make_post_err(PostError::Kind::NumericIssue,
                       std::move(prefix) + ": " + err.detail);
}

FitError post_to_fit(const PostError& err) {
  return make_fit_err(FitError::Kind::NumericIssue, err.detail);
}

fit_expected<void>
validate_extra_constraints(
    const estimate::frontier::ExtraNonlinearEqConstraints& extra,
    const Eigen::VectorXd& theta0, Eigen::Index npar, const char* who) {
  if (!extra.active()) return {};
  if (!extra.h || !extra.jacobian) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        std::string(who) +
            ": active extra nonlinear constraints require both h and jacobian"));
  }
  const Eigen::VectorXd h0 = extra.h(theta0);
  if (h0.size() != extra.n_constraint) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        std::string(who) + ": extra h(theta) length (" +
            std::to_string(h0.size()) + ") != n_constraint (" +
            std::to_string(extra.n_constraint) + ")"));
  }
  const Eigen::MatrixXd J0 = extra.jacobian(theta0);
  if (J0.rows() != extra.n_constraint || J0.cols() != npar) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        std::string(who) + ": extra jacobian shape (" +
            std::to_string(J0.rows()) + " x " + std::to_string(J0.cols()) +
            ") != (" + std::to_string(extra.n_constraint) + " x " +
            std::to_string(npar) + ")"));
  }
  return {};
}















void add_unique_warning(std::vector<std::string>& warnings, std::string warning) {
  if (std::find(warnings.begin(), warnings.end(), warning) == warnings.end()) {
    warnings.push_back(std::move(warning));
  }
}

fit_expected<void> validate_h1_options(const FIMLH1Options& options) {
  if (options.max_iter < 1) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: max_iter must be >= 1"));
  }
  if (!(options.parameter_tol > 0.0) || !std::isfinite(options.parameter_tol)) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: parameter_tol must be finite and > 0"));
  }
  if (!(options.objective_tol >= 0.0) ||
      !std::isfinite(options.objective_tol)) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: objective_tol must be finite and >= 0"));
  }
  if (!(options.covariance_floor >= 0.0) ||
      !std::isfinite(options.covariance_floor)) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: covariance_floor must be finite and >= 0"));
  }
  if (!(options.covariance_warn >= 0.0) ||
      !std::isfinite(options.covariance_warn)) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: covariance_warn must be finite and >= 0"));
  }
  return {};
}

std::string fmt_sci(double value) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(6) << value;
  return os.str();
}

bool finite_observed_row(const Eigen::MatrixXd& X,
                         const std::vector<Eigen::Index>& obs,
                         Eigen::Index row) {
  for (Eigen::Index c : obs) {
    if (!std::isfinite(X(row, c))) return false;
  }
  return true;
}

fit_expected<void>
validate_raw_shape(const RawData& raw) {
  if (raw.X.empty()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML: RawData.X is empty"));
  }
  if (!raw.mask.empty() && raw.mask.size() != raw.X.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML: RawData.mask must be empty or have one block per X block"));
  }
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const auto& X = raw.X[b];
    if (X.rows() <= 0) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: block " + std::to_string(b) + " has no rows"));
    }
    if (X.cols() <= 0) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: block " + std::to_string(b) + " has no columns"));
    }
    if (!raw.mask.empty() &&
        (raw.mask[b].rows() != X.rows() || raw.mask[b].cols() != X.cols())) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML: mask shape mismatch in block " + std::to_string(b)));
    }
  }
  return {};
}

Eigen::MatrixXd select_square(const Eigen::MatrixXd& M,
                              const std::vector<Eigen::Index>& idx) {
  const Eigen::Index q = static_cast<Eigen::Index>(idx.size());
  Eigen::MatrixXd out(q, q);
  for (Eigen::Index j = 0; j < q; ++j)
    for (Eigen::Index i = 0; i < q; ++i)
      out(i, j) = M(idx[static_cast<std::size_t>(i)],
                    idx[static_cast<std::size_t>(j)]);
  return out;
}

Eigen::VectorXd select_vector(const Eigen::VectorXd& v,
                              const std::vector<Eigen::Index>& idx) {
  const Eigen::Index q = static_cast<Eigen::Index>(idx.size());
  Eigen::VectorXd out(q);
  for (Eigen::Index i = 0; i < q; ++i) {
    out(i) = v(idx[static_cast<std::size_t>(i)]);
  }
  return out;
}

double log_det_from_llt(const Eigen::LLT<Eigen::MatrixXd>& llt) {
  double out = 0.0;
  const auto L = llt.matrixL();
  for (Eigen::Index i = 0; i < L.rows(); ++i) out += std::log(L(i, i));
  return 2.0 * out;
}

double symmetric_scale(const Eigen::MatrixXd& S) {
  if (S.size() == 0) return 1.0;
  double scale = S.cwiseAbs().maxCoeff();
  if (!std::isfinite(scale) || scale <= 0.0) scale = 1.0;
  return std::max(1.0, scale);
}

fit_expected<SymmetricFloorReport>
floor_symmetric_fit(Eigen::MatrixXd& S,
                    double absolute_floor,
                    double relative_floor,
                    const char* what) {
  if (S.rows() != S.cols()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        std::string(what) + " is not square"));
  }
  S = 0.5 * (S + S.transpose()).eval();
  if (!S.allFinite()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        std::string(what) + " contains non-finite values"));
  }
  SymmetricFloorReport report;
  const Eigen::Index q = S.rows();
  if (q == 0) {
    report.min_eigen_before = 0.0;
    report.min_eigen_after = 0.0;
    return report;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(S, Eigen::EigenvaluesOnly);
  if (eig.info() != Eigen::Success) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        std::string(what) + " eigen decomposition failed"));
  }
  report.min_eigen_before = eig.eigenvalues().minCoeff();
  const double floor =
      std::max(absolute_floor, relative_floor * symmetric_scale(S));
  if (report.min_eigen_before < floor) {
    report.applied = true;
    report.ridge = floor - report.min_eigen_before;
    S.diagonal().array() += report.ridge;
    S = 0.5 * (S + S.transpose()).eval();
    report.min_eigen_after = report.min_eigen_before + report.ridge;
  } else {
    report.min_eigen_after = report.min_eigen_before;
  }
  return report;
}

post_expected<SymmetricFloorReport>
floor_symmetric_post(Eigen::MatrixXd& S,
                     double absolute_floor,
                     double relative_floor,
                     std::string what) {
  if (S.rows() != S.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  S = 0.5 * (S + S.transpose()).eval();
  if (!S.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " contains non-finite values"));
  }
  SymmetricFloorReport report;
  const Eigen::Index q = S.rows();
  if (q == 0) {
    report.min_eigen_before = 0.0;
    report.min_eigen_after = 0.0;
    return report;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(S, Eigen::EigenvaluesOnly);
  if (eig.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " eigen decomposition failed"));
  }
  report.min_eigen_before = eig.eigenvalues().minCoeff();
  const double floor =
      std::max(absolute_floor, relative_floor * symmetric_scale(S));
  if (report.min_eigen_before < floor) {
    report.applied = true;
    report.ridge = floor - report.min_eigen_before;
    S.diagonal().array() += report.ridge;
    S = 0.5 * (S + S.transpose()).eval();
    report.min_eigen_after = report.min_eigen_before + report.ridge;
  } else {
    report.min_eigen_after = report.min_eigen_before;
  }
  return report;
}

std::vector<Eigen::Index>
fixed_x_observed_indices(const spec::LatentStructure& pt) {
  std::unordered_set<std::int32_t> exo_vars;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.exo[i] != 1) continue;
    if (i < pt.lhs_var.size() && pt.lhs_var[i] >= 0) exo_vars.insert(pt.lhs_var[i]);
    if (i < pt.rhs_var.size() && pt.rhs_var[i] >= 0) exo_vars.insert(pt.rhs_var[i]);
  }
  std::vector<Eigen::Index> out;
  std::unordered_set<Eigen::Index> seen;
  for (std::int32_t v : exo_vars) {
    if (v < 0 || static_cast<std::size_t>(v) >= pt.ov_pos.size()) continue;
    const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
    if (pos < 0) continue;
    const Eigen::Index idx = static_cast<Eigen::Index>(pos);
    if (seen.insert(idx).second) out.push_back(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<Eigen::Index>
observed_exogenous_indices(const spec::LatentStructure& pt) {
  std::vector<Eigen::Index> out;
  for (std::int32_t v = 0; v < pt.n_vars; ++v) {
    if (static_cast<std::size_t>(v) >= pt.var_role.size() ||
        pt.var_role[static_cast<std::size_t>(v)] != spec::VarRole::ExoOv ||
        static_cast<std::size_t>(v) >= pt.ov_pos.size()) {
      continue;
    }
    const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
    if (pos >= 0) out.push_back(static_cast<Eigen::Index>(pos));
  }
  std::sort(out.begin(), out.end());
  return out;
}

post_expected<double>
fixed_x_saturated_logl(const spec::LatentStructure& pt,
                       const SampleStats& samp) {
  const std::vector<Eigen::Index> exo_idx = fixed_x_observed_indices(pt);
  if (exo_idx.empty()) return 0.0;

  double logl = 0.0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd& S = samp.S[b];
    const Eigen::Index p = S.rows();
    if (exo_idx.back() >= p) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fixed.x exogenous sub-block index out of range"));
    }
    const Eigen::Index px = static_cast<Eigen::Index>(exo_idx.size());
    Eigen::MatrixXd Sxx(px, px);
    for (Eigen::Index r = 0; r < px; ++r) {
      for (Eigen::Index c = 0; c < px; ++c) {
        Sxx(r, c) = S(exo_idx[static_cast<std::size_t>(r)],
                      exo_idx[static_cast<std::size_t>(c)]);
      }
    }
    Eigen::LLT<Eigen::MatrixXd> llt_xx(Sxx);
    if (llt_xx.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "fixed.x exogenous sub-block for block " + std::to_string(b) +
              " is not positive definite"));
    }
    const double n_b = static_cast<double>(samp.n_obs[b]);
    logl += -0.5 * n_b *
        (static_cast<double>(px) * std::log(two_pi) +
         log_det_from_llt(llt_xx) + static_cast<double>(px));
  }
  return logl;
}

double observed_constant(const FIMLCache& cache) {
  double c = 0.0;
  for (const FIMLPattern& pat : cache.patterns) {
    c += static_cast<double>(pat.n_obs)
       * static_cast<double>(pat.observed.size()) * std::log(two_pi);
  }
  return c;
}

fit_expected<double>
h1_complete_data_value(const SampleStats& samp) {
  std::int64_t n_total = 0;
  for (auto n : samp.n_obs) n_total += n;
  if (n_total <= 0) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1: no observations"));
  }

  double f = 0.0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd S = 0.5 * (samp.S[b] + samp.S[b].transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSample,
          "FIML H1: complete-data sample covariance is not positive definite"));
    }
    const double scale = static_cast<double>(samp.n_obs[b]) /
                         static_cast<double>(n_total);
    f += scale * (log_det_from_llt(llt) + static_cast<double>(S.rows()));
  }
  return f;
}

Eigen::Index h1_chol_len(Eigen::Index p) {
  return p * (p + 1) / 2;
}

Eigen::VectorXd h1_start_for_block(const SampleStats& samp,
                                   std::size_t block) {
  const Eigen::Index p = samp.S[block].rows();
  Eigen::VectorXd x(p + h1_chol_len(p));
  x.head(p) = samp.mean[block];

  Eigen::MatrixXd S = 0.5 * (samp.S[block] + samp.S[block].transpose());
  const double max_diag = (p > 0) ? S.diagonal().cwiseAbs().maxCoeff() : 1.0;
  const double base = std::max(1.0, max_diag);
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  for (int k = 0; llt.info() != Eigen::Success && k < 10; ++k) {
    S.diagonal().array() += base * std::pow(10.0, -8.0 + static_cast<double>(k));
    llt.compute(S);
  }
  Eigen::MatrixXd L;
  if (llt.info() == Eigen::Success) {
    L = llt.matrixL();
  } else {
    L = Eigen::MatrixXd::Identity(p, p) * std::sqrt(base);
  }

  Eigen::Index off = p;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      x(off++) = (r == c) ? std::log(std::max(L(r, c), 1e-8)) : L(r, c);
    }
  }
  return x;
}

void h1_decode(const Eigen::VectorXd& x,
               Eigen::Index p,
               Eigen::VectorXd& mu,
               Eigen::MatrixXd& L,
               Eigen::MatrixXd& Sigma) {
  mu = x.head(p);
  L = Eigen::MatrixXd::Zero(p, p);
  Eigen::Index off = p;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      L(r, c) = (r == c) ? std::exp(x(off)) : x(off);
      ++off;
    }
  }
  Sigma.noalias() = L * L.transpose();
}

fit_expected<double>
h1_block_value_from_moments(const FIMLCache& cache,
                            std::size_t block,
                            const Eigen::VectorXd& mu,
                            const Eigen::MatrixXd& Sigma) {
  double f = 0.0;
  for (const FIMLPattern& pat : cache.patterns) {
    if (pat.block != block) continue;
    const Eigen::MatrixXd Sigma_o = select_square(Sigma, pat.observed);
    const Eigen::VectorXd Mu_o = select_vector(mu, pat.observed);
    const Eigen::VectorXd d = pat.mean - Mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "FIML H1: saturated observed-pattern covariance is not positive definite"));
    }
    const Eigen::MatrixXd SigmaInv_A = llt.solve(A);
    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);
    f += scale * (log_det_from_llt(llt) + SigmaInv_A.trace());
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_fit_err(FitError::Kind::NonFiniteObjective,
        "FIML H1 objective evaluated to non-finite"));
  }
  return f;
}

std::vector<Eigen::Index>
missing_indices(Eigen::Index p, const std::vector<Eigen::Index>& observed) {
  std::vector<unsigned char> seen(static_cast<std::size_t>(p), 0);
  for (Eigen::Index idx : observed) seen[static_cast<std::size_t>(idx)] = 1;
  std::vector<Eigen::Index> out;
  for (Eigen::Index i = 0; i < p; ++i) {
    if (!seen[static_cast<std::size_t>(i)]) out.push_back(i);
  }
  return out;
}

Eigen::MatrixXd select_rect(const Eigen::MatrixXd& M,
                            const std::vector<Eigen::Index>& rows,
                            const std::vector<Eigen::Index>& cols) {
  Eigen::MatrixXd out(static_cast<Eigen::Index>(rows.size()),
                      static_cast<Eigen::Index>(cols.size()));
  for (Eigen::Index j = 0; j < out.cols(); ++j) {
    for (Eigen::Index i = 0; i < out.rows(); ++i) {
      out(i, j) = M(rows[static_cast<std::size_t>(i)],
                    cols[static_cast<std::size_t>(j)]);
    }
  }
  return out;
}

// One EM sweep over the block's patterns: returns the H1 objective evaluated
// at the *input* (mu, Sigma) — sharing each pattern's Cholesky with the
// E-step — and writes the M-step update into (mu_next, Sigma_next).
fit_expected<H1EMStep>
h1_em_step_block(const FIMLCache& cache,
                 std::size_t block,
                 const Eigen::VectorXd& mu,
                 const Eigen::MatrixXd& Sigma,
                 const FIMLH1Options& options,
                 Eigen::VectorXd& mu_next,
                 Eigen::MatrixXd& Sigma_next) {
  const Eigen::Index p = cache.block_p[block];
  Eigen::VectorXd sum_x = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd sum_xx = Eigen::MatrixXd::Zero(p, p);
  std::int64_t n_block = 0;
  double f = 0.0;

  for (const FIMLPattern& pat : cache.patterns) {
    if (pat.block != block) continue;
    n_block += pat.n_obs;
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
          "FIML H1: saturated observed-pattern covariance is not positive definite"));
    }
    const Eigen::VectorXd mu_o = select_vector(mu, obs);
    const Eigen::VectorXd d = pat.mean - mu_o;
    const Eigen::MatrixXd A = pat.cov + d * d.transpose();
    const double scale = static_cast<double>(pat.n_obs) /
                         static_cast<double>(cache.n_total);

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

    sum_x.noalias() += static_cast<double>(pat.n_obs) * avg_x;
    sum_xx.noalias() += static_cast<double>(pat.n_obs) * avg_xx;
  }

  if (n_block <= 0) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1 EM: block has no observations"));
  }
  if (!std::isfinite(f)) {
    return std::unexpected(make_fit_err(FitError::Kind::NonFiniteObjective,
        "FIML H1 objective evaluated to non-finite"));
  }
  mu_next = sum_x / static_cast<double>(n_block);
  Sigma_next = sum_xx / static_cast<double>(n_block) -
               mu_next * mu_next.transpose();
  Sigma_next = 0.5 * (Sigma_next + Sigma_next.transpose());
  if (options.lavaan_covariance_ridge) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Sigma_next, Eigen::EigenvaluesOnly);
    if (eig.info() != Eigen::Success)
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "FIML H1 EM covariance eigen decomposition failed"));
    SymmetricFloorReport repair;
    repair.min_eigen_before = eig.eigenvalues().minCoeff();
    repair.applied = repair.min_eigen_before < 1e-6;
    repair.ridge = repair.applied ? Sigma_next.diagonal().maxCoeff() * 1e-8 : 0.0;
    Sigma_next.diagonal().array() += repair.ridge;
    repair.min_eigen_after = repair.min_eigen_before + repair.ridge;
    return H1EMStep{f, repair};
  }
  auto repair = floor_symmetric_fit(
      Sigma_next, options.covariance_floor, /*relative_floor=*/0.0,
      "FIML H1 EM covariance update");
  if (!repair.has_value()) return std::unexpected(repair.error());
  return H1EMStep{f, *repair};
}

double h1_em_parameter_change(const Eigen::VectorXd& mu,
                              const Eigen::MatrixXd& Sigma,
                              const Eigen::VectorXd& mu_next,
                              const Eigen::MatrixXd& Sigma_next) {
  double out = 0.0;
  if (mu.size() > 0) {
    out = std::max(out, (mu_next - mu).cwiseAbs().maxCoeff());
  }
  if (Sigma.size() > 0) {
    out = std::max(out, (Sigma_next - Sigma).cwiseAbs().maxCoeff());
  }
  return out;
}

// Squared extrapolation of the EM fixed-point map in (mu, vech(Sigma)):
// r=M(x)-x, v=M(M(x))-2*M(x)+x, x_sq=x+2*a*r+a*a*v,
// a=||r||/||v||. See Du & Varadhan, arXiv:1810.11163, Table 1.
// The pinned behavior clips a to [1, cap], stabilizes extrapolation with
// another EM update, and allows one total log-likelihood unit of decrease.
fit_expected<H1EMResult> h1_squarem_iterate_block(const FIMLCache& cache,
    std::size_t block, Eigen::VectorXd& mu, Eigen::MatrixXd& sigma,
    const FIMLH1Options& options) {
  const Eigen::Index p = mu.size();
  auto pack = [&](const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
    Eigen::VectorXd x(p + p*(p+1)/2);
    x.head(p) = mean;
    Eigen::Index k=p;
    for (Eigen::Index j=0;j<p;++j) for (Eigen::Index i=j;i<p;++i) x(k++)=cov(i,j);
    return x;
  };
  auto unpack = [&](const Eigen::VectorXd& x, Eigen::VectorXd& mean, Eigen::MatrixXd& cov) {
    mean=x.head(p); cov.resize(p,p);
    Eigen::Index k=p;
    for (Eigen::Index j=0;j<p;++j) for (Eigen::Index i=j;i<p;++i) cov(i,j)=cov(j,i)=x(k++);
  };
  H1EMResult result;
  auto update = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    ++result.iterations;
    Eigen::VectorXd mean, next_mean;
    Eigen::MatrixXd cov, next_cov;
    unpack(x,mean,cov);
    auto step=h1_em_step_block(cache,block,mean,cov,options,next_mean,next_cov);
    if (!step) return std::unexpected(step.error());
    result.min_covariance_eigen=std::min(result.min_covariance_eigen,step->sigma_repair.min_eigen_after);
    if (step->sigma_repair.applied) {
      ++result.covariance_repairs;
      result.max_covariance_ridge=std::max(result.max_covariance_ridge,step->sigma_repair.ridge);
    }
    return pack(next_mean,next_cov);
  };
  auto value = [&](const Eigen::VectorXd& x) {
    Eigen::VectorXd mean; Eigen::MatrixXd cov;
    unpack(x,mean,cov);
    return h1_block_value_from_moments(cache,block,mean,cov);
  };
  Eigen::VectorXd current=pack(mu,sigma);
  auto initial=value(current);
  if (!initial) return std::unexpected(initial.error());
  double previous=*initial, cap=1.0;
  const double allowed_loss=2.0/static_cast<double>(cache.n_total);
  while (result.iterations<options.max_iter) {
    auto first=update(current); if (!first) return std::unexpected(first.error());
    auto second=update(*first); if (!second) return std::unexpected(second.error());
    const Eigen::VectorXd r=*first-current;
    const Eigen::VectorXd v=*second-2.0*(*first)+current;
    double a=1.0;
    if (v.squaredNorm()>std::numeric_limits<double>::epsilon())
      a=std::clamp(std::sqrt(r.squaredNorm()/v.squaredNorm()),1.0,cap);
    Eigen::VectorXd next=*second;
    std::optional<double> candidate_value;
    if (a>1.01) {
      const Eigen::VectorXd extrapolated=current+2.0*a*r+a*a*v;
      auto stabilized=update(extrapolated);
      if (stabilized && stabilized->allFinite()) {
        auto proposed=value(*stabilized);
        if (proposed && std::isfinite(*proposed) && *proposed<=previous+allowed_loss) {
          next=std::move(*stabilized);
          candidate_value=*proposed;
        }
      }
      if (!candidate_value) {
        if (a>=cap) cap=std::max(1.0,cap/4.0);
        a=1.0;
      }
    }
    auto next_value=candidate_value ? fit_expected<double>(*candidate_value) : value(next);
    if (!next_value) return std::unexpected(next_value.error());
    if (!std::isfinite(*next_value) || *next_value>previous+allowed_loss) break;
    if (a>=cap) cap*=4.0;
    result.parameter_change=(next-current).cwiseAbs().maxCoeff();
    result.objective_change=std::abs(*next_value-previous);
    result.objective_converged=result.objective_change<=options.objective_tol*(1.0+std::abs(*next_value));
    current=std::move(next);
    previous=*next_value;
    if (result.parameter_change<options.parameter_tol) {
      result.converged=true;
      break;
    }
  }
  unpack(current,mu,sigma);
  result.value=previous;
  return result;
}

// Runs the EM iteration for one block. On return (mu, Sigma) hold the
// converged moments and the returned value is the H1 objective at exactly
// those moments — the same value/update ordering as the previous two-pass
// loop, with one Cholesky per pattern per iteration instead of two.
fit_expected<H1EMResult>
h1_em_iterate_block(const FIMLCache& cache,
                    std::size_t block,
                    Eigen::VectorXd& mu,
                    Eigen::MatrixXd& Sigma,
                    const FIMLH1Options& options) {
  Eigen::VectorXd mu_next;
  Eigen::MatrixXd Sigma_next;
  double prev = std::numeric_limits<double>::infinity();
  double cur = std::numeric_limits<double>::infinity();
  H1EMResult result;
  for (int iter = 0; iter < options.max_iter; ++iter) {
    auto step = h1_em_step_block(cache, block, mu, Sigma, options,
                                 mu_next, Sigma_next);
    if (!step.has_value()) return std::unexpected(step.error());
    cur = step->value;
    if (step->sigma_repair.applied) {
      ++result.covariance_repairs;
      result.max_covariance_ridge =
          std::max(result.max_covariance_ridge, step->sigma_repair.ridge);
    }
    result.min_covariance_eigen =
        std::min(result.min_covariance_eigen,
                 step->sigma_repair.min_eigen_after);
    result.parameter_change =
        h1_em_parameter_change(mu, Sigma, mu_next, Sigma_next);
    result.objective_change = std::isfinite(prev)
        ? std::abs(prev - cur)
        : std::numeric_limits<double>::infinity();
    result.objective_converged =
        result.objective_change <=
            options.objective_tol * (1.0 + std::abs(cur));
    if (result.parameter_change <= options.parameter_tol) {
      mu.swap(mu_next);
      Sigma.swap(Sigma_next);
      auto final_val = h1_block_value_from_moments(cache, block, mu, Sigma);
      if (!final_val.has_value()) return std::unexpected(final_val.error());
      result.value = *final_val;
      result.iterations = iter + 1;
      result.converged = true;
      return result;
    }
    prev = cur;
    mu.swap(mu_next);
    Sigma.swap(Sigma_next);
  }
  // Iteration cap reached: (mu, Sigma) carry one more update than `cur`
  // was evaluated at, so re-evaluate to keep value/moment consistency.
  auto final_val = h1_block_value_from_moments(cache, block, mu, Sigma);
  if (!final_val.has_value()) return std::unexpected(final_val.error());
  result.value = *final_val;
  result.iterations = options.max_iter;
  result.converged = false;
  return result;
}

// Light consistency check before indexing a caller-supplied FIMLH1 against a
// caller-supplied pack: the pack overloads trust the caller to have built both
// from the same raw data, but a block-count mismatch is always a usage bug.
fit_expected<void>
validate_h1_blocks(const FIMLCache& cache, const FIMLH1& h1) {
  if (h1.mu.size() != cache.block_p.size() ||
      h1.sigma.size() != cache.block_p.size()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "FIML H1: moments and pattern cache have different block counts"));
  }
  return {};
}

post_expected<Eigen::MatrixXd>
invert_symmetric(const Eigen::MatrixXd& A, std::string what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not square"));
  }
  const Eigen::Index q = A.rows();
  const Eigen::MatrixXd S = 0.5 * (A + A.transpose());
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  if (llt.info() == Eigen::Success) {
    return Eigen::MatrixXd(llt.solve(Eigen::MatrixXd::Identity(q, q)));
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        std::move(what) + " is not invertible"));
  }
  return Eigen::MatrixXd(ldlt.solve(Eigen::MatrixXd::Identity(q, q)));
}

// Per-observed-pattern moment factorization shared by all rows carrying that
// pattern: rows in the same pattern see the same Σ_oo, so the Cholesky and
// explicit inverse are computed once per pattern, not once per row.








post_expected<const PatternMoments*>
pattern_moments_for(PatternMomentMap& cache_map,
                    const std::vector<Eigen::Index>& obs,
                    const Eigen::VectorXd& Mu,
                    const Eigen::MatrixXd& Sigma,
                    const char* what) {
  auto it = cache_map.find(obs);
  if (it == cache_map.end()) {
    const Eigen::Index q = static_cast<Eigen::Index>(obs.size());
    const Eigen::MatrixXd Sigma_o = select_square(Sigma, obs);
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma_o);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(what) + " is not positive definite"));
    }
    PatternMoments pm;
    pm.SigmaInv = llt.solve(Eigen::MatrixXd::Identity(q, q));
    pm.SigmaInv = 0.5 * (pm.SigmaInv + pm.SigmaInv.transpose()).eval();
    pm.Mu_o = select_vector(Mu, obs);
    it = cache_map.emplace(obs, std::move(pm)).first;
  }
  return &it->second;
}


}  // namespace internal

}  // namespace magmaan::estimate::fiml
