#include "ordinal_internal.hpp"

namespace magmaan::estimate {

using namespace detail_ordinal;

namespace detail_ordinal {


Eigen::Index vech_sym_index(Eigen::Index p, Eigen::Index r,
                            Eigen::Index c) noexcept {
  if (r < c) std::swap(r, c);
  return vech_index(p, r, c);
}


void add_sigma_trace_weight(Eigen::Ref<Eigen::MatrixXd> G, Eigen::Index r,
                            Eigen::Index c, double value) {
  if (value == 0.0) return;
  if (r == c) {
    G(r, c) += value;
  } else {
    const double half = 0.5 * value;
    G(r, c) += half;
    G(c, r) += half;
  }
}


double sigma_deriv(const Eigen::MatrixXd& J_sigma, Eigen::Index sigma_off,
                   Eigen::Index p, Eigen::Index r, Eigen::Index c,
                   Eigen::Index param) {
  const Eigen::Index row = sigma_off + vech_sym_index(p, r, c);
  if (row < 0 || row >= J_sigma.rows() || param < 0 || param >= J_sigma.cols()) {
    return 0.0;
  }
  return J_sigma(row, param);
}


double mu_deriv(const Eigen::MatrixXd& J_mu, Eigen::Index mu_off,
                Eigen::Index idx, Eigen::Index param) {
  const Eigen::Index row = mu_off + idx;
  if (row < 0 || row >= J_mu.rows() || param < 0 || param >= J_mu.cols()) {
    return 0.0;
  }
  return J_mu(row, param);
}


double threshold_deriv(const ThresholdLayout& layout, std::size_t b,
                       Eigen::Index threshold, Eigen::Index param) {
  const std::int32_t fr = layout.free[b][static_cast<std::size_t>(threshold)];
  return fr > 0 && static_cast<Eigen::Index>(fr - 1) == param ? 1.0 : 0.0;
}


std::vector<char> threshold_parameter_mask(const ThresholdLayout& layout,
                                           Eigen::Index q) {
  std::vector<char> out(static_cast<std::size_t>(q), 0);
  for (std::size_t b = 0; b < layout.free.size(); ++b) {
    for (const std::int32_t fr : layout.free[b]) {
      if (fr > 0 && static_cast<Eigen::Index>(fr - 1) < q) {
        out[static_cast<std::size_t>(fr - 1)] = 1;
      }
    }
  }
  for (const auto& block : layout.delta_free)
    for (const auto fr : block)
      if (fr > 0 && fr <= q) out[static_cast<std::size_t>(fr - 1)] = 1;
  return out;
}


void add_assoc_gradient(Eigen::Ref<Eigen::MatrixXd> G,
                        const Eigen::MatrixXd& Sigma, Eigen::Index i,
                        Eigen::Index j, bool std_i, bool std_j,
                        double scale) {
  if (scale == 0.0) return;
  const double sij = Sigma(i, j);
  if (!std_i && !std_j) {
    add_sigma_trace_weight(G, i, j, scale);
    return;
  }
  if (std_i && std_j) {
    const double sii = Sigma(i, i);
    const double sjj = Sigma(j, j);
    const double inv_i = 1.0 / std::sqrt(sii);
    const double inv_j = 1.0 / std::sqrt(sjj);
    add_sigma_trace_weight(G, i, j, scale * inv_i * inv_j);
    add_sigma_trace_weight(G, i, i,
                           scale * (-0.5 * sij * inv_i / sii * inv_j));
    add_sigma_trace_weight(G, j, j,
                           scale * (-0.5 * sij * inv_i * inv_j / sjj));
    return;
  }
  const Eigen::Index o = std_i ? i : j;
  const double soo = Sigma(o, o);
  const double inv_o = 1.0 / std::sqrt(soo);
  add_sigma_trace_weight(G, i, j, scale * inv_o);
  add_sigma_trace_weight(G, o, o, scale * (-0.5 * sij * inv_o / soo));
}


double assoc_extra(const Eigen::MatrixXd& Sigma, const Eigen::MatrixXd& J_sigma,
                   Eigen::Index sigma_off, Eigen::Index p, Eigen::Index i,
                   Eigen::Index j, bool std_i, bool std_j, Eigen::Index a,
                   Eigen::Index b) {
  if (!std_i && !std_j) return 0.0;
  const double dx_a = sigma_deriv(J_sigma, sigma_off, p, i, j, a);
  const double dx_b = sigma_deriv(J_sigma, sigma_off, p, i, j, b);
  const double sij = Sigma(i, j);
  if (std_i && std_j) {
    const double sii = Sigma(i, i);
    const double sjj = Sigma(j, j);
    const double inv_i = 1.0 / std::sqrt(sii);
    const double inv_j = 1.0 / std::sqrt(sjj);
    const double di_a = sigma_deriv(J_sigma, sigma_off, p, i, i, a);
    const double di_b = sigma_deriv(J_sigma, sigma_off, p, i, i, b);
    const double dj_a = sigma_deriv(J_sigma, sigma_off, p, j, j, a);
    const double dj_b = sigma_deriv(J_sigma, sigma_off, p, j, j, b);
    const double f_xi = -0.5 * inv_i / sii * inv_j;
    const double f_xj = -0.5 * inv_i * inv_j / sjj;
    const double f_ii = 0.75 * sij * inv_i / (sii * sii) * inv_j;
    const double f_jj = 0.75 * sij * inv_i * inv_j / (sjj * sjj);
    const double f_ij = 0.25 * sij * inv_i / sii * inv_j / sjj;
    return f_xi * (dx_a * di_b + dx_b * di_a) +
           f_xj * (dx_a * dj_b + dx_b * dj_a) +
           f_ii * di_a * di_b + f_jj * dj_a * dj_b +
           f_ij * (di_a * dj_b + di_b * dj_a);
  }
  const Eigen::Index o = std_i ? i : j;
  const double soo = Sigma(o, o);
  const double inv_o = 1.0 / std::sqrt(soo);
  const double do_a = sigma_deriv(J_sigma, sigma_off, p, o, o, a);
  const double do_b = sigma_deriv(J_sigma, sigma_off, p, o, o, b);
  const double f_xo = -0.5 * inv_o / soo;
  const double f_oo = 0.75 * sij * inv_o / (soo * soo);
  return f_xo * (dx_a * do_b + dx_b * do_a) + f_oo * do_a * do_b;
}


double scaled_threshold_extra(double value_minus_mu, double sigma_ii,
                              double ds_a, double ds_b, double dtau_a,
                              double dtau_b, double dmu_a, double dmu_b) {
  const double inv = 1.0 / std::sqrt(sigma_ii);
  const double d1 = -0.5 * inv / sigma_ii;
  const double d2 = 0.75 * inv / (sigma_ii * sigma_ii);
  return d1 * ((dtau_a - dmu_a) * ds_b + (dtau_b - dmu_b) * ds_a) +
         value_minus_mu * d2 * ds_a * ds_b;
}


MomentCurvatureWeights ordinal_curvature_weights(
    const data::OrdinalStats& stats,
    const ThresholdLayout& layout,
    const model::ImpliedMoments& moments,
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& h,
    OrdinalParameterization param,
    std::size_t b) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  const Eigen::MatrixXd& Sig = moments.sigma[b];
  MomentCurvatureWeights out{Eigen::MatrixXd::Zero(p, p),
                             Eigen::VectorXd::Zero(p)};

  if (!theta_param) {
    for (Eigen::Index k = 0; k < nth; ++k) {
      const auto ov = stats.threshold_ov[b][static_cast<std::size_t>(k)];
      out.u(ov) -= h(k) * ordinal_delta(layout, theta, b, ov);
    }
    Eigen::Index r = nth;
    for (Eigen::Index j = 0; j < p; ++j) for (Eigen::Index i = j + 1; i < p; ++i)
      add_sigma_trace_weight(out.G, i, j, h(r++) * ordinal_delta(layout, theta, b, i) * ordinal_delta(layout, theta, b, j));
    return out;
  }
  const std::vector<std::int32_t> sf(static_cast<std::size_t>(p), 1);

  // mu (nu + Lambda*alpha) enters the threshold residual whenever the model
  // has a mean structure, whether or not this indicator is standardized:
  // theta and released-delta compare (tau-mu)/sqrt(Sigma_ii), and plain delta
  // compares tau-mu directly (see ordinal_residuals). `add_lisrel_second_order`
  // consumes `out.u` generically for the bilinear Lambda*alpha curvature in
  // every case; only the nonlinear 1/sqrt(Sigma_ii) chain-rule term (the
  // `scaled` branch below, via `ordinal_curvature_extra`) is standardization-
  // specific.
  const bool have_mu = b < moments.mu.size() && moments.mu[b].size() == p;
  const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
  for (Eigen::Index k = 0; k < nth; ++k) {
    const double hk = h(k);
    const Eigen::Index ov = stats.threshold_ov[b][static_cast<std::size_t>(k)];
    const bool scaled = sf[static_cast<std::size_t>(ov)] != 0;
    const double mu = have_mu ? moments.mu[b](ov) : 0.0;
    if (scaled) {
      const double s = Sig(ov, ov);
      const double inv = 1.0 / std::sqrt(s);
      const double v = it(k) - mu;
      add_sigma_trace_weight(out.G, ov, ov, hk * (-0.5 * v * inv / s));
      if (have_mu) out.u(ov) -= hk * inv;
    } else if (have_mu) {
      out.u(ov) -= hk;
    }
  }

  Eigen::Index row = nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      add_assoc_gradient(out.G, Sig, i, j,
                         sf[static_cast<std::size_t>(i)] != 0,
                         sf[static_cast<std::size_t>(j)] != 0,
                         h(row++));
    }
  }
  return out;
}


double ordinal_curvature_extra(const data::OrdinalStats& stats,
                               const ThresholdLayout& layout,
                               const model::ImpliedMoments& moments,
                               const Eigen::MatrixXd& J_sigma,
                               const Eigen::MatrixXd& J_mu,
                               const Eigen::VectorXd& theta,
                               const Eigen::VectorXd& h,
                               OrdinalParameterization param,
                               std::size_t blk,
                               Eigen::Index sigma_off,
                               Eigen::Index mu_off,
                               Eigen::Index a,
                               Eigen::Index c) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[blk].rows();
  const Eigen::Index nth = stats.thresholds[blk].size();
  const Eigen::MatrixXd& Sig = moments.sigma[blk];
  if (!theta_param) {
    double out = 0.0;
    for (Eigen::Index k = 0; k < nth; ++k) {
      const auto ov = stats.threshold_ov[blk][static_cast<std::size_t>(k)];
      const double da = ordinal_delta_deriv(layout, blk, ov, a), dc = ordinal_delta_deriv(layout, blk, ov, c);
      out += h(k) * (da * (threshold_deriv(layout, blk, k, c) - mu_deriv(J_mu, mu_off, ov, c)) +
                     dc * (threshold_deriv(layout, blk, k, a) - mu_deriv(J_mu, mu_off, ov, a)));
    }
    Eigen::Index r = nth;
    for (Eigen::Index j = 0; j < p; ++j) for (Eigen::Index i = j + 1; i < p; ++i) {
      const double di = ordinal_delta(layout, theta, blk, i), dj = ordinal_delta(layout, theta, blk, j);
      const double ia = ordinal_delta_deriv(layout, blk, i, a), ic = ordinal_delta_deriv(layout, blk, i, c);
      const double ja = ordinal_delta_deriv(layout, blk, j, a), jc = ordinal_delta_deriv(layout, blk, j, c);
      out += h(r++) * ((ia * jc + ic * ja) * Sig(i,j) +
          (ia * dj + di * ja) * sigma_deriv(J_sigma, sigma_off, p, i,j,c) +
          (ic * dj + di * jc) * sigma_deriv(J_sigma, sigma_off, p, i,j,a));
    }
    return out;
  }
  const std::vector<std::int32_t> sf(static_cast<std::size_t>(p), 1);

  double out = 0.0;
  const bool have_mu = blk < moments.mu.size() &&
                       moments.mu[blk].size() == p;
  const Eigen::VectorXd it = implied_thresholds(layout, theta, blk);
  for (Eigen::Index k = 0; k < nth; ++k) {
    const Eigen::Index ov =
        stats.threshold_ov[blk][static_cast<std::size_t>(k)];
    if (sf[static_cast<std::size_t>(ov)] == 0) continue;
    const double tau_a = threshold_deriv(layout, blk, k, a);
    const double tau_c = threshold_deriv(layout, blk, k, c);
    const double mu_a = have_mu ? mu_deriv(J_mu, mu_off, ov, a) : 0.0;
    const double mu_c = have_mu ? mu_deriv(J_mu, mu_off, ov, c) : 0.0;
    const double ds_a = sigma_deriv(J_sigma, sigma_off, p, ov, ov, a);
    const double ds_c = sigma_deriv(J_sigma, sigma_off, p, ov, ov, c);
    const double mu = have_mu ? moments.mu[blk](ov) : 0.0;
    out += h(k) * scaled_threshold_extra(
                      it(k) - mu, Sig(ov, ov), ds_a, ds_c,
                      tau_a, tau_c, mu_a, mu_c);
  }

  Eigen::Index row = nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out += h(row) * assoc_extra(
                          Sig, J_sigma, sigma_off, p, i, j,
                          sf[static_cast<std::size_t>(i)] != 0,
                          sf[static_cast<std::size_t>(j)] != 0,
                          a, c);
      ++row;
    }
  }
  return out;
}


MomentCurvatureWeights mixed_curvature_weights(
    const data::MixedOrdinalStats& stats,
    const ThresholdLayout& layout,
    const model::ImpliedMoments& moments,
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& h,
    OrdinalParameterization param,
    std::size_t b) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  const Eigen::Index p = stats.R[b].rows();
  const Eigen::Index nth = stats.thresholds[b].size();
  const Eigen::MatrixXd& Sig = moments.sigma[b];
  MomentCurvatureWeights out{Eigen::MatrixXd::Zero(p, p),
                             Eigen::VectorXd::Zero(p)};

  Eigen::Index row = 0;
  const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
  for (Eigen::Index k = 0; k < nth; ++k) {
    if (theta_param) {
      const Eigen::Index ov =
          stats.threshold_ov[b][static_cast<std::size_t>(k)];
      const double s = Sig(ov, ov);
      const double inv = 1.0 / std::sqrt(s);
      add_sigma_trace_weight(out.G, ov, ov,
                             h(row) * (-0.5 * it(k) * inv / s));
    }
    ++row;
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) out.u(j) -= h(row++);
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0)
      add_sigma_trace_weight(out.G, j, j, h(row++));
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool std_i = theta_param &&
                         stats.ordered[b][static_cast<std::size_t>(i)] != 0;
      const bool std_j = theta_param &&
                         stats.ordered[b][static_cast<std::size_t>(j)] != 0;
      add_assoc_gradient(out.G, Sig, i, j, std_i, std_j, h(row++));
    }
  }
  return out;
}


double mixed_curvature_extra(const data::MixedOrdinalStats& stats,
                             const ThresholdLayout& layout,
                             const model::ImpliedMoments& moments,
                             const Eigen::MatrixXd& J_sigma,
                             const Eigen::VectorXd& theta,
                             const Eigen::VectorXd& h,
                             OrdinalParameterization param,
                             std::size_t blk,
                             Eigen::Index sigma_off,
                             Eigen::Index a,
                             Eigen::Index c) {
  const bool theta_param = param == OrdinalParameterization::Theta;
  if (!theta_param) return 0.0;
  const Eigen::Index p = stats.R[blk].rows();
  const Eigen::Index nth = stats.thresholds[blk].size();
  const Eigen::MatrixXd& Sig = moments.sigma[blk];
  const Eigen::VectorXd it = implied_thresholds(layout, theta, blk);

  double out = 0.0;
  Eigen::Index row = 0;
  for (Eigen::Index k = 0; k < nth; ++k) {
    const Eigen::Index ov =
        stats.threshold_ov[blk][static_cast<std::size_t>(k)];
    const double tau_a = threshold_deriv(layout, blk, k, a);
    const double tau_c = threshold_deriv(layout, blk, k, c);
    const double ds_a = sigma_deriv(J_sigma, sigma_off, p, ov, ov, a);
    const double ds_c = sigma_deriv(J_sigma, sigma_off, p, ov, ov, c);
    out += h(row) * scaled_threshold_extra(
                       it(k), Sig(ov, ov), ds_a, ds_c,
                       tau_a, tau_c, 0.0, 0.0);
    ++row;
  }
  for (Eigen::Index j = 0; j < p; ++j)
    if (stats.ordered[blk][static_cast<std::size_t>(j)] == 0) ++row;
  for (Eigen::Index j = 0; j < p; ++j)
    if (stats.ordered[blk][static_cast<std::size_t>(j)] == 0) ++row;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool std_i =
          stats.ordered[blk][static_cast<std::size_t>(i)] != 0;
      const bool std_j =
          stats.ordered[blk][static_cast<std::size_t>(j)] != 0;
      out += h(row) * assoc_extra(Sig, J_sigma, sigma_off, p, i, j,
                                  std_i, std_j, a, c);
      ++row;
    }
  }
  return out;
}


post_expected<Eigen::MatrixXd>
ordinal_observed_bread_analytic(const spec::LatentStructure& pt,
                                const model::MatrixRep& rep,
                                const data::OrdinalStats& stats,
                                const Estimates& est,
                                const ThresholdLayout& layout,
                                const std::vector<Eigen::MatrixXd>& Ws,
                                const Eigen::MatrixXd& K,
                                OrdinalParameterization parameterization,
                                Eigen::MatrixXd* correction) {
  if (auto ok = require_linear_sensitivity(pt); !ok) return std::unexpected(ok.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  auto assembled = ev_or->assembled(est.theta);
  if (!assembled.has_value()) return std::unexpected(model_to_post(assembled.error()));
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(fit_to_post(n_or.error()));

  const Eigen::Index q = est.theta.size();
  if (K.rows() != q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal observed bread: K row count does not match theta length"));
  }
  const auto locs = ev_or->param_locations();
  if (locs.size() != static_cast<std::size_t>(q)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal observed bread: parameter-location count mismatch"));
  }
  const std::vector<char> threshold_mask = threshold_parameter_mask(layout, q);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(q, q);
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    const Eigen::MatrixXd Delta = ordinal_moment_jacobian_block(
        stats, layout, eval->moments, eval->J_sigma, est.theta,
        parameterization, eval->J_mu, b, sigma_off, mu_off);
    const Eigen::VectorXd d =
        ordinal_block_residual(stats, layout, eval->moments, est.theta,
                               parameterization, b);
    if (Ws[b].rows() != mb || Ws[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal observed bread: weight shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd h = Ws[b] * d;
    const double w_b = static_cast<double>(stats.n_obs[b]) /
                       static_cast<double>(*n_or);
    H.noalias() += w_b * (Delta.transpose() * Ws[b] * Delta);
    const auto curv = ordinal_curvature_weights(
        stats, layout, eval->moments, est.theta, h, parameterization, b);
    const bool has_mu_rows =
        eval->J_mu.rows() > 0 && mu_off + p <= eval->J_mu.rows();
    add_lisrel_second_order(
        correction != nullptr ? C : H, curv, assembled->blocks[b], locs, threshold_mask, b, has_mu_rows,
        w_b, [&](Eigen::Index a, Eigen::Index c) {
          return ordinal_curvature_extra(
              stats, layout, eval->moments, eval->J_sigma, eval->J_mu,
              est.theta, h, parameterization, b, sigma_off, mu_off, a, c);
        });
    sigma_off += vech_len(p);
    mu_off += p;
  }
  if (correction != nullptr) *correction = 0.5 * (C + C.transpose()).eval();
  if (correction != nullptr) H += C;
  if (!H.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal observed bread: non-finite Hessian"));
  }
  H = 0.5 * (H + H.transpose()).eval();
  Eigen::MatrixXd Halpha = K.transpose() * H * K;
  return Eigen::MatrixXd(0.5 * (Halpha + Halpha.transpose()).eval());
}


post_expected<Eigen::MatrixXd>
mixed_observed_bread_analytic(const spec::LatentStructure& pt,
                              const model::MatrixRep& rep,
                              const data::MixedOrdinalStats& stats,
                              const Estimates& est,
                              const ThresholdLayout& layout,
                              const std::vector<Eigen::MatrixXd>& Ws,
                              const Eigen::MatrixXd& K,
                              OrdinalParameterization parameterization,
                              Eigen::MatrixXd* correction) {
  if (auto ok = require_linear_sensitivity(pt); !ok) return std::unexpected(ok.error());
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) return std::unexpected(model_to_post(ev_or.error()));
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  auto assembled = ev_or->assembled(est.theta);
  if (!assembled.has_value()) return std::unexpected(model_to_post(assembled.error()));
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(fit_to_post(n_or.error()));

  const Eigen::Index q = est.theta.size();
  if (K.rows() != q) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal observed bread: K row count does not match theta length"));
  }
  const auto locs = ev_or->param_locations();
  if (locs.size() != static_cast<std::size_t>(q)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal observed bread: parameter-location count mismatch"));
  }
  const std::vector<char> threshold_mask = threshold_parameter_mask(layout, q);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(q, q);
  const Eigen::MatrixXd Delta_full = mixed_moment_jacobian(
      stats, layout, eval->moments, eval->J_sigma, eval->J_mu, est.theta,
      parameterization);
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  Eigen::Index moment_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.moments[b].size();
    const Eigen::MatrixXd Delta = Delta_full.block(moment_off, 0, mb, q);
    const Eigen::VectorXd d =
        mixed_model_moments(stats, layout, eval->moments, est.theta, b,
                            parameterization) -
        stats.moments[b];
    if (Ws[b].rows() != mb || Ws[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed ordinal observed bread: weight shape mismatch in block " +
              std::to_string(b)));
    }
    const Eigen::VectorXd h = Ws[b] * d;
    const double w_b = static_cast<double>(stats.n_obs[b]) /
                       static_cast<double>(*n_or);
    H.noalias() += w_b * (Delta.transpose() * Ws[b] * Delta);
    const auto curv = mixed_curvature_weights(
        stats, layout, eval->moments, est.theta, h, parameterization, b);
    const bool has_mu_rows =
        eval->J_mu.rows() > 0 && mu_off + p <= eval->J_mu.rows();
    add_lisrel_second_order(
        correction != nullptr ? C : H, curv, assembled->blocks[b], locs, threshold_mask, b, has_mu_rows,
        w_b, [&](Eigen::Index a, Eigen::Index c) {
          return mixed_curvature_extra(
              stats, layout, eval->moments, eval->J_sigma, est.theta, h,
              parameterization, b, sigma_off, a, c);
        });
    sigma_off += vech_len(p);
    mu_off += p;
    moment_off += mb;
  }
  if (correction != nullptr) *correction = 0.5 * (C + C.transpose()).eval();
  if (correction != nullptr) H += C;
  if (!H.allFinite()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal observed bread: non-finite Hessian"));
  }
  H = 0.5 * (H + H.transpose()).eval();
  Eigen::MatrixXd Halpha = K.transpose() * H * K;
  return Eigen::MatrixXd(0.5 * (Halpha + Halpha.transpose()).eval());
}


std::vector<Eigen::MatrixXd> dense_weights_from_factors(const WhitenFactors& factors) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(factors.size());
  for (const auto& f : factors) {
    const Eigen::MatrixXd F = f.to_dense();
    out.push_back(F * F.transpose());
  }
  return out;
}


post_expected<OrdinalNewtonRaw>
ordinal_newton_parts_prepared(const spec::LatentStructure& pt,
                              const model::MatrixRep& rep,
                              const data::OrdinalStats& stats,
                              const ThresholdLayout& layout,
                              const WhitenFactors& factors,
                              const Eigen::VectorXd& theta,
                              OrdinalParameterization parameterization) {
  const Eigen::Index q = theta.size();
  const auto Ws = dense_weights_from_factors(factors);
  Eigen::MatrixXd C;
  if (Ws.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal Newton parts: weight block count mismatch"));
  }
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(fit_to_post(n_or.error()));
  Estimates at;
  at.theta = theta;
  auto H = ordinal_observed_bread_analytic(pt, rep, stats, at, layout, Ws,
      Eigen::MatrixXd::Identity(q, q), parameterization, &C);
  if (!H.has_value()) return std::unexpected(H.error());
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return std::unexpected(model_to_post(ev.error()));
  auto eval = ev->evaluate(theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  Eigen::Index rows = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) rows += Ws[b].rows();
  Eigen::MatrixXd A(rows, q);
  Eigen::VectorXd score(rows);
  Eigen::Index off = 0;
  Eigen::Index sigma_off = 0, mu_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::MatrixXd Delta = ordinal_moment_jacobian_block(
        stats, layout, eval->moments, eval->J_sigma, theta, parameterization,
        eval->J_mu, b, sigma_off, mu_off);
    if (Ws[b].rows() != Delta.rows()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal Newton parts: weight dimension mismatch"));
    }
    const double root_n = std::sqrt(static_cast<double>(stats.n_obs[b]));
    A.middleRows(off, Delta.rows()) = factors[b].t_apply(root_n, Delta);
    const Eigen::VectorXd residual = ordinal_block_residual(
        stats, layout, eval->moments, theta, parameterization, b);
    score.segment(off, Delta.rows()) = factors[b].t_apply(root_n, residual);
    off += Delta.rows();
    sigma_off += vech_len(p);
    mu_off += p;
  }
  // Keep the factor and residual correction: neither is recovered by
  // subtracting or factoring rounded cross-products in weak directions.
  Eigen::MatrixXd M = A.transpose() * A;
  C *= static_cast<double>(*n_or);
  Eigen::MatrixXd observed = M + C;
  return OrdinalNewtonRaw{std::move(observed), std::move(M), std::move(C),
                          std::move(A), std::move(score)};
}


post_expected<OrdinalNewtonRaw>
mixed_newton_parts_prepared(const spec::LatentStructure& pt,
                            const model::MatrixRep& rep,
                            const data::MixedOrdinalStats& stats,
                            const ThresholdLayout& layout,
                            const WhitenFactors& factors,
                            const Eigen::VectorXd& theta,
                            OrdinalParameterization parameterization) {
  const Eigen::Index q = theta.size();
  const auto Ws = dense_weights_from_factors(factors);
  Eigen::MatrixXd C;
  if (Ws.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "mixed ordinal Newton parts: weight block count mismatch"));
  }
  auto n_or = total_n_obs(stats);
  if (!n_or.has_value()) return std::unexpected(fit_to_post(n_or.error()));
  Estimates at;
  at.theta = theta;
  auto H = mixed_observed_bread_analytic(pt, rep, stats, at, layout, Ws,
      Eigen::MatrixXd::Identity(q, q), parameterization, &C);
  if (!H.has_value()) return std::unexpected(H.error());
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) return std::unexpected(model_to_post(ev.error()));
  auto eval = ev->evaluate(theta, true, true);
  if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
  const Eigen::MatrixXd Delta_full = mixed_moment_jacobian(
      stats, layout, eval->moments, eval->J_sigma, eval->J_mu, theta, parameterization);
  Eigen::MatrixXd A(Delta_full.rows(), q);
  Eigen::VectorXd score(Delta_full.rows());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    if (Ws[b].rows() != mb || off + mb > Delta_full.rows()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "mixed ordinal Newton parts: weight dimension mismatch"));
    }
    const Eigen::MatrixXd Delta = Delta_full.block(off, 0, mb, q);
    const double root_n = std::sqrt(static_cast<double>(stats.n_obs[b]));
    A.middleRows(off, mb) = factors[b].t_apply(root_n, Delta);
    const Eigen::VectorXd residual = mixed_model_moments(
        stats, layout, eval->moments, theta, b, parameterization) - stats.moments[b];
    score.segment(off, mb) = factors[b].t_apply(root_n, residual);
    off += mb;
  }
  // Keep the factor and residual correction: neither is recovered by
  // subtracting or factoring rounded cross-products in weak directions.
  Eigen::MatrixXd M = A.transpose() * A;
  C *= static_cast<double>(*n_or);
  Eigen::MatrixXd observed = M + C;
  return OrdinalNewtonRaw{std::move(observed), std::move(M), std::move(C),
                          std::move(A), std::move(score)};
}


fit_expected<Eigen::VectorXd>
mixed_ordinal_residuals(const data::MixedOrdinalStats& stats,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const WhitenFactors& factors,
                        const Eigen::VectorXd& theta,
                        OrdinalParameterization param) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::VectorXd out(mixed_moment_rows(stats));
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    Eigen::VectorXd d =
        mixed_model_moments(stats, layout, moments, theta, b, param) -
        stats.moments[b];
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.segment(off, d.size()) = factors[b].t_apply(sw, d);
    off += d.size();
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "mixed ordinal LS residuals contain non-finite values"));
  }
  return out;
}


fit_expected<Eigen::MatrixXd>
mixed_ordinal_jacobian(const data::MixedOrdinalStats& stats,
                       const ThresholdLayout& layout,
                       const model::ImpliedMoments& moments,
                       const Eigen::MatrixXd& J_sigma,
                       const Eigen::MatrixXd& J_mu,
                       const WhitenFactors& factors,
                       const Eigen::VectorXd& theta,
                       OrdinalParameterization param,
                       const Eigen::MatrixXd* J_theta) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::MatrixXd Jfull =
      mixed_moment_jacobian(stats, layout, moments, J_sigma, J_mu, theta,
                            param, J_theta);
  Eigen::MatrixXd out(Jfull.rows(), Jfull.cols());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.block(off, 0, mb, Jfull.cols()) =
        factors[b].t_apply(sw, Jfull.block(off, 0, mb, Jfull.cols()));
    off += mb;
  }
  return out;
}


data::SampleStats sample_stats_for_starts(const data::OrdinalStats& stats) {
  data::SampleStats samp;
  samp.S = stats.R;
  samp.n_obs = stats.n_obs;
  return samp;
}

}  // namespace detail_ordinal


}  // namespace magmaan::estimate
