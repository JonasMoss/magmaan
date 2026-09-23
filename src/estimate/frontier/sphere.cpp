#include "magmaan/estimate/frontier/sphere.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include "magmaan/error.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/evaluate.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/parameter_map.hpp"
#include "magmaan/optim/reparameterize.hpp"
#include "magmaan/optim/terminal_audit.hpp"

#include "../detail_backend_dispatch.hpp"

namespace magmaan::estimate::frontier {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

std::size_t idx(std::int32_t i) { return static_cast<std::size_t>(i); }

FitError sphere_err(const char* who, std::string detail) {
  return FitError{FitError::Kind::NumericIssue,
                  std::string(who) + ": " + std::move(detail), 0, 0.0};
}

// Per-unit piece of the sphere map. The unit's loadings in raw coordinates are
// lambda = D * Q * beta / ||beta||, with Q an orthonormal basis of D^{-1} W.
struct UnitMap {
  Eigen::MatrixXd Q;
  Eigen::VectorXd D;
  // params[m][j]: internal free index of loading j of member m; -1 if fixed.
  std::vector<std::vector<Eigen::Index>> params;
  Eigen::Index offset = 0;  // position of beta in u
  Eigen::Index dim = 0;
};

// Everything the sphere problem needs. Held behind a shared_ptr so the map's
// closures stay valid for the life of the problem.
struct SphereSetup {
  GaugePlan                 plan;
  spec::LatentStructure     pt_user;
  spec::LatentStructure     pt_int;
  EqConstraints             con_user;
  EqConstraints             con_int;
  NonlinearEqConstraints    nl_int;
  Eigen::MatrixXd           K_rest;      // columns of con_int.K not touching loadings
  std::vector<Eigen::Index> rest_cols;
  std::vector<UnitMap>      units;
  Eigen::Index              n_rest = 0;
  Eigen::Index              n_u = 0;
  double                    pin_scale = 0.0;  // sqrt(2 rho)
};

// Internal gauge-free partable: loadings of gauge units whose span row is not
// identically zero become free, released std.lv variances become free, and
// the constraints that encoded the loading sets are removed (the span now
// carries them).
fit_expected<spec::LatentStructure>
internal_partable(const spec::LatentStructure& pt, const GaugePlan& plan,
                  std::vector<char>& loading_param, const char* who) {
  spec::LatentStructure out = pt;
  const std::int32_t n0 = pt.n_free();
  std::int32_t next = n0 + 1;
  std::vector<std::int32_t> released;  // new free indices, 1-based
  std::vector<std::int32_t> loading_free;  // 1-based free indices of loadings

  for (const auto& u : plan.units) {
    for (const auto& member : u.loading_rows) {
      for (std::size_t j = 0; j < member.size(); ++j) {
        const auto i = idx(member[j]);
        const bool zero_row =
            u.basis.row(static_cast<Eigen::Index>(j)).cwiseAbs().maxCoeff() < 1e-14;
        if (out.free[i] > 0) {
          loading_free.push_back(out.free[i]);
          continue;
        }
        if (zero_row) {
          if (std::abs(out.fixed_value[i]) > 1e-12) {
            return std::unexpected(sphere_err(who,
                "internal error: a loading outside the loading span is fixed "
                "at a nonzero value"));
          }
          continue;
        }
        out.free[i] = next;
        out.fixed_value[i] = kNaN;
        released.push_back(next);
        loading_free.push_back(next);
        ++next;
      }
    }
    if (u.kind == GaugeKind::Linear) {
      const auto i = idx(u.variance_row);
      out.free[i] = next;
      out.fixed_value[i] = kNaN;
      released.push_back(next);
      ++next;
    }
  }
  const std::int32_t n_int = next - 1;
  loading_param.assign(idx(n_int), 0);
  for (auto f : loading_free) loading_param[idx(f - 1)] = 1;

  // Merge groups: loadings of units become singletons; new params singletons.
  if (!pt.eq_groups.empty()) {
    std::int32_t fresh = 0;
    for (auto g : pt.eq_groups) fresh = std::max(fresh, g + 1);
    out.eq_groups.resize(idx(n_int));
    for (std::int32_t p = 0; p < n_int; ++p) {
      if (p >= n0 || loading_param[idx(p)]) out.eq_groups[idx(p)] = fresh++;
    }
  }

  // General-linear rows: widen, drop rows that live on unit loadings.
  const std::size_t n_lin = pt.lin_constraint_d.size();
  if (n_lin > 0) {
    std::vector<double> R;
    std::vector<double> d;
    for (std::size_t r = 0; r < n_lin; ++r) {
      bool on_loading = false;
      bool elsewhere = false;
      for (std::int32_t c = 0; c < n0; ++c) {
        const double v = pt.lin_constraint_R[r * idx(n0) + idx(c)];
        if (v == 0.0) continue;
        if (loading_param[idx(c)]) on_loading = true;
        else elsewhere = true;
      }
      if (on_loading && elsewhere) {
        return std::unexpected(sphere_err(who,
            "internal error: a linear constraint mixes unit loadings with "
            "other parameters"));
      }
      if (on_loading) continue;
      for (std::int32_t c = 0; c < n_int; ++c) {
        R.push_back(c < n0 ? pt.lin_constraint_R[r * idx(n0) + idx(c)] : 0.0);
      }
      d.push_back(pt.lin_constraint_d[r]);
    }
    out.lin_constraint_R = std::move(R);
    out.lin_constraint_d = std::move(d);
  }
  return out;
}

Eigen::VectorXd unit_metric(const spec::LatentStructure& pt,
                            const SampleStats& samp, const GaugeUnit& u,
                            SphereMetric metric) {
  const auto n = static_cast<Eigen::Index>(u.loading_rows.front().size());
  Eigen::VectorXd D = Eigen::VectorXd::Ones(n);
  if (metric == SphereMetric::Raw) return D;
  for (Eigen::Index j = 0; j < n; ++j) {
    double acc = 0.0;
    int cnt = 0;
    for (std::size_t m = 0; m < u.loading_rows.size(); ++m) {
      const auto row = idx(u.loading_rows[m][idx(static_cast<std::int32_t>(j))]);
      const std::int32_t var = pt.rhs_var[row];
      if (var < 0 || idx(var) >= pt.ov_pos.size()) continue;
      const std::int32_t pos = pt.ov_pos[idx(var)];
      const std::int32_t b = u.blocks[m] - 1;
      if (pos < 0 || b < 0 || idx(b) >= samp.S.size()) continue;
      const auto& S = samp.S[idx(b)];
      if (pos >= S.rows()) continue;
      const double s = S(pos, pos);
      if (s > 0.0 && std::isfinite(s)) {
        acc += s;
        ++cnt;
      }
    }
    if (cnt > 0) D(j) = std::sqrt(acc / cnt);
  }
  return D;
}

fit_expected<std::shared_ptr<SphereSetup>>
build_setup(const spec::LatentStructure& pt, const model::MatrixRep& rep,
            const SampleStats& samp, const Bounds& bounds,
            const SphereOptions& sopts, const char* who) {
  if (!(sopts.pin_weight > 0.0) || !std::isfinite(sopts.pin_weight)) {
    return std::unexpected(sphere_err(who, "pin_weight must be positive and finite"));
  }
  auto s = std::make_shared<SphereSetup>();
  s->pt_user = pt;
  if (auto e = resolve_fixed_x_from_sample(s->pt_user, rep, samp); !e.has_value())
    return std::unexpected(e.error());

  GaugeAnalysisOptions gopts;
  gopts.bounds = bounds;
  auto plan = analyze_gauge(s->pt_user, gopts);
  if (!plan) return std::unexpected(sphere_err(who, plan.error().detail));
  s->plan = std::move(*plan);

  auto con_user = build_eq_constraints(s->pt_user, /*allow_nonlinear=*/true);
  if (!con_user) return std::unexpected(sphere_err(who, con_user.error().detail));
  s->con_user = std::move(*con_user);

  std::vector<char> loading_param;
  auto pt_int = internal_partable(s->pt_user, s->plan, loading_param, who);
  if (!pt_int) return std::unexpected(pt_int.error());
  s->pt_int = std::move(*pt_int);

  auto con_int = build_eq_constraints(s->pt_int, /*allow_nonlinear=*/true);
  if (!con_int) return std::unexpected(sphere_err(who, con_int.error().detail));
  s->con_int = std::move(*con_int);
  s->nl_int = build_nl_constraints(s->pt_int);

  const Eigen::Index n_int = s->pt_int.n_free();
  const Eigen::MatrixXd& K = s->con_int.Kmat;
  for (Eigen::Index c = 0; c < K.cols(); ++c) {
    bool touches = false;
    for (Eigen::Index p = 0; p < n_int && !touches; ++p)
      if (loading_param[idx(static_cast<std::int32_t>(p))] && K(p, c) != 0.0)
        touches = true;
    if (!touches) s->rest_cols.push_back(c);
  }
  s->n_rest = static_cast<Eigen::Index>(s->rest_cols.size());
  s->K_rest.resize(n_int, s->n_rest);
  for (Eigen::Index k = 0; k < s->n_rest; ++k)
    s->K_rest.col(k) = K.col(s->rest_cols[idx(static_cast<std::int32_t>(k))]);

  Eigen::Index off = s->n_rest;
  for (const auto& u : s->plan.units) {
    UnitMap um;
    um.D = unit_metric(s->pt_user, samp, u, sopts.metric);
    const Eigen::MatrixXd scaled = um.D.cwiseInverse().asDiagonal() * u.basis;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(scaled);
    um.Q = qr.householderQ() * Eigen::MatrixXd::Identity(scaled.rows(), scaled.cols());
    um.dim = um.Q.cols();
    um.offset = off;
    off += um.dim;
    for (const auto& member : u.loading_rows) {
      std::vector<Eigen::Index> ps;
      for (auto row : member) {
        const auto f = s->pt_int.free[idx(row)];
        ps.push_back(f > 0 ? static_cast<Eigen::Index>(f - 1) : -1);
      }
      um.params.push_back(std::move(ps));
    }
    s->units.push_back(std::move(um));
  }
  s->n_u = off;
  s->pin_scale = std::sqrt(2.0 * sopts.pin_weight);
  return s;
}

Eigen::VectorXd expand_u(const SphereSetup& s, const Eigen::VectorXd& u) {
  Eigen::VectorXd theta = s.con_int.theta0;
  if (s.n_rest > 0) theta.noalias() += s.K_rest * u.head(s.n_rest);
  for (const auto& um : s.units) {
    const Eigen::VectorXd beta = u.segment(um.offset, um.dim);
    const double nb = beta.norm();
    const Eigen::VectorXd lam =
        nb > 0.0 ? Eigen::VectorXd(um.D.cwiseProduct(um.Q * beta) / nb)
                 : Eigen::VectorXd::Constant(um.D.size(), kNaN);
    for (const auto& member : um.params)
      for (std::size_t j = 0; j < member.size(); ++j)
        if (member[j] >= 0) theta(member[j]) = lam(static_cast<Eigen::Index>(j));
  }
  return theta;
}

Eigen::MatrixXd jacobian_u(const SphereSetup& s, const Eigen::VectorXd& u) {
  const Eigen::Index n_int = s.con_int.theta0.size();
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n_int, s.n_u);
  if (s.n_rest > 0) G.leftCols(s.n_rest) = s.K_rest;
  for (const auto& um : s.units) {
    const Eigen::VectorXd beta = u.segment(um.offset, um.dim);
    const double nb = beta.norm();
    if (!(nb > 0.0)) continue;
    const Eigen::VectorXd a = um.Q * beta / nb;
    const Eigen::Index n = a.size();
    const Eigen::MatrixXd J =
        um.D.asDiagonal() *
        ((Eigen::MatrixXd::Identity(n, n) - a * a.transpose()) * um.Q) / nb;
    for (const auto& member : um.params)
      for (std::size_t j = 0; j < member.size(); ++j)
        if (member[j] >= 0)
          G.row(member[j]).segment(um.offset, um.dim) =
              J.row(static_cast<Eigen::Index>(j));
  }
  return G;
}

optim::ParameterMap make_map(const std::shared_ptr<SphereSetup>& s) {
  optim::ParameterMap map;
  map.n_param = s->n_u;
  map.expand = [s](const Eigen::VectorXd& u) { return expand_u(*s, u); };
  map.jacobian = [s](const Eigen::VectorXd& u) { return jacobian_u(*s, u); };
  if (!s->units.empty()) {
    map.penalty_residual = [s](const Eigen::VectorXd& u) {
      Eigen::VectorXd r(static_cast<Eigen::Index>(s->units.size()));
      for (std::size_t k = 0; k < s->units.size(); ++k) {
        const auto& um = s->units[k];
        r(static_cast<Eigen::Index>(k)) =
            s->pin_scale * (u.segment(um.offset, um.dim).squaredNorm() - 1.0);
      }
      return r;
    };
    map.penalty_jacobian = [s](const Eigen::VectorXd& u) {
      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(s->units.size()), s->n_u);
      for (std::size_t k = 0; k < s->units.size(); ++k) {
        const auto& um = s->units[k];
        J.row(static_cast<Eigen::Index>(k)).segment(um.offset, um.dim) =
            2.0 * s->pin_scale * u.segment(um.offset, um.dim).transpose();
      }
      return J;
    };
  }
  return map;
}

// Start in u from user-chart start values.
fit_expected<Eigen::VectorXd>
start_u(const SphereSetup& s, const Eigen::VectorXd& x0_user, const char* who) {
  Eigen::VectorXd x0 = x0_user;
  if (s.con_user.active() && s.con_user.n_alpha > 0)
    x0 = s.con_user.expand(s.con_user.contract(x0));
  const Eigen::VectorXd rows0 = row_values(s.pt_user, x0);
  std::vector<Eigen::VectorXd> metric;
  for (const auto& um : s.units) metric.push_back(um.D);
  auto c = scales_to_sphere(s.plan, rows0, metric);
  if (!c) return std::unexpected(sphere_err(who, c.error().detail));
  const Eigen::VectorXd rows_sph = rescale_rows(s.plan, rows0, *c);
  Eigen::VectorXd theta_int = theta_from_rows(s.pt_int, rows_sph);
  for (Eigen::Index p = 0; p < theta_int.size(); ++p) {
    if (!std::isfinite(theta_int(p))) {
      return std::unexpected(sphere_err(who,
          "start values are not finite in the sphere chart"));
    }
  }
  Eigen::VectorXd u = Eigen::VectorXd::Zero(s.n_u);
  if (s.n_rest > 0) {
    const Eigen::VectorXd alpha = s.con_int.contract(theta_int);
    for (Eigen::Index k = 0; k < s.n_rest; ++k)
      u(k) = alpha(s.rest_cols[idx(static_cast<std::int32_t>(k))]);
  }
  for (std::size_t k = 0; k < s.units.size(); ++k) {
    const auto& um = s.units[k];
    const auto& rows = s.plan.units[k].loading_rows.front();
    Eigen::VectorXd lam(static_cast<Eigen::Index>(rows.size()));
    for (std::size_t j = 0; j < rows.size(); ++j)
      lam(static_cast<Eigen::Index>(j)) = rows_sph(rows[j]);
    Eigen::VectorXd beta = um.Q.transpose() * lam.cwiseQuotient(um.D);
    const double nb = beta.norm();
    if (!(nb > 0.0)) {
      beta = Eigen::VectorXd::Unit(um.dim, 0);
    } else {
      beta /= nb;
    }
    u.segment(um.offset, um.dim) = beta;
  }
  return u;
}

fit_expected<Bounds> driven_bounds(const SphereSetup& s, const Bounds& bounds,
                                   Eigen::VectorXd& u0, const char* who) {
  if (bounds.empty()) return Bounds{};
  if (s.con_int.group.empty()) {
    return std::unexpected(sphere_err(who,
        "box bounds together with general linear constraints are not "
        "supported in the sphere chart"));
  }
  const Eigen::Index n0 = s.pt_user.n_free();
  const Eigen::Index n_int = s.pt_int.n_free();
  Bounds bi;
  bi.lower = Eigen::VectorXd::Constant(n_int, -kInf);
  bi.upper = Eigen::VectorXd::Constant(n_int, kInf);
  bi.lower.head(n0) = bounds.lower;
  bi.upper.head(n0) = bounds.upper;
  for (const auto& u : s.plan.units) {
    if (u.kind != GaugeKind::Linear) continue;
    const auto f = s.pt_int.free[idx(u.variance_row)];
    if (f > 0) bi.lower(f - 1) = 0.0;
  }
  const Bounds folded = optim::fold_alpha_bounds(s.con_int, bi);
  Bounds out;
  out.lower = Eigen::VectorXd::Constant(s.n_u, -kInf);
  out.upper = Eigen::VectorXd::Constant(s.n_u, kInf);
  for (Eigen::Index k = 0; k < s.n_rest; ++k) {
    const auto c = s.rest_cols[idx(static_cast<std::int32_t>(k))];
    out.lower(k) = folded.lower(c);
    out.upper(k) = folded.upper(c);
  }
  u0 = u0.cwiseMax(out.lower).cwiseMin(out.upper);
  return out;
}

fit_expected<optim::OptimResult>
run_driven(const SphereSetup& s, const optim::ParameterMap& map,
           const optim::ScalarProblem* scalar, const optim::GmmProblem* ls,
           const Eigen::VectorXd& u0, const Bounds& ub, Backend backend,
           OptimOptions opts, const char* who) {
  if (s.nl_int.active()) {
    const optim::ScalarProblem sp =
        scalar ? optim::reparameterize(*scalar, map)
               : optim::scalarize(optim::reparameterize(*ls, map));
    const auto& nl = s.nl_int;
    auto h = [&nl, &map](const Eigen::VectorXd& u) { return nl.h(map.expand(u)); };
    auto J = [&nl, &map](const Eigen::VectorXd& u) {
      return Eigen::MatrixXd(nl.jacobian(map.expand(u)) * map.jacobian(u));
    };
    return backend_dispatch::dispatch_scalar_constrained(sp, h, J, nl.m(), u0, ub,
                                               backend, std::move(opts), who);
  }
  if (scalar) {
    return backend_dispatch::dispatch_scalar(optim::reparameterize(*scalar, map), u0, ub,
                                   backend, std::move(opts));
  }
  return backend_dispatch::dispatch_gmm(optim::reparameterize(*ls, map), u0, ub, backend,
                              std::move(opts));
}

using Finalizer =
    std::function<fit_expected<Estimates>(const Eigen::VectorXd& theta_user)>;

// Translate the driven solution into the user's chart and finalize there.
fit_expected<SphereFit>
finish(const std::shared_ptr<SphereSetup>& s, const Eigen::VectorXd& x0_user,
       const optim::OptimResult& r, const optim::ScalarProblem& internal_obj,
       const Finalizer& finalize, double pole_tol, const char* who) {
  SphereFit out;
  auto& rep_out = out.report;
  rep_out.plan = s->plan;
  rep_out.internal_pt = s->pt_int;
  rep_out.driven = r.x;
  rep_out.internal_theta = expand_u(*s, r.x);
  {
    Eigen::VectorXd g(rep_out.internal_theta.size());
    rep_out.fmin_internal = internal_obj.f(rep_out.internal_theta, g);
  }
  double pin = 0.0;
  for (const auto& um : s->units)
    pin = std::max(pin, std::abs(r.x.segment(um.offset, um.dim).norm() - 1.0));
  rep_out.pin_residual = pin;
  rep_out.optimizer_status = r.status;
  rep_out.iterations = r.iterations;
  rep_out.f_evals = r.f_evals;
  rep_out.g_evals = r.g_evals;
  rep_out.grad_inf_norm = r.grad_inf_norm;
  rep_out.driven_audit = r.audit;

  const Eigen::VectorXd rows_int = row_values(s->pt_int, rep_out.internal_theta);
  const Eigen::VectorXd rows_ref = row_values(s->pt_user, x0_user);
  rep_out.scales = scales_to_user_chart(s->plan, rows_int, &rows_ref, pole_tol);
  if (!rep_out.scales.singular_units.empty()) {
    out.user_chart = false;
    return out;
  }
  const Eigen::VectorXd rows_user =
      rescale_rows(s->plan, rows_int, rep_out.scales.scales);
  rep_out.residual = chart_residual(s->pt_user, s->con_user, rows_user);
  Eigen::VectorXd theta = theta_from_rows(s->pt_user, rows_user);
  if (s->con_user.active() && s->con_user.n_alpha > 0)
    theta = s->con_user.expand(s->con_user.contract(theta));
  for (Eigen::Index p = 0; p < theta.size(); ++p) {
    if (!std::isfinite(theta(p))) {
      return std::unexpected(sphere_err(who,
          "translated user-chart estimate is not finite"));
    }
  }
  auto est = finalize(theta);
  if (!est) return std::unexpected(est.error());
  est->iterations = r.iterations;
  est->f_evals = r.f_evals;
  est->g_evals = r.g_evals;
  est->optimizer_status = r.status;
  est->grad_inf_norm = r.grad_inf_norm;
  out.estimates = std::move(*est);
  out.user_chart = true;
  return out;
}

}  // namespace

fit_expected<SphereFit>
fit_ml_sphere(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              Bounds bounds, Backend backend, OptimOptions opts,
              SphereOptions sphere) {
  constexpr const char* who = "fit_ml_sphere";
  if (x0.size() != pt.n_free()) {
    return std::unexpected(sphere_err(who, "x0 size does not match n_free"));
  }
  auto s = build_setup(pt, rep, samp, bounds, sphere, who);
  if (!s) return std::unexpected(s.error());
  auto ev = model::ModelEvaluator::build((*s)->pt_int, rep);
  if (!ev) {
    return std::unexpected(sphere_err(who, "ModelEvaluator::build failed: " +
                                               ev.error().detail));
  }
  auto obj = ml_objective(*ev, samp);
  if (!obj) return std::unexpected(obj.error());
  const optim::ParameterMap map = make_map(*s);
  auto u0 = start_u(**s, x0, who);
  if (!u0) return std::unexpected(u0.error());
  auto ub = driven_bounds(**s, bounds, *u0, who);
  if (!ub) return std::unexpected(ub.error());
  opts.ml_sample_scaling = false;
  auto r = run_driven(**s, map, &*obj, nullptr, *u0, *ub, backend, opts, who);
  if (!r) return std::unexpected(r.error());
  const auto& pt_user = (*s)->pt_user;
  Finalizer finalize = [&](const Eigen::VectorXd& theta) {
    return evaluate_at(pt_user, rep, samp, theta, Estimator::ML, {}, bounds);
  };
  return finish(*s, x0, *r, *obj, finalize, sphere.pole_tol, who);
}

fit_expected<SphereProblem>
ml_sphere_problem(spec::LatentStructure pt, const model::MatrixRep& rep,
                  const SampleStats& samp, const Eigen::VectorXd& x0,
                  SphereOptions sphere) {
  constexpr const char* who = "ml_sphere_problem";
  if (x0.size() != pt.n_free()) {
    return std::unexpected(sphere_err(who, "x0 size does not match n_free"));
  }
  auto s = build_setup(pt, rep, samp, {}, sphere, who);
  if (!s) return std::unexpected(s.error());
  auto ev_or = model::ModelEvaluator::build((*s)->pt_int, rep);
  if (!ev_or) {
    return std::unexpected(sphere_err(who, "ModelEvaluator::build failed: " +
                                               ev_or.error().detail));
  }
  auto ev = std::make_shared<model::ModelEvaluator>(std::move(*ev_or));
  auto obj = ml_objective(*ev, samp);
  if (!obj) return std::unexpected(obj.error());
  auto u0 = start_u(**s, x0, who);
  if (!u0) return std::unexpected(u0.error());
  const optim::ParameterMap map = make_map(*s);
  optim::ScalarProblem driven = optim::reparameterize(*obj, map);
  // Keep the evaluator alive for the problem's lifetime.
  driven.f = [f = driven.f, ev](const Eigen::VectorXd& u, Eigen::VectorXd& g) {
    return f(u, g);
  };
  SphereProblem out;
  out.problem = std::move(driven);
  out.start = std::move(*u0);
  out.plan = (*s)->plan;
  out.internal_pt = (*s)->pt_int;
  out.internal_theta = [setup = *s](const Eigen::VectorXd& u) {
    return expand_u(*setup, u);
  };
  return out;
}

namespace {

fit_expected<SphereFit>
fit_ls_sphere(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              gmm::Weight weight, bool gls, Bounds bounds, Backend backend,
              OptimOptions opts, SphereOptions sphere, const char* who) {
  if (x0.size() != pt.n_free()) {
    return std::unexpected(sphere_err(who, "x0 size does not match n_free"));
  }
  auto s = build_setup(pt, rep, samp, bounds, sphere, who);
  if (!s) return std::unexpected(s.error());
  auto ev = model::ModelEvaluator::build((*s)->pt_int, rep);
  if (!ev) {
    return std::unexpected(sphere_err(who, "ModelEvaluator::build failed: " +
                                               ev.error().detail));
  }
  auto u0 = start_u(**s, x0, who);
  if (!u0) return std::unexpected(u0.error());
  const Eigen::VectorXd theta0 = expand_u(**s, *u0);
  if (gls) {
    auto W = gmm::normal_theory_weight(*ev, samp, theta0);
    if (!W) return std::unexpected(W.error());
    weight = std::move(*W);
  }
  auto prob = gmm::residuals(*ev, samp, theta0, weight);
  if (!prob) return std::unexpected(prob.error());
  const optim::ScalarProblem internal_obj = optim::scalarize(*prob);
  const optim::ParameterMap map = make_map(*s);
  auto ub = driven_bounds(**s, bounds, *u0, who);
  if (!ub) return std::unexpected(ub.error());
  auto r = run_driven(**s, map, nullptr, &*prob, *u0, *ub, backend, opts, who);
  if (!r) return std::unexpected(r.error());
  const Estimator est = gls ? Estimator::GLS
                            : (weight.empty() ? Estimator::ULS : Estimator::WLS);
  const auto& pt_user = (*s)->pt_user;
  const gmm::Weight final_weight = gls ? gmm::Weight{} : weight;
  Finalizer finalize = [&](const Eigen::VectorXd& theta) {
    return evaluate_at(pt_user, rep, samp, theta, est, final_weight, bounds);
  };
  return finish(*s, x0, *r, internal_obj, finalize, sphere.pole_tol, who);
}

}  // namespace

namespace {

// FIML objective on the 0.5 F scale, as in fit_fiml.
optim::ScalarProblem fiml_objective(const model::ModelEvaluator& ev,
                                    const data::RawData& raw,
                                    const fiml::FIMLCache& cache) {
  optim::ScalarProblem prob;
  prob.n_param = static_cast<Eigen::Index>(ev.param_locations().size());
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.f = [&ev, &raw, &cache](const Eigen::VectorXd& x,
                               Eigen::VectorXd& grad) -> double {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      grad = Eigen::VectorXd::Zero(x.size());
      return kInf;
    }
    auto vg = fiml::FIML{}.value_gradient(raw, cache, eval->moments,
                                          eval->J_sigma, eval->J_mu);
    if (!vg.has_value()) {
      grad = Eigen::VectorXd::Zero(x.size());
      return kInf;
    }
    grad = 0.5 * vg->gradient;
    return 0.5 * vg->value;
  };
  return prob;
}

}  // namespace

fit_expected<SphereFit>
fit_fiml_sphere(spec::LatentStructure pt, const model::MatrixRep& rep,
                const data::RawData& raw, const Eigen::VectorXd& x0,
                Backend backend, OptimOptions opts, SphereOptions sphere) {
  constexpr const char* who = "fit_fiml_sphere";
  if (x0.size() != pt.n_free()) {
    return std::unexpected(sphere_err(who, "x0 size does not match n_free"));
  }
  if (auto e = fiml::validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value())
    return std::unexpected(e.error());
  auto cache = fiml::FIML{}.prepare(raw);
  if (!cache) return std::unexpected(cache.error());
  auto start_samp = fiml::fiml_start_sample_stats(raw);
  if (!start_samp) return std::unexpected(start_samp.error());

  auto s = build_setup(pt, rep, *start_samp, {}, sphere, who);
  if (!s) return std::unexpected(s.error());
  auto ev = model::ModelEvaluator::build((*s)->pt_int, rep);
  if (!ev) {
    return std::unexpected(sphere_err(who, "ModelEvaluator::build failed: " +
                                               ev.error().detail));
  }
  const optim::ScalarProblem obj = fiml_objective(*ev, raw, *cache);
  const optim::ParameterMap map = make_map(*s);
  auto u0 = start_u(**s, x0, who);
  if (!u0) return std::unexpected(u0.error());
  auto r = run_driven(**s, map, &obj, nullptr, *u0, Bounds{}, backend, opts, who);
  if (!r) return std::unexpected(r.error());

  const auto& setup = **s;
  auto ev_user = model::ModelEvaluator::build(setup.pt_user, rep);
  if (!ev_user) {
    return std::unexpected(sphere_err(who, "ModelEvaluator::build failed: " +
                                               ev_user.error().detail));
  }
  const optim::ScalarProblem user_obj = fiml_objective(*ev_user, raw, *cache);
  const NonlinearEqConstraints nl_user = build_nl_constraints(setup.pt_user);
  Finalizer finalize = [&](const Eigen::VectorXd& theta)
      -> fit_expected<Estimates> {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(theta.size());
    const double f_at = user_obj.f(theta, grad);
    const Eigen::VectorXd lo = Eigen::VectorXd::Constant(theta.size(), -kInf);
    const Eigen::VectorXd hi = Eigen::VectorXd::Constant(theta.size(), kInf);
    Estimates est;
    est.theta = theta;
    est.fmin = f_at;
    est.audit = optim::audit_terminal_iterate(user_obj.f, theta, f_at, lo, hi);
    est.diagnostics = finalize_fit_diagnostics(theta, setup.pt_user, *ev_user,
                                               setup.con_user, nl_user, Bounds{});
    audit_full_model_fit(est.diagnostics, theta, grad, f_at, f_at,
                         setup.pt_user, *ev_user, setup.con_user, nl_user,
                         Bounds{});
    return est;
  };
  return finish(*s, x0, *r, obj, finalize, sphere.pole_tol, who);
}

namespace {

// Append a synthetic `==` row so that a programmatic nonlinear constraint has
// a partable row, as the partable-derived ones do.
void append_constraint_row(spec::LatentStructure& pt) {
  pt.op.push_back(parse::Op::EqConstraint);
  pt.lhs_var.push_back(-1);
  pt.rhs_var.push_back(-1);
  pt.group.push_back(0);
  if (!pt.level.empty()) pt.level.push_back(0);
  pt.free.push_back(0);
  pt.exo.push_back(0);
  pt.fixed_value.push_back(kNaN);
  for (auto& [k, v] : pt.extra_real) { (void)k; v.push_back(kNaN); }
  for (auto& [k, v] : pt.extra_int) { (void)k; v.push_back(0); }
  for (auto& [k, v] : pt.extra_str) { (void)k; v.emplace_back(); }
}

// sum_j (theta_{p_j} / d_j)^2 - 1 as a name-free expression tree.
spec::NlConstraint sphere_constraint(const std::vector<Eigen::Index>& params,
                                     const Eigen::VectorXd& D) {
  using Node = spec::NlExprNode;
  spec::NlConstraint c;
  auto push = [&c](Node n) {
    c.nodes.push_back(n);
    return static_cast<std::int32_t>(c.nodes.size() - 1);
  };
  std::int32_t sum = -1;
  for (std::size_t j = 0; j < params.size(); ++j) {
    if (params[j] < 0) continue;
    Node p;
    p.kind = Node::Kind::Param;
    p.free_idx = static_cast<std::int32_t>(params[j]);
    const std::int32_t ip = push(p);
    Node k;
    k.kind = Node::Kind::Const;
    k.constant = 1.0 / D(static_cast<Eigen::Index>(j));
    const std::int32_t ik = push(k);
    Node scaled;
    scaled.kind = Node::Kind::Binary;
    scaled.bin_op = parse::BinOp::Mul;
    scaled.lhs = ip;
    scaled.rhs = ik;
    const std::int32_t is = push(scaled);
    Node sq;
    sq.kind = Node::Kind::Binary;
    sq.bin_op = parse::BinOp::Mul;
    sq.lhs = is;
    sq.rhs = is;
    const std::int32_t iq = push(sq);
    if (sum < 0) {
      sum = iq;
    } else {
      Node add;
      add.kind = Node::Kind::Binary;
      add.bin_op = parse::BinOp::Add;
      add.lhs = sum;
      add.rhs = iq;
      sum = push(add);
    }
  }
  Node one;
  one.kind = Node::Kind::Const;
  one.constant = 1.0;
  const std::int32_t i1 = push(one);
  Node h;
  h.kind = Node::Kind::Binary;
  h.bin_op = parse::BinOp::Sub;
  h.lhs = sum;
  h.rhs = i1;
  c.root = push(h);
  return c;
}

// The internal partable with the sphere as explicit constraints: span and tie
// rows as linear equalities, unit norm as one nonlinear equality per unit.
spec::LatentStructure constrained_internal_partable(const SphereSetup& s) {
  spec::LatentStructure pt = s.pt_int;
  const auto n_int = static_cast<std::size_t>(pt.n_free());
  auto add_lin = [&pt, n_int](const std::vector<std::pair<Eigen::Index, double>>& row) {
    std::vector<double> r(n_int, 0.0);
    bool any = false;
    for (auto [p, v] : row) {
      if (p < 0 || std::abs(v) < 1e-14) continue;
      r[static_cast<std::size_t>(p)] += v;
      any = true;
    }
    if (!any) return;
    pt.lin_constraint_R.insert(pt.lin_constraint_R.end(), r.begin(), r.end());
    pt.lin_constraint_d.push_back(0.0);
  };
  for (std::size_t k = 0; k < s.units.size(); ++k) {
    const auto& um = s.units[k];
    const auto& basis = s.plan.units[k].basis;
    const Eigen::Index n = basis.rows();
    const auto& p0 = um.params.front();
    if (basis.cols() < n) {
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
      const Eigen::MatrixXd Qf = qr.householderQ();
      const Eigen::MatrixXd perp = Qf.rightCols(n - basis.cols());
      for (Eigen::Index c = 0; c < perp.cols(); ++c) {
        std::vector<std::pair<Eigen::Index, double>> row;
        for (Eigen::Index j = 0; j < n; ++j)
          row.emplace_back(p0[static_cast<std::size_t>(j)], perp(j, c));
        add_lin(row);
      }
    }
    for (std::size_t m = 1; m < um.params.size(); ++m) {
      for (std::size_t j = 0; j < p0.size(); ++j) {
        if (p0[j] < 0 || um.params[m][j] < 0) continue;
        add_lin({{um.params[m][j], 1.0}, {p0[j], -1.0}});
      }
    }
    append_constraint_row(pt);
    pt.nonlinear_eq_rows.push_back(static_cast<std::int32_t>(pt.size() - 1));
    pt.nl_constraints.push_back(sphere_constraint(p0, um.D));
  }
  return pt;
}

}  // namespace

fit_expected<SphereFit>
fit_ml_psd_sphere(spec::LatentStructure pt, const model::MatrixRep& rep,
                  const SampleStats& samp, const Eigen::VectorXd& x0,
                  Backend backend, OptimOptions opts, PsdFitOptions psd_opts,
                  SphereOptions sphere) {
  constexpr const char* who = "fit_ml_psd_sphere";
  if (x0.size() != pt.n_free()) {
    return std::unexpected(sphere_err(who, "x0 size does not match n_free"));
  }
  auto s = build_setup(pt, rep, samp, {}, sphere, who);
  if (!s) return std::unexpected(s.error());
  const auto& setup = **s;
  auto u0 = start_u(setup, x0, who);
  if (!u0) return std::unexpected(u0.error());
  const Eigen::VectorXd theta0 = expand_u(setup, *u0);
  const spec::LatentStructure pt_c = constrained_internal_partable(setup);
  auto rep_c = model::build_matrix_rep(pt_c);
  if (!rep_c) {
    return std::unexpected(sphere_err(who, "build_matrix_rep failed: " +
                                               rep_c.error().detail));
  }

  auto internal = fit_ml_psd(pt_c, *rep_c, samp, theta0, backend, opts, psd_opts);
  if (!internal) return std::unexpected(internal.error());

  auto ev_int = model::ModelEvaluator::build(setup.pt_int, rep);
  if (!ev_int) {
    return std::unexpected(sphere_err(who, "ModelEvaluator::build failed: " +
                                               ev_int.error().detail));
  }
  auto obj_int = ml_objective(*ev_int, samp);
  if (!obj_int) return std::unexpected(obj_int.error());

  // Report the terminal point in driven sphere coordinates as well.
  optim::OptimResult r;
  r.x = Eigen::VectorXd::Zero(setup.n_u);
  {
    const Eigen::VectorXd& th = internal->theta;
    if (setup.n_rest > 0) {
      const Eigen::VectorXd alpha = setup.con_int.contract(th);
      for (Eigen::Index k = 0; k < setup.n_rest; ++k)
        r.x(k) = alpha(setup.rest_cols[idx(static_cast<std::int32_t>(k))]);
    }
    for (const auto& um : setup.units) {
      const auto& p0 = um.params.front();
      Eigen::VectorXd lam = Eigen::VectorXd::Zero(um.D.size());
      for (std::size_t j = 0; j < p0.size(); ++j)
        if (p0[j] >= 0) lam(static_cast<Eigen::Index>(j)) = th(p0[j]);
      r.x.segment(um.offset, um.dim) = um.Q.transpose() * lam.cwiseQuotient(um.D);
    }
  }
  r.fmin = internal->fmin;
  r.iterations = internal->iterations;
  r.f_evals = internal->f_evals;
  r.g_evals = internal->g_evals;
  r.status = internal->optimizer_status;
  r.grad_inf_norm = internal->grad_inf_norm;
  r.audit = internal->audit;

  auto ev_user = model::ModelEvaluator::build(setup.pt_user, rep);
  if (!ev_user) {
    return std::unexpected(sphere_err(who, "ModelEvaluator::build failed: " +
                                               ev_user.error().detail));
  }
  auto user_obj = ml_objective(*ev_user, samp);
  if (!user_obj) return std::unexpected(user_obj.error());
  const NonlinearEqConstraints nl_user = build_nl_constraints(setup.pt_user);
  Finalizer finalize = [&](const Eigen::VectorXd& theta)
      -> fit_expected<Estimates> {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(theta.size());
    const double f_at = user_obj->f(theta, grad);
    const Eigen::VectorXd lo = Eigen::VectorXd::Constant(theta.size(), -kInf);
    const Eigen::VectorXd hi = Eigen::VectorXd::Constant(theta.size(), kInf);
    Estimates est;
    est.theta = theta;
    est.fmin = f_at;
    est.audit = optim::audit_terminal_iterate(user_obj->f, theta, f_at, lo, hi);
    est.diagnostics = finalize_fit_diagnostics(theta, setup.pt_user, *ev_user,
                                               setup.con_user, nl_user, Bounds{});
    audit_full_model_fit(est.diagnostics, theta, grad, f_at, f_at,
                         setup.pt_user, *ev_user, setup.con_user, nl_user,
                         Bounds{}, StationarityDomain::Psd);
    return est;
  };
  auto out = finish(*s, x0, r, *obj_int, finalize, sphere.pole_tol, who);
  if (out) {
    // The sphere is an equality constraint here, not a pin: report its
    // residual in the pin slot.
    double pin = 0.0;
    for (const auto& um : setup.units) {
      const auto& p0 = um.params.front();
      Eigen::VectorXd lam = Eigen::VectorXd::Zero(um.D.size());
      for (std::size_t j = 0; j < p0.size(); ++j)
        if (p0[j] >= 0) lam(static_cast<Eigen::Index>(j)) = internal->theta(p0[j]);
      pin = std::max(pin, std::abs(lam.cwiseQuotient(um.D).norm() - 1.0));
    }
    out->report.pin_residual = pin;
    out->report.internal_theta = internal->theta;
  }
  return out;
}

fit_expected<SphereFit>
fit_gmm_sphere(spec::LatentStructure pt, const model::MatrixRep& rep,
               const SampleStats& samp, const Eigen::VectorXd& x0,
               gmm::Weight weight, Bounds bounds, Backend backend,
               OptimOptions opts, SphereOptions sphere) {
  return fit_ls_sphere(std::move(pt), rep, samp, x0, std::move(weight), false,
                       std::move(bounds), backend, std::move(opts), sphere,
                       "fit_gmm_sphere");
}

fit_expected<SphereFit>
fit_gls_sphere(spec::LatentStructure pt, const model::MatrixRep& rep,
               const SampleStats& samp, const Eigen::VectorXd& x0,
               Bounds bounds, Backend backend, OptimOptions opts,
               SphereOptions sphere) {
  return fit_ls_sphere(std::move(pt), rep, samp, x0, {}, true,
                       std::move(bounds), backend, std::move(opts), sphere,
                       "fit_gls_sphere");
}

}  // namespace magmaan::estimate::frontier
