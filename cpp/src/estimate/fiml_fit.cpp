#include "fiml_internal.hpp"

namespace magmaan::estimate::fiml {
namespace internal {

// FIML Newton check: the analytic observed information of the fitted
// objective is both the curvature and the metric. `gradient` and `value` are
// the optimiser adapter's per-observation ½F scale at est.theta; the cache is
// the fit's own (raw-data FIML or a pattern NTML target).
void attach_fiml_newton_accuracy(Estimates& est, const spec::LatentStructure& pt,
                                 const model::MatrixRep& rep, const FIMLCache& cache,
                                 const SampleStats& start_samp,
                                 const Eigen::VectorXd& gradient, double value,
                                 StationarityDomain domain) {
  estimate::frontier::NewtonDerivatives d;
  d.theta = est.theta;
  d.objective_kind = NewtonObjectiveKind::Fiml;
  d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  const double n = static_cast<double>(cache.n_total);
  d.n_obs = n;
  d.native_to_total = n;
  d.objective = value;
  d.gradient = n * gradient;
  if (std::isfinite(value) && d.gradient.allFinite() && n > 0) {
    auto H = fiml_observed_hessian_analytic(pt, rep, cache, start_samp, est);
    if (H.has_value() && H->rows() == est.theta.size() && H->allFinite()) {
      d.hessian = 0.5 * n * (*H);
      d.status = NewtonAccuracyStatus::Available;
    } else {
      d.detail = H.has_value() ? "invalid FIML observed Hessian" : H.error().detail;
    }
  }
  est.diagnostics.newton_accuracy =
      estimate::frontier::audit_newton_derivatives(pt, rep, std::move(d), domain).diagnostics;
}

fit_expected<Estimates>
fit_fiml_impl(spec::LatentStructure pt,
              const model::MatrixRep& rep,
              const RawData& raw,
              const Eigen::VectorXd& x0,
              const FIMLCache& cache,
              const SampleStats& start_samp,
              FIML discrepancy,
              const estimate::frontier::ExtraNonlinearEqConstraints& extra,
              Backend backend,
              optim::OptimOptions opts,
              const char* who) {
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(e.error());
  }

  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp);
      !e.has_value()) {
    return std::unexpected(e.error());
  }

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail, 0, 0.0});
  }
  auto ev = std::move(*ev_or);

  if (x0.size() != pt.n_free()) {
    return std::unexpected(FitError{
        FitError::Kind::InvalidStartValues,
        std::string(who) + ": x0 size (" + std::to_string(x0.size()) + ") != n_free (" +
            std::to_string(pt.n_free()) + ")", 0, 0.0});
  }

  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail, 0, 0.0});
  }
  const EqConstraints& con = *con_or;
  const NonlinearEqConstraints nl = build_nl_constraints(pt);
  if (auto ok = validate_extra_constraints(extra, x0, pt.n_free(), who);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  const Eigen::Index n_constraint =
      static_cast<Eigen::Index>(nl.m()) + extra.n_constraint;
  auto h_theta = [&nl, &extra, n_constraint](const Eigen::VectorXd& theta) {
    Eigen::VectorXd out(n_constraint);
    Eigen::Index off = 0;
    if (nl.active()) {
      out.segment(off, nl.m()) = nl.h(theta);
      off += nl.m();
    }
    if (extra.active()) {
      out.segment(off, extra.n_constraint) = extra.h(theta);
    }
    return out;
  };
  auto J_theta = [&nl, &extra, n_constraint](const Eigen::VectorXd& theta) {
    Eigen::MatrixXd out(n_constraint, theta.size());
    Eigen::Index off = 0;
    if (nl.active()) {
      out.block(off, 0, nl.m(), theta.size()) = nl.jacobian(theta);
      off += nl.m();
    }
    if (extra.active()) {
      out.block(off, 0, extra.n_constraint, theta.size()) =
          extra.jacobian(theta);
    }
    return out;
  };
  auto eval_at = [&](const Eigen::VectorXd& x,
                     Eigen::VectorXd& grad) -> double {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    auto vg = discrepancy.value_gradient(raw, cache, eval->moments,
                                         eval->J_sigma, eval->J_mu);
    if (!vg.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    // ½·F scale: est.fmin = ½F, uniform with ML/LS. The FIML χ² stays the LRT
    // in fiml_extras, which recomputes the full-F deviance from est.theta and
    // is unaffected. INVARIANT: halve ONLY here in the optimiser adapter —
    // FIML::value / value_gradient must stay full-F because fiml_extras and
    // the observed-Hessian paths (analytic and FD) differentiate them.
    grad = 0.5 * vg->gradient;
    return 0.5 * vg->value;
  };

  // Optimizer coordinates from the start moments; raw coordinates when the
  // options ask for none or the map cannot be formed.
  std::optional<CoordinateMap> map;
  if (opts.coordinate_scaling != optim::CoordinateScaling::None && con.n_alpha > 0) {
    if (auto m = coordinate_map(opts.coordinate_scaling, opts.center_locations, pt,
                                rep, ev, con, start_samp, con.contract(x0)))
      map = std::move(*m);
  }

  auto finalize = [&](Estimates est) {
    if (map) est.coordinate_scaling = map->kind;
    est.diagnostics = finalize_fit_diagnostics(
        est.theta, pt, ev, con, nl, Bounds{});
    audit_observed_variances(est.diagnostics, start_samp.S, false);
    if (!extra.active()) {
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(est.theta.size());
      const double value = eval_at(est.theta, gradient);
      if (!std::isfinite(value)) {
        gradient.setConstant(std::numeric_limits<double>::quiet_NaN());
      }
      audit_full_model_fit(est.diagnostics, est.theta, gradient, est.fmin, value,
                           pt, ev, con, nl, Bounds{});
      attach_fiml_newton_accuracy(est, pt, rep, cache, start_samp, gradient,
                                  value, StationarityDomain::Ambient);
    }
    return est;
  };

  auto raw_fiml_scalar = [&](const optim::ScalarProblem& prob,
                             const Eigen::VectorXd& start)
      -> fit_expected<optim::OptimResult> {
    switch (backend) {
      case Backend::NloptSlsqp:
        return optim::nlopt_slsqp(prob, start, {}, opts);
      case Backend::NloptLbfgs:
        return optim::nlopt_lbfgs(prob, start, {}, opts);
      case Backend::NloptLbfgsSlsqpFallback:
        return optim::nlopt_lbfgs_slsqp_fallback(prob, start, {}, opts);
      case Backend::Ipopt:
#ifdef MAGMAAN_WITH_IPOPT
        return optim::ipopt(prob, start, {}, opts);
#else
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            std::string(who) + ": IPOPT backend requested but MAGMAAN_WITH_IPOPT is off"));
#endif
      default:
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            std::string(who) + ": requested optimizer backend is not supported; use "
            "nlopt-lbfgs, nlopt-lbfgs-slsqp-fallback, nlopt-slsqp, or ipopt"));
    }
  };

  auto raw_fiml_constrained = [&](const optim::ScalarProblem& prob,
                                  const optim::ConstraintFn& h,
                                  const optim::ConstraintJacFn& J_h,
                                  std::int32_t m,
                                  const Eigen::VectorXd& start)
      -> fit_expected<optim::OptimResult> {
    optim::ConstrainedScalarProblem cprob;
    cprob.objective = prob;
    cprob.h = h;
    cprob.J_h = J_h;
    cprob.n_constraint = m;
    cprob.constraint_lower = Eigen::VectorXd::Zero(m);
    cprob.constraint_upper = Eigen::VectorXd::Zero(m);
    if (backend == Backend::NloptSlsqp ||
        backend == Backend::NloptLbfgsSlsqpFallback) {
      return optim::nlopt_slsqp_constrained(cprob, start, {}, opts);
    }
#ifdef MAGMAAN_WITH_IPOPT
    if (backend == Backend::Ipopt) {
      return optim::ipopt_constrained(cprob, start, {}, opts);
    }
#else
    if (backend == Backend::Ipopt) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          std::string(who) + ": nonlinear equality constraints require optimizer "
          "\"nlopt-slsqp\" or an IPOPT-enabled build with optimizer "
          "\"ipopt\""));
    }
#endif
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        std::string(who) + ": nonlinear equality constraints require optimizer "
        "\"nlopt-slsqp\" or \"ipopt\""));
  };

  auto run_fiml_scalar = [&](const optim::ScalarProblem& prob,
                             const Eigen::VectorXd& start)
      -> fit_expected<optim::OptimResult> {
    if (!map) return raw_fiml_scalar(prob, start);
    return driven::run_in_coordinates(
        prob, *map, start, Bounds{},
        [&](const optim::ScalarProblem& q, const Eigen::VectorXd& z0,
            const Bounds&) { return raw_fiml_scalar(q, z0); });
  };
  auto run_fiml_constrained = [&](const optim::ScalarProblem& prob,
                                  const optim::ConstraintFn& h,
                                  const optim::ConstraintJacFn& J_h,
                                  std::int32_t m,
                                  const Eigen::VectorXd& start)
      -> fit_expected<optim::OptimResult> {
    if (!map) return raw_fiml_constrained(prob, h, J_h, m, start);
    return driven::run_in_coordinates(
        prob, h, J_h, m, *map, start, Bounds{},
        [&](const optim::ScalarProblem& q, const optim::ConstraintFn& hq,
            const optim::ConstraintJacFn& Jq, Eigen::Index mq,
            const Eigen::VectorXd& z0, const Bounds&) {
          return raw_fiml_constrained(q, hq, Jq, static_cast<std::int32_t>(mq), z0);
        });
  };

  if (!con.active()) {
    optim::ScalarProblem prob;
    prob.f       = eval_at;
    prob.n_param = x0.size();
    prob.expand  = [](const Eigen::VectorXd& x) { return x; };
    fit_expected<optim::OptimResult> out_or;
    if (n_constraint > 0) {
      out_or = run_fiml_constrained(
          prob, h_theta, J_theta, static_cast<std::int32_t>(n_constraint), x0);
    } else {
      out_or = run_fiml_scalar(prob, x0);
    }
    if (!out_or.has_value()) return std::unexpected(out_or.error());
    return finalize(Estimates{
        prob.expand(out_or->x), out_or->fmin, out_or->iterations,
        out_or->f_evals, out_or->g_evals, out_or->status,
        out_or->grad_inf_norm, std::move(out_or->audit)});
  }

  if (con.n_alpha == 0) {
    Eigen::VectorXd theta = con.expand(Eigen::VectorXd(0));
    if (n_constraint > 0) {
      const Eigen::VectorXd h = h_theta(theta);
      const double viol = h.size() > 0 ? h.cwiseAbs().maxCoeff() : 0.0;
      if (viol > 1e-6 || !std::isfinite(viol)) {
        return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
            std::string(who) + ": nonlinear equality constraints are infeasible at the "
            "fully linear-constrained point; violation " +
                std::to_string(viol)));
      }
    }
    Eigen::VectorXd scratch(theta.size());
    const double f = eval_at(theta, scratch);
    return finalize(Estimates{std::move(theta), f, 0});
  }

  optim::ScalarProblem prob_a;
  prob_a.n_param = con.n_alpha;
  prob_a.expand  = [&con](const Eigen::VectorXd& a) { return con.expand(a); };
  prob_a.f = [&con, &eval_at](const Eigen::VectorXd& a,
                              Eigen::VectorXd& grad_a) -> double {
    const Eigen::VectorXd x = con.expand(a);
    Eigen::VectorXd grad_x(x.size());
    const double v = eval_at(x, grad_x);
    grad_a = con.reduce_gradient(grad_x);
    return v;
  };
  const Eigen::VectorXd alpha0 = con.contract(x0);
  fit_expected<optim::OptimResult> out_or;
  if (n_constraint > 0) {
    auto h_a = [&h_theta, &con](const Eigen::VectorXd& a) {
      return h_theta(con.expand(a));
    };
    auto jac_a = [&J_theta, &con](const Eigen::VectorXd& a) {
      return Eigen::MatrixXd(J_theta(con.expand(a)) * con.K());
    };
    out_or = run_fiml_constrained(
        prob_a, h_a, jac_a, static_cast<std::int32_t>(n_constraint), alpha0);
  } else {
    out_or = run_fiml_scalar(prob_a, alpha0);
  }
  if (!out_or.has_value()) return std::unexpected(out_or.error());
  return finalize(Estimates{
      con.expand(out_or->x), out_or->fmin, out_or->iterations,
      out_or->f_evals, out_or->g_evals, out_or->status,
      out_or->grad_inf_norm, std::move(out_or->audit)});
}

}  // namespace

fit_expected<Estimates> evaluate_fiml_at(spec::LatentStructure pt,
    const model::MatrixRep& rep, const RawData& raw, const FIMLPack& pack,
    const Eigen::VectorXd& theta) {
  if (theta.size() != pt.n_free())
    return std::unexpected(make_fit_err(FitError::Kind::InvalidStartValues,
        "FIML endpoint size differs from n_free"));
  if (auto ok = validate_fiml_fixed_x_missing_policy(pt, raw); !ok)
    return std::unexpected(ok.error());
  if (auto ok = resolve_fixed_x_from_sample(pt, rep, pack.start_stats); !ok)
    return std::unexpected(ok.error());
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return std::unexpected(make_fit_err(FitError::Kind::NumericIssue, ev.error().detail));
  auto con = build_eq_constraints(pt);
  if (!con) return std::unexpected(make_fit_err(FitError::Kind::NumericIssue, con.error().detail));
  const auto nl = build_nl_constraints(pt);
  Eigen::VectorXd gradient = Eigen::VectorXd::Constant(theta.size(),
      std::numeric_limits<double>::quiet_NaN());
  double value = std::numeric_limits<double>::infinity();
  auto evaluated = ev->evaluate(theta, true, true);
  if (evaluated) {
    auto vg = FIML{}.value_gradient(raw, pack.cache, evaluated->moments,
                                  evaluated->J_sigma, evaluated->J_mu);
    if (vg) { value = 0.5 * vg->value; gradient = 0.5 * vg->gradient; }
  }
  Estimates est{theta, value, 0};
  const auto endpoint = [&](const Eigen::VectorXd&, Eigen::VectorXd& g) {
    g = gradient;
    return value;
  };
  const double inf = std::numeric_limits<double>::infinity();
  est.audit = optim::audit_terminal_iterate(endpoint, theta, value,
      Eigen::VectorXd::Constant(theta.size(), -inf),
      Eigen::VectorXd::Constant(theta.size(), inf));
  est.grad_inf_norm = est.audit.grad_inf_norm;
  est.f_evals = est.g_evals = 1;
  est.diagnostics = finalize_fit_diagnostics(theta, pt, *ev, *con, nl, Bounds{});
  audit_observed_variances(est.diagnostics, pack.start_stats.S, false);
  audit_full_model_fit(est.diagnostics, theta, gradient, value, value,
                      pt, *ev, *con, nl, Bounds{});
  attach_fiml_newton_accuracy(est, pt, rep, pack.cache, pack.start_stats,
                              gradient, value, StationarityDomain::Ambient);
  return est;
}

fit_expected<Estimates>
fit_fiml(spec::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         const Eigen::VectorXd& x0,
         FIML discrepancy,
         Backend backend,
         optim::OptimOptions opts) {
  // Policy validation precedes prepare so unsupported fixed.x-missing models
  // keep reporting the policy error, not a downstream pattern error.
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(e.error());
  }
  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) return std::unexpected(cache_or.error());
  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) return std::unexpected(start_samp_or.error());
  const estimate::frontier::ExtraNonlinearEqConstraints extra;
  return fit_fiml_impl(std::move(pt), rep, raw, x0, *cache_or, *start_samp_or,
                       discrepancy, extra, backend, std::move(opts),
                       "fit_fiml");
}

fit_expected<Estimates>
fit_fiml(spec::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         const Eigen::VectorXd& x0,
         const FIMLPack& pack,
         Backend backend,
         optim::OptimOptions opts) {
  const estimate::frontier::ExtraNonlinearEqConstraints extra;
  return fit_fiml_impl(std::move(pt), rep, raw, x0, pack.cache,
                       pack.start_stats, FIML{}, extra, backend,
                       std::move(opts), "fit_fiml");
}


}  // namespace magmaan::estimate::fiml
