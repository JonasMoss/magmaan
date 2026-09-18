// magmaan complete-data modular timing harness.
//
// Times each component of a complete-data linear SEM fit separately, over a
// crossed design of model structure x parameterization x p x n, and emits tidy
// long CSV. No analysis happens here: crossover detection, scaling slopes, and
// arm ranking are downstream groupbys.
//
// What "modular" buys, concretely: a reparameterization or a backend can win by
// making iterations cheaper or by needing fewer of them, and those are entirely
// different claims. `optimize_lbfgs` is reported alongside the iteration count
// it took, and the per-iteration primitives (sigma, dsigma, ml_value,
// ml_gradient, objective_call) are timed directly, so the two effects separate.
//
// Under complete data the n axis and the p axis hit disjoint stage sets:
// everything downstream of `sample_stats` works on S (p x p) and theta, where n
// enters only as a scalar multiplier. `sample_stats` is O(n p^2) and
// `info_cross_products` is O(n npar); every other stage here should be flat in
// n. That is a design consequence, and also a free validity check on this
// harness — if `optimize_lbfgs` moves with n at fixed p, the measurement is
// wrong.
//
// `staged_fit_ml` vs `fit_ml_end_to_end` is reported rather than disclaimed:
// the staged path rebuilds what the composite fit shares, so the gap prices the
// terminal audit and the fit diagnostics instead of leaving it to a footnote.
// That row earns its keep — it is what caught `optimize_lbfgs` failing to apply
// the constraint reparameterization for `effect.coding`, which had made that arm
// look like it beat marker when it actually loses to it.
//
// Stages are timed in a rotated shared arm set by default. That is the right
// design for comparing arms OF ONE STAGE and the wrong one for comparing
// different stages to each other, because multi-megabyte-allocating stages
// couple through the allocator. Pass `--isolate` for cross-stage comparisons.
// See the note on `time_arms` in timing.hpp.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/reparameterize.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/sim/normal.hpp"
#include "magmaan/spec/build.hpp"

#include "catalog.hpp"
#include "timing.hpp"

namespace {

using magmaan::bench::Arm;
using magmaan::bench::must;
using magmaan::bench::Param;
using magmaan::bench::StageStats;

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::string              cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

std::vector<int> split_ints(const std::string& s) {
  std::vector<int> out;
  for (auto& t : split_csv(s)) out.push_back(std::atoi(t.c_str()));
  return out;
}

Param param_from(const std::string& s) {
  if (s == "marker") return Param::Marker;
  if (s == "std.lv") return Param::StdLv;
  if (s == "effect.coding") return Param::EffectCoding;
  std::fprintf(stderr, "unknown parameterization '%s'\n", s.c_str());
  std::exit(2);
}

magmaan::spec::BuildOptions build_options_for(Param p) {
  magmaan::spec::BuildOptions o;
  switch (p) {
    case Param::Marker: break;
    case Param::StdLv: o.std_lv = true; break;
    case Param::EffectCoding: o.effect_coding = true; break;
  }
  return o;
}

struct CellKey {
  std::string model;
  std::string param;
  int         p = 0;
  int         n = 0;
};

struct CellOut {
  CellKey                 key;
  int                     npar           = 0;
  int                     fit_iterations = 0;
  int                     fit_f_evals    = 0;
  int                     fit_g_evals    = 0;
  double                  fit_fmin       = 0.0;
  std::vector<StageStats> stages;
};

// Stages whose medians reconstruct one `fit_ml` call. `fit_ml`'s own prelude is
// resolve_fixed_x + evaluator build + equality constraints + nonlinear
// constraints, then the objective build, then the optimizer, then diagnostics.
// The gap against the separately timed `fit_ml_end_to_end` therefore prices the
// terminal audit, the fit diagnostics, and whatever the composite path shares
// that the staged path rebuilds.
const std::vector<std::string>& staged_fit_members() {
  static const std::vector<std::string> v{"evaluator_build",    "constraints_eq",
                                          "constraints_nl",     "objective_ml_build",
                                          "reparameterize",     "optimize_lbfgs"};
  return v;
}

// One full user-facing analysis: syntax in, point estimates + expected-
// information SEs + chi-square + fit indices out.
const std::vector<std::string>& pipeline_members() {
  static const std::vector<std::string> v{
      "parse",          "build",        "matrix_rep",     "sample_stats",
      "evaluator_build", "starts_simple", "bounds_standard", "constraints_eq",
      "constraints_nl", "objective_ml_build", "reparameterize", "optimize_lbfgs",
      "info_expected",  "vcov",         "se",             "chi2_stat",
      "df_stat",        "baseline_chi2", "fit_measures"};
  return v;
}

// Sum the medians of `members`. Returns -1 when any member is missing, so a
// `--skip` cannot silently deflate a reported total.
double sum_members(const std::vector<StageStats>&  stages,
                   const std::vector<std::string>& members) {
  std::map<std::string, double> by_name;
  for (const auto& s : stages) by_name[s.name] = s.median_ns;
  double total = 0.0;
  for (const auto& m : members) {
    auto it = by_name.find(m);
    if (it == by_name.end()) return -1.0;
    total += it->second;
  }
  return total;
}

bool skipped(const std::vector<std::string>& skips, const std::string& name) {
  for (auto& s : skips)
    if (name.find(s) != std::string::npos) return true;
  return false;
}

CellOut run_cell(const CellKey& key, Param param,
                 const magmaan::bench::TimingOptions& topt,
                 const std::vector<std::string>&      skips,
                 std::uint64_t                        seed) {
  using namespace magmaan;

  CellOut out;
  out.key = key;

  const auto mc = bench::make_case(key.model, key.p);

  // ---- Untimed setup: one canonical pass through the whole pipeline -------
  // Every timed thunk below re-runs exactly one stage against this state.
  auto flat = must(parse::Parser::parse(mc.syntax), "parse");
  const auto bopts = build_options_for(param);
  spec::Starts      starts_hint;
  spec::LatentNames names;
  auto pt  = must(spec::build(flat, bopts, &starts_hint, &names), "build");
  auto rep = must(model::build_matrix_rep(pt, &names), "matrix_rep");

  std::mt19937_64 rng(seed);
  const Eigen::VectorXd zero_mean = Eigen::VectorXd::Zero(key.p);
  auto raw = must(sim::simulate_normal_raw(key.n, zero_mean, mc.sigma, rng),
                  "simulate_normal_raw");
  auto samp = must(data::sample_stats_from_raw(raw), "sample_stats");

  must(estimate::resolve_fixed_x_from_sample(pt, rep, samp), "resolve_fixed_x");
  auto ev = must(model::ModelEvaluator::build(pt, rep), "evaluator");
  out.npar = static_cast<int>(pt.n_free());

  const Eigen::VectorXd x0 =
      must(estimate::simple_start_values(pt, rep, samp, starts_hint), "starts");
  const estimate::Bounds bounds =
      must(estimate::standard_bounds(pt, samp), "bounds");

  // Reference fit, untimed. Supplies theta-hat for the post-fit stages and the
  // iteration count that contextualises `optimize_lbfgs`.
  const auto est = must(estimate::fit_ml(pt, rep, samp, x0, bounds,
                                         estimate::Backend::NloptLbfgs, {}),
                        "fit_ml");
  out.fit_iterations = est.iterations;
  out.fit_f_evals    = est.f_evals;
  out.fit_g_evals    = est.g_evals;
  out.fit_fmin       = est.fmin;

  const Eigen::VectorXd th = est.theta;

  // Hot-path inputs held stable so the micro stages measure only their own
  // work: implied moments and the Jacobian at theta-hat, and the ML cache.
  const model::ImpliedMoments implied = must(ev.sigma(th), "sigma");
  const Eigen::MatrixXd       Jsig    = must(ev.dsigma_dtheta(th), "dsigma");
  const estimate::MlCache     mlcache = must(estimate::ml_prepare(samp), "ml_prepare");

  // Objective problems. Their closures borrow `ev` and `samp`, which outlive
  // them here; `ev` is never moved after this point.
  const auto ml_prob  = must(estimate::ml_objective(ev, samp), "ml_objective");

  // Equality-constrained models are driven in the reduced alpha space, not in
  // full theta: `fit_ml` folds `build_eq_constraints` through
  // `optim::reparameterize` before handing anything to a backend. The
  // `optimize_lbfgs` arm has to do the same, or it silently solves a different
  // (unconstrained, differently-dimensioned) problem and converges to a
  // different point. `effect.coding` is the arm where this bites: it adds one
  // linear equality row per latent. The `staged_fit_ml` gap row is what caught
  // this — an unreparameterized arm read ~50% under the composite fit.
  const auto eqcon = must(estimate::build_eq_constraints(pt), "build_eq_constraints");
  const bool reduced = eqcon.rank > 0;
  const auto opt_prob =
      reduced ? optim::reparameterize(ml_prob, eqcon) : ml_prob;
  // `fold_alpha_bounds` is only defined for a *pure-merge* reparameterization,
  // flagged by a non-empty `con.group` (size npar there, empty otherwise).
  // `effect.coding` produces general-linear rows (sum of loadings == k), which
  // have no box-preserving alpha; those are optimized unbounded with a post-hoc
  // bound check, exactly as the header specifies. Calling it anyway indexes an
  // empty `group` and segfaults.
  const bool pure_merge = !eqcon.group.empty();
  const estimate::Bounds opt_bounds =
      !reduced ? bounds
               : (pure_merge ? optim::fold_alpha_bounds(eqcon, bounds)
                             : estimate::Bounds{});
  // x0 in the driven coordinate system: alpha solving theta0 + K*alpha = x0.
  const Eigen::VectorXd opt_x0 =
      reduced ? Eigen::VectorXd(eqcon.Kmat.colPivHouseholderQr().solve(
                    x0 - eqcon.theta0))
              : x0;
  const auto nt_w     = must(estimate::gmm::normal_theory_weight(ev, samp, x0),
                             "normal_theory_weight");
  const auto info_exp = must(inference::information_expected(pt, rep, samp, est),
                             "information_expected");
  const auto vcov_m   = must(inference::vcov(info_exp, pt, th), "vcov");
  const auto baseline = measures::baseline_chi2(pt, samp);
  const int  df       = must(inference::df_stat(pt, samp, th), "df_stat");
  const double chi2   = inference::chi2_stat(samp, est);

  // ---- The stage list ----------------------------------------------------
  std::vector<Arm> arms;
  auto add = [&](const char* name, std::function<double()> fn) {
    if (!skipped(skips, name)) arms.push_back(Arm{name, std::move(fn)});
  };

  // A. spec
  add("parse", [&] {
    auto r = parse::Parser::parse(mc.syntax);
    return r.has_value() ? 1.0 : 0.0;
  });
  add("build", [&] {
    auto r = spec::build(flat, bopts);
    return r.has_value() ? static_cast<double>(r->n_free()) : 0.0;
  });
  add("matrix_rep", [&] {
    auto r = model::build_matrix_rep(pt, &names);
    return r.has_value() ? 1.0 : 0.0;
  });

  // B. data reduction — the only O(n) stage on the ML path
  add("sample_stats", [&] {
    auto r = data::sample_stats_from_raw(raw);
    return r.has_value() ? r->S[0](0, 0) : 0.0;
  });

  // C. problem construction
  add("evaluator_build", [&] {
    auto r = model::ModelEvaluator::build(pt, rep);
    return r.has_value() ? static_cast<double>(r->n_free()) : 0.0;
  });
  add("starts_simple", [&] {
    auto r = estimate::simple_start_values(pt, rep, samp, starts_hint);
    return r.has_value() ? r->sum() : 0.0;
  });
  add("starts_fabin3", [&] {
    auto r = estimate::fabin_start_values(pt, rep, samp, starts_hint,
                                          estimate::FabinVariant::Fabin3);
    return r.has_value() ? r->sum() : 0.0;
  });
  add("bounds_standard", [&] {
    auto r = estimate::standard_bounds(pt, samp);
    return r.has_value() ? static_cast<double>(r->lower.size()) : 0.0;
  });
  add("constraints_eq", [&] {
    auto r = estimate::build_eq_constraints(pt);
    return r.has_value() ? static_cast<double>(r->rank) : 0.0;
  });
  add("constraints_nl", [&] {
    auto r = estimate::build_nl_constraints(pt);
    return static_cast<double>(r.rows.size());
  });
  add("objective_ml_build", [&] {
    auto r = estimate::ml_objective(ev, samp);
    return r.has_value() ? static_cast<double>(r->n_param) : 0.0;
  });
  add("weight_nt_build", [&] {
    auto r = estimate::gmm::normal_theory_weight(ev, samp, x0);
    return r.has_value() ? 1.0 : 0.0;
  });
  add("objective_gls_build", [&] {
    auto r = estimate::gmm::residuals(ev, samp, x0, nt_w);
    return r.has_value() ? static_cast<double>(r->n_resid) : 0.0;
  });
  add("objective_gls_trace_build", [&] {
    auto r = estimate::gmm::normal_theory_objective(ev, samp, x0);
    return r.has_value() ? static_cast<double>(r->n_param) : 0.0;
  });

  // D. per-iteration primitives, all at theta-hat
  add("sigma", [&] {
    auto r = ev.sigma(th);
    return r.has_value() ? r->sigma[0](0, 0) : 0.0;
  });
  add("dsigma", [&] {
    auto r = ev.dsigma_dtheta(th);
    return r.has_value() ? r->coeff(0, 0) : 0.0;
  });
  add("evaluate_sigma_and_jac", [&] {
    auto r = ev.evaluate(th, true, false);
    return r.has_value() ? r->J_sigma.coeff(0, 0) : 0.0;
  });
  add("ml_value", [&] {
    auto r = estimate::ml_value(samp, mlcache, implied);
    return r.has_value() ? *r : 0.0;
  });
  add("ml_gradient", [&] {
    auto r = estimate::ml_gradient(samp, implied, Jsig);
    return r.has_value() ? r->coeff(0) : 0.0;
  });
  add("ml_value_gradient", [&] {
    auto r = estimate::ml_value_gradient(samp, mlcache, implied, Jsig);
    return r.has_value() ? r->value : 0.0;
  });
  add("objective_call", [&] {
    Eigen::VectorXd g(ml_prob.n_param);
    return ml_prob.f(th, g);
  });

  // E. optimize. Reported with `fit_iterations` so cost-per-iteration and
  // iteration-count stay separable.
  add("reparameterize", [&] {
    if (!reduced) return 0.0;
    auto r = optim::reparameterize(ml_prob, eqcon);
    return static_cast<double>(r.n_param);
  });
  add("optimize_lbfgs", [&] {
    auto r = optim::nlopt_lbfgs(opt_prob, opt_x0, opt_bounds, {});
    return r.has_value() ? r->fmin : 0.0;
  });
  add("fit_ml_end_to_end", [&] {
    auto r = estimate::fit_ml(pt, rep, samp, x0, bounds,
                              estimate::Backend::NloptLbfgs, {});
    return r.has_value() ? r->fmin : 0.0;
  });

  // F. post-fit
  add("info_expected", [&] {
    auto r = inference::information_expected(pt, rep, samp, est);
    return r.has_value() ? r->trace() : 0.0;
  });
  add("info_observed_analytic", [&] {
    auto r = inference::information_observed_analytic(pt, rep, samp, est);
    return r.has_value() ? r->trace() : 0.0;
  });
  add("info_observed_fd", [&] {
    auto r = inference::information_observed_fd(pt, rep, samp, est);
    return r.has_value() ? r->trace() : 0.0;
  });
  add("info_cross_products", [&] {
    auto r = inference::information_cross_products(pt, rep, samp, raw, est);
    return r.has_value() ? r->trace() : 0.0;
  });
  add("vcov", [&] {
    auto r = inference::vcov(info_exp, pt, th);
    return r.has_value() ? r->trace() : 0.0;
  });
  add("se", [&] { return inference::se(vcov_m).sum(); });
  add("chi2_stat", [&] { return inference::chi2_stat(samp, est); });
  add("df_stat", [&] {
    auto r = inference::df_stat(pt, samp, th);
    return r.has_value() ? static_cast<double>(*r) : 0.0;
  });
  add("baseline_chi2", [&] { return measures::baseline_chi2(pt, samp).chi2; });
  add("fit_measures", [&] {
    return measures::fit_measures(chi2, df, baseline, samp).cfi;
  });
  add("fit_extras", [&] {
    auto r = measures::fit_extras(pt, rep, samp, est);
    return r.has_value() ? r->aic : 0.0;
  });

  out.stages = magmaan::bench::time_stages(arms, topt);
  return out;
}

void usage() {
  std::fprintf(stderr,
      "magmaan_timing_bench --out FILE [options]\n"
      "  --models   csv of structure ids (default: all)\n"
      "  --p        csv of p values     (default: 6,12,24,48,96)\n"
      "  --n        csv of n values     (default: 200,1000,5000,50000)\n"
      "  --params   csv of marker,std.lv,effect.coding (default: marker)\n"
      "  --reps N   timed reps per stage (default 15)\n"
      "  --warmups N                     (default 3)\n"
      "  --skip     csv substrings of stage names to omit\n"
      "  --isolate  time each stage alone instead of in a shared rotation;\n"
      "             required for comparing different stages to each other\n"
      "  --seed N                        (default 20260918)\n"
      "  --list     print the catalog and exit\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> models = magmaan::bench::all_ids();
  std::vector<int>         ps{6, 12, 24, 48, 96};
  std::vector<int>         ns{200, 1000, 5000, 50000};
  std::vector<std::string> params{"marker"};
  std::vector<std::string> skips;
  std::string              out_path;
  std::uint64_t            seed = 20260918ULL;
  magmaan::bench::TimingOptions topt;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--models")        models = split_csv(next());
    else if (a == "--p")        ps     = split_ints(next());
    else if (a == "--n")        ns     = split_ints(next());
    else if (a == "--params")   params = split_csv(next());
    else if (a == "--skip")     skips  = split_csv(next());
    else if (a == "--out")      out_path = next();
    else if (a == "--reps")     topt.reps = std::atoi(next().c_str());
    else if (a == "--warmups")  topt.warmups = std::atoi(next().c_str());
    else if (a == "--seed")     seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--isolate")  topt.isolate = true;
    else if (a == "--list") {
      for (auto& id : magmaan::bench::all_ids()) {
        std::fprintf(stderr, "%-16s p in {", id.c_str());
        for (int p : {6, 12, 24, 48, 96})
          if (magmaan::bench::supports(id, p)) std::fprintf(stderr, " %d", p);
        std::fprintf(stderr, " }%s\n",
                     magmaan::bench::has_latents(id) ? "" : "  (no latents)");
      }
      return 0;
    } else { usage(); return 2; }
  }

  std::FILE* out = stdout;
  if (!out_path.empty()) {
    out = std::fopen(out_path.c_str(), "w");
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", out_path.c_str());
      return 1;
    }
  }

  std::fprintf(out,
      "model,parameterization,p,n,npar,fit_iterations,fit_f_evals,fit_g_evals,"
      "fit_fmin,stage,reps,batch,median_ns,min_ns,p90_ns,iqr_ns,checksum\n");

  int cells = 0;
  for (const auto& model : models) {
    for (int p : ps) {
      if (!magmaan::bench::supports(model, p)) continue;
      for (const auto& pname : params) {
        const Param pr = param_from(pname);
        if (pr != Param::Marker && !magmaan::bench::has_latents(model)) continue;
        for (int n : ns) {
          CellKey key{model, pname, p, n};
          std::fprintf(stderr, "[cell] %-16s %-14s p=%-3d n=%-6d ",
                       model.c_str(), pname.c_str(), p, n);
          std::fflush(stderr);
          const auto r = run_cell(key, pr, topt, skips, seed);
          std::fprintf(stderr, "npar=%-4d f/g evals=%d/%d\n", r.npar,
                       r.fit_f_evals, r.fit_g_evals);

          auto emit = [&](const char* stage, int reps, int batch, double med,
                          double mn, double p90, double iqr, double chk) {
            std::fprintf(out,
                "%s,%s,%d,%d,%d,%d,%d,%d,%.12g,%s,%d,%d,%.1f,%.1f,%.1f,%.1f,%.10g\n",
                model.c_str(), pname.c_str(), p, n, r.npar, r.fit_iterations,
                r.fit_f_evals, r.fit_g_evals, r.fit_fmin, stage, reps, batch,
                med, mn, p90, iqr, chk);
          };
          for (const auto& s : r.stages)
            emit(s.name.c_str(), s.reps, s.batch, s.median_ns, s.min_ns,
                 s.p90_ns, s.iqr_ns, s.checksum);
          // Derived rows: additive reconstructions from the stage medians.
          const double staged = sum_members(r.stages, staged_fit_members());
          if (staged >= 0.0)
            emit("staged_fit_ml", 0, 0, staged, staged, staged, 0.0, 0.0);
          const double pipe = sum_members(r.stages, pipeline_members());
          if (pipe >= 0.0)
            emit("pipeline_total", 0, 0, pipe, pipe, pipe, 0.0, 0.0);
          std::fflush(out);
          ++cells;
        }
      }
    }
  }

  std::fprintf(stderr, "%d cells\n", cells);
  if (out != stdout) std::fclose(out);
  return 0;
}
