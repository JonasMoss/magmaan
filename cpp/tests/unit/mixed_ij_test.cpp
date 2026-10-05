#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include <random>
#include "../../src/data/detail_sampling_reference.hpp"
#include "../../src/estimate/ordinal_internal.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include "magmaan/robust/lr_test_satorra.hpp"

namespace {
using namespace magmaan;
using estimate::OrdinalParameterization;
using estimate::OrdinalWeightKind;

// One factor with an omitted residual association: estimated weights matter
// at the population projection, rather than only through sampling residuals.
Eigen::MatrixXd mixed_block(unsigned seed, Eigen::Index n) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z;
  Eigen::MatrixXd x(n, 6);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double f = z(rng), nuisance = z(rng);
    for (int j = 0; j < 6; ++j) {
      const double loading = 0.8 - 0.04*j;
      const double shared = j < 2 ? 0.35 : 0.0;
      const double y = loading*f + shared*nuisance +
          std::sqrt(1-loading*loading-shared*shared)*z(rng);
      x(i,j) = j < 3 ? 1.0+(y > -0.5)+(y > 0.5) : y+0.1*j;
    }
  }
  return x;
}
struct MixedModel { spec::LatentStructure pt; model::MatrixRep rep; };
MixedModel mixed_model(int groups, bool score = false, bool equal = false) {
  auto parsed = parse::Parser::parse(
      std::string(equal ? "f =~ x1 + x2 + x3 + a*x4 + a*x5 + x6\n"
                        : "f =~ x1 + x2 + x3 + x4 + x5 + x6\n") +
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\n" +
      std::string(score ? "x4 ~~ 0*x5\n" : ""));
  REQUIRE(parsed.has_value());
  spec::BuildOptions opts;
  opts.meanstructure = true;
  opts.n_groups = groups;
  auto pt = spec::build(*parsed, opts);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {*pt,*rep};
}
optim::OptimOptions tight() {
  optim::OptimOptions opts;
  opts.max_iter=3000; opts.ftol=1e-16; opts.gtol=1e-13;
  return opts;
}
void polish(const MixedModel& m, const data::MixedOrdinalStats& stats,
            estimate::Estimates& fit, OrdinalParameterization parameterization) {
  auto objective=estimate::frontier::mixed_ordinal_ls_objective(
      m.pt,m.rep,stats,fit,OrdinalWeightKind::DWLS,parameterization);
  REQUIRE(objective.has_value());
  double total_n=0; for (auto n : stats.n_obs) total_n+=static_cast<double>(n);
  for (int iteration=0;iteration<4;++iteration) {
    auto residual=objective->problem.r(fit.theta);
    auto jacobian=objective->problem.J(fit.theta);
    REQUIRE(residual.has_value()); REQUIRE(jacobian.has_value());
    const Eigen::VectorXd gradient=jacobian->transpose()*(*residual);
    if (gradient.lpNorm<Eigen::Infinity>() < 1e-13) return;
    auto parts=estimate::frontier::mixed_ordinal_ls_newton_parts(
        m.pt,m.rep,stats,fit.theta,OrdinalWeightKind::DWLS,parameterization);
    REQUIRE(parts.has_value());
    const Eigen::VectorXd step=parts->hessian.ldlt().solve(total_n*gradient);
    REQUIRE(step.allFinite());
    fit.theta-=step;
  }
  auto residual=objective->problem.r(fit.theta);
  auto jacobian=objective->problem.J(fit.theta);
  REQUIRE(residual.has_value()); REQUIRE(jacobian.has_value());
  CHECK((jacobian->transpose()*(*residual)).lpNorm<Eigen::Infinity>() < 1e-12);
}

}

TEST_CASE("Mixed DWLS IJ covariance agrees with the delete-one jackknife") {
  for (auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
    for (int groups : {1,2}) {
      double previous_error=1.0;
      for (int n : {600,1200}) {
        CAPTURE(n); CAPTURE(groups); CAPTURE(static_cast<int>(parameterization));
        std::vector<Eigen::MatrixXd> blocks;
        std::vector<std::vector<std::int32_t>> ordered(static_cast<std::size_t>(groups), {1,1,1,0,0,0});
        for (int g=0;g<groups;++g) blocks.push_back(mixed_block(6700u+static_cast<unsigned>(g),n));
        auto stats=data::mixed_ordinal_stats_from_data(blocks,ordered,false);
        REQUIRE(stats.has_value());
        const auto m=mixed_model(groups);
        auto fit=test::fit_mixed_ordinal_bounded(m.pt,m.rep,*stats,{},OrdinalWeightKind::DWLS,
            estimate::Backend::NloptLbfgs,tight(),parameterization);
        REQUIRE_MESSAGE(fit.has_value(), (fit.has_value()? "" : fit.error().detail));
        auto ij=estimate::robust_mixed_ordinal_ij(m.pt,m.rep,*stats,*fit,OrdinalWeightKind::DWLS,parameterization);
        REQUIRE_MESSAGE(ij.has_value(), (ij.has_value()? "" : ij.error().detail));
        auto fixed=estimate::robust_mixed_ordinal(m.pt,m.rep,*stats,*fit,OrdinalWeightKind::DWLS,
            parameterization,robust::Information::Observed);
        REQUIRE(fixed.has_value());
        const Eigen::Index k=fit->theta.size();
        Eigen::MatrixXd jackknife=Eigen::MatrixXd::Zero(k,k);
        for (int g=0;g<groups;++g) {
          Eigen::MatrixXd thetas(n,k);
          for (int i=0;i<n;++i) {
            auto reduced=blocks;
            auto& x=reduced[static_cast<std::size_t>(g)];
            Eigen::MatrixXd y(n-1,x.cols());
            y.topRows(i)=x.topRows(i); y.bottomRows(n-1-i)=x.bottomRows(n-1-i);
            x=std::move(y);
            auto s=data::mixed_ordinal_stats_from_data(reduced,ordered,false);
            REQUIRE(s.has_value());
            auto refit=estimate::fit_mixed_ordinal_bounded(m.pt,m.rep,*s,{},OrdinalWeightKind::DWLS,
                fit->theta,estimate::Backend::NloptLbfgs,tight(),parameterization);
            REQUIRE_MESSAGE(refit.has_value(), (refit.has_value()? "" : refit.error().detail));
            thetas.row(i)=refit->theta.transpose();
          }
          const Eigen::MatrixXd centered=thetas.rowwise()-thetas.colwise().mean();
          jackknife += (double(n-1)/n)*centered.transpose()*centered;
        }
        const double error=(ij->vcov-jackknife).norm()/jackknife.norm();
        const Eigen::VectorXd diagonal=jackknife.diagonal();
        const double diagonal_error=(ij->vcov.diagonal()-diagonal).norm()/diagonal.norm();
        const double fixed_error=(fixed->vcov.diagonal()-diagonal).norm()/diagonal.norm();
        MESSAGE("mixed IJ relative Frobenius error: " << error << ", diagonal error " << diagonal_error
            << ", fixed-weight diagonal error " << fixed_error);
        // Frozen N=1200 evidence: maximum diagonal error 2.03% (theta,
        // two groups), versus 4.23--8.42% for the fixed OPG sandwich. The
        // 2.5% gate is specific to these designs, not a universal 1/N bound.
        if (n==1200) {
          CHECK(diagonal_error < 0.025);
          CHECK(diagonal_error < previous_error);
        }
        CHECK(diagonal_error < fixed_error);
        previous_error=diagonal_error;
      }
    }
  }
}

TEST_CASE("Mixed IJ reduction routes require both variable types") {
  const Eigen::MatrixXd x=mixed_block(6700u,300);
  // The public mixed builder deliberately rejects the endpoints. A direct
  // mixed-fit reduction to either pure route is consequently unavailable.
  for (int flag : {0,1}) {
    const Eigen::MatrixXd pure=flag ? x.leftCols(3).eval() : x.rightCols(3).eval();
    auto stats=data::mixed_ordinal_stats_from_data({pure},{{flag,flag,flag}},false);
    REQUIRE_FALSE(stats.has_value());
    CHECK(stats.error().kind == PostError::Kind::NumericIssue);
    CHECK(stats.error().detail.find("must contain both ordered and continuous") != std::string::npos);
  }
}


TEST_CASE("Mixed DWLS IJ agrees with replicated case-weight finite differences") {
  // Integer replication implements empirical case weights independently of
  // the influence formulas. +/- one copy is a central weight step 1/copies.
  const int n=150, copies=100;
  const auto x=mixed_block(6700u,n);
  const std::vector<std::vector<std::int32_t>> ordered{{1,1,1,0,0,0}};
  auto stats=data::mixed_ordinal_stats_from_data({x},ordered,false);
  REQUIRE(stats.has_value());
  auto sampling=data::mixed_moment_sampling_influence(x,stats->ordered[0],stats->n_levels[0],
      stats->thresholds[0],stats->mean[0],stats->R[0]);
  REQUIRE(sampling.has_value());
  auto direct=data::mixed_gamma_diag_data_influence(x,stats->ordered[0],stats->n_levels[0],
      stats->thresholds[0],stats->mean[0],stats->R[0]);
  auto movement=data::mixed_gamma_diag_jacobian_fd(x,stats->ordered[0],stats->n_levels[0],
      stats->thresholds[0],stats->mean[0],stats->R[0]);
  REQUIRE(direct.has_value()); REQUIRE(movement.has_value());
  const Eigen::MatrixXd gamma_if=*direct+*sampling*movement->transpose();
  const auto m=mixed_model(1);
  for (auto parameterization : {OrdinalParameterization::Delta,OrdinalParameterization::Theta}) {
    auto fit=test::fit_mixed_ordinal_bounded(m.pt,m.rep,*stats,{},OrdinalWeightKind::DWLS,
        estimate::Backend::NloptLbfgs,tight(),parameterization);
    REQUIRE(fit.has_value());
    polish(m,*stats,*fit,parameterization);
    for (int row : {0,17,91}) {
      CAPTURE(row); CAPTURE(static_cast<int>(parameterization));
      Eigen::VectorXd theta[2];
      data::MixedOrdinalStats perturbed[2];
      for (int side=0;side<2;++side) {
        const int step=side==0 ? -1 : 1;
        Eigen::MatrixXd repeated(n*copies+step,x.cols());
        int off=0;
        for (int i=0;i<n;++i)
          for (int j=0;j<copies+(i==row ? step : 0);++j) repeated.row(off++)=x.row(i);
        auto st=data::mixed_ordinal_stats_from_data({repeated},ordered,false);
        REQUIRE(st.has_value());
        perturbed[side]=std::move(*st);
        auto refit=estimate::fit_mixed_ordinal_bounded(m.pt,m.rep,perturbed[side],{},
            OrdinalWeightKind::DWLS,fit->theta,estimate::Backend::NloptLbfgs,tight(),parameterization);
        REQUIRE(refit.has_value());
        polish(m,perturbed[side],*refit,parameterization);
        theta[side]=refit->theta;
      }
      const Eigen::VectorXd derivative=(theta[1]-theta[0])*(copies/2.0);
      const Eigen::VectorXd moment_fd=(perturbed[1].moments[0]-perturbed[0].moments[0])*(n*copies/2.0);
      // Marginal sample mean and ML variance have exact empirical
      // derivatives, independent of any normal-information approximation.
      Eigen::VectorXd marginal(6);
      for (int j=0;j<3;++j) {
        const double residual=x(row,j+3)-stats->mean[0](j+3);
        marginal(j)=-residual;
        marginal(j+3)=residual*residual-stats->R[0](j+3,j+3);
      }
      CHECK((moment_fd.segment(6,6)-marginal).norm()/marginal.norm() < 1e-6);
      const double moment_error=(moment_fd-sampling->row(row).transpose()).norm()/moment_fd.norm();
      const Eigen::VectorXd gamma_fd=(perturbed[1].NACOV[0].diagonal()-
          perturbed[0].NACOV[0].diagonal())*(n*copies/2.0);
      const double gamma_error=(gamma_fd-gamma_if.row(row).transpose()).norm()/gamma_fd.norm();
      // Isolate the selected row so its covariance is the outer product of
      // its predicted parameter derivative (the API does not expose rows).
      auto isolated=*stats;
      isolated.sampling_moment_influence={Eigen::MatrixXd::Zero(n,sampling->cols())};
      isolated.sampling_moment_influence[0].row(row)=sampling->row(row);
      isolated.gamma_diag_influence={Eigen::MatrixXd::Zero(n,gamma_if.cols())};
      isolated.gamma_diag_influence[0].row(row)=gamma_if.row(row);
      auto ij=estimate::robust_mixed_ordinal_ij(m.pt,m.rep,isolated,*fit,
          OrdinalWeightKind::DWLS,parameterization);
      REQUIRE(ij.has_value());
      const Eigen::MatrixXd fd_outer=derivative*derivative.transpose();
      const double error=(ij->vcov-fd_outer).norm()/fd_outer.norm();
      MESSAGE("case-weight outer-product relative error " << error << ", Gamma row error " << gamma_error << ", moment row error " << moment_error);
      // The public IJ returns covariance, so recover the rank-one row up to
      // its unobservable sign before testing parameter-level agreement.
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(ij->vcov);
      REQUIRE(eigen.info() == Eigen::Success);
      Eigen::VectorXd predicted=eigen.eigenvectors().col(derivative.size()-1)*
          std::sqrt(std::max(0.0,eigen.eigenvalues()(derivative.size()-1)));
      if (predicted.dot(derivative)<0) predicted=-predicted;
      CHECK((predicted-derivative).norm()/derivative.norm() < 1e-5);
      CHECK(error < 2e-5);
      CHECK(gamma_error < 1e-5);
      CHECK(moment_error < 1e-5);
    }
  }
}

TEST_CASE("Mixed WLS Gamma influence agrees with replicated case weights") {
  const int n = 150, copies = 100;
  const auto x = mixed_block(6700u, n);
  const std::vector<std::vector<std::int32_t>> ordered{{1,1,1,0,0,0}};
  auto stats = data::mixed_ordinal_stats_from_data({x}, ordered, false);
  REQUIRE(stats.has_value());
  auto sampling = data::mixed_moment_sampling_influence(x, stats->ordered[0],
      stats->n_levels[0], stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto direct = data::mixed_gamma_data_influence(x, stats->ordered[0],
      stats->n_levels[0], stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto movement = data::mixed_gamma_jacobian_fd(x, stats->ordered[0],
      stats->n_levels[0], stats->thresholds[0], stats->mean[0], stats->R[0]);
  REQUIRE(sampling.has_value()); REQUIRE(direct.has_value()); REQUIRE(movement.has_value());
  const Eigen::MatrixXd rows = *direct + *sampling * movement->transpose();
  for (int row : {0,17,91}) {
    Eigen::MatrixXd gamma[2];
    for (int side = 0; side < 2; ++side) {
      const int step = side == 0 ? -1 : 1;
      Eigen::MatrixXd repeated(n*copies+step, x.cols());
      int off = 0;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < copies+(i == row ? step : 0); ++j)
          repeated.row(off++) = x.row(i);
      auto st = data::mixed_ordinal_stats_from_data({repeated}, ordered, false);
      REQUIRE(st.has_value());
      gamma[side] = st->NACOV[0];
    }
    const Eigen::MatrixXd difference = (gamma[1]-gamma[0])*(n*copies/2.0);
    const Eigen::Map<const Eigen::VectorXd> fd(difference.data(), difference.size());
    const double error = (fd-rows.row(row).transpose()).norm()/fd.norm();
    MESSAGE("Full Gamma case-weight relative error " << error);
    CHECK(error < 1e-5);
  }
}

TEST_CASE("Mixed DWLS empirical IJ weight channel vanishes at exact fit") {
  const Eigen::MatrixXd x=mixed_block(6711u,300).rightCols(4);
  const std::vector<std::vector<std::int32_t>> ordered{{1,0,0,0}};
  auto stats=data::mixed_ordinal_stats_from_data({x},ordered,false);
  REQUIRE(stats.has_value());
  auto parsed=parse::Parser::parse(
      "x1 | t1 + t2\nx1 ~*~ 1*x1\nx1 ~~ 1*x1\nx1 ~ 0*1\n"
      "x1 ~~ x2 + x3 + x4\nx2 ~~ x3 + x4\nx3 ~~ x4\n");
  REQUIRE(parsed.has_value());
  spec::BuildOptions options; options.meanstructure=true;
  auto pt=spec::build(*parsed,options);
  REQUIRE(pt.has_value());
  auto rep=model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  for (auto parameterization : {OrdinalParameterization::Delta,OrdinalParameterization::Theta}) {
    auto fit=test::fit_mixed_ordinal_bounded(*pt,*rep,*stats,{},OrdinalWeightKind::DWLS,
        estimate::Backend::NloptLbfgs,tight(),parameterization);
    REQUIRE(fit.has_value());
    CHECK(fit->fmin < 1e-12);
    auto empirical=data::mixed_moment_sampling_influence(x,stats->ordered[0],stats->n_levels[0],
        stats->thresholds[0],stats->mean[0],stats->R[0]);
    REQUIRE(empirical.has_value());
    auto with_weight=estimate::robust_mixed_ordinal_ij(*pt,*rep,*stats,*fit,
        OrdinalWeightKind::DWLS,parameterization);
    REQUIRE(with_weight.has_value());
    auto no_weight=*stats;
    no_weight.sampling_moment_influence={*empirical};
    no_weight.gamma_diag_influence={Eigen::MatrixXd::Zero(x.rows(),empirical->cols())};
    auto without_weight=estimate::robust_mixed_ordinal_ij(*pt,*rep,no_weight,*fit,
        OrdinalWeightKind::DWLS,parameterization);
    REQUIRE(without_weight.has_value());
    CHECK((with_weight->vcov-without_weight->vcov).norm()/without_weight->vcov.norm() < 1e-6);
  }
}

TEST_CASE("Sparse exact first-stage Jacobian preserves dense FD scores and sampling rows") {
  using namespace magmaan;
  for (bool mixed : {false, true}) {
    for (int levels : {2, 5}) {
      for (int groups : {1, 2}) {
        CAPTURE(mixed); CAPTURE(levels); CAPTURE(groups);
        for (int group = 0; group < groups; ++group) {
          std::mt19937 rng(8300u + static_cast<unsigned>(group));
          std::normal_distribution<double> normal;
          Eigen::MatrixXd X(450, 6);
          std::vector<std::int32_t> ordered(6, 1);
          if (mixed) ordered = {1,1,1,0,0,0};
          for (Eigen::Index r = 0; r < X.rows(); ++r) {
            const double factor = normal(rng);
            for (Eigen::Index j = 0; j < X.cols(); ++j) {
              const double y = .65 * factor + .76 * normal(rng);
              if (ordered[static_cast<std::size_t>(j)]) {
                double category = 1;
                if (levels == 2) category += y > .7;
                else for (double threshold : {-1.4, -.6, .15, .9})
                  category += y > threshold;
                X(r,j) = category;
              } else X(r,j) = .3 + (1.2 + .1 * static_cast<double>(j)) * y;
            }
          }
          Eigen::VectorXd thresholds, mean;
          Eigen::MatrixXd R;
          std::vector<std::int32_t> n_levels;
          if (mixed) {
            auto stats = data::mixed_ordinal_stats_from_data({X}, {ordered}, false);
            REQUIRE(stats.has_value());
            thresholds = stats->thresholds[0]; mean = stats->mean[0];
            R = stats->R[0]; n_levels = stats->n_levels[0];
          } else {
            auto stats = data::ordinal_stats_from_integer_data({X}, false);
            REQUIRE(stats.has_value());
            thresholds = stats->thresholds[0]; R = stats->R[0];
            mean = Eigen::VectorXd::Zero(6); n_levels = stats->n_levels[0];
          }
          auto dense = data::validation::mixed_sampling_jacobian_for_validation(
              X, ordered, n_levels, thresholds, mean, R, false);
          auto sparse = data::validation::mixed_sampling_jacobian_for_validation(
              X, ordered, n_levels, thresholds, mean, R, true);
          REQUIRE(dense.has_value()); REQUIRE(sparse.has_value());
          const double jacobian_error = (*dense-*sparse).norm()/dense->norm();
          CAPTURE(jacobian_error);
          CHECK(jacobian_error <= 1e-7);
          auto reference = data::validation::mixed_sampling_influence_dense_reference(
              X, ordered, n_levels, thresholds, mean, R);
          auto rows = data::mixed_moment_sampling_influence(
              X, ordered, n_levels, thresholds, mean, R);
          REQUIRE(reference.has_value()); REQUIRE(rows.has_value());
          CHECK((*reference-*rows).norm()/reference->norm() <= 1e-7);
          const Eigen::MatrixXd gamma = rows->transpose()*(*rows)/450.0;
          const Eigen::MatrixXd gamma_reference = reference->transpose()*(*reference)/450.0;
          CHECK((gamma-gamma_reference).norm()/gamma_reference.norm() <= 1e-7);
        }
      }
    }
  }
}

TEST_CASE("Mixed estimated-weight MI reconstructs observed projection and augmented case scores") {
  using namespace estimate::detail_ordinal;
  for (auto weight : {OrdinalWeightKind::DWLS, OrdinalWeightKind::WLS}) {
    for (auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
      for (int groups : {1, 2}) {
        const int n = 150, copies = 100;
        const auto m = mixed_model(groups, true, true);
        std::vector<Eigen::MatrixXd> raw;
        for (std::size_t g = 0; g < static_cast<std::size_t>(groups); ++g) raw.push_back(mixed_block(6700u+static_cast<unsigned>(g), n));
        std::vector<std::vector<std::int32_t>> ordered(static_cast<std::size_t>(groups), {1,1,1,0,0,0});
        auto stats = data::mixed_ordinal_stats_from_data(raw, ordered, true);
        REQUIRE(stats.has_value());
        auto fit = test::fit_mixed_ordinal_bounded(m.pt, m.rep, *stats, {}, weight,
            estimate::Backend::NloptLbfgs, tight(), parameterization);
        REQUIRE(fit.has_value());
        auto objective = estimate::frontier::mixed_ordinal_ls_objective(
            m.pt, m.rep, *stats, *fit, weight, parameterization);
        REQUIRE(objective.has_value());
        auto aug = objective->pt;
        std::size_t row = 0;
        for (; row < aug.size(); ++row)
          if (aug.op[row] == parse::Op::Covariance && aug.free[row] == 0 &&
              aug.lhs_var[row] != aug.rhs_var[row]) break;
        REQUIRE(row < aug.size());
        const int q = aug.n_free();
        const double fixed = aug.fixed_value[row];
        aug.free[row] = q+1;
        aug.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
        if (static_cast<int>(aug.eq_groups.size()) == q) aug.eq_groups.push_back(q);
        else if (!aug.eq_groups.empty()) aug.eq_groups.clear();
        Eigen::VectorXd theta(q+1); theta.head(q) = fit->theta; theta(q) = fixed;
        auto layout = make_threshold_layout(aug, m.rep, *stats);
        auto ev = model::ModelEvaluator::build(aug, m.rep);
        REQUIRE(layout.has_value()); REQUIRE(ev.has_value());
        auto eval = ev->evaluate(theta, true, true);
        REQUIRE(eval.has_value());
        const auto delta = mixed_moment_jacobian(*stats, *layout, eval->moments,
            eval->J_sigma, eval->J_mu, theta, parameterization);
        auto parts = estimate::frontier::mixed_ordinal_ls_newton_parts_prepared(
            aug, m.rep, *stats, theta, weight, parameterization);
        REQUIRE(parts.has_value());
        const double N = n*groups;
        const Eigen::MatrixXd H = parts->hessian/N;
        const auto& weights = weight == OrdinalWeightKind::DWLS ? stats->W_dwls : stats->W_wls;
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(q+1,q+1);
        Eigen::VectorXd score = Eigen::VectorXd::Zero(q+1);
        Eigen::Index off = 0;
        for (std::size_t g = 0; g < static_cast<std::size_t>(groups); ++g) {
          const auto& x = raw[g];
          const auto& W = weights[g];
          const Eigen::Index mb = stats->moments[g].size();
          const Eigen::MatrixXd D = delta.middleRows(off, mb);
          const Eigen::VectorXd residual = mixed_model_moments(*stats, *layout,
              eval->moments, theta, g, parameterization)-stats->moments[g];
          auto sampling = data::mixed_moment_sampling_influence(x, ordered[g],
              stats->n_levels[g], stats->thresholds[g], stats->mean[g], stats->R[g]);
          REQUIRE(sampling.has_value());
          auto direct = weight == OrdinalWeightKind::DWLS
              ? data::mixed_gamma_diag_data_influence(x, ordered[g], stats->n_levels[g],
                  stats->thresholds[g], stats->mean[g], stats->R[g])
              : data::mixed_gamma_data_influence(x, ordered[g], stats->n_levels[g],
                  stats->thresholds[g], stats->mean[g], stats->R[g]);
          auto movement = weight == OrdinalWeightKind::DWLS
              ? data::mixed_gamma_diag_jacobian_fd(x, ordered[g], stats->n_levels[g],
                  stats->thresholds[g], stats->mean[g], stats->R[g])
              : data::mixed_gamma_jacobian_fd(x, ordered[g], stats->n_levels[g],
                  stats->thresholds[g], stats->mean[g], stats->R[g]);
          REQUIRE(direct.has_value()); REQUIRE(movement.has_value());
          const Eigen::MatrixXd gamma = *direct+*sampling*movement->transpose();
          Eigen::MatrixXd influence = *sampling*W;
          for (int i = 0; i < n; ++i) {
            Eigen::MatrixXd dW;
            if (weight == OrdinalWeightKind::DWLS) {
              dW = Eigen::MatrixXd::Zero(mb,mb);
              for (Eigen::Index k = 0; k < mb; ++k)
                dW(k,k) = -gamma(i,k)*W(k,k)*W(k,k);
            } else {
              const Eigen::VectorXd v = gamma.row(i).transpose();
              const Eigen::Map<const Eigen::MatrixXd> dGamma(v.data(),mb,mb);
              dW = -W*dGamma*W;
            }
            influence.row(i) -= residual.transpose()*dW;
          }
          const Eigen::MatrixXd rows = influence*D;
          B += rows.transpose()*rows/N;
          score -= D.transpose()*W*residual/groups;
          // Differentiate the augmented analytic score at a fixed parameter point.
          Eigen::VectorXd perturbed_score[2];
          for (int side = 0; side < 2; ++side) {
            const int step = side == 0 ? -1 : 1;
            auto blocks = raw;
            Eigen::MatrixXd repeated(n*copies+step, x.cols());
            int at = 0;
            for (int i = 0; i < n; ++i)
              for (int j = 0; j < copies+(i == 17 ? step : 0); ++j)
                repeated.row(at++) = x.row(i);
            // Replicate other groups equally, keeping group fractions fixed below.
            blocks[g] = std::move(repeated);
            auto st = data::mixed_ordinal_stats_from_data(blocks, ordered, true);
            REQUIRE(st.has_value());
            const auto& pw = weight == OrdinalWeightKind::DWLS ? st->W_dwls[g] : st->W_wls[g];
            perturbed_score[side] = -D.transpose()*pw*(
                mixed_model_moments(*stats, *layout, eval->moments, theta, g, parameterization)-st->moments[g]);
          }
          const Eigen::VectorXd fd = (perturbed_score[1]-perturbed_score[0])*(n*copies/2.0);
          CHECK((fd-rows.row(17).transpose()).norm()/fd.norm() < 1e-5);
          off += mb;
        }
        Eigen::VectorXd v = Eigen::VectorXd::Zero(q+1); v(q) = 1;
        auto constraints = estimate::build_eq_constraints(objective->pt);
        REQUIRE(constraints.has_value());
        const Eigen::MatrixXd K = constraints->K();
        Eigen::MatrixXd nuisance = Eigen::MatrixXd::Zero(q+1,K.cols());
        nuisance.topRows(q) = K;
        v -= nuisance*(nuisance.transpose()*H*nuisance).ldlt().solve(nuisance.transpose()*H*v);
        const double u = v.dot(score), variance = v.dot(B*v);
        // At a common evaluation point, the policy one-restriction law is
        // b/h in observed geometry; its corrected local quadratic is N*u²/b.
        Eigen::MatrixXd chart(q+1,K.cols()+1);
        chart.leftCols(K.cols()) = nuisance;
        chart.col(K.cols()) = Eigen::VectorXd::Unit(q+1,q);
        Eigen::MatrixXd restriction = Eigen::MatrixXd::Zero(1,chart.cols());
        restriction(0,chart.cols()-1) = 1;
        const Eigen::MatrixXd Hc = chart.transpose()*H*chart;
        const Eigen::MatrixXd Bc = chart.transpose()*B*chart;
        auto law = robust::compute_satorra2000_from_sandwich(Hc, Bc, restriction);
        REQUIRE(law.has_value());
        REQUIRE(law->eigenvalues.size() == 1);
        const double h = v.dot(H*v);
        CHECK(law->eigenvalues(0) == doctest::Approx(variance/h).epsilon(1e-10));
        CHECK((N*u*u/h)/law->eigenvalues(0) == doctest::Approx(N*u*u/variance).epsilon(1e-10));
        auto exact = *stats;
        for (std::size_t g = 0; g < static_cast<std::size_t>(groups); ++g)
          exact.moments[g] = mixed_model_moments(*stats, *layout, eval->moments,
              theta, g, parameterization);
        auto fixed_blocks = build_mixed_ordinal_ij_blocks(exact, *layout,
            eval->moments, theta, weights, delta, weight, parameterization, false);
        auto estimated_blocks = build_mixed_ordinal_ij_blocks(exact, *layout,
            eval->moments, theta, weights, delta, weight, parameterization, true);
        REQUIRE(fixed_blocks.has_value()); REQUIRE(estimated_blocks.has_value());
        auto fixed_meat = estimate::weighted_param_space_sandwich_ij(*fixed_blocks);
        auto estimated_meat = estimate::weighted_param_space_sandwich_ij(*estimated_blocks);
        REQUIRE(fixed_meat.has_value()); REQUIRE(estimated_meat.has_value());
        CHECK((fixed_meat->B1-estimated_meat->B1).norm() < 1e-10);
        auto mi = estimate::frontier::modification_indices_mixed_ordinal_robust(
            m.pt, m.rep, *stats, *fit, weight, {}, parameterization, true,
            robust::Information::Observed);
        REQUIRE(mi.has_value());
        bool found = false;
        for (const auto& result : mi->rows) if (result.candidate.row == row) {
          found = true;
          CHECK(result.mi_scaled == doctest::Approx(N*u*u/variance).epsilon(1e-10));
        }
        CHECK(found);
        auto releases = estimate::frontier::score_tests_mixed_ordinal_robust(
            m.pt, m.rep, *stats, *fit, weight, parameterization, true,
            robust::Information::Observed);
        REQUIRE(releases.has_value());
        REQUIRE_FALSE(releases->rows.empty());
        const Eigen::MatrixXd H0 = H.topLeftCorner(q,q), B0 = B.topLeftCorner(q,q);
        for (Eigen::Index r = 0; r < constraints->A_eq.rows(); ++r) {
          Eigen::MatrixXd remaining(constraints->A_eq.rows()-1,q);
          Eigen::Index at = 0;
          for (Eigen::Index i = 0; i < constraints->A_eq.rows(); ++i)
            if (i != r) remaining.row(at++) = constraints->A_eq.row(i);
          Eigen::MatrixXd relaxed;
          if (remaining.rows() == 0) relaxed = Eigen::MatrixXd::Identity(q,q);
          else {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(remaining,Eigen::ComputeFullV);
            svd.setThreshold(1e-9);
            relaxed = svd.matrixV().rightCols(q-svd.rank());
          }
          const Eigen::MatrixXd overlap = K.transpose()*relaxed;
          Eigen::JacobiSVD<Eigen::MatrixXd> complement(overlap,Eigen::ComputeFullV);
          complement.setThreshold(1e-9);
          REQUIRE(relaxed.cols()-complement.rank() == 1);
          const Eigen::VectorXd d = relaxed*complement.matrixV().col(relaxed.cols()-1);
          const Eigen::VectorXd efficient = d-K*(K.transpose()*H0*K).ldlt().solve(K.transpose()*H0*d);
          const double u0 = efficient.dot(score.head(q));
          const double b0 = efficient.dot(B0*efficient);
          bool matched = false;
          for (const auto& result : releases->rows) if (result.candidate.row == static_cast<std::size_t>(r)) {
            matched = true;
            CHECK(result.mi_scaled == doctest::Approx(N*u0*u0/b0).epsilon(1e-10));
          }
          CHECK(matched);
        }
      }
    }
  }
}
