// Diagnostic leaf: exact inputs emitted by --port-nls-witnesses.
// stdout: profile, expanded-original, returned-original, expansion gap.
// stderr: cold backend, historical objective gap, historical coordinate gap,
// last evaluation at endpoint, cold-versus-R endpoint coordinate gap.
#include <fstream>
#include <iomanip>
#include <iostream>

#include "magmaan/estimate/gmm/gp.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) return 2;
  auto parsed = magmaan::parse::Parser::parse(
      "X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~~ Y");
  if (!parsed) return 3;
  auto pt = magmaan::spec::build(*parsed);
  if (!pt) return 3;
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep) return 3;
  auto ev = magmaan::model::ModelEvaluator::build(*pt, *rep);
  if (!ev) return 3;
  std::ifstream input(argv[1]);
  Eigen::MatrixXd s(6, 6);
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i < 6; ++i) input >> s(i, j);
  Eigen::VectorXd w(21), start(13), endpoint(13);
  for (int i = 0; i < 21; ++i) input >> w[i];
  for (int i = 0; i < 13; ++i) input >> start[i];
  for (int i = 0; i < 13; ++i) input >> endpoint[i];
  if (!input) return 4;
  magmaan::data::SampleStats sample;
  sample.S = {s};
  sample.n_obs = {100};
  auto block = magmaan::estimate::gmm::BlockWeight::dense(
      w.asDiagonal(), magmaan::FitError::Kind::NumericIssue, "profile probe");
  if (!block) return 5;
  magmaan::estimate::gmm::Weight weight = {*block};
  auto base = magmaan::estimate::gmm::residuals(*ev, sample, start, weight);
  if (!base) return 5;
  auto profile = magmaan::estimate::gmm::gp(*base, *pt, *ev, start);
  if (!profile) return 5;

  // Retain every residual evaluation; the backend's stored value can then
  // be associated with an actual point rather than guessed from a stop code.
  auto problem = profile->problem;
  std::vector<Eigen::VectorXd> coordinates;
  std::vector<double> objectives;
  auto remember = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& r) {
    coordinates.push_back(x);
    objectives.push_back(0.5 * r.squaredNorm());
  };
  problem.r = [&](const Eigen::VectorXd& x) {
    auto r = profile->problem.r(x);
    if (r) remember(x, *r);
    return r;
  };
  problem.eval = [&](const Eigen::VectorXd& x) {
    auto e = profile->problem.eval(x);
    if (e) remember(x, e->residual);
    return e;
  };
  magmaan::optim::OptimOptions options;
  options.port.unbounded_routine = argc == 3 && std::string(argv[2]) == "unbounded";
  auto fit = argc == 3
      ? magmaan::optim::port(magmaan::optim::scalarize(problem), profile->beta0, {}, options)
      : magmaan::optim::port_nls(problem, profile->beta0, {}, {});
  if (!fit) {
    std::cerr << fit.error().detail;
    return 6;
  }
  if (argc == 3) {
    std::cout << std::setprecision(17) << fit->fmin;
    for (double coordinate : fit->x) std::cout << ',' << coordinate;
    const auto& t = *fit->audit.port_endpoint;
    std::cout << ',' << t.stored_objective << ',' << t.returned_x_objective
              << ',' << t.best_point_substituted;
    Eigen::VectorXd gradient(fit->x.size());
    const double final_objective = magmaan::optim::scalarize(problem).f(fit->x, gradient);
    auto original = base->r(problem.expand(fit->x));
    if (!original) return 5;
    std::cout << ',' << final_objective << ',' << 0.5 * original->squaredNorm()
              << ',' << fit->audit.raw_backend_status << '\n';
    return 0;
  }
  auto final_profile = magmaan::estimate::gmm::gp(*base, *pt, *ev, endpoint);
  if (!final_profile) return 5;
  double historical_f_gap = 1e300;
  double historical_x_gap = 0;
  double last_endpoint_f = -1;
  for (std::size_t k = 0; k < objectives.size(); ++k) {
    double gap = std::abs(objectives[k] - fit->fmin);
    if (gap < historical_f_gap) {
      historical_f_gap = gap;
      historical_x_gap = (coordinates[k] - fit->x).norm();
    }
    if ((coordinates[k].array() == fit->x.array()).all())
      last_endpoint_f = objectives[k];
  }
  std::cerr << std::setprecision(17) << fit->fmin << ',' << historical_f_gap
            << ',' << historical_x_gap << ',' << last_endpoint_f << ','
            << (fit->x - final_profile->beta0).norm() << ','
            << fit->audit.port_endpoint->stored_objective << ','
            << fit->audit.port_endpoint->returned_x_objective << ','
            << fit->audit.port_endpoint->best_point_substituted << ','
            << fit->audit.raw_backend_status << '\n';
  auto r = final_profile->problem.r(final_profile->beta0);
  auto expanded = final_profile->problem.expand(final_profile->beta0);
  auto expanded_r = base->r(expanded);
  auto returned_r = base->r(endpoint);
  if (!r || !expanded_r || !returned_r) return 5;
  std::cout << std::setprecision(17) << 0.5 * r->squaredNorm() << ','
            << 0.5 * expanded_r->squaredNorm() << ','
            << 0.5 * returned_r->squaredNorm() << ','
            << (expanded - endpoint).cwiseAbs().maxCoeff() << '\n';
}
