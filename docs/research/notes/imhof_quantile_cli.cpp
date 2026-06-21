#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/robust/weighted_chisq.hpp"

namespace {

Eigen::VectorXd
positive_lambdas(int argc, char** argv, int offset) {
  std::vector<double> xs;
  for (int i = offset; i < argc; ++i) {
    const double x = std::strtod(argv[i], nullptr);
    if (std::isfinite(x) && x > 1e-12) xs.push_back(x);
  }
  Eigen::VectorXd out(static_cast<Eigen::Index>(xs.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out(i) = xs[static_cast<std::size_t>(i)];
  }
  return out;
}

double
quantile(const Eigen::VectorXd& lambda, double prob) {
  if (lambda.size() == 0) return 0.0;
  prob = std::clamp(prob, 0.0, 1.0);
  if (prob <= 0.0) return 0.0;
  const double target_upper = 1.0 - prob;

  double lo = 0.0;
  double hi = std::max(1.0, lambda.sum() +
                                12.0 * std::sqrt(2.0 * lambda.squaredNorm()));
  while (magmaan::robust::imhof_upper(lambda, hi) > target_upper) {
    hi *= 2.0;
    if (hi > 1e12) break;
  }

  for (int iter = 0; iter < 80; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double upper = magmaan::robust::imhof_upper(lambda, mid);
    if (upper > target_upper) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

void
usage(const char* exe) {
  std::cerr << "usage:\n"
            << "  " << exe << " quantile <cdf_prob> <lambda...>\n"
            << "  " << exe << " upper <x> <lambda...>\n";
}

}  // namespace

int
main(int argc, char** argv) {
  if (argc < 4) {
    usage(argv[0]);
    return 2;
  }

  const std::string mode = argv[1];
  const double      x = std::strtod(argv[2], nullptr);
  const Eigen::VectorXd lambda = positive_lambdas(argc, argv, 3);

  std::cout << std::setprecision(17);
  if (mode == "quantile") {
    const double q = quantile(lambda, x);
    std::cout << q << ' ' << magmaan::robust::imhof_upper(lambda, q) << '\n';
    return 0;
  }
  if (mode == "upper") {
    std::cout << magmaan::robust::imhof_upper(lambda, x) << '\n';
    return 0;
  }

  usage(argv[0]);
  return 2;
}
