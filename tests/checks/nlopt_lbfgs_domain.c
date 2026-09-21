/* Advisory reproduction of NLopt L-BFGS domain backtracking exhaustion.
 * Example build from repo root:
 * cc -I build/opt/_deps/nlopt-src/src/api tests/checks/nlopt_lbfgs_domain.c
 *   build/opt/_deps/nlopt-build/libnlopt.a -lstdc++ -lm -o /tmp/lbfgs-domain
 * Run /tmp/lbfgs-domain. No particular failure is required by the test suite:
 * a future backend with a longer line search should solve all three cases.
 */
#include <math.h>
#include <stdio.h>
#include <nlopt.h>

static double variance_likelihood(unsigned n, const double *x,
                                  double *gradient, void *data) {
  (void)n;
  const double sample_variance = *(const double *)data;
  if (!(x[0] > 0)) {
    if (gradient) gradient[0] = 0;
    return INFINITY;
  }
  if (gradient)
    gradient[0] = 1 / x[0] - sample_variance / (x[0] * x[0]);
  return log(x[0] / sample_variance) + sample_variance / x[0] - 1;
}

int main(void) {
  double variances[] = {1, 0.01, 0.000001};
  puts("sample_variance,start,result,evaluations,estimate,discrepancy");
  for (unsigned i = 0; i < 3; ++i) {
    double start = 10 * variances[i], estimate = start, value;
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, 1);
    if (!opt) return 1;
    nlopt_set_min_objective(opt, variance_likelihood, &variances[i]);
    nlopt_set_maxeval(opt, 5000);
    int result = nlopt_optimize(opt, &estimate, &value);
    printf("%.12g,%.12g,%d,%d,%.12g,%.12g\n", variances[i], start,
           result, nlopt_get_numevals(opt), estimate, value);
    nlopt_destroy(opt);
  }
  return 0;
}
