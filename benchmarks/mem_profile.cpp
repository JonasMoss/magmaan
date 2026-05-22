// Memory-profiling harness for the three reduced-Gamma computations.
//
// Builds a synthetic one-factor CFA of size (n, p), fits it with ML, forms the
// U-factor and the casewise-contribution matrix Zc, then runs ONE of the three
// reduced-Gamma computations and reports the peak heap during that call (via
// the --wrap malloc counter) plus the process peak RSS.
//
//   materialized -> reduced_gamma_sample_materialized (forms the q x q Gamma)
//   batched      -> reduced_gamma_sample              (forms an n x k block)
//   rowwise      -> reduced_gamma_sample_streaming    (k x k accumulator only)
//
// Usage:  magmaan_mem_profile {materialized|batched|rowwise} [n] [p]

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <sys/resource.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/spec/build.hpp"

#include "malloc_peak.hpp"

namespace {

constexpr double kMiB = 1024.0 * 1024.0;

// One-factor population covariance: unit-variance indicators sharing a common
// loading, so Sigma = lambda*lambda' on the off-diagonal and 1 on the diagonal.
Eigen::MatrixXd one_factor_sigma(Eigen::Index p, double loading) {
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Constant(p, p, loading * loading);
  sigma.diagonal().setOnes();
  return sigma;
}

// An n x p sample from N(0, sigma).
Eigen::MatrixXd mvn_sample(Eigen::Index n, const Eigen::MatrixXd& sigma,
                           std::uint64_t seed) {
  const Eigen::Index p = sigma.rows();
  Eigen::LLT<Eigen::MatrixXd> llt(sigma);
  const Eigen::MatrixXd L = llt.matrixL();
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < p; ++j) X(i, j) = z(rng);
  return X * L.transpose();
}

std::string one_factor_model(Eigen::Index p) {
  std::string m = "f =~ x1";
  for (Eigen::Index j = 2; j <= p; ++j) m += " + x" + std::to_string(j);
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string kind = (argc > 1) ? argv[1] : "materialized";
  const Eigen::Index n =
      (argc > 2) ? static_cast<Eigen::Index>(std::atol(argv[2])) : 4000;
  const Eigen::Index p =
      (argc > 3) ? static_cast<Eigen::Index>(std::atol(argv[3])) : 70;

  if (kind != "materialized" && kind != "batched" && kind != "rowwise") {
    std::fprintf(stderr, "kind must be materialized|batched|rowwise\n");
    return 2;
  }

  // ---- synthetic data + ML fit ---------------------------------------------
  const Eigen::MatrixXd sigma = one_factor_sigma(p, 0.7);
  magmaan::data::RawData raw;
  raw.X.push_back(mvn_sample(n, sigma, 20260522ULL));

  auto samp = magmaan::data::sample_stats_from_raw(raw);
  if (!samp.has_value()) {
    std::fprintf(stderr, "sample_stats_from_raw failed\n");
    return 1;
  }

  auto flat = magmaan::parse::Parser::parse(one_factor_model(p));
  if (!flat.has_value()) {
    std::fprintf(stderr, "parse failed\n");
    return 1;
  }
  auto pt = magmaan::spec::build(*flat);
  if (!pt.has_value()) {
    std::fprintf(stderr, "spec::build failed\n");
    return 1;
  }
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep.has_value()) {
    std::fprintf(stderr, "build_matrix_rep failed\n");
    return 1;
  }

  auto x0 = magmaan::estimate::simple_start_values(*pt, *rep, *samp, {});
  if (!x0.has_value()) {
    std::fprintf(stderr, "simple_start_values failed\n");
    return 1;
  }
  auto est = magmaan::estimate::fit_ml(*pt, *rep, *samp, *x0);
  if (!est.has_value()) {
    std::fprintf(stderr, "fit_ml failed\n");
    return 1;
  }

  auto uf = magmaan::robust::build_u_factor(*pt, *rep, *samp, *est);
  if (!uf.has_value()) {
    std::fprintf(stderr, "build_u_factor failed\n");
    return 1;
  }
  auto Zc = magmaan::robust::casewise_contributions(raw, *samp);
  if (!Zc.has_value()) {
    std::fprintf(stderr, "casewise_contributions failed\n");
    return 1;
  }

  const long q = static_cast<long>(uf->pstar);
  const long k = static_cast<long>(uf->df);
  const double denom = static_cast<double>(n);

  // ---- the one measured computation ----------------------------------------
  std::size_t base = 0;
  std::size_t peak = 0;
  bool ok = true;

  if (kind == "rowwise") {
    std::vector<Eigen::VectorXd> rows;
    rows.reserve(static_cast<std::size_t>(Zc->rows()));
    for (Eigen::Index i = 0; i < Zc->rows(); ++i)
      rows.push_back(Zc->row(i).transpose());
    Zc->resize(0, 0);  // drop the matrix form; `rows` now holds the input
    base = memprof::current_bytes();
    memprof::reset_peak();
    auto M = magmaan::robust::reduced_gamma_sample_streaming(*uf, rows, denom);
    peak = memprof::peak_bytes();
    ok = M.has_value();
  } else if (kind == "batched") {
    base = memprof::current_bytes();
    memprof::reset_peak();
    auto M = magmaan::robust::reduced_gamma_sample(*uf, *Zc, denom);
    peak = memprof::peak_bytes();
    ok = M.has_value();
  } else {  // materialized
    base = memprof::current_bytes();
    memprof::reset_peak();
    auto M =
        magmaan::robust::reduced_gamma_sample_materialized(*uf, *Zc, denom);
    peak = memprof::peak_bytes();
    ok = M.has_value();
  }

  if (!ok) {
    std::fprintf(stderr, "%s computation failed\n", kind.c_str());
    return 1;
  }

  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  const double maxrss_mib = static_cast<double>(ru.ru_maxrss) / 1024.0;

  std::printf(
      "RESULT kind=%s n=%ld p=%ld q=%ld k=%ld "
      "peak_mib=%.2f delta_mib=%.2f base_mib=%.2f maxrss_mib=%.2f\n",
      kind.c_str(), static_cast<long>(n), static_cast<long>(p), q, k,
      static_cast<double>(peak) / kMiB,
      static_cast<double>(peak - base) / kMiB,
      static_cast<double>(base) / kMiB, maxrss_mib);
  return 0;
}
