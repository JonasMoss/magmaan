// Advisory Monte-Carlo check for the estimated-weight ("complete-sandwich")
// standardized residuals (measures::frontier::standardized_residuals_estimated_
// weight). It validates the FD / bootstrap property the analytic residual ACOV
// claims: the estimated-weight residual SE should match the actual sampling SD
// of the standardized residuals across resamples, and track it better than the
// normal-theory residual SE under non-normality (heavy tails).
//
// One base sample is drawn from a 4-indicator population that carries an extra
// residual covariance the one-factor model cannot reproduce, so real residuals
// remain. We fit continuous DWLS once, record the analytic NT and estimated-
// weight SEs for every off-diagonal correlation residual, then nonparametric-
// bootstrap the base sample B times (resample rows, refit, recompute the
// residuals) to get the empirical sampling SD of each residual. Reported: the
// mean ratio analytic-SE / bootstrap-SD for the NT and estimated-weight SEs;
// the estimated-weight ratio should sit near 1, the NT ratio below it.
//
// Build/run via the local justfile (links the prebuilt opt libmagmaan.a):
//   just quick      # n=1500, reps=300, df=6
//   just all        # n=3000, reps=1500, df=6

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/measures/residuals.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

namespace est = magmaan::estimate;

Eigen::Matrix4d population_cov() {
  Eigen::Vector4d lambda;  lambda << 1.0, 0.8, 0.7, 0.9;
  Eigen::Vector4d theta;   theta  << 0.6, 0.7, 0.8, 0.5;
  Eigen::Matrix4d S =
      lambda * lambda.transpose() * 1.4 + theta.asDiagonal().toDenseMatrix();
  S(1, 0) += 0.18;
  S(0, 1) = S(1, 0);
  return S;
}

Eigen::MatrixXd mvt_sample(std::mt19937& rng, Eigen::Index n,
                           const Eigen::MatrixXd& Sigma, double df) {
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::Index p = Sigma.rows();
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi(df);
  const double scale = std::sqrt((df - 2.0) / df);
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(p);
    for (Eigen::Index j = 0; j < p; ++j) zi(j) = z(rng);
    const double w = chi(rng) / df;
    X.row(i) = (scale * (L * zi) / std::sqrt(w)).transpose();
  }
  return X;
}

// Diagonal-ADF (continuous DWLS) weight W = diag(Γ̂)⁻¹ from raw rows.
magmaan::estimate::gmm::Weight dwls_weight(const Eigen::MatrixXd& X) {
  auto G = magmaan::data::empirical_gamma(X);
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(G->rows(), G->cols());
  for (Eigen::Index k = 0; k < G->rows(); ++k) W(k, k) = 1.0 / (*G)(k, k);
  return magmaan::estimate::gmm::Weight{W};
}

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

Handles build_model() {
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4");
  auto pt = magmaan::spec::build(*fp);
  auto rep = magmaan::model::build_matrix_rep(*pt);
  return Handles{std::move(*pt), std::move(*rep)};
}

// Fit continuous DWLS on raw X; returns the cor-metric off-diagonal residuals
// (length 6) or false on any failure.
bool dwls_cor_residuals(const Handles& h, const Eigen::MatrixXd& X,
                        Eigen::VectorXd& out) {
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  if (!samp.has_value()) return false;
  auto W = dwls_weight(X);
  auto x0 = est::simple_start_values(h.pt, h.rep, *samp, {});
  if (!x0.has_value()) return false;
  auto b = est::bounds_from_partable(h.pt);
  if (!b.has_value()) return false;
  auto e = est::fit_gmm(h.pt, h.rep, *samp, *x0, W, *b);
  if (!e.has_value()) return false;
  auto sr = magmaan::measures::standardized_residuals(h.pt, h.rep, *samp, *e);
  if (!sr.has_value()) return false;
  const Eigen::MatrixXd& C = sr->cov_cor[0];
  const Eigen::Index p = C.rows();
  out.resize(p * (p - 1) / 2);
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c)
    for (Eigen::Index r = c + 1; r < p; ++r) out(k++) = C(r, c);
  return true;
}

long parse_long(std::string_view s, long fallback) {
  long v = fallback;
  std::from_chars(s.data(), s.data() + s.size(), v);
  return v;
}
double parse_double(std::string_view s, double fallback) {
  double v = fallback;
  std::from_chars(s.data(), s.data() + s.size(), v);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  long n = 1500, reps = 300;
  double df = 6.0;
  for (int i = 1; i < argc; ++i) {
    std::string_view a(argv[i]);
    if (a.rfind("--n=", 0) == 0) n = parse_long(a.substr(4), n);
    else if (a.rfind("--reps=", 0) == 0) reps = parse_long(a.substr(7), reps);
    else if (a.rfind("--df=", 0) == 0) df = parse_double(a.substr(5), df);
  }

  const Handles h = build_model();
  const Eigen::Matrix4d Sigma = population_cov();
  std::mt19937 rng(20260623u);

  // Base sample + analytic NT / estimated-weight residual SEs.
  Eigen::MatrixXd X = mvt_sample(rng, n, Sigma, df);
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  auto W = dwls_weight(X);
  auto x0 = est::simple_start_values(h.pt, h.rep, *samp, {});
  auto bnd = est::bounds_from_partable(h.pt);
  auto est_hat = est::fit_gmm(h.pt, h.rep, *samp, *x0, W, *bnd);
  if (!est_hat.has_value()) {
    std::cerr << "base fit failed\n";
    return 1;
  }
  auto nt = magmaan::measures::standardized_residuals(h.pt, h.rep, *samp,
                                                      *est_hat);
  auto ew = magmaan::measures::frontier::standardized_residuals_estimated_weight(
      h.pt, h.rep, *samp, *est_hat, W, raw,
      est::ContinuousLsIJWeightMode::SampleEmpiricalDwls);
  if (!nt.has_value() || !ew.has_value()) {
    std::cerr << "base residuals failed\n";
    return 1;
  }
  const Eigen::Index p = nt->cov_cor[0].rows();
  const Eigen::Index npair = p * (p - 1) / 2;
  Eigen::VectorXd nt_se(npair), ew_se(npair);
  {
    Eigen::Index k = 0;
    for (Eigen::Index c = 0; c < p; ++c)
      for (Eigen::Index r = c + 1; r < p; ++r) {
        nt_se(k) = nt->cov_se[0](r, c);
        ew_se(k) = ew->cov_se[0](r, c);
        ++k;
      }
  }

  // Nonparametric bootstrap of the base sample -> sampling SD per residual.
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(npair);
  Eigen::VectorXd m2 = Eigen::VectorXd::Zero(npair);
  std::uniform_int_distribution<Eigen::Index> pick(0, n - 1);
  long used = 0, fail = 0;
  for (long b = 0; b < reps; ++b) {
    Eigen::MatrixXd Xb(n, X.cols());
    for (Eigen::Index i = 0; i < n; ++i) Xb.row(i) = X.row(pick(rng));
    Eigen::VectorXd res;
    if (!dwls_cor_residuals(h, Xb, res)) { ++fail; continue; }
    ++used;
    const Eigen::VectorXd d = res - mean;
    mean += d / static_cast<double>(used);
    m2 += d.cwiseProduct(res - mean);
  }
  if (used < 2) { std::cerr << "bootstrap produced too few fits\n"; return 1; }
  const Eigen::VectorXd boot_sd =
      (m2 / static_cast<double>(used - 1)).cwiseSqrt();

  double sum_nt = 0.0, sum_ew = 0.0;
  std::cout << "estimated-weight residual ACOV vs bootstrap (DWLS, t_" << df
            << ", n=" << n << ", reps=" << used << ")\n";
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  pair   boot.sd    nt.se   nt/boot    ew.se   ew/boot\n";
  for (Eigen::Index k = 0; k < npair; ++k) {
    const double rn = nt_se(k) / boot_sd(k);
    const double re = ew_se(k) / boot_sd(k);
    sum_nt += rn;
    sum_ew += re;
    std::cout << "  " << std::setw(4) << k << "  " << std::setw(8) << boot_sd(k)
              << " " << std::setw(8) << nt_se(k) << " " << std::setw(8) << rn
              << " " << std::setw(8) << ew_se(k) << " " << std::setw(8) << re
              << "\n";
  }
  const double dn = static_cast<double>(npair);
  std::cout << "  mean ratio  NT: " << sum_nt / dn
            << "   estimated-weight: " << sum_ew / dn << "\n";
  std::cout << "  (estimated-weight ratio should be nearer 1 than NT under "
               "heavy tails; "
            << fail << " bootstrap fits skipped)\n";
  return 0;
}
