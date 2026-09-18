#include "catalog.hpp"

#include <algorithm>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace magmaan::bench {

namespace {

constexpr double kLambda = 0.70;  // loading
constexpr double kTheta  = 0.51;  // residual variance (unit observed variance)
constexpr double kPhiOff = 0.30;  // first-order factor correlation
constexpr double kGamma  = 0.70;  // second-order loading
constexpr double kResCov = 0.15;  // residual covariance
constexpr double kBeta1  = 0.40;  // f3 <- f1, f3 <- f2
constexpr double kBeta2  = 0.50;  // f4 <- f3
constexpr double kPath   = 0.25;  // observed path coefficient, three lags

std::string xname(int i) { return "x" + std::to_string(i + 1); }
std::string vname(int i) { return "v" + std::to_string(i + 1); }
std::string fname(int i) { return "f" + std::to_string(i + 1); }

// Λ with p / n_fac consecutive indicators loading on each factor.
Eigen::MatrixXd block_lambda(int p, int n_fac) {
  Eigen::MatrixXd L   = Eigen::MatrixXd::Zero(p, n_fac);
  const int       per = p / n_fac;
  for (int f = 0; f < n_fac; ++f)
    for (int j = 0; j < per; ++j) L(f * per + j, f) = kLambda;
  return L;
}

// Compound-symmetric factor covariance with unit variances.
Eigen::MatrixXd compound_phi(int n_fac, double off) {
  Eigen::MatrixXd Phi = Eigen::MatrixXd::Constant(n_fac, n_fac, off);
  Phi.diagonal().setOnes();
  return Phi;
}

std::string block_cfa_syntax(int p, int n_fac) {
  const int   per = p / n_fac;
  std::string s;
  for (int f = 0; f < n_fac; ++f) {
    s += fname(f) + " =~ ";
    for (int j = 0; j < per; ++j) {
      if (j) s += " + ";
      s += xname(f * per + j);
    }
    s += "\n";
  }
  return s;
}

void standardize(Eigen::MatrixXd& S) {
  const Eigen::VectorXd d = S.diagonal().cwiseSqrt().cwiseInverse();
  S = d.asDiagonal() * S * d.asDiagonal();
  S.diagonal().setOnes();
}

Eigen::MatrixXd measurement_sigma(const Eigen::MatrixXd& L,
                                  const Eigen::MatrixXd& Psi) {
  const Eigen::Index p = L.rows();
  Eigen::MatrixXd    S = L * Psi * L.transpose();
  S.diagonal().array() += kTheta;
  return S;
}

}  // namespace

const char* param_name(Param p) {
  switch (p) {
    case Param::Marker:        return "marker";
    case Param::StdLv:         return "std.lv";
    case Param::EffectCoding:  return "effect.coding";
  }
  return "?";
}

std::vector<std::string> scaling_ids() {
  return {"cfa_1f", "cfa_3f", "soc_2nd", "sem_2x2"};
}

std::vector<std::string> hub_ids() {
  return {"cfa_3f_rescov", "path_recursive", "cfa_6f_2ind"};
}

std::vector<std::string> all_ids() {
  auto v = scaling_ids();
  for (auto& h : hub_ids()) v.push_back(h);
  return v;
}

bool supports(const std::string& id, int p) {
  if (id == "cfa_1f")         return p >= 3;
  if (id == "cfa_3f")         return p % 3 == 0 && p / 3 >= 2;
  if (id == "soc_2nd")        return p >= 12 && p % 3 == 0;
  if (id == "sem_2x2")        return p >= 12 && p % 4 == 0;
  if (id == "cfa_3f_rescov")  return p == 12;
  if (id == "path_recursive") return p == 12;
  if (id == "cfa_6f_2ind")    return p == 12;
  return false;
}

bool has_latents(const std::string& id) { return id != "path_recursive"; }

ModelCase make_case(const std::string& id, int p) {
  ModelCase c;
  c.id = id;
  c.p  = p;

  if (id == "cfa_1f") {
    const Eigen::MatrixXd L = block_lambda(p, 1);
    c.syntax = block_cfa_syntax(p, 1);
    c.sigma  = measurement_sigma(L, compound_phi(1, 0.0));

  } else if (id == "cfa_3f") {
    const Eigen::MatrixXd L = block_lambda(p, 3);
    c.syntax = block_cfa_syntax(p, 3);
    c.sigma  = measurement_sigma(L, compound_phi(3, kPhiOff));

  } else if (id == "cfa_3f_rescov") {
    const Eigen::MatrixXd L   = block_lambda(p, 3);
    const int             per = p / 3;
    c.syntax = block_cfa_syntax(p, 3);
    c.sigma  = measurement_sigma(L, compound_phi(3, kPhiOff));
    // One residual covariance inside each factor: (x1,x2), (x5,x6), (x9,x10).
    for (int f = 0; f < 3; ++f) {
      const int a = f * per, b = f * per + 1;
      c.syntax += xname(a) + " ~~ " + xname(b) + "\n";
      c.sigma(a, b) += kResCov;
      c.sigma(b, a) += kResCov;
    }

  } else if (id == "soc_2nd") {
    // Second-order factor g over three first-order factors. Ψ_first carries
    // the second-order structure: γγ' · var(g) + diag(1 - γ²).
    const Eigen::MatrixXd L     = block_lambda(p, 3);
    const Eigen::VectorXd gamma = Eigen::VectorXd::Constant(3, kGamma);
    Eigen::MatrixXd       Psi   = gamma * gamma.transpose();
    Psi.diagonal().array() += 1.0 - kGamma * kGamma;
    c.syntax = block_cfa_syntax(p, 3);
    c.syntax += "g =~ " + fname(0) + " + " + fname(1) + " + " + fname(2) + "\n";
    c.sigma = measurement_sigma(L, Psi);

  } else if (id == "sem_2x2") {
    // Two exogenous factors, two endogenous: f3 <- f1 + f2, f4 <- f3.
    const Eigen::MatrixXd L = block_lambda(p, 4);
    Eigen::MatrixXd       B = Eigen::MatrixXd::Zero(4, 4);
    B(2, 0) = kBeta1;
    B(2, 1) = kBeta1;
    B(3, 2) = kBeta2;
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(4, 4);
    Phi(0, 0) = 1.0;
    Phi(1, 1) = 1.0;
    Phi(0, 1) = kPhiOff;
    Phi(1, 0) = kPhiOff;
    // Disturbance variances giving unit total factor variance.
    Phi(2, 2) = 1.0 - (2.0 * kBeta1 * kBeta1 + 2.0 * kBeta1 * kBeta1 * kPhiOff);
    Phi(3, 3) = 1.0 - kBeta2 * kBeta2;
    const Eigen::MatrixXd I  = Eigen::MatrixXd::Identity(4, 4);
    const Eigen::MatrixXd Ai = (I - B).inverse();
    const Eigen::MatrixXd Psi = Ai * Phi * Ai.transpose();
    c.syntax = block_cfa_syntax(p, 4);
    c.syntax += fname(2) + " ~ " + fname(0) + " + " + fname(1) + "\n";
    c.syntax += fname(3) + " ~ " + fname(2) + "\n";
    c.sigma = measurement_sigma(L, Psi);

  } else if (id == "cfa_6f_2ind") {
    const int             n_fac = p / 2;
    const Eigen::MatrixXd L     = block_lambda(p, n_fac);
    c.syntax = block_cfa_syntax(p, n_fac);
    c.sigma  = measurement_sigma(L, compound_phi(n_fac, kPhiOff));

  } else if (id == "path_recursive") {
    // All-observed recursive chain: Λ = I, structure lives entirely in Β.
    // First three variables exogenous; each later one regressed on its three
    // predecessors.
    Eigen::MatrixXd B   = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(p, p);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) Phi(i, j) = (i == j) ? 1.0 : kPhiOff;
    for (int r = 3; r < p; ++r) {
      Phi(r, r) = 0.40;
      B(r, r - 1) = kPath;
      B(r, r - 2) = kPath;
      B(r, r - 3) = kPath;
      c.syntax += vname(r) + " ~ " + vname(r - 1) + " + " + vname(r - 2) +
                  " + " + vname(r - 3) + "\n";
    }
    const Eigen::MatrixXd I  = Eigen::MatrixXd::Identity(p, p);
    const Eigen::MatrixXd Ai = (I - B).inverse();
    c.sigma = Ai * Phi * Ai.transpose();
  }

  standardize(c.sigma);
  return c;
}

}  // namespace magmaan::bench
