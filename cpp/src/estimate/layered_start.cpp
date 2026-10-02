#include "magmaan/estimate/layered_start.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "magmaan/error.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::estimate {

namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using model::MatId;

constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();

std::size_t us(int i) { return static_cast<std::size_t>(i); }
int to_int(std::size_t i) { return static_cast<int>(i); }
int to_int(Index i) { return static_cast<int>(i); }
double sgn(double x) { return x < 0.0 ? -1.0 : 1.0; }
double clamp_to(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

// One matrix entry of the model: a free parameter, a fixed value, or absent
// (an absent entry is a fixed zero).
struct Slot {
  int free = -1;
  double value = 0.0;
  bool present = false;
  bool is_free() const { return present && free >= 0; }
  bool nonzero() const { return present && (free >= 0 || value != 0.0); }
  double fixed() const { return present && free < 0 ? value : 0.0; }
};

struct Grid {
  int cols = 0;
  std::vector<Slot> cells;
  void init(int r, int c) { cols = c; cells.assign(us(r) * us(c), Slot{}); }
  Slot& operator()(int r, int c) { return cells[us(r) * us(cols) + us(c)]; }
  const Slot& operator()(int r, int c) const { return cells[us(r) * us(cols) + us(c)]; }
};

struct Pattern {
  int p = 0, m = 0;
  Grid lam, the, bet, psi, nu, alp;
};

struct Loc {
  int block = -1;
  MatId mat = MatId::Lambda;
  int row = -1, col = -1;
};

struct Layout {
  std::vector<Pattern> blocks;
  std::vector<Loc> loc;  // per free parameter, first occurrence
};

void put(Pattern& P, MatId mat, int r, int c, const Slot& s) {
  switch (mat) {
    case MatId::Lambda: P.lam(r, c) = s; break;
    case MatId::Theta: P.the(r, c) = s; P.the(c, r) = s; break;
    case MatId::Psi: P.psi(r, c) = s; P.psi(c, r) = s; break;
    case MatId::Beta: P.bet(r, c) = s; break;
    case MatId::Nu: P.nu(r, 0) = s; break;
    case MatId::Alpha: P.alp(r, 0) = s; break;
  }
}

Layout build_layout(const spec::LatentStructure& pt, const model::MatrixRep& rep) {
  Layout out;
  out.blocks.resize(rep.dims.size());
  for (std::size_t b = 0; b < rep.dims.size(); ++b) {
    Pattern& P = out.blocks[b];
    P.p = rep.dims[b].n_observed;
    P.m = rep.dims[b].n_latent;
    P.lam.init(P.p, P.m); P.the.init(P.p, P.p);
    P.bet.init(P.m, P.m); P.psi.init(P.m, P.m);
    P.nu.init(P.p, 1); P.alp.init(P.m, 1);
  }
  out.loc.assign(us(pt.n_free()), Loc{});
  for (const auto& sc : rep.structural_cells) {
    if (sc.mat != MatId::Lambda || us(static_cast<int>(sc.block)) >= out.blocks.size()) continue;
    Slot s; s.present = true; s.value = sc.value;
    out.blocks[us(static_cast<int>(sc.block))].lam(sc.row, sc.col) = s;
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.is_constraint_row(i)) continue;
    const auto& c = rep.cell_for_row[i];
    if (!c.used) continue;
    const int b = static_cast<int>(c.block);
    if (b < 0 || us(b) >= out.blocks.size()) continue;
    Slot s; s.present = true;
    if (pt.free[i] > 0) s.free = pt.free[i] - 1;
    else s.value = std::isfinite(pt.fixed_value[i]) ? pt.fixed_value[i] : 0.0;
    put(out.blocks[us(b)], c.mat, c.row, c.col, s);
    if (s.free >= 0 && out.loc[us(s.free)].block < 0)
      out.loc[us(s.free)] = Loc{b, c.mat, c.row, c.col};
  }
  return out;
}

// Loading ratios relative to U[0] on a standardized covariance: FABIN3 (Hägglund
// 1982) for four or more indicators, triads for three, an instrumental ratio
// with instruments outside the set for two. `masked(a, b)` marks pairs whose
// residual covariance is free; they are neither equations nor instruments.
template <class Masked>
VectorXd loading_ratios(const MatrixXd& R, const std::vector<int>& U,
                        const std::vector<int>& outside, const Masked& masked) {
  const int n = to_int(U.size());
  VectorXd r = VectorXd::Ones(n);
  const int r0 = U[0];
  auto outside_ratio = [&](int y) {
    double num = 0.0, den = 0.0;
    for (int z : outside) {
      if (masked(z, r0) || masked(z, y)) continue;
      num += R(y, z) * R(r0, z);
      den += R(r0, z) * R(r0, z);
    }
    const double q = den > 1e-10 ? num / den : not_a_number;
    return (std::isfinite(q) && std::abs(q) > 1e-6) ? q : sgn(R(y, r0));
  };
  if (n == 1) return r;
  if (n == 2) { r(1) = outside_ratio(U[1]); return r; }
  if (n == 3) {
    const int y1 = U[1], y2 = U[2];
    const bool ok1 = !masked(y1, y2) && !masked(r0, y2) && std::abs(R(r0, y2)) > 1e-8;
    const bool ok2 = !masked(y1, y2) && !masked(r0, y1) && std::abs(R(r0, y1)) > 1e-8;
    r(1) = ok1 ? R(y1, y2) / R(r0, y2) : outside_ratio(y1);
    r(2) = ok2 ? R(y1, y2) / R(r0, y1) : outside_ratio(y2);
    return r;
  }
  for (int k = 1; k < n; ++k) {
    const int y = U[us(k)];
    std::vector<int> inst;
    for (int q = 1; q < n; ++q) {
      const int z = U[us(q)];
      if (z == y || masked(z, y) || masked(z, r0)) continue;
      inst.push_back(z);
    }
    const int ni = to_int(inst.size());
    double lam = not_a_number;
    if (ni > 0) {
      VectorXd s23(ni), s31(ni);
      MatrixXd S33(ni, ni);
      for (int a = 0; a < ni; ++a) {
        s23(a) = R(y, inst[us(a)]);
        s31(a) = R(inst[us(a)], r0);
        for (int b = 0; b < ni; ++b) S33(a, b) = R(inst[us(a)], inst[us(b)]);
      }
      Eigen::LDLT<MatrixXd> ldlt(S33);
      if (ldlt.info() == Eigen::Success) {
        const VectorXd t = ldlt.solve(s31);
        const double den = s31.dot(t);
        if (std::abs(den) > 1e-12) lam = s23.dot(t) / den;
      }
      if (!std::isfinite(lam)) {
        const double den = s31.squaredNorm();
        if (den > 1e-12) lam = s23.dot(s31) / den;
      }
    }
    r(k) = (std::isfinite(lam) && std::abs(lam) > 1e-6) ? lam : outside_ratio(y);
  }
  return r;
}

// Common variance of U[0] (standardized) from off-diagonal pairs, given ratios.
template <class Masked>
double reference_communality(const MatrixXd& R, const std::vector<int>& U,
                             const VectorXd& r, double fallback, const Masked& masked) {
  double num = 0.0, den = 0.0;
  const int n = to_int(U.size());
  for (int a = 0; a < n; ++a)
    for (int b = a + 1; b < n; ++b) {
      if (masked(U[us(a)], U[us(b)])) continue;
      const double w = r(a) * r(b);
      num += w * R(U[us(a)], U[us(b)]);
      den += w * w;
    }
  double psi = den > 1e-12 ? num / den : not_a_number;
  if (!std::isfinite(psi) || psi <= 0.0) psi = fallback;
  return clamp_to(psi, 0.05, 0.95);
}

// Eigenvalue floor on the correlation scale; keeps the diagonal.
MatrixXd floor_pd(const MatrixXd& A, double floor) {
  const Index n = A.rows();
  VectorXd d = A.diagonal();
  for (Index i = 0; i < n; ++i) if (!(d(i) > 0.0)) d(i) = 1.0;
  const VectorXd s = d.cwiseSqrt();
  MatrixXd C = s.cwiseInverse().asDiagonal() * (0.5 * (A + A.transpose())) *
               s.cwiseInverse().asDiagonal();
  C.diagonal().setOnes();
  Eigen::SelfAdjointEigenSolver<MatrixXd> es(C);
  if (es.info() != Eigen::Success) return s.asDiagonal() * MatrixXd::Identity(n, n) * s.asDiagonal();
  VectorXd ev = es.eigenvalues();
  bool clipped = false;
  for (Index i = 0; i < n; ++i) if (ev(i) < floor) { ev(i) = floor; clipped = true; }
  if (clipped) {
    C = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    const VectorXd cd = C.diagonal().cwiseSqrt().cwiseInverse();
    C = cd.asDiagonal() * C * cd.asDiagonal();
  }
  return s.asDiagonal() * C * s.asDiagonal();
}

// Per-block measurement work in the standardized observed metric.
struct Work {
  MatrixXd S, R;
  VectorXd sd, smc;
  std::vector<char> measured, estimated, exogenous;
  std::vector<int> ref_row;   // reference indicator per estimated latent
  MatrixXd Lw;                // working loadings: 0 absent, NaN unknown
  VectorXd theta_std;         // standardized uniqueness priors
  std::vector<int> M, mpos;   // measured latents and their positions
  MatrixXd Phi;               // working covariance of the measured latents
  VectorXd a;                 // working → user latent scale (signed)
  std::vector<char> determined;
  bool rank_deficient = false; // latent covariance not identified by the measurement layer
  VectorXd v;                 // user-chart latent variances (for weights)
  bool ok = true;
};

bool row_known(const Work& W, int i) {
  for (Index j = 0; j < W.Lw.cols(); ++j)
    if (W.Lw(i, j) != 0.0 && std::isfinite(W.Lw(i, j))) return true;
  return false;
}

// Weighted least squares for the working latent covariance. Diagonal equations
// carry the uniqueness prior (weight .1) unless the residual variance is fixed.
bool fit_latent_covariance(Work& W, const Pattern& P, std::vector<std::string>& notes, int block) {
  const int nM = to_int(W.M.size());
  if (nM == 0) { W.Phi.resize(0, 0); return true; }
  std::vector<int> uid(us(nM) * us(nM), -1);
  int nu = 0;
  for (int jj = 0; jj < nM; ++jj)
    for (int kk = jj; kk < nM; ++kk) {
      const int j = W.M[us(jj)], k = W.M[us(kk)];
      if (jj != kk && W.exogenous[us(j)] && W.exogenous[us(k)] && !P.psi(j, k).is_free() &&
          P.psi(j, k).fixed() == 0.0) continue;  // a structural zero
      uid[us(jj) * us(nM) + us(kk)] = nu++;
    }
  std::vector<int> rows;
  for (int i = 0; i < P.p; ++i) if (row_known(W, i)) rows.push_back(i);
  const int nr = to_int(rows.size());
  std::vector<std::pair<int, int>> eqs;
  for (int x = 0; x < nr; ++x)
    for (int y = x; y < nr; ++y) {
      const int a = rows[us(x)], b = rows[us(y)];
      if (a != b && P.the(a, b).is_free()) continue;
      eqs.emplace_back(a, b);
    }
  const int ne = to_int(eqs.size());
  MatrixXd X = MatrixXd::Zero(ne, nu);
  VectorXd yv(ne);
  for (int e = 0; e < ne; ++e) {
    const int a = eqs[us(e)].first, b = eqs[us(e)].second;
    double w = 1.0, target;
    if (a == b) {
      const bool fixed_theta = !P.the(a, a).is_free();
      target = W.R(a, a) - (fixed_theta ? P.the(a, a).fixed() / W.S(a, a) : W.theta_std(a));
      w = fixed_theta ? 1.0 : std::sqrt(0.1);
    } else {
      target = W.R(a, b) - P.the(a, b).fixed() / (W.sd(a) * W.sd(b));
    }
    yv(e) = w * target;
    for (int jj = 0; jj < nM; ++jj)
      for (int kk = jj; kk < nM; ++kk) {
        const int id = uid[us(jj) * us(nM) + us(kk)];
        if (id < 0) continue;
        const int j = W.M[us(jj)], k = W.M[us(kk)];
        const double coef = jj == kk ? W.Lw(a, j) * W.Lw(b, j)
                                     : W.Lw(a, j) * W.Lw(b, k) + W.Lw(a, k) * W.Lw(b, j);
        X(e, id) = w * coef;
      }
  }
  VectorXd sol = VectorXd::Zero(nu);
  W.rank_deficient = false;
  if (ne > 0 && nu > 0) {
    Eigen::CompleteOrthogonalDecomposition<MatrixXd> cod(X);
    sol = cod.solve(yv);
    W.rank_deficient = cod.rank() < nu;
  }
  W.Phi = MatrixXd::Zero(nM, nM);
  for (int jj = 0; jj < nM; ++jj)
    for (int kk = jj; kk < nM; ++kk) {
      const int id = uid[us(jj) * us(nM) + us(kk)];
      if (id < 0) continue;
      W.Phi(jj, kk) = W.Phi(kk, jj) = sol(id);
    }
  for (int jj = 0; jj < nM; ++jj) {
    const int id = uid[us(jj) * us(nM) + us(jj)];
    const bool informed = id >= 0 && ne > 0 && X.col(id).squaredNorm() > 1e-14;
    if (!informed || !(W.Phi(jj, jj) > 1e-6) || !std::isfinite(W.Phi(jj, jj))) {
      if (!informed)
        notes.push_back("block " + std::to_string(block + 1) +
                        ": a measured latent has no usable moments; unit working variance");
      W.Phi(jj, jj) = informed ? std::max(W.Phi(jj, jj), 0.05) : 1.0;
      if (!std::isfinite(W.Phi(jj, jj))) W.Phi(jj, jj) = 1.0;
    }
  }
  if (!W.Phi.allFinite()) return false;
  W.Phi = floor_pd(W.Phi, 1e-3);
  return W.Phi.allFinite();
}

// Refit the working loadings of one indicator row against the latent covariance.
void refit_row(Work& W, const Pattern& P, int a) {
  std::vector<int> J, F;
  for (int j = 0; j < P.m; ++j) {
    if (!P.lam(a, j).nonzero()) continue;
    if (W.estimated[us(j)]) J.push_back(j); else F.push_back(j);
  }
  if (J.empty()) return;
  const int nM = to_int(W.M.size());
  std::vector<VectorXd> xs;
  std::vector<double> ys;
  for (int b = 0; b < P.p; ++b) {
    if (b == a || P.the(a, b).is_free() || !row_known(W, b)) continue;
    VectorXd lb = VectorXd::Zero(nM);
    for (int jj = 0; jj < nM; ++jj) lb(jj) = W.Lw(b, W.M[us(jj)]);
    const VectorXd g = W.Phi * lb;
    double y = W.R(a, b) - P.the(a, b).fixed() / (W.sd(a) * W.sd(b));
    for (int j : F) y -= W.Lw(a, j) * g(W.mpos[us(j)]);
    VectorXd x(to_int(J.size()));
    for (std::size_t q = 0; q < J.size(); ++q) x(to_int(q)) = g(W.mpos[us(J[q])]);
    xs.push_back(x);
    ys.push_back(y);
  }
  // A small ridge toward the current values keeps loadings the moments do not
  // inform where they are instead of zeroing them.
  const int nj = to_int(J.size());
  const int ne = to_int(xs.size());
  MatrixXd X = MatrixXd::Zero(ne + nj, nj);
  VectorXd y = VectorXd::Zero(ne + nj);
  for (int e = 0; e < ne; ++e) { X.row(e) = xs[us(e)].transpose(); y(e) = ys[us(e)]; }
  const double ridge = std::sqrt(1e-4);
  for (int q = 0; q < nj; ++q) { X(ne + q, q) = ridge; y(ne + q) = ridge * W.Lw(a, J[us(q)]); }
  const VectorXd sol = X.colPivHouseholderQr().solve(y);
  if (!sol.allFinite()) return;
  for (std::size_t q = 0; q < J.size(); ++q)
    W.Lw(a, J[q]) = clamp_to(sol(to_int(q)), -std::sqrt(0.95), std::sqrt(0.95));
}

void update_uniqueness(Work& W, const Pattern& P) {
  const int nM = to_int(W.M.size());
  for (int i = 0; i < P.p; ++i) {
    if (!row_known(W, i)) { W.theta_std(i) = 1.0; continue; }
    VectorXd l = VectorXd::Zero(nM);
    for (int jj = 0; jj < nM; ++jj) l(jj) = W.Lw(i, W.M[us(jj)]);
    const double h = l.dot(W.Phi * l);
    W.theta_std(i) = std::isfinite(h) ? clamp_to(1.0 - h, 0.05, 0.95) : 0.5;
  }
}

bool measurement(Work& W, const Pattern& P, std::vector<std::string>& notes, int block) {
  const int p = P.p, m = P.m;
  W.sd = W.S.diagonal().cwiseSqrt();
  for (int i = 0; i < p; ++i) if (!(W.sd(i) > 0.0) || !std::isfinite(W.sd(i))) return false;
  W.R = W.sd.cwiseInverse().asDiagonal() * W.S * W.sd.cwiseInverse().asDiagonal();
  W.smc = VectorXd::Constant(p, 0.5);
  Eigen::LDLT<MatrixXd> ldlt(W.R);
  if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
    const MatrixXd Ri = ldlt.solve(MatrixXd::Identity(p, p));
    for (int i = 0; i < p; ++i) W.smc(i) = clamp_to(1.0 - 1.0 / Ri(i, i), 0.05, 0.95);
  } else {
    for (int i = 0; i < p; ++i) {
      double best = 0.05;
      for (int k = 0; k < p; ++k) if (k != i) best = std::max(best, W.R(i, k) * W.R(i, k));
      W.smc(i) = clamp_to(best, 0.05, 0.95);
    }
  }
  W.measured.assign(us(m), 0); W.estimated.assign(us(m), 0); W.exogenous.assign(us(m), 1);
  W.ref_row.assign(us(m), -1);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < p; ++i) {
      if (!P.lam(i, j).nonzero()) continue;
      W.measured[us(j)] = 1;
      if (P.lam(i, j).is_free()) W.estimated[us(j)] = 1;
    }
    for (int k = 0; k < m; ++k) if (k != j && P.bet(j, k).nonzero()) W.exogenous[us(j)] = 0;
  }
  W.M.clear(); W.mpos.assign(us(m), -1);
  for (int j = 0; j < m; ++j) if (W.measured[us(j)]) { W.mpos[us(j)] = to_int(W.M.size()); W.M.push_back(j); }

  auto masked = [&P](int a, int b) { return a != b && P.the(a, b).is_free(); };
  W.Lw = MatrixXd::Zero(p, m);
  std::vector<int> nnz(us(p), 0);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < m; ++j) if (P.lam(i, j).nonzero()) ++nnz[us(i)];
  for (int j = 0; j < m; ++j) {
    if (!W.measured[us(j)]) continue;
    for (int i = 0; i < p; ++i) {
      if (!P.lam(i, j).nonzero()) continue;
      W.Lw(i, j) = W.estimated[us(j)] ? not_a_number : P.lam(i, j).value / W.sd(i);
    }
  }
  for (int j = 0; j < m; ++j) {
    if (!W.estimated[us(j)]) continue;
    std::vector<int> all, pure;
    for (int i = 0; i < p; ++i) {
      if (!P.lam(i, j).nonzero()) continue;
      all.push_back(i);
      if (nnz[us(i)] == 1) pure.push_back(i);
    }
    // Pure indicators (any number) anchor the shape; cross-loadings are then
    // regressed on the anchored latent covariance. Without pure indicators the
    // shape uses every indicator of the latent.
    std::vector<int> U = pure.empty() ? all : pure;
    // The marker (first fixed nonzero loading) is the reference when it is in U.
    std::size_t ref = 0;
    for (std::size_t q = 0; q < U.size(); ++q)
      if (!P.lam(U[q], j).is_free()) { ref = q; break; }
    std::swap(U[0], U[ref]);
    std::vector<int> outside;
    for (int i = 0; i < p; ++i) if (!P.lam(i, j).nonzero() && nnz[us(i)] > 0) outside.push_back(i);
    const VectorXd r = loading_ratios(W.R, U, outside, masked);
    const double psi = reference_communality(W.R, U, r, W.smc(U[0]), masked);
    const double root = std::sqrt(psi);
    for (std::size_t q = 0; q < U.size(); ++q)
      W.Lw(U[q], j) = clamp_to(r(to_int(q)) * root, -std::sqrt(0.95), std::sqrt(0.95));
    W.ref_row[us(j)] = U[0];
  }
  // Rows loading on several latents, or outside every shape set, start at zero
  // on their unknown entries and are refitted against the latent covariance in
  // alternating sweeps with the covariance fit.
  std::vector<int> refit_rows;
  for (int i = 0; i < p; ++i) {
    bool needs = nnz[us(i)] > 1;
    for (int j = 0; j < m; ++j)
      if (std::isnan(W.Lw(i, j))) { needs = true; W.Lw(i, j) = 0.0; }
    if (needs) refit_rows.push_back(i);
  }
  // Uniqueness priors: standardized communality of the shape rows; rows of
  // latents with fixed loadings (user-chart working loadings) use the squared
  // multiple correlation, a lower bound on the communality.
  W.theta_std = VectorXd::Ones(p);
  for (int i = 0; i < p; ++i) {
    if (!row_known(W, i)) continue;
    double h = 0.0;
    bool fixed_chart = false;
    for (int j = 0; j < m; ++j) {
      if (W.Lw(i, j) == 0.0) continue;
      if (!W.estimated[us(j)]) fixed_chart = true;
      h += W.Lw(i, j) * W.Lw(i, j);
    }
    W.theta_std(i) = clamp_to(fixed_chart ? 1.0 - W.smc(i) : 1.0 - h, 0.05, 0.95);
  }
  std::vector<std::string> quiet;
  if (!fit_latent_covariance(W, P, refit_rows.empty() ? notes : quiet, block)) return false;
  for (int sweep = 0; sweep < 3 && !refit_rows.empty(); ++sweep) {
    for (int i : refit_rows) refit_row(W, P, i);
    update_uniqueness(W, P);
    if (!fit_latent_covariance(W, P, sweep == 2 ? notes : quiet, block)) return false;
  }
  update_uniqueness(W, P);
  return true;
}

// Variance of latent j when it and its whole ancestry are fixed by the model
// (e.g. a unit-variance phantom driving j through a fixed path); NaN otherwise.
double pinned_variance(const Pattern& P, int j) {
  std::vector<int> C{j};
  std::vector<char> in(us(P.m), 0);
  in[us(j)] = 1;
  for (std::size_t q = 0; q < C.size(); ++q)
    for (int x = 0; x < P.m; ++x)
      if (x != C[q] && P.bet(C[q], x).nonzero() && !in[us(x)]) { in[us(x)] = 1; C.push_back(x); }
  const int n = to_int(C.size());
  MatrixXd B = MatrixXd::Zero(n, n), Psi = MatrixXd::Zero(n, n);
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      const Slot& sb = P.bet(C[us(a)], C[us(b)]);
      const Slot& sp = P.psi(C[us(a)], C[us(b)]);
      if (sb.is_free() || sp.is_free()) return not_a_number;
      B(a, b) = a == b ? 0.0 : sb.fixed();
      Psi(a, b) = sp.fixed();
    }
  Eigen::FullPivLU<MatrixXd> lu(MatrixXd::Identity(n, n) - B);
  if (!lu.isInvertible()) return not_a_number;
  const MatrixXd A = lu.inverse();
  const double v = (A * Psi * A.transpose())(0, 0);
  return (std::isfinite(v) && v > 0.0) ? v : not_a_number;
}

struct ScaleEquation {
  std::vector<std::pair<int, double>> terms;
  double rhs = 0.0, weight = 1.0;
  bool hard = false;
};

// Solve per-(block, latent) working→user scales jointly in log scale.
void solve_scales(std::vector<Work>& works, const std::vector<Pattern>& pats,
                  const spec::LatentStructure& pt, const Layout& lay) {
  std::vector<std::vector<int>> node(works.size());
  std::vector<std::pair<int, int>> owner;
  for (std::size_t b = 0; b < works.size(); ++b) {
    node[b].assign(us(pats[b].m), -1);
    for (int j = 0; j < pats[b].m; ++j)
      if (works[b].ok && works[b].estimated[us(j)]) {
        node[b][us(j)] = to_int(owner.size());
        owner.emplace_back(to_int(b), j);
      }
  }
  const int nn = to_int(owner.size());
  for (std::size_t b = 0; b < works.size(); ++b) {
    works[b].a = VectorXd::Ones(pats[b].m);
    works[b].determined.assign(us(pats[b].m), 1);
  }
  if (nn == 0) return;
  std::vector<ScaleEquation> eqs;
  std::vector<double> anchor_sign(us(nn), 0.0);
  struct Link { int x, y; double rel; };
  std::vector<Link> links;
  std::vector<int> parent(us(nn));
  for (int i = 0; i < nn; ++i) parent[us(i)] = i;
  auto find = [&parent](int x) {
    while (parent[us(x)] != x) { parent[us(x)] = parent[us(parent[us(x)])]; x = parent[us(x)]; }
    return x;
  };
  std::vector<char> hard(us(nn), 0);
  auto work_loading = [&](int b, int i, int j) { return works[us(b)].sd(i) * works[us(b)].Lw(i, j); };

  for (int n = 0; n < nn; ++n) {
    const int b = owner[us(n)].first, j = owner[us(n)].second;
    const Pattern& P = pats[us(b)];
    for (int i = 0; i < P.p; ++i) {
      const Slot& s = P.lam(i, j);
      if (!s.present || s.is_free() || s.value == 0.0) continue;
      const double w = work_loading(b, i, j);
      if (!(std::abs(w) > 1e-10) || !std::isfinite(w)) continue;
      eqs.push_back({{{n, 1.0}}, std::log(std::abs(w / s.value)), 1e3, true});
      if (anchor_sign[us(n)] == 0.0) anchor_sign[us(n)] = sgn(w) * sgn(s.value);
      hard[us(n)] = 1;
    }
    const double pv = pinned_variance(P, j);
    const int jj = works[us(b)].mpos[us(j)];
    if (std::isfinite(pv) && jj >= 0 && works[us(b)].Phi(jj, jj) > 0.0) {
      eqs.push_back({{{n, 1.0}}, 0.5 * std::log(pv / works[us(b)].Phi(jj, jj)), 1e3, true});
      if (anchor_sign[us(n)] == 0.0) anchor_sign[us(n)] = 1.0;
      hard[us(n)] = 1;
    }
    const int r = works[us(b)].ref_row[us(j)];
    const double wr = r >= 0 ? work_loading(b, r, j) : 1.0;
    eqs.push_back({{{n, 1.0}}, std::log(std::abs(wr) > 1e-10 ? std::abs(wr) : 1.0), 1e-3, false});
  }
  // Effect coding: a linear row over the free loadings of one (block, latent).
  const std::size_t nf = us(pt.n_free());
  const std::size_t nlin = pt.lin_constraint_d.size();
  if (nf > 0 && pt.lin_constraint_R.size() == nlin * nf) {
    for (std::size_t r = 0; r < nlin; ++r) {
      const double d = pt.lin_constraint_d[r];
      if (d == 0.0) continue;
      int b = -1, j = -1;
      double sum = 0.0;
      bool ok = true, any = false;
      for (std::size_t k = 0; k < nf && ok; ++k) {
        const double c = pt.lin_constraint_R[r * nf + k];
        if (c == 0.0) continue;
        const Loc& L = lay.loc[k];
        if (L.block < 0 || L.mat != MatId::Lambda) { ok = false; break; }
        if (b < 0) { b = L.block; j = L.col; }
        if (L.block != b || L.col != j || node[us(b)][us(j)] < 0) { ok = false; break; }
        sum += c * work_loading(b, L.row, j);
        any = true;
      }
      if (!ok || !any || !std::isfinite(sum) || std::abs(sum) < 1e-12) continue;
      const int n = node[us(b)][us(j)];
      eqs.push_back({{{n, 1.0}}, std::log(std::abs(sum / d)), 1e3, true});
      if (anchor_sign[us(n)] == 0.0) anchor_sign[us(n)] = sgn(sum / d);
      hard[us(n)] = 1;
    }
  }
  // Cross-slot loading equalities (shared labels, group.equal) link scales.
  if (pt.eq_groups.size() == nf) {
    std::vector<int> first(nf, -1);
    for (std::size_t k = 0; k < nf; ++k) {
      const Loc& L = lay.loc[k];
      if (L.block < 0 || L.mat != MatId::Lambda || node[us(L.block)][us(L.col)] < 0) continue;
      const int g = pt.eq_groups[k];
      if (g < 0 || us(g) >= nf) continue;
      if (first[us(g)] < 0) { first[us(g)] = to_int(k); continue; }
      const Loc& F = lay.loc[us(first[us(g)])];
      const int x = node[us(F.block)][us(F.col)], y = node[us(L.block)][us(L.col)];
      if (x == y) continue;
      const double wx = work_loading(F.block, F.row, F.col), wy = work_loading(L.block, L.row, L.col);
      if (!(std::abs(wx) > 1e-10) || !(std::abs(wy) > 1e-10)) continue;
      eqs.push_back({{{x, 1.0}, {y, -1.0}}, std::log(std::abs(wx)) - std::log(std::abs(wy)), 1.0, false});
      links.push_back({x, y, sgn(wx) * sgn(wy)});
      parent[us(find(x))] = find(y);
    }
  }
  const int ne = to_int(eqs.size());
  MatrixXd X = MatrixXd::Zero(ne, nn);
  VectorXd y(ne);
  for (int e = 0; e < ne; ++e) {
    const auto& q = eqs[us(e)];
    for (const auto& t : q.terms) X(e, t.first) += q.weight * t.second;
    y(e) = q.weight * q.rhs;
  }
  VectorXd loga = X.colPivHouseholderQr().solve(y);
  std::vector<char> comp_hard(us(nn), 0);
  for (int n = 0; n < nn; ++n) if (hard[us(n)]) comp_hard[us(find(n))] = 1;
  // Signs: anchors first, propagated along equality links; otherwise positive.
  std::vector<double> sign = anchor_sign;
  for (bool changed = true; changed;) {
    changed = false;
    for (const auto& L : links) {
      if (sign[us(L.x)] != 0.0 && sign[us(L.y)] == 0.0) { sign[us(L.y)] = sign[us(L.x)] * L.rel; changed = true; }
      if (sign[us(L.y)] != 0.0 && sign[us(L.x)] == 0.0) { sign[us(L.x)] = sign[us(L.y)] * L.rel; changed = true; }
    }
  }
  for (int n = 0; n < nn; ++n) {
    const int b = owner[us(n)].first, j = owner[us(n)].second;
    const double la = std::isfinite(loga(n)) ? loga(n) : 0.0;
    works[us(b)].a(j) = (sign[us(n)] == 0.0 ? 1.0 : sign[us(n)]) * std::exp(la);
    works[us(b)].determined[us(j)] = comp_hard[us(find(n))];
  }
}

// ---- Structural layer --------------------------------------------------------

struct SubParam {
  char kind;  // 'B' path, 'P' latent (co)variance, 'A' log scale of a latent
  int r, c;
};

struct SubProblem {
  const Pattern* P = nullptr;
  const Work* W = nullptr;
  std::vector<SubParam> params;
  MatrixXd B0, Psi0;  // fixed parts
  VectorXd a0;        // signed scales of determined latents
  MatrixXd Linv;      // inverse Cholesky factor of the working target
  // Proper region for the start: latent variances stay above a fraction of
  // their initial values, free variances nonnegative, free covariances inside
  // their Cauchy-Schwarz bound, standardized free paths within `path_bound`.
  VectorXd var_floor;   // per latent (empty: unchecked)
  VectorXd path_bound;  // per parameter (B only)
};

bool proper(const SubProblem& sp, const MatrixXd& B, const MatrixXd& Psi, const MatrixXd& Phi) {
  if (sp.var_floor.size() == Phi.rows())
    for (Index j = 0; j < Phi.rows(); ++j)
      if (!(Phi(j, j) >= sp.var_floor(j))) return false;
  for (std::size_t q = 0; q < sp.params.size(); ++q) {
    const auto& s = sp.params[q];
    if (s.kind == 'P' && s.r == s.c && Psi(s.r, s.r) < 0.0) return false;
    if (s.kind == 'P' && s.r != s.c && Psi(s.r, s.r) > 0.0 && Psi(s.c, s.c) > 0.0 &&
        Psi(s.r, s.c) * Psi(s.r, s.c) > Psi(s.r, s.r) * Psi(s.c, s.c)) return false;
    if (s.kind == 'B' && sp.path_bound.size() > to_int(q) && Phi(s.r, s.r) > 0.0 &&
        std::abs(B(s.r, s.c)) * std::sqrt(std::max(Phi(s.c, s.c), 0.0) / Phi(s.r, s.r)) >
            sp.path_bound(to_int(q))) return false;
  }
  return true;
}

void unpack(const SubProblem& sp, const VectorXd& x, MatrixXd& B, MatrixXd& Psi, VectorXd& a) {
  B = sp.B0; Psi = sp.Psi0; a = sp.a0;
  for (std::size_t q = 0; q < sp.params.size(); ++q) {
    const auto& s = sp.params[q];
    const double v = x(to_int(q));
    if (s.kind == 'B') B(s.r, s.c) = v;
    else if (s.kind == 'P') { Psi(s.r, s.c) = v; Psi(s.c, s.r) = v; }
    else a(s.r) = std::exp(v);
  }
}

bool latent_moments(const MatrixXd& B, const MatrixXd& Psi, MatrixXd& A, MatrixXd& Phi) {
  const Index m = B.rows();
  // Observed-only models have an empty latent block; its moments need no solve.
  if (m == 0) {
    A.resize(0, 0);
    Phi.resize(0, 0);
    return true;
  }
  Eigen::FullPivLU<MatrixXd> lu(MatrixXd::Identity(m, m) - B);
  if (!lu.isInvertible()) return false;
  A = lu.inverse();
  Phi = A * Psi * A.transpose();
  return A.allFinite() && Phi.allFinite();
}

// GLS residuals of the latent-level fit, in the working gauge of the measured
// latents: E = L⁻¹ (Φ_MM ⊘ aaᵀ − Φ*) L⁻ᵀ with Φ* = LLᵀ, so ½‖E‖²_F is the GLS
// discrepancy (affine invariant). Jacobian columns are rank-two updates.
void pack(const MatrixXd& E, VectorXd& out, Index col_offset, MatrixXd* J, Index q) {
  const Index n = E.rows();
  Index e = 0;
  for (Index j = 0; j < n; ++j)
    for (Index k = j; k < n; ++k, ++e) {
      const double v = (j == k ? 1.0 : std::sqrt(2.0)) * E(j, k);
      if (J) (*J)(e, q) = v; else out(e + col_offset) = v;
    }
}

bool residuals(const SubProblem& sp, const VectorXd& x, VectorXd& res, MatrixXd* J) {
  MatrixXd B, Psi, A, Phi;
  VectorXd a;
  unpack(sp, x, B, Psi, a);
  if (!latent_moments(B, Psi, A, Phi)) return false;
  if (!proper(sp, B, Psi, Phi)) return false;
  const auto& M = sp.W->M;
  const MatrixXd& T = sp.W->Phi;
  const int nM = to_int(M.size());
  const int np = to_int(sp.params.size());
  const Index m = A.rows();
  VectorXd am(nM);
  MatrixXd AM(nM, m), PhiM(nM, m), Fw(nM, nM);
  for (int jj = 0; jj < nM; ++jj) {
    am(jj) = a(M[us(jj)]);
    AM.row(jj) = A.row(M[us(jj)]);
    PhiM.row(jj) = Phi.row(M[us(jj)]);
  }
  for (int jj = 0; jj < nM; ++jj)
    for (int kk = 0; kk < nM; ++kk) Fw(jj, kk) = Phi(M[us(jj)], M[us(kk)]) / (am(jj) * am(kk));
  const MatrixXd& Li = sp.Linv;
  const MatrixXd E = Li * (Fw - T) * Li.transpose();
  res.resize(nM * (nM + 1) / 2);
  pack(E, res, 0, nullptr, 0);
  if (!res.allFinite()) return false;
  if (!J) return true;
  J->setZero(res.size(), np);
  const VectorXd ainv = am.cwiseInverse();
  MatrixXd dE(nM, nM);
  for (int q = 0; q < np; ++q) {
    const auto& s = sp.params[us(q)];
    if (s.kind == 'B') {
      const VectorXd u = Li * ainv.cwiseProduct(AM.col(s.r));
      const VectorXd w = Li * ainv.cwiseProduct(PhiM.col(s.c));
      dE = u * w.transpose() + w * u.transpose();
    } else if (s.kind == 'P') {
      const VectorXd u = Li * ainv.cwiseProduct(AM.col(s.r));
      if (s.r == s.c) dE = u * u.transpose();
      else {
        const VectorXd v = Li * ainv.cwiseProduct(AM.col(s.c));
        dE = u * v.transpose() + v * u.transpose();
      }
    } else {
      const int uu = sp.W->mpos[us(s.r)];
      const VectorXd l = Li.col(uu);
      const VectorXd f = Li * Fw.col(uu);
      dE = -(l * f.transpose() + f * l.transpose());
    }
    pack(dE, res, 0, J, q);
  }
  return J->allFinite();
}

double half_norm2(const VectorXd& r) { return 0.5 * r.squaredNorm(); }

VectorXd levenberg_marquardt(const SubProblem& sp, VectorXd x, double& cost0, double& cost) {
  VectorXd r;
  if (!residuals(sp, x, r, nullptr)) { cost0 = cost = std::numeric_limits<double>::infinity(); return x; }
  cost0 = cost = half_norm2(r);
  const int np = to_int(x.size());
  if (np == 0) return x;
  MatrixXd J;
  double mu = -1.0;
  for (int it = 0; it < 300 && cost > 1e-24; ++it) {
    if (!residuals(sp, x, r, &J)) break;
    const MatrixXd H = J.transpose() * J;
    const VectorXd g = J.transpose() * r;
    const double hmax = std::max(H.diagonal().maxCoeff(), 1e-300);
    if (mu < 0.0) mu = 1e-3;
    bool accepted = false;
    for (int tries = 0; tries < 30; ++tries) {
      MatrixXd Hd = H;
      for (int q = 0; q < np; ++q) Hd(q, q) += mu * std::max(H(q, q), 1e-9 * hmax);
      const VectorXd step = Hd.ldlt().solve(-g);
      if (!step.allFinite()) { mu *= 4.0; continue; }
      VectorXd xn = x + step, rn;
      if (residuals(sp, xn, rn, nullptr)) {
        const double cn = half_norm2(rn);
        if (cn < cost) {
          const double drop = cost - cn;
          x = std::move(xn); cost = cn; mu = std::max(mu / 3.0, 1e-12);
          accepted = true;
          if (drop <= 1e-13 * (1.0 + cost)) it = 300;
          break;
        }
      }
      mu *= 4.0;
      if (mu > 1e14) break;
    }
    if (!accepted) break;
  }
  return x;
}

struct StructuralResult {
  MatrixXd B, Psi;
  VectorXd a, v;
  double cost0 = 0.0, cost = 0.0;
  bool ok = false;
};

StructuralResult structural_fit(const Pattern& P, Work& W) {
  StructuralResult out;
  const int m = P.m;
  const int nM = to_int(W.M.size());
  std::vector<char> inM(us(m), 0);
  for (int j : W.M) inM[us(j)] = 1;
  // User-chart target and latent variances.
  VectorXd v = VectorXd::Constant(m, not_a_number);
  double mean_t = 0.0;
  for (int jj = 0; jj < nM; ++jj) {
    const int j = W.M[us(jj)];
    v(j) = W.a(j) * W.a(j) * W.Phi(jj, jj);
    mean_t += v(j) / static_cast<double>(std::max(1, nM));
  }
  if (!(mean_t > 0.0)) mean_t = 1.0;
  auto T = [&](int j, int k) { return W.a(j) * W.a(k) * W.Phi(W.mpos[us(j)], W.mpos[us(k)]); };
  std::vector<int> npar(us(m), 0);
  std::vector<std::vector<int>> kids(us(m));
  for (int c = 0; c < m; ++c)
    for (int x = 0; x < m; ++x)
      if (x != c && P.bet(c, x).nonzero()) { ++npar[us(c)]; kids[us(x)].push_back(c); }
  auto share = [&](int c) {
    const Slot& s = P.psi(c, c);
    double delta = 0.5;
    if (!s.is_free()) delta = v(c) > 0.0 ? clamp_to(s.fixed() / v(c), 0.0, 0.95) : 0.5;
    return (1.0 - delta) / static_cast<double>(std::max(1, npar[us(c)]));
  };
  MatrixXd B = MatrixXd::Zero(m, m), Psi = MatrixXd::Zero(m, m);
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < m; ++c) {
      if (r != c && P.bet(r, c).present && !P.bet(r, c).is_free()) B(r, c) = P.bet(r, c).fixed();
      if (P.psi(r, c).present && !P.psi(r, c).is_free()) Psi(r, c) = P.psi(r, c).fixed();
    }
  std::vector<char> shaped(us(m), 0);  // structure-only latent initialized by a shape
  // Structure-only latents, children first.
  std::vector<char> done(us(m), 0);
  for (int j : W.M) done[us(j)] = 1;
  for (int pass = 0; pass <= m; ++pass) {
    bool progress = false;
    for (int q = 0; q < m; ++q) {
      if (done[us(q)]) continue;
      bool ready = true;
      for (int c : kids[us(q)]) if (!done[us(c)]) ready = false;
      if (!ready) continue;
      const Slot& d = P.psi(q, q);
      const bool fixed_var = !d.is_free() && d.fixed() > 0.0 && npar[us(q)] == 0;
      bool pure_shape = kids[us(q)].size() >= 3;
      for (int c : kids[us(q)])
        if (!inM[us(c)] || !P.bet(c, q).is_free() || npar[us(c)] != 1) pure_shape = false;
      if (pure_shape) {
        // A higher-order factor of measured latents: FABIN on their correlation.
        const auto& C = kids[us(q)];
        const int nc = to_int(C.size());
        MatrixXd Rc(nc, nc);
        for (int x = 0; x < nc; ++x)
          for (int y = 0; y < nc; ++y)
            Rc(x, y) = T(C[us(x)], C[us(y)]) / std::sqrt(v(C[us(x)]) * v(C[us(y)]));
        std::vector<int> U(us(nc));
        for (int x = 0; x < nc; ++x) U[us(x)] = x;
        auto dmask = [&](int x, int y) { return x != y && P.psi(C[us(x)], C[us(y)]).is_free(); };
        const VectorXd rr = loading_ratios(Rc, U, {}, dmask);
        const double root = std::sqrt(reference_communality(Rc, U, rr, 0.5, dmask));
        double vq = 0.0;
        VectorXd l(nc);
        for (int x = 0; x < nc; ++x) {
          l(x) = clamp_to(rr(x) * root, -std::sqrt(0.95), std::sqrt(0.95));
          vq += l(x) * l(x) * v(C[us(x)]) / static_cast<double>(nc);
        }
        if (fixed_var) vq = d.fixed();
        v(q) = vq > 0.0 ? vq : mean_t;
        for (int x = 0; x < nc; ++x) B(C[us(x)], q) = l(x) * std::sqrt(v(C[us(x)]) / v(q));
        shaped[us(q)] = 1;
      } else if (fixed_var) {
        v(q) = d.fixed();
      } else {
        std::vector<double> est;
        double mean_share = 0.0;
        for (int c : kids[us(q)]) {
          const Slot& s = P.bet(c, q);
          const double sv = share(c) * v(c);
          mean_share += sv / static_cast<double>(kids[us(q)].size());
          if (!s.is_free() && s.fixed() != 0.0) est.push_back(sv / (s.fixed() * s.fixed()));
        }
        if (!est.empty()) {
          std::sort(est.begin(), est.end());
          v(q) = est[est.size() / 2];
        } else if (!kids[us(q)].empty()) {
          v(q) = mean_share;
        } else {
          v(q) = d.fixed() > 0.0 ? d.fixed() : mean_t;
        }
        if (!(v(q) > 0.0) || !std::isfinite(v(q))) v(q) = mean_t;
      }
      done[us(q)] = 1;
      progress = true;
    }
    if (!progress) break;
  }
  for (int q = 0; q < m; ++q) if (!done[us(q)] || !(v(q) > 0.0)) v(q) = mean_t;

  // Paths.
  std::vector<double> explained(us(m), 0.0);
  for (int c = 0; c < m; ++c) {
    // Measured parents of a measured child: least squares on the target.
    if (inM[us(c)]) {
      std::vector<int> F, X;
      for (int x = 0; x < m; ++x) {
        if (x == c || !P.bet(c, x).nonzero() || !inM[us(x)]) continue;
        (P.bet(c, x).is_free() ? F : X).push_back(x);
      }
      if (!F.empty()) {
        const int nfp = to_int(F.size());
        MatrixXd TFF(nfp, nfp);
        VectorXd rhs(nfp);
        for (int x = 0; x < nfp; ++x) {
          rhs(x) = T(F[us(x)], c);
          for (int w : X) rhs(x) -= T(F[us(x)], w) * B(c, w);
          for (int y = 0; y < nfp; ++y) TFF(x, y) = T(F[us(x)], F[us(y)]);
        }
        Eigen::LDLT<MatrixXd> ldlt(TFF);
        VectorXd bf = ldlt.info() == Eigen::Success ? VectorXd(ldlt.solve(rhs)) : VectorXd(VectorXd::Zero(nfp));
        if (!bf.allFinite()) bf.setZero();
        for (int x = 0; x < nfp; ++x) B(c, F[us(x)]) = bf(x);
      }
      std::vector<int> Pm(F);
      Pm.insert(Pm.end(), X.begin(), X.end());
      double ex = 0.0;
      for (int x : Pm) for (int y : Pm) ex += B(c, x) * B(c, y) * T(x, y);
      if (ex > 0.9 * v(c) && !F.empty()) {
        const double f = std::sqrt(0.9 * v(c) / ex);
        for (int x : F) B(c, x) *= f;
        ex *= f * f;
      }
      explained[us(c)] += std::max(ex, 0.0);
    }
  }
  for (int q = 0; q < m; ++q) {
    if (inM[us(q)] || shaped[us(q)] || kids[us(q)].empty()) continue;
    // Reference child: a fixed path sets the sign, else the first child is positive.
    int c0 = -1;
    double b0sign = 1.0;
    for (int c : kids[us(q)])
      if (!P.bet(c, q).is_free()) { c0 = c; b0sign = sgn(P.bet(c, q).fixed()); break; }
    if (c0 < 0) c0 = kids[us(q)].front();
    for (int c : kids[us(q)]) {
      if (!P.bet(c, q).is_free()) continue;
      double s = b0sign;
      if (c != c0 && inM[us(c)] && inM[us(c0)]) {
        const double corr = T(c, c0) / std::sqrt(v(c) * v(c0));
        if (std::abs(corr) > 0.05) s = b0sign * sgn(corr);
      }
      B(c, q) = s * std::sqrt(std::max(share(c) * v(c), 1e-12 * v(c)) / v(q));
    }
  }
  for (int c = 0; c < m; ++c)
    for (int x = 0; x < m; ++x)
      if (x != c && !inM[us(x)] && P.bet(c, x).nonzero()) explained[us(c)] += B(c, x) * B(c, x) * v(x);
  // Variances and covariances.
  for (int j = 0; j < m; ++j) {
    if (!P.psi(j, j).is_free()) continue;
    if (inM[us(j)]) Psi(j, j) = std::max(v(j) - explained[us(j)], 0.1 * v(j));
    else Psi(j, j) = npar[us(j)] == 0 ? v(j) : 0.5 * v(j);
  }
  auto std_path = [&](int c, int q) { return B(c, q) * std::sqrt(v(q) / v(c)); };
  std::vector<std::pair<int, int>> residual_covs;
  for (int x = 0; x < m; ++x)
    for (int y = x + 1; y < m; ++y) {
      if (!P.psi(x, y).is_free()) continue;
      const bool exo = npar[us(x)] == 0 && npar[us(y)] == 0;
      if (inM[us(x)] && inM[us(y)]) {
        if (exo) Psi(x, y) = Psi(y, x) = T(x, y);
        else residual_covs.emplace_back(x, y);
      } else if (!inM[us(x)] && !inM[us(y)] && exo) {
        double num = 0.0, den = 0.0;
        for (int c1 : kids[us(x)])
          for (int c2 : kids[us(y)]) {
            if (c1 == c2 || !inM[us(c1)] || !inM[us(c2)]) continue;
            const double w = std_path(c1, x) * std_path(c2, y);
            num += w * T(c1, c2) / std::sqrt(v(c1) * v(c2));
            den += w * w;
          }
        const double rho = den > 1e-12 ? clamp_to(num / den, -0.95, 0.95) : 0.0;
        Psi(x, y) = Psi(y, x) = rho * std::sqrt(v(x) * v(y));
      }
    }
  // Keep (I - B) invertible.
  MatrixXd A, Phi;
  for (int shrink = 0; shrink < 20 && !latent_moments(B, Psi, A, Phi); ++shrink) {
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < m; ++c) if (P.bet(r, c).is_free()) B(r, c) *= 0.5;
  }
  if (!latent_moments(B, Psi, A, Phi)) return out;
  for (const auto& [x, y] : residual_covs) {
    double val = T(x, y) - Phi(x, y);
    const double lim = 0.9 * std::sqrt(std::max(Psi(x, x), 0.0) * std::max(Psi(y, y), 0.0));
    val = clamp_to(val, -lim, lim);
    Psi(x, y) = Psi(y, x) = std::isfinite(val) ? val : 0.0;
  }

  SubProblem sp;
  sp.P = &P; sp.W = &W;
  {
    Eigen::LLT<MatrixXd> llt(W.Phi);
    if (llt.info() == Eigen::Success)
      sp.Linv = llt.matrixL().solve(MatrixXd::Identity(nM, nM));
    else
      sp.Linv = W.Phi.diagonal().cwiseSqrt().cwiseInverse().asDiagonal();
  }
  sp.B0 = B; sp.Psi0 = Psi; sp.a0 = W.a;
  std::vector<double> x0;
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < m; ++c)
      if (r != c && P.bet(r, c).is_free()) { sp.params.push_back({'B', r, c}); x0.push_back(B(r, c)); }
  for (int r = 0; r < m; ++r)
    for (int c = r; c < m; ++c)
      if (P.psi(r, c).is_free()) { sp.params.push_back({'P', r, c}); x0.push_back(Psi(r, c)); }
  for (int j : W.M)
    if (W.estimated[us(j)] && !W.determined[us(j)]) {
      sp.params.push_back({'A', j, j});
      x0.push_back(std::log(std::abs(W.a(j)) > 1e-300 ? std::abs(W.a(j)) : 1.0));
      sp.a0(j) = std::abs(W.a(j));
    }
  VectorXd x = Eigen::Map<const VectorXd>(x0.data(), to_int(x0.size()));
  // The initial point defines the proper region's scale.
  sp.var_floor = 1e-2 * Phi.diagonal().cwiseMax(0.0);
  sp.path_bound = VectorXd::Constant(to_int(sp.params.size()), 2.0);
  for (std::size_t q = 0; q < sp.params.size(); ++q) {
    const auto& s = sp.params[q];
    if (s.kind != 'B' || !(Phi(s.r, s.r) > 0.0)) continue;
    const double std0 = std::abs(B(s.r, s.c)) * std::sqrt(std::max(Phi(s.c, s.c), 0.0) / Phi(s.r, s.r));
    sp.path_bound(to_int(q)) = std::max(2.0, 1.5 * std0);
  }
  {
    VectorXd r0;
    if (!residuals(sp, x, r0, nullptr)) { sp.var_floor.resize(0); sp.path_bound.resize(0); }
  }
  VectorXd xf = levenberg_marquardt(sp, x, out.cost0, out.cost);
  if (!(out.cost <= out.cost0) || !xf.allFinite()) xf = x;
  VectorXd a;
  unpack(sp, xf, out.B, out.Psi, a);
  // Bound the start inside proper values.
  VectorXd vfin = v;
  if (latent_moments(out.B, out.Psi, A, Phi))
    for (int j = 0; j < m; ++j) if (Phi(j, j) > 0.0) vfin(j) = Phi(j, j);
  for (int j = 0; j < m; ++j)
    if (P.psi(j, j).is_free()) out.Psi(j, j) = std::max(out.Psi(j, j), 1e-3 * vfin(j));
  for (int r = 0; r < m; ++r)
    for (int c = r + 1; c < m; ++c) {
      if (!P.psi(r, c).is_free() || !(out.Psi(r, r) > 0.0) || !(out.Psi(c, c) > 0.0)) continue;
      const double lim = 0.99 * std::sqrt(out.Psi(r, r) * out.Psi(c, c));
      out.Psi(r, c) = out.Psi(c, r) = clamp_to(out.Psi(r, c), -lim, lim);
    }
  out.a = a;
  out.v = vfin;
  out.ok = out.B.allFinite() && out.Psi.allFinite() && out.a.allFinite();
  return out;
}

}  // namespace

fit_expected<LayeredStartReport>
layered_start_report(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                     const data::SampleStats& samp, const spec::Starts& starts) {
  LayeredStartReport report;
  auto base = fabin_start_values(pt, rep, samp, starts, FabinVariant::Fabin3);
  if (!base) return std::unexpected(base.error());
  report.theta = std::move(*base);
  const int nf = pt.n_free();
  if (nf == 0) return report;
  if (pt.n_levels() > 1 || pt.composite_mode == spec::CompositeMode::FcSem ||
      !pt.composite_blocks.empty()) {
    report.notes.push_back("layered: two-level and native composite models are outside the "
                           "domain; FABIN3 starts returned");
    return report;
  }
  if (samp.S.size() != rep.dims.size())
    return std::unexpected(FitError{FitError::Kind::InvalidStartValues,
        "layered_start_values: sample block count does not match the model", 0, 0.0});

  spec::LatentStructure ptr = pt;
  (void)resolve_fixed_x_from_sample(ptr, rep, samp);
  const Layout lay = build_layout(ptr, rep);
  const std::size_t nb = lay.blocks.size();
  std::vector<Work> works(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const Pattern& P = lay.blocks[b];
    Work& W = works[b];
    W.S = samp.S[b];
    if (W.S.rows() != P.p || W.S.cols() != P.p) {
      W.ok = false;
      report.notes.push_back("block " + std::to_string(b + 1) + ": sample covariance size mismatch");
      continue;
    }
    W.ok = measurement(W, P, report.notes, to_int(b));
    if (!W.ok) {
      report.notes.push_back("block " + std::to_string(b + 1) +
                             ": measurement step failed; FABIN3 starts kept");
    } else if (W.rank_deficient) {
      // The latent covariance is identified only through the structural part
      // (e.g. trait-state models); the moment layers cannot separate it.
      W.ok = false;
      report.notes.push_back("block " + std::to_string(b + 1) +
                             ": measurement layer does not identify the latent covariance; "
                             "FABIN3 starts kept");
    }
  }
  solve_scales(works, lay.blocks, ptr, lay);

  Eigen::VectorXd theta = report.theta;
  auto put_value = [&theta](const Slot& s, double value) {
    if (s.is_free() && std::isfinite(value)) theta(s.free) = value;
  };
  for (std::size_t b = 0; b < nb; ++b) {
    const Pattern& P = lay.blocks[b];
    Work& W = works[b];
    if (!W.ok) continue;
    const StructuralResult sr = structural_fit(P, W);
    W.v = VectorXd::Ones(P.m);
    if (!sr.ok) {
      report.notes.push_back("block " + std::to_string(b + 1) +
                             ": structural step failed; FABIN3 path and variance starts kept");
    } else {
      report.structural_cost_initial += sr.cost0;
      report.structural_cost_final += sr.cost;
      W.a = sr.a;
      W.v = sr.v;
      for (int r = 0; r < P.m; ++r)
        for (int c = 0; c < P.m; ++c) {
          if (r != c) put_value(P.bet(r, c), sr.B(r, c));
          if (c >= r) put_value(P.psi(r, c), sr.Psi(r, c));
        }
    }
    for (int i = 0; i < P.p; ++i) {
      for (int j = 0; j < P.m; ++j)
        if (W.estimated[us(j)] && std::abs(W.a(j)) > 1e-300)
          put_value(P.lam(i, j), W.sd(i) * W.Lw(i, j) / W.a(j));
      put_value(P.the(i, i), W.S(i, i) * W.theta_std(i));
    }
    const int nM = to_int(W.M.size());
    for (int i = 0; i < P.p; ++i)
      for (int k = i + 1; k < P.p; ++k) {
        if (!P.the(i, k).is_free()) continue;
        VectorXd li = VectorXd::Zero(nM), lk = VectorXd::Zero(nM);
        for (int jj = 0; jj < nM; ++jj) { li(jj) = W.Lw(i, W.M[us(jj)]); lk(jj) = W.Lw(k, W.M[us(jj)]); }
        double rr = W.R(i, k) - li.dot(W.Phi * lk);
        const double lim = 0.9 * std::sqrt(W.theta_std(i) * W.theta_std(k));
        rr = std::isfinite(rr) ? clamp_to(rr, -lim, lim) : 0.0;
        put_value(P.the(i, k), W.sd(i) * W.sd(k) * rr);
      }
  }

  // Unit of each free parameter, for the scale-equivariant constraint projection.
  VectorXd unit = VectorXd::Ones(nf);
  VectorXd group_n = VectorXd::Ones(nf);
  for (int k = 0; k < nf; ++k) {
    const Loc& L = lay.loc[us(k)];
    if (L.block < 0 || !works[us(L.block)].ok) continue;
    const Work& W = works[us(L.block)];
    auto lat = [&](int j) {
      const double vj = W.v.size() > j ? W.v(j) : 1.0;
      return vj > 0.0 && std::isfinite(vj) ? std::sqrt(vj) : 1.0;
    };
    double u = 1.0;
    switch (L.mat) {
      case MatId::Lambda: u = W.sd(L.row) / lat(L.col); break;
      case MatId::Theta: u = W.sd(L.row) * W.sd(L.col); break;
      case MatId::Psi: u = lat(L.row) * lat(L.col); break;
      case MatId::Beta: u = lat(L.row) / lat(L.col); break;
      case MatId::Nu: u = W.sd(L.row); break;
      case MatId::Alpha: u = lat(L.row); break;
    }
    unit(k) = (u > 0.0 && std::isfinite(u)) ? u : 1.0;
    if (us(L.block) < samp.n_obs.size() && samp.n_obs[us(L.block)] > 0)
      group_n(k) = static_cast<double>(samp.n_obs[us(L.block)]);
  }
  auto con = build_eq_constraints(ptr, /*allow_nonlinear=*/true);
  auto project = [&](Eigen::VectorXd& t) {
    if (!con || !con->active()) return;
    const MatrixXd& K = con->Kmat;
    const VectorXd w = group_n.cwiseQuotient(unit.cwiseProduct(unit));
    const MatrixXd KW = K.transpose() * w.asDiagonal();
    Eigen::LDLT<MatrixXd> ldlt(KW * K);
    if (ldlt.info() != Eigen::Success) return;
    const VectorXd alpha = ldlt.solve(KW * (t - con->theta0));
    const VectorXd projected = con->theta0 + K * alpha;
    if (projected.allFinite()) t = projected;
  };
  if (!con) report.notes.push_back("layered: constraints not reducible; no projection");
  project(theta);

  // Means: linear least squares in the reduced constraint coordinates.
  std::vector<int> mean_params;
  for (int k = 0; k < nf; ++k) {
    const Loc& L = lay.loc[us(k)];
    if (L.block >= 0 && (L.mat == MatId::Nu || L.mat == MatId::Alpha)) mean_params.push_back(k);
  }
  bool means_ok = !mean_params.empty() && samp.mean.size() == nb;
  for (std::size_t b = 0; b < nb && means_ok; ++b)
    if (samp.mean[b].size() != lay.blocks[b].p || !works[b].ok) means_ok = false;
  if (!mean_params.empty() && !means_ok)
    report.notes.push_back("layered: no usable sample means; FABIN3 mean starts kept");
  if (means_ok) {
    auto mean_ev = model::ModelEvaluator::build(ptr, rep);
    auto evaluation = mean_ev ? mean_ev->evaluate(theta, false, true)
                              : model_expected<model::Evaluation>(std::unexpected(mean_ev.error()));
    if (evaluation && evaluation->J_mu.rows() > 0) {
      const MatrixXd& Jm = evaluation->J_mu;
      int ptot = 0;
      for (std::size_t b = 0; b < nb; ++b) ptot += lay.blocks[b].p;
      VectorXd ybar(ptot), mu(ptot), wrow(ptot);
      int off = 0;
      for (std::size_t b = 0; b < nb; ++b) {
        const int pb = lay.blocks[b].p;
        ybar.segment(off, pb) = samp.mean[b];
        mu.segment(off, pb) = evaluation->moments.mu.size() > b && evaluation->moments.mu[b].size() == pb
                                  ? evaluation->moments.mu[b] : VectorXd(VectorXd::Zero(pb));
        wrow.segment(off, pb) = works[b].sd.cwiseInverse();
        off += pb;
      }
      if (Jm.rows() == ptot) {
        const int nm = to_int(mean_params.size());
        MatrixXd Jsub(ptot, nm);
        VectorXd tm(nm);
        std::vector<char> is_mean(us(nf), 0);
        for (int q = 0; q < nm; ++q) {
          Jsub.col(q) = Jm.col(mean_params[us(q)]);
          tm(q) = theta(mean_params[us(q)]);
          is_mean[us(mean_params[us(q)])] = 1;
        }
        const VectorXd mu0 = mu - Jsub * tm;
        MatrixXd G;
        VectorXd base_m, rhs;
        MatrixXd Kmc;
        if (con && con->active()) {
          const MatrixXd& K = con->Kmat;
          std::vector<int> cols;
          for (Index c = 0; c < K.cols(); ++c) {
            bool touches_mean = false, touches_other = false;
            for (int k = 0; k < nf; ++k) {
              if (K(k, c) == 0.0) continue;
              (is_mean[us(k)] ? touches_mean : touches_other) = true;
            }
            if (touches_mean && !touches_other) cols.push_back(static_cast<int>(c));
          }
          const VectorXd alpha = con->contract(theta);
          VectorXd alpha_other = alpha;
          for (int c : cols) alpha_other(c) = 0.0;
          const VectorXd full_other = con->theta0 + K * alpha_other;
          Kmc.resize(nm, to_int(cols.size()));
          base_m.resize(nm);
          for (int q = 0; q < nm; ++q) {
            base_m(q) = full_other(mean_params[us(q)]);
            for (std::size_t c = 0; c < cols.size(); ++c) Kmc(q, to_int(c)) = K(mean_params[us(q)], cols[c]);
          }
        } else {
          Kmc = MatrixXd::Identity(nm, nm);
          base_m = VectorXd::Zero(nm);
        }
        G = wrow.asDiagonal() * (Jsub * Kmc);
        rhs = wrow.asDiagonal() * (ybar - mu0 - Jsub * base_m);
        if (G.cols() > 0) {
          Eigen::CompleteOrthogonalDecomposition<MatrixXd> cod(G);
          const VectorXd sol = cod.solve(rhs);
          const VectorXd tnew = base_m + Kmc * sol;
          if (tnew.allFinite())
            for (int q = 0; q < nm; ++q) theta(mean_params[us(q)]) = tnew(q);
        }
      }
    }
  }
  project(theta);

  for (int k = 0; k < nf; ++k)
    if (us(k) < starts.hint.size() && std::isfinite(starts.hint[us(k)])) theta(k) = starts.hint[us(k)];

  // Feasibility: inflate free variances until every block's Σ is PD.
  auto ev = model::ModelEvaluator::build(ptr, rep);
  if (ev) {
    auto pd = [&](const Eigen::VectorXd& t) {
      auto mom = ev->sigma(t);
      if (!mom) return false;
      for (const auto& s : mom->sigma) {
        Eigen::LLT<MatrixXd> llt(s);
        if (llt.info() != Eigen::Success || !s.allFinite()) return false;
      }
      return true;
    };
    if (!pd(theta)) {
      std::vector<int> diag;
      for (int k = 0; k < nf; ++k) {
        const Loc& L = lay.loc[us(k)];
        if (L.block < 0 || L.row != L.col || (L.mat != MatId::Theta && L.mat != MatId::Psi)) continue;
        if (us(k) < starts.hint.size() && std::isfinite(starts.hint[us(k)])) continue;
        diag.push_back(k);
      }
      bool fixed_up = false;
      for (double t : {0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0}) {
        Eigen::VectorXd trial = theta;
        for (int k : diag) trial(k) = std::abs(theta(k)) * (1.0 + t) + 1e-8 * unit(k) * unit(k);
        if (pd(trial)) { theta = trial; fixed_up = true; break; }
      }
      report.covariance_repaired = fixed_up;
      report.notes.push_back(fixed_up ? "layered: free variances inflated to reach a PD implied covariance"
                                      : "layered: implied covariance not PD at the start");
    }
  }
  if (!theta.allFinite()) {
    report.notes.push_back("layered: non-finite start; FABIN3 starts returned");
    return report;
  }
  report.theta = std::move(theta);
  return report;
}

fit_expected<Eigen::VectorXd>
layered_start_values(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                     const data::SampleStats& samp, const spec::Starts& starts) {
  auto out = layered_start_report(pt, rep, samp, starts);
  if (!out) return std::unexpected(out.error());
  return std::move(out->theta);
}

}  // namespace magmaan::estimate
