#include "magmaan/estimate/frontier/newton_accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::estimate {

std::string_view to_string(NewtonAccuracyStatus s) noexcept {
  switch (s) {
    case NewtonAccuracyStatus::Available: return "available";
    case NewtonAccuracyStatus::Unavailable: return "unavailable";
    case NewtonAccuracyStatus::Unsupported: return "unsupported";
    case NewtonAccuracyStatus::NonpositiveCurvature: return "nonpositive_curvature";
    case NewtonAccuracyStatus::IllConditioned: return "ill_conditioned";
    case NewtonAccuracyStatus::SolveUnreliable: return "solve_unreliable";
  }
  return "unavailable";
}

}  // namespace magmaan::estimate

namespace magmaan::estimate::frontier {

NewtonAccuracyDiagnostics
newton_accuracy_from(const Eigen::VectorXd& G, const Eigen::MatrixXd& I,
                     NewtonAccuracyOptions opts) {
  NewtonAccuracyDiagnostics a;
  a.checked = true;
  a.budget = opts.budget;
  a.n_reduced = static_cast<std::int32_t>(G.size());
  if (I.rows() != G.size() || I.cols() != G.size() || !G.allFinite() ||
      !I.allFinite()) {
    return a;
  }
  if (G.size() == 0) {
    a.status = NewtonAccuracyStatus::Available;
    a.distance = a.predicted_gain = a.max_step = a.solve_residual = 0.0;
    a.condition = 1.0;
    a.passed = true;
    return a;
  }
  if ((I.diagonal().array() <= 0.0).any()) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  // Diagonal equilibration makes the conditioning guard insensitive to
  // diagonal changes of units. No inverse is formed.
  const Eigen::VectorXd scale = I.diagonal().array().sqrt().inverse();
  Eigen::MatrixXd C = scale.asDiagonal() * I * scale.asDiagonal();
  C = (0.5 * (C + C.transpose())).eval();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C, Eigen::EigenvaluesOnly);
  if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <= 0.0) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  a.condition = eig.eigenvalues().maxCoeff() / eig.eigenvalues().minCoeff();
  if (a.condition > opts.max_condition) {
    a.status = NewtonAccuracyStatus::IllConditioned;
    return a;
  }
  Eigen::LLT<Eigen::MatrixXd> chol(C);
  if (chol.info() != Eigen::Success) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  const Eigen::VectorXd b = scale.asDiagonal() * G;
  const Eigen::VectorXd y = chol.solve(b);
  a.solve_residual =
      (C * y - b).norm() / (C.norm() * y.norm() + b.norm() + 1e-300);
  if (a.solve_residual > opts.max_solve_residual) {
    a.status = NewtonAccuracyStatus::SolveUnreliable;
    return a;
  }
  const double d2 = b.dot(y);
  if (!(d2 >= 0.0)) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  a.status = NewtonAccuracyStatus::Available;
  a.distance = std::sqrt(d2);
  a.predicted_gain = 0.5 * d2;
  a.max_step = (scale.asDiagonal() * y).cwiseAbs().maxCoeff();
  a.passed = a.distance <= opts.budget;
  return a;
}

namespace {

bool covariance_blocks_interior(const model::ModelEvaluator& ev,
                                const Eigen::VectorXd& theta, double tol) {
  auto mat = ev.assembled(theta);
  if (!mat) return false;
  for (const auto& b : mat->blocks) {
    for (const Eigen::MatrixXd* M : {&b.Psi, &b.Theta}) {
      if (M->size() == 0) continue;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(*M, Eigen::EigenvaluesOnly);
      if (es.info() != Eigen::Success) return false;
      const double top = es.eigenvalues().cwiseAbs().maxCoeff();
      if (es.eigenvalues().minCoeff() <= tol * top) return false;
    }
  }
  return true;
}

}  // namespace

NewtonAccuracyDiagnostics
newton_accuracy_ml(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                   const SampleStats& samp, const Estimates& est,
                   NewtonAccuracyOptions opts) {
  NewtonAccuracyDiagnostics fail;
  fail.checked = true;
  fail.budget = opts.budget;
  if (build_nl_constraints(pt).active()) {
    fail.status = NewtonAccuracyStatus::Unsupported;
    return fail;
  }
  auto con = build_eq_constraints(pt);
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!con || !ev) return fail;
  auto obj = ml_objective(*ev, samp);
  if (!obj) return fail;
  Eigen::VectorXd gradient;
  const double f = obj->f(est.theta, gradient);
  if (!std::isfinite(f) || !gradient.allFinite()) return fail;
  auto info = inference::information_observed_analytic(pt, rep, samp, est);
  if (!info) return fail;
  const double n = std::accumulate(samp.n_obs.begin(), samp.n_obs.end(), 0.0);
  const Eigen::VectorXd G = n * con->reduce_gradient(gradient);
  const Eigen::MatrixXd I = con->K().transpose() * (*info) * con->K();
  NewtonAccuracyDiagnostics a = newton_accuracy_from(G, I, opts);
  a.covariance_interior =
      covariance_blocks_interior(*ev, est.theta, opts.interior_eigen_tol);
  return a;
}

}  // namespace magmaan::estimate::frontier

namespace magmaan::estimate::frontier {

namespace {

// ⟨N, dC/dθ_k⟩_F for every free θ_k that lands in block `block` of `mat`.
Eigen::VectorXd block_adjoint(const Eigen::MatrixXd& N, model::MatId mat,
                              std::int32_t block,
                              const spec::LatentStructure& pt,
                              const model::MatrixRep& rep, Eigen::Index npar) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(npar);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    const model::Cell& cell = rep.cell_for_row[i];
    if (!cell.used || cell.mat != mat || cell.block != block) continue;
    const Eigen::Index k = pt.free[i] - 1;
    if (k < 0 || k >= npar || cell.row < 0 || cell.col < 0 ||
        cell.row >= N.rows() || cell.col >= N.cols()) continue;
    out(k) += cell.row == cell.col ? N(cell.row, cell.col)
                                   : 2.0 * N(cell.row, cell.col);
  }
  return out;
}

// Frobenius-orthonormal basis element of the symmetric q x q matrices.
Eigen::MatrixXd sym_basis(Eigen::Index q, Eigen::Index i, Eigen::Index j) {
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(q, q);
  if (i == j) {
    E(i, i) = 1.0;
  } else {
    E(i, j) = E(j, i) = 1.0 / std::sqrt(2.0);
  }
  return E;
}

struct FaceComponent {
  model::MatId mat = model::MatId::Psi;
  std::int32_t block = 0;
  Eigen::Index dim = 0;               // full block dimension
  std::vector<Eigen::Index> idx;      // component rows within the block
  Eigen::MatrixXd U;                  // |idx| x q numerical null basis
  Eigen::MatrixXd P;                  // pseudo-inverse on the range
  Eigen::MatrixXd Uc;                 // held null directions
  Eigen::VectorXd omega_c;            // their multipliers
};

// Embed a component-local matrix into its full block.
Eigen::MatrixXd embed(const FaceComponent& f, const Eigen::MatrixXd& local) {
  Eigen::MatrixXd full = Eigen::MatrixXd::Zero(f.dim, f.dim);
  for (std::size_t a = 0; a < f.idx.size(); ++a)
    for (std::size_t b = 0; b < f.idx.size(); ++b)
      full(f.idx[a], f.idx[b]) =
          local(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b));
  return full;
}

// Rows d/dθ of the basis components of W' C W, W a component-local basis.
void append_face_rows(std::vector<Eigen::VectorXd>& rows,
                      const FaceComponent& f, const Eigen::MatrixXd& W,
                      const spec::LatentStructure& pt,
                      const model::MatrixRep& rep, Eigen::Index npar) {
  const Eigen::Index q = W.cols();
  for (Eigen::Index j = 0; j < q; ++j)
    for (Eigen::Index i = 0; i <= j; ++i) {
      const Eigen::MatrixXd N = W * sym_basis(q, i, j) * W.transpose();
      rows.push_back(block_adjoint(embed(f, N), f.mat, f.block, pt, rep, npar));
    }
}

Eigen::MatrixXd stack_rows(const std::vector<Eigen::VectorXd>& rows,
                           Eigen::Index npar) {
  Eigen::MatrixXd A(static_cast<Eigen::Index>(rows.size()), npar);
  for (std::size_t r = 0; r < rows.size(); ++r)
    A.row(static_cast<Eigen::Index>(r)) = rows[r].transpose();
  return A;
}

}  // namespace

NewtonAccuracyDiagnostics
newton_accuracy_ml_psd(const spec::LatentStructure& pt,
                       const model::MatrixRep& rep, const SampleStats& samp,
                       const Estimates& est, NewtonAccuracyOptions opts) {
  NewtonAccuracyDiagnostics fail;
  fail.checked = true;
  fail.psd_domain = true;
  fail.budget = opts.budget;
  if (build_nl_constraints(pt).active()) {
    fail.status = NewtonAccuracyStatus::Unsupported;
    return fail;
  }
  auto con = build_eq_constraints(pt);
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!con || !ev) return fail;
  auto obj = ml_objective(*ev, samp);
  if (!obj) return fail;
  Eigen::VectorXd gradient;
  const double f = obj->f(est.theta, gradient);
  if (!std::isfinite(f) || !gradient.allFinite()) return fail;
  auto info = inference::information_observed_analytic(pt, rep, samp, est);
  auto mats = ev->assembled(est.theta);
  if (!info || !mats) return fail;
  const Eigen::Index npar = est.theta.size();
  const double n = std::accumulate(samp.n_obs.begin(), samp.n_obs.end(), 0.0);
  const Eigen::MatrixXd K = con->K();
  const Eigen::VectorXd G = n * con->reduce_gradient(gradient);
  const Eigen::MatrixXd I = K.transpose() * (*info) * K;

  // Singular components of every primitive block.
  std::vector<FaceComponent> faces;
  std::vector<Eigen::VectorXd> forced;  // fixed-zero-variance rows: dC_rj = 0
  bool infeasible = false;
  for (std::size_t b = 0; b < mats->blocks.size(); ++b) {
    for (const model::MatId mat : {model::MatId::Theta, model::MatId::Psi}) {
      const Eigen::MatrixXd& raw = mat == model::MatId::Theta
          ? mats->blocks[b].Theta : mats->blocks[b].Psi;
      if (raw.size() == 0) continue;
      const Eigen::MatrixXd C = 0.5 * (raw + raw.transpose());
      const Eigen::Index dim = C.rows();
      const auto block = static_cast<std::int32_t>(b);
      const double tol = opts.interior_eigen_tol *
          std::max(1.0, C.cwiseAbs().maxCoeff());
      // Structural pattern: free cells and fixed nonzero entries.
      Eigen::MatrixXi freecell = Eigen::MatrixXi::Zero(dim, dim);
      for (std::size_t i = 0; i < pt.size(); ++i) {
        if (pt.free[i] <= 0) continue;
        const model::Cell& cell = rep.cell_for_row[i];
        if (!cell.used || cell.mat != mat || cell.block != block) continue;
        freecell(cell.row, cell.col) = freecell(cell.col, cell.row) = 1;
      }
      std::vector<bool> active(static_cast<std::size_t>(dim), false);
      for (Eigen::Index r = 0; r < dim; ++r) {
        const bool zero_variance = freecell(r, r) == 0 && C(r, r) == 0.0;
        if (!zero_variance) {
          active[static_cast<std::size_t>(r)] = true;
          continue;
        }
        // A fixed zero variance forces its row to zero: hold free cells there.
        for (Eigen::Index c = 0; c < dim; ++c) {
          if (c == r || freecell(r, c) == 0) continue;
          Eigen::MatrixXd N = Eigen::MatrixXd::Zero(dim, dim);
          N(r, c) = N(c, r) = 0.5;
          forced.push_back(block_adjoint(N, mat, block, pt, rep, npar));
        }
      }
      // Connected components over structurally nonzero off-diagonals.
      std::vector<Eigen::Index> parent(static_cast<std::size_t>(dim));
      std::iota(parent.begin(), parent.end(), Eigen::Index{0});
      auto find = [&parent](Eigen::Index x) {
        while (parent[static_cast<std::size_t>(x)] != x)
          x = parent[static_cast<std::size_t>(x)] =
              parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
        return x;
      };
      for (Eigen::Index r = 0; r < dim; ++r)
        for (Eigen::Index c = 0; c < r; ++c)
          if (active[static_cast<std::size_t>(r)] &&
              active[static_cast<std::size_t>(c)] &&
              (freecell(r, c) != 0 || C(r, c) != 0.0))
            parent[static_cast<std::size_t>(find(r))] = find(c);
      std::vector<std::vector<Eigen::Index>> groups(static_cast<std::size_t>(dim));
      for (Eigen::Index r = 0; r < dim; ++r)
        if (active[static_cast<std::size_t>(r)])
          groups[static_cast<std::size_t>(find(r))].push_back(r);
      for (auto& idx : groups) {
        if (idx.empty()) continue;
        const auto m = static_cast<Eigen::Index>(idx.size());
        Eigen::MatrixXd Cs(m, m);
        for (Eigen::Index a = 0; a < m; ++a)
          for (Eigen::Index c = 0; c < m; ++c)
            Cs(a, c) = C(idx[static_cast<std::size_t>(a)], idx[static_cast<std::size_t>(c)]);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Cs);
        if (es.info() != Eigen::Success) return fail;
        const Eigen::VectorXd& lam = es.eigenvalues();
        if (lam.minCoeff() < -tol) infeasible = true;
        std::vector<Eigen::Index> nul, rng;
        for (Eigen::Index j = 0; j < m; ++j) (lam(j) <= tol ? nul : rng).push_back(j);
        if (nul.empty()) continue;
        FaceComponent fc;
        fc.mat = mat;
        fc.block = block;
        fc.dim = dim;
        fc.idx = idx;
        fc.U.resize(m, static_cast<Eigen::Index>(nul.size()));
        for (std::size_t j = 0; j < nul.size(); ++j)
          fc.U.col(static_cast<Eigen::Index>(j)) = es.eigenvectors().col(nul[j]);
        fc.P = Eigen::MatrixXd::Zero(m, m);
        for (const Eigen::Index j : rng)
          fc.P += es.eigenvectors().col(j) * es.eigenvectors().col(j).transpose() / lam(j);
        faces.push_back(std::move(fc));
      }
    }
  }
  if (infeasible) return fail;

  NewtonAccuracyDiagnostics out;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(npar, npar);
  std::vector<Eigen::VectorXd> held = forced;
  std::int32_t null_dirs = 0, held_dirs = 0;
  double min_multiplier = std::numeric_limits<double>::quiet_NaN();
  if (!faces.empty()) {
    // Multipliers: least squares of the reduced gradient on all face rows.
    std::vector<Eigen::VectorXd> rows = forced;
    std::vector<std::size_t> start;
    for (const auto& fc : faces) {
      start.push_back(rows.size());
      append_face_rows(rows, fc, fc.U, pt, rep, npar);
    }
    const Eigen::MatrixXd A = stack_rows(rows, npar) * K;
    const Eigen::VectorXd lambda =
        A.transpose().completeOrthogonalDecomposition().solve(G);
    std::vector<Eigen::VectorXd> omegas;
    double scale = 0.0;
    for (std::size_t c = 0; c < faces.size(); ++c) {
      auto& fc = faces[c];
      const Eigen::Index q = fc.U.cols();
      Eigen::MatrixXd L = Eigen::MatrixXd::Zero(q, q);
      std::size_t r = start[c];
      for (Eigen::Index j = 0; j < q; ++j)
        for (Eigen::Index i = 0; i <= j; ++i)
          L += lambda(static_cast<Eigen::Index>(r++)) * sym_basis(q, i, j);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
      if (es.info() != Eigen::Success) return fail;
      omegas.push_back(es.eigenvalues());
      scale = std::max(scale, es.eigenvalues().cwiseAbs().maxCoeff());
      fc.Uc = fc.U * es.eigenvectors();  // rotate into multiplier eigenbasis
    }
    const double omega_tol = 1e-10 * std::max(scale, 1e-300);
    for (std::size_t c = 0; c < faces.size(); ++c) {
      auto& fc = faces[c];
      const Eigen::VectorXd& w = omegas[c];
      null_dirs += static_cast<std::int32_t>(w.size());
      min_multiplier = std::isnan(min_multiplier) ? w.minCoeff()
                                                  : std::min(min_multiplier, w.minCoeff());
      std::vector<Eigen::Index> keep;
      for (Eigen::Index j = 0; j < w.size(); ++j)
        if (w(j) > omega_tol) keep.push_back(j);
      Eigen::MatrixXd Uc(fc.Uc.rows(), static_cast<Eigen::Index>(keep.size()));
      Eigen::VectorXd wc(static_cast<Eigen::Index>(keep.size()));
      for (std::size_t j = 0; j < keep.size(); ++j) {
        Uc.col(static_cast<Eigen::Index>(j)) = fc.Uc.col(keep[j]);
        wc(static_cast<Eigen::Index>(j)) = w(keep[j]);
      }
      fc.Uc = Uc;
      fc.omega_c = wc;
      held_dirs += static_cast<std::int32_t>(keep.size());
      if (keep.empty()) continue;
      append_face_rows(held, fc, fc.Uc, pt, rep, npar);
      // Curvature of the rank-constrained set: 2 tr(M D_k P D_l).
      const Eigen::MatrixXd M = fc.Uc * fc.omega_c.asDiagonal() * fc.Uc.transpose();
      const auto m = static_cast<Eigen::Index>(fc.idx.size());
      std::vector<std::pair<Eigen::Index, Eigen::MatrixXd>> D;
      for (std::size_t i = 0; i < pt.size(); ++i) {
        if (pt.free[i] <= 0) continue;
        const model::Cell& cell = rep.cell_for_row[i];
        if (!cell.used || cell.mat != fc.mat || cell.block != fc.block) continue;
        const auto a = std::find(fc.idx.begin(), fc.idx.end(), Eigen::Index{cell.row});
        const auto bb = std::find(fc.idx.begin(), fc.idx.end(), Eigen::Index{cell.col});
        if (a == fc.idx.end() || bb == fc.idx.end()) continue;
        const Eigen::Index ra = a - fc.idx.begin(), cb = bb - fc.idx.begin();
        Eigen::MatrixXd Dk = Eigen::MatrixXd::Zero(m, m);
        Dk(ra, cb) = 1.0;
        Dk(cb, ra) = 1.0;
        D.emplace_back(pt.free[i] - 1, Dk);
      }
      for (const auto& [k, Dk] : D) {
        const Eigen::MatrixXd T = M * Dk * fc.P;
        for (const auto& [l, Dl] : D) Q(k, l) += 2.0 * (T * Dl).trace();
      }
    }
  }

  const Eigen::MatrixXd H = I + K.transpose() * Q * K;
  Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(K.cols(), K.cols());
  if (!held.empty()) {
    const Eigen::MatrixXd A = stack_rows(held, npar) * K;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd& sv = svd.singularValues();
    const double cut = sv.size() ? 1e-10 * std::max(sv(0), 1e-300) : 0.0;
    Eigen::Index rank = 0;
    while (rank < sv.size() && sv(rank) > cut) ++rank;
    Z = svd.matrixV().rightCols(K.cols() - rank);
  }
  out = newton_accuracy_from(Z.transpose() * G, Z.transpose() * H * Z, opts);
  out.checked = true;
  out.psd_domain = true;
  out.null_directions = null_dirs;
  out.constrained_directions = held_dirs;
  out.min_multiplier = min_multiplier;
  out.covariance_interior = null_dirs == 0;
  return out;
}

}  // namespace magmaan::estimate::frontier
