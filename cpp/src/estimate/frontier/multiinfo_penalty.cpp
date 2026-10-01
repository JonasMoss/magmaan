#include "magmaan/estimate/frontier/multiinfo_penalty.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace magmaan::estimate::frontier {

namespace {

using model::MatId;

FitError penalty_err(std::string detail) {
  return FitError{FitError::Kind::NumericIssue,
                  "multiinfo penalty: " + std::move(detail), 0, 0.0};
}

using BoolMatrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

struct FreeCells {
  BoolMatrix psi, theta, beta, lambda;
};

FreeCells free_cells(const std::vector<model::ParamLocation>& locations,
                     std::size_t block, Eigen::Index m, Eigen::Index p) {
  FreeCells out{BoolMatrix::Constant(m, m, false),
                BoolMatrix::Constant(p, p, false),
                BoolMatrix::Constant(m, m, false),
                BoolMatrix::Constant(p, m, false)};
  for (const auto& loc : locations) {
    if (loc.row < 0 || loc.col < 0 || static_cast<std::size_t>(loc.block) != block) continue;
    switch (loc.mat) {
      case MatId::Psi:
        out.psi(loc.row, loc.col) = out.psi(loc.col, loc.row) = true;
        break;
      case MatId::Theta:
        out.theta(loc.row, loc.col) = out.theta(loc.col, loc.row) = true;
        break;
      case MatId::Beta:
        out.beta(loc.row, loc.col) = true;
        break;
      case MatId::Lambda:
        out.lambda(loc.row, loc.col) = true;
        break;
      default:
        break;
    }
  }
  return out;
}

// Kahn's algorithm on the directed graph j → i for pattern(i, j).
bool acyclic(const BoolMatrix& pattern) {
  const Eigen::Index m = pattern.rows();
  std::vector<int> indegree(static_cast<std::size_t>(m), 0);
  for (Eigen::Index i = 0; i < m; ++i) {
    if (pattern(i, i)) return false;
    for (Eigen::Index j = 0; j < m; ++j) {
      if (pattern(i, j)) ++indegree[static_cast<std::size_t>(i)];
    }
  }
  std::vector<Eigen::Index> ready;
  for (Eigen::Index i = 0; i < m; ++i) {
    if (indegree[static_cast<std::size_t>(i)] == 0) ready.push_back(i);
  }
  Eigen::Index visited = 0;
  while (!ready.empty()) {
    const Eigen::Index j = ready.back();
    ready.pop_back();
    ++visited;
    for (Eigen::Index i = 0; i < m; ++i) {
      if (pattern(i, j) && --indegree[static_cast<std::size_t>(i)] == 0) {
        ready.push_back(i);
      }
    }
  }
  return visited == m;
}

// Complete-data RAM matrices for one block: E = (I − A)⁻¹, S, C = E S Eᵀ.
struct CompleteData {
  Eigen::MatrixXd E;
  Eigen::MatrixXd S;
  Eigen::MatrixXd C;
};

CompleteData complete_data(const model::BlockMatrices& bm, Eigen::Index m,
                           Eigen::Index p) {
  const Eigen::Index n = m + p;
  CompleteData out;
  out.E = Eigen::MatrixXd::Identity(n, n);
  out.S = Eigen::MatrixXd::Zero(n, n);
  if (m > 0) {
    out.E.topLeftCorner(m, m) = bm.A;
    out.E.bottomLeftCorner(p, m) = bm.LamA;
    out.S.topLeftCorner(m, m) = bm.Psi;
  }
  out.S.bottomRightCorner(p, p) = bm.Theta;
  out.C = out.E * out.S * out.E.transpose();
  out.C = (0.5 * (out.C + out.C.transpose())).eval();
  return out;
}

Eigen::MatrixXd principal(const Eigen::MatrixXd& X,
                          const std::vector<std::int32_t>& keep) {
  const Eigen::Index k = static_cast<Eigen::Index>(keep.size());
  Eigen::MatrixXd out(k, k);
  for (Eigen::Index a = 0; a < k; ++a) {
    for (Eigen::Index c = 0; c < k; ++c) {
      out(a, c) = X(keep[static_cast<std::size_t>(a)],
                    keep[static_cast<std::size_t>(c)]);
    }
  }
  return out;
}

struct BlockGradient {
  Eigen::MatrixXd S;  // ∂P/∂S as a general n × n matrix (symmetric)
  Eigen::MatrixXd A;  // ∂P/∂A (RAM), entry (i, j) = ∂P/∂A_ij
};

// P_b and (optionally) its derivatives with respect to S and A.
//   dP = tr(M dC),  M = pad(C_KK⁻¹) − diag(1/C_ii),
//   dC = E dS Eᵀ + E dA C + C dAᵀ Eᵀ   ⇒   ∂P/∂S = Eᵀ M E, ∂P/∂A = 2 Eᵀ M C.
fit_expected<double> block_penalty(const MultiInfoPenaltyBlock& layout,
                                   const CompleteData& cd,
                                   std::size_t block, BlockGradient* grad) {
  if (layout.keep.empty()) {
    if (grad) {
      const Eigen::Index n = cd.C.rows();
      grad->S = Eigen::MatrixXd::Zero(n, n);
      grad->A = Eigen::MatrixXd::Zero(n, n);
    }
    return 0.0;
  }
  if (!cd.C.allFinite()) {
    return std::unexpected(penalty_err(
        "non-finite complete-data covariance in block " +
        std::to_string(block)));
  }
  const Eigen::MatrixXd C_KK = principal(cd.C, layout.keep);
  const Eigen::VectorXd d = C_KK.diagonal();
  if ((d.array() <= 0.0).any()) {
    return std::unexpected(penalty_err(
        "non-positive complete-data variance in block " +
        std::to_string(block)));
  }
  Eigen::LLT<Eigen::MatrixXd> llt(C_KK);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(penalty_err(
        "complete-data covariance C_KK not positive definite in block " +
        std::to_string(block)));
  }
  const Eigen::MatrixXd L = llt.matrixL();
  if ((L.diagonal().array() <= 0.0).any()) {
    return std::unexpected(penalty_err(
        "complete-data covariance C_KK singular in block " +
        std::to_string(block)));
  }
  const double value = 2.0 * L.diagonal().array().log().sum() -
                       d.array().log().sum();
  if (!std::isfinite(value)) {
    return std::unexpected(penalty_err(
        "non-finite penalty in block " + std::to_string(block)));
  }
  if (grad) {
    const Eigen::Index n = cd.C.rows();
    const Eigen::Index k = C_KK.rows();
    Eigen::MatrixXd inner =
        llt.solve(Eigen::MatrixXd::Identity(k, k));
    inner.diagonal() -= d.cwiseInverse();
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index a = 0; a < k; ++a) {
      for (Eigen::Index c = 0; c < k; ++c) {
        M(layout.keep[static_cast<std::size_t>(a)],
          layout.keep[static_cast<std::size_t>(c)]) = inner(a, c);
      }
    }
    const Eigen::MatrixXd EtM = cd.E.transpose() * M;
    grad->S = EtM * cd.E;
    grad->A = 2.0 * EtM * cd.C;
  }
  return value;
}

// Determinacy block. J is ordered (observed y, genuine latents L), so the
// Cholesky factor of C_JJ carries chol(Σ) in its leading block and chol(V),
// V = Var(η_L | y), in its trailing block.
//   P_b = 2 Σ log diag(chol V) − Σ_{j∈L} log C_jj,
//   dP = tr(M dC),  M = pad(C_JJ⁻¹) − pad(Σ⁻¹) − Σ_{j∈L} e_j e_jᵀ / C_jj.
struct DeterminacyParts {
  Eigen::MatrixXd chol_v;  // trailing Cholesky block, V = chol_v chol_vᵀ
  Eigen::VectorXd d;       // C_jj over L
};

fit_expected<double> block_determinacy(const MultiInfoPenaltyBlock& layout,
                                       const CompleteData& cd,
                                       std::size_t block, BlockGradient* grad,
                                       DeterminacyParts* parts) {
  const Eigen::Index n = cd.C.rows();
  if (layout.keep.empty()) {
    if (grad) {
      grad->S = Eigen::MatrixXd::Zero(n, n);
      grad->A = Eigen::MatrixXd::Zero(n, n);
    }
    return 0.0;
  }
  if (!cd.C.allFinite()) {
    return std::unexpected(penalty_err(
        "non-finite complete-data covariance in block " +
        std::to_string(block)));
  }
  const Eigen::Index m = layout.m;
  const Eigen::Index p = layout.p;
  const Eigen::Index l = static_cast<Eigen::Index>(layout.keep.size());
  std::vector<std::int32_t> order;
  order.reserve(static_cast<std::size_t>(p + l));
  for (Eigen::Index i = 0; i < p; ++i) {
    order.push_back(static_cast<std::int32_t>(m + i));
  }
  for (const std::int32_t j : layout.keep) order.push_back(j);
  const Eigen::MatrixXd C_JJ = principal(cd.C, order);
  const Eigen::VectorXd d = C_JJ.diagonal().tail(l);
  if ((d.array() <= 0.0).any()) {
    return std::unexpected(penalty_err(
        "non-positive latent variance in block " + std::to_string(block)));
  }
  Eigen::LLT<Eigen::MatrixXd> llt(C_JJ);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(penalty_err(
        "latent and observed covariance C_JJ not positive definite in block " +
        std::to_string(block)));
  }
  const Eigen::MatrixXd L = llt.matrixL();
  if ((L.diagonal().array() <= 0.0).any()) {
    return std::unexpected(penalty_err(
        "latent and observed covariance C_JJ singular in block " +
        std::to_string(block)));
  }
  const double value = 2.0 * L.diagonal().tail(l).array().log().sum() -
                       d.array().log().sum();
  if (!std::isfinite(value)) {
    return std::unexpected(penalty_err(
        "non-finite penalty in block " + std::to_string(block)));
  }
  if (parts) {
    parts->chol_v = L.bottomRightCorner(l, l);
    parts->d = d;
  }
  if (grad) {
    Eigen::MatrixXd inner =
        llt.solve(Eigen::MatrixXd::Identity(p + l, p + l));
    const Eigen::MatrixXd chol_sigma_inv =
        L.topLeftCorner(p, p).triangularView<Eigen::Lower>().solve(
            Eigen::MatrixXd::Identity(p, p));
    inner.topLeftCorner(p, p) -= chol_sigma_inv.transpose() * chol_sigma_inv;
    inner.diagonal().tail(l) -= d.cwiseInverse();
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index a = 0; a < p + l; ++a) {
      for (Eigen::Index c = 0; c < p + l; ++c) {
        M(order[static_cast<std::size_t>(a)],
          order[static_cast<std::size_t>(c)]) = inner(a, c);
      }
    }
    const Eigen::MatrixXd EtM = cd.E.transpose() * M;
    grad->S = EtM * cd.E;
    grad->A = 2.0 * EtM * cd.C;
  }
  return value;
}

fit_expected<double> block_value(const MultiInfoPenaltyLayout& layout,
                                 const CompleteData& cd, std::size_t block,
                                 BlockGradient* grad) {
  const auto& blk = layout.blocks[block];
  return layout.target == PenaltyTarget::Determinacy
             ? block_determinacy(blk, cd, block, grad, nullptr)
             : block_penalty(blk, cd, block, grad);
}

}  // namespace

fit_expected<MultiInfoPenaltyLayout>
multiinfo_penalty_layout(const model::ModelEvaluator& ev,
                         const Eigen::VectorXd& theta, PenaltyTarget target) {
  auto assembled = ev.assembled(theta);
  if (!assembled.has_value()) {
    return std::unexpected(penalty_err(
        "cannot assemble model matrices: " + assembled.error().detail));
  }
  const model::MatrixRep& rep = ev.matrix_rep();
  for (const auto& info : rep.block_info) {
    if (info.role != model::BlockLevel::Single) {
      return std::unexpected(penalty_err(
          "two-level models are not supported"));
    }
  }

  MultiInfoPenaltyLayout out;
  out.target = target;
  out.locations = ev.param_locations();
  for (std::size_t b = 0; b < assembled->blocks.size(); ++b) {
    const model::BlockMatrices& bm = assembled->blocks[b];
    const Eigen::Index m = rep.dims[b].n_latent;
    const Eigen::Index p = rep.dims[b].n_observed;
    const FreeCells free = free_cells(out.locations, b, m, p);

    MultiInfoPenaltyBlock blk;
    blk.m = static_cast<std::int16_t>(m);
    blk.p = static_cast<std::int16_t>(p);
    std::vector<bool> in_k(static_cast<std::size_t>(m + p), false);
    for (Eigen::Index i = 0; i < m; ++i) {
      in_k[static_cast<std::size_t>(i)] = free.psi(i, i) || bm.Psi(i, i) != 0.0;
    }
    for (Eigen::Index i = 0; i < p; ++i) {
      in_k[static_cast<std::size_t>(m + i)] =
          free.theta(i, i) || bm.Theta(i, i) != 0.0;
    }
    // A zero residual variance forces a zero residual row; anything else is
    // not a covariance matrix and the recursive barrier identity fails.
    for (Eigen::Index i = 0; i < m; ++i) {
      if (in_k[static_cast<std::size_t>(i)]) continue;
      for (Eigen::Index j = 0; j < m; ++j) {
        if (j != i && (free.psi(i, j) || bm.Psi(i, j) != 0.0)) {
          return std::unexpected(penalty_err(
              "latent " + std::to_string(i) + " in block " +
              std::to_string(b) +
              " has zero residual variance but a nonzero residual covariance"));
        }
      }
    }
    for (Eigen::Index i = 0; i < p; ++i) {
      if (in_k[static_cast<std::size_t>(m + i)]) continue;
      for (Eigen::Index j = 0; j < p; ++j) {
        if (j != i && (free.theta(i, j) || bm.Theta(i, j) != 0.0)) {
          return std::unexpected(penalty_err(
              "observed " + std::to_string(i) + " in block " +
              std::to_string(b) +
              " has zero residual variance but a nonzero residual covariance"));
        }
      }
    }
    if (target == PenaltyTarget::Determinacy) {
      // A latent that an error-free indicator reproduces exactly (phantom
      // ov.y / ov.x slots, zero-error single indicators) is observed data.
      std::vector<bool> exact(static_cast<std::size_t>(m), false);
      for (Eigen::Index i = 0; i < p; ++i) {
        if (free.theta(i, i) || bm.Theta(i, i) != 0.0) continue;
        Eigen::Index count = 0;
        Eigen::Index col = -1;
        for (Eigen::Index j = 0; j < m; ++j) {
          if (free.lambda(i, j) || bm.Lambda(i, j) != 0.0) {
            ++count;
            col = j;
          }
        }
        if (count == 1) {
          exact[static_cast<std::size_t>(col)] = true;
        } else if (count > 1) {
          return std::unexpected(penalty_err(
              "observed " + std::to_string(i) + " in block " +
              std::to_string(b) +
              " is an error-free indicator of several latents: Var(eta | y) "
              "is singular for every parameter value"));
        }
      }
      for (Eigen::Index j = 0; j < m; ++j) {
        if (in_k[static_cast<std::size_t>(j)] &&
            !exact[static_cast<std::size_t>(j)]) {
          blk.keep.push_back(static_cast<std::int32_t>(j));
        }
      }
    } else {
      for (Eigen::Index i = 0; i < m + p; ++i) {
        if (in_k[static_cast<std::size_t>(i)]) {
          blk.keep.push_back(static_cast<std::int32_t>(i));
        }
      }
    }
    if (m > 0) {
      BoolMatrix pattern(m, m);
      for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < m; ++j) {
          pattern(i, j) = free.beta(i, j) || bm.Beta(i, j) != 0.0;
        }
      }
      blk.recursive = acyclic(pattern);
    }
    out.blocks.push_back(std::move(blk));
  }
  return out;
}

namespace {

// One signed log-determinant term, sign * log det C_T, of a block penalty.
struct LogDetTerm {
  double sign = 1.0;
  std::vector<std::int32_t> idx;
};

// Joint: log det C_KK − Σ_K log C_ii. Determinacy: log det C_JJ − log det Σ −
// Σ_L log C_jj with J = (observed y, genuine latents L).
void penalty_terms(PenaltyTarget target, const MultiInfoPenaltyBlock& blk,
                   std::vector<LogDetTerm>& logdets,
                   std::vector<std::int32_t>& diag) {
  logdets.clear();
  diag = blk.keep;
  if (target == PenaltyTarget::Joint) {
    logdets.push_back({1.0, blk.keep});
    return;
  }
  LogDetTerm joint{1.0, {}};
  LogDetTerm observed{-1.0, {}};
  for (Eigen::Index i = 0; i < blk.p; ++i) {
    const auto v = static_cast<std::int32_t>(blk.m + i);
    joint.idx.push_back(v);
    observed.idx.push_back(v);
  }
  for (const std::int32_t j : blk.keep) joint.idx.push_back(j);
  logdets.push_back(std::move(joint));
  logdets.push_back(std::move(observed));
}

// A free parameter as a cell of the RAM matrices: S (symmetric) or A.
struct RamCell {
  bool in_s = true;
  Eigen::Index r = 0;
  Eigen::Index c = 0;
};

}  // namespace

fit_expected<Eigen::MatrixXd>
multiinfo_penalty_hessian(const MultiInfoPenaltyLayout& layout,
                          const model::ModelEvaluator& ev,
                          const Eigen::VectorXd& theta) {
  auto assembled = ev.assembled(theta);
  if (!assembled.has_value()) {
    return std::unexpected(penalty_err(
        "cannot assemble model matrices: " + assembled.error().detail));
  }
  if (assembled->blocks.size() != layout.blocks.size() ||
      layout.locations.size() != static_cast<std::size_t>(theta.size())) {
    return std::unexpected(penalty_err("layout/evaluator mismatch"));
  }
  const Eigen::Index q = theta.size();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(q, q);
  std::vector<LogDetTerm> logdets;
  std::vector<std::int32_t> diag;
  for (std::size_t b = 0; b < layout.blocks.size(); ++b) {
    const auto& blk = layout.blocks[b];
    if (blk.keep.empty()) continue;
    const Eigen::Index m = blk.m;
    const CompleteData cd = complete_data(assembled->blocks[b], m, blk.p);
    const Eigen::Index n = cd.C.rows();
    if (!cd.C.allFinite()) {
      return std::unexpected(penalty_err(
          "non-finite complete-data covariance in block " + std::to_string(b)));
    }
    penalty_terms(layout.target, blk, logdets, diag);

    // dP = tr(M dC); the inverses G_t of every log-det term.
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    std::vector<Eigen::MatrixXd> G;
    for (const auto& t : logdets) {
      const Eigen::MatrixXd Ct = principal(cd.C, t.idx);
      Eigen::LLT<Eigen::MatrixXd> llt(Ct);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(penalty_err(
            "penalty covariance not positive definite in block " + std::to_string(b)));
      }
      G.push_back(llt.solve(Eigen::MatrixXd::Identity(Ct.rows(), Ct.rows())));
      for (std::size_t a = 0; a < t.idx.size(); ++a)
        for (std::size_t c = 0; c < t.idx.size(); ++c)
          M(t.idx[a], t.idx[c]) += t.sign * G.back()(static_cast<Eigen::Index>(a),
                                                     static_cast<Eigen::Index>(c));
    }
    for (const std::int32_t i : diag) {
      if (!(cd.C(i, i) > 0.0)) {
        return std::unexpected(penalty_err(
            "non-positive complete-data variance in block " + std::to_string(b)));
      }
      M(i, i) -= 1.0 / cd.C(i, i);
    }

    // Free parameters of this block and their rank-two dC = x yᵀ + y xᵀ.
    std::vector<Eigen::Index> ks;
    std::vector<RamCell> cells;
    for (Eigen::Index k = 0; k < q; ++k) {
      const auto& loc = layout.locations[static_cast<std::size_t>(k)];
      if (loc.row < 0 || loc.col < 0 || static_cast<std::size_t>(loc.block) != b) continue;
      RamCell cell;
      switch (loc.mat) {
        case MatId::Psi: cell = {true, loc.row, loc.col}; break;
        case MatId::Theta: cell = {true, m + loc.row, m + loc.col}; break;
        case MatId::Beta: cell = {false, loc.row, loc.col}; break;
        case MatId::Lambda: cell = {false, m + loc.row, loc.col}; break;
        case MatId::Nu:
        case MatId::Alpha: continue;
      }
      ks.push_back(k);
      cells.push_back(cell);
    }
    const Eigen::Index qb = static_cast<Eigen::Index>(ks.size());
    if (qb == 0) continue;
    Eigen::MatrixXd X(n, qb), Y(n, qb);
    for (Eigen::Index a = 0; a < qb; ++a) {
      const auto& cell = cells[static_cast<std::size_t>(a)];
      if (cell.in_s) {
        X.col(a) = cd.E.col(cell.r);
        Y.col(a) = cell.r == cell.c ? Eigen::VectorXd(0.5 * cd.E.col(cell.r))
                                    : Eigen::VectorXd(cd.E.col(cell.c));
      } else {
        X.col(a) = cd.E.col(cell.r);
        Y.col(a) = cd.C.col(cell.c);
      }
    }
    Eigen::MatrixXd Hb = Eigen::MatrixXd::Zero(qb, qb);
    // −Σ_t sign_t tr(G_t dC_a G_t dC_b) on each term's principal block.
    for (std::size_t t = 0; t < logdets.size(); ++t) {
      const auto& idx = logdets[t].idx;
      Eigen::MatrixXd Xt(static_cast<Eigen::Index>(idx.size()), qb);
      Eigen::MatrixXd Yt(static_cast<Eigen::Index>(idx.size()), qb);
      for (std::size_t i = 0; i < idx.size(); ++i) {
        Xt.row(static_cast<Eigen::Index>(i)) = X.row(idx[i]);
        Yt.row(static_cast<Eigen::Index>(i)) = Y.row(idx[i]);
      }
      const Eigen::MatrixXd GX = G[t] * Xt;
      const Eigen::MatrixXd GY = G[t] * Yt;
      const Eigen::MatrixXd Pxx = Xt.transpose() * GX;
      const Eigen::MatrixXd Pxy = Xt.transpose() * GY;
      const Eigen::MatrixXd Pyy = Yt.transpose() * GY;
      Hb.array() -= logdets[t].sign * 2.0 *
          (Pxy.array() * Pxy.transpose().array() + Pxx.array() * Pyy.array());
    }
    // Σ_D (dC_ii)_a (dC_ii)_b / C_ii², with dC_ii = 2 x_i y_i.
    for (const std::int32_t i : diag) {
      const Eigen::VectorXd z = 2.0 * X.row(i).transpose().cwiseProduct(Y.row(i).transpose());
      Hb.noalias() += (z * z.transpose()) / (cd.C(i, i) * cd.C(i, i));
    }
    // tr(M d²C): S is linear and so is A, so only pairs with an A cell remain.
    const Eigen::MatrixXd N = cd.E.transpose() * M * cd.E;
    const Eigen::MatrixXd P = cd.C * M * cd.E;
    const auto& E = cd.E;
    const auto& C = cd.C;
    auto mixed_term = [&](const RamCell& a, const RamCell& s) {  // a in A, s in S
      if (s.r == s.c) return 2.0 * E(a.c, s.r) * N(s.r, a.r);
      return 2.0 * (E(a.c, s.r) * N(s.c, a.r) + E(a.c, s.c) * N(s.r, a.r));
    };
    for (Eigen::Index a = 0; a < qb; ++a) {
      const auto& ca = cells[static_cast<std::size_t>(a)];
      for (Eigen::Index c = a; c < qb; ++c) {
        const auto& cc = cells[static_cast<std::size_t>(c)];
        double v = 0.0;
        if (!ca.in_s && !cc.in_s) {
          v = 2.0 * (P(ca.c, cc.r) * E(cc.c, ca.r) + P(cc.c, ca.r) * E(ca.c, cc.r) +
                     N(cc.r, ca.r) * C(ca.c, cc.c));
        } else if (!ca.in_s) {
          v = mixed_term(ca, cc);
        } else if (!cc.in_s) {
          v = mixed_term(cc, ca);
        }
        Hb(a, c) += v;
        if (a != c) Hb(c, a) += v;
      }
    }
    for (Eigen::Index a = 0; a < qb; ++a)
      for (Eigen::Index c = 0; c < qb; ++c)
        H(ks[static_cast<std::size_t>(a)], ks[static_cast<std::size_t>(c)]) += Hb(a, c);
  }
  H = (0.5 * (H + H.transpose())).eval();
  if (!H.allFinite()) return std::unexpected(penalty_err("non-finite penalty Hessian"));
  return H;
}

fit_expected<MultiInfoPenaltyValue>
multiinfo_penalty(const MultiInfoPenaltyLayout& layout,
                  const model::ModelEvaluator& ev,
                  const Eigen::VectorXd& theta, bool with_gradient) {
  auto assembled = ev.assembled(theta);
  if (!assembled.has_value()) {
    return std::unexpected(penalty_err(
        "cannot assemble model matrices: " + assembled.error().detail));
  }
  if (assembled->blocks.size() != layout.blocks.size()) {
    return std::unexpected(penalty_err("layout/evaluator block mismatch"));
  }
  MultiInfoPenaltyValue out;
  std::vector<BlockGradient> grads(with_gradient ? layout.blocks.size() : 0);
  for (std::size_t b = 0; b < layout.blocks.size(); ++b) {
    const auto& blk = layout.blocks[b];
    const CompleteData cd = complete_data(assembled->blocks[b], blk.m, blk.p);
    auto v = block_value(layout, cd, b, with_gradient ? &grads[b] : nullptr);
    if (!v.has_value()) return std::unexpected(v.error());
    out.value += *v;
  }
  if (!with_gradient) return out;

  out.gradient = Eigen::VectorXd::Zero(theta.size());
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    const auto& loc = layout.locations[static_cast<std::size_t>(k)];
    if (loc.row < 0 || loc.col < 0) continue;  // Stage-1/LS threshold coordinate
    const std::size_t b = static_cast<std::size_t>(loc.block);
    const Eigen::Index m = layout.blocks[b].m;
    const BlockGradient& g = grads[b];
    const Eigen::Index r = loc.row;
    const Eigen::Index c = loc.col;
    switch (loc.mat) {
      case MatId::Psi:
        out.gradient(k) = r == c ? g.S(r, r) : g.S(r, c) + g.S(c, r);
        break;
      case MatId::Theta:
        out.gradient(k) = r == c ? g.S(m + r, m + r)
                                 : g.S(m + r, m + c) + g.S(m + c, m + r);
        break;
      case MatId::Beta:
        out.gradient(k) = g.A(r, c);
        break;
      case MatId::Lambda:
        out.gradient(k) = g.A(m + r, c);
        break;
      case MatId::Nu:
      case MatId::Alpha:
        break;
    }
  }
  return out;
}

fit_expected<MultiInfoPenaltyReport>
multiinfo_penalty_report(const MultiInfoPenaltyLayout& layout,
                         const model::ModelEvaluator& ev,
                         const Eigen::VectorXd& theta) {
  auto assembled = ev.assembled(theta);
  if (!assembled.has_value()) {
    return std::unexpected(penalty_err(
        "cannot assemble model matrices: " + assembled.error().detail));
  }
  MultiInfoPenaltyReport out;
  out.target = layout.target;
  out.recursive = layout.recursive();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t b = 0; b < layout.blocks.size(); ++b) {
    const auto& blk = layout.blocks[b];
    const CompleteData cd = complete_data(assembled->blocks[b], blk.m, blk.p);

    if (layout.target == PenaltyTarget::Determinacy) {
      DeterminacyParts parts;
      auto v = block_determinacy(blk, cd, b, nullptr, &parts);
      if (!v.has_value()) return std::unexpected(v.error());
      out.value += *v;
      out.block_value.push_back(*v);
      if (blk.keep.empty()) {
        out.residual_log_det_corr.push_back(0.0);
        out.total_correlation.push_back(0.0);
        out.mutual_information.push_back(0.0);
        continue;
      }
      const Eigen::VectorXd v_diag =
          (parts.chol_v * parts.chol_v.transpose()).diagonal();
      const double log_det_v =
          2.0 * parts.chol_v.diagonal().array().log().sum();
      out.residual_log_det_corr.push_back(log_det_v -
                                          v_diag.array().log().sum());
      const Eigen::MatrixXd C_LL = principal(cd.C, blk.keep);
      Eigen::LLT<Eigen::MatrixXd> llt(C_LL);
      double log_det_phi = nan;
      if (llt.info() == Eigen::Success) {
        const Eigen::MatrixXd L = llt.matrixL();
        log_det_phi = 2.0 * L.diagonal().array().log().sum();
      }
      out.total_correlation.push_back(
          -0.5 * (log_det_phi - parts.d.array().log().sum()));
      out.mutual_information.push_back(0.5 * (log_det_phi - log_det_v));
      for (std::size_t a = 0; a < blk.keep.size(); ++a) {
        MultiInfoPenaltyTerm term;
        term.block = static_cast<std::int16_t>(b);
        term.latent = true;
        term.index = static_cast<std::int16_t>(blk.keep[a]);
        const auto ai = static_cast<Eigen::Index>(a);
        term.log_one_minus_r2 = std::log(v_diag(ai) / parts.d(ai));
        out.terms.push_back(term);
      }
      continue;
    }

    auto v = block_penalty(blk, cd, b, nullptr);
    if (!v.has_value()) return std::unexpected(v.error());
    out.value += *v;
    out.block_value.push_back(*v);
    out.total_correlation.push_back(nan);
    out.mutual_information.push_back(nan);

    double residual = nan;
    if (!blk.keep.empty()) {
      const Eigen::MatrixXd S_KK = principal(cd.S, blk.keep);
      Eigen::LLT<Eigen::MatrixXd> llt(S_KK);
      if (llt.info() == Eigen::Success &&
          (S_KK.diagonal().array() > 0.0).all()) {
        const Eigen::MatrixXd L = llt.matrixL();
        residual = 2.0 * L.diagonal().array().log().sum() -
                   S_KK.diagonal().array().log().sum();
      }
    } else {
      residual = 0.0;
    }
    out.residual_log_det_corr.push_back(residual);

    for (const std::int32_t i : blk.keep) {
      MultiInfoPenaltyTerm term;
      term.block = static_cast<std::int16_t>(b);
      term.latent = i < blk.m;
      term.index = static_cast<std::int16_t>(term.latent ? i : i - blk.m);
      term.log_one_minus_r2 = std::log(cd.S(i, i) / cd.C(i, i));
      out.terms.push_back(term);
    }
  }
  return out;
}

fit_expected<double>
multiinfo_penalty_weight(const MultiInfoPenaltyOptions& options) {
  const double weight =
      std::isfinite(options.weight) ? options.weight : options.eta - 1.0;
  if (!std::isfinite(weight) || weight < 0.0) {
    return std::unexpected(penalty_err(
        "penalty weight must be finite and non-negative (eta >= 1)"));
  }
  return weight;
}

optim::ScalarProblem
multiinfo_penalized_problem(const optim::ScalarProblem& base,
                            const MultiInfoPenaltyLayout& layout,
                            const model::ModelEvaluator& ev,
                            double weight, double n_total) {
  optim::ScalarProblem out;
  out.n_param = base.n_param;
  out.expand = base.expand;
  const double scale = weight / n_total;
  out.f = [&base, &layout, &ev, scale](const Eigen::VectorXd& theta,
                                       Eigen::VectorXd& grad) -> double {
    const double f = base.f(theta, grad);
    if (!std::isfinite(f)) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    if (scale == 0.0) return f;
    // LS discrepancies do not themselves require a PD observed covariance.
    // Determinacy conditions on that block, so enforce its domain here too.
    auto moments = ev.sigma(theta);
    if (!moments) { grad.setZero(); return std::numeric_limits<double>::infinity(); }
    for (const auto& sigma : moments->sigma) {
      Eigen::LLT<Eigen::MatrixXd> chol(sigma);
      if (chol.info() != Eigen::Success) {
        grad.setZero();
        return std::numeric_limits<double>::infinity();
      }
    }
    auto pen = multiinfo_penalty(layout, ev, theta, true);
    if (!pen.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    grad -= scale * pen->gradient;
    return f - scale * pen->value;
  };
  return out;
}

}  // namespace magmaan::estimate::frontier
