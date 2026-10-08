#include "magmaan/estimate/frontier/identification.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/QR>
#include <Eigen/SVD>

#include "magmaan/parse/op.hpp"
#include "../detail_identification_probe.hpp"

#ifdef MAGMAAN_ENABLE_TEST_PROBES
#include <atomic>
#include <chrono>
#endif

namespace magmaan::estimate {

std::string_view to_string(IdentificationStatus s) noexcept {
  switch (s) {
    case IdentificationStatus::Unchecked: return "unchecked";
    case IdentificationStatus::Identified: return "identified";
    case IdentificationStatus::Unidentified: return "unidentified";
  }
  return "unchecked";
}

std::string_view to_string(IdentificationReason r) noexcept {
  switch (r) {
    case IdentificationReason::NotAttempted: return "not_attempted";
    case IdentificationReason::Rank: return "rank";
    case IdentificationReason::CountingRule: return "counting_rule";
    case IdentificationReason::NoFreeParameters: return "no_free_parameters";
    case IdentificationReason::NonlinearConstraints: return "nonlinear_constraints";
    case IdentificationReason::UnsupportedModel: return "unsupported_model";
    case IdentificationReason::ConstraintsUnavailable: return "constraints_unavailable";
    case IdentificationReason::EvaluationFailed: return "evaluation_failed";
    case IdentificationReason::AmbiguousGap: return "ambiguous_gap";
  }
  return "not_attempted";
}

std::string_view to_string(IdentificationMap m) noexcept {
  switch (m) {
    case IdentificationMap::None: return "none";
    case IdentificationMap::Covariance: return "covariance";
    case IdentificationMap::CovarianceMean: return "covariance_mean";
    case IdentificationMap::Ordinal: return "ordinal";
    case IdentificationMap::Mixed: return "mixed";
  }
  return "none";
}

}  // namespace magmaan::estimate

#ifdef MAGMAAN_ENABLE_TEST_PROBES
namespace magmaan::estimate::identification_test {
namespace {
std::atomic<Observer> current_observer{nullptr};
}  // namespace
void set_observer(Observer observer) noexcept { current_observer.store(observer); }
Observer observer() noexcept { return current_observer.load(); }
}  // namespace magmaan::estimate::identification_test
#endif

namespace magmaan::estimate::frontier {
namespace {

// SplitMix64 (Steele, Lea and Flood 2014) with an explicit 53-bit conversion:
// unlike the standard distributions it is specified bit for bit, so the
// draws, and with them every report, are the same on every platform.
struct Draws {
  std::uint64_t state = 0;
  std::uint64_t next() noexcept {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double uniform(double lo, double hi) noexcept {
    const double u = static_cast<double>(next() >> 11) * 0x1.0p-53;
    return lo + (hi - lo) * u;
  }
  double sign() noexcept { return (next() >> 63) != 0 ? -1.0 : 1.0; }
  // Magnitude in [lo, hi) with a random sign; the two draws are sequenced.
  double signed_uniform(double lo, double hi) noexcept {
    const double s = sign();
    return s * uniform(lo, hi);
  }
};

// Moderate values by parameter kind. Any absolutely continuous law finds the
// generic rank; these ranges keep every point away from the obvious special
// sets (zero loadings, singular I - B, nonpositive implied variances) so the
// relative singular values stay far from roundoff.
double draw_value(const spec::LatentStructure& pt, std::size_t row, Draws& d) {
  switch (pt.op[row]) {
    case parse::Op::Measurement: return d.uniform(0.5, 1.5);
    case parse::Op::Regression: return d.signed_uniform(0.2, 0.6);
    case parse::Op::Covariance:
      if (pt.lhs_var[row] == pt.rhs_var[row]) return d.uniform(0.5, 1.5);
      return d.signed_uniform(0.05, 0.3);
    case parse::Op::Intercept: return d.uniform(-1.0, 1.0);
    case parse::Op::Threshold: return d.uniform(-1.5, 1.5);
    case parse::Op::ResponseScale: return d.uniform(0.6, 1.4);
    default: return d.uniform(0.5, 1.5);
  }
}

// The affine space theta = theta0 + K alpha of the linear equalities.
class Reduction {
 public:
  bool init(const EqConstraints& con, Eigen::Index n) {
    con_ = &con;
    n_ = n;
    if (n == 0) { q_ = 0; identity_ = true; return true; }
    if (con.npar != n || con.Kmat.rows() != n || con.theta0.size() != n ||
        con.Kmat.cols() != con.n_alpha || con.n_alpha < 0 ||
        !con.Kmat.allFinite() || !con.theta0.allFinite()) return false;
    q_ = con.n_alpha;
    identity_ = con.rank == 0 && q_ == n && con.theta0.isZero(0.0) &&
        con.Kmat.isIdentity(0.0);
    merge_ = !identity_ && con.group.size() == static_cast<std::size_t>(n) &&
        con.theta0.isZero(0.0);
    if (merge_) {
      for (const auto g : con.group)
        if (g < 0 || g >= q_) { merge_ = false; break; }
    }
    if (!identity_ && q_ > 0) qr_.compute(con.Kmat);
    return true;
  }
  Eigen::Index q() const noexcept { return q_; }

  Eigen::VectorXd project(const Eigen::VectorXd& raw) const {
    if (identity_) return raw;
    if (q_ == 0) return con_->theta0;
    const Eigen::VectorXd alpha = qr_.solve(raw - con_->theta0);
    return con_->theta0 + con_->Kmat * alpha;
  }
  Eigen::MatrixXd reduce(const Eigen::MatrixXd& jacobian) const {
    if (identity_) return jacobian;
    if (merge_) {
      Eigen::MatrixXd out = Eigen::MatrixXd::Zero(jacobian.rows(), q_);
      for (Eigen::Index k = 0; k < n_; ++k)
        out.col(con_->group[static_cast<std::size_t>(k)]) += jacobian.col(k);
      return out;
    }
    return jacobian * con_->Kmat;
  }
  Eigen::VectorXd expand_direction(const Eigen::VectorXd& alpha) const {
    if (identity_) return alpha;
    return con_->Kmat * alpha;
  }
  Eigen::MatrixXd direction_coordinates(const Eigen::MatrixXd& full) const {
    if (identity_) return full;
    return qr_.solve(full);
  }
  Eigen::MatrixXd expand_directions(const Eigen::MatrixXd& alpha) const {
    if (identity_) return alpha;
    return con_->Kmat * alpha;
  }

 private:
  const EqConstraints* con_ = nullptr;
  Eigen::Index n_ = 0;
  Eigen::Index q_ = 0;
  bool identity_ = false;
  bool merge_ = false;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
};

Eigen::VectorXd draw_point(const spec::LatentStructure& pt,
                           const Reduction& reduction, Draws& draws) {
  const Eigen::Index n = pt.n_free();
  Eigen::VectorXd raw = Eigen::VectorXd::Ones(n);
  std::vector<char> seen(static_cast<std::size_t>(n), 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const std::int32_t f = i < pt.free.size() ? pt.free[i] : 0;
    if (f <= 0 || f > n || seen[static_cast<std::size_t>(f - 1)]) continue;
    seen[static_cast<std::size_t>(f - 1)] = 1;
    raw(f - 1) = draw_value(pt, i, draws);
  }
  return reduction.project(raw);
}

// Relative singular values of the reduced Jacobian after scaling every
// column to unit length. A column that is negligible against the largest one
// is a null direction by itself; it is zeroed rather than inflated to unit
// length, so roundoff in an inert parameter cannot pose as information.
struct PointRank {
  Eigen::VectorXd relative;  // q values, descending, zero padded
  Eigen::VectorXd scale;     // column scaling applied
  Eigen::MatrixXd V;         // q x q right singular vectors when requested
  std::int32_t rank = 0;
  double min_relative = 0.0;
};

PointRank rank_point(const Eigen::MatrixXd& reduced, double null_tolerance,
                     bool want_vectors) {
  const Eigen::Index q = reduced.cols();
  PointRank out;
  out.relative = Eigen::VectorXd::Zero(q);
  out.scale = Eigen::VectorXd::Ones(q);
  Eigen::MatrixXd scaled = reduced;
  const Eigen::VectorXd norms = reduced.colwise().norm().transpose();
  const double largest = q > 0 ? norms.maxCoeff() : 0.0;
  if (largest > 0.0 && std::isfinite(largest)) {
    for (Eigen::Index j = 0; j < q; ++j) {
      if (norms(j) > null_tolerance * largest) {
        out.scale(j) = 1.0 / norms(j);
        scaled.col(j) *= out.scale(j);
      } else {
        scaled.col(j).setZero();
      }
    }
  } else {
    scaled.setZero();
  }
  const unsigned int options =
      want_vectors ? static_cast<unsigned int>(Eigen::ComputeFullV) : 0u;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(scaled, options);
  const Eigen::VectorXd& values = svd.singularValues();
  if (values.size() > 0 && values(0) > 0.0)
    out.relative.head(values.size()) = values / values(0);
  for (Eigen::Index j = 0; j < q; ++j)
    if (out.relative(j) > null_tolerance) ++out.rank;
  out.min_relative = q > 0 ? out.relative(q - 1) : 1.0;
  if (want_vectors) out.V = svd.matrixV();
  return out;
}

Eigen::VectorXd ascending_tail(const Eigen::VectorXd& descending,
                               Eigen::Index count) {
  count = std::min(count, descending.size());
  Eigen::VectorXd out(count);
  for (Eigen::Index k = 0; k < count; ++k)
    out(k) = descending(descending.size() - 1 - k);
  return out;
}

struct NullBasis {
  Eigen::MatrixXd directions;
  PointRank point;
};

std::optional<NullBasis> null_basis_at(
    const Eigen::VectorXd& theta, const Reduction& reduction,
    const MomentJacobian& jacobian, Eigen::Index n, Eigen::Index nullity,
    double null_tolerance) {
  auto J = jacobian(theta);
  if (!J.has_value() || J->cols() != n || J->rows() == 0 || !J->allFinite())
    return std::nullopt;
  PointRank point = rank_point(reduction.reduce(*J), null_tolerance, true);
  const Eigen::Index q = point.V.cols();
  if (q < nullity || nullity <= 0) return std::nullopt;
  Eigen::MatrixXd out(n, nullity);
  for (Eigen::Index k = 0; k < nullity; ++k) {
    const Eigen::VectorXd alpha =
        point.scale.cwiseProduct(point.V.col(q - nullity + k));
    Eigen::VectorXd d = reduction.expand_direction(alpha);
    const double norm = d.norm();
    if (!(norm > 0.0) || !std::isfinite(norm)) return std::nullopt;
    d /= norm;
    Eigen::Index largest = 0;
    d.cwiseAbs().maxCoeff(&largest);
    if (d(largest) < 0.0) d = -d;
    out.col(k) = d;
  }
  return NullBasis{std::move(out), std::move(point)};
}


// Work in the same equilibrated reduced coordinates as rank_point. A full
// matrix-cell map also checks fixed/absent entries: dropping those entries
// would falsely call a pinned marker or a zero cross-loading a gauge.
void classify_directions(IdentificationReport& report,
                         const spec::LatentStructure& pt,
                         const Reduction& reduction,
                         const PointRank& point,
                         const Eigen::VectorXd& theta,
                         const model::ModelEvaluator * ev) {
  const Eigen::Index d = report.null_directions.cols();
  if (d == 0) return;
  report.deficit_dimension = static_cast<std::int32_t>(d);
  report.direction_types.assign(static_cast<std::size_t>(d), "deficit");
  report.direction_factors.resize(static_cast<std::size_t>(d));
  report.suggested_fixes.assign(static_cast<std::size_t>(d),
      "Information deficit: no automatic fix; add information or independent restrictions to the listed parameters and recheck identification.");
  if (ev == nullptr) return;
  auto assembled = ev->assembled(theta);
  if (!assembled) return;
  const Eigen::Index q = point.V.cols(), n = theta.size();
  const Eigen::MatrixXd N = point.V.rightCols(d);
  struct Layout { Eigen::Index lambda, psi, beta, nu, alpha, theta, p, m; };
  std::vector<Layout> layouts;
  Eigen::Index cells = 0;
  for (const auto& b : assembled->blocks) {
    const Eigen::Index p = b.Lambda.rows(), m = b.Lambda.cols();
    Layout l{cells, cells + p * m, cells + p * m + m * m,
             cells + p * m + 2 * m * m, cells + p * m + 2 * m * m + p,
             cells + p * m + 2 * m * m + p + m, p, m};
    cells = l.theta + p * p;
    layouts.push_back(l);
  }
  auto cell_index = [](const Layout& l, model::MatId mat, int row, int col) {
    switch (mat) {
      case model::MatId::Lambda: return l.lambda + row + l.p * col;
      case model::MatId::Psi: return l.psi + row + l.m * col;
      case model::MatId::Beta: return l.beta + row + l.m * col;
      case model::MatId::Nu: return l.nu + row;
      case model::MatId::Alpha: return l.alpha + row;
      case model::MatId::Theta: return l.theta + row + l.p * col;
      default: return Eigen::Index(-1);
    }
  };
  const Eigen::Index extra_cells = cells;
  cells += static_cast<Eigen::Index>(pt.size());
  std::vector<Eigen::Index> parameter_for_cell(static_cast<std::size_t>(cells), -1);
  const auto& rep = ev->matrix_rep();
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto& c = rep.cell_for_row[i];
    if (pt.free[i] <= 0) continue;
    if (!c.used) {
      // Thresholds and response scales are unchanged by a LISREL gauge.
      // Keeping their tangent entries also enforces equalities tying them to
      // matrix parameters, rather than dropping them from the reduction.
      parameter_for_cell[static_cast<std::size_t>(extra_cells) + i] = pt.free[i]-1;
      continue;
    }
    const auto& l = layouts[static_cast<std::size_t>(c.block)];
    const auto k = cell_index(l, c.mat, c.row, c.col);
    if (k < 0) continue;
    parameter_for_cell[static_cast<std::size_t>(k)] = pt.free[i]-1;
    if (c.mat == model::MatId::Psi || c.mat == model::MatId::Theta)
      parameter_for_cell[static_cast<std::size_t>(cell_index(l, c.mat, c.col, c.row))] = pt.free[i]-1;
  }
  struct Generator {
    Eigen::VectorXd cells;
    std::string type, factors;
  };
  std::vector<Generator> generators;
  for (std::size_t bi = 0; bi < layouts.size(); ++bi) {
    const auto& l = layouts[bi];
    const auto& b = assembled->blocks[bi];
    std::vector<int> factors;
    for (const auto v : pt.lv_ext_order)
      if (pt.is_user_latent[static_cast<std::size_t>(v)])
        factors.push_back(pt.lv_ext_pos[static_cast<std::size_t>(v)]);
    auto name = [&](int a) {
      std::string out = rep.lv_names[bi][static_cast<std::size_t>(a)];
      if (layouts.size() > 1) out += ".g" + std::to_string(bi + 1);
      return out;
    };
    auto pack = [&](const Eigen::MatrixXd& dl, const Eigen::MatrixXd& dp,
                    const Eigen::MatrixXd& db, const Eigen::VectorXd& dn,
                    const Eigen::VectorXd& da) {
      Eigen::VectorXd g = Eigen::VectorXd::Zero(cells);
      g.segment(l.lambda, l.p * l.m) = Eigen::Map<const Eigen::VectorXd>(dl.data(), dl.size());
      g.segment(l.psi, l.m * l.m) = Eigen::Map<const Eigen::VectorXd>(dp.data(), dp.size());
      g.segment(l.beta, l.m * l.m) = Eigen::Map<const Eigen::VectorXd>(db.data(), db.size());
      g.segment(l.nu, l.p) = dn;
      g.segment(l.alpha, l.m) = da;
      return g;
    };
    // The diagonal GL(k) generators are the scale transformations.
    for (int a : factors) for (int z : factors) {
      Eigen::MatrixXd E = Eigen::MatrixXd::Zero(l.m, l.m);
      E(a, z) = 1.0;
      Eigen::VectorXd da = Eigen::VectorXd::Zero(l.m);
      if (b.Alpha.size() == l.m) da = -E * b.Alpha;
      generators.push_back({pack(b.Lambda * E, -(E * b.Psi + b.Psi * E.transpose()),
          b.Beta * E - E * b.Beta, Eigen::VectorXd::Zero(l.p), da),
          a == z ? "scale" : "rotation", a == z ? name(a) : name(a)+", "+name(z)});
    }
    if (b.Nu.size() > 0) for (int a : factors) {
      Eigen::VectorXd shift = Eigen::VectorXd::Zero(l.m); shift(a) = 1.0;
      generators.push_back({pack(Eigen::MatrixXd::Zero(l.p, l.m),
          Eigen::MatrixXd::Zero(l.m, l.m), Eigen::MatrixXd::Zero(l.m, l.m),
          -b.Lambda * shift, shift - b.Beta * shift), "location", name(a)});
    }
  }
  const Eigen::Index g = static_cast<Eigen::Index>(generators.size());
  if (g == 0) return;
  Eigen::MatrixXd G(cells, g);
  for (Eigen::Index j = 0; j < g; ++j) {
    G.col(j) = generators[static_cast<std::size_t>(j)].cells;
    const double norm = G.col(j).norm();
    if (norm>0) G.col(j) /= norm;
  }
  // Every cell has one parameter owner, so C^T C is diagonal. Average
  // repeated (symmetric or aliased) cells, then reuse K's prepared QR. The
  // residual below still checks every fixed cell and alias: its kernel is
  // exactly the admissible generator span. No dense solve over cells is needed.
  Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(n, g);
  Eigen::VectorXd counts = Eigen::VectorXd::Zero(n);
  for (Eigen::Index i = 0; i < cells; ++i) {
    const auto k = parameter_for_cell[static_cast<std::size_t>(i)];
    if (k < 0) continue;
    raw.row(k) += G.row(i);
    counts(k) += 1.0;
  }
  if ((counts.array() <= 0.0).any()) return;
  raw = counts.cwiseInverse().asDiagonal() * raw;
  const Eigen::MatrixXd reduced_coordinates = reduction.direction_coordinates(raw);
  const Eigen::MatrixXd X = point.scale.cwiseInverse().asDiagonal() * reduced_coordinates;
  const Eigen::MatrixXd expanded = reduction.expand_directions(reduced_coordinates);
  // The first residual enforces all cell restrictions and linear equalities;
  // the second enforces membership in the normalized numerical null space.
  Eigen::MatrixXd residual(cells + q, g);
  residual.topRows(cells) = G;
  for (Eigen::Index i = 0; i < cells; ++i) {
    const auto k = parameter_for_cell[static_cast<std::size_t>(i)];
    if (k >= 0) residual.row(i) -= expanded.row(k);
  }
  residual.bottomRows(q) = X - N * (N.transpose() * X);
  std::vector<Eigen::VectorXd> directions, reported_directions;
  std::vector<std::string> types, names;
  auto append = [&](Eigen::VectorXd v, const std::string& type, const std::string& factors) {
    const double original = v.norm();
    const Eigen::VectorXd reported = v;
    for (const auto& old : directions) v -= old.dot(v) * old;
    // Reorthogonalize to keep the deficit complement stable.
    for (const auto& old : directions) v -= old.dot(v) * old;
    if (!(original > 0.0) || !(v.norm() > report.null_tolerance * original)) return;
    v.normalize();
    directions.push_back(v);
    // Orthogonalize only the internal span. The reported gauge column must
    // remain the actual generator (subtracting a scale from a translation
    // would give a mixed direction and invalidate its pure location label).
    reported_directions.push_back(type == "deficit" ? v : reported.normalized().eval());
    types.push_back(type);
    names.push_back(factors);
  };
  for (const auto& type : {"scale", "location"})
    for (Eigen::Index j = 0; j < g; ++j)
      if (generators[static_cast<std::size_t>(j)].type == type &&
          X.col(j).norm() > 0.0 &&
          residual.col(j).head(cells).norm() <= report.null_tolerance &&
          residual.col(j).tail(q).norm() <= report.null_tolerance * X.col(j).norm())
        append(N * (N.transpose() * X.col(j)), type, generators[static_cast<std::size_t>(j)].factors);
  // First restrict the generator span to directions honoring every cell and
  // linear equality. Then compute principal angles to the numerical null
  // space in the equilibrated coordinates used by the identification check.
  // Keep translations separate from GL(k): a mixture of the two must not
  // receive a pure location or rotation label merely from a large coefficient.
  for (bool locations : {true, false}) {
    std::vector<Eigen::Index> selected;
    for (Eigen::Index j = 0; j < g; ++j)
      if ((generators[static_cast<std::size_t>(j)].type == "location") == locations)
        selected.push_back(j);
    const Eigen::Index count = static_cast<Eigen::Index>(selected.size());
    if (count == 0) continue;
    Eigen::MatrixXd cell_residual(cells, count), coordinates(q, count);
    for (Eigen::Index j = 0; j < count; ++j) {
      cell_residual.col(j) = residual.col(selected[static_cast<std::size_t>(j)]).head(cells);
      coordinates.col(j) = X.col(selected[static_cast<std::size_t>(j)]);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> admissible(cell_residual, Eigen::ComputeFullV);
    Eigen::Index admissible_dim = 0;
    for (Eigen::Index j = 0; j < count; ++j)
      if (j >= admissible.singularValues().size() ||
          admissible.singularValues()(j) <= report.null_tolerance) ++admissible_dim;
    if (admissible_dim>0) {
      const Eigen::MatrixXd weights_basis = admissible.matrixV().rightCols(admissible_dim);
      const Eigen::MatrixXd allowed = coordinates * weights_basis;
      Eigen::JacobiSVD<Eigen::MatrixXd> span(allowed, Eigen::ComputeThinU | Eigen::ComputeThinV);
      span.setThreshold(report.null_tolerance);
      Eigen::Index span_dim = 0;
      const double largest = span.singularValues().size()>0 ? span.singularValues()(0) : 0.0;
      for (const double value : span.singularValues())
        if (value > report.null_tolerance * largest) ++span_dim;
      if (span_dim>0) {
        const Eigen::MatrixXd Q_G = span.matrixU().leftCols(span_dim);
        Eigen::JacobiSVD<Eigen::MatrixXd> angles(Q_G.transpose() * N,
            Eigen::ComputeThinU | Eigen::ComputeThinV);
        for (Eigen::Index j = 0; j < angles.singularValues().size(); ++j) {
          if (1.0 - angles.singularValues()(j)>report.null_tolerance) continue;
          Eigen::VectorXd v = N * angles.matrixV().col(j);
          // Recover the actual admissible generator combination. Keep the
          // reported column in this span; append removes dependent columns.
          const Eigen::VectorXd weights = weights_basis * span.solve(v);
          std::string pairs, other;
          const std::string type = locations ? "location" : "scale";
          const double max_weight = weights.cwiseAbs().maxCoeff();
          for (Eigen::Index k = 0; k < count; ++k) {
            const auto& gen = generators[static_cast<std::size_t>(selected[static_cast<std::size_t>(k)])];
            if (std::abs(weights(k)) <= report.null_tolerance * max_weight) continue;
            if (gen.type == "rotation") {
              if (!pairs.empty()) pairs += "; ";
              pairs += gen.factors;
            } else {
              if (!other.empty()) other += "; ";
              other += gen.factors;
            }
          }
          append(v, pairs.empty() ? type : "rotation", pairs.empty() ? other : pairs);
        }
      }
    }
  }
  report.gauge_dimension = static_cast<std::int32_t>(directions.size());
  for (Eigen::Index j = 0; j < d; ++j) append(N.col(j), "deficit", "");
  report.deficit_dimension = static_cast<std::int32_t>(directions.size()) - report.gauge_dimension;
  report.direction_types = std::move(types);
  report.direction_factors = std::move(names);
  report.suggested_fixes.clear();
  report.null_directions.resize(n, static_cast<Eigen::Index>(directions.size()));
  for (std::size_t j = 0; j < directions.size(); ++j) {
    Eigen::VectorXd v = reduction.expand_direction(
        point.scale.cwiseProduct(reported_directions[j]));
    v.normalize();
    Eigen::Index largest = 0;
    v.cwiseAbs().maxCoeff(&largest);
    if (v(largest) < 0.0) v = -v;
    report.null_directions.col(static_cast<Eigen::Index>(j)) = v;
    const auto& type = report.direction_types[j];
    const auto& f = report.direction_factors[j];
    if (type == "scale") {
      report.suggested_fixes.push_back("Scale freedom of " + f +
          ": fix one loading to 1, or its variance to 1 (std.lv); recheck identification.");
    } else if (type == "location") {
      report.suggested_fixes.push_back("Location freedom of " + f +
          ": fix its latent mean to 0, or one intercept; recheck identification.");
    } else if (type == "rotation") {
      report.suggested_fixes.push_back("Free rotation between factors " + f +
          ": add independent loading restrictions (each factor needs k - 1 zeros "
          "in a non-degenerate pattern); covariance restrictions can help but are "
          "not sufficient on their own; recheck identification after editing.");
    } else {
      report.suggested_fixes.push_back("Information deficit: no automatic fix; "
          "add information or independent restrictions to the listed parameters "
          "and recheck identification.");
    }
  }
}

IdentificationReport base_report(IdentificationMap map,
                                 const IdentificationOptions& options) {
  IdentificationReport r;
  r.map = map;
  r.null_tolerance = options.null_tolerance;
  r.identified_tolerance = options.identified_tolerance;
  r.seed = options.seed;
  return r;
}

bool valid_options(const IdentificationOptions& o) {
  return o.n_points >= 1 && std::isfinite(o.null_tolerance) &&
      std::isfinite(o.identified_tolerance) && o.null_tolerance > 0.0 &&
      o.null_tolerance < o.identified_tolerance && o.identified_tolerance < 1.0;
}

constexpr int draws_per_point = 5;

IdentificationReport rank_impl(const spec::LatentStructure& pt,
                               const EqConstraints& con,
                               bool nonlinear_constraints, IdentificationMap map,
                               const MomentJacobian& jacobian,
                               const Eigen::VectorXd* estimate,
                               const IdentificationOptions& options,
                               const model::ModelEvaluator* ev = nullptr) {
  IdentificationReport r = base_report(map, options);
  if (!valid_options(options)) return r;
  // Random points cannot be drawn on a nonlinear constraint manifold.
  if (nonlinear_constraints || !pt.nonlinear_eq_rows.empty()) {
    r.reason = IdentificationReason::NonlinearConstraints;
    return r;
  }
  if (pt.has_inequality_constraints || pt.has_unenforced_constraints) {
    r.reason = IdentificationReason::ConstraintsUnavailable;
    return r;
  }
  // Two-level moments pair within and between blocks, with the means on one
  // level; no route supplies that map yet.
  if (pt.n_levels() > 1 || !jacobian) {
    r.map = IdentificationMap::None;
    r.reason = IdentificationReason::UnsupportedModel;
    return r;
  }
  const Eigen::Index n = pt.n_free();
  Reduction reduction;
  if (!reduction.init(con, n)) {
    r.reason = IdentificationReason::ConstraintsUnavailable;
    return r;
  }
  const Eigen::Index q = reduction.q();
  r.n_parameters = static_cast<std::int32_t>(q);
  if (q == 0) {
    r.status = IdentificationStatus::Identified;
    r.reason = IdentificationReason::NoFreeParameters;
    r.rank = 0;
    return r;
  }

  const double null_tol = options.null_tolerance;
  const double identified_tol = options.identified_tolerance;
  Draws draws{options.seed};
  std::vector<double> minima;
  std::optional<PointRank> best;
  Eigen::VectorXd best_theta;
  bool counted = false;
  for (std::int32_t p = 0; p < options.n_points; ++p) {
    std::optional<Eigen::MatrixXd> reduced;
    Eigen::VectorXd theta;
    for (int attempt = 0; attempt < draws_per_point && !reduced; ++attempt) {
      theta = draw_point(pt, reduction, draws);
      auto J = jacobian(theta);
      if (!J.has_value() || J->cols() != n || J->rows() == 0 || !J->allFinite())
        continue;
      if (!counted) {
        counted = true;
        r.n_moments = static_cast<std::int32_t>(J->rows());
        r.counting_rule = q <= J->rows();
      }
      reduced = reduction.reduce(*J);
    }
    if (!reduced) continue;
    PointRank point = rank_point(*reduced, null_tol, false);
    minima.push_back(point.min_relative);
    if (point.min_relative >= identified_tol) {
      // Full column rank at one point proves the generic rank.
      r.status = IdentificationStatus::Identified;
      r.reason = IdentificationReason::Rank;
      r.rank = static_cast<std::int32_t>(q);
      r.n_points = static_cast<std::int32_t>(minima.size());
      r.min_relative_singular_values =
          Eigen::Map<const Eigen::VectorXd>(minima.data(), static_cast<Eigen::Index>(minima.size()));
      r.smallest_singular_values = ascending_tail(point.relative, 3);
      return r;
    }
    if (!best || point.rank > best->rank) {
      best = std::move(point);
      best_theta = theta;
    }
  }
  r.n_points = static_cast<std::int32_t>(minima.size());
  r.min_relative_singular_values =
      Eigen::Map<const Eigen::VectorXd>(minima.data(), static_cast<Eigen::Index>(minima.size()));
  if (!best) {
    r.reason = IdentificationReason::EvaluationFailed;
    return r;
  }

  // Every point is rank deficient beyond null_tol. The generic rank is the
  // largest rank seen; it counts only if its smallest nonnull value clears
  // identified_tol, so that the gap separates it from the null values.
  r.rank = best->rank;
  const Eigen::Index nullity = q - best->rank;
  r.smallest_singular_values = ascending_tail(best->relative, nullity + 3);
  const bool clear_gap =
      best->rank == 0 || best->relative(best->rank - 1) >= identified_tol;
  if (!r.counting_rule) {
    // rank(J K) <= n_moments < q: no numerical tolerance is involved.
    r.status = IdentificationStatus::Unidentified;
    r.reason = IdentificationReason::CountingRule;
  } else if (nullity > 0 && clear_gap) {
    r.status = IdentificationStatus::Unidentified;
    r.reason = IdentificationReason::Rank;
  } else {
    r.reason = IdentificationReason::AmbiguousGap;
    return r;
  }

  if (nullity > 0) {
    std::optional<NullBasis> basis;
    if (estimate != nullptr && estimate->size() == n && estimate->allFinite()) {
      basis = null_basis_at(*estimate, reduction, jacobian, n, nullity, null_tol);
      r.directions_at_estimate = basis.has_value();
    }
    if (!basis)
      basis = null_basis_at(best_theta, reduction, jacobian, n, nullity, null_tol);
    if (basis) {
      r.null_directions = std::move(basis->directions);
      classify_directions(r, pt, reduction, basis->point,
                          r.directions_at_estimate ? *estimate : best_theta, ev);
    }
  }
  return r;
}

bool has_ordinal_structure(const spec::LatentStructure& pt) {
  for (const auto op : pt.op)
    if (op == parse::Op::Threshold || op == parse::Op::ResponseScale) return true;
  for (const auto& block : pt.ordinal_preparation)
    for (const auto kind : block)
      if (kind != 0) return true;
  return false;
}

bool has_mean_structure(const spec::LatentStructure& pt) {
  for (const auto op : pt.op)
    if (op == parse::Op::Intercept) return true;
  return false;
}

#ifdef MAGMAAN_ENABLE_TEST_PROBES
using Clock = std::chrono::steady_clock;
struct Timer {
  Clock::time_point start = Clock::now();
};
IdentificationReport observed(IdentificationReport r, const Timer& timer) {
  if (auto* notify = identification_test::observer()) {
    const std::chrono::duration<double> elapsed = Clock::now() - timer.start;
    notify(r, elapsed.count());
  }
  return r;
}
#else
struct Timer {};
IdentificationReport observed(IdentificationReport r, const Timer&) { return r; }
#endif

}  // namespace

IdentificationReport check_identification_rank(
    const spec::LatentStructure& pt, const EqConstraints& con,
    bool nonlinear_constraints, IdentificationMap map,
    const MomentJacobian& jacobian, const Eigen::VectorXd* estimate,
    IdentificationOptions options, const model::ModelEvaluator* evaluator) {
  const Timer timer{};
  return observed(rank_impl(pt, con, nonlinear_constraints, map, jacobian,
                            estimate, options, evaluator), timer);
}

IdentificationReport check_structural_identification(
    const spec::LatentStructure& pt, const model::ModelEvaluator& ev,
    const EqConstraints& con, bool nonlinear_constraints,
    const Eigen::VectorXd* estimate, IdentificationOptions options) {
  const auto& rep = ev.matrix_rep();
  const IdentificationMap map = has_mean_structure(pt)
      ? IdentificationMap::CovarianceMean : IdentificationMap::Covariance;
  const IdentificationOptions defaults;
  if (rep.identification && rep.identification_n_free == pt.n_free() &&
      !has_ordinal_structure(pt) &&
      (rep.identification->map == map || rep.identification->map == IdentificationMap::None) &&
      (!nonlinear_constraints ||
       rep.identification->reason == IdentificationReason::NonlinearConstraints) &&
      options.seed == defaults.seed && options.n_points == defaults.n_points &&
      options.null_tolerance == defaults.null_tolerance &&
      options.identified_tolerance == defaults.identified_tolerance)
    return *rep.identification;
  const Timer timer{};
  // The covariance map is the wrong map for ordinal models (their moments are
  // thresholds and correlations) and FC-SEM uses another evaluator; an empty
  // Jacobian reports them unsupported.
  MomentJacobian jacobian;
  if (!has_ordinal_structure(pt) &&
      pt.composite_mode != spec::CompositeMode::FcSem &&
      ev.n_free() == static_cast<std::size_t>(pt.n_free())) {
    jacobian = [&ev](const Eigen::VectorXd& theta)
        -> model_expected<Eigen::MatrixXd> {
      auto e = ev.evaluate(theta, true, true);
      if (!e.has_value()) return std::unexpected(e.error());
      if (e->J_mu.rows() == 0) return std::move(e->J_sigma);
      Eigen::MatrixXd J(e->J_sigma.rows() + e->J_mu.rows(), e->J_sigma.cols());
      J << e->J_sigma, e->J_mu;
      return J;
    };
  }
  return observed(rank_impl(pt, con, nonlinear_constraints, map, jacobian,
                            estimate, options, &ev), timer);
}

fit_expected<IdentificationReport> check_structural_identification(
    const spec::LatentStructure& pt_in, const model::MatrixRep& rep,
    const Eigen::VectorXd* estimate, IdentificationOptions options) {
  // Before data arrive, fixed.x moments have no values. They are constants of
  // the moment map, so any generic values serve; draw them like free
  // parameters, from a stream separate from the parameter points.
  spec::LatentStructure pt = pt_in;
  Draws exogenous{options.seed ^ 0x65786f67656e6f75ULL};
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (i < pt.exo.size() && pt.exo[i] != 0 && i < pt.fixed_value.size() &&
        (i >= pt.free.size() || pt.free[i] <= 0) && std::isnan(pt.fixed_value[i]))
      pt.fixed_value[i] = draw_value(pt, i, exogenous);
  }
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "check_structural_identification: " + ev.error().detail, 0, 0.0});
  }
  auto con = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con.has_value()) {
    const Timer timer{};
    IdentificationReport r = base_report(IdentificationMap::None, options);
    r.reason = pt.nonlinear_eq_rows.empty()
        ? IdentificationReason::ConstraintsUnavailable
        : IdentificationReason::NonlinearConstraints;
    return observed(std::move(r), timer);
  }
  return check_structural_identification(pt, *ev, *con,
                                         !pt.nonlinear_eq_rows.empty(),
                                         estimate, options);
}

std::vector<std::string> free_parameter_labels(const spec::LatentStructure& pt,
                                               const spec::LatentNames& names) {
  const std::int32_t n = pt.n_free();
  std::vector<std::string> out(static_cast<std::size_t>(n));
  const bool grouped = pt.n_groups() > 1;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const std::int32_t f = i < pt.free.size() ? pt.free[i] : 0;
    if (f <= 0 || f > n || !out[static_cast<std::size_t>(f - 1)].empty()) continue;
    std::string label = i < names.row_lhs.size() ? names.row_lhs[i] : std::string();
    if (pt.op[i] == parse::Op::Intercept) {
      label += "~1";
    } else {
      label += std::string(parse::to_string(pt.op[i]));
      if (i < names.row_rhs.size()) label += names.row_rhs[i];
    }
    if (grouped && i < pt.group.size() && pt.group[i] > 1)
      label += ".g" + std::to_string(pt.group[i]);
    out[static_cast<std::size_t>(f - 1)] = std::move(label);
  }
  for (std::int32_t k = 0; k < n; ++k)
    if (out[static_cast<std::size_t>(k)].empty())
      out[static_cast<std::size_t>(k)] = "theta[" + std::to_string(k + 1) + "]";
  return out;
}

std::vector<std::string> describe_null_directions(
    const IdentificationReport& report, const std::vector<std::string>& labels,
    double drop) {
  std::vector<std::string> out;
  const Eigen::MatrixXd& D = report.null_directions;
  for (Eigen::Index c = 0; c < D.cols(); ++c) {
    const double largest = D.col(c).cwiseAbs().maxCoeff();
    std::string text;
    if (!(largest > 0.0)) { out.push_back(text); continue; }
    for (Eigen::Index k = 0; k < D.rows(); ++k) {
      const double v = D(k, c) / largest;
      if (std::abs(v) < drop) continue;
      char number[32];
      std::snprintf(number, sizeof number, "%.3g", std::abs(v));
      const std::string name = static_cast<std::size_t>(k) < labels.size()
          ? labels[static_cast<std::size_t>(k)]
          : "theta[" + std::to_string(k + 1) + "]";
      if (!text.empty()) text += v < 0 ? " - " : " + ";
      else if (v < 0) text += "-";
      text += name + " (" + number + ")";
    }
    out.push_back(std::move(text));
  }
  return out;
}

}  // namespace magmaan::estimate::frontier
