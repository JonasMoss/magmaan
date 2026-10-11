#include "magmaan/estimate/start_pipeline.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/coordinates.hpp"
#include "magmaan/estimate/layered_start.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace magmaan::estimate {
namespace {
std::size_t index(auto value) { return static_cast<std::size_t>(value); }
}

const char* start_transport_reason(StartTransportIssue issue) {
  switch (issue) {
    case StartTransportIssue::ConstructorRequiresMarker: return "constructor-requires-marker";
    case StartTransportIssue::None: return "none";
    case StartTransportIssue::EqualityConstraints: return "equality-constraints-unsupported";
    case StartTransportIssue::NonlinearConstraints: return "nonlinear-constraints-unsupported";
    case StartTransportIssue::MixedScaleForms: return "mixed-scale-forms-unsupported";
    case StartTransportIssue::RegressionEqualities: return "regression-equalities-unsupported";
    case StartTransportIssue::GeneralLinearEqualities: return "general-linear-equalities-unsupported";
    case StartTransportIssue::TangentDimension: return "transport-tangent-dimension-mismatch";
    case StartTransportIssue::FixedValues: return "fixed-values-unsupported";
    case StartTransportIssue::MarkerLayout: return "marker-layout-unsupported";
    case StartTransportIssue::SourceModel: return "source-model-invalid";
    case StartTransportIssue::SourceStarts: return "source-starts-invalid";
    case StartTransportIssue::InvalidScale: return "marker-scale-zero-or-nonfinite";
    case StartTransportIssue::NonfiniteValues: return "transport-nonfinite";
    case StartTransportIssue::FixedValueMismatch: return "transport-fixed-value-mismatch";
    case StartTransportIssue::InvalidVector: return "source-vector-invalid";
  }
  return "unknown";
}

std::expected<StdLvStartTransport, StartTransportIssue> prepare_std_lv_transport(
    const spec::LatentStructure& pt, const model::MatrixRep& rep) {
  auto con = build_eq_constraints(pt);
  if (!pt.nl_constraints.empty() || !pt.nonlinear_eq_rows.empty())
    return std::unexpected(StartTransportIssue::NonlinearConstraints);
  if (!con) return std::unexpected(StartTransportIssue::EqualityConstraints);

  std::vector<std::vector<int>> markers, variances;
  for (const auto& d : rep.dims) {
    markers.emplace_back(index(d.n_latent), -1);
    variances.emplace_back(index(d.n_latent), -1);
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto c = rep.cell_for_row[i];
    if (!c.used) {
      if (pt.free[i] > 0) return std::unexpected(StartTransportIssue::MarkerLayout);
      continue;
    }
    if (c.mat == model::MatId::Psi && c.row == c.col && pt.free[i] > 0)
      variances[index(c.block)][index(c.col)] = static_cast<int>(i);
    if (pt.free[i] > 0 || pt.fixed_value[i] == 0) continue;
    if (c.mat != model::MatId::Lambda || pt.fixed_value[i] != 1 ||
        markers[index(c.block)][index(c.col)] >= 0)
      return std::unexpected(StartTransportIssue::FixedValues);
    markers[index(c.block)][index(c.col)] = static_cast<int>(i);
  }
  // Scale classes are ordered by block, then factor; union always retains the
  // first representative so the chart is deterministic across callers.
  std::vector<int> offsets(markers.size() + 1, 0);
  for (std::size_t b = 0; b < markers.size(); ++b)
    offsets[b + 1] = offsets[b] + static_cast<int>(markers[b].size());
  std::vector<int> parent(index(offsets.back()));
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](int j) { while (parent[index(j)] != j) j = parent[index(j)]; return j; };
  auto tie = [&](int j, int k) {
    j = root(j); k = root(k);
    parent[index(std::max(j, k))] = std::min(j, k);
  };
  struct Form { model::MatId kind; std::vector<int> coefficients; };
  std::vector<Form> forms(index(pt.n_free()), {model::MatId::Theta, std::vector<int>(parent.size(), 0)});
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    const auto c = rep.cell_for_row[i];
    if (!c.used) return std::unexpected(StartTransportIssue::MarkerLayout);
    auto& f = forms[index(pt.free[i] - 1)];
    f.kind = c.mat;
    const auto factor = [&](int j) { return index(offsets[index(c.block)] + j); };
    if (c.mat == model::MatId::Lambda) f.coefficients[factor(c.col)] = 1;
    if (c.mat == model::MatId::Alpha) f.coefficients[factor(c.row)] = -1;
    if (c.mat == model::MatId::Psi) {
      --f.coefficients[factor(c.row)]; --f.coefficients[factor(c.col)];
    }
    if (c.mat == model::MatId::Beta) {
      --f.coefficients[factor(c.row)]; ++f.coefficients[factor(c.col)];
    }
  }
  std::vector<std::vector<int>> equality_members;
  std::vector<int> group_leader(index(pt.n_free()), -1);
  const bool have_groups = pt.eq_groups.size() == index(pt.n_free());
  for (int j = 0; j < pt.n_free(); ++j) {
    const int group = have_groups ? pt.eq_groups[index(j)] : j;
    // Resolved merge IDs are contiguous and less than n_free().
    if (group < 0 || group >= pt.n_free()) return std::unexpected(StartTransportIssue::EqualityConstraints);
    auto& leader = group_leader[index(group)];
    if (leader < 0) { leader = static_cast<int>(equality_members.size()); equality_members.emplace_back(); }
    equality_members[index(leader)].push_back(j);
  }
  const std::size_t merge_count = equality_members.size();
  for (std::size_t r = 0; r < pt.lin_constraint_d.size(); ++r) {
    equality_members.emplace_back();
    for (int j = 0; j < pt.n_free(); ++j)
      if (pt.lin_constraint_R[r * index(pt.n_free()) + index(j)] != 0)
        equality_members.back().push_back(j);
  }
  // Only single-factor loading/mean ties introduce new scale identifications.
  // Covariance forms are accepted only if existing ties make them identical.
  for (const auto& members : equality_members) {
    if (members.size() < 2) continue;
    const auto& f = forms[index(members.front())];
    for (int member : members) {
      const auto& g = forms[index(member)];
      if (f.kind == model::MatId::Beta || g.kind == model::MatId::Beta)
        return std::unexpected(StartTransportIssue::RegressionEqualities);
      if (f.kind == g.kind && (f.kind == model::MatId::Lambda || f.kind == model::MatId::Alpha)) {
        int fj = 0, fk = 0;
        for (std::size_t l = 0; l < parent.size(); ++l) {
          if (f.coefficients[l]) fj = static_cast<int>(l);
          if (g.coefficients[l]) fk = static_cast<int>(l);
        }
        tie(fj, fk);
      }
    }
  }
  for (std::size_t r = 0; r < equality_members.size(); ++r) {
    const auto& members = equality_members[r];
    std::vector<int> form(parent.size(), 0);
    bool first_member = true;
    for (int member : members) {
      if (r >= merge_count && forms[index(member)].kind == model::MatId::Beta)
        return std::unexpected(StartTransportIssue::RegressionEqualities);
      std::vector<int> current(parent.size(), 0);
      for (std::size_t l = 0; l < parent.size(); ++l)
        current[index(root(static_cast<int>(l)))] += forms[index(member)].coefficients[l];
      if (!first_member && form != current)
        return std::unexpected(StartTransportIssue::MixedScaleForms);
      form = std::move(current); first_member = false;
    }
    if (r >= merge_count && pt.lin_constraint_d[r - merge_count] != 0 &&
        std::any_of(form.begin(), form.end(), [](int coefficient) { return coefficient != 0; }))
      return std::unexpected(StartTransportIssue::GeneralLinearEqualities);
  }
  auto standard = pt;
  std::vector<int> representative_markers(parent.size(), -1);
  for (std::size_t b = 0; b < markers.size(); ++b)
    for (std::size_t j = 0; j < markers[b].size(); ++j) {
      if (markers[b][j] < 0 || variances[b][j] < 0)
        return std::unexpected(StartTransportIssue::MarkerLayout);
      const int representative = root(offsets[b] + static_cast<int>(j));
      standard.free[index(markers[b][j])] = 1;
      if (representative == offsets[b] + static_cast<int>(j)) {
        standard.free[index(variances[b][j])] = 0;
        standard.fixed_value[index(variances[b][j])] = 1;
        representative_markers[index(representative)] = markers[b][j];
      }
    }
  int next = 0;
  std::vector<int> remap(index(pt.n_free()), -1);
  for (std::size_t i = 0; i < standard.size(); ++i) {
    if (standard.free[i] <= 0) continue;
    standard.free[i] = ++next;
    if (pt.free[i] > 0) remap[index(pt.free[i] - 1)] = next - 1;
  }
  standard.eq_groups.resize(index(next));
  std::iota(standard.eq_groups.begin(), standard.eq_groups.end(), 0);
  // Retain every original equality, including an equality whose representative
  // variance becomes fixed. Such a row becomes affine (the other variance is
  // now equal to one), rather than disappearing with the old free index.
  std::vector<double> fixed(index(pt.n_free()), 0.0);
  for (std::size_t i = 0; i < pt.size(); ++i)
    if (pt.free[i] > 0 && standard.free[i] == 0)
      fixed[index(pt.free[i] - 1)] = standard.fixed_value[i];
  standard.lin_constraint_R.assign(index(con->rank * next), 0.0);
  standard.lin_constraint_d.resize(index(con->rank));
  for (int r = 0; r < con->rank; ++r) {
    double rhs = con->b_eq(r);
    for (int j = 0; j < pt.n_free(); ++j) {
      const double coefficient = con->A_eq(r, j);
      if (remap[index(j)] >= 0)
        standard.lin_constraint_R[index(r * next + remap[index(j)])] += coefficient;
      else rhs -= coefficient * fixed[index(j)];
    }
    standard.lin_constraint_d[index(r)] = rhs;
  }
  std::vector<std::vector<int>> scale_markers = markers;
  for (std::size_t b = 0; b < markers.size(); ++b)
    for (std::size_t j = 0; j < markers[b].size(); ++j) {
      const int representative = root(offsets[b] + static_cast<int>(j));
      const int marker = representative_markers[index(representative)];
      scale_markers[b][j] = marker;
      standard.eq_groups[index(standard.free[index(markers[b][j])] - 1)] = standard.free[index(marker)] - 1;
    }
  // Ordered affine rows use the old free indices; native audit constraints use
  // eq_groups. Clear the preset-only projection rather than retain stale rows.
  standard.ordered_affine_R.clear(); standard.ordered_affine_d.clear();
  auto standard_con = build_eq_constraints(standard);
  if (!standard_con || standard_con->n_alpha != con->n_alpha)
    return std::unexpected(StartTransportIssue::TangentDimension);
  auto sr = model::build_matrix_rep(standard);
  if (!sr) return std::unexpected(StartTransportIssue::SourceModel);
  return StdLvStartTransport{pt, std::move(standard), rep, std::move(*sr), std::move(markers), std::move(scale_markers)};
}

std::expected<Eigen::VectorXd, StartTransportIssue> transport_start_values(
    const StdLvStartTransport& plan, const Eigen::VectorXd& source) {
  if (source.size() != plan.source.n_free() || !source.allFinite())
    return std::unexpected(StartTransportIssue::InvalidVector);
  const auto& pt = plan.target;
  const auto& rep = plan.target_rep;
  const auto& markers = plan.markers;
  auto ev = model::ModelEvaluator::build(plan.source, plan.source_rep);
  if (!ev) return std::unexpected(StartTransportIssue::SourceModel);
  auto assembled = ev->assembled(source);
  if (!assembled) return std::unexpected(StartTransportIssue::SourceStarts);
  std::vector<Eigen::VectorXd> scales;
  for (std::size_t b = 0; b < markers.size(); ++b) {
    Eigen::VectorXd a(static_cast<Eigen::Index>(markers[b].size()));
    for (Eigen::Index j = 0; j < a.size(); ++j) {
      const auto c = rep.cell_for_row[index(plan.scale_markers[b][index(j)])];
      a(j) = assembled->blocks[index(c.block)].Lambda(c.row, c.col);
      if (!std::isfinite(a(j)) || std::abs(a(j)) <= 1e-12) return std::unexpected(StartTransportIssue::InvalidScale);
    }
    scales.push_back(std::move(a));
  }
  for (std::size_t b = 0; b < markers.size(); ++b) {
    auto& m = assembled->blocks[b];
    const auto& a = scales[b];
    m.Lambda = (m.Lambda * a.cwiseInverse().asDiagonal()).eval();
    m.Beta = (a.asDiagonal() * m.Beta * a.cwiseInverse().asDiagonal()).eval();
    m.Psi = (a.asDiagonal() * m.Psi * a.asDiagonal()).eval();
    if (m.Alpha.size()) m.Alpha = (a.asDiagonal() * m.Alpha).eval();
  }
  Eigen::VectorXd candidate = Eigen::VectorXd::Zero(pt.n_free());
  for (std::size_t i = 0; i < pt.size(); ++i) {
    auto c = rep.cell_for_row[i];
    if (!c.used) continue;
    const auto& m = assembled->blocks[index(c.block)];
    double v = 0;
    switch (c.mat) {
      case model::MatId::Lambda: v = m.Lambda(c.row,c.col); break;
      case model::MatId::Beta: v = m.Beta(c.row,c.col); break;
      case model::MatId::Psi: v = m.Psi(c.row,c.col); break;
      case model::MatId::Theta: v = m.Theta(c.row,c.col); break;
      case model::MatId::Nu: v = m.Nu(c.row); break;
      case model::MatId::Alpha: v = m.Alpha(c.row); break;
    }
    if (!std::isfinite(v)) return std::unexpected(StartTransportIssue::NonfiniteValues);
    if (pt.free[i] > 0) candidate(pt.free[i]-1) = v;
    else if (std::abs(v-pt.fixed_value[i]) > 1e-8) return std::unexpected(StartTransportIssue::FixedValueMismatch);
  }
  return candidate;
}

fit_expected<Eigen::VectorXd> construct_start_values(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const data::SampleStats& samp, StartMethod method, const spec::Starts& hints,
    std::vector<std::string>* notes) {
  if (method == StartMethod::Layered) {
    auto report = layered_start_report(pt, rep, samp, hints);
    if (!report) return std::unexpected(report.error());
    if (notes) *notes = std::move(report->notes);
    if (report->theta.size() != pt.n_free() || !report->theta.allFinite())
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          "Start constructor returned an invalid vector", 0, 0});
    return std::move(report->theta);
  }
  auto out = method == StartMethod::Guttman ? guttman_start_values(pt, rep, samp, hints)
      : method == StartMethod::Bentler1982 ? bentler1982_start_values(pt, rep, samp, hints)
      : method == StartMethod::JamesStein ? jamesstein_start_values(pt, rep, samp, hints)
      : method == StartMethod::Simple ? simple_start_values(pt, rep, samp, hints)
      : fabin_start_values(pt, rep, samp, hints,
          method == StartMethod::Fabin2 ? FabinVariant::Fabin2 : FabinVariant::Fabin3);
  if (out && (out->size() != pt.n_free() || !out->allFinite()))
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "Start constructor returned an invalid vector", 0, 0});
  return out;
}

fit_expected<StartValues> explicit_start_values(
    const spec::LatentStructure& pt, const Eigen::VectorXd& theta) {
  if (theta.size() != pt.n_free() || !theta.allFinite())
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "Explicit start vector must be finite and match model n_free", 0, 0});
  StartValues out;
  out.theta = theta;
  out.explicit_vector = true;
  return out;
}

fit_expected<StartValues> start_values(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const data::SampleStats& samp, const StartPolicy& policy, const spec::Starts& hints) {
  StartTransportIssue reason = StartTransportIssue::None;
  if (policy.transport != StartTransport::Native && policy.method != StartMethod::Layered) {
    const bool needs_marker = policy.method == StartMethod::Guttman ||
        policy.method == StartMethod::Bentler1982 || policy.method == StartMethod::JamesStein;
    // TASK-162 owns adoption in automatic starts. Keep TASK-131's constrained
    // normalized-unit fallback even when the audit transport is available.
    auto original_con = build_eq_constraints(pt);
    const bool constrained = !original_con || original_con->active() || !pt.nl_constraints.empty();
    auto plan = needs_marker
        ? std::expected<StdLvStartTransport, StartTransportIssue>(
            std::unexpected(StartTransportIssue::ConstructorRequiresMarker))
        : constrained
        ? std::expected<StdLvStartTransport, StartTransportIssue>(
            std::unexpected(StartTransportIssue::EqualityConstraints))
        : prepare_std_lv_transport(pt, rep);
    if (!plan) reason = plan.error();
    else {
      auto source = construct_start_values(plan->source, plan->source_rep, samp, policy.method);
      if (!source) reason = StartTransportIssue::SourceStarts;
      else {
        auto candidate = transport_start_values(*plan, *source);
        if (!candidate) reason = candidate.error();
        else {
          for (Eigen::Index k = 0; k < candidate->size(); ++k)
            if (index(k) < hints.hint.size() && std::isfinite(hints.hint[index(k)]))
              (*candidate)(k) = hints.hint[index(k)];
          return StartValues{std::move(*candidate), StartBranch::TransportedStdLv,
                          StartTransportIssue::None, policy.method, policy.transport};
        }
      }
    }
    if (policy.transport == StartTransport::RequireStdLv)
      return std::unexpected(FitError{FitError::Kind::NumericIssue,
          start_transport_reason(reason), 0, 0});
  }
  std::vector<std::string> notes;
  // Equality constraints prevent the std.lv chart change, but not a change
  // of data units. Construct the fallback in normalized sample units so
  // the simple constructor's latent variance constants scale with the data.
  if (reason == StartTransportIssue::EqualityConstraints &&
      ml_normalization_supported(pt, rep)) {
    // A model the normalizer rejects keeps the plain native fallback below.
    if (auto normalized = normalize_ml_model(pt, rep, samp)) {
      auto scaled_hints = hints;
      for (std::size_t k = 0; k < scaled_hints.hint.size() &&
           k < static_cast<std::size_t>(normalized->parameter_units.size()); ++k)
        scaled_hints.hint[k] /= normalized->parameter_units(static_cast<Eigen::Index>(k));
      auto native = construct_start_values(normalized->structure, normalized->representation,
          normalized->sample, policy.method, scaled_hints, &notes);
      if (!native) return std::unexpected(native.error());
      native->array() *= normalized->parameter_units.array();
      StartValues out{std::move(*native), StartBranch::Native, reason, policy.method, policy.transport};
      out.notes = std::move(notes);
      return out;
    }
  }
  auto native = construct_start_values(pt, rep, samp, policy.method, hints, &notes);
  if (!native) return std::unexpected(native.error());
  StartValues out{std::move(*native), StartBranch::Native, reason, policy.method, policy.transport};
  out.notes = std::move(notes);
  return out;
}

fit_expected<StartValues> normalized_ml_start_values(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const data::SampleStats& samp, const StartPolicy& policy, const spec::Starts& hints) {
  if (!ml_normalization_supported(pt, rep)) return start_values(pt, rep, samp, policy, hints);
  auto normalized = normalize_ml_model(pt, rep, samp);
  if (!normalized) return std::unexpected(normalized.error());
  auto scaled_hints = hints;
  for (std::size_t k = 0; k < scaled_hints.hint.size() &&
       k < static_cast<std::size_t>(normalized->parameter_units.size()); ++k)
    scaled_hints.hint[k] /= normalized->parameter_units(static_cast<Eigen::Index>(k));
  auto out = start_values(normalized->structure, normalized->representation,
                          normalized->sample, policy, scaled_hints);
  if (out) out->theta.array() *= normalized->parameter_units.array();
  return out;
}

} // namespace magmaan::estimate
