#include "magmaan/fit/score.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/fit/concepts.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/inference.hpp"
#include "magmaan/fit/resolve_fixed_x.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

PostError fit_to_post(FitError e) {
  return make_err(PostError::Kind::NumericIssue, std::move(e.detail));
}

PostError model_to_post(ModelError e) {
  return make_err(PostError::Kind::NumericIssue, std::move(e.detail));
}

post_expected<double> total_n(const SampleStats& samp) {
  double n = 0.0;
  for (auto nb : samp.n_obs) n += static_cast<double>(nb);
  if (!(n > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: SampleStats has non-positive total n_obs"));
  }
  return n;
}

bool fixed_candidate_row(const partable::LatentStructure& pt,
                         const model::MatrixRep& rep,
                         std::size_t row) {
  if (row >= pt.size() || row >= rep.cell_for_row.size()) return false;
  if (pt.is_constraint_row(row)) return false;
  if (!rep.cell_for_row[row].used) return false;
  if (pt.free[row] != 0) return false;
  if (row < pt.exo.size() && pt.exo[row] != 0) return false;
  if (row < pt.fixed_value.size() && !std::isfinite(pt.fixed_value[row])) {
    return false;
  }
  return true;
}

void add_new_free_group(partable::LatentStructure& pt, std::int32_t old_n) {
  if (static_cast<std::int32_t>(pt.eq_groups.size()) == old_n) {
    pt.eq_groups.push_back(old_n);
  } else if (!pt.eq_groups.empty()) {
    pt.eq_groups.clear();
  }
}

post_expected<partable::LatentStructure>
with_fixed_row_freed(partable::LatentStructure pt, std::size_t row,
                     const SampleStats& samp, const model::MatrixRep& rep) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  if (!fixed_candidate_row(pt, rep, row)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: requested row is not an eligible fixed parameter"));
  }
  const std::int32_t old_n = pt.n_free();
  pt.free[row] = old_n + 1;
  pt.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
  add_new_free_group(pt, old_n);
  return pt;
}

Eigen::VectorXd append_theta(const Eigen::VectorXd& theta, double value) {
  Eigen::VectorXd out(theta.size() + 1);
  if (theta.size() > 0) out.head(theta.size()) = theta;
  out(theta.size()) = value;
  return out;
}

post_expected<Eigen::MatrixXd> invert_symmetric(const Eigen::MatrixXd& A,
                                                const char* what) {
  if (A.rows() != A.cols()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(what) + " is not square"));
  }
  if (A.rows() == 0) return Eigen::MatrixXd(0, 0);
  Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (A + A.transpose()));
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        std::string(what) + " is not positive definite"));
  }
  return Eigen::MatrixXd(ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols())));
}

post_expected<ScoreTestResult>
score_for_direction(const ScoreCandidate& candidate,
                    const Eigen::VectorXd& score_full,
                    const Eigen::MatrixXd& info_full,
                    const Eigen::MatrixXd& K_nuisance,
                    const Eigen::VectorXd& direction) {
  if (score_full.size() != info_full.rows() || info_full.rows() != info_full.cols() ||
      direction.size() != score_full.size() ||
      K_nuisance.rows() != score_full.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: incompatible score/information/direction shapes"));
  }

  const Eigen::VectorXd I_d = info_full * direction;
  double score_eff = direction.dot(score_full);
  double info_eff = direction.dot(I_d);

  if (K_nuisance.cols() > 0) {
    const Eigen::MatrixXd I_aa =
        K_nuisance.transpose() * info_full * K_nuisance;
    const Eigen::VectorXd I_ab = K_nuisance.transpose() * I_d;
    const Eigen::VectorXd score_a = K_nuisance.transpose() * score_full;
    auto Iaa_inv = invert_symmetric(I_aa, "score tests nuisance information");
    if (!Iaa_inv.has_value()) return std::unexpected(Iaa_inv.error());
    score_eff -= I_ab.dot((*Iaa_inv) * score_a);
    info_eff -= I_ab.dot((*Iaa_inv) * I_ab);
  }

  const double tol = 1e-10 * std::max<double>(1.0, std::abs(info_eff));
  if (!(info_eff > tol) || !std::isfinite(score_eff)) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        "score tests: efficient information is not positive"));
  }

  ScoreTestResult out;
  out.candidate = candidate;
  out.score = score_eff;
  out.information = info_eff;
  out.mi = (score_eff * score_eff) / info_eff;
  out.df = 1;
  out.p_value = chi2_pvalue(out.mi, 1);
  out.epc = score_eff / info_eff;
  return out;
}

post_expected<Eigen::MatrixXd>
null_space(const Eigen::MatrixXd& A, Eigen::Index n_cols) {
  if (A.cols() != n_cols) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: constraint matrix has incompatible column count"));
  }
  if (n_cols == 0) return Eigen::MatrixXd(0, 0);
  if (A.rows() == 0) return Eigen::MatrixXd::Identity(n_cols, n_cols);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  svd.setThreshold(1e-9);
  const Eigen::Index rank = svd.rank();
  return Eigen::MatrixXd(svd.matrixV().rightCols(n_cols - rank));
}

post_expected<Eigen::VectorXd>
release_direction(const EqConstraints& con, Eigen::Index release_row) {
  if (release_row < 0 || release_row >= con.A_eq.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: equality-release row is out of range"));
  }
  if (con.K().cols() >= con.npar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: model has no active equality constraints"));
  }

  Eigen::MatrixXd A_rel(con.A_eq.rows() - 1, con.A_eq.cols());
  Eigen::Index out = 0;
  for (Eigen::Index r = 0; r < con.A_eq.rows(); ++r) {
    if (r == release_row) continue;
    A_rel.row(out++) = con.A_eq.row(r);
  }
  auto K_rel = null_space(A_rel, con.npar);
  if (!K_rel.has_value()) return std::unexpected(K_rel.error());
  if (K_rel->cols() <= con.K().cols()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: equality release did not add a parameter direction"));
  }

  const Eigen::MatrixXd M = K_rel->transpose() * con.K();
  auto z_or = null_space(M.transpose(), K_rel->cols());
  if (!z_or.has_value()) return std::unexpected(z_or.error());
  if (z_or->cols() != 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: equality release is not one-dimensional"));
  }
  Eigen::VectorXd d = (*K_rel) * z_or->col(0);
  const double norm = d.norm();
  if (!(norm > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: equality-release direction is degenerate"));
  }
  d /= norm;
  return d;
}

post_expected<model::ModelEvaluator>
build_eval(const partable::LatentStructure& pt, const model::MatrixRep& rep) {
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    return std::unexpected(model_to_post(ev.error()));
  }
  return std::move(*ev);
}

post_expected<void>
evaluate_augmented_ml(const partable::LatentStructure& pt,
                      const model::MatrixRep& rep,
                      const SampleStats& samp,
                      const Estimates& est,
                      ML discrepancy,
                      ScoreInformation information,
                      double score_scale,
                      Eigen::VectorXd& score_full,
                      Eigen::MatrixXd& info_full) {
  auto ev = build_eval(pt, rep);
  if (!ev.has_value()) return std::unexpected(ev.error());
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(model_to_post(eval.error()));
  }
  auto cache = discrepancy.prepare(samp);
  if (!cache.has_value()) return std::unexpected(fit_to_post(cache.error()));
  auto vg = discrepancy.value_gradient(samp, *cache, eval->moments,
                                       eval->J_sigma, eval->J_mu);
  if (!vg.has_value()) return std::unexpected(fit_to_post(vg.error()));
  score_full = -score_scale * vg->gradient;
  auto info = information == ScoreInformation::Expected
      ? information_expected(pt, rep, samp, est)
      : information_observed_analytic(pt, rep, samp, est);
  if (!info.has_value()) return std::unexpected(info.error());
  info_full = *info;
  return {};
}

template <class D>
post_expected<void>
evaluate_augmented_ls(const partable::LatentStructure& pt,
                      const model::MatrixRep& rep,
                      const SampleStats& samp,
                      const Estimates& est,
                      D discrepancy,
                      double n_total,
                      Eigen::VectorXd& score_full,
                      Eigen::MatrixXd& info_full) {
  auto ev = build_eval(pt, rep);
  if (!ev.has_value()) return std::unexpected(ev.error());
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(model_to_post(eval.error()));
  }
  auto r = discrepancy.residuals(samp, eval->moments);
  if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
  auto J = discrepancy.residual_jacobian(samp, eval->moments,
                                         eval->J_sigma, eval->J_mu);
  if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));
  if (J->rows() != r->size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: LS residual/Jacobian shape mismatch"));
  }
  score_full = -n_total * (J->transpose() * *r);
  info_full = n_total * (J->transpose() * *J);
  info_full = 0.5 * (info_full + info_full.transpose());
  return {};
}

post_expected<void>
evaluate_augmented_fiml(const partable::LatentStructure& pt,
                        const model::MatrixRep& rep,
                        const RawData& raw,
                        const Estimates& est,
                        FIML discrepancy,
                        double h_step,
                        Eigen::VectorXd& score_full,
                        Eigen::MatrixXd& info_full) {
  if (!(h_step > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests FIML: h_step must be > 0"));
  }
  auto cache = discrepancy.prepare(raw);
  if (!cache.has_value()) return std::unexpected(fit_to_post(cache.error()));
  const double n_total = static_cast<double>(cache->n_total);
  if (!(n_total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests FIML: non-positive total n"));
  }
  auto ev = build_eval(pt, rep);
  if (!ev.has_value()) return std::unexpected(ev.error());

  auto gradient_at = [&](const Eigen::VectorXd& theta)
      -> post_expected<Eigen::VectorXd> {
    auto eval = ev->evaluate(theta, true, true);
    if (!eval.has_value()) return std::unexpected(model_to_post(eval.error()));
    auto vg = discrepancy.value_gradient(raw, *cache, eval->moments,
                                         eval->J_sigma, eval->J_mu);
    if (!vg.has_value()) return std::unexpected(fit_to_post(vg.error()));
    return vg->gradient;
  };

  auto g0 = gradient_at(est.theta);
  if (!g0.has_value()) return std::unexpected(g0.error());
  score_full = -0.5 * n_total * *g0;

  const Eigen::Index p = est.theta.size();
  info_full = Eigen::MatrixXd::Zero(p, p);
  for (Eigen::Index k = 0; k < p; ++k) {
    Eigen::VectorXd xp = est.theta;
    Eigen::VectorXd xm = est.theta;
    xp(k) += h_step;
    xm(k) -= h_step;
    auto gp = gradient_at(xp);
    if (!gp.has_value()) return std::unexpected(gp.error());
    auto gm = gradient_at(xm);
    if (!gm.has_value()) return std::unexpected(gm.error());
    info_full.col(k) = 0.5 * n_total * ((*gp - *gm) / (2.0 * h_step));
  }
  info_full = 0.5 * (info_full + info_full.transpose());
  return {};
}

template <class Evaluator>
post_expected<ScoreTestTable>
fixed_parameter_tests(partable::LatentStructure pt,
                      const model::MatrixRep& rep,
                      const Estimates& est,
                      Evaluator eval_score_info) {
  ScoreTestTable out;
  auto con0 = build_eq_constraints(pt);
  if (!con0.has_value()) return std::unexpected(con0.error());
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: fitted theta length does not match partable"));
  }

  for (std::size_t row = 0; row < pt.size(); ++row) {
    if (!fixed_candidate_row(pt, rep, row)) continue;
    const double fixed_value = pt.fixed_value[row];
    auto aug_or = eval_score_info.make_augmented(pt, row);
    if (!aug_or.has_value()) return std::unexpected(aug_or.error());
    partable::LatentStructure aug_pt = std::move(*aug_or);
    Estimates aug_est{append_theta(est.theta, fixed_value), est.fmin, est.iterations};

    Eigen::VectorXd score_full;
    Eigen::MatrixXd info_full;
    if (auto e = eval_score_info.evaluate(aug_pt, aug_est, score_full, info_full);
        !e.has_value()) {
      return std::unexpected(e.error());
    }

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(score_full.size());
    direction(score_full.size() - 1) = 1.0;
    ScoreCandidate cand;
    cand.kind = ScoreCandidateKind::FixedParam;
    cand.row = row;
    cand.op = pt.op[row];
    cand.lhs_var = pt.lhs_var[row];
    cand.rhs_var = pt.rhs_var[row];
    cand.group = pt.group[row];
    Eigen::MatrixXd K_aug = Eigen::MatrixXd::Zero(score_full.size(),
                                                  con0->K().cols());
    if (con0->K().rows() > 0) {
      K_aug.topRows(con0->K().rows()) = con0->K();
    }
    auto r = score_for_direction(cand, score_full, info_full, K_aug, direction);
    if (r.has_value()) out.rows.push_back(*r);
  }
  return out;
}

template <class Evaluator>
post_expected<ScoreTestTable>
equality_release_tests(partable::LatentStructure pt,
                       const model::MatrixRep& rep,
                       const Estimates& est,
                       Evaluator eval_score_info) {
  (void)rep;
  ScoreTestTable out;
  auto con = build_eq_constraints(pt);
  if (!con.has_value()) return std::unexpected(con.error());
  if (!con->active()) return out;
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "score tests: fitted theta length does not match partable"));
  }

  Eigen::VectorXd score_full;
  Eigen::MatrixXd info_full;
  if (auto e = eval_score_info.evaluate(pt, est, score_full, info_full);
      !e.has_value()) {
    return std::unexpected(e.error());
  }

  for (Eigen::Index r = 0; r < con->A_eq.rows(); ++r) {
    auto d = release_direction(*con, r);
    if (!d.has_value()) return std::unexpected(d.error());
    ScoreCandidate cand;
    cand.kind = ScoreCandidateKind::EqualityRelease;
    cand.row = static_cast<std::size_t>(r);
    cand.op = parse::Op::EqConstraint;
    auto res = score_for_direction(cand, score_full, info_full, con->K(), *d);
    if (res.has_value()) out.rows.push_back(*res);
  }
  return out;
}

struct ContinuousMlEvaluator {
  const model::MatrixRep& rep;
  const SampleStats& samp;
  ML discrepancy;
  ScoreInformation information;
  double score_scale;

  post_expected<partable::LatentStructure>
  make_augmented(partable::LatentStructure pt, std::size_t row) const {
    return with_fixed_row_freed(std::move(pt), row, samp, rep);
  }

  post_expected<void>
  evaluate(const partable::LatentStructure& pt, const Estimates& est,
           Eigen::VectorXd& score, Eigen::MatrixXd& info) const {
    return evaluate_augmented_ml(pt, rep, samp, est, discrepancy, information,
                                 score_scale, score, info);
  }
};

template <class D>
struct ContinuousLsEvaluator {
  const model::MatrixRep& rep;
  const SampleStats& samp;
  D discrepancy;
  double n_total;

  post_expected<partable::LatentStructure>
  make_augmented(partable::LatentStructure pt, std::size_t row) const {
    return with_fixed_row_freed(std::move(pt), row, samp, rep);
  }

  post_expected<void>
  evaluate(const partable::LatentStructure& pt, const Estimates& est,
           Eigen::VectorXd& score, Eigen::MatrixXd& info) const {
    return evaluate_augmented_ls(pt, rep, samp, est, discrepancy, n_total,
                                 score, info);
  }
};

struct FimlEvaluator {
  const model::MatrixRep& rep;
  const RawData& raw;
  FIML discrepancy;
  double h_step;

  post_expected<partable::LatentStructure>
  make_augmented(partable::LatentStructure pt, std::size_t row) const {
    auto start_samp = fiml_start_sample_stats(raw);
    if (!start_samp.has_value()) return std::unexpected(fit_to_post(start_samp.error()));
    return with_fixed_row_freed(std::move(pt), row, *start_samp, rep);
  }

  post_expected<void>
  evaluate(const partable::LatentStructure& pt, const Estimates& est,
           Eigen::VectorXd& score, Eigen::MatrixXd& info) const {
    return evaluate_augmented_fiml(pt, rep, raw, est, discrepancy, h_step,
                                   score, info);
  }
};

}  // namespace

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     ML discrepancy,
                     ScoreInformation information) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousMlEvaluator ev{rep, samp, discrepancy, information, 0.5 * *n};
  return fixed_parameter_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            ML discrepancy,
            ScoreInformation information) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousMlEvaluator ev{rep, samp, discrepancy, information, 0.5 * *n};
  return equality_release_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     ULS discrepancy,
                     ScoreInformation) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousLsEvaluator<ULS> ev{rep, samp, discrepancy, *n};
  return fixed_parameter_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            ULS discrepancy,
            ScoreInformation) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousLsEvaluator<ULS> ev{rep, samp, discrepancy, *n};
  return equality_release_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     GLS discrepancy,
                     ScoreInformation) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousLsEvaluator<GLS> ev{rep, samp, discrepancy, *n};
  return fixed_parameter_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            GLS discrepancy,
            ScoreInformation) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousLsEvaluator<GLS> ev{rep, samp, discrepancy, *n};
  return equality_release_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
modification_indices(partable::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const SampleStats& samp,
                     const Estimates& est,
                     WLS discrepancy,
                     ScoreInformation) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousLsEvaluator<WLS> ev{rep, samp, std::move(discrepancy), *n};
  return fixed_parameter_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
score_tests(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const SampleStats& samp,
            const Estimates& est,
            WLS discrepancy,
            ScoreInformation) {
  auto n = total_n(samp);
  if (!n.has_value()) return std::unexpected(n.error());
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  ContinuousLsEvaluator<WLS> ev{rep, samp, std::move(discrepancy), *n};
  return equality_release_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
modification_indices_fiml(partable::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const RawData& raw,
                          const Estimates& est,
                          FIML discrepancy,
                          double h_step) {
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto start_samp = fiml_start_sample_stats(raw);
  if (!start_samp.has_value()) return std::unexpected(fit_to_post(start_samp.error()));
  if (auto e = resolve_fixed_x_from_sample(pt, rep, *start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  FimlEvaluator ev{rep, raw, discrepancy, h_step};
  return fixed_parameter_tests(std::move(pt), rep, est, ev);
}

post_expected<ScoreTestTable>
score_tests_fiml(partable::LatentStructure pt,
                 const model::MatrixRep& rep,
                 const RawData& raw,
                 const Estimates& est,
                 FIML discrepancy,
                 double h_step) {
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  auto start_samp = fiml_start_sample_stats(raw);
  if (!start_samp.has_value()) return std::unexpected(fit_to_post(start_samp.error()));
  if (auto e = resolve_fixed_x_from_sample(pt, rep, *start_samp); !e.has_value()) {
    return std::unexpected(fit_to_post(e.error()));
  }
  FimlEvaluator ev{rep, raw, discrepancy, h_step};
  return equality_release_tests(std::move(pt), rep, est, ev);
}

}  // namespace magmaan::fit
