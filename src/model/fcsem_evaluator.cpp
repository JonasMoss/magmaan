#include "magmaan/model/fcsem_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::model {

namespace {

ModelError make_err(ModelError::Kind k, std::string detail) {
  return ModelError{k, std::move(detail)};
}

bool is_constraint_op(parse::Op op) noexcept {
  return op == parse::Op::EqConstraint || op == parse::Op::LtConstraint ||
         op == parse::Op::GtConstraint || op == parse::Op::DefineParam;
}

bool is_composite_var(const spec::LatentStructure& pt,
                      std::int32_t var) noexcept {
  for (const auto& c : pt.composite_blocks)
    if (c.composite_var == var) return true;
  return false;
}

bool same_composite_indicator_block(const spec::LatentStructure& pt,
                                    std::int32_t lhs,
                                    std::int32_t rhs) noexcept {
  for (const auto& c : pt.composite_blocks) {
    bool has_lhs = false;
    bool has_rhs = false;
    for (auto v : c.indicator_vars) {
      if (v == lhs) has_lhs = true;
      if (v == rhs) has_rhs = true;
    }
    if (has_lhs && has_rhs) return true;
  }
  return false;
}

std::int32_t ov_pos(const spec::LatentStructure& pt, std::int32_t var) noexcept {
  if (var < 0 || var >= pt.n_vars) return -1;
  return pt.ov_pos[static_cast<std::size_t>(var)];
}

std::int32_t lv_pos(const spec::LatentStructure& pt, std::int32_t var) noexcept {
  if (var < 0 || var >= pt.n_vars) return -1;
  return pt.lv_ext_pos[static_cast<std::size_t>(var)];
}

bool is_user_lv(const spec::LatentStructure& pt, std::int32_t var) noexcept {
  return var >= 0 && var < pt.n_vars &&
         pt.is_user_latent[static_cast<std::size_t>(var)] != 0;
}

model_expected<double>
row_value(const spec::LatentStructure& pt,
          std::size_t                  row,
          Eigen::Ref<const Eigen::VectorXd> theta) {
  const std::int32_t free = pt.free[row];
  if (free > 0) {
    const Eigen::Index k = static_cast<Eigen::Index>(free - 1);
    if (k < 0 || k >= theta.size()) {
      return std::unexpected(make_err(
          ModelError::Kind::UnknownVariable,
          "free parameter index is outside theta"));
    }
    return theta(k);
  }
  const double v = pt.fixed_value[row];
  if (!std::isfinite(v)) {
    return std::unexpected(make_err(
        ModelError::Kind::UnsupportedRowKind,
        "fixed row has no finite value in the native FC-SEM evaluator"));
  }
  return v;
}

void set_symmetric(Eigen::MatrixXd& M, Eigen::Index r, Eigen::Index c,
                   double value) {
  M(r, c) = value;
  if (r != c) M(c, r) = value;
}

inline Eigen::Index vech_index(Eigen::Index p, Eigen::Index r,
                               Eigen::Index c) noexcept {
  if (r < c) std::swap(r, c);
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

inline Eigen::Index vech_len(Eigen::Index p) noexcept {
  return p * (p + 1) / 2;
}

struct FcSemEvaluation {
  ImpliedMoments moments;
  std::vector<Eigen::MatrixXd> construct_covariance;
};

model_expected<FcSemEvaluation>
evaluate_fcsem(const spec::LatentStructure& pt, std::size_t n_free,
               std::size_t n_blocks, Eigen::Index p_, Eigen::Index m_,
               const data::SampleStats& samp,
               Eigen::Ref<const Eigen::VectorXd> theta) {
  if (static_cast<std::size_t>(theta.size()) != n_free) {
    return std::unexpected(make_err(
        ModelError::Kind::UnknownVariable,
        std::string("theta has size ") + std::to_string(theta.size()) +
            "; FcSemEvaluator expects " + std::to_string(n_free)));
  }
  if (samp.S.size() != n_blocks) {
    return std::unexpected(make_err(
        ModelError::Kind::EmptyMatrix,
        "sample covariance block count does not match the model"));
  }

  FcSemEvaluation out;
  out.moments.sigma.reserve(n_blocks);
  out.moments.mu.reserve(n_blocks);
  out.construct_covariance.reserve(n_blocks);

  std::vector<Eigen::Index> derived_idx;
  derived_idx.reserve(pt.composite_blocks.size());
  for (const auto& c : pt.composite_blocks) {
    derived_idx.push_back(static_cast<Eigen::Index>(lv_pos(pt, c.composite_var)));
  }

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const auto& S = samp.S[b];
    if (S.rows() != p_ || S.cols() != p_) {
      return std::unexpected(make_err(
          ModelError::Kind::EmptyMatrix,
          "sample covariance dimensions do not match the observed ordering"));
    }

    Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(p_, m_);
    Eigen::MatrixXd W      = Eigen::MatrixXd::Zero(p_, m_);
    Eigen::MatrixXd Theta  = Eigen::MatrixXd::Zero(p_, p_);
    Eigen::MatrixXd Psi    = Eigen::MatrixXd::Zero(m_, m_);
    Eigen::MatrixXd Beta   = Eigen::MatrixXd::Zero(m_, m_);

    // Phantom observed variables in the extended latent set keep their usual
    // identity measurement row, matching the Reduced LISREL path.
    for (auto v : pt.lv_ext_order) {
      const std::size_t vi = static_cast<std::size_t>(v);
      if (pt.is_user_latent[vi] != 0) continue;
      const std::int32_t oi = ov_pos(pt, v);
      const std::int32_t li = lv_pos(pt, v);
      if (oi >= 0 && li >= 0) Lambda(oi, li) = 1.0;
    }

    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (is_constraint_op(pt.op[i])) continue;
      if (pt.group[i] != static_cast<std::int32_t>(b + 1)) continue;

      const std::int32_t L = pt.lhs_var[i];
      const std::int32_t R = pt.rhs_var[i];

      switch (pt.op[i]) {
        case parse::Op::Composite: {
          const std::int32_t r = ov_pos(pt, R);
          const std::int32_t c = lv_pos(pt, L);
          if (r < 0 || c < 0) {
            return std::unexpected(make_err(
                ModelError::Kind::UnknownVariable,
                "native composite row references an unknown variable"));
          }
          auto v = row_value(pt, i, theta);
          if (!v.has_value()) return std::unexpected(v.error());
          W(r, c) = *v;
          break;
        }
        case parse::Op::Measurement: {
          auto v = row_value(pt, i, theta);
          if (!v.has_value()) return std::unexpected(v.error());
          if (is_user_lv(pt, R)) {
            const std::int32_t row = lv_pos(pt, R);
            const std::int32_t col = lv_pos(pt, L);
            if (row < 0 || col < 0) {
              return std::unexpected(make_err(
                  ModelError::Kind::UnknownVariable,
                  "higher-order measurement row references an unknown latent"));
            }
            Beta(row, col) = *v;
          } else {
            const std::int32_t row = ov_pos(pt, R);
            const std::int32_t col = lv_pos(pt, L);
            if (row < 0 || col < 0) {
              return std::unexpected(make_err(
                  ModelError::Kind::UnknownVariable,
                  "measurement row references an unknown variable"));
            }
            Lambda(row, col) = *v;
          }
          break;
        }
        case parse::Op::Regression: {
          const std::int32_t row = lv_pos(pt, L);
          const std::int32_t col = lv_pos(pt, R);
          if (row < 0 || col < 0) {
            return std::unexpected(make_err(
                ModelError::Kind::UnsupportedRowKind,
                "native FC-SEM currently requires regression endpoints to be "
                "constructs or reduced-form observed variables"));
          }
          auto v = row_value(pt, i, theta);
          if (!v.has_value()) return std::unexpected(v.error());
          Beta(row, col) = *v;
          break;
        }
        case parse::Op::Covariance: {
          if (same_composite_indicator_block(pt, L, R)) {
            if (pt.free[i] != 0) {
              return std::unexpected(make_err(
                  ModelError::Kind::UnsupportedRowKind,
                  "free composite-indicator T rows are not supported yet"));
            }
            break;  // T is read from the sample covariance block below.
          }

          const std::int32_t l = lv_pos(pt, L);
          const std::int32_t r = lv_pos(pt, R);
          if (l >= 0 || r >= 0) {
            if (l < 0 || r < 0) {
              return std::unexpected(make_err(
                  ModelError::Kind::UnsupportedRowKind,
                  "native FC-SEM mixed observed/construct covariance rows are "
                  "not supported yet"));
            }
            if (L == R && is_composite_var(pt, L)) {
              break;  // derived from W' T W and the structural model.
            }
            auto v = row_value(pt, i, theta);
            if (!v.has_value()) return std::unexpected(v.error());
            set_symmetric(Psi, l, r, *v);
          } else {
            const std::int32_t lo = ov_pos(pt, L);
            const std::int32_t ro = ov_pos(pt, R);
            if (lo < 0 || ro < 0) {
              return std::unexpected(make_err(
                  ModelError::Kind::UnknownVariable,
                  "observed covariance row references an unknown variable"));
            }
            auto v = row_value(pt, i, theta);
            if (!v.has_value()) return std::unexpected(v.error());
            set_symmetric(Theta, lo, ro, *v);
          }
          break;
        }
        case parse::Op::Intercept:
          return std::unexpected(make_err(
              ModelError::Kind::UnsupportedRowKind,
              "native FC-SEM evaluator is covariance-only for now"));
        default:
          return std::unexpected(make_err(
              ModelError::Kind::UnsupportedRowKind,
              "row kind is not supported by the native FC-SEM evaluator"));
      }
    }

    std::vector<double> target_var(pt.composite_blocks.size(), 0.0);
    for (std::size_t ci = 0; ci < pt.composite_blocks.size(); ++ci) {
      const auto& cb = pt.composite_blocks[ci];
      const Eigen::Index comp = static_cast<Eigen::Index>(lv_pos(pt, cb.composite_var));
      const Eigen::Index k = static_cast<Eigen::Index>(cb.indicator_vars.size());
      Eigen::VectorXd w(k);
      Eigen::MatrixXd T(k, k);
      for (Eigen::Index r = 0; r < k; ++r) {
        const Eigen::Index orow =
            static_cast<Eigen::Index>(ov_pos(pt, cb.indicator_vars[static_cast<std::size_t>(r)]));
        w(r) = W(orow, comp);
        for (Eigen::Index c = 0; c < k; ++c) {
          const Eigen::Index ocol =
              static_cast<Eigen::Index>(ov_pos(pt, cb.indicator_vars[static_cast<std::size_t>(c)]));
          T(r, c) = S(orow, ocol);
        }
      }
      const double v = (w.transpose() * T * w)(0, 0);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(
            ModelError::Kind::NonPositiveDefinite,
            "composite variance W' T W is not positive"));
      }
      target_var[ci] = v;
      const Eigen::VectorXd lambda = (T * w) / v;
      for (Eigen::Index r = 0; r < k; ++r) {
        const Eigen::Index orow =
            static_cast<Eigen::Index>(ov_pos(pt, cb.indicator_vars[static_cast<std::size_t>(r)]));
        Lambda(orow, comp) = lambda(r);
      }
      const Eigen::MatrixXd theta_c = T - lambda * (v * lambda.transpose());
      for (Eigen::Index r = 0; r < k; ++r) {
        const Eigen::Index orow =
            static_cast<Eigen::Index>(ov_pos(pt, cb.indicator_vars[static_cast<std::size_t>(r)]));
        for (Eigen::Index c = 0; c < k; ++c) {
          const Eigen::Index ocol =
              static_cast<Eigen::Index>(ov_pos(pt, cb.indicator_vars[static_cast<std::size_t>(c)]));
          Theta(orow, ocol) = theta_c(r, c);
        }
      }
    }

    Eigen::MatrixXd ImB = -Beta;
    ImB.diagonal().array() += 1.0;
    Eigen::FullPivLU<Eigen::MatrixXd> imb_lu(ImB);
    if (!imb_lu.isInvertible()) {
      return std::unexpected(make_err(
          ModelError::Kind::NonPositiveDefinite,
          "I - B is singular in native FC-SEM evaluator"));
    }
    const Eigen::MatrixXd A = imb_lu.inverse();

    Eigen::MatrixXd base_psi = Psi;
    for (Eigen::Index idx : derived_idx) base_psi(idx, idx) = 0.0;
    const Eigen::MatrixXd base_veta = A * base_psi * A.transpose();

    const Eigen::Index q = static_cast<Eigen::Index>(derived_idx.size());
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(q, q);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(q);
    for (Eigen::Index r = 0; r < q; ++r) {
      const Eigen::Index target = derived_idx[static_cast<std::size_t>(r)];
      rhs(r) = target_var[static_cast<std::size_t>(r)] - base_veta(target, target);
      for (Eigen::Index c = 0; c < q; ++c) {
        const Eigen::Index source = derived_idx[static_cast<std::size_t>(c)];
        C(r, c) = A(target, source) * A(target, source);
      }
    }
    Eigen::FullPivLU<Eigen::MatrixXd> c_lu(C);
    if (!c_lu.isInvertible()) {
      return std::unexpected(make_err(
          ModelError::Kind::NonPositiveDefinite,
          "derived composite variance system is singular"));
    }
    const Eigen::VectorXd derived = c_lu.solve(rhs);
    for (Eigen::Index j = 0; j < q; ++j) {
      if (!std::isfinite(derived(j))) {
        return std::unexpected(make_err(
            ModelError::Kind::NonPositiveDefinite,
            "derived composite disturbance variance is not finite"));
      }
      Psi(derived_idx[static_cast<std::size_t>(j)],
          derived_idx[static_cast<std::size_t>(j)]) = derived(j);
    }

    const Eigen::MatrixXd Veta = A * Psi * A.transpose();
    Eigen::MatrixXd Sigma = Lambda * Veta * Lambda.transpose() + Theta;
    Sigma = 0.5 * (Sigma + Sigma.transpose());
    out.construct_covariance.push_back(0.5 * (Veta + Veta.transpose()));
    out.moments.sigma.push_back(std::move(Sigma));
    out.moments.mu.emplace_back();
  }

  return out;
}

}  // namespace

model_expected<FcSemEvaluator>
FcSemEvaluator::build(const spec::LatentStructure& pt) {
  if (pt.composite_mode != spec::CompositeMode::FcSem ||
      pt.composite_blocks.empty()) {
    return std::unexpected(make_err(
        ModelError::Kind::UnsupportedRowKind,
        "FcSemEvaluator requires a native FC-SEM composite model"));
  }
  if (pt.ov_order.empty() || pt.lv_ext_order.empty()) {
    return std::unexpected(make_err(
        ModelError::Kind::EmptyMatrix,
        "native FC-SEM model has no observed or construct variables"));
  }
  for (const auto& c : pt.composite_blocks) {
    if (lv_pos(pt, c.composite_var) < 0) {
      return std::unexpected(make_err(
          ModelError::Kind::UnknownVariable,
          "composite construct is not in the extended latent ordering"));
    }
    if (c.indicator_vars.size() < 2) {
      return std::unexpected(make_err(
          ModelError::Kind::UnsupportedRowKind,
          "native FC-SEM composite has fewer than two indicators"));
    }
    for (auto v : c.indicator_vars) {
      if (ov_pos(pt, v) < 0) {
        return std::unexpected(make_err(
            ModelError::Kind::UnknownVariable,
            "composite indicator is not in the observed ordering"));
      }
    }
  }

  FcSemEvaluator out;
  out.pt_ = &pt;
  out.n_free_ = static_cast<std::size_t>(pt.n_free());
  out.n_blocks_ = static_cast<std::size_t>(pt.n_groups());
  out.p_ = static_cast<Eigen::Index>(pt.ov_order.size());
  out.m_ = static_cast<Eigen::Index>(pt.lv_ext_order.size());
  return out;
}

model_expected<ImpliedMoments>
FcSemEvaluator::sigma(const data::SampleStats& samp,
                      Eigen::Ref<const Eigen::VectorXd> theta) const {
  auto eval = evaluate_fcsem(*pt_, n_free_, n_blocks_, p_, m_, samp, theta);
  if (!eval.has_value()) return std::unexpected(eval.error());
  return std::move(eval->moments);
}

model_expected<std::vector<Eigen::MatrixXd>>
FcSemEvaluator::construct_covariance(
    const data::SampleStats& samp,
    Eigen::Ref<const Eigen::VectorXd> theta) const {
  auto eval = evaluate_fcsem(*pt_, n_free_, n_blocks_, p_, m_, samp, theta);
  if (!eval.has_value()) return std::unexpected(eval.error());
  return std::move(eval->construct_covariance);
}

model_expected<Eigen::MatrixXd>
FcSemEvaluator::dsigma_dtheta(const data::SampleStats& samp,
                              Eigen::Ref<const Eigen::VectorXd> theta,
                              double rel_step) const {
  if (!(rel_step > 0.0) || !std::isfinite(rel_step)) {
    return std::unexpected(make_err(
        ModelError::Kind::UnsupportedRowKind,
        "FC-SEM covariance Jacobian step must be positive and finite"));
  }
  auto base = sigma(samp, theta);
  if (!base.has_value()) return std::unexpected(base.error());

  Eigen::Index total_vech = 0;
  for (const auto& Sig : base->sigma) {
    if (Sig.rows() != Sig.cols()) {
      return std::unexpected(make_err(
          ModelError::Kind::EmptyMatrix,
          "FC-SEM implied covariance block is not square"));
    }
    total_vech += vech_len(Sig.rows());
  }

  Eigen::MatrixXd J =
      Eigen::MatrixXd::Zero(total_vech, static_cast<Eigen::Index>(n_free_));
  if (n_free_ == 0) return J;

  for (std::size_t k = 0; k < n_free_; ++k) {
    const Eigen::Index col = static_cast<Eigen::Index>(k);
    const double h = rel_step * std::max(1.0, std::abs(theta(col)));
    Eigen::VectorXd plus = theta;
    Eigen::VectorXd minus = theta;
    plus(col) += h;
    minus(col) -= h;

    auto sp = sigma(samp, plus);
    if (!sp.has_value()) return std::unexpected(sp.error());
    auto sm = sigma(samp, minus);
    if (!sm.has_value()) return std::unexpected(sm.error());
    if (sp->sigma.size() != base->sigma.size() ||
        sm->sigma.size() != base->sigma.size()) {
      return std::unexpected(make_err(
          ModelError::Kind::EmptyMatrix,
          "FC-SEM covariance Jacobian block count changed under perturbation"));
    }

    Eigen::Index off = 0;
    for (std::size_t b = 0; b < base->sigma.size(); ++b) {
      const Eigen::Index p = base->sigma[b].rows();
      if (sp->sigma[b].rows() != p || sp->sigma[b].cols() != p ||
          sm->sigma[b].rows() != p || sm->sigma[b].cols() != p) {
        return std::unexpected(make_err(
            ModelError::Kind::EmptyMatrix,
            "FC-SEM covariance Jacobian block shape changed under "
            "perturbation"));
      }
      const Eigen::MatrixXd dS = (sp->sigma[b] - sm->sigma[b]) / (2.0 * h);
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          J(off + vech_index(p, r, c), col) = dS(r, c);
        }
      }
      off += vech_len(p);
    }
  }
  return J;
}

}  // namespace magmaan::model
