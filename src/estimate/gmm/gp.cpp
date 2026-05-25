#include "magmaan/estimate/gmm/gp.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate::gmm {

namespace {

using estimate::build_eq_constraints;
using estimate::EqConstraints;

FitError fit_err(FitError::Kind kind, std::string detail) {
  return FitError{kind, std::move(detail), 0, 0.0};
}

// Golub–Pereyra splits the free parameters into two blocks: the *nonlinear*
// loadings/structural coefficients (Λ, Β) that drive the outer optimization,
// and the *conditionally-linear* (co)variances/intercepts (Θ, Ψ, ν, α) that an
// inner linear least-squares solves for at each outer step.
enum class BlockKind : std::uint8_t { None, Nonlinear, Linear };

BlockKind block_kind_for(model::MatId mat) noexcept {
  switch (mat) {
    case model::MatId::Lambda:
    case model::MatId::Beta:
      return BlockKind::Nonlinear;
    case model::MatId::Theta:
    case model::MatId::Psi:
    case model::MatId::Nu:
    case model::MatId::Alpha:
      return BlockKind::Linear;
  }
  return BlockKind::None;
}

std::string_view block_kind_name(BlockKind k) noexcept {
  switch (k) {
    case BlockKind::None: return "none";
    case BlockKind::Nonlinear: return "nonlinear";
    case BlockKind::Linear: return "linear";
  }
  return "?";
}

// The β/α split: which equality-constraint basis columns carry the nonlinear
// parameters, which carry the linear ones, plus the affine reparameterization
// pieces (θ₀, K_β, K_α) restricted to each block.
struct Classification {
  EqConstraints constraints;
  std::vector<Eigen::Index> beta_cols;
  std::vector<Eigen::Index> alpha_cols;
  Eigen::VectorXd beta0;
  Eigen::MatrixXd K_beta;
  Eigen::MatrixXd K_alpha;
};

// Classify the free parameters into the nonlinear (β) and linear (α) blocks.
// Fails (NumericIssue) when the model is not separable: an unsupported matrix
// location, an equality constraint that mixes the two kinds, or no linear
// parameter to profile.
fit_expected<Classification>
classify(const spec::LatentStructure& pt, const model::ModelEvaluator& ev,
         const Eigen::VectorXd& theta_start) {
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS compatibility: constraint: " + con_or.error().detail));
  }
  EqConstraints con = std::move(*con_or);
  const auto locs = ev.param_locations();
  if (static_cast<std::int32_t>(locs.size()) != pt.n_free()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS compatibility: parameter-location count does not match n_free"));
  }

  Classification out;
  out.constraints = std::move(con);
  std::vector<BlockKind> full_kind(locs.size(), BlockKind::None);
  for (std::size_t k = 0; k < locs.size(); ++k) {
    full_kind[k] = block_kind_for(locs[k].mat);
    if (full_kind[k] == BlockKind::None) {
      return std::unexpected(fit_err(
          FitError::Kind::NumericIssue,
          "SNLLS compatibility: free parameter " + std::to_string(k + 1) +
              " has unsupported matrix location"));
    }
  }

  constexpr double tol = 1e-10;
  for (Eigen::Index c = 0; c < out.constraints.Kmat.cols(); ++c) {
    BlockKind kind = BlockKind::None;
    for (Eigen::Index r = 0; r < out.constraints.Kmat.rows(); ++r) {
      if (std::abs(out.constraints.Kmat(r, c)) <= tol) continue;
      const BlockKind rk = full_kind[static_cast<std::size_t>(r)];
      if (kind == BlockKind::None) {
        kind = rk;
      } else if (kind != rk) {
        return std::unexpected(fit_err(
            FitError::Kind::NumericIssue,
            "SNLLS compatibility: equality/linear constraint basis column " +
                std::to_string(c + 1) + " mixes " +
                std::string(block_kind_name(kind)) + " and " +
                std::string(block_kind_name(rk)) + " parameters"));
      }
    }
    if (kind == BlockKind::None) continue;
    if (kind == BlockKind::Nonlinear) out.beta_cols.push_back(c);
    if (kind == BlockKind::Linear) out.alpha_cols.push_back(c);
  }

  if (out.alpha_cols.empty()) {
    return std::unexpected(fit_err(
        FitError::Kind::NumericIssue,
        "SNLLS compatibility: no conditionally linear free parameters to profile"));
  }

  const Eigen::VectorXd q0 = out.constraints.contract(theta_start);
  out.beta0.resize(static_cast<Eigen::Index>(out.beta_cols.size()));
  for (std::size_t i = 0; i < out.beta_cols.size(); ++i) {
    out.beta0(static_cast<Eigen::Index>(i)) = q0(out.beta_cols[i]);
  }

  out.K_beta.resize(out.constraints.npar,
                    static_cast<Eigen::Index>(out.beta_cols.size()));
  for (std::size_t j = 0; j < out.beta_cols.size(); ++j) {
    out.K_beta.col(static_cast<Eigen::Index>(j)) =
        out.constraints.Kmat.col(out.beta_cols[j]);
  }
  out.K_alpha.resize(out.constraints.npar,
                     static_cast<Eigen::Index>(out.alpha_cols.size()));
  for (std::size_t j = 0; j < out.alpha_cols.size(); ++j) {
    out.K_alpha.col(static_cast<Eigen::Index>(j)) =
        out.constraints.Kmat.col(out.alpha_cols[j]);
  }
  return out;
}

// One profiled point: full θ, the profiled residual, and the two Jacobian
// pieces needed downstream (H = J_r·K_α for the inner solve, J_r·K_β for the
// outer Jacobian).
struct ProfilePoint {
  Eigen::VectorXd theta;
  Eigen::VectorXd residual;
  Eigen::MatrixXd H;
  Eigen::MatrixXd jacobian;
};

// Solve the inner linear least-squares for α̂ at a given β.
fit_expected<ProfilePoint>
profile_at(const optim::GmmProblem& base, const Classification& cls,
           const Eigen::VectorXd& beta) {
  Eigen::VectorXd theta_base = cls.constraints.theta0;
  if (cls.K_beta.cols() > 0) theta_base.noalias() += cls.K_beta * beta;

  Eigen::VectorXd r0;
  Eigen::MatrixXd Jr0;
  if (base.eval) {
    auto e0 = base.eval(theta_base);
    if (!e0.has_value()) return std::unexpected(e0.error());
    r0 = std::move(e0->residual);
    Jr0 = std::move(e0->jacobian);
  } else {
    auto r0_or = base.r(theta_base);
    if (!r0_or.has_value()) return std::unexpected(r0_or.error());
    auto Jr0_or = base.J(theta_base);
    if (!Jr0_or.has_value()) return std::unexpected(Jr0_or.error());
    r0 = std::move(*r0_or);
    Jr0 = std::move(*Jr0_or);
  }

  ProfilePoint out;
  // H = ∂r/∂α is invariant in the conditionally-linear α (the residual is
  // affine in Θ/Ψ/ν/α), so evaluating it at θ_base is exact — and so is the
  // inner linear least-squares it drives.
  out.H = Jr0 * cls.K_alpha;

  const Eigen::Index n_alpha = cls.K_alpha.cols();
  if (n_alpha == 0) {
    out.residual = std::move(r0);
    out.theta    = theta_base;
    out.jacobian = Jr0 * cls.K_beta;
    return out;
  }

  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> H_qr(out.H);
  const Eigen::VectorXd alpha_hat = H_qr.solve(-r0);
  out.residual = r0 + out.H * alpha_hat;
  out.theta    = theta_base + cls.K_alpha * alpha_hat;

  // The outer (β) Jacobian must be taken at the *profiled* point θ̂(β):
  // ∂r/∂Λ depends on the latent (co)variances Ψ that α̂ has just set, so
  // J_β at θ_base would be wrong.
  auto Jr = base.J(out.theta);
  if (!Jr.has_value()) return std::unexpected(Jr.error());
  out.jacobian = *Jr * cls.K_beta;
  if (out.jacobian.cols() > 0) {
    const Eigen::MatrixXd coeff = H_qr.solve(out.jacobian);
    out.jacobian.noalias() -= out.H * coeff;
  }
  return out;
}

}  // namespace

bool gp_compatible(const spec::LatentStructure& pt,
                   const model::ModelEvaluator& ev,
                   const Eigen::VectorXd& theta0) {
  return classify(pt, ev, theta0).has_value();
}

fit_expected<GpProblem>
gp(const optim::GmmProblem& base, const spec::LatentStructure& pt,
   const model::ModelEvaluator& ev, const Eigen::VectorXd& theta0) {
  auto cls_or = classify(pt, ev, theta0);
  if (!cls_or.has_value()) return std::unexpected(cls_or.error());
  auto cls = std::make_shared<Classification>(std::move(*cls_or));

  // The optimizer queries residual and Jacobian at the same β; cache the
  // profile so the inner solve runs once per distinct β.
  struct Cache {
    bool            valid = false;
    Eigen::VectorXd beta;
    ProfilePoint    point;
  };
  auto cache = std::make_shared<Cache>();
  auto profiled = [base, cls, cache](const Eigen::VectorXd& beta)
      -> fit_expected<std::reference_wrapper<const ProfilePoint>> {
    if (cache->valid && cache->beta.size() == beta.size() &&
        (cache->beta.array() == beta.array()).all()) {
      return std::cref(cache->point);
    }
    auto p = profile_at(base, *cls, beta);
    if (!p.has_value()) return std::unexpected(p.error());
    cache->valid = true;
    cache->beta  = beta;
    cache->point = std::move(*p);
    return std::cref(cache->point);
  };

  optim::GmmProblem out;
  out.n_resid = base.n_resid;
  out.n_param = cls->K_beta.cols();
  out.r = [profiled](const Eigen::VectorXd& beta)
      -> fit_expected<Eigen::VectorXd> {
    auto p = profiled(beta);
    if (!p.has_value()) return std::unexpected(p.error());
    return p->get().residual;
  };
  out.J = [profiled](const Eigen::VectorXd& beta)
      -> fit_expected<Eigen::MatrixXd> {
    auto p = profiled(beta);
    if (!p.has_value()) return std::unexpected(p.error());
    return p->get().jacobian;
  };
  out.eval = [profiled](const Eigen::VectorXd& beta)
      -> fit_expected<optim::LsEvaluation> {
    auto p = profiled(beta);
    if (!p.has_value()) return std::unexpected(p.error());
    return optim::LsEvaluation{p->get().residual, p->get().jacobian};
  };
  out.expand = [profiled, cls](const Eigen::VectorXd& beta) -> Eigen::VectorXd {
    auto p = profiled(beta);
    if (p.has_value()) return p->get().theta;
    Eigen::VectorXd theta = cls->constraints.theta0;
    if (cls->K_beta.cols() > 0) theta.noalias() += cls->K_beta * beta;
    return theta;
  };
  return GpProblem{
      std::move(out), cls->beta0,
      static_cast<std::int32_t>(cls->beta_cols.size()),
      static_cast<std::int32_t>(cls->alpha_cols.size())};
}

}  // namespace magmaan::estimate::gmm
