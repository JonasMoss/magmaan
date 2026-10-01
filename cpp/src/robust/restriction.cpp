#include "magmaan/robust/restriction.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <tuple>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/estimate/start_values.hpp"
#include <utility>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::robust {

using estimate::EqConstraints;


namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

namespace {

Eigen::MatrixXd pinv_times(const Eigen::Ref<const Eigen::MatrixXd>& A,
                           const Eigen::Ref<const Eigen::MatrixXd>& B,
                           double tol) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd& s = svd.singularValues();
  const double scale = s.size() > 0 ? s(0) : 0.0;
  Eigen::VectorXd inv = Eigen::VectorXd::Zero(s.size());
  const double cutoff = tol * static_cast<double>(std::max(A.rows(), A.cols())) *
                        std::max(1.0, scale);
  for (Eigen::Index i = 0; i < s.size(); ++i) {
    if (s(i) > cutoff) inv(i) = 1.0 / s(i);
  }
  return svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose() * B;
}

}  // namespace

post_expected<RestrictionAlpha>
restriction_alpha_from_K(const EqConstraints& K_H1,
                         const EqConstraints& K_H0,
                         double               tol_range_F,
                         double               tol_singular) {
  const Eigen::Index npar = K_H1.Kmat.rows();
  const Eigen::Index r1   = K_H1.Kmat.cols();
  const Eigen::Index r0   = K_H0.Kmat.cols();

  if (K_H0.Kmat.rows() != npar) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: K_H1.npar (" + std::to_string(npar) +
        ") != K_H0.npar (" + std::to_string(K_H0.Kmat.rows()) + ")"));
  }
  if (r0 > r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: r_H0 (" + std::to_string(r0) +
        ") > r_H1 (" + std::to_string(r1) + ") — H0 has *more* free directions"
        " than H1, so the pair is not nested in the expected order."));
  }

  const Eigen::Index m = r1 - r0;
  if (m == 0 && (K_H1.Kmat-K_H0.Kmat).norm()==0 &&
      (K_H1.theta0-K_H0.theta0).norm()==0) {
    // Degenerate but valid: H0 ≡ H1 (no extra restriction). Return an empty
    // (0 × r1) restriction; downstream Satorra returns T_diff = 0, df = 0.
    return RestrictionAlpha{
        /*A=*/Eigen::MatrixXd::Zero(0, r1),
        /*b=*/Eigen::VectorXd::Zero(0)};
  }

  // 1. Project K_H0 onto K_H1's column space.  Both K matrices are
  //    orthonormal-column (built by SVD or as 0/1 group-membership with
  //    unit-norm columns after the QR step in `build_eq_constraints`), so
  //    when range(K_H0) ⊆ range(K_H1) the projector reduces to
  //
  //        M = K_H1ᵀ · K_H0     (r1 × r0,  M is then column-orthonormal).
  //
  //    We use a QR solve instead of plain matrix multiply because either K
  //    can come out of `build_eq_constraints` non-strictly-orthonormal in the
  //    pure-merge case (the 0/1 K has columns of unit norm only after the
  //    helper rescales).
  Eigen::HouseholderQR<Eigen::MatrixXd> qr_K1(K_H1.Kmat);
  const Eigen::MatrixXd M = qr_K1.solve(K_H0.Kmat);  // r1 × r0

  // Range-inclusion check: ‖K_H1·M − K_H0‖_F should be ≈ 0 when H0 nests
  // inside H1.  Tolerance is absolute, scaled to be robust against the
  // numerical noise that JacobiSVD introduces in K_H0.
  const double inclusion_resid = (K_H1.Kmat * M - K_H0.Kmat).norm();
  if (inclusion_resid > tol_range_F) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: range(K_H0) is not contained in "
        "range(K_H1) — residual ‖K_H1·M − K_H0‖_F = " +
        std::to_string(inclusion_resid) +
        " exceeds tol_range_F = " + std::to_string(tol_range_F) +
        ". H0 is not nested in H1."));
  }

  // 2. Build A_α as the orthonormal basis of null(Mᵀ) inside R^{r1}.
  //    Equivalently: the left-singular vectors of Mᵀ corresponding to its
  //    *small* singular values (i.e. its (r1 − r0) trailing left singular
  //    vectors), reshaped as rows of A.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU);
  const Eigen::MatrixXd& U = svd.matrixU();        // r1 × r1
  const Eigen::VectorXd& sv = svd.singularValues();// length min(r1,r0) = r0

  // Defensive: count actual zero singular values.  In pathological cases the
  // user-provided K may be near-rank-deficient and `m` may not match the
  // null-space dimension we predicted from `r1 - r0`.
  Eigen::Index n_zero = r1 - r0;  // expected
  for (Eigen::Index k = 0; k < sv.size(); ++k) {
    if (sv(k) < tol_singular) ++n_zero;
  }
  if (n_zero != m) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_from_K: null-space dimension " +
        std::to_string(n_zero) + " from SVD does not match r1−r0 = " +
        std::to_string(m) + " — K_H1 or K_H0 likely rank-deficient."));
  }

  // The last m columns of U span null(Mᵀ); rows of A are those vectors.
  Eigen::MatrixXd A(m, r1);
  A = U.rightCols(m).transpose();

  // 3. Inhomogeneous shift b = A · b₀ where K_H1·b₀ = θ₀_H0 − θ₀_H1
  //    (least-squares solve; b₀ is unique up to a vector in null(K_H1) which
  //    A annihilates anyway, so b is well-defined regardless).
  Eigen::VectorXd b = Eigen::VectorXd::Zero(m);
  // Only bother if either θ₀ is non-zero (pure-merge case has θ₀ ≡ 0).
  if (K_H1.theta0.size() == npar && K_H0.theta0.size() == npar &&
      (K_H0.theta0 - K_H1.theta0).squaredNorm() > 0.0) {
    const Eigen::VectorXd b0 = qr_K1.solve(K_H0.theta0 - K_H1.theta0);
    if ((K_H1.Kmat*b0-(K_H0.theta0-K_H1.theta0)).norm()>tol_range_F)
      return std::unexpected(make_err(PostError::Kind::NotNested,
          "restriction_alpha_from_K: affine null offset is outside H1"));
    b = A * b0;
  }

  return RestrictionAlpha{std::move(A), std::move(b)};
}

namespace {
using RowKey = std::tuple<std::string, parse::Op, std::string, int, int>;

std::string variable_name(const spec::LatentStructure& pt,
                          const model::MatrixRep& rep, int id, int block) {
  if (id < 0) return "";
  for (std::size_t k = 0; k < pt.ov_order.size(); ++k)
    if (pt.ov_order[k] == id) return rep.ov_names[static_cast<std::size_t>(block)][k];
  for (std::size_t k = 0; k < pt.lv_ext_order.size(); ++k)
    if (pt.lv_ext_order[k] == id) return rep.lv_names[static_cast<std::size_t>(block)][k];
  return "v" + std::to_string(id);
}
std::vector<RowKey> row_keys(const spec::LatentStructure& pt,
                            const model::MatrixRep& rep) {
  std::vector<RowKey> keys(pt.size());
  std::map<std::tuple<std::string,int,int>,int> thresholds;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.is_constraint_row(i)) continue;
    const int block = pt.block_of(i)-1;
    auto lhs = variable_name(pt,rep,pt.lhs_var[i],block);
    auto rhs = variable_name(pt,rep,pt.rhs_var[i],block);
    const int level = pt.level.empty() ? 1 : std::max(1,pt.level[i]);
    // Threshold RHSs are ordinal indices, not model variables. They have
    // canonical within-variable order in the prepared ordinal partable.
    if (pt.op[i] == parse::Op::Threshold)
      rhs = "t" + std::to_string(++thresholds[{lhs,pt.group[i],level}]);
    if (pt.op[i] == parse::Op::Covariance && rhs < lhs) std::swap(lhs,rhs);
    keys[i] = {lhs,pt.op[i],rhs,pt.group[i],level};
  }
  return keys;
}
bool zero_if_absent(const RowKey& key) {
  const auto op = std::get<1>(key);
  return op == parse::Op::Measurement || op == parse::Op::Regression ||
      (op == parse::Op::Covariance && std::get<0>(key) != std::get<2>(key));
}
bool same_ambient(const spec::LatentStructure& p1, const model::MatrixRep& r1,
                  const spec::LatentStructure& p0, const model::MatrixRep& r0) {
  if (r1.ov_names != r0.ov_names || r1.lv_names != r0.lv_names ||
      p1.n_free() != p0.n_free() || r1.form != r0.form) return false;
  using Entry = std::tuple<int,int,int,int,int,double>;
  auto entries = [](const spec::LatentStructure& p, const model::MatrixRep& r) {
    std::vector<Entry> out;
    for (std::size_t i=0;i<r.cell_for_row.size();++i) {
      const auto& c=r.cell_for_row[i];
      if (c.used) out.emplace_back(static_cast<int>(c.mat),c.block,c.row,c.col,
          p.free[i],p.free[i] ? 0.0 : p.fixed_value[i]);
    }
    for (const auto& c:r.structural_cells)
      out.emplace_back(static_cast<int>(c.mat),c.block,c.row,c.col,0,c.value);
    std::sort(out.begin(),out.end()); return out;
  };
  // Unused threshold/scale rows must match too; their slots still enter the
  // ordinal criterion even though the covariance evaluator does not read them.
  auto unused = [](const spec::LatentStructure& p, const model::MatrixRep& r) {
    std::vector<std::tuple<RowKey,int,double>> out;
    const auto keys=row_keys(p,r);
    for (std::size_t i=0;i<p.size();++i)
      if (!p.is_constraint_row(i) && !r.cell_for_row[i].used)
        out.emplace_back(keys[i],p.free[i],p.free[i] ? 0.0 : p.fixed_value[i]);
    return out;
  };
  return entries(p1,r1)==entries(p0,r0) && unused(p1,r1)==unused(p0,r0);
}
post_expected<void> check_moments(
    const model::Evaluation& a, const model::Evaluation& b) {
  if (a.moments.sigma.size()!=b.moments.sigma.size() ||
      a.moments.mu.size()!=b.moments.mu.size())
    return std::unexpected(make_err(PostError::Kind::NotNested,
        "nested embedding: covariance/mean layouts differ"));
  for (std::size_t k=0;k<a.moments.sigma.size();++k) {
    const auto& x=a.moments.sigma[k]; const auto& y=b.moments.sigma[k];
    if (x.rows()!=y.rows() || x.cols()!=y.cols() ||
        (x-y).norm()>1e-10*std::max({1.0,x.norm(),y.norm()}))
      return std::unexpected(make_err(PostError::Kind::NotNested,
          "nested embedding: implied covariances disagree at the null point"));
  }
  for (std::size_t k=0;k<a.moments.mu.size();++k) {
    const auto& x=a.moments.mu[k]; const auto& y=b.moments.mu[k];
    if (x.size()!=y.size() ||
        (x-y).norm()>1e-10*std::max({1.0,x.norm(),y.norm()}))
      return std::unexpected(make_err(PostError::Kind::NotNested,
          "nested embedding: implied means disagree at the null point"));
  }
  return {};
}
}

static post_expected<NestedEmbedding> embed_key_nested_null(
    const spec::LatentStructure& p1, const model::MatrixRep& r1,
    const spec::LatentStructure& p0, const model::MatrixRep& r0,
    const Eigen::VectorXd& theta0, const EqConstraints& c1, const EqConstraints& c0) {
  if (theta0.size()!=p0.n_free() || c1.npar!=p1.n_free() || c0.npar!=p0.n_free())
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "nested embedding: estimates and constraint dimensions disagree"));
  if (c0.n_alpha>c1.n_alpha)
    return std::unexpected(make_err(PostError::Kind::NotNested,
        "nested embedding: H0 has more free directions than H1"));
  if (r1.ov_names!=r0.ov_names || p1.n_groups()!=p0.n_groups() ||
      p1.n_levels()!=p0.n_levels())
    return std::unexpected(make_err(PostError::Kind::NotNested,
        "nested embedding: observed variables or group/level layouts differ"));
  NestedEmbedding out;
  out.same_ambient = same_ambient(p1,r1,p0,r0);
  if (out.same_ambient) {
    out.null_constraints=c0; out.theta=theta0;
  } else {
    const auto keys1=row_keys(p1,r1), keys0=row_keys(p0,r0);
    auto factors = [](const std::vector<RowKey>& keys) {
      std::vector<std::string> out;
      for (const auto& key:keys)
        if (std::get<1>(key)==parse::Op::Measurement && !std::get<0>(key).empty())
          out.push_back(std::get<0>(key));
      std::sort(out.begin(),out.end());
      out.erase(std::unique(out.begin(),out.end()),out.end());
      return out;
    };
    if (factors(keys1)!=factors(keys0))
      return std::unexpected(make_err(PostError::Kind::UnsupportedNesting,
          "nested only through moments; not supported by a parameter-key embedding"));
    std::map<RowKey,std::size_t> rows0, rows1;
    for (std::size_t i=0;i<p0.size();++i)
      if (!p0.is_constraint_row(i)) rows0.emplace(keys0[i],i);
    for (std::size_t i=0;i<p1.size();++i)
      if (!p1.is_constraint_row(i)) rows1.emplace(keys1[i],i);
    Eigen::MatrixXd T=Eigen::MatrixXd::Zero(c1.npar,c0.npar);
    Eigen::VectorXd offset=Eigen::VectorXd::Zero(c1.npar);
    for (const auto& [key,i]:rows1) {
      auto found=rows0.find(key);
      if (found==rows0.end() && !zero_if_absent(key))
        return std::unexpected(make_err(PostError::Kind::UnsupportedNesting,
            "nested embedding: absent intercept, mean, variance or scale has no zero default"));
      const int f1=p1.free[i]-1;
      const int f0=found==rows0.end() ? -1 : p0.free[found->second]-1;
      const double fixed0=found==rows0.end() ? 0.0 : p0.fixed_value[found->second];
      if (f1>=0) {
        if (f0>=0) T(f1,f0)=1.0;
        else offset(f1)=fixed0;
      } else {
        const double value=f0>=0 ? c0.theta0(f0) : fixed0;
        if ((f0>=0 && c0.Kmat.row(f0).norm()>1e-10) ||
            !std::isfinite(value) || std::abs(value-p1.fixed_value[i])>1e-10)
          return std::unexpected(make_err(PostError::Kind::NotNested,
              "nested embedding: H0 changes an H1 fixed parameter"));
      }
    }
    for (const auto& [key,i]:rows0) {
      if (rows1.contains(key)) continue;
      const int f=p0.free[i]-1;
      const double value=f>=0 ? c0.theta0(f) : p0.fixed_value[i];
      if (!zero_if_absent(key) || (f>=0 && c0.Kmat.row(f).norm()>1e-10) ||
          std::abs(value)>1e-10)
        return std::unexpected(make_err(PostError::Kind::UnsupportedNesting,
            "nested embedding: parameter correspondence unavailable; moment nesting requires a same-point solution"));
    }
    auto& c=out.null_constraints;
    c.npar=c1.npar; c.n_alpha=c0.n_alpha; c.rank=c.npar-c.n_alpha;
    c.theta0=offset+T*c0.theta0; c.Kmat=T*c0.Kmat;
    out.theta=offset+T*theta0;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(c.Kmat,Eigen::ComputeFullU);
    svd.setThreshold(1e-10);
    if (svd.rank()!=c.n_alpha)
      return std::unexpected(make_err(PostError::Kind::BoundaryNesting,
          "nested embedding: null tangent loses rank in H1 slots"));
    c.A_eq=svd.matrixU().rightCols(c.rank).transpose();
    c.b_eq=c.A_eq*c.theta0;
  }
  auto restriction=restriction_alpha_from_K(c1,out.null_constraints);
  if (!restriction) return std::unexpected(PostError{PostError::Kind::NotNested,restriction.error().detail});
  out.restriction=std::move(*restriction);
  auto ev1=model::ModelEvaluator::build(p1,r1), ev0=model::ModelEvaluator::build(p0,r0);
  if (!ev1 || !ev0)
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "nested embedding: model evaluator build failed"));
  auto m1=ev1->evaluate(out.theta,false,false), m0=ev0->evaluate(theta0,false,false);
  if (!m1 || !m0)
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "nested embedding: null moment evaluation failed"));
  auto moments=check_moments(*m1,*m0);
  if (!moments) return std::unexpected(moments.error());
  return out;
}

namespace {
Eigen::VectorXd moment_vector(const model::Evaluation& e) {
  Eigen::Index n=0;
  for (const auto& s:e.moments.sigma) n+=s.rows()*(s.rows()+1)/2;
  for (const auto& u:e.moments.mu) n+=u.size();
  Eigen::VectorXd out(n); Eigen::Index k=0;
  for (const auto& s:e.moments.sigma)
    for (Eigen::Index j=0;j<s.cols();++j)
      for (Eigen::Index i=j;i<s.rows();++i) out(k++)=s(i,j);
  for (const auto& u:e.moments.mu) {out.segment(k,u.size())=u; k+=u.size();}
  return out;
}
Eigen::MatrixXd moment_jacobian(const model::Evaluation& e) {
  Eigen::MatrixXd out(e.J_sigma.rows()+e.J_mu.rows(),e.J_sigma.cols());
  out.topRows(e.J_sigma.rows())=e.J_sigma;
  if (e.J_mu.rows()) out.bottomRows(e.J_mu.rows())=e.J_mu;
  return out;
}
post_expected<void> interior_tangent(const spec::LatentStructure& pt,
                                    model::ModelEvaluator& ev,
                                    const Eigen::VectorXd& theta,
                                    const EqConstraints& con,
                                    const model::Evaluation& eval) {
  Eigen::MatrixXd tangent=moment_jacobian(eval)*con.Kmat;
  // Rank must not depend on observed or latent measurement units. Equilibrate
  // both the moment rows and the parameter directions before the SVD gate.
  for (Eigen::Index i=0;i<tangent.rows();++i) {
    const double norm=tangent.row(i).norm();
    if (norm>0.0) tangent.row(i)/=norm;
  }
  for (Eigen::Index j=0;j<tangent.cols();++j) {
    const double norm=tangent.col(j).norm();
    if (norm>0.0) tangent.col(j)/=norm;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(tangent);
  svd.setThreshold(1e-9);
  if (svd.rank()!=con.n_alpha)
    return std::unexpected(make_err(PostError::Kind::BoundaryNesting,
        "nested embedding: alternative tangent loses rank at the null point"));
  auto matrices=ev.assembled(theta);
  if (!matrices) return std::unexpected(make_err(PostError::Kind::NumericIssue,matrices.error().detail));
  std::vector<int> latent;
  for (std::size_t i=0;i<pt.lv_ext_order.size();++i)
    if (pt.is_user_latent[static_cast<std::size_t>(pt.lv_ext_order[i])]) latent.push_back(static_cast<int>(i));
  for (const auto& block:matrices->blocks) {
    // Test user-factor covariance, excluding the structural zero rows of
    // phantom latents. Fixed zero residuals are valid identification choices.
    if (!latent.empty()) {
      Eigen::MatrixXd psi(static_cast<Eigen::Index>(latent.size()),static_cast<Eigen::Index>(latent.size()));
      for (std::size_t i=0;i<latent.size();++i)
        for (std::size_t j=0;j<latent.size();++j) psi(static_cast<Eigen::Index>(i),static_cast<Eigen::Index>(j))=block.Psi(latent[i],latent[j]);
      if ((psi.diagonal().array()<=0.0).any())
        return std::unexpected(make_err(PostError::Kind::BoundaryNesting,
            "nested embedding: null point is on the factor-variance boundary"));
      const Eigen::VectorXd scale=psi.diagonal().array().sqrt().inverse();
      psi=scale.asDiagonal()*psi*scale.asDiagonal();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(psi);
      if (es.info()!=Eigen::Success || es.eigenvalues().minCoeff()<=
          1e-9*std::max(1.0,psi.norm()))
        return std::unexpected(make_err(PostError::Kind::BoundaryNesting,
            "nested embedding: null point is on the factor-covariance boundary"));
    }
  }
  return {};
}
post_expected<NestedEmbedding> embed_moment_nested_null(
    const spec::LatentStructure& p1,const model::MatrixRep& r1,
    const spec::LatentStructure& p0,const model::MatrixRep& r0,
    const Eigen::VectorXd& theta0,const EqConstraints& c1,const EqConstraints& c0,
    const Eigen::VectorXd* theta1) {
  if (r1.ov_names!=r0.ov_names || p1.n_groups()!=p0.n_groups() ||
      p1.n_levels()!=p0.n_levels() || c0.n_alpha>=c1.n_alpha)
    return std::unexpected(make_err(PostError::Kind::NotNested,
        "same-point nesting: incompatible layout or no restrictions released"));
  auto ev1=model::ModelEvaluator::build(p1,r1), ev0=model::ModelEvaluator::build(p0,r0);
  if (!ev1 || !ev0) return std::unexpected(make_err(PostError::Kind::NumericIssue,
      "same-point nesting: evaluator build failed"));
  auto target=ev0->evaluate(theta0,true,true);
  if (!target) return std::unexpected(make_err(PostError::Kind::NumericIssue,target.error().detail));
  Eigen::VectorXd initial;
  if (theta1) initial=*theta1;
  else {
    data::SampleStats sample; sample.S=target->moments.sigma;
    sample.mean=target->moments.mu;
    sample.n_obs.resize(sample.S.size(),1);
    auto start=estimate::simple_start_values(p1,r1,sample);
    if (!start) return std::unexpected(make_err(PostError::Kind::NumericIssue,start.error().detail));
    initial=std::move(*start);
  }
  Eigen::VectorXd alpha=c1.contract(initial);
  const Eigen::VectorXd y=moment_vector(*target);
  const Eigen::VectorXd scale=y.cwiseAbs().cwiseMax(1.0);
  double damping=1e-6;
  bool solved=false;
  // Fit the alternative to the fitted null moments, with its own constraints.
  // Damped Gauss-Newton uses analytic moment derivatives, never a mixed-point
  // Jacobian. A zero residual alone is insufficient: tangent inclusion below
  // also checks the entire null tangent at this common point.
  for (int iteration=0;iteration<300;++iteration) {
    auto e=ev1->evaluate(c1.expand(alpha),true,true);
    if (!e || moment_vector(*e).size()!=y.size()) break;
    const Eigen::VectorXd residual=(moment_vector(*e)-y).cwiseQuotient(scale);
    if (residual.norm()<1e-12) {solved=true; break;}
    Eigen::MatrixXd J=moment_jacobian(*e)*c1.Kmat;
    J.array().colwise()/=scale.array();
    const Eigen::MatrixXd gram=J.transpose()*J;
    Eigen::MatrixXd damped=gram;
    damped.diagonal().array()+=damping;
    const Eigen::VectorXd step=damped.ldlt().solve(-J.transpose()*residual);
    if (!step.allFinite()) break;
    auto trial=ev1->evaluate(c1.expand(alpha+step),false,false);
    if (trial && moment_vector(*trial).size()==y.size() &&
        (moment_vector(*trial)-y).cwiseQuotient(scale).squaredNorm()<residual.squaredNorm()) {
      alpha+=step; damping=std::max(1e-14,damping*0.2);
    } else damping*=10.0;
    if (damping>1e16) break;
  }
  if (!solved) return std::unexpected(make_err(PostError::Kind::NotNested,
      "same-point nesting: alternative cannot reproduce the fitted null moments"));
  NestedEmbedding out; out.through_moments=true; out.theta=c1.expand(alpha);
  auto point=ev1->evaluate(out.theta,true,true);
  if (!point) return std::unexpected(make_err(PostError::Kind::NumericIssue,point.error().detail));
  auto equal=check_moments(*point,*target);
  if (!equal) return std::unexpected(equal.error());
  auto interior=interior_tangent(p1,*ev1,out.theta,c1,*point);
  if (!interior) return std::unexpected(interior.error());
  const Eigen::MatrixXd J1=moment_jacobian(*point)*c1.Kmat;
  const Eigen::MatrixXd J0=moment_jacobian(*target)*c0.Kmat;
  const Eigen::MatrixXd M=pinv_times(J1,J0,1e-10);
  if ((J1*M-J0).norm()>1e-8*std::max(1.0,J0.norm()))
    return std::unexpected(make_err(PostError::Kind::NotNested,
        "same-point nesting: null tangent is outside the alternative tangent"));
  Eigen::JacobiSVD<Eigen::MatrixXd> tangent(M,Eigen::ComputeFullU);
  tangent.setThreshold(1e-9);
  if (tangent.rank()!=c0.n_alpha)
    return std::unexpected(make_err(PostError::Kind::BoundaryNesting,
        "same-point nesting: null tangent loses rank"));
  auto& c=out.null_constraints;
  c.npar=c1.npar; c.n_alpha=c0.n_alpha; c.rank=c.npar-c.n_alpha;
  c.Kmat=c1.Kmat*M; c.theta0=out.theta;
  Eigen::JacobiSVD<Eigen::MatrixXd> ambient(c.Kmat,Eigen::ComputeFullU);
  c.A_eq=ambient.matrixU().rightCols(c.rank).transpose(); c.b_eq=c.A_eq*c.theta0;
  auto restriction=restriction_alpha_from_K(c1,c);
  if (!restriction) return std::unexpected(restriction.error());
  out.restriction=std::move(*restriction);
  return out;
}
}

post_expected<NestedEmbedding> embed_nested_null(
    const spec::LatentStructure& p1,const model::MatrixRep& r1,
    const spec::LatentStructure& p0,const model::MatrixRep& r0,
    const Eigen::VectorXd& theta0,const EqConstraints& c1,const EqConstraints& c0,
    bool allow_moment_nesting,const Eigen::VectorXd* theta1) {
  auto key=embed_key_nested_null(p1,r1,p0,r0,theta0,c1,c0);
  if (key) {
    // Checking regularity leaves the existing same-slot numeric pipeline intact.
    // For ordinal models unused threshold slots make the covariance tangent
    // incomplete; their criterion owns identification checks.
    if (std::none_of(p1.op.begin(),p1.op.end(),
        [](parse::Op op){return op==parse::Op::Threshold || op==parse::Op::ResponseScale;})) {
      auto ev=model::ModelEvaluator::build(p1,r1);
      if (!ev) return std::unexpected(make_err(PostError::Kind::NumericIssue,ev.error().detail));
      auto e=ev->evaluate(key->theta,true,true);
      if (!e) return std::unexpected(make_err(PostError::Kind::NumericIssue,e.error().detail));
      auto interior=interior_tangent(p1,*ev,key->theta,c1,*e);
      if (!interior) return std::unexpected(interior.error());
    }
    return key;
  }
  if (!allow_moment_nesting || key.error().kind==PostError::Kind::NumericIssue ||
      key.error().kind==PostError::Kind::BoundaryNesting) return key;
  return embed_moment_nested_null(p1,r1,p0,r0,theta0,c1,c0,theta1);
}

spec::LatentStructure embedded_null_structure(
    const spec::LatentStructure& pt, const EqConstraints& c) {
  spec::LatentStructure out=pt;
  out.eq_groups.clear(); out.lin_constraint_R.clear(); out.lin_constraint_d.clear();
  for (Eigen::Index i=0;i<c.A_eq.rows();++i) {
    for (Eigen::Index j=0;j<c.A_eq.cols();++j) out.lin_constraint_R.push_back(c.A_eq(i,j));
    out.lin_constraint_d.push_back(c.b_eq(i));
  }
  return out;
}

post_expected<Eigen::MatrixXd>
restriction_alpha_delta_from_jacobians(
    const Eigen::Ref<const Eigen::MatrixXd>& Pi_H1_alpha,
    const Eigen::Ref<const Eigen::MatrixXd>& Pi_H0_alpha,
    int                                      expected_m,
    double                                   tol_singular) {
  if (Pi_H1_alpha.rows() != Pi_H0_alpha.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_delta_from_jacobians: H1/H0 moment Jacobians have "
        "different row counts"));
  }
  const Eigen::Index r1 = Pi_H1_alpha.cols();
  if (expected_m < 0 || expected_m > r1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_delta_from_jacobians: expected restriction rank m = " +
        std::to_string(expected_m) + " is outside [0, r_H1 = " +
        std::to_string(r1) + "]"));
  }
  if (expected_m == 0) {
    return Eigen::MatrixXd::Zero(0, r1);
  }

  const Eigen::MatrixXd H = pinv_times(Pi_H1_alpha, Pi_H0_alpha, tol_singular);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU);
  const Eigen::VectorXd& s = svd.singularValues();
  const double scale = s.size() > 0 ? s(0) : 0.0;
  const double cutoff = tol_singular *
                        static_cast<double>(std::max(H.rows(), H.cols())) *
                        std::max(1.0, scale);
  Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < s.size(); ++i) {
    if (s(i) > cutoff) ++rank;
  }
  const Eigen::Index null_dim = r1 - rank;
  if (null_dim != expected_m) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "restriction_alpha_delta_from_jacobians: null-space dimension " +
        std::to_string(null_dim) + " from H = pinv(Π_H1)Π_H0 does not match "
        "df_H0−df_H1 = " + std::to_string(expected_m)));
  }
  return svd.matrixU().rightCols(expected_m).transpose();
}

}  // namespace magmaan::robust
