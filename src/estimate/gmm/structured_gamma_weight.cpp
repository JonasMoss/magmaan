#include "magmaan/estimate/gmm/structured_gamma_weight.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>

#include "magmaan/error.hpp"

#include "detail_vech.hpp"

namespace magmaan::estimate::frontier {

namespace {

using detail::vech_index;
using detail::vech_len;

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

fit_expected<void>
validate_structured_gamma_shapes(const data::SampleStats& samp,
                                 const data::RawData& raw,
                                 const model::ImpliedMoments& m) {
  if (raw.X.size() != samp.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "structured_gamma_weight: RawData and SampleStats block counts differ"));
  }
  if (!raw.mask.empty()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "structured_gamma_weight: complete raw data is required"));
  }
  if (samp.S.size() != m.sigma.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "structured_gamma_weight: SampleStats and model block counts differ"));
  }
  if (samp.n_obs.size() != samp.S.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "structured_gamma_weight: n_obs block count does not match sample "
        "covariances"));
  }
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    if (b < m.mu.size() && m.mu[b].size() > 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "structured_gamma_weight: mean-structure models are not supported"));
    }

    const Eigen::MatrixXd& S = samp.S[b];
    const Eigen::MatrixXd& X = raw.X[b];
    const Eigen::MatrixXd& Sigma = m.sigma[b];
    const Eigen::Index p = S.rows();
    if (S.rows() != S.cols() || Sigma.rows() != Sigma.cols()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "structured_gamma_weight: block " + std::to_string(b) +
          " covariance matrix is not square"));
    }
    if (Sigma.rows() != p || X.cols() != p || X.rows() != samp.n_obs[b]) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "structured_gamma_weight: block " + std::to_string(b) +
          " raw data, sample covariance, and implied covariance shapes differ"));
    }
  }
  return {};
}

struct Quad {
  Eigen::Index i;
  Eigen::Index j;
  Eigen::Index k;
  Eigen::Index l;
};

std::vector<Quad> nondecreasing_quads(Eigen::Index p) {
  std::vector<Quad> out;
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i; j < p; ++j) {
      for (Eigen::Index k = j; k < p; ++k) {
        for (Eigen::Index l = k; l < p; ++l) {
          out.push_back({i, j, k, l});
        }
      }
    }
  }
  return out;
}

double empirical_fourth_moment(const Eigen::MatrixXd& Xc, const Quad& q) {
  return (Xc.col(q.i).array() * Xc.col(q.j).array() *
          Xc.col(q.k).array() * Xc.col(q.l).array()).mean();
}

Eigen::VectorXd source_cumulants_ols(const Eigen::MatrixXd& X,
                                     const Eigen::MatrixXd& Lambda,
                                     const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = X.cols();
  const Eigen::Index q = Lambda.cols();
  const std::vector<Quad> quads = nondecreasing_quads(p);
  const Eigen::VectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd Xc = X.rowwise() - mean.transpose();

  Eigen::MatrixXd B(static_cast<Eigen::Index>(quads.size()), q + p);
  Eigen::VectorXd c(static_cast<Eigen::Index>(quads.size()));

  for (Eigen::Index r = 0; r < static_cast<Eigen::Index>(quads.size()); ++r) {
    const Quad& h = quads[static_cast<std::size_t>(r)];
    c(r) = empirical_fourth_moment(Xc, h) -
           Sigma(h.i, h.j) * Sigma(h.k, h.l) -
           Sigma(h.i, h.k) * Sigma(h.j, h.l) -
           Sigma(h.i, h.l) * Sigma(h.j, h.k);

    for (Eigen::Index a = 0; a < q; ++a) {
      B(r, a) = Lambda(h.i, a) * Lambda(h.j, a) *
                Lambda(h.k, a) * Lambda(h.l, a);
    }
    for (Eigen::Index i = 0; i < p; ++i) {
      B(r, q + i) = (h.i == i && h.j == i && h.k == i && h.l == i) ? 1.0 : 0.0;
    }
  }

  return B.completeOrthogonalDecomposition().solve(c);
}

Eigen::MatrixXd structured_gamma(const Eigen::MatrixXd& Lambda,
                                 const Eigen::MatrixXd& Sigma,
                                 const Eigen::VectorXd& kappa) {
  const Eigen::Index p = Sigma.rows();
  const Eigen::Index q = Lambda.cols();
  const Eigen::Index pstar = vech_len(p);
  Eigen::MatrixXd Gamma(pstar, pstar);

  for (Eigen::Index c1 = 0; c1 < p; ++c1) {
    for (Eigen::Index r1 = c1; r1 < p; ++r1) {
      const Eigen::Index a_idx = vech_index(p, r1, c1);
      for (Eigen::Index c2 = 0; c2 < p; ++c2) {
        for (Eigen::Index r2 = c2; r2 < p; ++r2) {
          const Eigen::Index b_idx = vech_index(p, r2, c2);
          double fourth_cumulant = 0.0;
          for (Eigen::Index a = 0; a < q; ++a) {
            fourth_cumulant += Lambda(r1, a) * Lambda(c1, a) *
                               Lambda(r2, a) * Lambda(c2, a) * kappa(a);
          }
          if (r1 == c1 && c1 == r2 && r2 == c2) {
            fourth_cumulant += kappa(q + r1);
          }

          Gamma(a_idx, b_idx) =
              Sigma(r1, r2) * Sigma(c1, c2) +
              Sigma(r1, c2) * Sigma(c1, r2) +
              fourth_cumulant;
        }
      }
    }
  }
  return Gamma;
}

}  // namespace

fit_expected<gmm::Weight>
structured_gamma_weight(const model::ModelEvaluator& ev,
                        const model::MatrixRep& rep,
                        const data::SampleStats& samp,
                        const data::RawData& raw,
                        const Eigen::VectorXd& theta0) {
  if (rep.form != model::RepForm::PureCFA) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "structured_gamma_weight: only pure CFA matrix representations are "
        "supported"));
  }

  auto eval0 = ev.evaluate(theta0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "structured_gamma_weight: theta0: " + eval0.error().detail));
  }
  auto shapes = validate_structured_gamma_shapes(samp, raw, eval0->moments);
  if (!shapes.has_value()) return std::unexpected(shapes.error());

  auto matrices = ev.assembled(theta0);
  if (!matrices.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
        "structured_gamma_weight: theta0: " + matrices.error().detail));
  }

  gmm::Weight W;
  W.reserve(raw.X.size());
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const model::BlockMatrices& mb = matrices->blocks[b];
    if (mb.Lambda.cols() == 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "structured_gamma_weight: block " + std::to_string(b) +
          " has no latent factor columns"));
    }

    const Eigen::VectorXd kappa =
        source_cumulants_ols(raw.X[b], mb.Lambda, eval0->moments.sigma[b]);
    const Eigen::MatrixXd Gamma =
        structured_gamma(mb.Lambda, eval0->moments.sigma[b], kappa);

    Eigen::LLT<Eigen::MatrixXd> g_llt(Gamma);
    if (g_llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "structured_gamma_weight: block " + std::to_string(b) +
          " structured Gamma is not positive definite"));
    }
    const Eigen::MatrixXd Wb =
        g_llt.solve(Eigen::MatrixXd::Identity(Gamma.rows(), Gamma.cols()));
    Eigen::MatrixXd Wsym = 0.5 * (Wb + Wb.transpose());
    W.push_back(std::move(Wsym));
  }
  return W;
}

}  // namespace magmaan::estimate::frontier
