// R-shaped caller NACOV validation shared by the inference adapters.
#pragma once
#include "glue_internal.h"
#include <Eigen/Eigenvalues>

namespace magmaanr::fitglue {

inline std::vector<Eigen::Index> continuous_gamma_dimensions(const Ctx& ctx) {
  std::vector<Eigen::Index> dims;
  for (const auto& S : ctx.samp.S) {
    const auto p = S.rows();
    dims.push_back(p * (p + 1) / 2 + (ctx.meanstructure ? p : 0));
  }
  return dims;
}

inline std::vector<Eigen::MatrixXd> supplied_gamma_blocks(
    SEXP arg, const std::vector<Eigen::Index>& dims) {
  Rcpp::List blocks;
  if (Rf_isMatrix(arg)) {
    if (dims.size() != 1) Rcpp::stop("gamma: use a list of matrices for multiple groups");
    blocks = Rcpp::List::create(arg);
  } else if (TYPEOF(arg) == VECSXP) {
    blocks = Rcpp::List(arg);
  } else {
    Rcpp::stop("gamma: expected a numeric matrix or a list of per-group matrices");
  }
  if (static_cast<std::size_t>(blocks.size()) != dims.size())
    Rcpp::stop("gamma: group count does not match the fit");
  std::vector<Eigen::MatrixXd> out;
  for (std::size_t b = 0; b < dims.size(); ++b) {
    if (dims[b] <= 0) Rcpp::stop("gamma: fit has no covariance layout in group %d", static_cast<int>(b + 1));
    SEXP block = blocks[b];
    if (!Rf_isMatrix(block) || !(TYPEOF(block) == REALSXP || TYPEOF(block) == INTSXP))
      Rcpp::stop("gamma: each group must be a numeric matrix");
    Eigen::MatrixXd G = Rcpp::as<Eigen::MatrixXd>(block);
    if (G.rows() != dims[b] || G.cols() != dims[b])
      Rcpp::stop("gamma: dimension mismatch in group %d; expected %d x %d",
                 static_cast<int>(b + 1), static_cast<int>(dims[b]), static_cast<int>(dims[b]));
    if (!G.allFinite()) Rcpp::stop("gamma: entries must be finite");
    const double scale = G.cwiseAbs().maxCoeff();
    const double tolerance = 1e-10 * scale;
    if ((G - G.transpose()).cwiseAbs().maxCoeff() > tolerance)
      Rcpp::stop("gamma: matrix must be symmetric in group %d", static_cast<int>(b + 1));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(G, Eigen::EigenvaluesOnly);
    if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() < -tolerance)
      Rcpp::stop("gamma: matrix must be positive semidefinite (PSD) in group %d",
                 static_cast<int>(b + 1));
    out.push_back(std::move(G));
  }
  return out;
}

inline Eigen::MatrixXd supplied_ml_gamma(
    const std::vector<Eigen::MatrixXd>& blocks, const Ctx& ctx) {
  Eigen::Index size = 0;
  double total_n = 0;
  for (const auto& G : blocks) size += G.rows();
  for (const auto n : ctx.samp.n_obs) total_n += static_cast<double>(n);
  Eigen::MatrixXd full = Eigen::MatrixXd::Zero(size, size);
  Eigen::Index offset = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto m = blocks[b].rows();
    full.block(offset, offset, m, m) =
        (static_cast<double>(ctx.samp.n_obs[b]) / total_n) * blocks[b];
    offset += m;
  }
  return full;
}

template <typename Stats>
void replace_score_nacov(Stats& stats, SEXP gamma) {
  if (Rf_isNull(gamma)) return;
  std::vector<Eigen::Index> dims;
  for (const auto& G : stats.NACOV) dims.push_back(G.rows());
  // Only the meat changes: W_dwls/W_wls remain the actual fitting weights.
  stats.NACOV = supplied_gamma_blocks(gamma, dims);
}

inline void validate_score_gamma_request(SEXP gamma, bool estimated_weight,
                                         const std::string& estimator,
                                         const std::string& cov) {
  if (Rf_isNull(gamma)) return;
  if (estimated_weight)
    stop_post({magmaan::PostError::Kind::UnsupportedInference,
               "supplied gamma cannot carry casewise weight influence; set estimated_weight = FALSE explicitly"});
  if (estimator == "FIML" || is_ml2s_estimator_label(estimator))
    stop_post({magmaan::PostError::Kind::UnsupportedInference,
               "supplied gamma is unavailable for FIML and ML2S score routes"});
  if (cov != "empirical")
    Rcpp::stop("gamma: supplied covariance requires cov='empirical'; do not also select model-implied or Browne covariance");
}

}  // namespace magmaanr::fitglue
