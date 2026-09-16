#pragma once

#include <memory>
#include <optional>
#include "magmaan/robust/robust.hpp"

namespace magmaan::robust::frontier {
// These owning snapshots are shared by native consumers. Inputs must not be
// mutated after preparation. Lazy caches are local to a session, not thread-safe.
enum class ContributionStorage { Auto, Casewise, Tiled };
struct NTMLData {
  RawData raw;
  SampleStats sample;
  estimate::fiml::FIMLPack pack;
  bool has_means = false;
  ContributionStorage storage = ContributionStorage::Auto;
  std::optional<Eigen::MatrixXd> contributions;
  std::size_t contribution_builds = 0, projection_passes = 0;
};
struct NTMLQuadratic {
  double statistic = 0;
  int df = 0;
  Eigen::MatrixXd rows; // scaled rows: crossprod(rows) is the reduced covariance
  std::optional<Eigen::MatrixXd> reduced;
  std::optional<Eigen::VectorXd> eigenvalues;
  std::size_t spectrum_builds = 0;
  bool row_space = false;
};
struct NTMLFit {
  std::shared_ptr<NTMLData> data;
  spec::LatentStructure pt;
  model::MatrixRep rep;
  Estimates estimates;
  std::optional<NTMLGeometry> geometry;
  std::optional<UFactor> expected, observed;
  std::optional<Eigen::MatrixXd> weighted_delta, information, covariance, robust_covariance;
  std::optional<Eigen::MatrixXd> projected_rows;
  std::shared_ptr<NTMLQuadratic> score, lr;
  std::optional<Eigen::VectorXd> unbiased_spectrum;
  std::vector<std::pair<std::shared_ptr<NTMLFit>,std::shared_ptr<NTMLQuadratic>>> nested_lr;
  std::size_t geometry_builds = 0, u_builds = 0, information_builds = 0;
};
struct NTMLHypothesis {
  std::shared_ptr<NTMLFit> null_fit, alternative;
  RestrictionAlpha restriction;
  std::shared_ptr<NTMLQuadratic> score, lr;
};
post_expected<std::shared_ptr<NTMLData>> prepare_ntml_data(
    RawData raw, bool has_means, ContributionStorage storage = ContributionStorage::Auto);
post_expected<std::shared_ptr<NTMLFit>> prepare_ntml_fit(
    std::shared_ptr<NTMLData> data, spec::LatentStructure pt,
    model::MatrixRep rep, Estimates estimates);
post_expected<const NTMLGeometry*> ntml_geometry(NTMLFit& fit);
post_expected<const UFactor*> ntml_factor(NTMLFit& fit, Information bread = Information::Expected);
post_expected<std::shared_ptr<NTMLQuadratic>> ntml_quadratic(NTMLFit& fit, bool score);
post_expected<std::shared_ptr<NTMLHypothesis>> prepare_ntml_hypothesis(
    std::shared_ptr<NTMLFit> null_fit, std::shared_ptr<NTMLFit> alternative);
post_expected<std::shared_ptr<NTMLQuadratic>> ntml_quadratic(NTMLHypothesis& hypothesis, bool score);
post_expected<const Eigen::VectorXd*> ntml_unbiased_spectrum(NTMLFit& fit);
post_expected<const Eigen::VectorXd*> ntml_spectrum(NTMLQuadratic& quadratic);
post_expected<const Eigen::MatrixXd*> ntml_covariance(NTMLFit& fit, bool robust = false);
post_expected<const Eigen::MatrixXd*> ntml_information(NTMLFit& fit);
} // namespace magmaan::robust::frontier
