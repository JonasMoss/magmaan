#pragma once

// Test-only convenience wrapper. The library itself returns each
// inference primitive as a separate object (information_*, vcov, se,
// chi2_stat, df_stat) — that's the public-API contract, and tests should
// usually exercise the individual pieces. But many existing tests assert
// against the whole {info, vcov, se, chi2, df} bundle at once, and
// re-stitching the pieces inline at each call site is just churn that
// makes the test less readable. So we provide one small helper here that
// chains the new free functions and returns a struct with the obvious
// fields — purely for test ergonomics. Nothing in src/ or include/
// references this.

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/fit.hpp"
#include "magmaan/fit/inference.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/partable/partable.hpp"

namespace magmaan::test {

struct InferenceBundle {
  Eigen::MatrixXd info;
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
  double          chi2 = 0.0;
  int             df   = 0;
};

template <class InfoFn>
inline post_expected<InferenceBundle>
make_bundle(InfoFn&& info_fn,
            const partable::LatentStructure& pt,
            const fit::SampleStats&          samp,
            const fit::Estimates&            est) {
  auto info_or = info_fn();
  if (!info_or.has_value()) return std::unexpected(info_or.error());
  auto vcov_or = fit::vcov(*info_or, pt);
  if (!vcov_or.has_value()) return std::unexpected(vcov_or.error());
  auto df_or   = fit::df_stat(pt, samp);
  if (!df_or.has_value())   return std::unexpected(df_or.error());
  InferenceBundle out;
  out.info = std::move(*info_or);
  out.vcov = std::move(*vcov_or);
  out.se   = fit::se(out.vcov);
  out.chi2 = fit::chi2_stat(samp, est);
  out.df   = *df_or;
  return out;
}

// Convenience: build the bundle from each of the three information methods.
inline post_expected<InferenceBundle>
expected_inference(partable::LatentStructure pt,
                   const model::MatrixRep&   rep,
                   const fit::SampleStats&   samp,
                   const fit::Estimates&     est) {
  return make_bundle(
      [&] { return fit::information_expected(pt, rep, samp, est); },
      pt, samp, est);
}

inline post_expected<InferenceBundle>
fd_observed_inference(partable::LatentStructure pt,
                      const model::MatrixRep&   rep,
                      const fit::SampleStats&   samp,
                      const fit::Estimates&     est,
                      double                    h_step = 1e-4) {
  return make_bundle(
      [&] { return fit::information_observed_fd(pt, rep, samp, est, h_step); },
      pt, samp, est);
}

inline post_expected<InferenceBundle>
analytic_observed_inference(partable::LatentStructure pt,
                            const model::MatrixRep&   rep,
                            const fit::SampleStats&   samp,
                            const fit::Estimates&     est) {
  return make_bundle(
      [&] { return fit::information_observed_analytic(pt, rep, samp, est); },
      pt, samp, est);
}

}  // namespace magmaan::test
