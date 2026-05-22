#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/model/model_evaluator.hpp"  // ImpliedMoments
#include "magmaan/spec/partable.hpp"

namespace magmaan::model {

// Native FC-SEM covariance evaluator for `<~` models. This is deliberately
// separate from ModelEvaluator: FC-SEM has W/T semantics and derived composite
// variances that are not ordinary LISREL cells.
class FcSemEvaluator {
 public:
  static model_expected<FcSemEvaluator>
  build(const spec::LatentStructure& pt);

  std::size_t n_free() const noexcept { return n_free_; }
  std::size_t n_blocks() const noexcept { return n_blocks_; }

  model_expected<ImpliedMoments>
  sigma(const data::SampleStats& samp,
        Eigen::Ref<const Eigen::VectorXd> theta) const;

 private:
  const spec::LatentStructure* pt_ = nullptr;
  std::size_t n_free_ = 0;
  std::size_t n_blocks_ = 0;
  Eigen::Index p_ = 0;
  Eigen::Index m_ = 0;
};

}  // namespace magmaan::model
