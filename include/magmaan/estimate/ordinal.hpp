#pragma once

#include <string_view>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/concepts.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

enum class OrdinalWeightKind {
  DWLS,
  WLS,
};

fit_expected<void>
prepare_ordinal_delta_partable(spec::LatentStructure& pt,
                                const data::OrdinalStats& stats,
                                spec::Starts* starts = nullptr);

template <optim::LsBoundedOptimizer O = optim::LbfgsBOptimizer>
fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    O optimizer = {},
                    spec::Starts starts = {});

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    optim::LbfgsBOptimizer optimizer,
                    spec::Starts starts = {});

extern template fit_expected<Estimates>
fit_ordinal_bounded<optim::LbfgsBOptimizer>(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    Bounds bounds,
    OrdinalWeightKind weights,
    optim::LbfgsBOptimizer optimizer,
    spec::Starts starts);

}  // namespace magmaan::estimate
