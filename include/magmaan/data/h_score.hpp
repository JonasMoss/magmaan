#pragma once

#include <limits>

#include "magmaan/expected.hpp"

namespace magmaan::data {

enum class PolychoricHScoreKind {
  ML,
  WmaHardCap,
  SmoothCap,
  ExpCap,
};

struct PolychoricHScoreOptions {
  PolychoricHScoreKind kind = PolychoricHScoreKind::ML;
  double k = std::numeric_limits<double>::infinity();
  double a = 1.6;
  double b = 2.2;
  double lambda = 0.2;
};

struct PolychoricHScoreEval {
  double h = 0.0;
  double dh = 0.0;
  double phi = 0.0;
};

post_expected<PolychoricHScoreEval>
eval_polychoric_h_score(double t, const PolychoricHScoreOptions& options = {});

}  // namespace magmaan::data
