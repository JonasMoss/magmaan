#pragma once

// The complete-data timing catalog: a small set of model structures, each
// paired with the exact population covariance it was built from.
//
// The population is constructed here in closed form rather than recovered from
// a partable, because the structures are generated here too: we know Λ, Β, Ψ,
// and Θ by construction, so Σ = Σ(θ₀) is exact and the fitted model is
// correctly specified by construction. That matters for timing — a
// misspecified population changes iteration counts, and iteration count is one
// of the two things being measured.
//
// Every population is standardized to unit observed variances. All three
// parameterizations (marker, std.lv, effect-coding) describe the *same*
// population, so a parameterization arm is a pure change of coordinates: same
// Σ, same optimum, different geometry on the way there.
//
// Population constants:
//   loadings 0.70, residual variances 0.51  (unit observed variance)
//   first-order factor correlations 0.30     (cfa_3f family)
//   second-order loadings 0.70               (soc_2nd; implies 0.49 among f)
//   structural regressions 0.40 / 0.50       (sem_2x2)
//   residual covariances 0.15                (cfa_3f_rescov)
//   path coefficients 0.25 on three lags     (path_recursive)

#include <string>
#include <vector>

#include <Eigen/Core>

namespace magmaan::bench {

// Latent scaling / identification convention. These map onto
// spec::BuildOptions flags, not onto different models.
enum class Param { Marker, StdLv, EffectCoding };

const char* param_name(Param p);

struct ModelCase {
  std::string     id;
  std::string     syntax;
  Eigen::MatrixXd sigma;  // population covariance, p x p, unit diagonal
  int             p = 0;
};

// Structures that scale across the whole p ladder.
std::vector<std::string> scaling_ids();

// Structures pinned to the p = 12 variety hub.
std::vector<std::string> hub_ids();

std::vector<std::string> all_ids();

// Whether this structure is defined at this p (divisibility, minimum
// indicators per factor, hub-only pinning).
bool supports(const std::string& id, int p);

// Whether latent-scaling variants mean anything for this structure.
// False for path_recursive, which has no latent variables.
bool has_latents(const std::string& id);

ModelCase make_case(const std::string& id, int p);

}  // namespace magmaan::bench
