#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/robust/satorra2000.hpp"

namespace magmaan::robust::frontier {

// Foldnes-Moss-Gronneberg eigenvalue tests.
//
// This is deliberately not a lavaan-style string parser. Callers choose the
// method and optional tuning value explicitly, then supply the source
// chi-square statistic and the UGamma eigenvalues from the existing robust
// machinery.
enum class FmgMethod {
  StandardChiSquare,
  SatorraBentler,
  ScaledShifted,
  MeanVarAdjusted,   // Satterthwaite mean-and-variance adjusted (fractional df)
  ScaledF,
  All,
  PenalizedAll,
  Eba,
  Peba,
  Pols
};

struct FmgOptions {
  FmgMethod method = FmgMethod::Peba;
  double    param  = 4.0;   // Peba: block count j; Pols: shrinkage gamma.
  bool      truncate_negative = true;
};

struct FmgTestResult {
  double chi2_source = 0.0;
  int    df          = 0;
  double p_value     = 0.0;
  double chi2_equiv  = 0.0;  // qchisq(p_value, df, lower.tail = FALSE)

  Eigen::VectorXd lambdas_raw;       // descending, top df, before truncation
  Eigen::VectorXd lambdas;           // descending, after truncation
  Eigen::VectorXd lambdas_reference; // weights used by the chosen method
  int             n_truncated = 0;

  FmgMethod method = FmgMethod::Peba;
  double    param  = 4.0;
};

FmgTestResult
fmg_test(double chi2_source,
         int df,
         const Eigen::Ref<const Eigen::VectorXd>& ugamma_eigenvalues,
         FmgOptions options = {});

post_expected<FmgTestResult>
fmg_test_from_reduced_matrix(double chi2_source,
                             int df,
                             const Eigen::Ref<const Eigen::MatrixXd>& M,
                             FmgOptions options = {});

post_expected<FmgTestResult>
lr_test_fmg(double T_diff,
            const SatorraDiffResult& sd,
            FmgOptions options = {FmgMethod::PenalizedAll, 0.0, true});

}  // namespace magmaan::robust::frontier
