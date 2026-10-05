# Lab inference defaults: implementation inventory

Audit and approved phase-two implementation on `lane/task-61`. Authority:
[scope requirement](../scope.md#misspecification-robust-inference-requirement).
R = robust already (for the listed argument only); F = generic default to flip;
C = explicit named compatibility/convention recipe to retain; Q = decision needed.
A row marked R does not certify the entire function. Correct-model GOF permits
expected geometry with complete data, but not normal-theory meat under arbitrary
nonnormal sampling; MAR FIML requires observed geometry. Nested and covariance
components cannot assume structural correctness. Expected score metric may stay.

## Coverage and counting

One row per inferential default in hand-written R wrappers and generated
RcppExports.R, plus implicit dispatch rows. Generated `_impl` rows deliberately
show glue defaults separately: changing wrappers alone leaves primitive defaults.
Public `magmaan_core` aliases inherit the corresponding `infer_*` / `_impl`
rows (alias assignments and parallel canonical/implementation group lists in
R/zzz_core.R); aliases are not counted again. All rows are local source findings,
not claims of external validation. Observed-sensitivity score routes for continuous LS and ML2S remain unsupported; an explicit expected bread selects their comparator. C names identify the lavaan function/convention
being reproduced, not a new parity certification. Estimation tuning, robust
outlier-fitting switches, simulation metrics, confidence levels and eigenvalue
cutoffs are excluded: they are not structural-misspecification reference choices.
Caller-supplied matrices/Gamma without defaults and explicitly named NT/expected/
observed/empirical primitives prescribe a component, not a hidden default.

Final classes: R=132, F=0, C=43, Q=0. Historical phase-one counts: R=48, F=46, C=11, Q=70. TASK-77 resolves all 19 moment-choice rows; R still certifies only the listed choice within the stated sampling contract.

## Decision table

| Function | Argument | Current default | Class | Estimand / null | Assessment | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `frontier_sam_impl` | `se` | `"twostep"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Diagnostic case-change/SAM standardization convention, unchanged under decision #5; not a generic inferential covariance guarantee | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_rbm_impl` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_modification_indices_robust` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_modification_indices_robust` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_modification_indices_robust` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_modification_indices_robust` | `information` | `"expected"` | R | Score metric for release test | Expected metric can stay; bread controls sensitivity | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_modification_indices_robust` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_score_tests_robust` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_score_tests_robust` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_score_tests_robust` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_score_tests_robust` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_score_flip_test` | `sensitivity` | `"observed"` | R | Nested restriction at pseudo-true value; larger model may be wrong | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_score_flip_test_model` | `sensitivity` | `"observed"` | R | Nested restriction at pseudo-true value; larger model may be wrong | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_global_score_flip_test` | `sensitivity` | `observed FIML / expected complete` | R | Global correct-model GOF, or reusable covariance/nested geometry | MAR FIML uses observed sensitivity; complete-data global correct-model null permits expected | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `inference_global_score_flip_test` | `metric` | `"expected"` | R | Score quadratic metric; no model-correctness restriction | Expected metric allowed with matching empirical spectrum (caller matrix otherwise) | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `measures_reliability_cov` | `gamma` | `NULL` | R | Sampling covariance / nested spectrum | NULL derives empirical Gamma from raw_data; absent raw data is an error; explicit Gamma remains available | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `measures_reliability_omega_multidim` | `gamma` | `NULL` | R | Sampling covariance / nested spectrum | NULL derives empirical Gamma from raw_data; absent raw data is an error; explicit Gamma remains available | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `fiml_fit_measures_impl` | `robust` | `TRUE` | R | Global GOF / fit-index uncertainty (FIML under MAR) | Model-based default needs empirical sandwich/reference; FIML must use observed | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_inference_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_modindices_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_se_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_wald_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_difference_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_grouped_inference_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_pseudo_lrt_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_constrained_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `noniterative_cfa_scalar_impl` | `gamma` | `"nt"` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ml_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ml_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_gmm_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_gmm_impl` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_gmm_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ordinal_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ordinal_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ml_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ml_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_gmm_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_gmm_impl` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_gmm_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ordinal_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ordinal_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_fiml_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_fiml_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ml2s_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ml2s_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ml2s_impl` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ml2s_nt_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_ml2s_nt_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_mixed_ordinal_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_parameter_mixed_ordinal_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_fiml_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_fiml_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ml2s_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ml2s_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ml2s_impl` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ml2s_nt_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_ml2s_nt_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_mixed_ordinal_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_parameter_mixed_ordinal_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ordinal_polychoric_omega_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ordinal_polychoric_omega_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_ordinal_polychoric_omega_impl` | `robust` | `FALSE` | R | Restriction/interval at pseudo-true value; larger model may be wrong | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `frontier_profile_lrt_ci_ordinal_polychoric_omega_impl` | `reference` | `NULL -> misspec_scaled (robust=FALSE)` | R | Pseudo-true profile restriction or interval | Default MisspecScaled; explicit ordinary and robust_scaled remain comparators; robust=TRUE with NULL retains RobustScaled. Supply raw data for empirical laws | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_continuous_ls_robust` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_continuous_ls_robust` | `gamma` | `"empirical"` | R | Moment/score covariance for covariance, nested or global reference | Empirical meat; robustness still depends on geometry and estimated weights | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_lr_test_satorra2000` | `gamma` | `"empirical"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_continuous_ls_lr_test_satorra2000` | `gamma` | `"empirical"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_fiml_lr_test_satorra2000` | `gamma` | `"empirical"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_fiml_lr_test_satorra2000` | `convention` | `"magmaan"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_ml2s_lr_test_satorra2000` | `gamma` | `"empirical"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_ml2s_lr_test_satorra2000` | `convention` | `"magmaan"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_lr_test_satorra_bentler2001` | `gamma` | `"empirical"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_lr_test_satorra_bentler2010` | `gamma` | `"empirical"` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_build_u_factor` | `bread` | `required` | R | Global correct-model GOF, or reusable covariance/nested geometry | Explicit bread required: caller must choose covariance/nested observed versus correct-model complete-data GOF expected | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_build_u_factor` | `moments` | `"structured"` | C | Correct-model global GOF moment quadratic; explicit convention for other uses | Structured NT metric is a correct-null/SB convention, not a pseudo-true score meat; unstructured is an explicit metric comparator; see derivation below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_build_u_factor_parts` | `bread` | `required` | R | Global correct-model GOF, or reusable covariance/nested geometry | Explicit bread required: caller must choose covariance/nested observed versus correct-model complete-data GOF expected | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_build_u_factor_parts` | `moments` | `"structured"` | C | Correct-model global GOF moment quadratic; explicit convention for other uses | Structured NT metric is a correct-null/SB convention, not a pseudo-true score meat; unstructured is an explicit metric comparator; see derivation below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_build_u_factor_pairwise` | `bread` | `required` | R | Global correct-model GOF, or reusable covariance/nested geometry | Explicit bread required: caller must choose covariance/nested observed versus correct-model complete-data GOF expected | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_test_moments_both_breads_zc` | `moments` | `"structured"` | C | Correct-model global GOF moment quadratic; explicit convention for other uses | Structured NT metric is a correct-null/SB convention, not a pseudo-true score meat; unstructured is an explicit metric comparator; see derivation below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_test_moments_both_breads_gamma` | `moments` | `"structured"` | C | Correct-model global GOF moment quadratic; explicit convention for other uses | Structured NT metric is a correct-null/SB convention, not a pseudo-true score meat; unstructured is an explicit metric comparator; see derivation below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_ordinal_robust` | `bread` | `"ij"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_mixed_ordinal_robust` | `bread` | `"ij"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_ordinal_fit_measures_misspec` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_mixed_ordinal_rmsea_misspec` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_mixed_ordinal_crmr_misspec` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_mixed_ordinal_cfi_tli_misspec` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_mixed_ordinal_fit_measures_misspec` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_parts` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_parts` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_parts` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_raw` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_raw` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_raw` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_raw_parts` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_raw_parts` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_raw_parts` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_zc` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_zc` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_zc` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_both_breads` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_both_breads` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_both_breads_raw` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_both_breads_raw` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_both_breads_zc` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `infer_robust_se_both_breads_zc` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [RcppExports.R](../../r-package/R/RcppExports.R) |
| `est_change` | `se` | `c("standard", "robust.sem", "robust.huber.white")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Diagnostic case-change/SAM standardization convention, unchanged under decision #5; not a generic inferential covariance guarantee | [case_influence.R](../../r-package/R/case_influence.R) |
| `residuals.magmaan_fit` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [context.R](../../r-package/R/context.R) |
| `lav_residuals` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [context.R](../../r-package/R/context.R) |
| `modification_indices_robust` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [context.R](../../r-package/R/context.R) |
| `modification_indices_robust` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [context.R](../../r-package/R/context.R) |
| `modification_indices_robust` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [context.R](../../r-package/R/context.R) |
| `modification_indices_robust` | `information` | `"expected"` | R | Score metric for release test | Expected metric can stay; bread controls sensitivity | [context.R](../../r-package/R/context.R) |
| `modification_indices_robust` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [context.R](../../r-package/R/context.R) |
| `score_tests_robust` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [context.R](../../r-package/R/context.R) |
| `score_tests_robust` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [context.R](../../r-package/R/context.R) |
| `score_tests_robust` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [context.R](../../r-package/R/context.R) |
| `score_tests_robust` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [context.R](../../r-package/R/context.R) |
| `fmg_nested` | `gamma` | `NULL` | C | Sampling covariance / nested spectrum | FMG/semTests replication-compatibility route; retains explicit NT/fixed-weight conventions, not the ordinary policy | [fmg.R](../../r-package/R/fmg.R) |
| `fmg_tests` | `gamma` | `c("empirical", "normal")` | R | Moment/score covariance for covariance, nested or global reference | Empirical meat; robustness still depends on geometry and estimated weights | [fmg.R](../../r-package/R/fmg.R) |
| `fmg_pvalues` | `gamma` | `c("empirical", "normal")` | R | Moment/score covariance for covariance, nested or global reference | Empirical meat; robustness still depends on geometry and estimated weights | [fmg.R](../../r-package/R/fmg.R) |
| `fit_measures` | `robust` | `NULL` | R | Global GOF / fit-index uncertainty (FIML under MAR) | Model-based default needs empirical sandwich/reference; FIML must use observed | [fmg.R](../../r-package/R/fmg.R) |
| `fit_measures_misspec` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [fmg.R](../../r-package/R/fmg.R) |
| `fit_measures_misspec_mixed_ordinal` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [fmg.R](../../r-package/R/fmg.R) |
| `modification_indices_lrt` | `robust` | `TRUE` | R | Nested release restriction | TRUE already; confirm downstream observed/weight law | [mi_lrt.R](../../r-package/R/mi_lrt.R) |
| `score_tests_lrt` | `gamma` | `c("empirical", "NT")` | R | Moment/score covariance for covariance, nested or global reference | Empirical meat; robustness still depends on geometry and estimated weights | [mi_lrt.R](../../r-package/R/mi_lrt.R) |
| `sam` | `se` | `c("twostep", "twostep.robust", "standard", "none")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Diagnostic case-change/SAM standardization convention, unchanged under decision #5; not a generic inferential covariance guarantee | [model_data.R](../../r-package/R/model_data.R) |
| `fit_model` | `se` | `"none"` | R | No inference requested | none is estimate-only, not an incorrect inferential reference | [model_data.R](../../r-package/R/model_data.R) |
| `fit_model` | `test` | `"none"` | R | No inference requested | none is estimate-only | [model_data.R](../../r-package/R/model_data.R) |
| `infer_build_u_factor_fit` | `bread` | `required` | R | Global correct-model GOF, or reusable covariance/nested geometry | Explicit bread required: caller must choose covariance/nested observed versus correct-model complete-data GOF expected | [model_data.R](../../r-package/R/model_data.R) |
| `infer_build_u_factor_fit` | `moments` | `"structured"` | C | Correct-model global GOF moment quadratic; explicit convention for other uses | Structured NT metric is a correct-null/SB convention, not a pseudo-true score meat; unstructured is an explicit metric comparator; see derivation below | [model_data.R](../../r-package/R/model_data.R) |
| `infer_robust_se_fit` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [model_data.R](../../r-package/R/model_data.R) |
| `infer_robust_se_fit` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [model_data.R](../../r-package/R/model_data.R) |
| `infer_robust_se_fit` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [model_data.R](../../r-package/R/model_data.R) |
| `infer_robust_se_raw_fit` | `bread` | `"observed"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [model_data.R](../../r-package/R/model_data.R) |
| `infer_robust_se_raw_fit` | `moments` | `"auto"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Auto selects exact uncentered likelihood projections for empirical ML SE/score meats; explicit structured/unstructured retain moment-weight comparators; LS dispatch retains fitting-weight influence within its stated contract; see gates below | [model_data.R](../../r-package/R/model_data.R) |
| `infer_robust_se_raw_fit` | `cov` | `"empirical"` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Empirical meat already; bread/weight caveats remain | [model_data.R](../../r-package/R/model_data.R) |
| `score_flip_test` | `sensitivity` | `c("observed", "expected")` | R | Nested restriction at pseudo-true value; larger model may be wrong | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [nested_test.R](../../r-package/R/nested_test.R) |
| `global_score_flip_test` | `sensitivity` | `observed FIML / expected complete` | R | Global correct-model GOF, or reusable covariance/nested geometry | MAR FIML uses observed sensitivity; complete-data global correct-model null permits expected | [nested_test.R](../../r-package/R/nested_test.R) |
| `global_score_flip_test` | `metric` | `c("expected", "observed", "observed-h1")` | R | Score quadratic metric; no model-correctness restriction | Expected metric allowed with matching empirical spectrum (caller matrix otherwise) | [nested_test.R](../../r-package/R/nested_test.R) |
| `nested_score_test` | `sensitivity` | `c("observed", "expected")` | R | Nested restriction at pseudo-true value; larger model may be wrong | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [nested_test.R](../../r-package/R/nested_test.R) |
| `robust_nested_lrt` | `gamma` | `c("empirical", "unbiased", "NT", "both")` | R | Moment/score covariance for covariance, nested or global reference | Empirical meat; robustness still depends on geometry and estimated weights | [nested_test.R](../../r-package/R/nested_test.R) |
| `robust_nested_lrt` | `convention` | `c("magmaan", "lavaan")` | R | Nested parameter restrictions; larger model may be wrong | Generic default calls policy_nested for ML/DWLS; typed unsupported condition elsewhere. Explicit method selects compatibility | [nested_test.R](../../r-package/R/nested_test.R) |
| `nestedTest` | `gamma` | `c("empirical", "unbiased", "NT", "both")` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [nested_test.R](../../r-package/R/nested_test.R) |
| `nestedTest` | `convention` | `c("magmaan", "lavaan")` | C | Nested parameter restrictions; larger model may be wrong | Named Satorra convention: lavaan::lavTestLRT; expected/fixed-weight geometry is compatibility, not policy | [nested_test.R](../../r-package/R/nested_test.R) |
| `noniterative_cfa_inference` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_se` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_modification_indices` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_wald` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_difference_test` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_pseudo_lrt` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_grouped_inference` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_constrained` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `noniterative_cfa_scalar_invariance` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative.R](../../r-package/R/noniterative.R) |
| `fit_measures_noniterative` | `gamma` | `c("nt", "empirical")` | C | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Postponed noniterative area: documented NT comparator; not a robust default | [noniterative_postfit.R](../../r-package/R/noniterative_postfit.R) |
| `score_components` | `sensitivity` | `c("observed", "expected", "observed-h1", "observed-shrink-light", "observed-shrink-sqrt")` | R | Nested restriction at pseudo-true value; larger model may be wrong | Use observed sensitivity/bread; LS estimated weights additionally need IJ law | [scores.R](../../r-package/R/scores.R) |
| `score_components` | `metric` | `c("expected", "observed", "observed-h1")` | R | Score quadratic metric; no model-correctness restriction | Expected metric allowed with matching empirical spectrum (caller matrix otherwise) | [scores.R](../../r-package/R/scores.R) |
| `score_quadratic` | `meat` | `required` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Caller supplies meat explicitly; metric remains a free choice | [scores.R](../../r-package/R/scores.R) |
| `inference_information` | `type` | `c("observed", "expected")` | R | Generic curvature used as covariance bread or nested sensitivity | Default observed; explicit expected remains a metric/convention option | [scores.R](../../r-package/R/scores.R) |
| `parameter_covariance` | `meat` | `NULL` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | NULL derives empirical meat from retained fitting observations; explicit meat="model" requests inverse information | [scores.R](../../r-package/R/scores.R) |
| `score_components_from_matrices` | `metric` | `sensitivity` | R | Score quadratic metric; no model-correctness restriction | Expected metric allowed with matching empirical spectrum (caller matrix otherwise) | [scores.R](../../r-package/R/scores.R) |
| `inference_quadratic` | `geometry` | `NULL: observed nested / expected global` | R | Nested restriction at pseudo-true value; larger model may be wrong | Observed nested pseudo-true sensitivity; expected complete-data correct-model GOF | [scores.R](../../r-package/R/scores.R) |
| `inference_covariance` | `robust` | `TRUE` | R | Pseudo-true parameter covariance / Wald or interval; no correct-model assumption | Model-based default needs empirical sandwich/reference; FIML must use observed | [scores.R](../../r-package/R/scores.R) |
| `frontier_rbm` | `estimated_weight` | `TRUE` | R | Sampling influence of fitted weights; no correct-model assumption | TRUE includes weight estimation; does not repair another non-robust default | [zzz_core.R](../../r-package/R/zzz_core.R) |
| `vcov.magmaan_fit` | `regime` | `NULL: observed sandwich / IJ / postponed NT / diagnostic stored` | R | Pseudo-true covariance | ML/FIML observed empirical sandwich; categorical/estimated GLS-WLS IJ; noniterative NT postponed, SAM stored diagnostic | [R](../../r-package/R) |
| `modification_indices / score_tests` | `implicit route` | `observed / empirical / estimated weight` | R | Nested release restriction at pseudo-true value | Generic robust composition; unsupported observed LS/ML2S laws fail, historical unscaled comparator requires explicit model_implied/fixed-weight choice | [R](../../r-package/R) |
| `policy_inference / policy_nested` | `implicit recipe` | `observed ML / IJ DWLS` | R | Covariance; global correct-model null; nested pseudo-true restrictions | Already robust within documented supported scope; unavailable elsewhere | [R](../../r-package/R) |
| `convention_inference / convention_nested` | `convention` | `required named bundle` | C | Bundle-specific covariance/global/nested null | lavaan::lavaan and lavaan::lavTestLRT; preserve explicit ML/MLM/MLR/DWLS/WLSMV/ULS/ULSMV/WLS conventions | [R](../../r-package/R) |
| `fmg_nested` | `implicit LS Gamma` | `ULS NT / GLS,WLS empirical` | C | Nested restrictions at pseudo-true value | FMG/semTests replication-compatibility route; retains explicit NT/fixed-weight conventions, not the ordinary policy | [R](../../r-package/R) |
| `infer_ordinal_robust / infer_mixed_ordinal_robust` | `implicit weight law` | `IJ covariance + expected GOF` | R | Pseudo-true covariance (also returns global GOF spectrum) | Observed bread alone insufficient for estimated DWLS/WLS weights; use IJ covariance, retain valid complete-data global spectrum | [R](../../r-package/R) |
| `infer_continuous_ls_robust` | `implicit weight law` | `fixed fitting weight` | C | Pseudo-true covariance and global spectrum | Caller-fixed-weight law; data-estimated nt/adf/dwls/dls recipes require IJ unless fixed_weight=TRUE explicitly | [R](../../r-package/R) |
| `case_approx_parts / est_change_approx / est_change_raw_approx` | `type` | `standard` | C | Case influence / estimator derivative, not a null test | Diagnostic case-change/SAM standardization convention, unchanged under decision #5; not a generic inferential covariance guarantee | [R](../../r-package/R) |
| `est_change / est_change_raw` | `se` | `standard` | C | Case-change standardization | Diagnostic case-change/SAM standardization convention, unchanged under decision #5; not a generic inferential covariance guarantee | [R](../../r-package/R) |

## Approved decisions and validation

Board comments #5, #8 and #12 approve the flips and comparator arguments.
Noniterative normal-theory development is postponed. SAM and case influence
standardizations remain diagnostics. TASK-77 derives the moment choices below; the actual comparator spelling is
`unstructured`, not `saturated`. Named Satorra,
lavaan bundles and FMG/semTests replication routes remain compatibility APIs.

Generic covariance uses observed bread and empirical meat; estimated weights
require IJ. Generic nested tests use the ML/DWLS policy composer and typed
unsupported reasons elsewhere. Required bread/meat arguments expose choices
that cannot be made safely from the primitive alone. Explicit compatibility
arguments retain all existing oracle expectations and tolerances. The ordinary
package implementation is unchanged.

## TASK-77: estimating equations and consistent meats

`moments="structured"` and `"unstructured"` select M = fitted Sigma or
sample S in W = Gamma_NT(M)^(-1). Both previously projected **sample-centered**
moment influence. They do not choose how observations are centered. Neither
is generally the likelihood-score meat with restricted misspecified means.
`"auto"` now selects `"likelihood"` for empirical complete-data ML SE/score
meats and `"structured"` for explicitly model-implied comparators. Score-table
dispatch selects its existing fitting-weight route for other estimators.
The C++ `InferenceSpec` remains an explicit convention container; its historical
expected/model-implied initialization is not a generic robust recipe.

### ML SE, equality releases and modification indices

Write r_i = x_i - mu(theta), C = Sigma(theta)^(-1), and D_mu, D_Sigma for
parameter derivatives. The estimator solves sum_i score_i(theta) = 0, where

    score_ik = D_mu,k' C r_i
               + 1/2 tr[C (r_i r_i' - Sigma) C D_Sigma,k].

Covariance-only ML profiles unrestricted observed means, so r_i uses the
sample mean there. The pseudo-true joint-sampling meat is E[score_i score_i']
at a stationary target; observed sensitivity H = -E[d score_i/d theta'] gives
H^(-1) B H^(-1)/N. Expected bread remains an explicit geometry comparator.
MI and equality releases use this same B on the augmented parameter space,
then apply their existing nuisance projection and score metric.

Let z_i contain x_i - xbar and vech[(x_i-xbar)(x_i-xbar)' - S]. Put d = xbar -
mu_hat and T = W( Sigma_hat ) Delta. The exact score is z_i' L + c_g, with
L's mean rows augmented by the covariance projection of
(x_i-xbar)d' + d(x_i-xbar)', and

    c_g = d' T_mu + vech(S - Sigma_hat + d d')' T_Sigma.

The shared `likelihood_projection` factors TASK-66's `likelihood_rows`
construction; prepared policy and lab use the same correction. Raw/Zc paths
retain c_g on every group row. Caller Gamma (centered moment NACOV, including
mean/covariance cross blocks when means are fitted) uses L' Gamma L plus
sum_g (n_g/N) c_g c_g'. It must have the documented block n_g/N weighting.
Gamma-only and Zc interfaces therefore need no reconstructed observations.
Zc must be centered sample-moment rows in fitted group order, with the fitting
counts and total N; this is not an arbitrary score matrix API.

The separate group constants are required by the primary **joint sampling**
contract in [scope](../scope.md#group-allocation-and-likelihood-score-covariance).
Centering within groups would instead estimate fixed-allocation covariance.
Structured and exact projected rows coincide when means match and each group's
projected score mean is zero. Structured and unstructured metrics coincide
when both fitted covariance and means match sample moments (full saturation).
Saturated means alone imply neither statement in a covariance-restricted
multi-group model.

### Continuous LS SE, score and MI

For caller-fixed fitting weights the estimating equation is
sum_g w_g Delta_g' W_g [s_g - sigma_g(theta)] = 0, with sample moments
s_g = (xbar_g, vech S_g) or vech S_g alone. Its conditional, fixed-allocation
influence uses z_ig centered at each group's sample moments. The consistent
conditional meat is sum_g w_g Delta_g' W_g Gamma_g W_g Delta_g. Fitted mean
misspecification does not change this moment derivative: replacing xbar_g by
mu_hat in z would be wrong. The empirical route uses the **recorded fitting
weight**, so the structured/unstructured NT comparator does not change it.
For estimated weights, the weight derivative multiplying the residual is also
needed; supported routes retain their existing IJ law. Observed LS score
sensitivity is still unavailable where already documented; expected is an
explicit comparator, not newly certified here.

In multi-group LS these within-group centered meats are a **fixed-allocation**
contract. They do not include random group-composition derivatives when group
estimating-equation means differ. The existing joint-sampling extension remains
a limitation/follow-up; this card changes no LS numbers and does not certify
joint robustness from a stratified gate. Single-group moment influence has no
such allocation distinction. The same qualification applies to current
estimated-weight moment IJ routes.

### U-factor and test-moment builders

These functions construct the global correct-null discrepancy quadratic,
not an estimator's arbitrary pseudo-true score covariance. Linearization at
the correct-model null uses centered sample-moment influence with covariance
Gamma and U = W - W Delta (Delta' W Delta)^(-1) Delta' W (or the explicitly
requested observed-Hessian convention). Test moments reduce U Gamma through
a U-factor. Structured NT W is legitimate here: at the correct null fitted and
population moments agree. Unstructured remains an explicit alternative metric.
The C-class rows retain these primitives without a generic covariance claim;
`likelihood` is rejected as a U-factor convention. Pseudo-true covariance or
nested score consumers must use exact likelihood SE/score construction, or
explicitly own the alternative sampling law.

### Deterministic gates

- `cpp/tests/unit/policy_test.cpp`, restricted-mean covariance case: two groups
  with equal fitted intercepts and unequal population means; raw, Zc and Gamma
  SEs and augmented score meat agree with independently finite-differenced
  casewise log densities and the TASK-66 policy. Both-bread observed SE agrees.
- `r-package/tests/testthat/test_likelihood_moments.R`: saturated model equality;
  single-group free mean/fixed variance equality with structured only; two-group
  free means/shared variance deliberately differs (exact .08, structured 0).
  Group variance-score means are +/- .08 and total information is 4, hence
  (1/4)^2 * 200 * .08^2 = .08, including shared-slot covariance.
- The same lab test differentiates case weights in the exact linear ULS
  sample-moment estimator with unequal group means. Its stratified influence
  cross-product agrees with the centered fixed-weight LS covariance. This gate
  validates fixed allocation only. Existing score/MI oracle gates continue to
  select their explicit comparator conventions.
