# Tests are built with the same -fno-exceptions -fno-rtti compile model as
# magmaan itself. doctest is built with DOCTEST_CONFIG_NO_EXCEPTIONS so its
# CHECK_THROWS-style assertions vanish.

magmaan_find_or_fetch(NAME doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
  FIND_PACKAGE_ARGS CONFIG)

# Test-only JSON dependency. Used by the fixture loader and the regen
# mode. JSON_NOEXCEPTION makes nlohmann/json compatible with our
# -fno-exceptions build (errors std::abort instead of throwing; the
# loader uses the non-throwing parse(..., allow_exceptions=false) API
# so we never reach an abort path on bad fixture JSON).
magmaan_find_or_fetch(NAME nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  FIND_PACKAGE_ARGS CONFIG)

include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

set(MAGMAAN_TEST_TARGETS "")

function(magmaan_add_test_target target label)
  add_executable(${target}
    doctest_main.cpp
    ${ARGN})

  target_link_libraries(${target}
    PRIVATE
      magmaan::magmaan
      magmaan::warnings
      magmaan::sanitizers
      doctest::doctest
      nlohmann_json::nlohmann_json)

  target_compile_definitions(${target} PRIVATE
    DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS=1
    DOCTEST_CONFIG_USE_STD_HEADERS=1
    JSON_NOEXCEPTION=1
    MAGMAAN_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

  doctest_discover_tests(${target}
    TEST_PREFIX "${label}: "
    PROPERTIES
      LABELS "${label}"
      ENVIRONMENT "LSAN_OPTIONS=detect_leaks=0")

  set(MAGMAAN_TEST_TARGETS ${MAGMAAN_TEST_TARGETS} ${target} PARENT_SCOPE)
endfunction()

magmaan_add_test_target(magmaan_test_smoke smoke
  unit/version_test.cpp)

magmaan_add_test_target(magmaan_test_spec spec
  unit/lexer_test.cpp
  unit/parser_test.cpp
  unit/lavaanify_test.cpp
  unit/composite_expand_test.cpp
  unit/composite_fold_test.cpp
  unit/matrix_rep_test.cpp
  unit/model_evaluator_test.cpp
  unit/fcsem_evaluator_test.cpp
  unit/auto_identification_test.cpp
  golden/lexer_golden_test.cpp
  golden/parser_golden_test.cpp
  golden/lavaanify_golden_test.cpp
  golden/matrix_rep_golden_test.cpp
  golden/fit_implied_golden_test.cpp)

set(MAGMAAN_ESTIMATE_TEST_SOURCES
  unit/ml_test.cpp
  unit/ml_continuation_test.cpp
  unit/fcsem_ml_test.cpp
  unit/fiml_test.cpp
  unit/pairwise_gls_test.cpp
  unit/start_fabin_test.cpp
  unit/start_cfa_noniter_test.cpp
  unit/snlls_test.cpp
  unit/bounds_test.cpp
  unit/stabilizer_bounds_test.cpp
  unit/terminal_audit_test.cpp
  unit/fit_diagnostics_test.cpp
  unit/ceres_optimizer_test.cpp
  unit/ceres_integration_test.cpp
  unit/nlopt_optimizer_test.cpp
  unit/ipopt_optimizer_test.cpp
  unit/port_optimizer_test.cpp
  unit/optimizer_crosscheck_test.cpp
  unit/ls_path_test.cpp
  unit/constraints_test.cpp
  unit/lin_constraints_test.cpp
  unit/property_test.cpp
  unit/raw_data_test.cpp
  unit/shrinkage_test.cpp
  unit/dls_weight_test.cpp
  unit/structured_gamma_weight_test.cpp
  golden/fit_theta_golden_test.cpp
  golden/ls_golden_test.cpp
  golden/fiml_golden_test.cpp
  golden/lin_constraint_golden_test.cpp)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/convergence_diagnostics_test.cpp")
  list(APPEND MAGMAAN_ESTIMATE_TEST_SOURCES
    unit/convergence_diagnostics_test.cpp)
endif()

magmaan_add_test_target(magmaan_test_estimate estimate
  ${MAGMAAN_ESTIMATE_TEST_SOURCES})

magmaan_add_test_target(magmaan_test_inference inference
  unit/inference_test.cpp
  unit/score_test.cpp
  unit/effects_test.cpp
  unit/composite_weights_test.cpp
  unit/standardized_test.cpp
  unit/fit_measures_test.cpp
  unit/residuals_test.cpp
  unit/factor_scores_test.cpp
  unit/robust_test.cpp
  unit/pairwise_casewise_test.cpp
  unit/fmg_test.cpp
  unit/weighted_inference_test.cpp
  unit/restriction_test.cpp
  unit/satorra2000_test.cpp
  unit/weighted_chisq_test.cpp
  golden/inference_golden_test.cpp
  golden/score_golden_test.cpp
  golden/observed_inference_golden_test.cpp
  golden/multigroup_inference_golden_test.cpp
  golden/fit_measures_golden_test.cpp
  golden/composite_golden_test.cpp
  golden/std_lv_golden_test.cpp
  golden/standardized_golden_test.cpp
  golden/test_stats_golden_test.cpp)

magmaan_add_test_target(magmaan_test_ordinal ordinal
  unit/ordinal_test.cpp
  golden/ordinal_golden_test.cpp
  golden/pairwise_golden_test.cpp)

magmaan_add_test_target(magmaan_test_api api
  unit/api_sem_test.cpp)

# Depth coverage: real-data lavaan-parity goldens. Its own executable because
# it fits the full real datasets (largest is bfi at 2436×25) — heavier than
# the synthetic-corpus tests, and isolable via `ctest -L parity`.
magmaan_add_test_target(magmaan_test_parity parity
  golden/lavaan_parity_golden_test.cpp
  golden/geiser_golden_test.cpp
  golden/mplus_sem_golden_test.cpp
  golden/paper_corpus_golden_test.cpp
  golden/textbook_corpus_golden_test.cpp)

# Robust-polychoric sanity suite: magmaan's bivariate ordinal estimators vs
# the canonical robcat R package. Frozen JSON fixtures (tests/fixtures/robcat/,
# regenerated by tests/tools/regen_robcat_fixtures.R) — no R at run time.
magmaan_add_test_target(magmaan_test_robcat robcat
  golden/robcat_parity_golden_test.cpp)

add_custom_target(magmaan_tests DEPENDS ${MAGMAAN_TEST_TARGETS})
