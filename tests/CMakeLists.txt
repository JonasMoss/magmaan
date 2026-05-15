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

add_executable(magmaan_tests
  doctest_main.cpp
  unit/version_test.cpp
  unit/lexer_test.cpp
  unit/parser_test.cpp
  unit/lavaanify_test.cpp
  unit/matrix_rep_test.cpp
  unit/model_evaluator_test.cpp
  unit/ml_test.cpp
  unit/fiml_test.cpp
  unit/uls_test.cpp
  unit/gls_wls_test.cpp
  unit/snlls_test.cpp
  unit/concepts_test.cpp
  unit/namespace_alias_test.cpp
  unit/bounds_test.cpp
  unit/lbfgs_optimizer_test.cpp
  unit/lbfgsb_optimizer_test.cpp
  unit/lbfgsb_integration_test.cpp
  unit/ceres_optimizer_test.cpp
  unit/ceres_integration_test.cpp
  unit/ls_path_test.cpp
  unit/constraints_test.cpp
  unit/lin_constraints_test.cpp
  unit/inference_test.cpp
  unit/score_test.cpp
  unit/effects_test.cpp
  unit/standardized_test.cpp
  unit/fit_measures_test.cpp
  unit/raw_data_test.cpp
  unit/ordinal_test.cpp
  unit/robust_test.cpp
  unit/restriction_test.cpp
  unit/satorra2000_test.cpp
  unit/weighted_chisq_test.cpp
  golden/lexer_golden_test.cpp
  golden/parser_golden_test.cpp
  golden/lavaanify_golden_test.cpp
  golden/matrix_rep_golden_test.cpp
  golden/fit_implied_golden_test.cpp
  golden/fit_theta_golden_test.cpp
  golden/ls_golden_test.cpp
  golden/fiml_golden_test.cpp
  golden/inference_golden_test.cpp
  golden/observed_inference_golden_test.cpp
  golden/multigroup_inference_golden_test.cpp
  golden/fit_measures_golden_test.cpp
  golden/ordinal_golden_test.cpp
  golden/std_lv_golden_test.cpp
  golden/standardized_golden_test.cpp
  golden/test_stats_golden_test.cpp
  golden/lin_constraint_golden_test.cpp)

target_link_libraries(magmaan_tests
  PRIVATE
    magmaan::magmaan
    magmaan::warnings
    magmaan::sanitizers
    doctest::doctest
    nlohmann_json::nlohmann_json)

target_compile_definitions(magmaan_tests PRIVATE
  DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS=1
  DOCTEST_CONFIG_USE_STD_HEADERS=1
  JSON_NOEXCEPTION=1
  MAGMAAN_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# CMake's automatic test discovery for doctest.
include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")
doctest_discover_tests(magmaan_tests)
