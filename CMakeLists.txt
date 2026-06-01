cmake_minimum_required(VERSION 3.28)

project(magmaan
  VERSION      0.0.1
  DESCRIPTION  "C++23 toolkit for linear SEM (lavaan-equivalent core)"
  LANGUAGES    CXX C)
# C is enabled for the vendored PORT (Bell Labs) optimizer routines under
# third_party/port/; see cmake/PortVendor.cmake.

# Honor parent project's settings if we're embedded as a subdir.
set(MAGMAAN_IS_TOP_LEVEL FALSE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(MAGMAAN_IS_TOP_LEVEL TRUE)
endif()

option(MAGMAAN_BUILD_TESTS  "Build the magmaan test suite"      ${MAGMAAN_IS_TOP_LEVEL})
option(MAGMAAN_BUILD_BENCH  "Build the magmaan benchmarks"      OFF)
option(MAGMAAN_WITH_CERES   "Enable the Ceres Solver optimizer backend (optional, heavy first build)"  OFF)
option(MAGMAAN_WITH_PORT    "Enable the vendored PORT (Bell Labs) optimizer backend — nlminb-equivalent trust region (TOMS 611) plus NL2SOL with bounds (TOMS 573)"  ON)
option(MAGMAAN_WITH_IPOPT   "Enable the IPOPT interior-point optimizer backend (optional system dependency)" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(MagmaanCompilerCheck)
include(MagmaanWarnings)
include(MagmaanNoExceptNoRtti)
include(MagmaanSanitizers)
include(FindOrFetch)

# Eigen — math backbone for ModelEvaluator and beyond. Built with
# EIGEN_NO_EXCEPTIONS (set in magmaan::no_exc_no_rtti) so any internal
# eigen_assert under -fno-exceptions routes through std::abort rather
# than throw. We pin to 3.4.0.
magmaan_find_or_fetch(NAME Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  FIND_PACKAGE_ARGS CONFIG)

# Ceres Solver — optional alternative optimizer backend, behind MAGMAAN_WITH_CERES.
# OFF by default because Ceres is a heavy first build (~5–15 min via FetchContent).
# The Ceres-specific build flags (no LAPACK / SuiteSparse / glog / gflags /
# examples / tests / docs / Schur specializations / shared libs) live in
# `cmake/CeresFetch.cmake` so the top-level CMakeLists stays tidy.
if(MAGMAAN_WITH_CERES)
  include(CeresFetch)
endif()

# IPOPT — optional system-provided interior-point optimizer. We deliberately do
# not FetchContent IPOPT: its BLAS/LAPACK and sparse-linear-solver stack is a
# toolchain choice, so enabling this backend means pointing CMake/pkg-config at
# an installed IPOPT.
if(MAGMAAN_WITH_IPOPT)
  include(IpoptFind)
endif()

# NLopt — required scalar optimizer backend. Lightweight C library; its L-BFGS
# algorithm is the current default, while SLSQP / BOBYQA / TNEWTON / VAR2 give
# optimizer-family cross-checks.
# Build knobs live in `cmake/NloptFetch.cmake`, mirroring CeresFetch.cmake.
include(NloptFetch)

# PORT — vendored Bell Labs optimizer routines (f2c-translated C), behind
# MAGMAAN_WITH_PORT. Default ON because the vendored source is small (~620 KB),
# self-contained (no external Fortran or libf2c dependency), BSD-3 licensed,
# and adds the nlminb/nls trust-region algorithms lavaan itself uses. Build
# knobs live in `cmake/PortVendor.cmake`; vendor manifest in
# `third_party/port/README.md`.
if(MAGMAAN_WITH_PORT)
  include(PortVendor)
endif()

# QUADPACK — vendored `qagi` adaptive integrator (f2c-translated C, public
# domain). Always built: `imhof_upper` (the weighted-chi-square tail behind every
# SB/pEBA/pOLS p-value and the robust nested tests) depends on it. Knobs in
# `cmake/QuadpackVendor.cmake`; vendor manifest in `third_party/quadpack/README.md`.
include(QuadpackVendor)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(magmaan STATIC
  src/util/version.cpp
  src/parse/lexer.cpp
  src/parse/parser.cpp
  src/spec/build.cpp
  src/spec/composite_expand.cpp
  src/spec/lin_constraints.cpp
  src/compat/lavaan/partable_view.cpp
  src/compat/lavaan/composite_fold.cpp
  src/model/matrix_rep.cpp
  src/model/model_evaluator.cpp
  src/model/fcsem_evaluator.cpp
  src/model/auto_identification.cpp
  src/data/raw_data.cpp
  src/data/pairwise_cov.cpp
  src/data/shrinkage.cpp
  src/data/h_score.cpp
  src/data/pairwise_mixed.cpp
  src/data/pairwise_ordinal.cpp
  src/data/ordinal.cpp
  src/estimate/fit.cpp
  src/estimate/evaluate.cpp
  src/estimate/backend_strings.cpp
  src/estimate/nt.cpp
  src/estimate/ml_continuation.cpp
  src/estimate/fiml.cpp
  src/estimate/reparameterize.cpp
  src/estimate/bounds.cpp
  src/estimate/ordinal.cpp
  src/estimate/pairwise.cpp
  src/estimate/start_values.cpp
  src/estimate/start_fabin.cpp
  src/estimate/start_guttman.cpp
  src/estimate/start_bentler.cpp
  src/estimate/start_jamesstein.cpp
  src/estimate/cfa_utils.cpp
  src/estimate/resolve_fixed_x.cpp
  src/estimate/constraints.cpp
  src/estimate/expr_eval.cpp
  src/estimate/nl_constraints.cpp
  src/estimate/diagnostics.cpp
  src/estimate/gmm/moment_quadratic.cpp
  src/estimate/gmm/gp.cpp
  src/estimate/gmm/dls_weight.cpp
  src/estimate/gmm/structured_gamma_weight.cpp
  src/api/sem.cpp
  src/inference/inference.cpp
  src/inference/score.cpp
  src/robust/robust.cpp
  src/robust/fmg.cpp
  src/robust/restriction.cpp
  src/robust/satorra2000.cpp
  src/robust/lr_test_satorra.cpp
  src/robust/weighted_chisq.cpp
  src/robust/weighted_inference.cpp
  src/sim/norta.cpp
  src/sim/vale_maurelli.cpp
  src/measures/fit_measures.cpp
  src/measures/standardized.cpp
  src/measures/effects.cpp
  src/measures/composite_weights.cpp
  src/measures/residuals.cpp
  src/measures/factor_scores.cpp
  src/optim/nlopt_optimizer.cpp
  src/optim/optimizers.cpp
  src/optim/terminal_audit.cpp)
add_library(magmaan::magmaan ALIAS magmaan)

target_compile_features(magmaan PUBLIC cxx_std_23)
set_target_properties(magmaan PROPERTIES
  CXX_EXTENSIONS         OFF
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(magmaan
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(magmaan
  PUBLIC  magmaan::no_exc_no_rtti Eigen3::Eigen
  PRIVATE magmaan::warnings magmaan::sanitizers NLopt::nlopt quadpack)

# Ceres optimizer adapter — only compiled when MAGMAAN_WITH_CERES=ON. Mirrors
# the old exception-isolated optimizer pattern: TU is `-fexceptions` (Ceres
# uses exceptions internally, we catch them at the boundary and return FitError
# values), and a PUBLIC compile definition lets downstream code (tests, R
# bindings) detect the build configuration via `#ifdef MAGMAAN_WITH_CERES`.
if(MAGMAAN_WITH_CERES)
  target_sources(magmaan PRIVATE src/optim/ceres_optimizer.cpp)
  target_link_libraries(magmaan PRIVATE Ceres::ceres)
  target_compile_definitions(magmaan PUBLIC MAGMAAN_WITH_CERES)
  set_source_files_properties(src/optim/ceres_optimizer.cpp
    PROPERTIES COMPILE_OPTIONS "-fexceptions")
endif()

# PORT optimizer adapter — only compiled when MAGMAAN_WITH_PORT=ON. PORT is a
# pure C library (f2c-translated Fortran) with no exceptions on either side
# of the boundary, so unlike the Ceres adapter this TU does not need
# -fexceptions. The `port` static library is defined by `cmake/PortVendor.cmake`.
# A PUBLIC compile definition lets downstream code (tests, R bindings) detect
# the build via `#ifdef MAGMAAN_WITH_PORT`.
if(MAGMAAN_WITH_PORT)
  target_sources(magmaan PRIVATE src/optim/port_optimizer.cpp)
  target_link_libraries(magmaan PRIVATE port)
  target_compile_definitions(magmaan PUBLIC MAGMAAN_WITH_PORT)
endif()

# IPOPT optimizer adapter — only compiled when MAGMAAN_WITH_IPOPT=ON. The
# adapter uses IPOPT's C interface and limited-memory Hessian approximation.
if(MAGMAAN_WITH_IPOPT)
  target_sources(magmaan PRIVATE src/optim/ipopt_optimizer.cpp)
  target_link_libraries(magmaan PRIVATE ${MAGMAAN_IPOPT_TARGET})
  target_compile_definitions(magmaan PUBLIC MAGMAAN_WITH_IPOPT)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(MAGMAAN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(MAGMAAN_BUILD_BENCH)
  add_subdirectory(benchmarks)
endif()
