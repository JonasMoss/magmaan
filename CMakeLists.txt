cmake_minimum_required(VERSION 3.28)

project(magmaan
  VERSION      0.0.1
  DESCRIPTION  "C++23 toolkit for linear SEM (lavaan-equivalent core)"
  LANGUAGES    CXX)

# Honor parent project's settings if we're embedded as a subdir.
set(MAGMAAN_IS_TOP_LEVEL FALSE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(MAGMAAN_IS_TOP_LEVEL TRUE)
endif()

option(MAGMAAN_BUILD_TESTS  "Build the magmaan test suite"      ${MAGMAAN_IS_TOP_LEVEL})
option(MAGMAAN_BUILD_BENCH  "Build the magmaan benchmarks"      OFF)
option(MAGMAAN_WITH_CERES   "Enable the Ceres Solver optimizer backend (optional, heavy first build)"  OFF)
option(MAGMAAN_WITH_NLOPT   "Enable the NLopt optimizer backend (SLSQP cross-check; augmented-Lagrangian-ready)"  OFF)

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

# LBFGS++ — header-only L-BFGS solver. We use the standard unconstrained
# `LBFGSSolver`. lbfgspp depends on Eigen (which we just fetched).
magmaan_find_or_fetch(NAME lbfgspp
  GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
  GIT_TAG        v0.4.0
  FIND_PACKAGE_ARGS CONFIG)

# CppNumericalSolvers — header-only, Eigen-native optimizer collection. We use
# its Newton trust-region solver as an independent cross-check of the L-BFGS
# optima (lavaan's nlminb is itself a trust region). v2.0.0 is fetched and
# patched (a TrustRegionNewton dimension bug); see cmake/CppNumericalSolversFetch.cmake.
include(CppNumericalSolversFetch)

# Ceres Solver — optional alternative optimizer backend, behind MAGMAAN_WITH_CERES.
# OFF by default because Ceres is a heavy first build (~5–15 min via FetchContent).
# The Ceres-specific build flags (no LAPACK / SuiteSparse / glog / gflags /
# examples / tests / docs / Schur specializations / shared libs) live in
# `cmake/CeresFetch.cmake` so the top-level CMakeLists stays tidy.
if(MAGMAAN_WITH_CERES)
  include(CeresFetch)
endif()

# NLopt — optional optimizer backend, behind MAGMAAN_WITH_NLOPT. Lightweight C
# library; the SLSQP algorithm gives a sequential-quadratic-programming
# cross-check, and AUGLAG is reserved for future nonlinear-constraint support.
# Build knobs live in `cmake/NloptFetch.cmake`, mirroring CeresFetch.cmake.
if(MAGMAAN_WITH_NLOPT)
  include(NloptFetch)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(magmaan STATIC
  src/util/version.cpp
  src/parse/lexer.cpp
  src/parse/parser.cpp
  src/spec/build.cpp
  src/spec/lin_constraints.cpp
  src/compat/lavaan/partable_view.cpp
  src/model/matrix_rep.cpp
  src/model/model_evaluator.cpp
  src/data/raw_data.cpp
  src/data/shrinkage.cpp
  src/data/h_score.cpp
  src/data/pairwise_mixed.cpp
  src/data/pairwise_ordinal.cpp
  src/data/ordinal.cpp
  src/estimate/fit.cpp
  src/estimate/nt.cpp
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
  src/estimate/gmm/moment_quadratic.cpp
  src/estimate/gmm/gp.cpp
  src/estimate/gmm/dls_weight.cpp
  src/api/sem.cpp
  src/inference/inference.cpp
  src/inference/score.cpp
  src/robust/robust.cpp
  src/robust/restriction.cpp
  src/robust/satorra2000.cpp
  src/robust/lr_test_satorra.cpp
  src/robust/weighted_chisq.cpp
  src/robust/weighted_inference.cpp
  src/measures/fit_measures.cpp
  src/measures/standardized.cpp
  src/measures/effects.cpp
  src/optim/lbfgs_optimizer.cpp
  src/optim/lbfgsb_optimizer.cpp
  src/optim/trust_region_optimizer.cpp
  src/optim/optimizers.cpp)
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
  PRIVATE magmaan::warnings magmaan::sanitizers lbfgspp
          CppNumericalSolvers::CppNumericalSolvers)

# LBFGS++ has `throw std::invalid_argument` in its Param check. GCC under
# -fno-exceptions refuses to compile any throw statement (Clang elides it
# silently). Compile just this TU with -fexceptions; we never invoke the
# throw path with our default parameters, but the binary needs to accept
# the source.
set_source_files_properties(src/optim/lbfgs_optimizer.cpp
  PROPERTIES COMPILE_OPTIONS "-fexceptions")
# LBFGS-B adapter — same rationale: LBFGSBSolver's More–Thuente line search
# throws std::runtime_error on exhaustion, and LBFGSBParam::check_param
# can throw std::invalid_argument. Compile this TU with -fexceptions so
# we can catch at the boundary; the rest of magmaan stays -fno-exceptions.
set_source_files_properties(src/optim/lbfgsb_optimizer.cpp
  PROPERTIES COMPILE_OPTIONS "-fexceptions")
# Trust-region adapter — CppNumericalSolvers is header-only and compiled into
# this TU; like the LBFGS++ TUs it may instantiate code that `throw`s, which
# GCC rejects under -fno-exceptions. Compile just this TU with -fexceptions.
set_source_files_properties(src/optim/trust_region_optimizer.cpp
  PROPERTIES COMPILE_OPTIONS "-fexceptions")

# Ceres optimizer adapter — only compiled when MAGMAAN_WITH_CERES=ON. Mirrors
# the LBFGS++ pattern: TU is `-fexceptions` (Ceres uses exceptions internally,
# we catch them at the boundary and return FitError values), and a PUBLIC
# compile definition lets downstream code (tests, R bindings) detect the
# build configuration via `#ifdef MAGMAAN_WITH_CERES`.
if(MAGMAAN_WITH_CERES)
  target_sources(magmaan PRIVATE src/optim/ceres_optimizer.cpp)
  target_link_libraries(magmaan PRIVATE Ceres::ceres)
  target_compile_definitions(magmaan PUBLIC MAGMAAN_WITH_CERES)
  set_source_files_properties(src/optim/ceres_optimizer.cpp
    PROPERTIES COMPILE_OPTIONS "-fexceptions")
endif()

# NLopt optimizer adapter — only compiled when MAGMAAN_WITH_NLOPT=ON. The
# adapter uses NLopt's C API (`<nlopt.h>`, return-code based), so unlike the
# Ceres TU it needs no -fexceptions. A PUBLIC compile definition lets
# downstream code (tests, R bindings) detect the build via `#ifdef`.
if(MAGMAAN_WITH_NLOPT)
  target_sources(magmaan PRIVATE src/optim/nlopt_optimizer.cpp)
  target_link_libraries(magmaan PRIVATE NLopt::nlopt)
  target_compile_definitions(magmaan PUBLIC MAGMAAN_WITH_NLOPT)
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
