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

# Ceres Solver — optional alternative optimizer backend, behind MAGMAAN_WITH_CERES.
# OFF by default because Ceres is a heavy first build (~5–15 min via FetchContent).
# The Ceres-specific build flags (no LAPACK / SuiteSparse / glog / gflags /
# examples / tests / docs / Schur specializations / shared libs) live in
# `cmake/CeresFetch.cmake` so the top-level CMakeLists stays tidy.
if(MAGMAAN_WITH_CERES)
  include(CeresFetch)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(magmaan STATIC
  src/util/version.cpp
  src/parse/lexer.cpp
  src/parse/parser.cpp
  src/spec/lavaanify.cpp
  src/spec/lin_constraints.cpp
  src/lavaan/partable_view.cpp
  src/model/matrix_rep.cpp
  src/model/model_evaluator.cpp
  src/nt/ml.cpp
  src/gls/uls.cpp
  src/gls/gls_wls.cpp
  src/estimate/bounds.cpp
  src/estimate/ordinal.cpp
  src/estimate/start_values.cpp
  src/optim/lbfgs_optimizer.cpp
  src/optim/lbfgsb_optimizer.cpp
  src/estimate/resolve_fixed_x.cpp
  src/estimate/constraints.cpp
  src/nt/inference.cpp
  src/nt/effects.cpp
  src/nt/standardized.cpp
  src/nt/fit_measures.cpp
  src/data/raw_data.cpp
  src/data/ordinal.cpp
  src/nt/robust.cpp
  src/nt/restriction.cpp
  src/nt/satorra2000.cpp
  src/nt/lr_test_satorra.cpp
  src/nt/weighted_chisq.cpp)
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
  PRIVATE magmaan::warnings magmaan::sanitizers lbfgspp)

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
