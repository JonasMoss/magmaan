cmake_minimum_required(VERSION 3.28)

project(latva
  VERSION      0.0.1
  DESCRIPTION  "C++23 toolkit for linear SEM (lavaan-equivalent core)"
  LANGUAGES    CXX)

# Honor parent project's settings if we're embedded as a subdir.
set(LATVA_IS_TOP_LEVEL FALSE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(LATVA_IS_TOP_LEVEL TRUE)
endif()

option(LATVA_BUILD_TESTS  "Build the latva test suite"      ${LATVA_IS_TOP_LEVEL})
option(LATVA_BUILD_BENCH  "Build the latva benchmarks"      OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(LatvaCompilerCheck)
include(LatvaWarnings)
include(LatvaNoExceptNoRtti)
include(LatvaSanitizers)
include(FindOrFetch)

# Eigen — math backbone for ModelEvaluator and beyond. Built with
# EIGEN_NO_EXCEPTIONS (set in latva::no_exc_no_rtti) so any internal
# eigen_assert under -fno-exceptions routes through std::abort rather
# than throw. We pin to 3.4.0.
latva_find_or_fetch(NAME Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  FIND_PACKAGE_ARGS CONFIG)

# LBFGS++ — header-only L-BFGS solver. We use the standard unconstrained
# `LBFGSSolver`. lbfgspp depends on Eigen (which we just fetched).
latva_find_or_fetch(NAME lbfgspp
  GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
  GIT_TAG        v0.4.0
  FIND_PACKAGE_ARGS CONFIG)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(latva STATIC
  src/util/version.cpp
  src/parse/lexer.cpp
  src/parse/parser.cpp
  src/partable/lavaanify.cpp
  src/model/matrix_rep.cpp
  src/model/model_evaluator.cpp
  src/fit/ml.cpp
  src/fit/start_values.cpp
  src/fit/lbfgs_optimizer.cpp
  src/fit/resolve_fixed_x.cpp
  src/fit/constraints.cpp
  src/fit/inference.cpp
  src/fit/effects.cpp
  src/fit/standardized.cpp
  src/fit/fit_measures.cpp
  src/fit/raw_data.cpp
  src/fit/robust.cpp)
add_library(latva::latva ALIAS latva)

target_compile_features(latva PUBLIC cxx_std_23)
set_target_properties(latva PROPERTIES
  CXX_EXTENSIONS         OFF
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(latva
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(latva
  PUBLIC  latva::no_exc_no_rtti Eigen3::Eigen
  PRIVATE latva::warnings latva::sanitizers lbfgspp)

# LBFGS++ has `throw std::invalid_argument` in its Param check. GCC under
# -fno-exceptions refuses to compile any throw statement (Clang elides it
# silently). Compile just this TU with -fexceptions; we never invoke the
# throw path with our default parameters, but the binary needs to accept
# the source.
set_source_files_properties(src/fit/lbfgs_optimizer.cpp
  PROPERTIES COMPILE_OPTIONS "-fexceptions")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(LATVA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(LATVA_BUILD_BENCH)
  add_subdirectory(benchmarks)
endif()
