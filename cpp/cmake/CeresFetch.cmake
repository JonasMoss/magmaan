# Ceres Solver — optional optimizer backend, fetched on demand.
#
# Set the minimum-needed Ceres build options BEFORE calling
# `magmaan_find_or_fetch` so FetchContent picks them up when Ceres'
# CMakeLists.txt is added to the build. Defaults chosen to keep the Ceres
# build small + fast:
#
#   • MINIGLOG=ON, GFLAGS=OFF — bundled tiny glog; no glog/gflags system deps.
#   • SUITESPARSE / CXSPARSE / ACCELERATESPARSE / LAPACK = OFF — Eigen's
#     dense + sparse solvers are enough for the trust-region + line-search
#     usage we do. Skipping LAPACK avoids dragging in a system BLAS/LAPACK.
#   • EIGENSPARSE=ON — keep at least one sparse direct solver available.
#   • SCHUR_SPECIALIZATIONS=OFF — big compile-time cut; we don't have
#     bundle-adjustment-style sparse Hessians.
#   • BUILD_TESTING / EXAMPLES / BENCHMARKS / DOCUMENTATION / SHARED_LIBS =
#     OFF — only the Ceres library, statically linkable into libmagmaan.a.
#   • USE_CUDA=OFF, EXPORT_BUILD_DIR=OFF — keep the fetched build hermetic.
#
# All set as cache variables WITHOUT FORCE so a user that's invoking cmake
# with `-DSUITESPARSE=ON` (e.g. wants the system SuiteSparse) can still
# override. The defaults just steer the common case.

set(MINIGLOG                 ON  CACHE BOOL "miniglog (bundled) — avoids glog system dep")
set(GFLAGS                   OFF CACHE BOOL "gflags command-line parsing — not needed for the lib")
set(SUITESPARSE              OFF CACHE BOOL "SuiteSparse direct solvers — heavy dep, skip")
set(CXSPARSE                 OFF CACHE BOOL "CXSparse direct solver — not used")
set(ACCELERATESPARSE         OFF CACHE BOOL "Apple Accelerate sparse — macOS-only, off here")
set(LAPACK                   OFF CACHE BOOL "Dense LAPACK direct solver — skip to avoid system BLAS dep")
set(EIGENSPARSE              ON  CACHE BOOL "Eigen's sparse direct solver — Eigen we already have")
set(SCHUR_SPECIALIZATIONS    OFF CACHE BOOL "Schur eliminator template specializations — cuts compile time")
set(USE_CUDA                 OFF CACHE BOOL "CUDA acceleration — not needed")
set(BUILD_TESTING            OFF CACHE BOOL "Ceres test suite — off, we run our own")
set(BUILD_EXAMPLES           OFF CACHE BOOL "Ceres example programs — off")
set(BUILD_BENCHMARKS         OFF CACHE BOOL "Ceres benchmarks — off")
set(BUILD_DOCUMENTATION      OFF CACHE BOOL "Ceres Sphinx docs — off")
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "Static library — linkable into libmagmaan.a")
set(EXPORT_BUILD_DIR         OFF CACHE BOOL "Don't export Ceres' build tree to the user package registry")
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Don't define a global `uninstall` target")

magmaan_find_or_fetch(NAME Ceres
  GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
  GIT_TAG        2.2.0
  FIND_PACKAGE_ARGS CONFIG)
