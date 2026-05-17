# CppNumericalSolvers — header-only Newton trust-region solver, fetched + patched.
#
# We always FetchContent this dependency and never accept a system copy:
# v2.0.0's TrustRegionNewton has a bug — InitializeSolver() records the problem
# dimension and then calls ResetInternal(), which zeroes it again, so the
# CG-Steihaug subproblem allocates a size-0 step vector and aborts on any
# dynamically-sized problem (which is every magmaan problem). The PATCH_COMMAND
# below reorders those two statements. A system copy would silently
# reintroduce the bug, so a controlled, patched fetch is the only safe path.

include(FetchContent)

# Skip the library's install() / export(PACKAGE) rules — we consume it in-tree,
# and export(PACKAGE) would pollute the user's ~/.cmake package registry.
set(CppNumericalSolvers_INSTALL OFF CACHE BOOL "" FORCE)

# PATCH_COMMAND is re-run by FetchContent on every reconfigure, so it must be
# idempotent: restore the target header to its pristine committed state first,
# then apply the patch. A bare `git apply` would fail the second time round
# ("patch does not apply") because the change is already present.
set(_cppnsolvers_patch
  "${CMAKE_CURRENT_LIST_DIR}/patches/cppnumericalsolvers-v2.0.0-trust-region-dim.patch")

FetchContent_Declare(cppnumericalsolvers
  GIT_REPOSITORY https://github.com/PatWie/CppNumericalSolvers.git
  GIT_TAG        v2.0.0
  GIT_SHALLOW    TRUE
  SYSTEM
  EXCLUDE_FROM_ALL
  PATCH_COMMAND  sh -c
    "git checkout -- include/cppoptlib/solver/trust_region_newton.h && git apply --ignore-whitespace '${_cppnsolvers_patch}'")

FetchContent_MakeAvailable(cppnumericalsolvers)

# The fetched build exposes the plain `CppNumericalSolvers` interface target;
# normalize to the namespaced spelling the top-level link line uses.
if(NOT TARGET CppNumericalSolvers::CppNumericalSolvers
   AND TARGET CppNumericalSolvers)
  add_library(CppNumericalSolvers::CppNumericalSolvers
              ALIAS CppNumericalSolvers)
endif()
