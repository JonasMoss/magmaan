# NLopt — optional optimizer backend, fetched on demand. Mirrors CeresFetch.cmake.
#
# magmaan only needs NLopt's C-API algorithms (SLSQP for the cross-check now,
# AUGLAG for nonlinear constraints later), so every language binding and the
# NLopt test suite are switched off, and the library is built static so it
# links straight into libmagmaan.a. The knobs are set as cache variables
# BEFORE `magmaan_find_or_fetch` so FetchContent picks them up when NLopt's
# CMakeLists.txt is added to the build.
#
# Set WITHOUT FORCE so a user configuring with their own `-D...` still wins.

set(NLOPT_PYTHON  OFF CACHE BOOL "NLopt Python bindings — not needed")
set(NLOPT_OCTAVE  OFF CACHE BOOL "NLopt Octave bindings — not needed")
set(NLOPT_MATLAB  OFF CACHE BOOL "NLopt Matlab bindings — not needed")
set(NLOPT_GUILE   OFF CACHE BOOL "NLopt Guile bindings — not needed")
set(NLOPT_SWIG    OFF CACHE BOOL "NLopt SWIG wrapper generation — not needed")
set(NLOPT_TESTS   OFF CACHE BOOL "NLopt test suite — off, we run our own")
set(NLOPT_FORTRAN OFF CACHE BOOL "NLopt Fortran bindings — not needed")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static NLopt — linkable into libmagmaan.a")

magmaan_find_or_fetch(NAME NLopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG        v2.10.1
  FIND_PACKAGE_ARGS CONFIG)

# `magmaan_find_or_fetch` may have found a system NLopt (which exports the
# `NLopt::nlopt` target via NLoptConfig.cmake) or fetched + add_subdirectory'd
# it (build-tree target `nlopt`). Normalize to `NLopt::nlopt` so the link line
# in the top-level CMakeLists.txt is the same either way.
if(NOT TARGET NLopt::nlopt AND TARGET nlopt)
  add_library(NLopt::nlopt ALIAS nlopt)
endif()
