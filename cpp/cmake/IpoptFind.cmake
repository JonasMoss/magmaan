# IPOPT — optional system dependency.
#
# Prefer a CMake package if the install provides one, otherwise fall back to
# pkg-config's imported target. Do not fetch/build IPOPT here: its BLAS/LAPACK
# and sparse linear solver stack is intentionally left to the local toolchain.

find_package(Ipopt CONFIG QUIET)

if(TARGET Ipopt::ipopt)
  set(MAGMAAN_IPOPT_TARGET Ipopt::ipopt)
elseif(TARGET IPOPT::IPOPT)
  set(MAGMAAN_IPOPT_TARGET IPOPT::IPOPT)
elseif(TARGET ipopt)
  set(MAGMAAN_IPOPT_TARGET ipopt)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)
  set(MAGMAAN_IPOPT_TARGET PkgConfig::IPOPT)
endif()

message(STATUS "magmaan: using system IPOPT")
