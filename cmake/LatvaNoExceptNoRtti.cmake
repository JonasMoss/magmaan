# INTERFACE target carrying the no-exceptions / no-RTTI compile model and
# the matching Eigen configuration macros. Linked PUBLIC by latva so that
# every consumer sees the same Eigen behaviour.

add_library(latva_no_exc_no_rtti INTERFACE)
add_library(latva::no_exc_no_rtti ALIAS latva_no_exc_no_rtti)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(latva_no_exc_no_rtti INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti
                              -fvisibility=hidden -fvisibility-inlines-hidden>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(latva_no_exc_no_rtti INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/EHs-c- /GR->)
  target_compile_definitions(latva_no_exc_no_rtti INTERFACE _HAS_EXCEPTIONS=0)
endif()

# Eigen configuration. These are macro flags only; they cost nothing if Eigen
# is not transitively included. They are PUBLIC because Eigen types may appear
# in latva's public API and consumers must agree on the configuration.
target_compile_definitions(latva_no_exc_no_rtti INTERFACE
  EIGEN_NO_EXCEPTIONS=1
  EIGEN_NO_AUTOMATIC_RESIZING=1
  EIGEN_RUNTIME_NO_MALLOC=1
  EIGEN_DONT_PARALLELIZE=1
  EIGEN_MAX_ALIGN_BYTES=64
  $<$<CONFIG:Debug>:EIGEN_INITIALIZE_MATRICES_BY_NAN=1>)
