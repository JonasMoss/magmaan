cmake_minimum_required(VERSION 3.25)
project(corpus_domain_probe LANGUAGES CXX)
get_filename_component(MAGMAAN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(MAGMAAN_BUILD "${MAGMAAN_ROOT}/cpp/build/opt" CACHE PATH "Prebuilt optimized magmaan")
find_path(EIGEN_INCLUDE_DIR Eigen/Core REQUIRED
  HINTS "${MAGMAAN_BUILD}/_deps/eigen3-src" ENV EIGEN_INCLUDE_DIR
  PATH_SUFFIXES eigen3)
add_executable(domain_probe domain_probe.cpp)
target_compile_features(domain_probe PRIVATE cxx_std_23)
target_compile_options(domain_probe PRIVATE -O2 -march=native -fno-exceptions -fno-rtti)
target_compile_definitions(domain_probe PRIVATE MAGMAAN_WITH_PORT EIGEN_MAX_ALIGN_BYTES=64 EIGEN_NO_EXCEPTIONS)
target_include_directories(domain_probe PRIVATE "${MAGMAAN_ROOT}/cpp/include" "${EIGEN_INCLUDE_DIR}")
target_link_libraries(domain_probe PRIVATE
  "${MAGMAAN_BUILD}/libmagmaan.a" "${MAGMAAN_BUILD}/libport.a"
  "${MAGMAAN_BUILD}/libquadpack.a" "${MAGMAAN_BUILD}/_deps/nlopt-build/libnlopt.a" m)
