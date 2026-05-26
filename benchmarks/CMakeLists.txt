# R-first benchmark scaffold.
#
# The public benchmark harness lives under benchmarks/r and is intentionally
# not built by CMake. This placeholder keeps MAGMAAN_BUILD_BENCH=ON valid even
# when no C++ target is requested. The C++ memory profiler below is opt-in.
add_custom_target(magmaan_benchmark_scaffold
  COMMAND ${CMAKE_COMMAND} -E echo "Use Rscript benchmarks/r/check_sources.R for the benchmark scaffold."
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(magmaan_ordinal_workspace_bench
  ordinal_workspace_bench.cpp)

target_link_libraries(magmaan_ordinal_workspace_bench PRIVATE magmaan::magmaan)

target_compile_features(magmaan_ordinal_workspace_bench PRIVATE cxx_std_23)

add_executable(magmaan_ordinal_inference_workspace_bench
  ordinal_inference_workspace_bench.cpp)

target_link_libraries(magmaan_ordinal_inference_workspace_bench PRIVATE magmaan::magmaan)

target_compile_features(magmaan_ordinal_inference_workspace_bench PRIVATE cxx_std_23)

add_executable(magmaan_ordinal_snlls_speed_bench
  ordinal_snlls_speed_bench.cpp)

target_link_libraries(magmaan_ordinal_snlls_speed_bench PRIVATE magmaan::magmaan)

target_compile_features(magmaan_ordinal_snlls_speed_bench PRIVATE cxx_std_23)

add_executable(magmaan_ordinal_threshold_constraint_bench
  ordinal_threshold_constraint_bench.cpp)

target_link_libraries(magmaan_ordinal_threshold_constraint_bench PRIVATE magmaan::magmaan)

target_compile_features(magmaan_ordinal_threshold_constraint_bench PRIVATE cxx_std_23)

add_executable(magmaan_ordinal_construction_bench
  ordinal_construction_bench.cpp)

target_link_libraries(magmaan_ordinal_construction_bench PRIVATE magmaan::magmaan)

target_compile_features(magmaan_ordinal_construction_bench PRIVATE cxx_std_23)

# ---------------------------------------------------------------------------
# magmaan_mem_profile — peak-heap profiler for the reduced-Gamma computations.
#
# malloc_peak.cpp installs a heap high-water counter via linker `--wrap`, so
# Eigen's aligned_malloc allocations (which bypass operator new) are captured.
# Links only magmaan::magmaan: that brings -fno-exceptions/-fno-rtti and Eigen,
# but NOT the PRIVATE sanitizers (an AddressSanitizer build would replace
# malloc and defeat the counter).
# ---------------------------------------------------------------------------
add_executable(magmaan_mem_profile
  mem_profile.cpp
  malloc_peak.cpp)

target_link_libraries(magmaan_mem_profile PRIVATE magmaan::magmaan)

target_compile_features(magmaan_mem_profile PRIVATE cxx_std_23)

target_link_options(magmaan_mem_profile PRIVATE
  -Wl,--wrap=malloc
  -Wl,--wrap=free
  -Wl,--wrap=calloc
  -Wl,--wrap=realloc
  -Wl,--wrap=posix_memalign
  -Wl,--wrap=aligned_alloc)
