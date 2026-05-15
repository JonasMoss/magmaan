# R-first benchmark scaffold.
#
# The public benchmark harness lives under benchmarks/r and is intentionally
# not built by CMake yet. This placeholder keeps MAGMAAN_BUILD_BENCH=ON valid
# while C++ microbenchmarks are still future work.
add_custom_target(magmaan_benchmark_scaffold
  COMMAND ${CMAKE_COMMAND} -E echo "Use Rscript benchmarks/r/check_sources.R for the benchmark scaffold."
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
