#pragma once

// Process-wide heap high-water-mark counter for the memory-profiling harness.
//
// The harness is linked with `-Wl,--wrap=malloc` (and friends), so every
// allocation that reaches libc malloc -- including Eigen's aligned_malloc,
// which bypasses `operator new` -- is counted here. Use it by bracketing a
// region: call reset_peak(), run the region, read peak_bytes().

#include <cstddef>

namespace memprof {

// Reset the high-water mark to the current live-bytes level.
void reset_peak();

// Current outstanding heap bytes (summed malloc_usable_size of live blocks).
std::size_t current_bytes();

// High-water mark of live bytes since the last reset_peak() (process start
// if never reset).
std::size_t peak_bytes();

}  // namespace memprof
