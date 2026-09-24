# Reject toolchains that lack the one C++23 feature magmaan relies on:
# std::expected (the error model). That single feature sets the floor, and it
# belongs to the compiler *and* standard-library pair, not the compiler alone:
# clang 17 and 18 have std::expected with libc++ but not with libstdc++, whose
# <expected> needs the C++20 concepts level clang reports only from 19 on. The
# version checks below are therefore a coarse first gate with friendly
# messages; the compile probe at the end decides.
#
# AppleClang has historically lagged libc++ shipping std::expected, so we
# refuse it explicitly and ask macOS users to install Homebrew LLVM.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR
      "magmaan requires GCC >= 13 (have ${CMAKE_CXX_COMPILER_VERSION}). "
      "GCC 13 is the first version with usable std::expected in libstdc++.")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
    message(FATAL_ERROR
      "magmaan requires Clang >= 17 (have ${CMAKE_CXX_COMPILER_VERSION}); "
      "Clang >= 19 when building against libstdc++.")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(FATAL_ERROR
    "AppleClang is not supported: it lacks std::expected. "
    "Install Homebrew LLVM and pass -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.37)
    message(FATAL_ERROR
      "magmaan requires MSVC >= 19.37 (have ${CMAKE_CXX_COMPILER_VERSION}).")
  endif()
else()
  message(WARNING
    "Untested compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}. "
    "Build may succeed but is not on the supported matrix.")
endif()

include(CheckCXXSourceCompiles)
block()
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  check_cxx_source_compiles([=[
#include <expected>
#if !defined(__cpp_lib_expected)
#error "std::expected is not available"
#endif
int main() {
  std::expected<int, int> e{1};
  return e.has_value() ? 0 : e.error();
}
]=] MAGMAAN_HAVE_STD_EXPECTED)
endblock()
if(NOT MAGMAAN_HAVE_STD_EXPECTED)
  message(FATAL_ERROR
    "magmaan needs std::expected (C++23), which this compiler and standard "
    "library pair (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}) does "
    "not provide. Known-good pairs: GCC >= 13; Clang >= 19 with libstdc++; "
    "Clang >= 17 with libc++ (-stdlib=libc++). Clang 17 and 18 cannot use "
    "libstdc++'s <expected>. The failing probe is logged in "
    "CMakeFiles/CMakeConfigureLog.yaml.")
endif()
