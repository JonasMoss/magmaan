# Reject toolchains that lack the C++23 features magmaan relies on (chiefly
# std::expected and std::variant in the configurations we use). AppleClang
# has historically lagged libc++ shipping std::expected, so we refuse it
# explicitly and ask macOS users to install Homebrew LLVM.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR
      "magmaan requires GCC >= 13 (have ${CMAKE_CXX_COMPILER_VERSION}). "
      "GCC 13 is the first version with usable std::expected in libstdc++.")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
    message(FATAL_ERROR
      "magmaan requires Clang >= 17 (have ${CMAKE_CXX_COMPILER_VERSION}).")
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
