# Sanitizer wiring. Driven by the cache variable LATVA_SANITIZER which is
# set by CMake presets (asan/ubsan) or by the user on the command line.

set(LATVA_SANITIZER "" CACHE STRING
  "Sanitizer to enable: '', asan, ubsan, asan+ubsan, tsan")
set_property(CACHE LATVA_SANITIZER PROPERTY STRINGS
  "" "asan" "ubsan" "asan+ubsan" "tsan")

add_library(latva_sanitizers INTERFACE)
add_library(latva::sanitizers ALIAS latva_sanitizers)

if(NOT LATVA_SANITIZER STREQUAL "")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "Sanitizers are only wired for GCC/Clang; ignoring "
                    "LATVA_SANITIZER=${LATVA_SANITIZER}")
  else()
    set(_san_flags "")
    if(LATVA_SANITIZER MATCHES "asan")
      list(APPEND _san_flags -fsanitize=address)
    endif()
    if(LATVA_SANITIZER MATCHES "ubsan")
      list(APPEND _san_flags -fsanitize=undefined)
    endif()
    if(LATVA_SANITIZER STREQUAL "tsan")
      list(APPEND _san_flags -fsanitize=thread)
    endif()
    list(APPEND _san_flags
      -fno-omit-frame-pointer
      -fno-sanitize-recover=all)
    target_compile_options(latva_sanitizers INTERFACE ${_san_flags})
    target_link_options   (latva_sanitizers INTERFACE ${_san_flags})
    message(STATUS "latva: sanitizers active = ${LATVA_SANITIZER}")
  endif()
endif()
