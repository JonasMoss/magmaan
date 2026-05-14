# Sanitizer wiring. Driven by the cache variable MAGMAAN_SANITIZER which is
# set by CMake presets (asan/ubsan) or by the user on the command line.

set(MAGMAAN_SANITIZER "" CACHE STRING
  "Sanitizer to enable: '', asan, ubsan, asan+ubsan, tsan")
set_property(CACHE MAGMAAN_SANITIZER PROPERTY STRINGS
  "" "asan" "ubsan" "asan+ubsan" "tsan")

add_library(magmaan_sanitizers INTERFACE)
add_library(magmaan::sanitizers ALIAS magmaan_sanitizers)

if(NOT MAGMAAN_SANITIZER STREQUAL "")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "Sanitizers are only wired for GCC/Clang; ignoring "
                    "MAGMAAN_SANITIZER=${MAGMAAN_SANITIZER}")
  else()
    set(_san_flags "")
    if(MAGMAAN_SANITIZER MATCHES "asan")
      list(APPEND _san_flags -fsanitize=address)
    endif()
    if(MAGMAAN_SANITIZER MATCHES "ubsan")
      list(APPEND _san_flags -fsanitize=undefined)
    endif()
    if(MAGMAAN_SANITIZER STREQUAL "tsan")
      list(APPEND _san_flags -fsanitize=thread)
    endif()
    list(APPEND _san_flags
      -fno-omit-frame-pointer
      -fno-sanitize-recover=all)
    target_compile_options(magmaan_sanitizers INTERFACE ${_san_flags})
    target_link_options   (magmaan_sanitizers INTERFACE ${_san_flags})
    message(STATUS "magmaan: sanitizers active = ${MAGMAAN_SANITIZER}")
  endif()
endif()
