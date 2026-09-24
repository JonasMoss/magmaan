# INTERFACE target carrying magmaan's warning wall. Linked PRIVATE by magmaan
# itself and by magmaan_tests so that downstream consumers do not inherit
# our -Werror.

add_library(magmaan_warnings INTERFACE)
add_library(magmaan::warnings ALIAS magmaan_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(magmaan_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wshadow
      -Wconversion
      -Wsign-conversion
      -Wnon-virtual-dtor
      -Wold-style-cast
      -Wcast-align
      -Wunused
      -Woverloaded-virtual
      -Wdouble-promotion
      -Wformat=2
      -Wno-unknown-pragmas
    >
    # GCC 12 and 13 report false -Warray-bounds and -Wstringop-overflow
    # positives inside libstdc++'s std::string (memcpy in
    # string_fortified.h), which -Werror turns fatal.
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,14>>:
      -Wno-array-bounds
      -Wno-stringop-overflow
    >)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(magmaan_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:
      /W4
      /WX
      /permissive-
      /w14242 /w14254 /w14263 /w14265 /w14287
      /we4289 /w14296 /w14311 /w14545 /w14546 /w14547
      /w14549 /w14555 /w14619 /w14640 /w14826 /w14905
      /w14906 /w14928
    >)
endif()
