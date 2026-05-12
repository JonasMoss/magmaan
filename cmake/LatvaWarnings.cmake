# INTERFACE target carrying latva's warning wall. Linked PRIVATE by latva
# itself and by latva_tests so that downstream consumers do not inherit
# our -Werror.

add_library(latva_warnings INTERFACE)
add_library(latva::warnings ALIAS latva_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(latva_warnings INTERFACE
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
    >)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(latva_warnings INTERFACE
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
