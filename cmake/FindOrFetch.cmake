# magmaan_find_or_fetch(NAME <pkg>
#                     [FIND_PACKAGE_ARGS <args>...]
#                     GIT_REPOSITORY <url>
#                     GIT_TAG <tag>
#                     [SUBDIR <subdir>])
#
# Try find_package first; fall back to FetchContent. Fetched dependencies
# are marked SYSTEM and EXCLUDE_FROM_ALL so their warnings do not break -Werror.

include(FetchContent)

function(magmaan_find_or_fetch)
  set(_options "")
  set(_one NAME GIT_REPOSITORY GIT_TAG SUBDIR)
  set(_multi FIND_PACKAGE_ARGS)
  cmake_parse_arguments(LFF "${_options}" "${_one}" "${_multi}" ${ARGN})

  if(NOT LFF_NAME)
    message(FATAL_ERROR "magmaan_find_or_fetch: NAME is required")
  endif()

  find_package(${LFF_NAME} ${LFF_FIND_PACKAGE_ARGS} QUIET)
  if(${LFF_NAME}_FOUND)
    message(STATUS "magmaan: using system ${LFF_NAME}")
    return()
  endif()

  message(STATUS "magmaan: fetching ${LFF_NAME} from ${LFF_GIT_REPOSITORY}@${LFF_GIT_TAG}")
  string(TOLOWER ${LFF_NAME} _lower)
  FetchContent_Declare(${_lower}
    GIT_REPOSITORY ${LFF_GIT_REPOSITORY}
    GIT_TAG        ${LFF_GIT_TAG}
    GIT_SHALLOW    TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE)
  set(_old_build_testing "${BUILD_TESTING}")
  set(BUILD_TESTING OFF CACHE BOOL "Disable tests for fetched dependencies" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
  FetchContent_MakeAvailable(${_lower})
  set(BUILD_TESTING "${_old_build_testing}" CACHE BOOL "Build tests" FORCE)
endfunction()
