# QuadpackVendor.cmake — builds the vendored QUADPACK `qagi` adaptive integrator
# as a static library `quadpack`, linkable into libmagmaan.a.
#
# Sources are committed under `third_party/quadpack/` (f2c-translated, public
# domain). See `third_party/quadpack/README.md` for the manifest and license.
#
# Unlike PORT this is NOT behind a feature option: `robust::weighted_chisq::
# imhof_upper` (the weighted-chi-square tail behind every SB/pEBA/pOLS p-value
# and the robust nested tests) depends on it, so it is always built.

set(_quadpack_root "${CMAKE_SOURCE_DIR}/third_party/quadpack")

# Five small TUs (dqagie/dqk15i/dqelg/dqpsrt + d1mach + pow_dd). Glob to mirror
# PortVendor and survive re-vendoring; captured at configure time.
file(GLOB _quadpack_sources CONFIGURE_DEPENDS "${_quadpack_root}/*.c")

add_library(quadpack STATIC ${_quadpack_sources})

# f2c.h lives next to the TUs; the translated .c include it unqualified.
target_include_directories(quadpack PRIVATE "${_quadpack_root}")

# `d1mach_` and `pow_dd` also exist in third_party/port. Rename QUADPACK's copies
# to magmaan-private symbols so the two vendored static archives never collide,
# independent of link order or LTO. QUADPACK's own routine names (dqagie_, ...)
# are already unique.
target_compile_definitions(quadpack PRIVATE
  "d1mach_=magmaan_qpk_d1mach_"
  "pow_dd=magmaan_qpk_pow_dd"
)

# Vendored f2c output is not warning-clean; silence the TU set like PORT.
target_compile_options(quadpack PRIVATE -w)

# Pure C, no exceptions/RTTI to worry about; match the magmaan baseline.
set_target_properties(quadpack PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME magmaan_quadpack
  FOLDER "third_party"
)

unset(_quadpack_sources)
unset(_quadpack_root)
