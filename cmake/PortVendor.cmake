# PortVendor.cmake — builds the vendored PORT (Bell Labs) routines as a
# static library `port`, linkable into libmagmaan.a.
#
# Sources are committed under `third_party/port/`. See
# `third_party/port/README.md` for the vendor manifest, license chain
# (BSD-3-Clause, AMPL/ASL + Fermi-LAT), and TOMS 573 / TOMS 611 citations.
# Re-vendoring is mechanical — `magmaan_find_or_fetch` is intentionally not
# used: there is no `master` to track since upstream is decades-stable.
#
# Only invoked from the top-level CMakeLists when MAGMAAN_WITH_PORT is ON.

set(_port_root "${CMAKE_SOURCE_DIR}/third_party/port")

# All AMPL/ASL `cport/` translation units plus Fermi-LAT's `drmnfb_routines.c`
# and our two libf2c-runtime replacement TUs. Glob the cport/ tree because the
# file list is large (~75 entries) and they are all algorithmic content; a
# manual enumeration would just be noise. Glob result is captured at configure
# time, which is fine for vendored sources that change only when a maintainer
# re-vendors.
file(GLOB _port_cport_sources CONFIGURE_DEPENDS "${_port_root}/cport/*.c")

add_library(port STATIC
  ${_port_cport_sources}
  "${_port_root}/drmnfb_routines.c"
  "${_port_root}/port_io.c"
  "${_port_root}/f2c_intrinsics.c"
)

# `f2c.h` lives next to the cport TUs; both top-level TUs (port_io.c,
# f2c_intrinsics.c, drmnfb_routines.c-after-include-patch) include it
# unqualified. The header is exported via INTERFACE so the magmaan adapter
# (third_party-internal symbols) can pull in the cilist / doublereal /
# integer typedefs if it ever needs to.
target_include_directories(port
  PUBLIC  "${_port_root}/cport"
  PRIVATE "${_port_root}"
)

# Vendored f2c output is famously not warning-clean (implicit declarations,
# variable shadowing, unused labels). Silence the whole TU set — warnings
# inside vendored code are not our bugs to fix, and turning them into errors
# would only force endless local patches.
target_compile_options(port PRIVATE -w)

# Pure C, no C++ exceptions, no RTTI — match the magmaan baseline rather
# than the -fexceptions adapters. PORT is reverse-communication, no
# callback throws to worry about.
set_target_properties(port PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Tag the build so anyone who pulls in target `port` later can introspect.
set_target_properties(port PROPERTIES
  EXPORT_NAME magmaan_port
  FOLDER "third_party"
)

unset(_port_cport_sources)
unset(_port_root)
