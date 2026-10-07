#pragma once

#include "magmaan/estimate/diagnostics.hpp"

#ifdef MAGMAAN_ENABLE_TEST_PROBES
namespace magmaan::estimate::identification_test {

// Private production-compiler seam for the identification sweep: when set,
// every completed structural identification check is reported with its wall
// time in seconds. Test builds only; the observer must not throw or allocate
// from a signal context. Pass nullptr to remove it.
using Observer = void (*)(const IdentificationReport& report, double seconds);
void set_observer(Observer observer) noexcept;

}  // namespace magmaan::estimate::identification_test
#endif
