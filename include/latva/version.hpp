#pragma once

#include <string_view>

namespace latva {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 0;
inline constexpr int version_patch = 1;

std::string_view version() noexcept;

}
