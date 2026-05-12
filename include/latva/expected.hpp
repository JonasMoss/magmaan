#pragma once

#include <expected>

#include "latva/error.hpp"

namespace latva {

template <class T>
using parse_expected = std::expected<T, ParseError>;

template <class T>
using partable_expected = std::expected<T, PartableError>;

template <class T>
using model_expected = std::expected<T, ModelError>;

template <class T>
using fit_expected = std::expected<T, FitError>;

template <class T>
using post_expected = std::expected<T, PostError>;

}
