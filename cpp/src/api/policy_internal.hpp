#pragma once
#include "magmaan/api/policy.hpp"
namespace magmaan::api {
post_expected<data::MixedOrdinalStats> mixed_policy_stats(
    const data::MixedOrdinalStats&, MixedDwlsPolicyFit::Impl*);
template<class Stats, class Cache>
post_expected<Stats> dwls_policy_stats(const Stats&, Cache*);
extern template post_expected<data::OrdinalStats> dwls_policy_stats(
    const data::OrdinalStats&, DwlsPolicyFit::Impl*);
extern template post_expected<data::MixedOrdinalStats> dwls_policy_stats(
    const data::MixedOrdinalStats&, MixedDwlsPolicyFit::Impl*);
}
