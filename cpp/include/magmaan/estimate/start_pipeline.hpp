#pragma once

#include <string>
#include <vector>

#include "magmaan/estimate/start_values.hpp"

namespace magmaan::estimate {

// Layered is the layered moment start (estimate/layered_start.hpp), the default
// of complete-data ML and GLS.
// It already constructs in the target identification, so it is never transported.
enum class StartMethod { Simple, Fabin2, Fabin3, Guttman, Bentler1982, JamesStein, Layered };
enum class StartTransport { Native, AutoStdLv, RequireStdLv };
enum class StartTransportIssue {
  None, ConstructorRequiresMarker, EqualityConstraints, FixedValues, MarkerLayout, SourceModel,
  SourceStarts, InvalidScale, NonfiniteValues, FixedValueMismatch, InvalidVector
};
const char* start_transport_reason(StartTransportIssue);

// A prepared identification transformation. Construct with prepare_std_lv_transport;
// source and target must not be mutated afterwards. No sample statistics or start
// algorithm are embedded here: any producer can supply the source vector.
struct StdLvStartTransport {
  spec::LatentStructure target, source;
  model::MatrixRep target_rep, source_rep;
  std::vector<std::vector<int>> markers;
};
std::expected<StdLvStartTransport, StartTransportIssue> prepare_std_lv_transport(
    const spec::LatentStructure&, const model::MatrixRep&);
std::expected<Eigen::VectorXd, StartTransportIssue> transport_start_values(
    const StdLvStartTransport&, const Eigen::VectorXd& source);

struct StartPolicy {
  StartMethod method = StartMethod::Fabin3;
  StartTransport transport = StartTransport::AutoStdLv;
};
enum class StartBranch { Native, TransportedStdLv, NativeFabin = Native };
struct StartValues {
  // Supplied fit input, before any fitter repair/projection/profiling.
  Eigen::VectorXd theta;
  StartBranch branch = StartBranch::Native;
  StartTransportIssue fallback_reason = StartTransportIssue::None;
  // Requested constructor; internal per-factor substitutions remain possible.
  StartMethod method = StartMethod::Fabin3;
  StartTransport requested_transport = StartTransport::Native;
  bool explicit_vector = false;
  // Constructor diagnostics (fallbacks, repairs); currently filled by Layered.
  std::vector<std::string> notes{};
};

fit_expected<StartValues> explicit_start_values(
    const spec::LatentStructure&, const Eigen::VectorXd&);

fit_expected<Eigen::VectorXd> construct_start_values(
    const spec::LatentStructure&, const model::MatrixRep&,
    const data::SampleStats&, StartMethod, const spec::Starts& = {},
    std::vector<std::string>* notes = nullptr);

// Compose constructor -> optional transport -> target-coordinate hints. Auto
// falls back to the same native constructor; RequireStdLv returns an error.
// Marker-only constructors cannot be composed with std.lv source construction.
// Finite values are required, but PSD feasibility belongs to the fitter.
fit_expected<StartValues> start_values(
    const spec::LatentStructure&, const model::MatrixRep&,
    const data::SampleStats&, const StartPolicy&, const spec::Starts& = {});

// Construct starts in the same normalized model used by ML/PSD fitting, then
// return them in caller units. Hints retain their original-unit interpretation.
fit_expected<StartValues> normalized_ml_start_values(
    const spec::LatentStructure&, const model::MatrixRep&,
    const data::SampleStats&, const StartPolicy&, const spec::Starts& = {});

} // namespace magmaan::estimate
