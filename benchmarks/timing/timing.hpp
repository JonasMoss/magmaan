#pragma once

// Shared timing core for magmaan benchmarks.
//
// This is the extraction of the per-file timer that grew independently in
// `ordinal_gamma_influence_bench.cpp` and five other places. Three things it
// adds over the copies it replaces:
//
//   * Batch auto-calibration. A stage like `ev.sigma()` at p = 6 runs in
//     hundreds of nanoseconds, well under `steady_clock`'s useful resolution
//     once loop and call overhead are counted. Each stage is calibrated to a
//     batch size that puts one timed sample near `target_ns`, and the reported
//     figure is ns *per call*.
//
//   * Arm rotation. Reps are the outer loop and arms the inner one, with the
//     arm order rotated by rep index. Running all reps of arm A and then all
//     reps of arm B attributes turbo decay, thermal throttle, and page-cache
//     warmth to whichever arm ran last, which is exactly the bias that matters
//     when the question is "which of these is faster".
//
//   * A checksum return from every thunk. At -O3 -march=native a timed call
//     whose result is unused is legitimately dead code. Returning a double
//     derived from the result and sinking it through an asm barrier keeps the
//     work alive, and comparing checksums across arms is a free tripwire for
//     "these two arms did not compute the same thing".
//
// Medians, not means: a stolen scheduler slice is a one-sided contaminant, and
// the mean has no defence against it.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace magmaan::bench {

using clock_type = std::chrono::steady_clock;

// Keep a computed value alive across the timing barrier.
inline void sink(double value) noexcept {
  asm volatile("" : : "x"(value) : "memory");
}

inline void clobber() noexcept { asm volatile("" : : : "memory"); }

// `std::expected` unwrap for a -fno-exceptions binary: `.value()` on an error
// would call the throwing accessor, so check first and use operator*.
template <class Expected>
decltype(auto) must(Expected&& e, const char* what) {
  if (!e.has_value()) {
    std::fprintf(stderr, "fatal: %s returned an error\n", what);
    std::exit(1);
  }
  return *std::forward<Expected>(e);
}

// A named unit of timed work. Returns a checksum derived from its result.
struct Arm {
  std::string             name;
  std::function<double()> run;
};

struct StageStats {
  std::string name;
  double      median_ns = 0.0;
  double      min_ns    = 0.0;
  double      p90_ns    = 0.0;
  double      iqr_ns    = 0.0;
  int         reps      = 0;
  int         batch     = 1;
  double      checksum  = 0.0;
};

namespace detail {

inline double quantile_sorted(const std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  if (v.size() == 1) return v.front();
  const double pos = q * static_cast<double>(v.size() - 1);
  const auto   lo  = static_cast<std::size_t>(std::floor(pos));
  const auto   hi  = static_cast<std::size_t>(std::ceil(pos));
  const double w   = pos - static_cast<double>(lo);
  return v[lo] * (1.0 - w) + v[hi] * w;
}

inline StageStats summarize(std::string name, std::vector<double> per_call_ns,
                            int batch, double checksum) {
  std::sort(per_call_ns.begin(), per_call_ns.end());
  StageStats s;
  s.name      = std::move(name);
  s.reps      = static_cast<int>(per_call_ns.size());
  s.batch     = batch;
  s.checksum  = checksum;
  if (per_call_ns.empty()) return s;
  s.min_ns    = per_call_ns.front();
  s.median_ns = quantile_sorted(per_call_ns, 0.5);
  s.p90_ns    = quantile_sorted(per_call_ns, 0.9);
  s.iqr_ns    = quantile_sorted(per_call_ns, 0.75) - quantile_sorted(per_call_ns, 0.25);
  return s;
}

// One untimed call, used to size the batch so a timed sample lands near
// `target_ns`. Stages slower than the target get batch 1.
inline int calibrate_batch(const Arm& arm, double target_ns, int max_batch) {
  const auto t0 = clock_type::now();
  const double c = arm.run();
  const auto t1 = clock_type::now();
  sink(c);
  const double one_ns =
      std::chrono::duration<double, std::nano>(t1 - t0).count();
  if (!(one_ns > 0.0)) return max_batch;
  if (one_ns >= target_ns) return 1;
  const double want = std::ceil(target_ns / one_ns);
  if (want >= static_cast<double>(max_batch)) return max_batch;
  return std::max(1, static_cast<int>(want));
}

}  // namespace detail

struct TimingOptions {
  int    reps       = 15;
  int    warmups    = 3;
  double target_ns  = 200000.0;  // 200 us per timed sample
  int    max_batch  = 1 << 20;
  // Time each arm alone instead of in a shared rotation. See the note on
  // `time_arms` — required for comparing *different* stages to each other.
  bool   isolate    = false;
};

// Time one arm on its own. Equivalent to `time_arms` with a single arm.
inline StageStats time_arm(const Arm& arm, const TimingOptions& opt) {
  const int batch = detail::calibrate_batch(arm, opt.target_ns, opt.max_batch);
  double checksum = 0.0;
  for (int w = 0; w < opt.warmups; ++w) checksum += arm.run();
  sink(checksum);

  std::vector<double> per_call;
  per_call.reserve(static_cast<std::size_t>(opt.reps));
  for (int r = 0; r < opt.reps; ++r) {
    double acc = 0.0;
    clobber();
    const auto t0 = clock_type::now();
    for (int b = 0; b < batch; ++b) acc += arm.run();
    const auto t1 = clock_type::now();
    clobber();
    sink(acc);
    checksum = acc;
    per_call.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                       static_cast<double>(batch));
  }
  return detail::summarize(arm.name, std::move(per_call), batch, checksum);
}

// Time several arms as a paired comparison: reps outer, arms inner, arm order
// rotated by rep so no arm has a systematically privileged position in the run.
//
// WHAT ROTATION DOES AND DOES NOT BUY. Rotation cancels *time-ordered* drift —
// turbo decay, thermal throttle, a background job ramping up — so two arms that
// do comparable work are compared fairly. It does NOT make arms independent of
// each other. Arms allocating multi-megabyte buffers (a q x n_free Jacobian at
// p = 96 is ~7 MB) interact through the allocator's mmap threshold and the page
// cache, so a stage's absolute median depends on what else is in the arm set.
// Measured: `dsigma` read 3.6x `evaluate_sigma_and_jac` inside a 30-arm
// rotation and 0.98x it alone, when the two call the same code to produce the
// same matrix.
//
// So: use the rotation to compare ARMS OF ONE STAGE (marker vs std.lv vs
// effect.coding on `optimize_lbfgs`; one algorithm against another), where both
// arms allocate alike and the comparison is what rotation protects. Use
// `isolate` when comparing DIFFERENT STAGES to each other, or when an absolute
// number has to mean something on its own.
inline std::vector<StageStats> time_arms(const std::vector<Arm>& arms,
                                         const TimingOptions&    opt) {
  const std::size_t k = arms.size();
  std::vector<StageStats> out;
  if (k == 0) return out;

  std::vector<int>    batch(k, 1);
  std::vector<double> checksum(k, 0.0);
  for (std::size_t i = 0; i < k; ++i)
    batch[i] = detail::calibrate_batch(arms[i], opt.target_ns, opt.max_batch);

  for (int w = 0; w < opt.warmups; ++w) {
    double acc = 0.0;
    for (std::size_t i = 0; i < k; ++i) acc += arms[i].run();
    sink(acc);
  }

  std::vector<std::vector<double>> per_call(k);
  for (std::size_t i = 0; i < k; ++i)
    per_call[i].reserve(static_cast<std::size_t>(opt.reps));

  for (int r = 0; r < opt.reps; ++r) {
    for (std::size_t j = 0; j < k; ++j) {
      const std::size_t i = (static_cast<std::size_t>(r) + j) % k;
      double acc = 0.0;
      clobber();
      const auto t0 = clock_type::now();
      for (int b = 0; b < batch[i]; ++b) acc += arms[i].run();
      const auto t1 = clock_type::now();
      clobber();
      sink(acc);
      checksum[i] = acc;
      per_call[i].push_back(
          std::chrono::duration<double, std::nano>(t1 - t0).count() /
          static_cast<double>(batch[i]));
    }
  }

  out.reserve(k);
  for (std::size_t i = 0; i < k; ++i)
    out.push_back(detail::summarize(arms[i].name, std::move(per_call[i]),
                                    batch[i], checksum[i]));
  return out;
}

// Dispatch on `opt.isolate`: shared rotation by default, each arm alone when
// asked. Prefer this over calling `time_arms` directly so a harness picks up
// the isolate switch for free.
inline std::vector<StageStats> time_stages(const std::vector<Arm>& arms,
                                           const TimingOptions&    opt) {
  if (!opt.isolate) return time_arms(arms, opt);
  std::vector<StageStats> out;
  out.reserve(arms.size());
  for (const auto& a : arms) out.push_back(time_arm(a, opt));
  return out;
}

}  // namespace magmaan::bench
