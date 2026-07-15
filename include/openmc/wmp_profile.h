#ifndef OPENMC_WMP_PROFILE_H
#define OPENMC_WMP_PROFILE_H

//! \file wmp_profile.h
//! Lightweight rdtsc cycle counters for attributing WMP fast-range flight
//! and collision costs (diagnostic branch wmp-fast-profile only).
//! Thread-local accumulators, aggregated + printed at simulation end.

#include <cstdint>
#include <x86intrin.h>

// Compile-time switch: 1 = counters on (attribution build; inflates the
// instrumented paths by ~60 cyc per section), 0 = all profiling compiled
// out (clean-rate build for true A/B timing).
#ifndef WMP_PROF_ENABLED
#define WMP_PROF_ENABLED 0
#endif

namespace openmc {

enum WmpProfCat {
  WPROF_FLIGHT_ACE = 0, //!< calculate_xs, tabulated (non-multipole) path
  WPROF_FLIGHT_FAST,    //!< calculate_xs, fast-multipole path (whole call)
  WPROF_KERNEL_FAST,    //!< evaluate_fast (subset of FLIGHT_FAST)
  WPROF_NU,             //!< nu() on the fast path (subset of FLIGHT_FAST)
  WPROF_URR,            //!< calculate_urr_xs (subset of the owning flight)
  WPROF_FLIGHT_RES,     //!< calculate_xs, resolved-multipole path
  WPROF_COLL_GRID,      //!< ensure_grid_index (deferred ACE grid search)
  WPROF_COLL_CHAN,      //!< evaluate_fast_channels (collision kernel)
  WPROF_PREAMBLE,       //!< temp select + lazy-grid branch (fast path subset)
  WPROF_PHOTON,         //!< photon_prod interpolation (fast path subset)
  WPROF_N
};

struct WmpProfCounters {
  uint64_t cyc[WPROF_N] {};
  uint64_t cnt[WPROF_N] {};
};

//! Thread-local counters, registered globally on first use per thread.
WmpProfCounters& wmp_prof_local();

inline uint64_t wmp_prof_tick()
{
#if WMP_PROF_ENABLED
  return __rdtsc();
#else
  return 0;
#endif
}

inline void wmp_prof_add(int cat, uint64_t t0)
{
#if WMP_PROF_ENABLED
  auto& c = wmp_prof_local();
  c.cyc[cat] += __rdtsc() - t0;
  ++c.cnt[cat];
#else
  (void)cat;
  (void)t0;
#endif
}

//! Aggregate all threads' counters and print the attribution table.
void wmp_prof_report();

} // namespace openmc

#endif // OPENMC_WMP_PROFILE_H
