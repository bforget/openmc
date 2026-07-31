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

// Nested SUBSET counters. Each costs an rdtsc pair whose opening tick lands in
// the ENCLOSING span, so with six subsets inside one calculate_xs the parent
// carries a few hundred cycles of pure instrumentation that looks like an
// "unattributed remainder". Build with 0 to measure the top-level flight
// categories undistorted; build with 1 to attribute within them. The two
// builds answer different questions and their totals are NOT comparable.
#ifndef WMP_PROF_SUBSETS
#define WMP_PROF_SUBSETS 0
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
  WPROF_FAST_HEAD,      //!< entry -> temperature branch (micro cache init)
  WPROF_FAST_TAIL,      //!< after the pole block -> return (sab/URR gate/cache)
  WPROF_URR_SEARCH,     //!< URR: energy bsearch + future_prn + 2 cdf bsearches
  WPROF_URR_INTERP,     //!< URR: band interpolation (log_log = 6 log + 3 exp)
  WPROF_URR_NU,         //!< URR: the SECOND nu() call, on the slow path

  // SYMMETRIC per-region attribution. The subset counters above fire only on
  // the fast path, so FLIGHT_FAST carried six nested rdtsc pairs while
  // FLIGHT_RESOLVED carried none and FLIGHT_ACE one -- which made the fast
  // path look far worse than it is and the resolved path look like parity
  // with ACE. These regions are timed with the SAME NUMBER OF TICKS AT THE
  // SAME POINTS on every path; only the attribution differs, so cross-path
  // differences are real and the instrumentation cancels in the comparison.
  WPROF_FL_SEARCH,   //!< fast flight: sub-window lookup
  WPROF_FL_CURVE,    //!< fast flight: Horner curvefit (no Doppler)
  WPROF_FL_SWEEP,    //!< fast flight: retained-pole sweep
  WPROF_RG_ACE_HEAD1,
  WPROF_RG_ACE_HEAD2,
  WPROF_RG_ACE_TEMP,
  WPROF_RG_ACE_GRID,
  WPROF_RG_ACE_XS,
  WPROF_RG_ACE_PHOT,
  WPROF_RG_ACE_EVAL,
  WPROF_RG_ACE_TAIL1,
  WPROF_RG_ACE_TAIL2,
  WPROF_RG_FAST_HEAD1,
  WPROF_RG_FAST_HEAD2,
  WPROF_RG_FAST_TEMP,
  WPROF_RG_FAST_GRID,
  WPROF_RG_FAST_XS,
  WPROF_RG_FAST_PHOT,
  WPROF_RG_FAST_EVAL,
  WPROF_RG_FAST_TAIL1,
  WPROF_RG_FAST_TAIL2,
  WPROF_RG_RES_HEAD1,
  WPROF_RG_RES_HEAD2,
  WPROF_RG_RES_TEMP,
  WPROF_RG_RES_GRID,
  WPROF_RG_RES_XS,
  WPROF_RG_RES_PHOT,
  WPROF_RG_RES_EVAL,
  WPROF_RG_RES_TAIL1,
  WPROF_RG_RES_TAIL2,
  WPROF_N
};

struct WmpProfCounters {
  uint64_t cyc[WPROF_N] {};
  uint64_t cnt[WPROF_N] {};
  bool registered {false};
};

//! Register a thread's counter block with the aggregator (first use only).
void wmp_prof_register(WmpProfCounters* c);

//! Thread-local counters. CONSTANT-initialized (all zero) so there is no
//! dynamic-init guard, and INLINE so there is no cross-TU call. The previous
//! version was an out-of-line function holding a `static thread_local` with
//! dynamic initialization, so every wmp_prof_add() paid a call plus a guard
//! check -- and because that ran BEFORE the closing rdtsc, its cost was
//! charged to the enclosing span instead of the section being measured. With
//! six nested subsets per fast lookup that manufactured a ~500 cyc
//! "unattributed" remainder in calculate_xs that was pure instrumentation.
extern thread_local WmpProfCounters wmp_prof_tls;
inline WmpProfCounters& wmp_prof_local()
{
  if (!wmp_prof_tls.registered) {
    wmp_prof_tls.registered = true;
    wmp_prof_register(&wmp_prof_tls);
  }
  return wmp_prof_tls;
}

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
  uint64_t dt = __rdtsc() - t0;
  auto& c = wmp_prof_local();
  c.cyc[cat] += dt;
  ++c.cnt[cat];
#else
  (void)cat;
  (void)t0;
#endif
}

//! Add a cycle count measured elsewhere (for a region whose category is not
//! known until after it has been timed -- the fast path only learns it IS the
//! fast path partway through calculate_xs).
inline void wmp_prof_add_cycles(int cat, uint64_t cyc)
{
#if WMP_PROF_ENABLED
  auto& c = wmp_prof_local();
  c.cyc[cat] += cyc;
  ++c.cnt[cat];
#else
  (void)cat;
  (void)cyc;
#endif
}

//! Serializing fence used ONLY to pin a region's out-of-order drain inside its
//! own span while profiling. MUST compile to nothing in a production build.
#if WMP_PROF_ENABLED && WMP_PROF_SUBSETS
#define WMP_PROF_FENCE() _mm_lfence()
#else
#define WMP_PROF_FENCE() ((void)0)
#endif

//! Subset-counter variants: compiled out entirely when WMP_PROF_SUBSETS=0.
inline uint64_t wmp_prof_tick_sub()
{
#if WMP_PROF_SUBSETS
  return wmp_prof_tick();
#else
  return 0;
#endif
}

inline void wmp_prof_add_sub(int cat, uint64_t t0)
{
#if WMP_PROF_SUBSETS
  wmp_prof_add(cat, t0);
#else
  (void)cat;
  (void)t0;
#endif
}

inline void wmp_prof_add_cycles_sub(int cat, uint64_t cyc)
{
#if WMP_PROF_SUBSETS
  wmp_prof_add_cycles(cat, cyc);
#else
  (void)cat;
  (void)cyc;
#endif
}

//! Aggregate all threads' counters and print the attribution table.
void wmp_prof_report();

} // namespace openmc

#endif // OPENMC_WMP_PROFILE_H
