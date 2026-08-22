#include "openmc/wmp_profile.h"

#include <cstdio>
#include <mutex>
#include <vector>

namespace openmc {

namespace {
std::mutex reg_mutex;
std::vector<WmpProfCounters*> registry;
} // namespace

thread_local WmpProfCounters wmp_prof_tls;

void wmp_prof_register(WmpProfCounters* c)
{
  std::lock_guard<std::mutex> lock(reg_mutex);
  registry.push_back(c);
}

void wmp_prof_report()
{
#if !WMP_PROF_ENABLED
  // NOTHING TO REPORT IN A PRODUCTION BUILD, so print nothing. The counters
  // are all zero (wmp_prof_add compiles out), so the table body printed no
  // rows -- but the header and the overhead line were emitted unconditionally
  // on every run, and the overhead figure was garbage: wmp_prof_tick() returns
  // 0 when profiling is off, so the calibration below computed
  // (__rdtsc() - 0)/1000, i.e. the raw TSC over 1000, and reported ~1.75e10
  // cyc/call. It also burned 1000 rdtsc calls at finalize for nothing.
  // Build with -DWMP_PROF_ENABLED=1 to get the real report back.
  return;
#else
  WmpProfCounters tot;
  {
    std::lock_guard<std::mutex> lock(reg_mutex);
    for (auto* c : registry) {
      for (int i = 0; i < WPROF_N; ++i) {
        tot.cyc[i] += c->cyc[i];
        tot.cnt[i] += c->cnt[i];
      }
    }
  }
  // Calibrate the rdtsc pair overhead so per-call numbers can be read
  // against it (not subtracted -- just printed for context).
  uint64_t t0 = wmp_prof_tick();
  volatile uint64_t sink = 0;
  for (int i = 0; i < 1000; ++i)
    sink += __rdtsc();
  double pair_cyc = (double)(__rdtsc() - t0) / 1000.0;

  static const char* names[WPROF_N] = {
    "flight ACE-tabulated  (calculate_xs)",
    "flight FAST-multipole (calculate_xs)",
    "  evaluate_fast kernel     [subset]",
    "  nu() on fast path        [subset]",
    "  calculate_urr_xs         [subset]",
    "flight RESOLVED-mp    (calculate_xs)",
    "collision ensure_grid_index",
    "collision evaluate_fast_channels",
    "  flight preamble (temp+lazy)  [subset]",
    "  flight photon_prod interp    [subset]",
    "  flight head  (micro init)    [subset]",
    "  flight tail  (sab/urr/cache) [subset]",
    "    urr search (bsearch+prn)   [subset]",
    "    urr band interpolation     [subset]",
    "    urr nu() (2nd, slow path)  [subset]",
  };
  std::printf("\n ===== WMP fast-range cycle attribution (rdtsc) =====\n");
  std::printf(" %-38s %14s %12s %10s\n", "category", "calls", "Mcycles",
    "cyc/call");
  for (int i = 0; i < WPROF_N; ++i) {
    if (tot.cnt[i] == 0)
      continue;
    std::printf(" %-38s %14llu %12.1f %10.1f\n", names[i],
      (unsigned long long)tot.cnt[i], tot.cyc[i] / 1e6,
      (double)tot.cyc[i] / tot.cnt[i]);
  }
  std::printf(" (rdtsc pair overhead ~%.0f cyc/call is INCLUDED in each "
              "measured section)\n",
    pair_cyc);
  std::printf(" =====================================================\n");
#endif
}

} // namespace openmc
