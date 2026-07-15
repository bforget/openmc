#include "openmc/wmp_profile.h"

#include <cstdio>
#include <mutex>
#include <vector>

namespace openmc {

namespace {
std::mutex reg_mutex;
std::vector<WmpProfCounters*> registry;
} // namespace

WmpProfCounters& wmp_prof_local()
{
  static thread_local WmpProfCounters* local = [] {
    auto* c = new WmpProfCounters();
    std::lock_guard<std::mutex> lock(reg_mutex);
    registry.push_back(c);
    return c;
  }();
  return *local;
}

void wmp_prof_report()
{
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
}

} // namespace openmc
