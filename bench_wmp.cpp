// Per-window evaluate() timing: cost = f(n_poles). Usage: bench_wmp file nuc sqrtkT
#include "openmc/wmp.h"
#include "openmc/hdf5_interface.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <tuple>

int main(int argc, char* argv[])
{
  hid_t file = openmc::file_open(argv[1], 'r');
  openmc::check_wmp_version(file);
  hid_t group = openmc::open_group(file, argv[2]);
  openmc::WindowedMultipole wmp(group);
  double sqrtkT = atof(argv[3]);
  const int REP = 200000;
  volatile double sink = 0.0;
  int n_win = wmp.win_start_.size();
  int stride = n_win > 60 ? n_win / 60 : 1;   // ~60 sample windows
  for (int w = 0; w < n_win; w += stride) {
    double umid = 0.5 * (wmp.sqrtE_bounds_[w] + wmp.sqrtE_bounds_[w + 1]);
    double E = umid * umid;
    int npoles = wmp.win_end_[w] - wmp.win_start_[w];
    auto t0 = std::chrono::steady_clock::now();
    double s, a, f;
    for (int r = 0; r < REP; ++r) {
      // jitter E slightly to defeat any value caching, stay in window
      double Ej = E * (1.0 + 1e-7 * (r % 97));
      std::tie(s, a, f) = wmp.evaluate(Ej, sqrtkT);
      sink += s;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / REP;
    printf("%d %d %.2f\n", w, npoles, ns);
  }
  fprintf(stderr, "sink %g\n", sink);
  return 0;
}
