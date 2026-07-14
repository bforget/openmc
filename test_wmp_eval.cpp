// Standalone xs-dump driver for global-pole WMP verification (both segments).
// Usage: test_wmp_eval <file.h5> <nuclide> <sqrtkT_eV>  (E values on stdin)
// Output: E total elastic absorption fission
#include "openmc/wmp.h"
#include "openmc/hdf5_interface.h"
#include <cstdio>
#include <cstdlib>
#include <tuple>

int main(int argc, char* argv[])
{
  if (argc != 4) { fprintf(stderr, "args: file nuclide sqrtkT\n"); return 1; }
  hid_t file = openmc::file_open(argv[1], 'r');
  openmc::check_wmp_version(file);
  hid_t group = openmc::open_group(file, argv[2]);
  openmc::WindowedMultipole wmp(group);
  double sqrtkT = atof(argv[3]);
  double E, t, s, a, f;
  while (scanf("%lf", &E) == 1) {
    if (wmp.fast_present_ && E > wmp.E_max_ && E <= wmp.fast_E_max_) {
      std::tie(t, s, a, f) = wmp.evaluate_fast(E);
    } else {
      std::tie(s, a, f) = wmp.evaluate(E, sqrtkT);
      t = s + a;
    }
    printf("%.10e %.12e %.12e %.12e %.12e\n", E, t, s, a, f);
  }
  return 0;
}
