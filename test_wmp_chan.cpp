// Dump per-channel fast xs: E followed by nm channel values.
#include "openmc/wmp.h"
#include "openmc/hdf5_interface.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char* argv[]) {
  hid_t file = openmc::file_open(argv[1], 'r');
  openmc::check_wmp_version(file);
  hid_t group = openmc::open_group(file, argv[2]);
  openmc::WindowedMultipole wmp(group);
  int nm = wmp.fast_n_mts_;
  // header: MT list
  printf("MTS");
  for (int c = 0; c < nm; ++c) printf(" %d", wmp.fast_mts_[c]);
  printf("\n");
  std::vector<double> buf(nm);
  double E;
  while (scanf("%lf", &E) == 1) {
    wmp.evaluate_fast_channels(E, buf.data());
    printf("%.10e", E);
    for (int c = 0; c < nm; ++c) printf(" %.10e", buf[c]);
    printf("\n");
  }
  return 0;
}
