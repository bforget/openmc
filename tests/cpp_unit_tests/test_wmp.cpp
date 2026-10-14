#include <cmath>
#include <complex>
#include <cstdio>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "hdf5.h"
#include "openmc/constants.h"
#include "openmc/external/Faddeeva.hh"
#include "openmc/hdf5_interface.h"
#include "openmc/wmp.h"

using namespace openmc;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Complex = std::complex<double>;

namespace {

// A small synthetic version 2 library. Every residue is -i*A with A > 0 so
// each pole contributes a positive Lorentzian and no evaluator clamp fires.
struct Library {
  const char* name = "Xx1";
  double sqrt_awr = 10.0;

  // Resolved segment: 9 uniform windows of width 1 in sqrt(E) over [1, 100] eV
  double E_min = 1.0, E_max = 100.0;
  std::vector<int64_t> mts {2, 18, 102, 107};
  std::vector<Complex> poles {
    {0.0, 0.5}, {2.5, 0.05}, {5.5, 0.08}, {8.2, 0.04}};
  std::vector<Complex> residues {{0, 0}, {0, 0}, {0, -3.0},
    {0, -0.5},                               // 1/v pole
    {0, -0.5}, {0, -0.2}, {0, -0.3}, {0, 0}, //
    {0, -1.0}, {0, 0}, {0, -0.6}, {0, -0.1}, //
    {0, -0.7}, {0, -0.1}, {0, -0.4}, {0, 0}};
  std::vector<int64_t> wstart {0, 0, 0, 0, 1, 2, 2, 3, 3};
  std::vector<int64_t> wend {2, 2, 3, 3, 3, 4, 4, 4, 4};
  std::vector<double> bounds {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // curvefit terms (1/E, 1/sqrt(E), 1) per channel, identical in each window
  std::vector<double> cf_terms {
    1.0, 0.1, 0.5, 0.05, 0.2, 0.0, 0.1, 0.0, 3.0, 0.05, 0.1, 0.01};

  // Low-energy segment
  double lowe_E_min = 1.0e-5, lowe_E_max = 1.0;
  std::vector<int64_t> lowe_mts {2, 102, 107};
  std::vector<double> lowe_cf {0.1, 1.0, 0.2, 0.05, 0.3, 0.0, 2.0, 0.01, 0.0};

  // Fast segment: 3 windows over [100, 1e6] eV
  double fast_E_max = 1.0e6;
  std::vector<double> fast_bounds {10, 30, 100, 1000};
  std::vector<int64_t> fast_mts {2, 51, 18, 102, 16};
  std::vector<double> E_on {0.0, 5000.0, 0.0, 0.0, 2.0e5};
  std::vector<double> E_off {1e6, 1e6, 1e6, 1e6, 1e6};
  std::vector<Complex> fast_poles {{20.0, 1.0}, {60.0, 2.0}};
  std::vector<Complex> fast_residues {{0, -2.0}, {0, -0.5}, {0, -0.1},
    {0, -0.3}, {0, -0.2}, {0, -3.0}, {0, -1.0}, {0, -0.2}, {0, -0.2},
    {0, -0.4}};
  std::vector<int64_t> fast_wstart {0, 0, 1};
  std::vector<int64_t> fast_wend {1, 2, 2};
  std::vector<Complex> ext_poles {{15.0, 30.0}, {60.0, 50.0}, {300.0, 200.0}};
  std::vector<Complex> ext_residues {{0, -20}, {0, -2}, {0, -1}, {0, -3},
    {0, -1}, {0, -30}, {0, -5}, {0, -1}, {0, -2}, {0, -2}, {0, -100}, {0, -10},
    {0, -2}, {0, -3}, {0, -5}};
  // curvefit terms (1/E, 1/sqrt(E)) per channel, identical in each window
  std::vector<double> fast_cf_terms {
    10, 1, 0.5, 2, 0.5, 1, 0.1, 0.05, 0.2, 0.1};
};

const Library& lib()
{
  static Library L;
  return L;
}

void write_complex(hid_t group, const char* name, std::vector<hsize_t> dims,
  const std::vector<Complex>& data)
{
  struct complex_t {
    double re;
    double im;
  };
  hid_t ctype = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
  H5Tinsert(ctype, "r", HOFFSET(complex_t, re), H5T_NATIVE_DOUBLE);
  H5Tinsert(ctype, "i", HOFFSET(complex_t, im), H5T_NATIVE_DOUBLE);
  hid_t space = H5Screate_simple(dims.size(), dims.data(), nullptr);
  hid_t dset =
    H5Dcreate(group, name, ctype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dset);
  H5Sclose(space);
  H5Tclose(ctype);
}

template<typename T>
void write_array(hid_t group, const char* name, std::vector<hsize_t> dims,
  const std::vector<T>& data)
{
  write_dataset_lowlevel(group, dims.size(), dims.data(), name,
    H5TypeMap<T>::type_id, H5S_ALL, false, data.data());
}

//! Repeat a per-window block n_win times
template<typename T>
std::vector<T> repeat(const std::vector<T>& block, int n)
{
  std::vector<T> out;
  for (int i = 0; i < n; ++i)
    out.insert(out.end(), block.begin(), block.end());
  return out;
}

void write_library(const std::string& path)
{
  const Library& L = lib();
  hid_t file = file_open(path, 'w');
  write_attribute(file, "filetype", "data_wmp");
  write_attribute(file, "version", WMP_VERSION);
  hid_t nuc = create_group(file, L.name);

  hid_t res = create_group(nuc, "resolved");
  write_dataset(res, "sqrtAWR", L.sqrt_awr);
  write_dataset(res, "E_min", L.E_min);
  write_dataset(res, "E_max", L.E_max);
  write_array(res, "mts", {4}, L.mts);
  write_complex(res, "poles", {4}, L.poles);
  write_complex(res, "residues", {4, 4}, L.residues);
  write_array(res, "win_pole_start", {9}, L.wstart);
  write_array(res, "win_pole_end", {9}, L.wend);
  write_array(res, "sqrtE_bounds", {10}, L.bounds);
  write_array(res, "curvefit", {9, 3, 4}, repeat(L.cf_terms, 9));
  write_attribute(res, "grid_uniform", 1);
  write_attribute(res, "grid_u_lowe", 1.0);
  write_attribute(res, "grid_spacing", 1.0);
  close_group(res);

  hid_t lowe = create_group(nuc, "lowe");
  write_dataset(lowe, "sqrtAWR", L.sqrt_awr);
  write_dataset(lowe, "E_min", L.lowe_E_min);
  write_dataset(lowe, "E_max", L.lowe_E_max);
  write_array(lowe, "mts", {3}, L.lowe_mts);
  write_array(lowe, "curvefit", {3, 3}, L.lowe_cf);
  close_group(lowe);

  hid_t fast = create_group(nuc, "fast");
  write_dataset(fast, "sqrtAWR", L.sqrt_awr);
  write_dataset(fast, "E_min", L.E_max);
  write_dataset(fast, "E_max", L.fast_E_max);
  write_array(fast, "mts", {5}, L.fast_mts);
  write_array(fast, "mt_E_on", {5}, L.E_on);
  write_array(fast, "mt_E_off", {5}, L.E_off);
  write_complex(fast, "poles", {2}, L.fast_poles);
  write_complex(fast, "residues", {2, 5}, L.fast_residues);
  write_array(fast, "win_pole_start", {3}, L.fast_wstart);
  write_array(fast, "win_pole_end", {3}, L.fast_wend);
  write_array(fast, "sqrtE_bounds", {4}, L.fast_bounds);
  write_complex(fast, "ext_poles", {3, 1}, L.ext_poles);
  write_complex(fast, "ext_residues", {3, 1, 5}, L.ext_residues);
  write_array(fast, "curvefit", {3, 2, 5}, repeat(L.fast_cf_terms, 3));
  close_group(fast);

  close_group(nuc);
  file_close(file);
}

//! Basis factors E^(k/2-1), broadened when T > 0
std::vector<double> basis(double E, double T, int n)
{
  std::vector<double> fac(std::max(n, 3));
  if (T > 0.0) {
    broaden_wmp_polynomials(
      E, lib().sqrt_awr / std::sqrt(K_BOLTZMANN * T), fac.size(), fac.data());
  } else {
    for (int k = 0; k < n; ++k)
      fac[k] = std::pow(E, 0.5 * k - 1.0);
  }
  fac.resize(n);
  return fac;
}

//! Reference per-channel cross sections in the low-energy segment
std::vector<double> ref_lowe(double E, double T)
{
  const Library& L = lib();
  auto fac = basis(E, T, 3);
  std::vector<double> xs(3, 0.0);
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c)
      xs[c] += L.lowe_cf[k * 3 + c] * fac[k];
  return xs;
}

//! Reference per-channel cross sections in the resolved segment, using the
//! exact Faddeeva function
std::vector<double> ref_resolved(double E, double T)
{
  const Library& L = lib();
  double u = std::sqrt(E);
  int w = std::min(std::max(static_cast<int>(u - 1.0), 0), 8);
  double xi = T > 0.0 ? std::sqrt(K_BOLTZMANN * T) / L.sqrt_awr : 0.0;
  std::vector<double> xs(4, 0.0);
  for (int64_t i = L.wstart[w]; i < L.wend[w]; ++i) {
    Complex p = L.poles[i];
    Complex b;
    if (xi == 0.0 || p.real() == 0.0) {
      b = 1.0 / (u - p);
    } else {
      Complex z = (p - u) / xi;
      b = ((u + p) + p * p * Complex(0.0, SQRT_PI / xi) * Faddeeva::w(z)) / E;
    }
    for (int c = 0; c < 4; ++c)
      xs[c] += (L.residues[i * 4 + c] * b).real();
  }
  auto fac = basis(E, T, 3);
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 4; ++c)
      xs[c] += L.cf_terms[k * 4 + c] * fac[k];
  return xs;
}

//! Reference per-channel cross sections in the fast segment
std::vector<double> ref_fast(double E)
{
  const Library& L = lib();
  double u = std::sqrt(E);
  int w = 0;
  while (w < 2 && u >= L.fast_bounds[w + 1])
    ++w;
  std::vector<double> xs(5, 0.0);
  for (int64_t i = L.fast_wstart[w]; i < L.fast_wend[w]; ++i) {
    Complex phi = 1.0 / (u - L.fast_poles[i]);
    for (int c = 0; c < 5; ++c)
      xs[c] += (L.fast_residues[i * 5 + c] * phi).real();
  }
  Complex phi = 1.0 / (u - L.ext_poles[w]);
  for (int c = 0; c < 5; ++c)
    xs[c] += (L.ext_residues[w * 5 + c] * phi).real();
  auto fac = basis(E, 0.0, 2);
  for (int k = 0; k < 2; ++k)
    for (int c = 0; c < 5; ++c)
      xs[c] += L.fast_cf_terms[k * 5 + c] * fac[k];
  for (int c = 0; c < 5; ++c)
    if (E < L.E_on[c] || E > L.E_off[c])
      xs[c] = 0.0;
  return xs;
}

struct Fixture {
  std::string path = "test_wmp_library.h5";
  WindowedMultipole wmp;
  Fixture()
  {
    write_library(path);
    hid_t file = file_open(path, 'r');
    REQUIRE(check_wmp_version(file) == WMP_VERSION[0]);
    hid_t group = open_group(file, lib().name);
    wmp = WindowedMultipole(group, WMP_VERSION[0]);
    close_group(group);
    file_close(file);
  }
  ~Fixture() { std::remove(path.c_str()); }
};

} // namespace

TEST_CASE("WMP version 2: low-energy segment")
{
  Fixture f;
  REQUIRE(f.wmp.lowe_present_);
  REQUIRE(f.wmp.fissionable_);
  for (double T : {0.0, 300.0, 1200.0}) {
    for (double E : {1.0e-4, 1.0e-2, 0.5}) {
      double sqrtkT = std::sqrt(K_BOLTZMANN * T);
      auto ref = ref_lowe(E, T);
      double dep[WMP_N_DEP] {};
      auto [s, a, fis] = f.wmp.evaluate(E, sqrtkT, dep);
      CHECK_THAT(s, WithinRel(ref[0], 1e-12));
      CHECK_THAT(a, WithinRel(ref[1] + ref[2], 1e-12));
      CHECK(fis == 0.0);
      CHECK_THAT(dep[0], WithinRel(ref[1], 1e-12)); // (n,gamma)
      CHECK_THAT(dep[2], WithinRel(ref[2], 1e-12)); // (n,alpha)
      CHECK(dep[1] == 0.0);
      CHECK(dep[3] == 0.0);
    }
  }
}

TEST_CASE("WMP version 2: resolved segment")
{
  Fixture f;
  std::vector<double> energies {1.5, 6.25, 7.0, 30.25, 67.24, 99.0};

  SECTION("0 K pole sum is exact")
  {
    for (double E : energies) {
      auto ref = ref_resolved(E, 0.0);
      double dep[WMP_N_DEP] {};
      auto [s, a, fis] = f.wmp.evaluate(E, 0.0, dep);
      CHECK_THAT(s, WithinRel(ref[0], 1e-12));
      CHECK_THAT(a, WithinRel(ref[1] + ref[2] + ref[3], 1e-12));
      CHECK_THAT(fis, WithinRel(ref[1], 1e-12));
      CHECK_THAT(dep[0], WithinRel(ref[2], 1e-12));
      CHECK_THAT(dep[2], WithinRel(ref[3], 1e-12));
    }
  }

  SECTION("Broadened pole sum matches the exact Faddeeva function")
  {
    for (double T : {300.0, 900.0, 2500.0}) {
      double sqrtkT = std::sqrt(K_BOLTZMANN * T);
      for (double E : energies) {
        auto ref = ref_resolved(E, T);
        double dep[WMP_N_DEP] {};
        auto [s, a, fis] = f.wmp.evaluate(E, sqrtkT, dep);
        CHECK_THAT(s, WithinRel(ref[0], 1e-4));
        CHECK_THAT(a, WithinRel(ref[1] + ref[2] + ref[3], 1e-4));
        CHECK_THAT(fis, WithinRel(ref[1], 1e-4));
        CHECK_THAT(dep[0], WithinRel(ref[2], 1e-4));
        CHECK_THAT(dep[2], WithinRel(ref[3], 1e-4));
        CHECK(dep[1] == 0.0);
      }
    }
  }

  SECTION("Depletion channels are not filled when not requested")
  {
    auto [s, a, fis] = f.wmp.evaluate(6.25, 0.0);
    auto ref = ref_resolved(6.25, 0.0);
    CHECK_THAT(s, WithinRel(ref[0], 1e-12));
  }
}

TEST_CASE("WMP version 2: fast segment")
{
  Fixture f;
  REQUIRE(f.wmp.fast_present_);
  REQUIRE(f.wmp.fast_channel_of_mt(51) == 1);
  REQUIRE(f.wmp.fast_channel_of_mt(4) == -1);

  // 4000 eV and 1e5 eV sit below the MT51 and MT16 thresholds, which fall
  // inside their windows
  for (double E : {500.0, 4000.0, 6000.0, 1.0e5, 3.0e5, 9.0e5}) {
    auto ref = ref_fast(E);
    double dep[WMP_N_DEP] {};
    auto [total, a, fis] = f.wmp.evaluate_fast(E, dep);
    double ref_total = ref[0] + ref[1] + ref[2] + ref[3] + ref[4];
    CHECK_THAT(total, WithinRel(ref_total, 1e-12));
    CHECK_THAT(a, WithinRel(ref[2] + ref[3], 1e-12));
    CHECK_THAT(fis, WithinRel(ref[2], 1e-12));
    CHECK_THAT(f.wmp.evaluate_fast_elastic(E), WithinRel(ref[0], 1e-12));
    CHECK_THAT(dep[0], WithinRel(ref[3], 1e-12)); // (n,gamma)
    CHECK_THAT(dep[3], WithinAbs(ref[4], 1e-12)); // (n,2n), gated
    CHECK(dep[1] == 0.0);

    WindowedMultipole::FastChanCtx ctx;
    f.wmp.fast_chan_begin(E, ctx);
    for (int c = 0; c < 5; ++c)
      CHECK_THAT(f.wmp.fast_chan_xs(ctx, E, c), WithinAbs(ref[c], 1e-12));
  }
}

TEST_CASE("WMP version 2: temperature derivative")
{
  Fixture f;
  const double T = 600.0, h = 10.0;
  // Central differences of evaluate() carry the rational Faddeeva kernel's
  // relative error (~1e-5) divided by the step, so allow that on top of 1%
  auto check = [&](double d, double xp, double xm) {
    double fd = (xp - xm) / (2 * h);
    double tol = 1e-2 * std::abs(fd) + 2e-5 * std::max(xp, xm) / h;
    CHECK_THAT(d, WithinAbs(fd, tol));
  };
  for (double E : {0.5, 1.5, 6.25, 30.25}) {
    auto [ds, da, df] = f.wmp.evaluate_deriv(E, std::sqrt(K_BOLTZMANN * T));
    auto [sp, ap, fp] = f.wmp.evaluate(E, std::sqrt(K_BOLTZMANN * (T + h)));
    auto [sm, am, fm] = f.wmp.evaluate(E, std::sqrt(K_BOLTZMANN * (T - h)));
    check(ds, sp, sm);
    check(da, ap, am);
    check(df, fp, fm);
  }
  // The fast segment is not broadened
  auto [ds, da, df] = f.wmp.evaluate_deriv(1.0e5, std::sqrt(K_BOLTZMANN * T));
  CHECK(ds == 0.0);
  CHECK(da == 0.0);
  CHECK(df == 0.0);
}
