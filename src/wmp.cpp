#include "openmc/wmp.h"

#include "openmc/constants.h"
#include "openmc/cross_sections.h"
#include "openmc/error.h" // for writing messages
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/nuclide.h"

#include <fmt/core.h>

#include <algorithm> // for min
#include <cmath>

namespace openmc {

//========================================================================
// WindowedeMultipole implementation
//========================================================================

WindowedMultipole::WindowedMultipole(hid_t group)
{
  // Get name of nuclide from group, removing leading '/'
  name_ = object_name(group).substr(1);

  // Two-segment global-pole format: /resolved (analytic Doppler) +
  // /fast (T=0 pole sum with per-channel residues and threshold clamps).
  if (object_exists(group, "resolved")) {
    read_global_segment(open_group(group, "resolved"));
    if (object_exists(group, "fast")) {
      read_fast_segment(open_group(group, "fast"));
    }
    return;
  }

  // Global-pole format (AAA-fit libraries): detected by the presence of
  // the external-pole dataset. See wmp.h for the format description.
  if (object_exists(group, "ext_poles")) {
    read_global_segment(group);
    return;
  }

  // Read scalar values.
  read_dataset(group, "spacing", inv_spacing_);
  inv_spacing_ = 1.0 / inv_spacing_;
  read_dataset(group, "sqrtAWR", sqrt_awr_);
  read_dataset(group, "E_min", E_min_);
  read_dataset(group, "E_max", E_max_);

  // Read the "data" array.  Use its shape to figure out the number of poles
  // and residue types in this data.
  read_dataset(group, "data", data_);
  int n_residues = data_.shape(1) - 1;

  // Check to see if this data includes fission residues.
  fissionable_ = (n_residues == 3);

  // Read the "windows" array and use its shape to figure out the number of
  // windows.
  tensor::Tensor<int> windows;
  read_dataset(group, "windows", windows);
  int n_windows = windows.shape(0);
  windows -= 1; // Adjust to 0-based indices

  // Read the "broaden_poly" arrays.
  tensor::Tensor<bool> broaden_poly;
  read_dataset(group, "broaden_poly", broaden_poly);
  if (n_windows != broaden_poly.shape(0)) {
    fatal_error("broaden_poly array shape is not consistent with the windows "
                "array shape in WMP library for " +
                name_ + ".");
  }

  // Read the "curvefit" array.
  read_dataset(group, "curvefit", curvefit_);
  if (n_windows != curvefit_.shape(0)) {
    fatal_error("curvefit array shape is not consistent with the windows "
                "array shape in WMP library for " +
                name_ + ".");
  }
  fit_order_ = curvefit_.shape(1) - 1;

  // Check the code is compiling to work with sufficiently high fit order
  if (fit_order_ + 1 > MAX_POLY_COEFFICIENTS) {
    fatal_error(fmt::format(
      "Need to compile with WindowedMultipole::MAX_POLY_COEFFICIENTS = {}",
      fit_order_ + 1));
  }

  // Move window information into a vector
  window_info_.resize(n_windows);
  for (int i = 0; i < n_windows; ++i) {
    window_info_[i].index_start = windows(i, 0);
    window_info_[i].index_end = windows(i, 1);
    window_info_[i].broaden_poly = broaden_poly[i];
  }
}

void WindowedMultipole::read_global_segment(hid_t group)
{
    global_format_ = true;
    read_dataset(group, "sqrtAWR", sqrt_awr_);
    read_dataset(group, "E_min", E_min_);
    read_dataset(group, "E_max", E_max_);

    tensor::Tensor<std::complex<double>> poles;    // (n_pool,)
    tensor::Tensor<std::complex<double>> residues; // (n_pool, 3): s, f, g
    read_dataset(group, "poles", poles);
    read_dataset(group, "residues", residues);
    fissionable_ = true; // [MT2, MT18, MT102] layout
    if (residues.shape(1) == 2)
      fissionable_ = false; // [MT2, MT102]

    int n_pool = poles.shape(0);
    gdata_ = tensor::Tensor<std::complex<double>>({(size_t)n_pool, 4});
    for (int i = 0; i < n_pool; ++i) {
      gdata_(i, MP_EA) = poles[i];
      if (fissionable_) {
        gdata_(i, MP_RS) = residues(i, 0);              // MT2
        gdata_(i, MP_RA) = residues(i, 1) + residues(i, 2); // MT18 + MT102
        gdata_(i, MP_RF) = residues(i, 1);              // MT18
      } else {
        gdata_(i, MP_RS) = residues(i, 0);
        gdata_(i, MP_RA) = residues(i, 1);
        gdata_(i, MP_RF) = 0.0;
      }
    }

    tensor::Tensor<int64_t> wstart, wend;
    read_dataset(group, "win_pole_start", wstart);
    read_dataset(group, "win_pole_end", wend);
    int n_win = wstart.shape(0);
    win_start_.resize(n_win);
    win_end_.resize(n_win);
    for (int i = 0; i < n_win; ++i) {
      win_start_[i] = wstart[i];
      win_end_[i] = wend[i];
    }

    tensor::Tensor<double> bounds;
    read_dataset(group, "sqrtE_bounds", bounds);
    sqrtE_bounds_.assign(bounds.begin(), bounds.end());

    // Build the two-level O(1) window index (see wmp.h)
    {
      int n_cells = 4 * n_win;
      double u_lo = sqrtE_bounds_.front();
      double u_hi = sqrtE_bounds_.back();
      cell_inv_du_ = n_cells / (u_hi - u_lo);
      cell_first_.resize(n_cells);
      int w = 0;
      for (int i = 0; i < n_cells; ++i) {
        double u_cell = u_lo + i / cell_inv_du_;
        while (w < n_win - 1 && sqrtE_bounds_[w + 1] <= u_cell)
          ++w;
        cell_first_[i] = w;
      }
    }

    tensor::Tensor<std::complex<double>> xp;  // (n_win, 4)
    tensor::Tensor<std::complex<double>> xr;  // (n_win, 4, 3)
    read_dataset(group, "ext_poles", xp);
    read_dataset(group, "ext_residues", xr);
    int n_ext = xp.shape(1);
    ext_data_ =
      tensor::Tensor<std::complex<double>>({(size_t)n_win, (size_t)n_ext, 4});
    for (int i = 0; i < n_win; ++i) {
      for (int j = 0; j < n_ext; ++j) {
        ext_data_(i, j, MP_EA) = xp(i, j);
        if (fissionable_) {
          ext_data_(i, j, MP_RS) = xr(i, j, 0);
          ext_data_(i, j, MP_RA) = xr(i, j, 1) + xr(i, j, 2);
          ext_data_(i, j, MP_RF) = xr(i, j, 1);
        } else {
          ext_data_(i, j, MP_RS) = xr(i, j, 0);
          ext_data_(i, j, MP_RA) = xr(i, j, 1);
          ext_data_(i, j, MP_RF) = 0.0;
        }
      }
    }
}

void WindowedMultipole::read_fast_segment(hid_t group)
{
  fast_present_ = true;
  read_dataset(group, "E_max", fast_E_max_);

  tensor::Tensor<std::complex<double>> poles;    // (n_pool,)
  tensor::Tensor<std::complex<double>> residues; // (n_pool, n_mts)
  tensor::Tensor<int64_t> mts, wstart, wend;
  tensor::Tensor<double> e_on, e_off, bounds;
  tensor::Tensor<std::complex<double>> xp; // (n_win, n_ext)
  tensor::Tensor<std::complex<double>> xr; // (n_win, n_ext, n_mts)
  read_dataset(group, "poles", poles);
  read_dataset(group, "residues", residues);
  read_dataset(group, "mts", mts);
  read_dataset(group, "mt_E_on", e_on);
  read_dataset(group, "mt_E_off", e_off);
  read_dataset(group, "win_pole_start", wstart);
  read_dataset(group, "win_pole_end", wend);
  read_dataset(group, "sqrtE_bounds", bounds);
  read_dataset(group, "ext_poles", xp);
  read_dataset(group, "ext_residues", xr);

  int n_win = wstart.shape(0);
  int n_mts = mts.shape(0);
  int n_ext = xp.shape(1);
  fast_bounds_.assign(bounds.begin(), bounds.end());

  // Transport kind of each channel: 0 other-scatter (inelastic, (n,xn),
  // MT5, ...), 1 fission, 2 disappearance lump, 3 elastic.
  vector<int> kind(n_mts, 0);
  for (int c = 0; c < n_mts; ++c) {
    int mt = mts[c];
    if (mt == 2) {
      kind[c] = 3;
    } else if (mt == 18) {
      kind[c] = 1;
    } else if (mt == 101 || mt == 102 || mt == 103 || mt == 107) {
      kind[c] = 2;
    }
  }

  // Flatten each window's pointer range + externals; collapse residues of
  // fully-active channels into the four transport vectors, keep channels
  // that turn on/off inside the window as individually-clamped partials.
  fw_off_.assign(n_win + 1, 0);
  fast_partials_.resize(n_win);
  for (int w = 0; w < n_win; ++w) {
    int np = (wend[w] - wstart[w]) + n_ext;
    if (np > 96) {
      fatal_error(fmt::format("Fast WMP window with {} active poles exceeds "
                              "the evaluator's stack buffer (96) for {}.",
        np, name_));
    }
    fw_off_[w + 1] = fw_off_[w] + np;
  }
  int n_flat = fw_off_[n_win];
  fw_pole_.assign(n_flat, 0.0);
  fw_r_el_.assign(n_flat, 0.0);
  fw_r_fis_.assign(n_flat, 0.0);
  fw_r_dis_.assign(n_flat, 0.0);
  fw_r_oth_.assign(n_flat, 0.0);

  for (int w = 0; w < n_win; ++w) {
    double e_lo = fast_bounds_[w] * fast_bounds_[w];
    double e_hi = fast_bounds_[w + 1] * fast_bounds_[w + 1];
    int np = wend[w] - wstart[w];
    int base = fw_off_[w];
    for (int k = 0; k < np; ++k)
      fw_pole_[base + k] = poles[wstart[w] + k];
    for (int j = 0; j < n_ext; ++j)
      fw_pole_[base + np + j] = xp(w, j);

    for (int c = 0; c < n_mts; ++c) {
      // Overlap of the channel's active range with this window
      if (e_on[c] >= e_hi || e_off[c] <= e_lo)
        continue; // channel inactive across the whole window
      // Exact comparison: a channel whose clamp lies even 1 ulp inside the
      // window is kept as a partial (clamped pointwise) — collapsing it
      // would leak the fit's unconstrained below-threshold value into the
      // sliver between the window edge and the clamp.
      bool full = (e_on[c] <= e_lo) && (e_off[c] >= e_hi);
      if (full) {
        auto& acc = (kind[c] == 3)   ? fw_r_el_
                    : (kind[c] == 1) ? fw_r_fis_
                    : (kind[c] == 2) ? fw_r_dis_
                                     : fw_r_oth_;
        for (int k = 0; k < np; ++k)
          acc[base + k] += residues(wstart[w] + k, c);
        for (int j = 0; j < n_ext; ++j)
          acc[base + np + j] += xr(w, j, c);
      } else {
        // Partially active: keep separate, clamp at evaluation time
        FastPartial part;
        part.E_on = e_on[c];
        part.E_off = e_off[c];
        part.kind = kind[c];
        part.res_off = (int)fp_res_.size();
        for (int k = 0; k < np; ++k)
          fp_res_.push_back(residues(wstart[w] + k, c));
        for (int j = 0; j < n_ext; ++j)
          fp_res_.push_back(xr(w, j, c));
        fast_partials_[w].push_back(part);
      }
    }
  }

  // O(1) two-level index over the fast windows
  {
    int n_cells = 4 * n_win;
    double u_lo = fast_bounds_.front();
    double u_hi = fast_bounds_.back();
    fast_cell_inv_du_ = n_cells / (u_hi - u_lo);
    fast_cell_first_.resize(n_cells);
    int w = 0;
    for (int i = 0; i < n_cells; ++i) {
      double u_cell = u_lo + i / fast_cell_inv_du_;
      while (w < n_win - 1 && fast_bounds_[w + 1] <= u_cell)
        ++w;
      fast_cell_first_[i] = w;
    }
  }
}

std::tuple<double, double, double, double> WindowedMultipole::evaluate_fast(
  double E) const
{
  double sqrtE = std::sqrt(E);

  // O(1) window lookup over the fast bounds
  int n_win = fw_off_.size() - 1;
  int cell = (int)((sqrtE - fast_bounds_[0]) * fast_cell_inv_du_);
  cell = std::min(std::max(cell, 0), (int)fast_cell_first_.size() - 1);
  int w = fast_cell_first_[cell];
  while (w < n_win - 1 && sqrtE >= fast_bounds_[w + 1])
    ++w;

  int k0 = fw_off_[w];
  int k1 = fw_off_[w + 1];
  int np = k1 - k0;

  // One reciprocal per pole; accumulate the four collapsed transport sums
  constexpr int FAST_MAX_ACTIVE = 96;
  std::complex<double> phi[FAST_MAX_ACTIVE];
  double el = 0.0, fis = 0.0, dis = 0.0, oth = 0.0;
  for (int k = 0; k < np; ++k) {
    phi[k] = 1.0 / (sqrtE - fw_pole_[k0 + k]);
    el += (fw_r_el_[k0 + k] * phi[k]).real();
    fis += (fw_r_fis_[k0 + k] * phi[k]).real();
    dis += (fw_r_dis_[k0 + k] * phi[k]).real();
    oth += (fw_r_oth_[k0 + k] * phi[k]).real();
  }

  // Partially-clamped channels (rare; only near their thresholds)
  for (const auto& part : fast_partials_[w]) {
    if (E < part.E_on || E > part.E_off)
      continue;
    double s = 0.0;
    for (int k = 0; k < np; ++k)
      s += (fp_res_[part.res_off + k] * phi[k]).real();
    switch (part.kind) {
    case 3:
      el += s;
      break;
    case 1:
      fis += s;
      break;
    case 2:
      dis += s;
      break;
    default:
      oth += s;
    }
  }

  // Clip tiny negative excursions from the fit (thresholds, near-zero xs)
  el = std::max(el, 0.0);
  fis = std::max(fis, 0.0);
  dis = std::max(dis, 0.0);
  oth = std::max(oth, 0.0);

  double total = el + fis + dis + oth;
  return {total, el, dis + fis, fis};
}

std::tuple<double, double, double> WindowedMultipole::evaluate(
  double E, double sqrtkT) const
{
  using namespace std::complex_literals;

  // ==========================================================================
  // Bookkeeping

  // Define some frequently used variables.
  double sqrtE = std::sqrt(E);
  double invE = 1.0 / E;

  // ==========================================================================
  // Global-pole format (see wmp.h): variable-width windows, pool pointer
  // range + external poles, sigma-basis analytic broadening.

  if (global_format_) {
    // Variable-width windows: O(1) two-level index (uniform coarse grid in
    // sqrt(E) + short forward walk; see wmp.h)
    int n_win = win_start_.size();
    int cell = (int)((sqrtE - sqrtE_bounds_[0]) * cell_inv_du_);
    cell = std::min(std::max(cell, 0), (int)cell_first_.size() - 1);
    int i_window = cell_first_[cell];
    while (i_window < n_win - 1 && sqrtE >= sqrtE_bounds_[i_window + 1])
      ++i_window;

    double sig_s = 0.0, sig_a = 0.0, sig_f = 0.0;
    int n_ext = ext_data_.shape(1);

    auto accumulate = [&](const std::complex<double>& p,
                        const std::complex<double>& rs,
                        const std::complex<double>& ra,
                        const std::complex<double>& rf, double xi) {
      std::complex<double> base;
      if (xi <= 0.0 || p.real() == 0.0) {
        // 0K pole sum; the 1/v pole (Re p = 0) is exactly invariant under
        // free-gas broadening and is always evaluated unbroadened.
        base = 1.0 / (sqrtE - p);
      } else {
        std::complex<double> z = (p - sqrtE) / xi;
        if (std::abs(z.real()) > 1.0e2 || std::abs(z.imag()) > 1.0e2) {
          // |z| >> 1: broadening changes the tail by O(1/|z|^2) < 1e-4,
          // while the Faddeeva form suffers catastrophic cancellation
          // ((u+p) vs p^2*w terms) amplified by (|p|/u)^2 -- very broad
          // background poles evaluated at very low E. Use the 0K form.
          base = 1.0 / (sqrtE - p);
        } else {
          // (1/E) Re[r ((u+p) + p^2 (i sqrt(pi)/xi) w((p-u)/xi))]
          std::complex<double> w_val = faddeeva(z);
          base = ((sqrtE + p) +
                   p * p * (std::complex<double>(0.0, SQRT_PI) / xi) * w_val) *
                 invE;
        }
      }
      sig_s += (rs * base).real();
      sig_a += (ra * base).real();
      if (fissionable_)
        sig_f += (rf * base).real();
    };

    // xi = sqrt(kT/AWR) in sqrt(eV); sqrtkT = sqrt(kT)
    double xi = (sqrtkT > 0.0) ? sqrtkT / sqrt_awr_ : 0.0;
    for (int i = win_start_[i_window]; i < win_end_[i_window]; ++i) {
      accumulate(
        gdata_(i, MP_EA), gdata_(i, MP_RS), gdata_(i, MP_RA), gdata_(i, MP_RF), xi);
    }
    for (int j = 0; j < n_ext; ++j) {
      accumulate(ext_data_(i_window, j, MP_EA), ext_data_(i_window, j, MP_RS),
        ext_data_(i_window, j, MP_RA), ext_data_(i_window, j, MP_RF), xi);
    }
    return std::make_tuple(sig_s, sig_a, sig_f);
  }

  // Locate window containing energy
  int i_window = std::min(window_info_.size() - 1,
    static_cast<size_t>((sqrtE - std::sqrt(E_min_)) * inv_spacing_));
  const auto& window {window_info_[i_window]};

  // Initialize the ouptut cross sections
  double sig_s = 0.0;
  double sig_a = 0.0;
  double sig_f = 0.0;

  // ==========================================================================
  // Add the contribution from the curvefit polynomial.

  if (sqrtkT > 0.0 && window.broaden_poly) {
    // Broaden the curvefit.
    double dopp = sqrt_awr_ / sqrtkT;
    array<double, MAX_POLY_COEFFICIENTS> broadened_polynomials;
    broaden_wmp_polynomials(
      E, dopp, fit_order_ + 1, broadened_polynomials.data());
    for (int i_poly = 0; i_poly < fit_order_ + 1; ++i_poly) {
      sig_s +=
        curvefit_(i_window, i_poly, FIT_S) * broadened_polynomials[i_poly];
      sig_a +=
        curvefit_(i_window, i_poly, FIT_A) * broadened_polynomials[i_poly];
      if (fissionable_) {
        sig_f +=
          curvefit_(i_window, i_poly, FIT_F) * broadened_polynomials[i_poly];
      }
    }
  } else {
    // Evaluate as if it were a polynomial
    double temp = invE;
    for (int i_poly = 0; i_poly < fit_order_ + 1; ++i_poly) {
      sig_s += curvefit_(i_window, i_poly, FIT_S) * temp;
      sig_a += curvefit_(i_window, i_poly, FIT_A) * temp;
      if (fissionable_) {
        sig_f += curvefit_(i_window, i_poly, FIT_F) * temp;
      }
      temp *= sqrtE;
    }
  }

  // ==========================================================================
  // Add the contribution from the poles in this window.

  if (sqrtkT == 0.0) {
    // If at 0K, use asymptotic form.
    for (int i_pole = window.index_start; i_pole <= window.index_end;
         ++i_pole) {
      std::complex<double> psi_chi = -1.0i / (data_(i_pole, MP_EA) - sqrtE);
      std::complex<double> c_temp = psi_chi * invE;
      sig_s += (data_(i_pole, MP_RS) * c_temp).real();
      sig_a += (data_(i_pole, MP_RA) * c_temp).real();
      if (fissionable_) {
        sig_f += (data_(i_pole, MP_RF) * c_temp).real();
      }
    }
  } else {
    // At temperature, use Faddeeva function-based form.
    double dopp = sqrt_awr_ / sqrtkT;
    for (int i_pole = window.index_start; i_pole <= window.index_end;
         ++i_pole) {
      std::complex<double> z = (sqrtE - data_(i_pole, MP_EA)) * dopp;
      std::complex<double> w_val = faddeeva(z) * dopp * invE * SQRT_PI;
      sig_s += (data_(i_pole, MP_RS) * w_val).real();
      sig_a += (data_(i_pole, MP_RA) * w_val).real();
      if (fissionable_) {
        sig_f += (data_(i_pole, MP_RF) * w_val).real();
      }
    }
  }

  return std::make_tuple(sig_s, sig_a, sig_f);
}

std::tuple<double, double, double> WindowedMultipole::evaluate_deriv(
  double E, double sqrtkT) const
{
  // ==========================================================================
  // Bookkeeping

  // Define some frequently used variables.
  double sqrtE = std::sqrt(E);
  double invE = 1.0 / E;
  double T = sqrtkT * sqrtkT / K_BOLTZMANN;

  if (sqrtkT == 0.0) {
    fatal_error("Windowed multipole temperature derivatives are not implemented"
                " for 0 Kelvin cross sections.");
  }

  if (global_format_) {
    fatal_error("Windowed multipole temperature derivatives are not "
                "implemented for global-pole format libraries.");
  }

  // Locate us
  int i_window = (sqrtE - std::sqrt(E_min_)) * inv_spacing_;
  const auto& window {window_info_[i_window]};

  // Initialize the ouptut cross sections.
  double sig_s = 0.0;
  double sig_a = 0.0;
  double sig_f = 0.0;

  // TODO Polynomials: Some of the curvefit polynomials Doppler broaden so
  // rigorously we should be computing the derivative of those.  But in
  // practice, those derivatives are only large at very low energy and they
  // have no effect on reactor calculations.

  // ==========================================================================
  // Add the contribution from the poles in this window.

  double dopp = sqrt_awr_ / sqrtkT;
  for (int i_pole = window.index_start; i_pole <= window.index_end; ++i_pole) {
    std::complex<double> z = (sqrtE - data_(i_pole, MP_EA)) * dopp;
    std::complex<double> w_val = -invE * SQRT_PI * 0.5 * w_derivative(z, 2);
    sig_s += (data_(i_pole, MP_RS) * w_val).real();
    sig_a += (data_(i_pole, MP_RA) * w_val).real();
    if (fissionable_) {
      sig_f += (data_(i_pole, MP_RF) * w_val).real();
    }
  }
  double norm = -0.5 * sqrt_awr_ / std::sqrt(K_BOLTZMANN) * std::pow(T, -1.5);
  sig_s *= norm;
  sig_a *= norm;
  sig_f *= norm;

  return std::make_tuple(sig_s, sig_a, sig_f);
}

//========================================================================
// Non-member functions
//========================================================================

void check_wmp_version(hid_t file)
{
  if (attribute_exists(file, "version")) {
    array<int, 2> version;
    read_attribute(file, "version", version);
    if (version[0] != WMP_VERSION[0]) {
      fatal_error(fmt::format(
        "WMP data format uses version {}.{} whereas your installation of "
        "OpenMC expects version {}.x data.",
        version[0], version[1], WMP_VERSION[0]));
    }
  } else {
    fatal_error(fmt::format("WMP data does not indicate a version. Your "
                            "installation of OpenMC expects version {}x data.",
      WMP_VERSION[0]));
  }
}

void read_multipole_data(int i_nuclide)
{
  // Look for WMP data in cross_sections.xml
  const auto& nuc {data::nuclides[i_nuclide]};
  auto it = data::library_map.find({Library::Type::wmp, nuc->name_});

  // If no WMP library for this nuclide, just return
  if (it == data::library_map.end())
    return;

  // Check if WMP library exists
  int idx = it->second;
  std::string& filename = data::libraries[idx].path_;

  // Display message
  write_message(6, "Reading {} WMP data from {}", nuc->name_, filename);

  // Open file and make sure version is sufficient
  hid_t file = file_open(filename, 'r');
  check_wmp_version(file);

  // Read nuclide data from HDF5
  hid_t group = open_group(file, nuc->name_.c_str());
  nuc->multipole_ = make_unique<WindowedMultipole>(group);
  close_group(group);
  file_close(file);
}

void broaden_wmp_polynomials(double E, double dopp, int n, double factors[])
{
  // Broadening of polynomials follows procedure outlined in C. Josey, P. Ducru,
  // B. Forget, and K. Smith, "Windowed multipole for cross section Doppler
  // broadening," J. Comput. Phys., 307, 715-727 (2016).
  // https://doi.org/10.1016/j.jcp.2015.08.013

  // Factors is already pre-allocated
  double sqrtE = std::sqrt(E);
  double beta = sqrtE * dopp;
  double half_inv_dopp2 = 0.5 / (dopp * dopp);
  double quarter_inv_dopp4 = half_inv_dopp2 * half_inv_dopp2;

  double erf_beta;    // error function of beta
  double exp_m_beta2; // exp(-beta**2)
  if (beta > 6.0) {
    // Save time, ERF(6) is 1 to machine precision.
    // beta/sqrtpi*exp(-beta**2) is also approximately 1 machine epsilon.
    erf_beta = 1.;
    exp_m_beta2 = 0.;
  } else {
    erf_beta = std::erf(beta);
    exp_m_beta2 = std::exp(-beta * beta);
  }

  // Assume that, for sure, we'll use a second order (1/E, 1/V, const)
  // fit, and no less.
  factors[0] = erf_beta / E;
  factors[1] = 1. / sqrtE;
  factors[2] =
    factors[0] * (half_inv_dopp2 + E) + exp_m_beta2 / (beta * SQRT_PI);
  if (n > 3)
    factors[3] = factors[1] * (E + 3.0 * half_inv_dopp2);

  // Perform recursive broadening of high order components (Eq. 16)
  for (int i = 1; i < n - 3; i++) {
    double ip1_dbl = i + 1;
    factors[i + 3] =
      -factors[i - 1] * (ip1_dbl - 1.) * ip1_dbl * quarter_inv_dopp4 +
      factors[i + 1] * (E + (1. + 2. * ip1_dbl) * half_inv_dopp2);
  }
}

} // namespace openmc
