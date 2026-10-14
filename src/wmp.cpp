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
#include <complex>

namespace openmc {

//========================================================================
// Helpers for the version 2 format
//========================================================================

//! Transport group of a reaction channel identified by its MT number
static int channel_kind(int mt)
{
  if (mt == 2)
    return WindowedMultipole::KIND_ELASTIC;
  if (mt == 18)
    return WindowedMultipole::KIND_FISSION;
  if (mt == 101 || (mt >= 102 && mt <= 117))
    return WindowedMultipole::KIND_DISAPPEARANCE;
  return WindowedMultipole::KIND_OTHER;
}

//! Add a channel's value to the collapsed (scatter, absorption, fission)
//! triplet. Fission counts in absorption; other scattering counts in scatter.
template<typename T>
static void collapse_channel(int kind, const T& v, T& s, T& a, T& f)
{
  switch (kind) {
  case WindowedMultipole::KIND_FISSION:
    f += v;
    a += v;
    break;
  case WindowedMultipole::KIND_DISAPPEARANCE:
    a += v;
    break;
  default:
    s += v;
    break;
  }
}

//! Map a segment's MT list onto the DEPLETION_RX slots: dep[j] receives the
//! channel index carrying DEPLETION_RX[j], or -1 if absent.
static void build_dep_map(const vector<int>& mts, array<int, WMP_N_DEP>& dep)
{
  for (int j = 0; j < WMP_N_DEP; ++j) {
    dep[j] = -1;
    for (int c = 0; c < static_cast<int>(mts.size()); ++c) {
      if (mts[c] == DEPLETION_RX[j]) {
        dep[j] = c;
        break;
      }
    }
  }
}

//========================================================================
// WindowedMultipole implementation
//========================================================================

WindowedMultipole::WindowedMultipole(hid_t group, int version_major)
{
  // Get name of nuclide from group, removing leading '/'
  name_ = object_name(group).substr(1);

  if (version_major == WMP_VERSION_MAJOR_LEGACY) {
    if (object_exists(group, "resolved")) {
      fatal_error(fmt::format("WMP data for {} uses the version {}.x layout "
                              "but is stamped version {}.x. Update the file's "
                              "version attribute.",
        name_, WMP_VERSION[0], version_major));
    }
    read_legacy(group);
    return;
  }

  if (!object_exists(group, "resolved")) {
    fatal_error(fmt::format(
      "WMP data for {} does not contain a resolved segment.", name_));
  }
  hid_t resolved = open_group(group, "resolved");
  read_resolved_segment(resolved);
  close_group(resolved);

  if (object_exists(group, "lowe")) {
    hid_t lowe = open_group(group, "lowe");
    read_lowe_segment(lowe);
    close_group(lowe);
  }

  if (object_exists(group, "fast")) {
    hid_t fast = open_group(group, "fast");
    read_fast_segment(fast);
    close_group(fast);
  }
}

void WindowedMultipole::read_legacy(hid_t group)
{
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

void WindowedMultipole::read_lowe_segment(hid_t group)
{
  tensor::Tensor<double> cf; // (n_cf, n_ch)
  tensor::Tensor<int64_t> mts_ds;
  read_dataset(group, "curvefit", cf);
  read_dataset(group, "mts", mts_ds);
  read_dataset(group, "E_min", lowe_E_min_);
  read_dataset(group, "E_max", lowe_E_max_);
  int n_cf = cf.shape(0);
  int n_ch = cf.shape(1);
  if (mts_ds.shape(0) != n_ch) {
    fatal_error(fmt::format(
      "Low-energy curvefit and MT list have inconsistent shapes for {}.",
      name_));
  }
  if (n_cf > MAX_POLY_COEFFICIENTS) {
    fatal_error(fmt::format(
      "Need to compile with WindowedMultipole::MAX_POLY_COEFFICIENTS = {}",
      n_cf));
  }
  lowe_n_ = n_cf;

  vector<int> mts(n_ch);
  for (int c = 0; c < n_ch; ++c)
    mts[c] = static_cast<int>(mts_ds[c]);

  // Collapse the per-MT coefficients to the transport triplet
  lowe_cf_.assign(static_cast<size_t>(n_cf) * 3, 0.0);
  for (int k = 0; k < n_cf; ++k) {
    double vs = 0.0, va = 0.0, vf = 0.0;
    for (int c = 0; c < n_ch; ++c)
      collapse_channel(channel_kind(mts[c]), cf(k, c), vs, va, vf);
    lowe_cf_[3 * k + 0] = vs;
    lowe_cf_[3 * k + 1] = va;
    lowe_cf_[3 * k + 2] = vf;
  }

  // Keep the per-MT coefficients of the depletion reactions
  build_dep_map(mts, lowe_dep_ch_);
  lowe_dep_cf_.assign(static_cast<size_t>(n_cf) * WMP_N_DEP, 0.0);
  for (int j = 0; j < WMP_N_DEP; ++j) {
    const int c = lowe_dep_ch_[j];
    if (c < 0)
      continue;
    for (int k = 0; k < n_cf; ++k)
      lowe_dep_cf_[static_cast<size_t>(k) * WMP_N_DEP + j] = cf(k, c);
  }
  lowe_present_ = true;
}

void WindowedMultipole::read_resolved_segment(hid_t group)
{
  global_format_ = true;
  read_dataset(group, "sqrtAWR", sqrt_awr_);
  read_dataset(group, "E_min", E_min_);
  read_dataset(group, "E_max", E_max_);

  tensor::Tensor<std::complex<double>> poles;    // (n_pool,)
  tensor::Tensor<std::complex<double>> residues; // (n_pool, n_ch)
  tensor::Tensor<int64_t> mts_ds;                // (n_ch,)
  read_dataset(group, "poles", poles);
  read_dataset(group, "residues", residues);
  read_dataset(group, "mts", mts_ds);

  int n_pool = poles.shape(0);
  int n_ch = residues.shape(1);
  if (mts_ds.shape(0) != n_ch) {
    fatal_error(fmt::format(
      "Resolved residues and MT list have inconsistent shapes for {}.", name_));
  }

  // Classify the channels
  mts_.resize(n_ch);
  vector<int> kind(n_ch);
  fissionable_ = false;
  int n_dis = 0;
  for (int c = 0; c < n_ch; ++c) {
    mts_[c] = static_cast<int>(mts_ds[c]);
    kind[c] = channel_kind(mts_[c]);
    if (kind[c] == KIND_FISSION)
      fissionable_ = true;
    if (kind[c] == KIND_DISAPPEARANCE)
      ++n_dis;
  }

  // Depletion channel map. (n,gamma) = absorption - fission is exact only
  // when MT102 is the single disappearance channel.
  build_dep_map(mts_, dep_ch_);
  dep_needs_split_ = !(n_dis == 1 && dep_ch_[0] >= 0);
  res_mt_ = residues;

  // Build one record per pole with the residues collapsed to the transport
  // triplet. The 1/v pole keeps zero residues in its record since it is
  // added separately, unbroadened.
  gs_stride_ = fissionable_ ? 8 : 6;
  cf_ch_ = fissionable_ ? 3 : 2;
  vector<double> records(static_cast<size_t>(n_pool) * gs_stride_, 0.0);
  for (int i = 0; i < n_pool; ++i) {
    const std::complex<double> p = poles[i];
    std::complex<double> rs {0.0, 0.0}, ra {0.0, 0.0}, rf {0.0, 0.0};
    for (int c = 0; c < n_ch; ++c)
      collapse_channel(kind[c], residues(i, c), rs, ra, rf);

    double* r = &records[static_cast<size_t>(i) * gs_stride_];
    r[0] = p.real();
    r[1] = p.imag();
    if (p.real() == 0.0) {
      if (onev_present_) {
        fatal_error(fmt::format(
          "WMP data for {} carries more than one 1/v pole.", name_));
      }
      onev_present_ = true;
      onev_idx_ = i;
      onev_p_ = p;
      onev_rs_ = rs;
      onev_ra_ = ra;
      onev_rf_ = rf;
    } else {
      r[2] = rs.real();
      r[3] = rs.imag();
      r[4] = ra.real();
      r[5] = ra.imag();
      if (fissionable_) {
        r[6] = rf.real();
        r[7] = rf.imag();
      }
    }
  }

  // Windows: pointer ranges into the pole table and bounds in sqrt(E)
  tensor::Tensor<int64_t> wstart, wend;
  tensor::Tensor<double> bounds;
  read_dataset(group, "win_pole_start", wstart);
  read_dataset(group, "win_pole_end", wend);
  read_dataset(group, "sqrtE_bounds", bounds);
  int n_win = wstart.shape(0);
  if (wend.shape(0) != n_win || bounds.shape(0) != n_win + 1) {
    fatal_error(fmt::format(
      "Resolved window arrays have inconsistent shapes for {}.", name_));
  }
  win_start_.resize(n_win);
  vector<int> win_end(n_win);
  for (int w = 0; w < n_win; ++w) {
    win_start_[w] = wstart[w];
    win_end[w] = std::max(static_cast<int>(wend[w]), win_start_[w]);
    if (win_end[w] - win_start_[w] > MAX_WINDOW_POLES) {
      fatal_error(fmt::format("Resolved WMP window with {} poles exceeds "
                              "MAX_WINDOW_POLES = {} for {}.",
        win_end[w] - win_start_[w], MAX_WINDOW_POLES, name_));
    }
  }
  win_u_lo_ = bounds[0];

  // Window lookup: uniform mesh if the library says so, else a coarse grid
  if (attribute_exists(group, "grid_uniform")) {
    int gu = 0;
    double ul = 0.0, sp = 0.0;
    read_attribute(group, "grid_uniform", gu);
    read_attribute(group, "grid_u_lowe", ul);
    read_attribute(group, "grid_spacing", sp);
    if (gu == 1 && sp > 0.0) {
      grid_uniform_ = true;
      grid_u_lowe_ = ul;
      grid_inv_spacing_ = 1.0 / sp;
    }
  }
  if (!grid_uniform_) {
    int n_cells = 4 * n_win;
    double u_hi = bounds[n_win];
    cell_inv_du_ = n_cells / (u_hi - win_u_lo_);
    cell_first_.resize(n_cells);
    int w = 0;
    for (int i = 0; i < n_cells; ++i) {
      double u_cell = win_u_lo_ + i / cell_inv_du_;
      while (w < n_win - 1 && bounds[w + 1] <= u_cell)
        ++w;
      cell_first_[i] = w;
    }
  }

  if (object_exists(group, "ext_poles")) {
    fatal_error(fmt::format("Resolved-segment external poles are not "
                            "supported (WMP data for {}).",
      name_));
  }

  // Per-window curvefit, collapsed to the transport triplet and interleaved
  // per window as [s0, a0, (f0), s1, a1, (f1), ...]
  vector<double> cf_blk;
  if (object_exists(group, "curvefit")) {
    cf_present_ = true;
    tensor::Tensor<double> cf; // (n_win, n_cf, n_ch)
    read_dataset(group, "curvefit", cf);
    if (cf.shape(0) != n_win || cf.shape(2) != n_ch) {
      fatal_error(fmt::format(
        "Resolved curvefit array has inconsistent shape for {}.", name_));
    }
    cf_n_ = cf.shape(1);
    if (cf_n_ > MAX_POLY_COEFFICIENTS) {
      fatal_error(fmt::format(
        "Need to compile with WindowedMultipole::MAX_POLY_COEFFICIENTS = {}",
        cf_n_));
    }
    cf_mt_ = cf;
    cf_blk.assign(static_cast<size_t>(n_win) * cf_n_ * cf_ch_, 0.0);
    for (int w = 0; w < n_win; ++w) {
      for (int k = 0; k < cf_n_; ++k) {
        double vs = 0.0, va = 0.0, vf = 0.0;
        for (int c = 0; c < n_ch; ++c)
          collapse_channel(kind[c], cf(w, k, c), vs, va, vf);
        double* d = &cf_blk[(static_cast<size_t>(w) * cf_n_ + k) * cf_ch_];
        d[0] = vs;
        d[1] = va;
        if (cf_ch_ > 2)
          d[2] = vf;
      }
    }
  }

  // Materialize the per-window blocks: curvefit then pole records, each
  // block starting on a 64-byte boundary so it spans as few cache lines as
  // possible.
  constexpr size_t DPL = 64 / sizeof(double); // doubles per cache line
  const int ncf = cf_present_ ? cf_n_ * cf_ch_ : 0;
  win_.resize(n_win);
  vector<size_t> sz(n_win);
  size_t total = 0;
  for (int w = 0; w < n_win; ++w) {
    int np = win_end[w] - win_start_[w];
    win_[w].u_hi = bounds[w + 1];
    win_[w].np = static_cast<uint16_t>(np);
    win_[w].flags = 0;
    if (onev_present_ && onev_idx_ >= win_start_[w] && onev_idx_ < win_end[w])
      win_[w].flags |= WIN_HAS_ONEV;
    sz[w] = (static_cast<size_t>(ncf) + static_cast<size_t>(np) * gs_stride_ +
              DPL - 1) /
            DPL * DPL;
    total += sz[w];
  }
  // std::vector only guarantees 16-byte alignment, so over-allocate one
  // line and start the first block on the next 64-byte boundary
  blk_.assign(total + DPL, 0.0);
  size_t off = ((64 - (reinterpret_cast<uintptr_t>(blk_.data()) & 63)) & 63) /
               sizeof(double);
  for (int w = 0; w < n_win; ++w) {
    win_[w].blk_off = static_cast<uint32_t>(off);
    double* b = &blk_[off];
    if (ncf) {
      const double* c = &cf_blk[static_cast<size_t>(w) * ncf];
      std::copy(c, c + ncf, b);
      b += ncf;
    }
    const double* src =
      &records[static_cast<size_t>(win_start_[w]) * gs_stride_];
    std::copy(src, src + static_cast<size_t>(win_[w].np) * gs_stride_, b);
    off += sz[w];
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
  if (wend.shape(0) != n_win || bounds.shape(0) != n_win + 1 ||
      residues.shape(1) != n_mts || e_on.shape(0) != n_mts ||
      e_off.shape(0) != n_mts || xp.shape(0) != n_win || xr.shape(0) != n_win ||
      xr.shape(1) != n_ext || xr.shape(2) != n_mts) {
    fatal_error(fmt::format(
      "Fast segment arrays have inconsistent shapes for {}.", name_));
  }
  if (n_win > 65535) {
    fatal_error(fmt::format(
      "{} fast WMP windows exceeds the 65535 the window table can address "
      "for {}.",
      n_win, name_));
  }
  fast_bounds_.assign(bounds.begin(), bounds.end());

  // Channels: MT numbers, transport groups, and threshold clamps. The clamps
  // are compared with the window bounds in sqrt(E) so that a threshold
  // sitting exactly on a window edge is recognized as such.
  fast_n_mts_ = n_mts;
  fast_mts_.resize(n_mts);
  fast_mt_on_.resize(n_mts);
  fast_mt_off_.resize(n_mts);
  fast_mt_to_chan_.fill(-1);
  vector<int> kind(n_mts);
  vector<double> u_on(n_mts), u_off(n_mts);
  for (int c = 0; c < n_mts; ++c) {
    fast_mts_[c] = mts[c];
    fast_mt_on_[c] = e_on[c];
    fast_mt_off_[c] = e_off[c];
    kind[c] = channel_kind(fast_mts_[c]);
    u_on[c] = std::sqrt(e_on[c]);
    u_off[c] = std::sqrt(e_off[c]);
    if (mts[c] >= 0 && mts[c] < MAX_MT)
      fast_mt_to_chan_[mts[c]] = c;
  }
  fast_ch_el_ = fast_channel_of_mt(2);
  for (int j = 0; j < WMP_N_DEP; ++j)
    fast_dep_ch_[j] = fast_channel_of_mt(DEPLETION_RX[j]);

  // Per-window curvefit in the E^(k/2-1) basis (optional)
  tensor::Tensor<double> cf; // (n_win, n_cf, n_mts)
  const bool have_cf = object_exists(group, "curvefit");
  if (have_cf) {
    read_dataset(group, "curvefit", cf);
    if (cf.shape(0) != n_win || cf.shape(2) != n_mts) {
      fatal_error(fmt::format(
        "Fast curvefit array has inconsistent shape for {}.", name_));
    }
    fast_cf_n_ = cf.shape(1);
    fw_chan_cf_.assign(static_cast<size_t>(n_win) * fast_cf_n_ * n_mts, 0.0);
    for (int w = 0; w < n_win; ++w)
      for (int k = 0; k < fast_cf_n_; ++k)
        for (int c = 0; c < n_mts; ++c)
          fw_chan_cf_[(static_cast<size_t>(w) * fast_cf_n_ + k) * n_mts + c] =
            cf(w, k, c);
  }

  // Flatten each window's pointer range plus its external poles
  fw_off_.assign(n_win + 1, 0);
  for (int w = 0; w < n_win; ++w) {
    int np = static_cast<int>(wend[w] - wstart[w]) + n_ext;
    if (np > FAST_MAX_POLES) {
      fatal_error(fmt::format("Fast WMP window with {} poles exceeds "
                              "FAST_MAX_POLES = {} for {}.",
        np, FAST_MAX_POLES, name_));
    }
    fw_off_[w + 1] = fw_off_[w] + np;
  }
  const int n_flat = fw_off_[n_win];
  fw_poles_.assign(n_flat, 0.0);
  fw_chan_res_.assign(static_cast<size_t>(n_flat) * n_mts, 0.0);
  fast_partials_.resize(n_win);
  fast_has_part_.assign(n_win, 0);

  // Residues and curvefit collapsed to [scatter, fission, disappearance]
  vector<std::complex<double>> coll(static_cast<size_t>(n_flat) * 3, 0.0);
  vector<double> cf_coll(
    static_cast<size_t>(n_win) * fast_cf_n_ * FAST_CF_SLOTS, 0.0);

  for (int w = 0; w < n_win; ++w) {
    const double u_lo = fast_bounds_[w];
    const double u_hi = fast_bounds_[w + 1];
    const int np = static_cast<int>(wend[w] - wstart[w]);
    const int np_tot = np + n_ext;
    const int base = fw_off_[w];

    for (int k = 0; k < np; ++k)
      fw_poles_[base + k] = poles[wstart[w] + k];
    for (int j = 0; j < n_ext; ++j)
      fw_poles_[base + np + j] = xp(w, j);

    // Per-channel residues, channel-major within the window
    const size_t wbase = static_cast<size_t>(base) * n_mts;
    for (int c = 0; c < n_mts; ++c) {
      std::complex<double>* dst =
        &fw_chan_res_[wbase + static_cast<size_t>(c) * np_tot];
      for (int k = 0; k < np; ++k)
        dst[k] = residues(wstart[w] + k, c);
      for (int j = 0; j < n_ext; ++j)
        dst[np + j] = xr(w, j, c);
    }

    for (int c = 0; c < n_mts; ++c) {
      // Skip channels inactive across the whole window
      if (u_on[c] >= u_hi || u_off[c] <= u_lo)
        continue;

      const int slot = (kind[c] == KIND_FISSION)         ? 1
                       : (kind[c] == KIND_DISAPPEARANCE) ? 2
                                                         : 0;
      if (u_on[c] <= u_lo && u_off[c] >= u_hi) {
        // Fully active: collapse residues and curvefit into the group
        for (int k = 0; k < np_tot; ++k)
          coll[static_cast<size_t>(base + k) * 3 + slot] +=
            fw_chan_res_[wbase + static_cast<size_t>(c) * np_tot + k];
        for (int k = 0; k < fast_cf_n_; ++k)
          cf_coll[(static_cast<size_t>(w) * fast_cf_n_ + k) * FAST_CF_SLOTS +
                  slot] += cf(w, k, c);
      } else {
        // Threshold inside the window: evaluate individually behind the clamp
        FastPartial part;
        part.E_on = e_on[c];
        part.E_off = e_off[c];
        part.kind = kind[c];
        part.ch = c;
        fast_partials_[w].push_back(part);
        fast_has_part_[w] = 1;
      }
    }
  }

  // Which groups this nuclide uses; the block layout omits the others
  fast_has_fis_ = fast_has_dis_ = false;
  for (int i = 0; i < n_flat; ++i) {
    if (coll[static_cast<size_t>(i) * 3 + 1] != 0.0)
      fast_has_fis_ = true;
    if (coll[static_cast<size_t>(i) * 3 + 2] != 0.0)
      fast_has_dis_ = true;
  }
  for (size_t i = 0; i < cf_coll.size(); i += FAST_CF_SLOTS) {
    if (cf_coll[i + 1] != 0.0)
      fast_has_fis_ = true;
    if (cf_coll[i + 2] != 0.0)
      fast_has_dis_ = true;
  }
  const int n_groups =
    1 + static_cast<int>(fast_has_fis_) + static_cast<int>(fast_has_dis_);
  fast_soa_stride_ = 2 + 2 * n_groups;

  // Per-window blocks: curvefit first, then the pole coefficients
  const int cfw = FAST_CF_SLOTS * fast_cf_n_;
  fast_blk_off_.assign(n_win + 1, 0);
  size_t off = 0;
  for (int w = 0; w < n_win; ++w) {
    fast_blk_off_[w] = off;
    off += static_cast<size_t>(cfw) + static_cast<size_t>(fast_soa_stride_) *
                                        (fw_off_[w + 1] - fw_off_[w]);
  }
  fast_blk_off_[n_win] = off;
  fw_soa_.assign(off, 0.0);
  for (int w = 0; w < n_win; ++w) {
    const int k0 = fw_off_[w];
    const int np = fw_off_[w + 1] - k0;
    double* d = &fw_soa_[fast_blk_off_[w]];
    std::copy(&cf_coll[static_cast<size_t>(w) * cfw],
      &cf_coll[static_cast<size_t>(w) * cfw] + cfw, d);
    double* B = d + cfw;
    for (int k = 0; k < np; ++k) {
      const double a = fw_poles_[k0 + k].real();
      const double b = fw_poles_[k0 + k].imag();
      B[k] = a;
      B[np + k] = b * b;
      int slot = 0;
      for (int g = 0; g < 3; ++g) {
        if ((g == 1 && !fast_has_fis_) || (g == 2 && !fast_has_dis_))
          continue;
        const std::complex<double> r =
          coll[static_cast<size_t>(k0 + k) * 3 + g];
        B[(2 + 2 * slot) * np + k] = r.real();
        B[(3 + 2 * slot) * np + k] = -(r.real() * a + r.imag() * b);
        ++slot;
      }
    }
  }

  // Exponent-keyed window table (see wmp.h)
  fast_key_lo_ = fast_key(fast_bounds_.front());
  const int64_t key_hi = fast_key(fast_bounds_.back());
  // One bucket per distinct key plus a trailing bucket so the clamp in
  // fast_window() lands on a real entry at or above the last bound
  const size_t n_bucket = static_cast<size_t>(key_hi - fast_key_lo_) + 2;
  fast_bidx_.resize(n_bucket);
  int w = 0;
  for (size_t i = 0; i < n_bucket; ++i) {
    uint64_t bits =
      static_cast<uint64_t>(fast_key_lo_ + static_cast<int64_t>(i))
      << FAST_KEY_SHIFT;
    double u_bucket;
    std::memcpy(&u_bucket, &bits, sizeof(u_bucket));
    while (w < n_win - 1 && fast_bounds_[w + 1] <= u_bucket)
      ++w;
    fast_bidx_[i] = static_cast<uint16_t>(w);
  }
}

//========================================================================
// Fast segment evaluation
//========================================================================

//! Fill the pole reciprocals 1/(u - p) of window w at sqrt(E)
static void fast_reciprocals(const vector<std::complex<double>>& poles,
  const vector<int>& fw_off, int w, double sqrtE,
  WindowedMultipole::FastChanCtx& ctx)
{
  ctx.w = w;
  ctx.k0 = fw_off[w];
  ctx.np = fw_off[w + 1] - ctx.k0;
  for (int k = 0; k < ctx.np; ++k) {
    const std::complex<double>& p = poles[ctx.k0 + k];
    double dr = sqrtE - p.real();
    double di = -p.imag();
    double inv = 1.0 / (dr * dr + di * di);
    ctx.pr[k] = dr * inv;
    ctx.pi[k] = -di * inv;
  }
}

void WindowedMultipole::fast_chan_begin(double E, FastChanCtx& ctx) const
{
  double sqrtE = std::sqrt(E);
  fast_reciprocals(fw_poles_, fw_off_, fast_window(sqrtE), sqrtE, ctx);
}

double WindowedMultipole::fast_chan_xs(
  const FastChanCtx& ctx, double E, int c) const
{
  if (E < fast_mt_on_[c] || E > fast_mt_off_[c])
    return 0.0;

  // Pole sum, Re[r / (u - p)] = Re(r) Re(1/d) - Im(r) Im(1/d)
  const std::complex<double>* r =
    &fw_chan_res_[static_cast<size_t>(ctx.k0) * fast_n_mts_ +
                  static_cast<size_t>(c) * ctx.np];
  double s = 0.0;
  for (int k = 0; k < ctx.np; ++k)
    s += r[k].real() * ctx.pr[k] - r[k].imag() * ctx.pi[k];

  // This channel's curvefit background
  if (fast_cf_n_ > 0) {
    const double* cf =
      &fw_chan_cf_[static_cast<size_t>(ctx.w) * fast_cf_n_ * fast_n_mts_ + c];
    const double sqrtE = std::sqrt(E);
    double fac = 1.0 / E;
    for (int k = 0; k < fast_cf_n_; ++k, cf += fast_n_mts_) {
      s += *cf * fac;
      fac *= sqrtE;
    }
  }
  return (s > 0.0) ? s : 0.0;
}

//! Sweep over a window's poles in the fast segment, specialized at compile
//! time on the groups present so the slot offsets are constants
template<bool FIS, bool DIS>
static inline void fast_sweep(
  const double* B, int np, double u, double& sct, double& fis, double& dis)
{
  constexpr int I_FIS = 1;
  constexpr int I_DIS = I_FIS + (FIS ? 1 : 0);
  const double* al = B;
  const double* be2 = B + np;
#pragma omp simd reduction(+ : sct, fis, dis)
  for (int k = 0; k < np; ++k) {
    double du = u - al[k];
    double inv = 1.0 / (du * du + be2[k]);
    sct += (B[2 * np + k] * u + B[3 * np + k]) * inv;
    if (FIS)
      fis +=
        (B[(2 + 2 * I_FIS) * np + k] * u + B[(3 + 2 * I_FIS) * np + k]) * inv;
    if (DIS)
      dis +=
        (B[(2 + 2 * I_DIS) * np + k] * u + B[(3 + 2 * I_DIS) * np + k]) * inv;
  }
}

std::tuple<double, double, double> WindowedMultipole::evaluate_fast(
  double E, double* dep) const
{
  const double sqrtE = std::sqrt(E);
  const int w = fast_window(sqrtE);
  const int np = fw_off_[w + 1] - fw_off_[w];

  // Per-channel context, built only if a partial channel or a depletion
  // reaction needs it
  FastChanCtx ctx;
  bool have_ctx = false;

  // Depletion reactions from the per-channel residues
  if (dep) {
    fast_reciprocals(fw_poles_, fw_off_, w, sqrtE, ctx);
    have_ctx = true;
    for (int j = 0; j < WMP_N_DEP; ++j) {
      const int c = fast_dep_ch_[j];
      if (c >= 0)
        dep[j] += fast_chan_xs(ctx, E, c);
    }
  }

  // Collapsed pole sum
  const double* blk = &fw_soa_[fast_blk_off_[w]];
  const double* B = blk + FAST_CF_SLOTS * fast_cf_n_;
  double sct = 0.0, fis = 0.0, dis = 0.0;
  switch ((fast_has_fis_ ? 2 : 0) | (fast_has_dis_ ? 1 : 0)) {
  case 0:
    fast_sweep<false, false>(B, np, sqrtE, sct, fis, dis);
    break;
  case 1:
    fast_sweep<false, true>(B, np, sqrtE, sct, fis, dis);
    break;
  case 2:
    fast_sweep<true, false>(B, np, sqrtE, sct, fis, dis);
    break;
  default:
    fast_sweep<true, true>(B, np, sqrtE, sct, fis, dis);
    break;
  }

  // Channels with a threshold inside this window
  if (fast_has_part_[w]) {
    for (const auto& part : fast_partials_[w]) {
      if (E < part.E_on || E > part.E_off)
        continue;
      if (!have_ctx) {
        fast_reciprocals(fw_poles_, fw_off_, w, sqrtE, ctx);
        have_ctx = true;
      }
      const double s = fast_chan_xs(ctx, E, part.ch);
      switch (part.kind) {
      case KIND_FISSION:
        fis += s;
        break;
      case KIND_DISAPPEARANCE:
        dis += s;
        break;
      default:
        sct += s;
        break;
      }
    }
  }

  // Curvefit background in the E^(k/2-1) basis
  if (fast_cf_n_ > 0) {
    const double* c = blk;
    double fac = 1.0 / E;
    for (int k = 0; k < fast_cf_n_; ++k, c += FAST_CF_SLOTS) {
      sct += c[0] * fac;
      if (fast_has_fis_)
        fis += c[1] * fac;
      if (fast_has_dis_)
        dis += c[2] * fac;
      fac *= sqrtE;
    }
  }

  // Guard against fits dipping below zero. Scattering is floored rather than
  // clamped so the total stays positive.
  sct = (sct > 0.0) ? sct : 1.0e-5;
  fis = (fis > 0.0) ? fis : 0.0;
  dis = (dis > 0.0) ? dis : 0.0;

  return {sct + fis + dis, dis + fis, fis};
}

double WindowedMultipole::evaluate_fast_elastic(double E) const
{
  if (fast_ch_el_ < 0)
    return 1.0e-5;
  FastChanCtx ctx;
  fast_chan_begin(E, ctx);
  double s = fast_chan_xs(ctx, E, fast_ch_el_);
  // Same floor as the scattering group so channel selection never divides
  // by zero
  return (s > 0.0) ? s : 1.0e-5;
}

//========================================================================
// Resolved segment evaluation
//========================================================================

//! Rational approximation of the Faddeeva function w(z) for Im(z) > 0.
//!
//! A sixth-order shifted rational form, w(z) ~ N(z + 1.5i) / D(z + 1.5i), with
//! D even and the leading numerator coefficient pinned to i/sqrt(pi) so the
//! large-|z| tail is reproduced without an asymptotic branch. The
//! construction follows B. Forget, J. Yu, and G. Ridley, "Direct Doppler
//! broadening of windowed multipole cross sections with a rational
//! approximation of the Faddeeva function," Proc. PHYSOR 2022. Fitted over
//! |Re z| <= 600, 1e-4 <= Im z <= 1200 with a maximum relative error of
//! 1.6e-5. Written out in real arithmetic so the pole sweep vectorizes.
static inline void faddeeva_rational(
  double zr, double zi, double& wr, double& wi)
{
  zi += 1.5;                                   // shift
  double nr = 0.0, ni = 0.5641895835477563, t; // leading coefficient
  t = nr * zr - ni * zi - 0.8463988774687549;
  ni = nr * zi + ni * zr;
  nr = t;
  t = nr * zr - ni * zi;
  ni = nr * zi + ni * zr - 4.262789761626753;
  nr = t;
  t = nr * zr - ni * zi + 5.57957343594273;
  ni = nr * zi + ni * zr;
  nr = t;
  t = nr * zr - ni * zi;
  ni = nr * zi + ni * zr + 8.876835541419577;
  nr = t;
  t = nr * zr - ni * zi - 8.2804011586059;
  ni = nr * zi + ni * zr;
  nr = t;
  double zzr = zr * zr - zi * zi, zzi = 2.0 * zr * zi;
  double dnr = 1.0, dni = 0.0;
  t = dnr * zzr - dni * zzi - 5.810043673216703;
  dni = dnr * zzi + dni * zzr;
  dnr = t;
  t = dnr * zzr - dni * zzi + 6.844707994387722;
  dni = dnr * zzi + dni * zzr;
  dnr = t;
  t = dnr * zzr - dni * zzi - 0.901452218111202;
  dni = dnr * zzi + dni * zzr;
  dnr = t;
  double dm = 1.0 / (dnr * dnr + dni * dni);
  wr = (nr * dnr + ni * dni) * dm;
  wi = (ni * dnr - nr * dni) * dm;
}

//! Broadened sweep over a window's pole records, specialized at compile time
//! on the record stride. With SAVE, each pole's broadened kernel (br, bi) is
//! stored so a per-MT pass can reuse it without evaluating w(z) again.
template<int STRIDE, bool FISS, bool SAVE = false>
static inline void pole_sweep(const double* pol, int np, double sqrtE,
  double invE, double inv_xi, double& sig_s, double& sig_a, double& sig_f,
  double* brbi = nullptr)
{
  const double sq = SQRT_PI * inv_xi;
#pragma omp simd reduction(+ : sig_s, sig_a, sig_f)
  for (int k = 0; k < np; ++k) {
    const double* r = pol + static_cast<size_t>(k) * STRIDE;
    double pr = r[0], pi = r[1];
    double p2r = pr * pr - pi * pi, p2i = 2.0 * pr * pi;
    double zr = (pr - sqrtE) * inv_xi;
    double zi = pi * inv_xi;
    double wr, wi;
    faddeeva_rational(zr, zi, wr, wi);
    double wgr = -sq * wi, wgi = sq * wr; // i sqrt(pi)/xi w(z)
    double hr = p2r * wgr - p2i * wgi, hi = p2r * wgi + p2i * wgr;
    double br = (sqrtE + pr + hr) * invE;
    double bi = (pi + hi) * invE;
    if (SAVE) {
      brbi[2 * k] = br;
      brbi[2 * k + 1] = bi;
    }
    sig_s += r[2] * br - r[3] * bi;
    sig_a += r[4] * br - r[5] * bi;
    if (FISS)
      sig_f += r[6] * br - r[7] * bi;
  }
}

std::tuple<double, double, double> WindowedMultipole::evaluate(
  double E, double sqrtkT, double* dep) const
{
  using namespace std::complex_literals;

  // ==========================================================================
  // Bookkeeping

  // Define some frequently used variables.
  double sqrtE = std::sqrt(E);
  double invE = 1.0 / E;

  if (global_format_) {
    // ========================================================================
    // Low-energy segment: curvefit only

    if (E < lowe_E_max_) {
      array<double, MAX_POLY_COEFFICIENTS> fac;
      if (sqrtkT > 0.0) {
        broaden_wmp_polynomials(E, sqrt_awr_ / sqrtkT, lowe_n_, fac.data());
      } else {
        double t = invE;
        for (int k = 0; k < lowe_n_; ++k) {
          fac[k] = t;
          t *= sqrtE;
        }
      }
      double sig_s = 0.0, sig_a = 0.0, sig_f = 0.0;
      for (int k = 0; k < lowe_n_; ++k) {
        sig_s += lowe_cf_[3 * k + 0] * fac[k];
        sig_a += lowe_cf_[3 * k + 1] * fac[k];
        sig_f += lowe_cf_[3 * k + 2] * fac[k];
      }
      if (dep) {
        for (int k = 0; k < lowe_n_; ++k) {
          const double* c = &lowe_dep_cf_[static_cast<size_t>(k) * WMP_N_DEP];
          for (int j = 0; j < WMP_N_DEP; ++j)
            dep[j] += c[j] * fac[k];
        }
        for (int j = 0; j < WMP_N_DEP; ++j)
          dep[j] = std::max(dep[j], 0.0);
      }
      sig_s = (sig_s > 0.0) ? sig_s : 1.0e-5;
      sig_a = (sig_a > 0.0) ? sig_a : 0.0;
      sig_f = (sig_f > 0.0) ? sig_f : 0.0;
      return std::make_tuple(sig_s, sig_a, sig_f);
    }

    // ========================================================================
    // Resolved segment

    const int i_window = resolved_window(sqrtE);
    const WinRec& wrec = win_[i_window];
    const int ncf = cf_present_ ? cf_n_ * cf_ch_ : 0;
    const double* blk = &blk_[wrec.blk_off];
    const double* pol = blk + ncf;
    const int np = wrec.np;
    const int ps = win_start_[i_window];

    // xi = sqrt(kT/AWR) in sqrt(eV)
    double xi = (sqrtkT > 0.0) ? sqrtkT / sqrt_awr_ : 0.0;
    double inv_xi = (xi > 0.0) ? 1.0 / xi : 0.0;

    // Per-MT depletion channels are only needed when (n,gamma) cannot be
    // taken as absorption - fission. The broadened kernel of each pole is
    // saved by the sweep and reused here.
    const bool do_dep = (dep != nullptr) && dep_needs_split_;
    array<double, 2 * MAX_WINDOW_POLES> brbi;
    auto accum_dep = [&]() {
      for (int i = 0; i < np; ++i) {
        if (pol[static_cast<size_t>(i) * gs_stride_] == 0.0)
          continue; // 1/v pole, added separately below
        const double br = brbi[2 * i], bi = brbi[2 * i + 1];
        for (int j = 0; j < WMP_N_DEP; ++j) {
          const int c = dep_ch_[j];
          if (c < 0)
            continue;
          const std::complex<double> r = res_mt_(ps + i, c);
          dep[j] += r.real() * br - r.imag() * bi;
        }
      }
    };

    double sig_s = 0.0, sig_a = 0.0, sig_f = 0.0;
    if (xi <= 0.0) {
      // 0 K: plain pole sum, Re[r / (u - p)]
      const double* rec = pol;
      for (int i = 0; i < np; ++i, rec += gs_stride_) {
        double dr = sqrtE - rec[0], di = -rec[1];
        double minv = 1.0 / (dr * dr + di * di);
        double br = dr * minv, bi = -di * minv;
        if (do_dep) {
          brbi[2 * i] = br;
          brbi[2 * i + 1] = bi;
        }
        sig_s += rec[2] * br - rec[3] * bi;
        sig_a += rec[4] * br - rec[5] * bi;
        if (fissionable_)
          sig_f += rec[6] * br - rec[7] * bi;
      }
      if (do_dep)
        accum_dep();
    } else if (do_dep) {
      if (fissionable_)
        pole_sweep<8, true, true>(
          pol, np, sqrtE, invE, inv_xi, sig_s, sig_a, sig_f, brbi.data());
      else
        pole_sweep<6, false, true>(
          pol, np, sqrtE, invE, inv_xi, sig_s, sig_a, sig_f, brbi.data());
      accum_dep();
    } else if (fissionable_) {
      pole_sweep<8, true>(pol, np, sqrtE, invE, inv_xi, sig_s, sig_a, sig_f);
    } else {
      pole_sweep<6, false>(pol, np, sqrtE, invE, inv_xi, sig_s, sig_a, sig_f);
    }

    // 1/v pole: broadening-invariant, so its 0 K value is exact at any T
    if (wrec.flags & WIN_HAS_ONEV) {
      double dr = sqrtE - onev_p_.real(), di = -onev_p_.imag();
      double minv = 1.0 / (dr * dr + di * di);
      double br = dr * minv, bi = -di * minv;
      sig_s += onev_rs_.real() * br - onev_rs_.imag() * bi;
      sig_a += onev_ra_.real() * br - onev_ra_.imag() * bi;
      sig_f += onev_rf_.real() * br - onev_rf_.imag() * bi;
      if (do_dep) {
        for (int j = 0; j < WMP_N_DEP; ++j) {
          const int c = dep_ch_[j];
          if (c < 0)
            continue;
          const std::complex<double> r = res_mt_(onev_idx_, c);
          dep[j] += r.real() * br - r.imag() * bi;
        }
      }
    }

    // Per-window curvefit, broadened exactly in the E^(k/2-1) basis
    if (cf_present_) {
      array<double, MAX_POLY_COEFFICIENTS> fac;
      if (sqrtkT > 0.0) {
        broaden_wmp_polynomials(E, sqrt_awr_ / sqrtkT, cf_n_, fac.data());
      } else {
        double t = invE;
        for (int k = 0; k < cf_n_; ++k) {
          fac[k] = t;
          t *= sqrtE;
        }
      }
      if (do_dep) {
        for (int k = 0; k < cf_n_; ++k) {
          for (int j = 0; j < WMP_N_DEP; ++j) {
            const int c = dep_ch_[j];
            if (c >= 0)
              dep[j] += cf_mt_(i_window, k, c) * fac[k];
          }
        }
      }
      const double* c = blk;
      if (fissionable_) {
        for (int k = 0; k < cf_n_; ++k, c += cf_ch_) {
          sig_s += c[0] * fac[k];
          sig_a += c[1] * fac[k];
          sig_f += c[2] * fac[k];
        }
      } else {
        for (int k = 0; k < cf_n_; ++k, c += cf_ch_) {
          sig_s += c[0] * fac[k];
          sig_a += c[1] * fac[k];
        }
      }
    }

    // Guard against fits dipping below zero. Scattering is floored rather
    // than clamped so the total stays positive. Note that (x > 0) ? x : 0
    // also traps NaN, unlike std::max.
    sig_s = (sig_s > 0.0) ? sig_s : 1.0e-5;
    sig_a = (sig_a > 0.0) ? sig_a : 0.0;
    sig_f = (sig_f > 0.0) ? sig_f : 0.0;
    if (dep) {
      if (do_dep) {
        for (int j = 0; j < WMP_N_DEP; ++j)
          dep[j] = std::max(dep[j], 0.0);
      } else {
        // MT102 is the only disappearance channel, so (n,gamma) is exact
        dep[0] = std::max(sig_a - sig_f, 0.0);
        for (int j = 1; j < WMP_N_DEP; ++j)
          dep[j] = 0.0;
      }
    }
    return std::make_tuple(sig_s, sig_a, sig_f);
  }

  // ==========================================================================
  // Version 1 format

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

  // Version 1 libraries carry no per-MT data; (n,gamma) is the only
  // depletion reaction available
  if (dep) {
    dep[0] = std::max(sig_a - sig_f, 0.0);
    for (int j = 1; j < WMP_N_DEP; ++j)
      dep[j] = 0.0;
  }

  return std::make_tuple(sig_s, sig_a, sig_f);
}

std::tuple<double, double, double> WindowedMultipole::evaluate_deriv(
  double E, double sqrtkT) const
{
  using namespace std::complex_literals;

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

  // Initialize the ouptut cross sections.
  double sig_s = 0.0;
  double sig_a = 0.0;
  double sig_f = 0.0;

  // Every broadened quantity depends on temperature through
  // dopp = sqrt(AWR / kT), so d/dT = (d dopp/dT) d/d dopp
  double dopp = sqrt_awr_ / sqrtkT;
  double ddopp_dT = -0.5 * dopp / T;

  if (global_format_) {
    // The fast segment is not broadened, so its derivative vanishes
    if (E > E_max_)
      return std::make_tuple(0.0, 0.0, 0.0);

    // Derivative of the broadened curvefit basis
    array<double, MAX_POLY_COEFFICIENTS> fac, dfac;
    auto add_curvefit = [&](const double* c, int n, int stride) {
      broaden_wmp_polynomials_deriv(E, dopp, n, fac.data(), dfac.data());
      for (int k = 0; k < n; ++k, c += stride) {
        sig_s += c[0] * dfac[k];
        sig_a += c[1] * dfac[k];
        if (stride > 2)
          sig_f += c[2] * dfac[k];
      }
    };

    if (E < lowe_E_max_) {
      add_curvefit(lowe_cf_.data(), lowe_n_, 3);
    } else {
      const int i_window = resolved_window(sqrtE);
      const WinRec& wrec = win_[i_window];
      const int ncf = cf_present_ ? cf_n_ * cf_ch_ : 0;
      const double* blk = &blk_[wrec.blk_off];

      // Poles: d/d dopp of (i sqrt(pi) dopp w(z)) with z = (p - u) dopp is
      // i sqrt(pi) (w + z w') = -i sqrt(pi) w'' / 2. The 1/v pole is
      // broadening-invariant and carries zero residues in its record.
      const double* rec = blk + ncf;
      for (int i = 0; i < wrec.np; ++i, rec += gs_stride_) {
        std::complex<double> p {rec[0], rec[1]};
        if (p.real() == 0.0)
          continue;
        std::complex<double> z = (p - sqrtE) * dopp;
        std::complex<double> val =
          -0.5i * SQRT_PI * p * p * w_derivative(z, 2) * invE;
        sig_s += (std::complex<double> {rec[2], rec[3]} * val).real();
        sig_a += (std::complex<double> {rec[4], rec[5]} * val).real();
        if (fissionable_)
          sig_f += (std::complex<double> {rec[6], rec[7]} * val).real();
      }

      if (cf_present_)
        add_curvefit(blk, cf_n_, cf_ch_);
    }

    sig_s *= ddopp_dT;
    sig_a *= ddopp_dT;
    sig_f *= ddopp_dT;
    return std::make_tuple(sig_s, sig_a, sig_f);
  }

  // ==========================================================================
  // Version 1 format

  // Locate us
  int i_window = (sqrtE - std::sqrt(E_min_)) * inv_spacing_;
  const auto& window {window_info_[i_window]};

  // TODO Polynomials: Some of the curvefit polynomials Doppler broaden so
  // rigorously we should be computing the derivative of those.  But in
  // practice, those derivatives are only large at very low energy and they
  // have no effect on reactor calculations.

  // ==========================================================================
  // Add the contribution from the poles in this window.

  for (int i_pole = window.index_start; i_pole <= window.index_end; ++i_pole) {
    std::complex<double> z = (sqrtE - data_(i_pole, MP_EA)) * dopp;
    std::complex<double> w_val = -invE * SQRT_PI * 0.5 * w_derivative(z, 2);
    sig_s += (data_(i_pole, MP_RS) * w_val).real();
    sig_a += (data_(i_pole, MP_RA) * w_val).real();
    if (fissionable_) {
      sig_f += (data_(i_pole, MP_RF) * w_val).real();
    }
  }
  sig_s *= ddopp_dT;
  sig_a *= ddopp_dT;
  sig_f *= ddopp_dT;

  return std::make_tuple(sig_s, sig_a, sig_f);
}

//========================================================================
// Non-member functions
//========================================================================

int check_wmp_version(hid_t file)
{
  if (attribute_exists(file, "version")) {
    array<int, 2> version;
    read_attribute(file, "version", version);
    if (version[0] != WMP_VERSION[0] &&
        version[0] != WMP_VERSION_MAJOR_LEGACY) {
      fatal_error(fmt::format(
        "WMP data format uses version {}.{} whereas your installation of "
        "OpenMC expects version {}.x or {}.x data.",
        version[0], version[1], WMP_VERSION[0], WMP_VERSION_MAJOR_LEGACY));
    }
    return version[0];
  } else {
    fatal_error(fmt::format("WMP data does not indicate a version. Your "
                            "installation of OpenMC expects version {}.x data.",
      WMP_VERSION[0]));
  }
  return 0;
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
  int version_major = check_wmp_version(file);

  // Read nuclide data from HDF5
  hid_t group = open_group(file, nuc->name_.c_str());
  nuc->multipole_ = make_unique<WindowedMultipole>(group, version_major);
  close_group(group);
  file_close(file);

  // Mirror the energy bounds into the nuclide (see nuclide.h)
  const auto& mp = *nuc->multipole_;
  nuc->mp_present_ = true;
  nuc->mp_fast_present_ = mp.fast_present_;
  nuc->mp_E_min_ =
    mp.lowe_present_ ? std::min(mp.E_min_, mp.lowe_E_min_) : mp.E_min_;
  nuc->mp_E_max_ = mp.E_max_;
  nuc->mp_fast_E_max_ = mp.fast_E_max_;

  if (mp.fast_present_)
    nuc->init_nu_fast_map(mp.E_max_, mp.fast_E_max_);
}

//! erf(beta) and exp(-beta^2) as one shared rational approximation, valid for
//! 0 <= beta <= 6. With t = beta^2,
//!   erf(beta) = beta P1(t) / Q(t),   exp(-beta^2) = P2(t) / Q(t),
//! so the pair costs a single division and three independent Horner chains.
//! exp_m_beta2 is accurate only in the combination broaden_wmp_polynomials
//! uses it in, exp(-beta^2)/(beta sqrt(pi)) added to erf(beta)/E * (...),
//! whose sensitivity to it falls off as fast as the term itself; it is not a
//! general-purpose exp(-beta^2). Verified against std::erf and std::exp to
//! better than 2.5e-6 in the broadened cross section over the fitted range.
static inline void doppler_erf_exp(
  double beta, double& erf_beta, double& exp_m_beta2)
{
  const double t = beta * beta;
  double p1 = 7.474374146806876e-08;
  p1 = p1 * t + 2.693973668033131e-05;
  p1 = p1 * t + 0.0009953139016093899;
  p1 = p1 * t + 0.01127708612050287;
  p1 = p1 * t + 0.08205945692613878;
  p1 = p1 * t + 0.3212541350292576;
  p1 = p1 * t + 1.128379117742098; // -> 2/sqrt(pi) as t -> 0
  double p2 = 5.216660591492671e-08;
  p2 = p2 * t + -5.551042589775674e-06;
  p2 = p2 * t + 0.0002352498383684777;
  p2 = p2 * t + -0.005117030892861882;
  p2 = p2 * t + 0.06093163280748515;
  p2 = p2 * t + -0.3820119593411849;
  p2 = p2 * t + 1.000000102493979; // -> 1 as t -> 0
  double q = 2.142899192132085e-06;
  q = q * t + 0.0001993803551938137;
  q = q * t + 0.003696070390878379;
  q = q * t + 0.03146603019871784;
  q = q * t + 0.1788100116816972;
  q = q * t + 0.618022547204684;
  q = q * t + 1.0;
  const double inv_q = 1.0 / q;
  erf_beta = beta * p1 * inv_q;
  exp_m_beta2 = p2 * inv_q;
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
    doppler_erf_exp(beta, erf_beta, exp_m_beta2);
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

void broaden_wmp_polynomials_deriv(
  double E, double dopp, int n, double factors[], double dfactors[])
{
  // Same construction as broaden_wmp_polynomials, differentiated with respect
  // to dopp. Uses the exact erf and exp since this is not a hot path.
  double sqrtE = std::sqrt(E);
  double beta = sqrtE * dopp;
  double h = 0.5 / (dopp * dopp); // half_inv_dopp2
  double dh = -2.0 * h / dopp;
  double q = h * h; // quarter_inv_dopp4
  double dq = 2.0 * h * dh;

  double erf_beta = std::erf(beta);
  double exp_m_beta2 = std::exp(-beta * beta);

  // d(beta)/d(dopp) = sqrtE
  factors[0] = erf_beta / E;
  dfactors[0] = 2.0 / SQRT_PI * exp_m_beta2 * sqrtE / E;
  factors[1] = 1. / sqrtE;
  dfactors[1] = 0.0;
  double g = exp_m_beta2 / (beta * SQRT_PI);
  double dg = -exp_m_beta2 / SQRT_PI * (2.0 + 1.0 / (beta * beta)) * sqrtE;
  factors[2] = factors[0] * (h + E) + g;
  dfactors[2] = dfactors[0] * (h + E) + factors[0] * dh + dg;
  if (n > 3) {
    factors[3] = factors[1] * (E + 3.0 * h);
    dfactors[3] = factors[1] * 3.0 * dh;
  }

  for (int i = 1; i < n - 3; i++) {
    double ip1_dbl = i + 1;
    double a = (ip1_dbl - 1.) * ip1_dbl;
    double b = 1. + 2. * ip1_dbl;
    factors[i + 3] = -factors[i - 1] * a * q + factors[i + 1] * (E + b * h);
    dfactors[i + 3] = -dfactors[i - 1] * a * q - factors[i - 1] * a * dq +
                      dfactors[i + 1] * (E + b * h) + factors[i + 1] * b * dh;
  }
}

} // namespace openmc
