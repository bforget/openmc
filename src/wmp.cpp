#include "openmc/wmp.h"

#include "openmc/constants.h"
#include "openmc/cross_sections.h"
#include "openmc/error.h" // for writing messages
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/nuclide.h"
#include "openmc/wmp_profile.h"

#include <fmt/core.h>

#include <algorithm> // for min
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace openmc {

namespace {

// 12-point Gauss-Legendre nodes/weights on [-1, 1] (the base panel of the
// low-E correction, used on the hot path with no narrow in-range poles;
// precomputed so leggauss/Newton is never run there).
constexpr int N_GL12 {12};
const double GL12_X[N_GL12] = {-9.81560634246719244e-01,
  -9.04117256370474798e-01, -7.69902674194304693e-01,
  -5.87317954286617483e-01, -3.67831498998180184e-01,
  -1.25233408511468913e-01, 1.25233408511468913e-01, 3.67831498998180184e-01,
  5.87317954286617483e-01, 7.69902674194304693e-01, 9.04117256370474798e-01,
  9.81560634246719244e-01};
const double GL12_W[N_GL12] = {4.71753363865120220e-02, 1.06939325995318885e-01,
  1.60078328543346110e-01, 2.03167426723065647e-01, 2.33492536538354639e-01,
  2.49147045813402690e-01, 2.49147045813402690e-01, 2.33492536538354639e-01,
  2.03167426723065647e-01, 1.60078328543346110e-01, 1.06939325995318885e-01,
  4.71753363865120220e-02};

//! Gauss-Legendre nodes/weights on [-1, 1] via Newton iteration on P_n.
void leggauss(int n, std::vector<double>& x, std::vector<double>& w)
{
  x.resize(n);
  w.resize(n);
  for (int i = 0; i < (n + 1) / 2; ++i) {
    double z = std::cos(PI * (i + 0.75) / (n + 0.5));
    double z1, pp;
    do {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 0; j < n; ++j) {
        double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / pp;
    } while (std::abs(z - z1) > 1e-14);
    x[i] = -z;
    x[n - 1 - i] = z;
    double ww = 2.0 / ((1.0 - z * z) * pp * pp);
    w[i] = ww;
    w[n - 1 - i] = ww;
  }
}

//! Composite Gauss-Legendre abscissae/weights on (0, hi) with refined pockets
//! (16-64 pts) around any narrow in-range pole and a 12-pt base elsewhere.
//! Ports evalxs.lowe_nodes exactly. RARE path only: reached solely when a
//! window has a narrow in-range background pole (e.g. Na-23); the common case
//! (no narrow pole -> a single GL-12 panel) is handled inline in
//! lowe_correction without touching this function.
void lowe_nodes(double hi, const std::complex<double>* Pc, int nP,
  std::vector<double>& ug, std::vector<double>& wg)
{
  const int n_base = 12;
  std::vector<std::array<double, 3>> cuts; // {lo, hi, beta}
  for (int ip = 0; ip < nP; ++ip) {
    double a = Pc[ip].real(), b = Pc[ip].imag();
    if (b < hi / 4.0 && -8.0 * b < a && a < hi + 8.0 * b) {
      double lo = std::max(a - 8.0 * b, 0.0), hp = std::min(a + 8.0 * b, hi);
      if (hp > lo)
        cuts.push_back({lo, hp, b});
    }
  }
  std::sort(cuts.begin(), cuts.end(),
    [](const std::array<double, 3>& A, const std::array<double, 3>& B) {
      return A[0] < B[0];
    });
  std::vector<std::array<double, 3>> merged;
  for (const auto& c : cuts) {
    if (!merged.empty() && c[0] <= merged.back()[1]) {
      merged.back()[1] = std::max(merged.back()[1], c[1]);
      merged.back()[2] = std::min(merged.back()[2], c[2]);
    } else {
      merged.push_back(c);
    }
  }
  struct Seg {
    double lo, hi;
    int n;
  };
  std::vector<Seg> segs;
  double prev = 0.0;
  for (const auto& m : merged) {
    if (m[0] > prev)
      segs.push_back({prev, m[0], n_base});
    double nf = std::min(std::max(4.0 * (m[1] - m[0]) / m[2], 16.0), 64.0);
    segs.push_back({m[0], m[1], static_cast<int>(nf)});
    prev = m[1];
  }
  if (prev < hi)
    segs.push_back({prev, hi, n_base});
  ug.clear();
  wg.clear();
  std::vector<double> X, W;
  for (const auto& s : segs) {
    leggauss(s.n, X, W);
    double c0 = 0.5 * (s.hi - s.lo), c1 = 0.5 * (s.hi + s.lo);
    for (int i = 0; i < s.n; ++i) {
      ug.push_back(c0 * X[i] + c1);
      wg.push_back(c0 * W[i]);
    }
  }
}

} // namespace

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

    // Build the branch-free SoA pool arrays (see wmp.h). The 1/v pole
    // (Re p == 0) is broadening-invariant and must NOT go through zpf8h:
    // zero its SoA residues so the vector loop skips it, and stash its data
    // + owning window for a separate exact-0K add.
    gs_pr_.resize(n_pool); gs_pi_.resize(n_pool);
    gs_sr_.resize(n_pool); gs_si_.resize(n_pool);
    gs_ar_.resize(n_pool); gs_ai_.resize(n_pool);
    gs_fr_.resize(n_pool); gs_fi_.resize(n_pool);
    for (int i = 0; i < n_pool; ++i) {
      std::complex<double> p = gdata_(i, MP_EA);
      gs_pr_[i] = p.real(); gs_pi_[i] = p.imag();
      bool is1v = (p.real() == 0.0);
      if (is1v) {
        onev_present_ = true;
        onev_p_ = p;
        onev_rs_ = gdata_(i, MP_RS);
        onev_ra_ = gdata_(i, MP_RA);
        onev_rf_ = gdata_(i, MP_RF);
        gs_sr_[i] = gs_si_[i] = gs_ar_[i] = gs_ai_[i] = 0.0;
        gs_fr_[i] = gs_fi_[i] = 0.0;
      } else {
        gs_sr_[i] = gdata_(i, MP_RS).real(); gs_si_[i] = gdata_(i, MP_RS).imag();
        gs_ar_[i] = gdata_(i, MP_RA).real(); gs_ai_[i] = gdata_(i, MP_RA).imag();
        gs_fr_[i] = gdata_(i, MP_RF).real(); gs_fi_[i] = gdata_(i, MP_RF).imag();
      }
    }
    if (onev_present_) {
      for (int w = 0; w < n_win; ++w) {
        for (int i = win_start_[w]; i < win_end_[w]; ++i) {
          if (gs_pr_[i] == 0.0) { onev_window_ = w; break; }
        }
        if (onev_window_ >= 0) break;
      }
    }

    // Optional low-E polynomial segment (see wmp.h). curvefit shape is
    // [n_poly, order+1, n_ch]; collapse the per-MT channels to the transport
    // triplet exactly as the residues are collapsed above.
    if (object_exists(group, "poly_curvefit")) {
      poly_present_ = true;
      read_dataset(group, "poly_E_max", poly_E_max_);
      read_dataset(group, "poly_sqrtE_bounds", poly_sqrtE_bounds_);
      if (attribute_exists(group, "poly_order"))
        read_attribute(group, "poly_order", poly_order_);
      tensor::Tensor<double> cf; // [n_poly, order+1, n_ch]
      read_dataset(group, "poly_curvefit", cf);
      int n_poly = cf.shape(0), no = cf.shape(1), nch = cf.shape(2);
      poly_order_ = no - 1;
      poly_s_.resize(n_poly * no);
      poly_a_.resize(n_poly * no);
      poly_f_.resize(n_poly * no);
      for (int w = 0; w < n_poly; ++w) {
        for (int k = 0; k < no; ++k) {
          int idx = w * no + k;
          poly_s_[idx] = cf(w, k, 0);
          if (nch == 3) {
            poly_a_[idx] = cf(w, k, 1) + cf(w, k, 2); // MT18 + MT102
            poly_f_[idx] = cf(w, k, 1);               // MT18
          } else {
            poly_a_[idx] = cf(w, k, 1);               // MT102
            poly_f_[idx] = 0.0;
          }
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
  fw_data_.assign((size_t)n_flat * FW_STRIDE, 0.0);

  // Per-channel residues for the collision kernel: keep the full n_mts
  // columns (the collapse above is only for the flight path). MT list,
  // clamps, and the MT -> channel index map.
  fast_n_mts_ = n_mts;
  fast_mts_.resize(n_mts);
  fast_mt_on_.resize(n_mts);
  fast_mt_off_.resize(n_mts);
  fast_mt_to_chan_.fill(-1);
  for (int c = 0; c < n_mts; ++c) {
    fast_mts_[c] = mts[c];
    fast_mt_on_[c] = e_on[c];
    fast_mt_off_[c] = e_off[c];
    if (mts[c] >= 0 && mts[c] < 900)
      fast_mt_to_chan_[mts[c]] = c;
  }
  fw_chan_res_.assign((size_t)n_flat * n_mts, 0.0);

  for (int w = 0; w < n_win; ++w) {
    double e_lo = fast_bounds_[w] * fast_bounds_[w];
    double e_hi = fast_bounds_[w + 1] * fast_bounds_[w + 1];
    int np = wend[w] - wstart[w];
    int base = fw_off_[w];
    for (int k = 0; k < np; ++k)
      fw_data_[(size_t)(base + k) * FW_STRIDE] = poles[wstart[w] + k];
    for (int j = 0; j < n_ext; ++j)
      fw_data_[(size_t)(base + np + j) * FW_STRIDE] = xp(w, j);

    // Per-channel residues (uncollapsed), CHANNEL-MAJOR within the window
    // block (base*n_mts + c*np_tot + k): the collision walk evaluates one
    // channel at a time, so each channel's residues must be contiguous.
    {
      int np_tot = np + n_ext;
      size_t wbase = (size_t)base * n_mts;
      for (int c = 0; c < n_mts; ++c) {
        for (int k = 0; k < np; ++k)
          fw_chan_res_[wbase + (size_t)c * np_tot + k] =
            residues(wstart[w] + k, c);
        for (int j = 0; j < n_ext; ++j)
          fw_chan_res_[wbase + (size_t)c * np_tot + np + j] = xr(w, j, c);
      }
    }

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
        // Interleaved slot: 1 elastic, 2 fission, 3 disappearance, 4 other
        int slot = (kind[c] == 3)   ? 1
                   : (kind[c] == 1) ? 2
                   : (kind[c] == 2) ? 3
                                    : 4;
        for (int k = 0; k < np; ++k)
          fw_data_[(size_t)(base + k) * FW_STRIDE + slot] +=
            residues(wstart[w] + k, c);
        for (int j = 0; j < n_ext; ++j)
          fw_data_[(size_t)(base + np + j) * FW_STRIDE + slot] += xr(w, j, c);
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

  // Real second-order-section flight arrays from fw_data_ (see wmp.h):
  // block-interleaved per window, denominator kept FACTORED (alpha, beta^2)
  // -- the expanded u^2+d1*u+d0 form is catastrophically unstable near real
  // poles (see wmp.h).
  fw_soa_.assign((size_t)10 * n_flat, 0.0);
  for (int w = 0; w < n_win; ++w) {
    int k0 = fw_off_[w];
    int np = fw_off_[w + 1] - k0;
    double* B = &fw_soa_[(size_t)10 * k0];
    for (int k = 0; k < np; ++k) {
      const std::complex<double>* rec = &fw_data_[(size_t)(k0 + k) * FW_STRIDE];
      double a = rec[0].real(), b = rec[0].imag();
      B[k] = a;
      B[np + k] = b * b;
      for (int g = 0; g < 4; ++g) {
        double rr = rec[1 + g].real(), ri = rec[1 + g].imag();
        B[(2 + 2 * g) * np + k] = rr;
        B[(3 + 2 * g) * np + k] = -(rr * a + ri * b);
      }
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

  // Real second-order-section sweep (SoA, plain double -> vectorizes). The
  // fast range is unbroadened, so sigma = (n1*u + n0)/((u-alpha)^2 + beta^2)
  // exactly, with the denominator FACTORED (see wmp.h: the expanded form
  // cancels to 0.0 near real poles -> inf/NaN). phi (for partial-channel
  // clamps) is computed from fw_data_ only when the window has partials.
  constexpr int FAST_MAX_ACTIVE = 96;
  std::complex<double> phi[FAST_MAX_ACTIVE];
  bool need_phi = !fast_partials_[w].empty();
  double u = sqrtE;
  const double* B = &fw_soa_[(size_t)10 * k0];
  const double *al = B, *be2 = B + np;
  const double *ne1 = B + 2 * np, *ne0 = B + 3 * np;
  const double *nf1 = B + 4 * np, *nf0 = B + 5 * np;
  const double *nd1 = B + 6 * np, *nd0 = B + 7 * np;
  const double *no1 = B + 8 * np, *no0 = B + 9 * np;
  double el = 0.0, fis = 0.0, dis = 0.0, oth = 0.0;
  // The reduction clause is required for vectorization: without it the four
  // FP sums are order-dependent and GCC keeps the loop scalar.
#pragma omp simd reduction(+ : el, fis, dis, oth)
  for (int k = 0; k < np; ++k) {
    double du = u - al[k];
    double inv = 1.0 / (du * du + be2[k]);
    el += (ne1[k] * u + ne0[k]) * inv;
    fis += (nf1[k] * u + nf0[k]) * inv;
    dis += (nd1[k] * u + nd0[k]) * inv;
    oth += (no1[k] * u + no0[k]) * inv;
  }
  if (need_phi) {
    const std::complex<double>* rec = &fw_data_[(size_t)k0 * FW_STRIDE];
    for (int k = 0; k < np; ++k, rec += FW_STRIDE) {
      double dr = sqrtE - rec[0].real(), di = -rec[0].imag();
      double iv = 1.0 / (dr * dr + di * di);
      phi[k] = {dr * iv, -di * iv};
    }
  }

  // Partially-clamped channels (rare; only near their thresholds)
  for (const auto& part : fast_partials_[w]) {
    if (E < part.E_on || E > part.E_off)
      continue;
    double s = 0.0;
    for (int k = 0; k < np; ++k)
      s += fp_res_[part.res_off + k].real() * phi[k].real() -
           fp_res_[part.res_off + k].imag() * phi[k].imag();
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

void WindowedMultipole::evaluate_fast_channels(double E, double* out) const
{
  uint64_t prof_t0 = wmp_prof_tick();
  double sqrtE = std::sqrt(E);

  // O(1) window lookup (same as evaluate_fast)
  int n_win = fw_off_.size() - 1;
  int cell = (int)((sqrtE - fast_bounds_[0]) * fast_cell_inv_du_);
  cell = std::min(std::max(cell, 0), (int)fast_cell_first_.size() - 1);
  int w = fast_cell_first_[cell];
  while (w < n_win - 1 && sqrtE >= fast_bounds_[w + 1])
    ++w;

  int k0 = fw_off_[w];
  int np = fw_off_[w + 1] - k0;
  int nm = fast_n_mts_;

  // One reciprocal per pole (pr, pi) = (Re(1/d), Im(1/d)), reused across
  // all channels, same convention as evaluate_fast.
  constexpr int FAST_MAX_ACTIVE = 96;
  double pr_[FAST_MAX_ACTIVE], pi_[FAST_MAX_ACTIVE];
  const std::complex<double>* rec = &fw_data_[(size_t)k0 * FW_STRIDE];
  for (int k = 0; k < np; ++k, rec += FW_STRIDE) {
    double dr = sqrtE - rec[0].real();
    double di = -rec[0].imag();
    double inv = 1.0 / (dr * dr + di * di);
    pr_[k] = dr * inv;
    pi_[k] = -di * inv;
  }

  // Channel-major layout: channel c's residues are contiguous
  const std::complex<double>* cr = &fw_chan_res_[(size_t)k0 * nm];
  for (int c = 0; c < nm; ++c) {
    if (E < fast_mt_on_[c] || E > fast_mt_off_[c]) {
      out[c] = 0.0;
      continue;
    }
    const std::complex<double>* crc = cr + (size_t)c * np;
    double s = 0.0; // Re[r/(u-p)] = r.re*Re(1/d) - r.im*Im(1/d)
    for (int k = 0; k < np; ++k)
      s += crc[k].real() * pr_[k] - crc[k].imag() * pi_[k];
    out[c] = (s > 0.0) ? s : 0.0;
  }
  wmp_prof_add(WPROF_COLL_CHAN, prof_t0);
}

void WindowedMultipole::fast_chan_begin(double E, FastChanCtx& ctx) const
{
  double sqrtE = std::sqrt(E);

  // O(1) window lookup (same as evaluate_fast)
  int n_win = fw_off_.size() - 1;
  int cell = (int)((sqrtE - fast_bounds_[0]) * fast_cell_inv_du_);
  cell = std::min(std::max(cell, 0), (int)fast_cell_first_.size() - 1);
  int w = fast_cell_first_[cell];
  while (w < n_win - 1 && sqrtE >= fast_bounds_[w + 1])
    ++w;

  ctx.k0 = fw_off_[w];
  ctx.np = fw_off_[w + 1] - ctx.k0;

  // Shared pole reciprocals, computed once; channels evaluated on demand
  // by fast_chan_xs() as the selection walk reaches them.
  const std::complex<double>* rec = &fw_data_[(size_t)ctx.k0 * FW_STRIDE];
  for (int k = 0; k < ctx.np; ++k, rec += FW_STRIDE) {
    double dr = sqrtE - rec[0].real();
    double di = -rec[0].imag();
    double inv = 1.0 / (dr * dr + di * di);
    ctx.pr[k] = dr * inv;
    ctx.pi[k] = -di * inv;
  }
}

std::tuple<double, double, double> WindowedMultipole::lowe_correction(
  double sqrtE, double xi, int i_window) const
{
  // Gather this window's non-1/v poles (pool + externals) with their
  // per-channel residues into fixed stack buffers as struct-of-arrays doubles
  // (no heap, and the inner loop is hand-rolled/vectorizable -- std::complex
  // division/multiply route through guarded libcalls, far too slow for a path
  // that fires on nearly every thermal lookup for light nuclides). The 1/v
  // pole (Re p == 0) is excluded: it is broadening-invariant and already
  // exact (its pool SoA residues are zeroed and it is added separately).
  constexpr int MAXP = 256;
  double Ppr[MAXP], Ppi[MAXP];
  double Rsr[MAXP], Rsi[MAXP], Rar[MAXP], Rai[MAXP], Rfr[MAXP], Rfi[MAXP];
  int n = 0;
  double hi = sqrtE + 5.0 * xi;
  bool narrow = false; // any pole triggers a refined pocket in lowe_nodes?
  auto take = [&](double a, double b, std::complex<double> rs,
                std::complex<double> ra, std::complex<double> rf) {
    if (std::abs(a) < 1e-12 || n >= MAXP)
      return; // 1/v pole (or overflow guard)
    Ppr[n] = a;
    Ppi[n] = b;
    Rsr[n] = rs.real();
    Rsi[n] = rs.imag();
    Rar[n] = ra.real();
    Rai[n] = ra.imag();
    Rfr[n] = rf.real();
    Rfi[n] = rf.imag();
    if (b < hi * 0.25 && -8.0 * b < a && a < hi + 8.0 * b)
      narrow = true;
    ++n;
  };
  int ps = win_start_[i_window], pe = win_end_[i_window];
  for (int i = ps; i < pe; ++i)
    take(gs_pr_[i], gs_pi_[i], {gs_sr_[i], gs_si_[i]}, {gs_ar_[i], gs_ai_[i]},
      {gs_fr_[i], gs_fi_[i]});
  int n_ext = ext_data_.shape(1);
  for (int j = 0; j < n_ext; ++j) {
    const std::complex<double>* rec = &ext_data_(i_window, j, 0);
    take(rec[0].real(), rec[0].imag(), rec[1], rec[2], rec[3]);
  }
  if (n == 0)
    return std::make_tuple(0.0, 0.0, 0.0);

  // Quadrature of the image-kernel integral: exp(-((u'+u)/xi)^2) times the
  // even part of u'^2 * sigma(u') over (0, u + 5*xi). One GL node contributes
  // (hand-rolled: 1/(u'-p) has real part (u'-pr)/den, imag part pi/den):
  double cs = 0.0, ca = 0.0, cf = 0.0;
  auto add_node = [&](double up, double wgt) {
    double kp = std::exp(-((up + sqrtE) / xi) * ((up + sqrtE) / xi));
    double sk = 0.0, ak = 0.0, fk = 0.0;
#pragma omp simd reduction(+ : sk, ak, fk)
    for (int m = 0; m < n; ++m) {
      double pr = Ppr[m], pi = Ppi[m];
      double d1 = up - pr, d2 = -up - pr;
      double inv1 = 1.0 / (d1 * d1 + pi * pi);
      double inv2 = 1.0 / (d2 * d2 + pi * pi);
      double phr = d1 * inv1 + d2 * inv2; // Re[1/(u'-p) + 1/(-u'-p)]
      double phi = pi * (inv1 + inv2);    // Im of the same
      sk += Rsr[m] * phr - Rsi[m] * phi;
      ak += Rar[m] * phr - Rai[m] * phi;
      fk += Rfr[m] * phr - Rfi[m] * phi;
    }
    double wk = wgt * kp * up * up;
    cs += wk * sk;
    ca += wk * ak;
    cf += wk * fk;
  };

  if (!narrow) {
    // HOT PATH (H1/O16/actinides): a single GL-12 panel over (0, hi) from the
    // static table -- no heap, no Newton. Bit-identical to lowe_nodes here,
    // which returns exactly this when no pole is narrow enough for a pocket.
    double c = 0.5 * hi;
    for (int k = 0; k < N_GL12; ++k)
      add_node(c * (GL12_X[k] + 1.0), c * GL12_W[k]);
  } else {
    // RARE PATH: composite pockets around narrow background pole(s).
    std::complex<double> Pc[MAXP];
    for (int m = 0; m < n; ++m)
      Pc[m] = {Ppr[m], Ppi[m]};
    std::vector<double> ug, wg;
    lowe_nodes(hi, Pc, n, ug, wg);
    for (std::size_t g = 0; g < ug.size(); ++g)
      add_node(ug[g], wg[g]);
  }
  double norm = 1.0 / (xi * SQRT_PI * sqrtE * sqrtE);
  return std::make_tuple(cs * norm, ca * norm, cf * norm);
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
    // Low-E polynomial segment: below poly_E_max_ the smooth cross section is a
    // windowed curvefit broadened EXACTLY (image kernel included) -- cheap, and
    // no G-L correction. Replaces poles+externals here.
    if (poly_present_ && E < poly_E_max_) {
      int n_poly = poly_sqrtE_bounds_.size() - 1;
      int w = std::upper_bound(poly_sqrtE_bounds_.begin(),
                poly_sqrtE_bounds_.end(), sqrtE) -
              poly_sqrtE_bounds_.begin() - 1;
      w = std::min(std::max(w, 0), n_poly - 1);
      int no = poly_order_ + 1;
      array<double, MAX_POLY_COEFFICIENTS> fac;
      if (sqrtkT > 0.0) {
        double dopp = sqrt_awr_ / sqrtkT;
        broaden_wmp_polynomials(E, dopp, no, fac.data());
      } else {
        double t = invE; // 0K basis E^(i/2-1): 1/E, 1/sqrtE, 1, sqrtE, ...
        for (int k = 0; k < no; ++k) { fac[k] = t; t *= sqrtE; }
      }
      double sig_s = 0.0, sig_a = 0.0, sig_f = 0.0;
      const double* cs = &poly_s_[w * no];
      const double* ca = &poly_a_[w * no];
      const double* cf = &poly_f_[w * no];
      for (int k = 0; k < no; ++k) {
        sig_s += cs[k] * fac[k];
        sig_a += ca[k] * fac[k];
        sig_f += cf[k] * fac[k];
      }
      return std::make_tuple(sig_s, sig_a, sig_f);
    }

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

    // xi = sqrt(kT/AWR) in sqrt(eV); sqrtkT = sqrt(kT)
    double xi = (sqrtkT > 0.0) ? sqrtkT / sqrt_awr_ : 0.0;
    double inv_xi = (xi > 0.0) ? 1.0 / xi : 0.0;
    // Loop-invariant i*sqrt(pi)/xi
    std::complex<double> cxi(0.0, SQRT_PI * inv_xi);

    // Hand-rolled complex arithmetic throughout (like the fast-segment
    // loop): std::complex division and multiplication route through
    // guarded libcalls / NaN-check branches; poles never coincide with
    // real sqrtE so the naive forms are safe. Records [p, rs, ra, rf]
    // are contiguous rows of gdata_ / ext_data_ walked by raw pointer.
    double xi2 = xi * xi;
    auto accumulate = [&](const std::complex<double>* rec) {
      double pr = rec[0].real();
      double pi = rec[0].imag();
      // d = u - p
      double dr = sqrtE - pr;
      double di = -pi;
      double br, bi; // base
      // Broadening decision: 1/v pole (pr == 0) and T=0 are exact 0K;
      // |z| >= 5 takes the stable asymptotic form (which includes the
      // old |z| > 100 territory as its far tail); |z| < 5 pays zpf8h.
      double m2 = dr * dr + di * di;
      if (xi <= 0.0 || pr == 0.0) {
        double minv = 1.0 / m2;
        br = dr * minv;
        bi = -di * minv;
      } else if (m2 >= 25.0 * xi2) {
        // Moderate-to-large |z|: base = [1 + (p/u)^2 c]/(u-p),
        // c = t (1/2 + t (3/4 + 15 t/8)), t = xi^2/(u-p)^2 = 1/z^2.
        // Substituting the Laurent series of w(z) ANALYTICALLY cancels
        // the huge (u+p) vs p^2 w terms (naive substitution amplifies
        // the truncation by the ~1e6 cancellation ratio). c -> 0
        // recovers the 0K form; truncation <= ~3e-5 of c.
        double minv = 1.0 / m2;
        double ir = dr * minv; // inv_d
        double ii = -di * minv;
        // t = xi^2 * inv_d^2
        double tr = xi2 * (ir * ir - ii * ii);
        double ti = xi2 * (2.0 * ir * ii);
        // c = t*(0.5 + t*(0.75 + t*1.875))
        double cr = 0.75 + 1.875 * tr;
        double ci = 1.875 * ti;
        double hr = 0.5 + (tr * cr - ti * ci);
        double hi = tr * ci + ti * cr;
        cr = tr * hr - ti * hi;
        ci = tr * hi + ti * hr;
        // q = p^2 * invE
        double p2r = (pr * pr - pi * pi) * invE;
        double p2i = (2.0 * pr * pi) * invE;
        // base = (1 + q*c) * inv_d
        double gr = 1.0 + (p2r * cr - p2i * ci);
        double gi = p2r * ci + p2i * cr;
        br = gr * ir - gi * ii;
        bi = gr * ii + gi * ir;
      } else {
        // Near pole: full Faddeeva evaluation
        std::complex<double> z(-dr * inv_xi, pi * inv_xi);
        std::complex<double> w_val = faddeeva(z);
        std::complex<double> p = rec[0];
        std::complex<double> b =
          ((sqrtE + p) + p * p * cxi * w_val) * invE;
        br = b.real();
        bi = b.imag();
      }
      sig_s += rec[1].real() * br - rec[1].imag() * bi;
      sig_a += rec[2].real() * br - rec[2].imag() * bi;
      if (fissionable_)
        sig_f += rec[3].real() * br - rec[3].imag() * bi;
    };
    // Pool poles: branch-free SoA sweep (vectorizable). Every pole takes the
    // SAME uniform path -- no runtime |z| test. At T>0 that is the inlined
    // zpf8h Faddeeva sigma basis (Im(z)=beta/xi>0 for resonance poles, so no
    // conjugation); the 1/v pole's SoA residues are zeroed here and its exact
    // 0K contribution is added below. At T=0 the plain pole sum.
    int ps = win_start_[i_window], pe = win_end_[i_window];
    if (xi <= 0.0) {
      for (int i = ps; i < pe; ++i) {
        double dr = sqrtE - gs_pr_[i], di = -gs_pi_[i];
        double minv = 1.0 / (dr * dr + di * di);
        double br = dr * minv, bi = -di * minv;
        sig_s += gs_sr_[i] * br - gs_si_[i] * bi;
        sig_a += gs_ar_[i] * br - gs_ai_[i] * bi;
        sig_f += gs_fr_[i] * br - gs_fi_[i] * bi;
      }
    } else {
      // NEAR/FAR PARTITION -- 1x work AND vectorized (strictly beats
      // compute-both). The cheap analytic Laurent base is computed for EVERY
      // pole (vectorizes); the EXPENSIVE zpf8h Faddeeva is done ONLY for the
      // NEAR poles (|z|^2 < 25*xi^2), which are first COMPACTED into a
      // contiguous scratch so that loop vectorizes too, then scattered back
      // over the analytic value. Far poles keep the analytic base (also the
      // cancellation-safe form they require). This removes the wasted zpf8h
      // on far poles that made Na-23 (sparse, mostly-far windows) slower.
      // 1/v pole: zeroed SoA residues -> contributes 0 in the reduction.
      double sq = SQRT_PI * inv_xi;
      double xi2 = xi * xi, thr = 25.0 * xi2;
      constexpr int MAXP = 64;
      double bR[MAXP], bI[MAXP], m2v[MAXP];
      double nPr[MAXP], nPi[MAXP], nbR[MAXP], nbI[MAXP];
      int nK[MAXP];
      for (int base = ps; base < pe; base += MAXP) {
        int np = std::min(MAXP, pe - base);
        // PASS 1: analytic Laurent base for ALL poles (vectorizes)
#pragma omp simd
        for (int k = 0; k < np; ++k) {
          double pr = gs_pr_[base + k], pi = gs_pi_[base + k];
          double dr = sqrtE - pr, di = -pi;
          double m2 = dr * dr + di * di;
          m2v[k] = m2;
          double p2r = pr * pr - pi * pi, p2i = 2.0 * pr * pi;
          double minv = 1.0 / m2;
          double ir = dr * minv, ii = -di * minv;
          double tr = xi2 * (ir * ir - ii * ii), ti = xi2 * (2.0 * ir * ii);
          double cr0 = 0.75 + 1.875 * tr, ci0 = 1.875 * ti;
          double hr0 = 0.5 + (tr * cr0 - ti * ci0), hi0 = tr * ci0 + ti * cr0;
          double cr = tr * hr0 - ti * hi0, ci = tr * hi0 + ti * hr0;
          double qr = p2r * invE, qi = p2i * invE;
          double gr = 1.0 + (qr * cr - qi * ci), gi = qr * ci + qi * cr;
          bR[k] = gr * ir - gi * ii;
          bI[k] = gr * ii + gi * ir;
        }
        // PASS 2: compact the near poles (|z|<5) into contiguous scratch
        int nn = 0;
        for (int k = 0; k < np; ++k) {
          if (m2v[k] < thr) {
            nPr[nn] = gs_pr_[base + k];
            nPi[nn] = gs_pi_[base + k];
            nK[nn] = k;
            ++nn;
          }
        }
        // PASS 3: zpf8h Faddeeva base for near poles only (contiguous ->
        // vectorizes). zs = (pr-sqrtE, pi)/xi + (0, 0.9)
#pragma omp simd
        for (int j = 0; j < nn; ++j) {
          double pr = nPr[j], pi = nPi[j];
          double p2r = pr * pr - pi * pi, p2i = 2.0 * pr * pi;
          double zr = (pr - sqrtE) * inv_xi;
          double zi = pi * inv_xi + 0.9;
          double nr = 0.0, ni = 0.564189504758109, t;
          t = nr*zr - ni*zi; ni = nr*zi + ni*zr; nr = t - 0.507771291744591;
          t = nr*zr - ni*zi; ni = nr*zi + ni*zr - 8.07354660639634; nr = t;
          t = nr*zr - ni*zi + 6.75847413957232; ni = nr*zi + ni*zr; nr = t;
          t = nr*zr - ni*zi; ni = nr*zi + ni*zr + 31.490536152863; nr = t;
          t = nr*zr - ni*zi - 21.9357456686406; ni = nr*zi + ni*zr; nr = t;
          t = nr*zr - ni*zi; ni = nr*zi + ni*zr - 32.310199761603; nr = t;
          t = nr*zr - ni*zi + 11.7559071436993; ni = nr*zi + ni*zr; nr = t;
          double zzr = zr*zr - zi*zi, zzi = 2.0*zr*zi;
          double dnr = 1.0, dni = 0.0;
          t = dnr*zzr - dni*zzi - 14.0; dni = dnr*zzi + dni*zzr; dnr = t;
          t = dnr*zzr - dni*zzi + 52.5; dni = dnr*zzi + dni*zzr; dnr = t;
          t = dnr*zzr - dni*zzi - 52.5; dni = dnr*zzi + dni*zzr; dnr = t;
          t = dnr*zzr - dni*zzi + 6.5625; dni = dnr*zzi + dni*zzr; dnr = t;
          double dm = 1.0 / (dnr*dnr + dni*dni);
          double wr = (nr*dnr + ni*dni) * dm;
          double wi = (ni*dnr - nr*dni) * dm;
          double wgr = -sq * wi, wgi = sq * wr;
          double hr = p2r * wgr - p2i * wgi, hi = p2r * wgi + p2i * wgr;
          nbR[j] = (sqrtE + pr + hr) * invE;
          nbI[j] = (pi + hi) * invE;
        }
        // PASS 4: scatter near results over the analytic base
        for (int j = 0; j < nn; ++j) {
          bR[nK[j]] = nbR[j];
          bI[nK[j]] = nbI[j];
        }
        // PASS 5: residue reduction (vectorizes)
#pragma omp simd reduction(+ : sig_s, sig_a, sig_f)
        for (int k = 0; k < np; ++k) {
          sig_s += gs_sr_[base + k] * bR[k] - gs_si_[base + k] * bI[k];
          sig_a += gs_ar_[base + k] * bR[k] - gs_ai_[base + k] * bI[k];
          sig_f += gs_fr_[base + k] * bR[k] - gs_fi_[base + k] * bI[k];
        }
      }
    }
    // 1/v background pole: exact 0K value (broadening-invariant, any T).
    if (onev_present_ && i_window == onev_window_) {
      double dr = sqrtE - onev_p_.real(), di = -onev_p_.imag();
      double minv = 1.0 / (dr * dr + di * di);
      double br = dr * minv, bi = -di * minv;
      sig_s += onev_rs_.real() * br - onev_rs_.imag() * bi;
      sig_a += onev_ra_.real() * br - onev_ra_.imag() * bi;
      sig_f += onev_rf_.real() * br - onev_rf_.imag() * bi;
    }
    // Externals (4/window): scalar branchy accumulate -- their |z| spans
    // ~5 to >100 across E, genuinely needing the analytic/Faddeeva split.
    for (int j = 0; j < n_ext; ++j) {
      accumulate(&ext_data_(i_window, j, 0));
    }

    // Low-E free-gas image-kernel correction: the one-kernel form above
    // neglects the exp(-((u'+u)/xi)^2) image kernel below u < LOWE_THR*xi. For
    // light nuclides (H1) this spans the thermal range and is essential for a
    // correct moderator (scattering) cross section. Both the correction AND
    // the one-kernel error decay as exp(-(u/xi)^2), so the firing threshold is
    // a SIGNIFICANCE bound, not just a form-validity bound: measured H1/O16
    // elastic error vs ACE is unchanged from 4.5 down to 2.5, while 4.5 makes
    // abundant mid-mass nuclides (O16 at u/xi~4 at the thermal peak) pay the
    // quadrature for a ~1e-7 correction. 3.5 keeps the correction wherever it
    // matters (<=5e-6 relative skipped) and stops O16 firing at thermal.
    constexpr double LOWE_THR = 3.5;
    if (xi > 0.0 && sqrtE < LOWE_THR * xi) {
      double cs, ca, cf;
      std::tie(cs, ca, cf) = lowe_correction(sqrtE, xi, i_window);
      sig_s -= cs;
      sig_a -= ca;
      sig_f -= cf;
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

  // Build the O(1) total-nu map for the fast segment's query range (exact,
  // self-validated; no-op for non-tabulated nu).
  if (nuc->multipole_->fast_present_) {
    nuc->init_nu_fast_map(
      nuc->multipole_->E_max_, nuc->multipole_->fast_E_max_);
  }
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
