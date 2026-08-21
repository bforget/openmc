#include <cstring>
#include <string>

#include <x86intrin.h>

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
    // LOW-ENERGY segment: its own group, its own [E_min, E_max], no poles.
    // Kept separate rather than folded in as resolved window 0 so each
    // segment's bounds mean what they say (Ben, 2026-08-13).
    if (object_exists(group, "lowe")) {
      read_lowe_segment(open_group(group, "lowe"));
    }
    return;
  }

  // Global-pole format (AAA-fit libraries): detected by the window pointer
  // arrays, which are unique to it (the OLD format also has a "curvefit"
  // dataset, so that name cannot be the discriminator). See wmp.h.
  if (object_exists(group, "win_pole_start")) {
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

//! Transport kind of a channel: 3 elastic, 1 fission, 2 disappearance,
//! 0 other (MT5, inelastic -- treated as scatter). Collapsing BY MT rather
//! than by column position is what lets v3's wider channel sets work; see
//! read_global_segment.
static int wmp_channel_kind(int mt)
{
  if (mt == 2)
    return 3;
  if (mt == 18)
    return 1;
  if (mt == 101 || (mt >= 102 && mt <= 117))
    return 2;
  return 0;
}

//! Read the LOW-ENERGY segment: curvefit only, no poles.
void WindowedMultipole::read_lowe_segment(hid_t group)
{
  tensor::Tensor<double> cf; // [n_cf, n_ch]
  read_dataset(group, "curvefit", cf);
  tensor::Tensor<int64_t> mts_ds;
  read_dataset(group, "mts", mts_ds);
  read_dataset(group, "E_min", lowe_E_min_);
  read_dataset(group, "E_max", lowe_E_max_);
  int n_cf = cf.shape(0), n_ch = cf.shape(1);
  lowe_n_ = n_cf;
  lowe_cf_.assign((size_t)n_cf * 3, 0.0);
  for (int k = 0; k < n_cf; ++k) {
    double vs = 0.0, va = 0.0, vf = 0.0;
    for (int c = 0; c < n_ch && c < mts_ds.shape(0); ++c) {
      const double v = cf(k, c);
      switch (wmp_channel_kind(static_cast<int>(mts_ds[c]))) {
      case 3: vs += v; break;
      case 1: vf += v; va += v; break;
      case 2: va += v; break;
      default: vs += v; break;
      }
    }
    lowe_cf_[3 * k + 0] = vs;
    lowe_cf_[3 * k + 1] = va;
    lowe_cf_[3 * k + 2] = vf;
  }
  lowe_present_ = true;
}

void WindowedMultipole::read_global_segment(hid_t group)
{
    global_format_ = true;
    read_dataset(group, "sqrtAWR", sqrt_awr_);
    read_dataset(group, "E_min", E_min_);
    read_dataset(group, "E_max", E_max_);

    tensor::Tensor<std::complex<double>> poles;    // (n_pool,)
    tensor::Tensor<std::complex<double>> residues; // (n_pool, n_ch)
    read_dataset(group, "poles", poles);
    read_dataset(group, "residues", residues);

    // COLLAPSE BY MT, NOT BY POSITION. This used to assume the channel set
    // was [MT2, MT18, MT102] when it had 3 columns and [MT2, MT102]
    // otherwise. v3 libraries carry wider sets -- U-235 is
    // [2, 18, 5, 102, 107] and B-10 is [2, 102, 103, 105, 107] -- and under
    // the old rule U-235 was marked NON-FISSIONABLE (5 != 3) and B-10's
    // absorption kept only MT102, dropping the (n,alpha) that is essentially
    // all of it. Both are central to a borated-water core.
    //
    // The per-MT arrays are KEPT (res_mt_, mts_): the triplet is only what
    // the FLIGHT kernel needs, while tallies and collision sampling still
    // want the individual channels (Ben, 2026-08-13).
    tensor::Tensor<int64_t> mts_ds;
    read_dataset(group, "mts", mts_ds);
    int n_ch = residues.shape(1);
    mts_.assign(n_ch, 0);
    for (int c = 0; c < n_ch && c < mts_ds.shape(0); ++c)
      mts_[c] = static_cast<int>(mts_ds[c]);
    // kind: 3 elastic, 1 fission, 2 disappearance, 0 other (MT5, inelastic)
    vector<int> kind(n_ch, 0);
    fissionable_ = false;
    for (int c = 0; c < n_ch; ++c) {
      int mt = mts_[c];
      if (mt == 2) {
        kind[c] = 3;
      } else if (mt == 18) {
        kind[c] = 1;
        fissionable_ = true;
      } else if (mt == 101 || (mt >= 102 && mt <= 117)) {
        kind[c] = 2;
      }
    }

    int n_pool = poles.shape(0);
    res_mt_ = residues;                       // keep per-MT for tallies
    gdata_ = tensor::Tensor<std::complex<double>>({(size_t)n_pool, 4});
    for (int i = 0; i < n_pool; ++i) {
      gdata_(i, MP_EA) = poles[i];
      std::complex<double> rs {0.0, 0.0}, ra {0.0, 0.0}, rf {0.0, 0.0};
      for (int c = 0; c < n_ch; ++c) {
        const std::complex<double> v = residues(i, c);
        switch (kind[c]) {
        case 3: rs += v; break;               // elastic
        case 1: rf += v; ra += v; break;      // fission counts in absorption
        case 2: ra += v; break;               // disappearance
        default: rs += v; break;              // MT5 / inelastic: scatter-like
        }
      }
      gdata_(i, MP_RS) = rs;
      gdata_(i, MP_RA) = ra;
      gdata_(i, MP_RF) = rf;
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

    // Grid convention (optional attributes; absent in older libraries).
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

    // (The DYADIC grid was removed 2026-08-13. v3 measured it against a
    // uniform mesh on Cr-50 and U-238 and it LOST on the 1/E-flux-weighted
    // cost metric -- refinement is driven by poles per window, so it coarsens
    // exactly the low-energy region that carries most of the flux. Every
    // library we produce is uniform, so the branch was dead weight in the
    // hottest loop in the code.)

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

    // External background poles are OPTIONAL: current libraries carry a
    // per-window curvefit instead (read below). Older global-format files
    // still have them, so both are supported.
    if (object_exists(group, "ext_poles")) {
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
    } else {
      ext_data_ = tensor::Tensor<std::complex<double>>({(size_t)n_win, 0, 4});
    }
    n_ext_ = (int)ext_data_.shape(1);
    n_win_ = (int)win_start_.size();
    {
    }

    // Build the branch-free SoA pool arrays (see wmp.h). The 1/v pole
    // (Re p == 0) is broadening-invariant and must NOT go through zpf8h:
    // zero its SoA residues so the vector loop skips it, and stash its data
    // + owning window for a separate exact-0K add.
    gs_stride_ = fissionable_ ? 8 : 6;   // drop the all-zero fission residue
    cf_ch_ = fissionable_ ? 3 : 2;
    gs_aos_.assign((size_t)n_pool * gs_stride_, 0.0);
    for (int i = 0; i < n_pool; ++i) {
      double* r = &gs_aos_[(size_t)i * gs_stride_];
      std::complex<double> p = gdata_(i, MP_EA);
      r[0] = p.real(); r[1] = p.imag();
      if (p.real() == 0.0) {
        // 1/v pole: residues stay ZERO so the vector loop adds nothing for
        // it; its exact 0K contribution is added separately per window.
        onev_present_ = true;
        onev_p_ = p;
        onev_rs_ = gdata_(i, MP_RS);
        onev_ra_ = gdata_(i, MP_RA);
        onev_rf_ = gdata_(i, MP_RF);
        if (onev_idx_ < 0)
          onev_idx_ = i;   // FIRST such pole, matching the old second pass
      } else {
        r[2] = gdata_(i, MP_RS).real(); r[3] = gdata_(i, MP_RS).imag();
        r[4] = gdata_(i, MP_RA).real(); r[5] = gdata_(i, MP_RA).imag();
        if (fissionable_) {
          r[6] = gdata_(i, MP_RF).real(); r[7] = gdata_(i, MP_RF).imag();
        }
      }
    }

    // Per-window smooth-background CURVEFIT (see wmp.h): ADDED to the pole
    // sum. curvefit shape [n_win, n_cf, n_ch]; collapse the per-MT channels
    // to the transport triplet exactly as the residues are collapsed above.
    if (object_exists(group, "curvefit")) {
      cf_present_ = true;
      tensor::Tensor<double> cf; // [n_win, n_cf, n_ch]
      read_dataset(group, "curvefit", cf);
      int n_cw = cf.shape(0), nc = cf.shape(1), nch = cf.shape(2);
      cf_n_ = nc;
      // INTERLEAVED per window: [s0,a0,f0, s1,a1,f1, ...]. Three separate
      // flat arrays meant three scattered lines per lookup; one window's
      // block is 3*cf_n_ doubles (120 B at cf_n_=5) = 2 lines, prefetched.
      cf_blk_.assign((size_t)n_cw * nc * cf_ch_, 0.0);
      // Collapsed BY MT with the same `kind` map as the residues above --
      // positional collapse silently mis-assigned every channel set wider
      // than 3 (see the residue block).
      cf_mt_ = cf;                            // keep per-MT for tallies
      for (int w = 0; w < n_cw; ++w) {
        for (int k = 0; k < nc; ++k) {
          double* d = &cf_blk_[((size_t)w * nc + k) * cf_ch_];
          double vs = 0.0, va = 0.0, vf = 0.0;
          for (int c = 0; c < nch && c < (int)kind.size(); ++c) {
            const double v = cf(w, k, c);
            switch (kind[c]) {
            case 3: vs += v; break;
            case 1: vf += v; va += v; break;
            case 2: va += v; break;
            default: vs += v; break;
            }
          }
          d[0] = vs;
          d[1] = va;
          if (cf_ch_ > 2)
            d[2] = vf;
        }
      }
    }

    // ---- materialise the PER-WINDOW CONTIGUOUS BLOCKS (see wmp.h) ----
    // Everything one lookup touches, laid out in the order it is read. Built
    // once here; the hot path then does a single offset and a sequential run.
    {
      int n_win = win_start_.size();
      int ncf3 = cf_present_ ? cf_n_ * cf_ch_ : 0;
      win_.resize(n_win);
      win_u_lo_ = sqrtE_bounds_.front();
      // 64-BYTE ALIGN every block. Packed flat, a block landing at an
      // arbitrary offset STRADDLES an extra cache line -- a 96 B block spans
      // three lines instead of two whenever it is not line-aligned, which for
      // uniformly distributed offsets is most of the time. Static line budget
      // over the 32-nuclide library: 4.06 -> 3.56 lines per lookup (-12%).
      // Costs only the padding rounded up to the next 8 doubles per window.
      constexpr size_t DPL = 64 / sizeof(double);   // doubles per cache line
      vector<size_t> sz(n_win);
      size_t total = 0;
      for (int w = 0; w < n_win; ++w) {
        int np = win_end_[w] - win_start_[w];
        if (np < 0)
          np = 0;                       // empty pointer range is legal
        win_[w].u_hi = sqrtE_bounds_[w + 1];
        win_[w].np = (uint16_t)np;
        win_[w].flags = 0;
        if (onev_present_ && onev_idx_ >= win_start_[w]
            && onev_idx_ < win_end_[w])
          win_[w].flags |= WIN_HAS_ONEV;
        sz[w] = ((size_t)ncf3 + (size_t)np * gs_stride_ + DPL - 1)
                / DPL * DPL;            // round up to a whole line
        total += sz[w];
      }
      // std::vector guarantees only 16-B alignment, so over-allocate one line
      // and start the first block at the next 64-B boundary.
      blk_.assign(total + DPL, 0.0);
      size_t pad = ((64 - (reinterpret_cast<uintptr_t>(blk_.data()) & 63)) & 63)
                   / sizeof(double);
      size_t off = pad;
      for (int w = 0; w < n_win; ++w) {
        win_[w].blk_off = (uint32_t)off;
        off += sz[w];
      }
      for (int w = 0; w < n_win; ++w) {
        double* b = &blk_[win_[w].blk_off];
        if (ncf3) {
          const double* c = &cf_blk_[(size_t)w * ncf3];
          std::copy(c, c + ncf3, b);
          b += ncf3;
        }
        const double* src = &gs_aos_[(size_t)win_start_[w] * gs_stride_];
        std::copy(src, src + (size_t)win_[w].np * gs_stride_, b);
      }
      // the source arrays are now redundant on the hot path
      vector<double>().swap(gs_aos_);
      vector<double>().swap(cf_blk_);
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

  // LAZY ELASTIC (see wmp.h fast_merge_sct_): fold elastic into the
  // scatter group for the flight and recover it on demand at collisions.
  // The rewindowed flight path builds its blocks against the historical
  // four-group order, so it keeps the unmerged layout.
  fast_merge_sct_ = !object_exists(group, "flight_bounds");

  // Transport kind of each channel: 0 other-scatter (inelastic, (n,xn),
  // MT5, ...), 1 fission, 2 disappearance lump, 3 elastic.
  vector<int> kind(n_mts, 0);
  for (int c = 0; c < n_mts; ++c) {
    int mt = mts[c];
    if (mt == 2) {
      kind[c] = 3;
    } else if (mt == 18) {
      kind[c] = 1;
    } else if (mt == 101 || (mt >= 102 && mt <= 117)) {
      // FULL disappearance range, matching the resolved map above. This used
      // to list only 101/102/103/107, so any other disappearance channel --
      // MT104 (n,t), MT105 (n,d), MT106, MT108 ... -- fell through to
      // kind 0 "other scatter" and was added to SCATTERING instead of
      // absorption, silently. Harmless while the fast segment carried only
      // the MT101 lump; a real defect from 2026-08-18, when disappearance
      // reactions started being carried INDIVIDUALLY so depletion and tallies
      // can resolve them (B-10 carries MT104 and MT105, B-11 carries MT105).
      kind[c] = 2;
    }
  }

  // Flatten each window's pointer range + externals; collapse residues of
  // fully-active channels into the four transport vectors, keep channels
  // that turn on/off inside the window as individually-clamped partials.
  // PER-WINDOW CURVEFIT for the fast segment. v3 replaces the 4 external
  // poles with an ncf-term curvefit fitted to the externals' own
  // contribution on a self-generated grid -- the externals stored COMPLEX
  // residues (4*nch*16 B), the curvefit stores REAL coefficients
  // (ncf*nch*8 B), halving the background term, which dominated the smooth
  // nuclides (U-238's externals were 87% of its fast bytes).
  //
  // The fast range is UNBROADENED, so the basis is simply E^(k/2 - 1) and
  // needs no broaden_wmp_polynomials call. Collapsed into the same four
  // channels the pole sweep uses (elastic / fission / disappearance / other).
  if (object_exists(group, "curvefit")) {
    tensor::Tensor<double> cf; // [n_win, n_cf, n_ch]
    read_dataset(group, "curvefit", cf);
    int ncw = cf.shape(0), nc = cf.shape(1), nch = cf.shape(2);
    fast_cf_n_ = nc;
    fast_cf_tmp_.assign((size_t)ncw * nc * 4, 0.0);
    // PER-MT copy for the collision kernel. The collapse below is for the
    // FLIGHT only; channel selection needs each MT's own background, exactly
    // as fw_chan_res_ keeps each MT's residues (Ben, 2026-08-13: "combine at
    // load for the flight, but keep everything separate for collision").
    fw_chan_cf_.assign((size_t)ncw * nc * n_mts, 0.0);
    for (int w = 0; w < ncw; ++w)
      for (int k = 0; k < nc; ++k)
        for (int c = 0; c < nch && c < n_mts; ++c)
          fw_chan_cf_[((size_t)w * nc + k) * n_mts + c] = cf(w, k, c);
    for (int w = 0; w < ncw; ++w) {
      for (int k = 0; k < nc; ++k) {
        double* d = &fast_cf_tmp_[((size_t)w * nc + k) * 4];
        const double _elo = fast_bounds_[w] * fast_bounds_[w];
        const double _ehi = fast_bounds_[w + 1] * fast_bounds_[w + 1];
        for (int c = 0; c < nch && c < n_mts; ++c) {
          // PARTIAL CHANNELS ARE EXCLUDED, exactly as they are from the
          // collapsed RESIDUES below. A channel whose clamp cuts inside this
          // window has its POLES gated at evaluation; leaving its curvefit in
          // the collapsed groups gated NOTHING and applied the background
          // across the whole window. Ne-20 window 25 (13.79-20 MeV) holds
          // MT16, which turns on at 17.91 MeV: the flight read 13% low below
          // that and exactly right above it (2026-08-18).
          if (!(e_on[c] <= _elo && e_off[c] >= _ehi))
            continue;
          const double v = cf(w, k, c);
          // Merged slot order is [scatter, fission, disappearance, ELASTIC];
          // elastic lands in BOTH slot 0 (summed with other-scatter, for the
          // flight) and slot 3 (alone, for evaluate_fast_elastic).
          switch (kind[c]) {
          case 3:
            d[0] += v;                              // elastic -> scatter
            if (fast_merge_sct_) d[3] += v;         // elastic-only copy
            break;
          case 1: d[1] += v; break;                 // fission
          case 2: d[2] += v; break;                 // disappearance
          default:
            d[fast_merge_sct_ ? 0 : 3] += v;        // other (MT5, inelastic)
            break;
          }
        }
      }
    }
  }

  fw_off_.assign(n_win + 1, 0);
  fast_partials_.resize(n_win);
  fast_has_part_.assign(n_win, 0);
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
  // Elastic channel, cached for the lazy split (evaluate_fast_elastic).
  fast_ch_el_ = fast_channel_of_mt(2);
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
        part.ch = c;
        part.res_off = (int)fp_res_.size();
        for (int k = 0; k < np; ++k)
          fp_res_.push_back(residues(wstart[w] + k, c));
        for (int j = 0; j < n_ext; ++j)
          fp_res_.push_back(xr(w, j, c));
        fast_partials_[w].push_back(part);
        fast_has_part_[w] = 1;
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
  // CHANNEL SPECIALISATION. The sweep reserves two numerator slots for each of
  // the four collapsed channels (elastic, fission, disappearance=capture,
  // other), so a pole always cost 10 doubles. Most nuclides do not use all
  // four -- H-1 in the fast range is elastic + capture only -- and packing
  // just the channels a nuclide actually uses removes those bytes at zero
  // accuracy cost, with no refit and no change to the pole count. Unlike the
  // resolved range's failed byte reductions this does not disturb the access
  // pattern: same SoA sweep, fewer streams.
  //
  // Note `dis` is CAPTURE (kind 2), which nearly every nuclide has -- do not
  // confuse it with an unused channel. Every one of the eight combinations is
  // instantiated below; a fall-through reads the wrong slots and segfaults
  // downstream in Nuclide::calculate_xs, not here.
  fast_has_fis_ = fast_has_dis_ = fast_has_oth_ = false;
  for (size_t i = 0; i < (size_t)n_flat; ++i) {
    const std::complex<double>* rec = &fw_data_[i * FW_STRIDE];
    if (rec[2] != 0.0) fast_has_fis_ = true;
    if (rec[3] != 0.0) fast_has_dis_ = true;
    if (rec[4] != 0.0) fast_has_oth_ = true;
  }
  // With the lazy split, other-scatter no longer gets a slot of its own --
  // it is summed into the elastic slot below, which then means SCATTER.
  const int nch = 1 + (int)fast_has_fis_ + (int)fast_has_dis_
                    + (int)(fast_has_oth_ && !fast_merge_sct_);
  fast_soa_stride_ = 2 + 2 * nch;
  if (std::getenv("WMP_FASTCH_DEBUG"))
    std::fprintf(stderr,
      "[fastch] %-8s %s%s%s -> stride %d/10 (%.0f%% of bytes)%s\n",
      name_.c_str(),
      (fast_merge_sct_ && fast_has_oth_) ? "sct(el+oth)" : "el",
      fast_has_fis_ ? "+fis" : "", fast_has_dis_ ? "+dis" : "",
      fast_soa_stride_, 100.0 * fast_soa_stride_ / 10.0,
      fast_merge_sct_ ? "  [lazy elastic]" : "");
  // One block per window: curvefit first, then the SoA pole run.
  const int cfw = 4 * fast_cf_n_;
  fast_blk_off_.assign(n_win + 1, 0);
  {
    size_t off = 0;
    for (int w = 0; w < n_win; ++w) {
      fast_blk_off_[w] = off;
      off += (size_t)cfw
             + (size_t)fast_soa_stride_ * (fw_off_[w + 1] - fw_off_[w]);
    }
    fast_blk_off_[n_win] = off;
    fw_soa_.assign(off, 0.0);
  }
  for (int w = 0; w < n_win; ++w) {
    int k0 = fw_off_[w];
    int np = fw_off_[w + 1] - k0;
    if (cfw) {
      double* d = &fw_soa_[fast_blk_off_[w]];
      for (int i = 0; i < cfw; ++i)
        d[i] = fast_cf_tmp_[(size_t)w * cfw + i];
    }
    double* B = &fw_soa_[fast_blk_off_[w] + cfw];
    for (int k = 0; k < np; ++k) {
      const std::complex<double>* rec = &fw_data_[(size_t)(k0 + k) * FW_STRIDE];
      double a = rec[0].real(), b = rec[0].imag();
      B[k] = a;
      B[np + k] = b * b;
      int slot = 0;
      for (int g = 0; g < 4; ++g) {
        if ((g == 1 && !fast_has_fis_) || (g == 2 && !fast_has_dis_)
            || (g == 3 && !fast_has_oth_))
          continue;
        // LAZY SPLIT: other-scatter (g == 3) is summed into the elastic slot
        // instead of taking one of its own, so slot 0 holds SCATTER. Doing it
        // on the residues here -- rather than on the two numerators below --
        // is what removes the pair; the numerator form is affine in the
        // residue, so the sum of numerators equals the numerator of the sum.
        if (g == 3 && fast_merge_sct_)
          continue;
        std::complex<double> res = rec[1 + g];
        if (g == 0 && fast_merge_sct_ && fast_has_oth_)
          res += rec[4];
        double rr = res.real(), ri = res.imag();
        B[(2 + 2 * slot) * np + k] = rr;
        B[(3 + 2 * slot) * np + k] = -(rr * a + ri * b);
        ++slot;
      }
    }
  }  fast_cf_tmp_.clear();
  fast_cf_tmp_.shrink_to_fit();


  // ---- rewindowed FLIGHT path (optional; see wmp.h FlWin) ----------------
  // Libraries without the flight_* datasets keep the full-pole sweep above.
  if (object_exists(group, "flight_bounds")) {
    tensor::Tensor<double> flb, flcf;
    tensor::Tensor<int64_t> fls, fle, fli;
    read_dataset(group, "flight_bounds", flb);
    read_dataset(group, "flight_win_start", fls);
    read_dataset(group, "flight_win_end", fle);
    read_dataset(group, "flight_pole_idx", fli);
    read_dataset(group, "flight_curvefit", flcf);
    tensor::Tensor<double> slo, shi;
    tensor::Tensor<int64_t> sw0, snw;
    read_dataset(group, "flight_seg_lo", slo);
    read_dataset(group, "flight_seg_hi", shi);
    read_dataset(group, "flight_seg_win0", sw0);
    read_dataset(group, "flight_seg_nwin", snw);

    int n_sub = fls.shape(0);
    fl_order_ = flcf.shape(1);
    fl_bounds_.assign(flb.begin(), flb.end());

    // Stored slot order is elastic, fission, disappearance, other -- the same
    // channel specialisation the full sweep uses -- and kslot maps each stored
    // slot back to the pipeline's `kind` index so the curvefit and the
    // numerators land in the same place.
    int kslot[4];
    fl_nch_ = 0;
    kslot[fl_nch_++] = 3;
    if (fast_has_fis_) kslot[fl_nch_++] = 1;
    if (fast_has_dis_) kslot[fl_nch_++] = 2;
    if (fast_has_oth_) kslot[fl_nch_++] = 0;

    constexpr size_t DPL = 8; // doubles per 64-byte line
    vector<size_t> sz(n_sub);
    size_t tot = 0;
    for (int w = 0; w < n_sub; ++w) {
      size_t npw = (size_t)(fle[w] - fls[w]);
      size_t need = (size_t)FL_HDR + (size_t)fl_order_ * fl_nch_
                    + (size_t)fast_soa_stride_ * npw;
      sz[w] = (need + DPL - 1) / DPL * DPL;
      tot += sz[w];
    }
    fl_blk_.assign(tot + DPL, 0.0);
    size_t pad = ((64 - ((uintptr_t)fl_blk_.data() % 64)) % 64) / sizeof(double);
    fl_base_ = fl_blk_.data() + pad;

    fl_win_.resize(n_sub);
    size_t off = 0;
    for (int w = 0; w < n_sub; ++w) {
      double lo = fl_bounds_[w], hi = fl_bounds_[w + 1];
      int npw = (int)(fle[w] - fls[w]);
      double wid = hi - lo;
      fl_win_[w] = {lo, hi, wid > 0.0 ? 1.0 / wid : 0.0, (uint32_t)off,
                    (uint32_t)npw};
      double* Bw = fl_base_ + off;
      Bw[0] = lo;
      Bw[1] = hi;
      Bw[2] = wid > 0.0 ? 1.0 / wid : 0.0;
      // [3] packs (next block offset, pole count). np cannot be derived from
      // the block size because sizes are padded up to a 64-byte line. 0 marks
      // the last block so the advance has a stop without a bounds array.
      {
        uint32_t two[2] = {
          (w + 1 < n_sub) ? (uint32_t)(off + sz[w]) : 0u, (uint32_t)npw};
        std::memcpy(&Bw[3], two, sizeof(double));
      }
      Bw += FL_HDR;
      for (int g = 0; g < fl_nch_; ++g)
        for (int i = 0; i < fl_order_; ++i)
          Bw[g * fl_order_ + i] = flcf(w, i, kslot[g]);

      double* Pp = Bw + (size_t)fl_order_ * fl_nch_;
      double E_lo = lo * lo, E_hi = hi * hi;
      for (int k = 0; k < npw; ++k) {
        int gi = (int)fli[fls[w] + k];
        double a = poles[gi].real(), b = poles[gi].imag();
        Pp[k] = a;
        Pp[npw + k] = b * b;
        for (int g = 0; g < fl_nch_; ++g) {
          std::complex<double> r = 0.0;
          for (int c = 0; c < n_mts; ++c) {
            // thresholds are sub-window boundaries, so a channel is either
            // fully on or fully off here -- same test the pipeline fitted with
            if (kind[c] != kslot[g])
              continue;
            if (!(e_on[c] <= E_lo * (1.0 + 1e-12) &&
                  e_off[c] >= E_hi * (1.0 - 1e-12)))
              continue;
            r += residues(gi, c);
          }
          double rr = r.real(), ri = r.imag();
          Pp[(2 + 2 * g) * npw + k] = rr;
          Pp[(3 + 2 * g) * npw + k] = -(rr * a + ri * b);
        }
      }
      off += sz[w];
    }

    // Segment table + a small uniform cell index onto it. Both are tiny
    // (measured 3.1 sub-windows per segment over the library), so they stay
    // cache-hot, and the per-lookup work becomes: one cell load, a short
    // advance, then ARITHMETIC for the sub-window inside the segment.
    int n_seg = sw0.shape(0);
    fl_seg_.resize(n_seg);
    for (int i = 0; i < n_seg; ++i) {
      double a = slo[i], b = shi[i];
      int nw = (int)snw[i];
      fl_seg_[i] = {a, b, nw / (b - a), (int32_t)sw0[i], (int32_t)nw};
    }
    // Old strategy's index, kept so both can be A/B'd in ONE binary: the
    // machine drifts 7-64% between runs here, far more than the difference
    // being measured, so separate-binary comparisons cannot resolve it.
    {
      int nc0 = 4 * n_sub;
      double a = fl_bounds_.front(), b = fl_bounds_.back();
      fl_cell_inv_du_ = nc0 / (b - a);
      fl_cell_first_.resize(nc0);
      int ww = 0;
      for (int i = 0; i < nc0; ++i) {
        double u_cell = a + i / fl_cell_inv_du_;
        while (ww < n_sub - 1 && fl_bounds_[ww + 1] <= u_cell)
          ++ww;
        fl_cell_first_[i] = ww;
      }
    }
    {
      int nc0 = (int)fl_cell_first_.size();
      fl_cell_blk_.resize(nc0);
      for (int i = 0; i < nc0; ++i)
        fl_cell_blk_[i] = fl_win_[fl_cell_first_[i]].off;
    }
    fl_u0_ = fl_bounds_.front();
    double u_top = fl_bounds_.back();
    int n_cells = 4 * n_seg;
    fl_seg_cell_inv_du_ = n_cells / (u_top - fl_u0_);
    fl_cell_seg_.resize(n_cells);
    int sg = 0;
    for (int i = 0; i < n_cells; ++i) {
      double u_cell = fl_u0_ + i / fl_seg_cell_inv_du_;
      while (sg < n_seg - 1 && fl_seg_[sg].u_hi <= u_cell)
        ++sg;
      fl_cell_seg_[i] = sg;
    }
    fl_have_ = true;
  }
}

//! Fast-range sweep, specialised at COMPILE TIME on which collapsed channels
//! the nuclide uses. Real second-order sections, denominator kept FACTORED.
template <bool FIS, bool DIS, bool OTH>
static inline void fast_sweep(const double* B, int np, double u, double& el,
  double& fis, double& dis, double& oth)
{
  constexpr int I_FIS = 1;
  constexpr int I_DIS = I_FIS + (FIS ? 1 : 0);
  constexpr int I_OTH = I_DIS + (DIS ? 1 : 0);
  const double* al = B;
  const double* be2 = B + np;
#pragma omp simd reduction(+ : el, fis, dis, oth)
  for (int k = 0; k < np; ++k) {
    double du = u - al[k];
    double inv = 1.0 / (du * du + be2[k]);
    el += (B[2 * np + k] * u + B[3 * np + k]) * inv;
    if (FIS)
      fis += (B[(2 + 2 * I_FIS) * np + k] * u
              + B[(3 + 2 * I_FIS) * np + k]) * inv;
    if (DIS)
      dis += (B[(2 + 2 * I_DIS) * np + k] * u
              + B[(3 + 2 * I_DIS) * np + k]) * inv;
    if (OTH)
      oth += (B[(2 + 2 * I_OTH) * np + k] * u
              + B[(3 + 2 * I_OTH) * np + k]) * inv;
  }
}

//! Rewindowed flight evaluation: window-local Horner curvefit seeded into the
//! accumulators, then the retained-pole sweep adds on top. No phi[], no
//! partial-channel clamps, no fw_data_ touch -- thresholds are window edges.
template <bool FIS, bool DIS, bool OTH>
static inline void fl_eval(const double* B, int order, int np, double u,
  double t, double& el, double& fis, double& dis, double& oth)
{
  constexpr int I_FIS = 1;
  constexpr int I_DIS = I_FIS + (FIS ? 1 : 0);
  constexpr int I_OTH = I_DIS + (DIS ? 1 : 0);
  constexpr int NC = 1 + (FIS ? 1 : 0) + (DIS ? 1 : 0) + (OTH ? 1 : 0);
  auto horner = [&](int g) {
    const double* c = B + g * order;
    double v = c[order - 1];
    for (int i = order - 2; i >= 0; --i)
      v = v * t + c[i];
    return v;
  };
  el = horner(0);
  if (FIS)
    fis = horner(I_FIS);
  if (DIS)
    dis = horner(I_DIS);
  if (OTH)
    oth = horner(I_OTH);
}

//! Pole base pointer for a block whose curvefit occupies order*NC doubles.
template <bool FIS, bool DIS, bool OTH>
static inline const double* fl_poles(const double* B, int order)
{
  constexpr int NC = 1 + (FIS ? 1 : 0) + (DIS ? 1 : 0) + (OTH ? 1 : 0);
  return B + order * NC;
}

std::tuple<double, double, double> WindowedMultipole::evaluate_fast(
  double E) const
{
  // DEFENSIVE FLOOR. The low-E curvefit basis starts at 1/E, so E == 0 (or a
  // NaN arriving from a bad collision) turns every channel into NaN, which
  // then defeats both bounds checks in Nuclide::calculate_xs and segfaults.
  // Transport now kills such particles in Material::calculate_neutron_xs, but
  // this path is also reachable from tallies and the C API. Small POSITIVE
  // energies are fine and are NOT clamped -- the 1/v extrapolation is exact
  // down to at least 1e-11 eV (verified 2026-08-13).
  if (!(E > 0.0))
    E = (E_min_ > 0.0) ? E_min_ : 1.0e-11;

  double sqrtE = std::sqrt(E);

  if (fl_have_) {
    uint64_t prof_fl0 = wmp_prof_tick_sub();
    // DEFAULT is the cell table + linear advance. The segment-arithmetic
    // alternative (WMP_FL_SEARCH=arith) is 24% SLOWER on the search and 1.7%
    // on the whole fast flight -- measured same-binary, order reversed, four
    // runs, identical leakage. Not because arithmetic is expensive: both do
    // three loads, but the cell grid is 4x finer than the windows so the
    // advance almost always exits at once, letting the CPU speculate `w` and
    // issue fl_win_[w] IN PARALLEL with the bounds check (chain depth 2).
    // Under the segment scheme, `w = S.win0 + (u-S.u_lo)*S.inv_du` is a true
    // data dependency on the segment record, so fl_win_[w] cannot issue until
    // it returns, with an FP multiply and convert inside the chain (depth 3).
    // Making the index arithmetic serialised two loads that used to overlap.
    static const bool use_loop = [] {
      const char* e = std::getenv("WMP_FL_SEARCH");
      return !(e && std::string(e) == "arith");
    }();
    // WMP_FL_HDR=0 restores the separate fl_win_ record for A/B.
    static const bool use_hdr = [] {
      const char* e = std::getenv("WMP_FL_HDR");
      return !(e && e[0] == '0');
    }();
    if (use_hdr && use_loop) {
      int cell = (int)((sqrtE - fl_u0_) * fl_cell_inv_du_);
      cell = std::min(std::max(cell, 0), (int)fl_cell_blk_.size() - 1);
      const double* H = fl_base_ + fl_cell_blk_[cell];
      uint32_t two[2];
      std::memcpy(two, &H[3], sizeof(double));
      // advance: u_hi and the next offset both come from the line already
      // loaded, so a step costs no extra array
      while (sqrtE >= H[1] && two[0] != 0u) {
        H = fl_base_ + two[0];
        std::memcpy(two, &H[3], sizeof(double));
      }
      double t = (sqrtE - H[0]) * H[2];
      int np = (int)two[1];
      const double* B = H + FL_HDR;
      double el = 0.0, fis = 0.0, dis = 0.0, oth = 0.0;
      int mask = (fast_has_fis_ ? 4 : 0) | (fast_has_dis_ ? 2 : 0)
                 | (fast_has_oth_ ? 1 : 0);
      switch (mask) {
      case 0: fl_eval<false,false,false>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<false,false,false>(fl_poles<false,false,false>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      case 1: fl_eval<false,false,true>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<false,false,true>(fl_poles<false,false,true>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      case 2: fl_eval<false,true,false>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<false,true,false>(fl_poles<false,true,false>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      case 3: fl_eval<false,true,true>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<false,true,true>(fl_poles<false,true,true>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      case 4: fl_eval<true,false,false>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<true,false,false>(fl_poles<true,false,false>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      case 5: fl_eval<true,false,true>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<true,false,true>(fl_poles<true,false,true>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      case 6: fl_eval<true,true,false>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<true,true,false>(fl_poles<true,true,false>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      default: fl_eval<true,true,true>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);
        fast_sweep<true,true,true>(fl_poles<true,true,true>(B, fl_order_), np, sqrtE, el, fis, dis, oth); break;
      }
      el = (el > 0.0) ? el : 1.0e-5; fis = (fis > 0.0) ? fis : 0.0;
      dis = (dis > 0.0) ? dis : 0.0; oth = (oth > 0.0) ? oth : 0.0;
      return {el + fis + dis + oth, dis + fis, fis};
    }
    int w;
    if (use_loop) {
      int n_sub2 = (int)fl_win_.size();
      int cell = (int)((sqrtE - fl_u0_) * fl_cell_inv_du_);
      cell = std::min(std::max(cell, 0), (int)fl_cell_first_.size() - 1);
      w = fl_cell_first_[cell];
      // WMP_FL_BOUNDS=1 restores the old separate-array check for A/B.
      static const bool use_bounds_array =
        std::getenv("WMP_FL_BOUNDS") != nullptr;
      if (use_bounds_array) {
        while (w < n_sub2 - 1 && sqrtE >= fl_bounds_[w + 1])
          ++w;
      } else {
        while (w < n_sub2 - 1 && sqrtE >= fl_win_[w].u_hi)
          ++w;
      }
    } else {
      int cell = (int)((sqrtE - fl_u0_) * fl_seg_cell_inv_du_);
      cell = std::min(std::max(cell, 0), (int)fl_cell_seg_.size() - 1);
      int sgi = fl_cell_seg_[cell];
      int n_seg = (int)fl_seg_.size();
      while (sgi < n_seg - 1 && sqrtE >= fl_seg_[sgi].u_hi)
        ++sgi;
      const FlSeg& S = fl_seg_[sgi];
      int j = (int)((sqrtE - S.u_lo) * S.inv_du);
      j = std::min(std::max(j, 0), S.nwin - 1);
      w = S.win0 + j;
    }
    const FlWin& W = fl_win_[w];
    const double* B = fl_base_ + W.off + FL_HDR;
    double t = (sqrtE - W.u_lo) * W.inv_w;
    int np = (int)W.np;
    double el = 0.0, fis = 0.0, dis = 0.0, oth = 0.0;
    int mask = (fast_has_fis_ ? 4 : 0) | (fast_has_dis_ ? 2 : 0)
               | (fast_has_oth_ ? 1 : 0);
    WMP_PROF_FENCE();
    wmp_prof_add_sub(WPROF_FL_SEARCH, prof_fl0);
    uint64_t prof_fl1 = wmp_prof_tick_sub();
#define FL_CASE(M, A, B_, C)                                                   \
  case M:                                                                      \
    fl_eval<A, B_, C>(B, fl_order_, np, sqrtE, t, el, fis, dis, oth);          \
    WMP_PROF_FENCE();                                                          \
    wmp_prof_add_sub(WPROF_FL_CURVE, prof_fl1);                                \
    prof_fl1 = wmp_prof_tick_sub();                                            \
    fast_sweep<A, B_, C>(                                                      \
      fl_poles<A, B_, C>(B, fl_order_), np, sqrtE, el, fis, dis, oth);         \
    break;
    switch (mask) {
      FL_CASE(0, false, false, false)
      FL_CASE(1, false, false, true)
      FL_CASE(2, false, true, false)
      FL_CASE(3, false, true, true)
      FL_CASE(4, true, false, false)
      FL_CASE(5, true, false, true)
      FL_CASE(6, true, true, false)
    default:
      FL_CASE(7, true, true, true)
    }
#undef FL_CASE
    WMP_PROF_FENCE();
    wmp_prof_add_sub(WPROF_FL_SWEEP, prof_fl1);
    el = std::max(el, 0.0);
    fis = std::max(fis, 0.0);
    dis = std::max(dis, 0.0);
    oth = std::max(oth, 0.0);
    return {el + fis + dis + oth, dis + fis, fis};
  }

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
  bool need_phi = fast_has_part_[w];   // flag, not a vector<> indirection
  double u = sqrtE;
  const double* BLK = &fw_soa_[fast_blk_off_[w]];
  const double* B = BLK + 4 * fast_cf_n_;
  double el = 0.0, fis = 0.0, dis = 0.0, oth = 0.0;
  // One switch on a per-nuclide constant; each instantiation carries
  // COMPILE-TIME slot offsets and sweeps only the channels that exist.
  // All eight combinations are instantiated -- an unhandled one would read
  // the wrong slots and feed garbage cross sections to the transport loop.
  // With the lazy split, slot 0 holds SCATTER (elastic + other) and there is
  // no other-scatter slot, so the OTH instantiations are simply not selected
  // -- `el` receives the merged scatter sum.
  int mask = (fast_has_fis_ ? 4 : 0) | (fast_has_dis_ ? 2 : 0)
             | ((fast_has_oth_ && !fast_merge_sct_) ? 1 : 0);
  switch (mask) {
  case 0: fast_sweep<false, false, false>(B, np, u, el, fis, dis, oth); break;
  case 1: fast_sweep<false, false, true>(B, np, u, el, fis, dis, oth); break;
  case 2: fast_sweep<false, true, false>(B, np, u, el, fis, dis, oth); break;
  case 3: fast_sweep<false, true, true>(B, np, u, el, fis, dis, oth); break;
  case 4: fast_sweep<true, false, false>(B, np, u, el, fis, dis, oth); break;
  case 5: fast_sweep<true, false, true>(B, np, u, el, fis, dis, oth); break;
  case 6: fast_sweep<true, true, false>(B, np, u, el, fis, dis, oth); break;
  default: fast_sweep<true, true, true>(B, np, u, el, fis, dis, oth); break;
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
    // ...and this channel's OWN curvefit, which the collapsed groups omit.
    if (fast_cf_n_ > 0 && !fw_chan_cf_.empty() && part.ch >= 0) {
      const double* cfp =
        &fw_chan_cf_[((size_t)w * fast_cf_n_) * fast_n_mts_ + part.ch];
      double fac = 1.0 / E;
      for (int k = 0; k < fast_cf_n_; ++k, cfp += fast_n_mts_) {
        s += *cfp * fac;
        fac *= sqrtE;
      }
    }
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
      // other-scatter joins the scatter group under the lazy split
      if (fast_merge_sct_)
        el += s;
      else
        oth += s;
    }
  }

  // Per-window curvefit background (replaces the external poles). T = 0 in
  // the fast range, so the basis factors are the plain E^(k/2 - 1).
  if (fast_cf_n_ > 0) {
    // Channels the nuclide does not have are skipped, matching fast_sweep's
    // compile-time elimination -- accumulating into all four unconditionally
    // wasted three FMAs per coefficient on an elastic-only nuclide.
    const double* c = BLK;   // curvefit is at the FRONT of this block
    double fac = 1.0 / E;
    const bool hf = fast_has_fis_, hd = fast_has_dis_;
    // Under the lazy split slot 0 is already elastic+other and slot 3 is the
    // elastic-only copy that evaluate_fast_elastic reads -- the flight must
    // NOT add it, or elastic would be counted twice.
    const bool ho = fast_has_oth_ && !fast_merge_sct_;
    for (int k = 0; k < fast_cf_n_; ++k, c += 4) {
      el += c[0] * fac;
      if (hf) fis += c[1] * fac;
      if (hd) dis += c[2] * fac;
      if (ho) oth += c[3] * fac;
      fac *= sqrtE;
    }
  }

  // Clip tiny negative excursions from the fit (thresholds, near-zero xs)
  el = (el > 0.0) ? el : 1.0e-5;  // floor, not clamp: see evaluate()
  fis = (fis > 0.0) ? fis : 0.0;
  dis = (dis > 0.0) ? dis : 0.0;
  oth = (oth > 0.0) ? oth : 0.0;

  double total = el + fis + dis + oth;
  return {total, dis + fis, fis};
}

//! Elastic alone, on demand. See wmp.h: the flight never computes this, so
//! nothing here is on the hot path -- it runs once per collision that needs
//! to sample a channel, and only in the fast segment.
double WindowedMultipole::evaluate_fast_elastic(double E) const
{
  // DEFENSIVE FLOOR. The low-E curvefit basis starts at 1/E, so E == 0 (or a
  // NaN arriving from a bad collision) turns every channel into NaN, which
  // then defeats both bounds checks in Nuclide::calculate_xs and segfaults.
  // Transport now kills such particles in Material::calculate_neutron_xs, but
  // this path is also reachable from tallies and the C API. Small POSITIVE
  // energies are fine and are NOT clamped -- the 1/v extrapolation is exact
  // down to at least 1e-11 eV (verified 2026-08-13).
  if (!(E > 0.0))
    E = (E_min_ > 0.0) ? E_min_ : 1.0e-11;

  // Unmerged (rewindowed) libraries never take the lazy path, but answer
  // correctly anyway; a nuclide with no fitted MT2 has no elastic to give.
  if (fast_ch_el_ < 0)
    return 1.0e-5;

  double sqrtE = std::sqrt(E);

  // Same O(1) window lookup as evaluate_fast's full-sweep path.
  int n_win = fw_off_.size() - 1;
  int cell = (int)((sqrtE - fast_bounds_[0]) * fast_cell_inv_du_);
  cell = std::min(std::max(cell, 0), (int)fast_cell_first_.size() - 1);
  int w = fast_cell_first_[cell];
  while (w < n_win - 1 && sqrtE >= fast_bounds_[w + 1])
    ++w;

  int k0 = fw_off_[w];
  int np = fw_off_[w + 1] - k0;

  // Pole sum over the MT2 column of the per-channel residues. This is the
  // SAME data the collision kernel already uses (fast_chan_xs), so the split
  // costs no extra storage -- but it is a different, COLD cache stream from
  // the flight SoA, which is precisely why it is not done during the flight.
  double s = 0.0;
  const int c = fast_ch_el_;
  if (E >= fast_mt_on_[c] && E <= fast_mt_off_[c]) {
    const std::complex<double>* cr =
      &fw_chan_res_[(size_t)k0 * fast_n_mts_ + (size_t)c * np];
    const std::complex<double>* rec = &fw_data_[(size_t)k0 * FW_STRIDE];
    for (int k = 0; k < np; ++k, rec += FW_STRIDE) {
      double dr = sqrtE - rec[0].real(), di = -rec[0].imag();
      double iv = 1.0 / (dr * dr + di * di);
      // phi = 1/(u - p), same convention as the partial-channel clamps
      s += cr[k].real() * (dr * iv) - cr[k].imag() * (-di * iv);
    }
  }

  // Elastic-only curvefit: slot 3 when merged (slot 0 is elastic+other),
  // slot 0 otherwise.
  if (fast_cf_n_ > 0) {
    const double* cf = &fw_soa_[fast_blk_off_[w]] + (fast_merge_sct_ ? 3 : 0);
    double fac = 1.0 / E;
    for (int k = 0; k < fast_cf_n_; ++k, cf += 4) {
      s += *cf * fac;
      fac *= sqrtE;
    }
  }

  // Same floor as the flight's scatter group: a zero elastic would make
  // collision channel selection divide by zero.
  return (s > 0.0) ? s : 1.0e-5;
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
  wmp_prof_add_sub(WPROF_COLL_CHAN, prof_t0);
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

  ctx.w = w;
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

//! Broadened pole sweep, specialised at COMPILE TIME on the record stride and
//! whether the nuclide is fissionable.
//!
//! 30 of the 32 nuclides in a PWR problem are non-fissionable and carried an
//! all-zero fission residue -- 2 of every 8 doubles in a pole record. Removing
//! it was tried once with a RUNTIME stride and an in-loop `if (fiss)` and cost
//! 20%: the compiler could no longer fold constant offsets or unroll. As
//! template parameters, both variants keep folded offsets and neither branches.
template <int STRIDE, bool FISS>
static inline void pole_sweep(const double* pol, int np, double sqrtE,
  double invE, double inv_xi, double& sig_s, double& sig_a, double& sig_f)
{
  const double sq = SQRT_PI * inv_xi;
#pragma omp simd reduction(+ : sig_s, sig_a, sig_f)
  for (int k = 0; k < np; ++k) {
    const double* r = pol + (size_t)k * STRIDE;
    double pr = r[0], pi = r[1];
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
    double br = (sqrtE + pr + hr) * invE;
    double bi = (pi + hi) * invE;
    sig_s += r[2] * br - r[3] * bi;
    sig_a += r[4] * br - r[5] * bi;
    if (FISS)
      sig_f += r[6] * br - r[7] * bi;
  }
}

std::tuple<double, double, double> WindowedMultipole::evaluate(
  double E, double sqrtkT) const
{
  // DEFENSIVE FLOOR. The low-E curvefit basis starts at 1/E, so E == 0 (or a
  // NaN arriving from a bad collision) turns every channel into NaN, which
  // then defeats both bounds checks in Nuclide::calculate_xs and segfaults.
  // Transport now kills such particles in Material::calculate_neutron_xs, but
  // this path is also reachable from tallies and the C API. Small POSITIVE
  // energies are fine and are NOT clamped -- the 1/v extrapolation is exact
  // down to at least 1e-11 eV (verified 2026-08-13).
  if (!(E > 0.0))
    E = (E_min_ > 0.0) ? E_min_ : 1.0e-11;

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
    // LOW-ENERGY SEGMENT: curvefit alone, no poles. The basis broadens in
    // closed form (image kernel included), so this is exact at temperature
    // and needs no low-E correction.
    // ONE compare, not a bool load AND a compare: lowe_E_max_ stays 0 when
    // there is no low-E segment, so E < 0 is never true. This is the first
    // test on every resolved lookup.
    if (E < lowe_E_max_) {
      array<double, MAX_POLY_COEFFICIENTS> fac;
      if (sqrtkT > 0.0) {
        broaden_wmp_polynomials(E, sqrt_awr_ / sqrtkT, lowe_n_, fac.data());
      } else {
        double t = invE;
        for (int k = 0; k < lowe_n_; ++k) { fac[k] = t; t *= sqrtE; }
      }
      double s0 = 0.0, a0 = 0.0, f0 = 0.0;
      for (int k = 0; k < lowe_n_; ++k) {
        s0 += lowe_cf_[3 * k + 0] * fac[k];
        a0 += lowe_cf_[3 * k + 1] * fac[k];
        f0 += lowe_cf_[3 * k + 2] * fac[k];
      }
      // SAME FLOOR AS THE RESOLVED BRANCH. scatter() reaches the inelastic
      // loop only when prob <= cutoff, and in the multipole path that loop
      // indexes rx->xs(micro) with index_grid = -1 (the resolved path poisons
      // it deliberately). prob = micro.elastic and cutoff = prn()*(total -
      // absorption) = prn()*sig_s, so a ZERO sig_s makes prob > cutoff false
      // and drops into that loop. Measured 2026-08-13: all 27 libraries are
      // safely positive here (min 0.60 b, Fe-57), but leaving it unguarded
      // makes correctness depend on the fit rather than on the code.
      s0 = (s0 > 0.0) ? s0 : 1.0e-5;
      a0 = (a0 > 0.0) ? a0 : 0.0;
      f0 = (f0 > 0.0) ? f0 : 0.0;
      return std::make_tuple(s0, a0, fissionable_ ? f0 : 0.0);
    }

    // Variable-width windows: O(1) two-level index (uniform coarse grid in
    // sqrt(E) + short forward walk; see wmp.h)
    int n_win = win_start_.size();
    int i_window;
    if (grid_uniform_) {
      // ARITHMETIC ONLY -- no table, no scan, no dependent load
      // Window 0 is the FIRST MESH WINDOW. (It used to be a low-E slot, with
      // the mesh at 1..N and a `1 +` here; the low-E region is now its own
      // segment handled above, so the offset would be off by one.)
      i_window = (int)((sqrtE - grid_u_lowe_) * grid_inv_spacing_);
      i_window = std::min(std::max(i_window, 0), n_win - 1);
    } else {
      int cell = (int)((sqrtE - win_u_lo_) * cell_inv_du_);
      cell = std::min(std::max(cell, 0), (int)cell_first_.size() - 1);
      i_window = cell_first_[cell];
      while (i_window < n_win - 1 && sqrtE >= win_[i_window].u_hi)
        ++i_window;
    }

    double sig_s = 0.0, sig_a = 0.0, sig_f = 0.0;
    // n_ext_ is a MEMBER: ext_data_.shape(1) reads the tensor's shape vector
    // off the heap on every lookup, to learn a per-nuclide constant that is
    // ZERO for every v3 library (the curvefit replaced the externals).
    const int n_ext = n_ext_;

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
    // ONE offset into this window's contiguous block, then a sequential run:
    // curvefit triplets first, then this window's pole records.
    const int ncf3 = cf_present_ ? cf_n_ * cf_ch_ : 0;
    const WinRec& wrec = win_[i_window];   // one line: bound, offset, np, flags
    const double* blk = &blk_[wrec.blk_off];
    const double* pol = blk + ncf3;
    const int np_w = wrec.np;
    int ps = win_start_[i_window], pe = win_end_[i_window];
    if (xi <= 0.0) {
      // one 64 B record per pole, sequential -> one cache line per pole
      const double* rec = pol;
      const int gstr = gs_stride_;
      for (int i = 0; i < np_w; ++i, rec += gstr) {
        double dr = sqrtE - rec[0], di = -rec[1];
        double minv = 1.0 / (dr * dr + di * di);
        double br = dr * minv, bi = -di * minv;
        sig_s += rec[2] * br - rec[3] * bi;
        sig_a += rec[4] * br - rec[5] * bi;
        if (fissionable_)
          sig_f += rec[6] * br - rec[7] * bi;
      }
    } else {
      // UNIFORM zpf8h SWEEP -- no near/far partition, no branch, one pass.
      // PROJECT.md §5.6 recorded "catastrophic cancellation in the sigma basis
      // at large |z| -- needs the analytic Laurent form". Measured 2026-07-29
      // that does NOT apply to the zpf8h path: its error in w(z) is ~1e-7 flat
      // from |z| = 4 to 50 and the sigma basis amplifies it by 1.0x (worst case
      // 1e-3 AT the resonance, which we already accept); the exact form holds
      // to 1e-15 at |z| = 100 against mpmath. The old note concerns naively
      // substituting the Laurent SERIES of w(z), a different construction.
      // So the partition bought speed, not accuracy, at the cost of a compare
      // per pole plus a compaction and a scatter pass.
      // one predictable branch on a member constant; both instantiations
      // carry compile-time offsets (see pole_sweep).
      if (fissionable_)
        pole_sweep<8, true>(pol, np_w, sqrtE, invE, inv_xi,
                            sig_s, sig_a, sig_f);
      else
        pole_sweep<6, false>(pol, np_w, sqrtE, invE, inv_xi,
                             sig_s, sig_a, sig_f);
    }
    // 1/v background pole: exact 0K value (broadening-invariant, any T).
    // Tested against THIS window's pointer range -- several windows can point
    // at it, and each of them owes its 1/v contribution.
    if (wrec.flags & WIN_HAS_ONEV) {
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

    // Per-window smooth-background CURVEFIT, ADDED to the pole sum (the
    // original WMP decomposition; see wmp.h). Broadened by
    // broaden_wmp_polynomials, which is the EXACT free-gas broadening of the
    // E^(i/2-1) basis (image kernel included) -- so this part needs no low-E
    // correction, and it replaces the 4 external Faddeeva calls per window.
    if (cf_present_) {
      array<double, MAX_POLY_COEFFICIENTS> fac;
      if (sqrtkT > 0.0) {
        broaden_wmp_polynomials(E, sqrt_awr_ / sqrtkT, cf_n_, fac.data());
      } else {
        double t = invE; // 0K basis: 1/E, 1/sqrtE, 1, sqrtE, ...
        for (int k = 0; k < cf_n_; ++k) { fac[k] = t; t *= sqrtE; }
      }
      // curvefit lives at the FRONT of this window's block (see wmp.h);
      // cf_blk_ was freed once the blocks were materialised at load.
      const double* c = blk;
      const int nch = cf_ch_;
      // fissionable_ hoisted OUT of the loop: it is a per-nuclide constant,
      // and testing it per coefficient per lookup is a branch in the hot path
      // for nothing.
      if (fissionable_) {
        for (int k = 0; k < cf_n_; ++k, c += nch) {
          sig_s += c[0] * fac[k];
          sig_a += c[1] * fac[k];
          sig_f += c[2] * fac[k];
        }
      } else {
        for (int k = 0; k < cf_n_; ++k, c += nch) {
          sig_s += c[0] * fac[k];
          sig_a += c[1] * fac[k];
        }
      }
    }

    // CLAMP. The fast path already clips its four channels; the resolved path
    // did not, so a window whose fit dips below zero produced a NEGATIVE
    // TOTAL, hence a negative distance to collision, surfacing far away as
    // "negative distance to a lattice boundary". Ar-40 reached -47 b at
    // 114 keV; several nuclides show harmless -1e-5 b excursions in capture.
    //
    // THIS HIDES A LIBRARY FAULT RATHER THAN FIXING IT (Ben, 2026-08-13:
    // clamp "so that we can test the efficiency before chasing the culprit").
    // Set WMP_CHECK=1 to report every clamp with nuclide and energy.
    // NOTE THE FORM: std::max(NaN, 0.0) returns NaN -- it evaluates
    // `NaN < 0.0` as false and hands back the first argument. A NaN then
    // reaches the transport as a NaN energy, and `(int)NaN` is INT_MIN, which
    // indexes the log-union grid out of bounds and segfaults in
    // Nuclide::calculate_xs. `(x > 0.0) ? x : 0.0` traps NaN and negatives
    // together.
    // TEMPORARY FLOOR, NOT A FIX (Ben, 2026-08-13: "let's make it 1e-5
    // instead of 0 and run the case, later we can go back and fix the
    // libraries"). Clamping to exactly 0 was not enough: a ZERO micro total
    // breaks collision sampling, which divides through it, and the resulting
    // garbage energy makes `log(E)` -inf so `(int)` gives INT_MIN and
    // Material::calculate_neutron_xs indexes grid_index[] out of bounds.
    //
    // Elastic carries the floor so the total is always > 0; the other
    // channels clamp to zero. THREE LIBRARIES NEED REPAIR: Al-27, Cr-50 and
    // Ar-40 evaluate non-positive somewhere (12, 6 and 140 points of an
    // 80000-point sweep at 294 K). The non-negativity constraint was only
    // ever applied to the RESOLVED phase-3 fit, never to the fast segment.
    sig_s = (sig_s > 0.0) ? sig_s : 1.0e-5;
    sig_a = (sig_a > 0.0) ? sig_a : 0.0;
    sig_f = (sig_f > 0.0) ? sig_f : 0.0;
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
  // Mirror the bounds into the Nuclide so the per-lookup range tests in
  // calculate_xs need no dereference of the multipole object (see nuclide.h).
  nuc->mp_present_ = true;
  nuc->mp_fast_present_ = nuc->multipole_->fast_present_;
  // The gate opens at the bottom of the LOW-E segment when one is present:
  // E_min_ is the resolved segment's lower bound, which is the low-E seam.
  nuc->mp_E_min_ = nuc->multipole_->lowe_present_
                     ? std::min(nuc->multipole_->E_min_,
                                nuc->multipole_->lowe_E_min_)
                     : nuc->multipole_->E_min_;
  nuc->mp_E_max_ = nuc->multipole_->E_max_;
  nuc->mp_fast_E_max_ = nuc->multipole_->fast_E_max_;
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
