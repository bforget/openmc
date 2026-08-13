#ifndef OPENMC_WMP_H
#define OPENMC_WMP_H

#include "hdf5.h"
#include "openmc/tensor.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <tuple>

#include "openmc/array.h"
#include "openmc/vector.h"

namespace openmc {

//========================================================================
// Constants
//========================================================================

// Constants that determine which value to access
constexpr int MP_EA {0}; // Pole
constexpr int MP_RS {1}; // Residue scattering
constexpr int MP_RA {2}; // Residue absorption
constexpr int MP_RF {3}; // Residue fission

// Polynomial fit indices
constexpr int FIT_S {0}; // Scattering
constexpr int FIT_A {1}; // Absorption
constexpr int FIT_F {2}; // Fission

// Multipole HDF5 file version
constexpr array<int, 2> WMP_VERSION {1, 1};

//========================================================================
// Windowed multipole data
//========================================================================

class WindowedMultipole {
public:
  // Types
  struct WindowInfo {
    int index_start;   // Index of starting pole
    int index_end;     // Index of ending pole
    bool broaden_poly; // Whether to broaden polynomial curvefit
  };

  // Constructors, destructors
  WindowedMultipole() = default;
  WindowedMultipole(hid_t group);

  //! Read a global-pole-format segment (poles/residues/windows/externals)
  //! into the resolved-segment members.
  void read_global_segment(hid_t group);

  //! Read the fast segment and build the collapsed transport residues.
  void read_fast_segment(hid_t group);

  //! Read the LOW-ENERGY segment: a curvefit-only region below the resolved
  //! range, carrying no poles at all. Its basis is the same E^(i/2-1) one the
  //! window curvefits use, so broaden_wmp_polynomials broadens it exactly
  //! (image kernel included) and it needs no low-E correction.
  void read_lowe_segment(hid_t group);

  // Methods

  //! \brief Evaluate the windowed multipole equations for cross sections in the
  //! resolved resonance regions
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of elastic scattering, absorption, and fission cross
  //! sections in [b]
  std::tuple<double, double, double> evaluate(double E, double sqrtkT) const;

  //! \brief Pull this nuclide's window block toward cache for energy E.
  //!
  //! MEMORY-LEVEL PARALLELISM. A material XS evaluation calls calculate_xs for
  //! EVERY nuclide it contains (3-18 here) at the SAME energy, and each call
  //! walks an independent chain of dependent loads: window index -> WinRec ->
  //! block. Serialised behind per-nuclide function calls, the out-of-order
  //! window cannot span them, so 18 miss latencies are paid one after another.
  //! Issuing them from a TIGHT loop first lets the hardware overlap them --
  //! this vectorises the WAITING, not the arithmetic, which is the right
  //! target for a latency-bound kernel (measured 763 cyc/call against ~100 cyc
  //! of actual work). Cheap and side-effect free: address arithmetic plus a
  //! prefetch hint, no evaluation.
  void prefetch(double E) const
  {
    if (!global_format_ || win_.empty())
      return;
    double sqrtE = std::sqrt(E);
    int n_win = static_cast<int>(win_.size());
    int i;
    if (grid_uniform_) {
      // window 0 is the first mesh window; see evaluate()
      i = static_cast<int>((sqrtE - grid_u_lowe_) * grid_inv_spacing_);
      i = std::min(std::max(i, 0), n_win - 1);
    } else {
      int cell = static_cast<int>((sqrtE - win_u_lo_) * cell_inv_du_);
      cell = std::min(std::max(cell, 0), static_cast<int>(cell_first_.size()) - 1);
      i = cell_first_[cell];
      while (i < n_win - 1 && sqrtE >= win_[i].u_hi)
        ++i;
    }
    // this load is what the next one depends on; across nuclides these are
    // independent, so a tight loop overlaps them
    const char* b = reinterpret_cast<const char*>(blk_.data() + win_[i].blk_off);
    __builtin_prefetch(b, 0, 3);
    __builtin_prefetch(b + 64, 0, 3);   // blocks commonly span two lines
  }

  //! \brief Evaluates the windowed multipole equations for the derivative of
  //! cross sections in the resolved resonance regions with respect to
  //! temperature.
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of derivatives of elastic scattering, absorption, and
  //!         fission cross sections in [b/K]
  std::tuple<double, double, double> evaluate_deriv(
    double E, double sqrtkT) const;

  //! \brief Evaluate the fast-segment (URR+fast) pole representation.
  //!
  //! T=0 pole sum over all fitted channels, collapsed at load time to the
  //! transport quantities the FLIGHT needs; threshold clamps honored
  //! (channels that turn on/off inside a window are evaluated individually
  //! and gated).
  //!
  //! ELASTIC IS NOT RETURNED. The flight kernel needs only three independent
  //! numbers -- total, absorption, fission -- so elastic and other-scatter
  //! are summed into ONE "scatter" group at load time (see fast_merge_sct_),
  //! dropping a residue pair per pole. Elastic is split back out only when a
  //! collision actually samples a channel, via evaluate_fast_elastic().
  //!
  //! \param E Incident neutron energy in [eV], seam < E <= fast_E_max_
  //! \return Tuple of total, absorption (fission + disappearance), and
  //!         fission cross sections in [b]
  std::tuple<double, double, double> evaluate_fast(double E) const;

  //! \brief Elastic (MT2) alone in the fast segment -- the LAZY half of
  //!        evaluate_fast, resolved on demand at a collision.
  //!
  //! Rebuilt from the per-channel residues (fw_chan_res_, which the collision
  //! kernel already keeps in full) plus the elastic-only curvefit slot, so it
  //! needs no storage the library did not already carry. Taking elastic from
  //! the ACE table instead would mix sources -- total from poles, elastic
  //! from ACE -- and turn our fast-range fit error into spurious inelastic
  //! events, since OpenMC derives inelastic as total - elastic - absorption.
  //!
  //! \param E Incident neutron energy in [eV], seam < E <= fast_E_max_
  //! \return Elastic scattering cross section in [b]
  double evaluate_fast_elastic(double E) const;

  //! \brief Evaluate every fitted fast channel at E (collision kernel).
  //!
  //! One set of pole reciprocals; fills out[c] with the pole cross section
  //! of channel c (0 outside its clamp). Used only at collisions to select
  //! the scattering/reaction channel from poles rather than ACE tables.
  //!
  //! \param E   Incident neutron energy [eV], seam < E <= fast_E_max_
  //! \param out Output buffer, size >= fast_n_mts_
  void evaluate_fast_channels(double E, double* out) const;

  //! Sequential per-channel evaluation for collision channel selection:
  //! scatter() walks channels accumulating probability until the sampled
  //! cutoff is crossed, so pre-evaluating ALL channels (the full n_mts x np
  //! residue block, ~13 KB for U-235) wastes most of the work. begin()
  //! computes the shared pole reciprocals once; channel_xs() then evaluates
  //! ONE channel on demand (contiguous np-length dot product in the
  //! channel-major residue layout).
  struct FastChanCtx {
    int k0;          //!< first flattened pole of the window
    int np;          //!< number of active poles
    int w;           //!< window index, for the per-MT curvefit background
    double pr[96];   //!< Re of pole reciprocals 1/(u - p)
    double pi[96];   //!< Im of pole reciprocals
  };
  void fast_chan_begin(double E, FastChanCtx& ctx) const;
  double fast_chan_xs(const FastChanCtx& ctx, double E, int c) const
  {
    if (E < fast_mt_on_[c] || E > fast_mt_off_[c])
      return 0.0;
    const std::complex<double>* cr =
      &fw_chan_res_[(size_t)ctx.k0 * fast_n_mts_ + (size_t)c * ctx.np];
    double s = 0.0;
    for (int k = 0; k < ctx.np; ++k)
      s += cr[k].real() * ctx.pr[k] - cr[k].imag() * ctx.pi[k];
    // PER-MT CURVEFIT BACKGROUND -- REQUIRED, not an refinement. v3 replaced
    // the fast external poles with a per-window curvefit that carries most of
    // the fast-range magnitude (87% of U-238's fast bytes). A pole-only
    // channel value is therefore far too small, while scatter()'s cutoff is
    // normalised on poles+curvefit; the selection walk then under-accumulates
    // and falls through to the LAST inelastic level. Measured 2026-08-13:
    // that alone was ~+390 pcm on coreI.
    if (fast_cf_n_ > 0 && !fw_chan_cf_.empty()) {
      const double* cf =
        &fw_chan_cf_[((size_t)ctx.w * fast_cf_n_) * fast_n_mts_ + c];
      const double sqrtE = std::sqrt(E);
      double fac = 1.0 / E;
      for (int k = 0; k < fast_cf_n_; ++k, cf += fast_n_mts_) {
        s += *cf * fac;
        fac *= sqrtE;
      }
    }
    return (s > 0.0) ? s : 0.0;
  }

  //! Channel index for a given MT in the fast segment, or -1 if not fit.
  int fast_channel_of_mt(int mt) const
  {
    return (mt >= 0 && mt < 900) ? fast_mt_to_chan_[mt] : -1;
  }

  // Data members
  std::string name_;               //!< Name of nuclide
  //! LOW-ENERGY segment (optional). Below lowe_E_max_ the cross section is
  //! this curvefit ALONE -- no poles -- which is why it is a segment in its
  //! own right rather than window 0 of the resolved mesh (Ben, 2026-08-13).
  //! Per-MT data kept alongside the collapsed triplet: the flight kernel
  //! only needs (scatter, absorption, fission), but tallies and collision
  //! sampling want the individual channels (Ben, 2026-08-13).
  vector<int> mts_;                              //!< resolved channel MTs
  tensor::Tensor<std::complex<double>> res_mt_;  //!< (n_pool, n_ch)
  tensor::Tensor<double> cf_mt_;                 //!< (n_win, n_cf, n_ch)

  //! Fast-segment per-window curvefit, collapsed to four slots and
  //! interleaved per window. When fast_merge_sct_ the slot order is
  //! [scatter=elastic+other, fission, disappearance, ELASTIC-ONLY]: the
  //! flight reads slots 0..2 and evaluate_fast_elastic() reads slot 3, so
  //! the lazy split costs no extra array. Otherwise (dormant rewindowed
  //! path) it is the historical [elastic, fission, disappearance, other].
  //! Fast curvefit, INTERLEAVED into the same per-window block as the poles.
  //! A separate array is a second cache-line stream for every fast lookup,
  //! which costs most in the SPARSE windows -- U-238 averages 0.57
  //! poles/window, so its pole block is one line and the curvefit was a
  //! second. Contiguity is the only thing that has ever improved this
  //! evaluator's runtime, so the curvefit now sits at the FRONT of the
  //! window's block, exactly as the resolved path does.
  //!   block(w) = [4 * fast_cf_n_ curvefit][fast_soa_stride_ * np poles]
  int fast_cf_n_ {0};
  vector<size_t> fast_blk_off_;     //!< block start per window
  vector<uint8_t> fast_has_part_;   //!< 1 if the window has partial channels
  vector<double> fast_cf_tmp_;      //!< load-time only, freed after interleave

  //! LAZY ELASTIC. The flight needs total/absorption/fission only; elastic is
  //! needed just when a collision samples a channel. Summing elastic and
  //! other-scatter into one group drops a residue PAIR from every pole:
  //! stride 8->6 for the 25 non-fissionable nuclides and 10->8 for U-235/238,
  //! 24.8% off the fast pole block, which matters because this evaluator is
  //! memory-bound. Disabled for the rewindowed flight path (fl_*), whose
  //! blocks are built against the historical four-group slot order.
  bool fast_merge_sct_ {false};
  int fast_ch_el_ {-1};             //!< channel index of MT2, -1 if not fit

  int n_ext_ {0};   //!< externals per window; 0 for every v3 library
  int n_win_ {0};   //!< resolved window count

  bool lowe_present_ {false};
  double lowe_E_min_ {0.0};        //!< Lower bound of the low-E segment [eV]
  double lowe_E_max_ {0.0};        //!< Upper bound of the low-E segment [eV]
  int lowe_n_ {0};                 //!< Number of curvefit terms
  //! [n_cf][3] collapsed to the transport triplet (scatter, absorb, fission)
  vector<double> lowe_cf_;

  double E_min_;                   //!< Minimum energy in [eV]
  double E_max_;                   //!< Maximum energy in [eV]
  double sqrt_awr_;                //!< Square root of atomic weight ratio
  double inv_spacing_;             //!< 1 / spacing in sqrt(E) space
  int fit_order_;                  //!< Order of the fit
  bool fissionable_;               //!< Is the nuclide fissionable?
  vector<WindowInfo> window_info_; // Information about a window
  tensor::Tensor<double>
    curvefit_; // Curve fit coefficients (window, poly order, reaction)
  tensor::Tensor<std::complex<double>> data_; //!< Poles and residues

  // Global-pole format (AAA-fit libraries, 2026): global pole/residue
  // table + contiguous per-window pointer ranges + 4 external poles per
  // window replacing the polynomial curvefit. sigma_mt(E) =
  // sum Re[r/(sqrt(E)-p)] over the owning window's range + externals;
  // at temperature each pole broadens analytically as
  // (1/E) Re[r ((u+p) + p^2 (i sqrt(pi)/xi) w((p-u)/xi))], with the 1/v
  // pole (Re p = 0) evaluated unbroadened (exactly invariant).
  // Below cf_lowe_xi * xi(T_max) the library carries only the 1/v pole
  // (broadening-invariant) plus the per-window curvefit (closed-form
  // broadening WITH the image kernel), so the one-kernel Faddeeva form is
  // never evaluated where it would be invalid and NO runtime low-E
  // correction exists. TEST-PHASE: evaluate_deriv not implemented.
  bool global_format_ {false};   //!< New-format library?
  vector<double> sqrtE_bounds_;  //!< Window boundaries in sqrt(E) [n_win+1]
  vector<int> win_start_;        //!< First pool-pole index per window
  vector<int> win_end_;          //!< One-past-last pool-pole index
  // Two-level O(1) window lookup over the variable-width windows: a uniform
  // coarse grid in sqrt(E) (4 cells per window on average) stores the first
  // window overlapping each cell; a lookup is one multiply + a short forward
  // walk (bounded by windows-per-cell, ~1). Replaces the binary search.
  //! UNIFORM-GRID fast index. When the library reports grid_uniform, the
  //! window is located by ARITHMETIC ALONE -- one compare, one FMA, NO
  //! dependent load:
  //!     i = (u < u_lowe) ? 0 : 1 + (u - u_lowe) * inv_spacing
  //! versus the table path's `cell_first_[cell]` load followed by a scan over
  //! per-window bounds (two dependent loads and a branch). Load LATENCY does
  //! not improve with cache size, so this transfers to any CPU.
  //! The grid is one low-E window (below u_lowe it carries only the 1/v pole
  //! plus the curvefit, both exactly broadenable) then uniform spacing above.
  bool grid_uniform_ {false};
  double grid_u_lowe_ {0.0};
  double grid_inv_spacing_ {0.0};

  //! (A DYADIC index lived here until 2026-08-13. v3 measured a dyadic mesh
  //! against a uniform one on Cr-50 and U-238 and it LOST on the
  //! 1/E-flux-weighted cost metric: refinement is driven by poles per window,
  //! so it coarsens exactly the low-energy region carrying most of the flux.
  //! Every library v3 produces is uniform, so the extra branch was removed
  //! from evaluate() and prefetch().)
  vector<int> cell_first_;       //!< First window overlapping each grid cell
  double cell_inv_du_ {0.0};     //!< 1 / coarse-cell width in sqrt(E)

  // Fast segment (URR + fast range, seam..fast_E_max_, T=0 pole sum).
  // The HDF5 stores per-channel residues (n_pool x n_mts) + 4 external
  // poles/window + per-channel threshold clamps [mt_E_on, mt_E_off].
  // At load, each window's pointer range AND its 4 externals are flattened
  // into one pole array with residues COLLAPSED to the four transport
  // quantities (elastic / fission / disappearance / other-scattering) over
  // the channels fully active across that window. Channels that turn on or
  // off strictly inside a window (boundary-merge tolerance) are kept as
  // per-window "partials" and evaluated individually behind their clamp.
  bool fast_present_ {false};
  double fast_E_max_ {0.0};       //!< Upper energy of fast segment [eV]
  vector<double> fast_bounds_;    //!< Window bounds in sqrt(E) [n_fwin+1]
  vector<int> fast_cell_first_;   //!< O(1) index over fast windows
  double fast_cell_inv_du_ {0.0};
  vector<int> fw_off_;            //!< Flattened pole offsets [n_fwin+1]
  // Interleaved per-pole record (single memory stream per lookup):
  // [pole, r_elastic, r_fission, r_disappearance, r_other] x n_flat
  static constexpr int FW_STRIDE = 5;
  vector<std::complex<double>> fw_data_;
  // Real second-order-section flight-path arrays (SoA, plain double, so the
  // hot loop vectorizes -- std::complex defeats the auto-vectorizer). The
  // fast range is NOT Doppler-broadened, so sigma = Re[sum r/(sqrtE-p)] is
  // exact real arithmetic: each pole -> (n1*u + n0)/((u-alpha)^2 + beta^2),
  // n1=Re(r), n0=-(Re(r)Re(p)+Im(r)Im(p)). The denominator is stored and
  // evaluated FACTORED, never expanded to u^2 + d1*u + d0: for a near-real
  // pole (beta ~ 0) at large u the expanded form cancels catastrophically
  // (u^2, d1*u, d0 each ~1e6 cancelling to ~1e-11) and can round to exactly
  // 0.0 -> 1/0 = inf -> NaN total -> "did not sample any nuclide" abort.
  // Built from fw_data_ at load. Groups: 0 elastic, 1 fission,
  // 2 disappearance, 3 other-scatter.
  // BLOCK-INTERLEAVED single array: window w's record starts at 10*fw_off_[w]
  // with layout [alpha(np) | beta2(np) | n1_el(np) | n0_el(np) | n1_fis |
  // n0_fis | n1_dis | n0_dis | n1_oth | n0_oth] -- ONE contiguous stream per
  // lookup (a 10-separate-vectors SoA made every lookup touch 10 scattered
  // regions; transport jumps windows randomly so all 10 were cache-cold).
  // Each sub-array is still unit-stride, so the sweep still vectorizes.
  //! Which collapsed channels this nuclide's fast segment actually uses.
  //! Measured over the 32-nuclide library: dis is empty for ALL, fis for 30.
  bool fast_has_fis_ {false}, fast_has_dis_ {false}, fast_has_oth_ {false};
  int fast_soa_stride_ {10};       //!< 2 + 2*(channels in use)
  vector<double> fw_soa_; //!< [10 * n_flat] per-window interleaved blocks
  vector<double> fw_gn1_[4], fw_gn0_[4];    //!< [n_flat] per-group numerators
  struct FastPartial {
    double E_on;  //!< Clamp: zero below this energy [eV]
    double E_off; //!< Clamp: zero above this energy [eV]
    int kind;     //!< 0 other-scatter, 1 fission, 2 disappearance, 3 elastic
    int res_off;  //!< Offset into fp_res_ (window pole count entries)
  };
  // ---------------------------------------------------------------------
  // REWINDOWED FLIGHT PATH (fast segment). Optional: present only if the
  // library carries the flight_* datasets, otherwise the original full-pole
  // sweep above is used unchanged.
  //
  // The fast range costs 1927.9 cyc/lookup against ACE's 1075.1 while the
  // resolved range sits at parity, and the cause is 16.5 poles per lookup
  // against the resolved range's 1.55 -- a long contiguous stream, so unlike
  // the resolved evaluator (latency-bound on one pole, immune to byte cuts)
  // it should respond to touching fewer poles. Each original window is
  // subdivided and the far poles inside each piece are replaced by a
  // per-window curvefit. Measured 0.53x the bytes for <1% error.
  //
  // Two consequences worth knowing:
  //  * Threshold energies are sub-window boundaries, so every channel is
  //    uniformly on or off across a piece: NO clamped partials survive here,
  //    which is why this path needs neither phi[] nor any fw_data_ read.
  //  * The curvefit basis is the WINDOW-LOCAL t = (u-lo)/(hi-lo), NOT the
  //    resolved range's u^(i-2). Sub-windows are narrow (du/u ~ 5e-4) and the
  //    power columns go collinear across one, which silently truncates the
  //    fit. Nothing here is Doppler-broadened, so no broadenable basis is
  //    needed. Evaluated by Horner.
  //
  // The COLLISION kernel is deliberately left on the full pole set and all
  // MTs (1.3% of evaluator cost). The flight total and the sum of the
  // collision per-MT values therefore come from different representations
  // and may disagree by up to the fit tolerance.
  //! One sub-window's metadata, 32 B. u_hi is carried HERE rather than read
  //! from a separate bounds array: the search must load this record anyway, so
  //! folding the upper edge in makes the boundary check free and removes a
  //! whole array from the per-lookup dependent chain (4 touches -> 3).
  struct FlWin {
    double u_lo;    //!< Sub-window lower edge in sqrt(E)
    double u_hi;    //!< Upper edge; the advance test reads it from this load
    double inv_w;   //!< 1/(u_hi-u_lo): maps u to the local curvefit variable
    uint32_t off;   //!< Offset of this window's block in fl_blk_ (doubles)
    uint32_t np;    //!< Retained poles in this window
  };
  //! Threshold/window-bound delimited SEGMENT, inside which the sub-windows
  //! are UNIFORM in sqrt(E) so one is found by arithmetic instead of a search.
  //! The old index walked a 4*n_sub cell table into a bounds array with a
  //! linear advance -- two large cold arrays and a data-dependent loop, and
  //! rewindowing had made it worse by creating 6.5x more windows. Measured at
  //! 39% of the fast kernel, more than the pole sweep itself.
  struct FlSeg {
    double u_lo;
    double u_hi;
    double inv_du; //!< n_win / (u_hi - u_lo)
    int32_t win0;  //!< First sub-window index of this segment
    int32_t nwin;
  };
  vector<FlSeg> fl_seg_;
  vector<int32_t> fl_cell_seg_; //!< uniform cell -> first segment in that cell
  double fl_u0_ {0.0};
  double fl_seg_cell_inv_du_ {0.0};
  bool fl_have_ {false};
  int fl_order_ {0};              //!< Curvefit coefficients per channel
  int fl_nch_ {0};                //!< Channels stored (channel specialisation)
  double fl_cell_inv_du_ {0.0};
  vector<double> fl_bounds_;      //!< [n_sub+1] sub-window edges in sqrt(E)
  vector<int> fl_cell_first_;     //!< Uniform-cell -> first sub-window index
  vector<FlWin> fl_win_;

  //! BLOCK-HEADER layout. fl_win_ is a separate array fetched only to find the
  //! block, so it is a whole link in the per-lookup dependent chain
  //! (cell -> record -> block). Moving those 4 doubles to the head of the block
  //! itself and storing BLOCK OFFSETS in the cell table makes it
  //! cell -> block: the first line carries the metadata AND the start of the
  //! curvefit. No extra data is stored -- the metadata moves rather than
  //! duplicates. Header is [u_lo, u_hi, inv_w, next_off]; the pole count is
  //! derived from next_off, so it needs no slot of its own.
  static constexpr int FL_HDR = 4;
  vector<uint32_t> fl_cell_blk_; //!< uniform cell -> block offset (doubles)
  vector<double> fl_blk_;         //!< [curvefit | pole SoA] blocks, 64B aligned
  double* fl_base_ {nullptr};     //!< Aligned base pointer into fl_blk_
  vector<vector<FastPartial>> fast_partials_; //!< Per-window partial channels
  vector<std::complex<double>> fp_res_;       //!< Partial-channel residues

  // Per-channel fast residues for the COLLISION kernel (channel selection).
  // Kept alongside the collapsed transport residues; read only at collisions,
  // never on the flight path. fw_chan_res_ is laid out [flat_pole * n_mts +
  // channel], flat_pole in the same order as fw_data_ (pool poles then the 4
  // externals per window). fast_mts_ gives the MT of each channel; the clamps
  // gate each channel to [on, off].
  //! Per-MT curvefit for the collision kernel, [n_win][fast_cf_n_][n_mts].
  //! The flight collapses the curvefit into four transport groups; channel
  //! selection needs it per MT, exactly as fw_chan_res_ keeps the residues.
  //! Cold: touched only at collisions.
  vector<double> fw_chan_cf_;
  int fast_n_mts_ {0};
  vector<int> fast_mts_;                       //!< MT number per channel
  vector<double> fast_mt_on_;                  //!< Per-channel turn-on [eV]
  vector<double> fast_mt_off_;                 //!< Per-channel turn-off [eV]
  vector<std::complex<double>> fw_chan_res_;   //!< [n_flat * n_mts] residues
  array<int, 900> fast_mt_to_chan_;            //!< MT -> channel index, -1 none

  // Pool: [pole, r_elastic, r_absorption, r_fission] like data_ columns
  tensor::Tensor<std::complex<double>> gdata_; //!< (n_pool, 4)
  // Externals: 4 per window, same column convention
  tensor::Tensor<std::complex<double>> ext_data_; //!< (n_win, 4, 4)

  // Branch-free resolved pool kernel (SoA plain double, so the hot loop
  // vectorizes -- std::complex + the runtime |z| branch defeated the
  // auto-vectorizer). Every pool pole takes ONE uniform zpf8h Faddeeva path
  // (no |z| test): the pointer ranges are sized for kernel reach at
  // T_max=3000K, where ~100% of actinide pool poles are within 5*xi, so the
  // old |z|>=5 asymptotic branch was a low-T-only optimization; running
  // zpf8h everywhere costs a small low-E/low-T accuracy gray zone
  // (5<=|z|<~40, ~1.26% U-235 elastic 294K) accepted for a clean
  // vectorizable baseline (Ben, 2026-07-14). Parallel to gdata_ rows.
  // Resonance poles have Im(p)=beta>0, so z is always upper-half-plane and
  // the faddeeva() conjugation branch never fires. The 1/v pole (Re p=0,
  // broadening-invariant) has its SoA residues ZEROED (the vector loop adds
  // nothing for it) and its exact 0K contribution is added separately;
  // externals (giant beta, |z| up to >100 -> zpf8h cancellation) keep the
  // scalar branchy accumulate.
  // POLE RECORDS ARE AoS, 8 doubles = 64 B = EXACTLY ONE CACHE LINE:
  //   [pr, pi, sr, si, ar, ai, fr, fi]
  // This was eight separate SoA vectors. SoA is right when a window streams
  // MANY poles -- the multiple streams amortise and the loop vectorises. But
  // the library work drove poles/lookup down to 1-2, and in THAT regime SoA
  // is the worst possible layout: evaluating ONE pole touched EIGHT distinct
  // memory regions, i.e. up to 8 cache misses to gather 8 doubles. Measured
  // 2026-07-29 with WMP_PROF_ENABLED: 830.6 cyc per calculate_xs call against
  // ACE's 584.3 in the same binary on identical call counts, for arithmetic
  // worth ~100 cyc -- the signature of a scattered working set (~15 lines per
  // lookup), and consistent with the microbenchmark reading 40-125 ns
  // hot-cache while the real run reads ~237 ns. The vectorised passes are
  // unaffected: a window's poles are copied ONCE, sequentially, into local
  // SoA scratch (np cache lines, one per pole) and the passes run unchanged.
  //! Pole-record stride: 8 doubles when fissionable, 6 otherwise (30 of 32
  //! nuclides in a PWR problem carried an all-zero fission residue).
  int gs_stride_ {8};
  int cf_ch_ {3};                  //!< curvefit channels stored
  //! Below this many poles in a window, take the SCALAR path: vectorising a
  //! handful of elements does not pay for the AoS->SoA shuffle it needs, and
  //! the library now averages 1-2 poles per lookup.
  static constexpr int SIMD_MIN_POLES = 4;
  vector<double> gs_aos_;                         //!< [n_pool * GS_STRIDE]

  // PER-WINDOW CONTIGUOUS BLOCKS -- what a lookup actually reads, in the
  // order it reads it:
  //     [ curvefit (cf_n_ x 3 s,a,f) ][ pole records (np x 8) ]
  // Built ONCE at load from gs_aos_ + the pointer ranges, so a lookup is a
  // single offset and then one sequential run of memory: no indirection, no
  // per-evaluation gather, no AoS->SoA shuffling. Context poles shared with
  // a neighbouring window are duplicated, which is the whole point -- the
  // file keeps storing each pole once (windows point in by contiguous index
  // range, per the format rule), and the DUPLICATION IS A LOAD-TIME
  // DECISION, invisible to the format. Cost is small: sum(np) is ~2.5x
  // n_win, so ~1.5 MB of pole records for the 32-nuclide library.
  vector<double> blk_;              //!< all per-window blocks, back to back

  //! ONE 16-byte record per window: everything the hot path needs to locate
  //! and size a window, in a single cache line (4 windows per line).
  //!
  //! This was FOUR separate arrays -- cell_first_, sqrtE_bounds_,
  //! win_blk_off_, win_np_ -- plus win_start_/win_end_ for the 1/v membership
  //! test, so reaching one ~96 B block could touch up to 5 lines of METADATA.
  //! Packing them is the same fix that made the per-window pole blocks worth
  //! +18%, applied to the half that was left as SoA. The 1/v test becomes a
  //! flag bit, which removes win_start_/win_end_ from the hot path entirely.
  struct WinRec {
    double u_hi;        //!< upper window bound in sqrt(E) (the scan compares this)
    uint32_t blk_off;   //!< offset of this window's block in blk_
    uint16_t np;        //!< pole records in the block (<= 64)
    uint16_t flags;     //!< bit 0: pointer range contains the 1/v pole
  };
  static constexpr uint16_t WIN_HAS_ONEV = 1u;
  vector<WinRec> win_;              //!< [n_win]
  double win_u_lo_ {0.0};           //!< sqrtE_bounds_.front(), hoisted
  bool onev_present_ {false};                     //!< 1/v background pole?
  int onev_idx_ {-1};                             //!< its POOL INDEX. The 1/v
  // pole can lie inside the pointer range of SEVERAL windows, so membership
  // must be tested per lookup: caching a single "owning window" silently
  // dropped its contribution (and halved 1/v absorption) in every other
  // window that pointed at it.
  std::complex<double> onev_p_, onev_rs_, onev_ra_, onev_rf_; //!< its data

  // Per-window smooth-background CURVEFIT (global format): every window adds
  // sum_i c_i E^(i/2-1) to its pole sum -- the original WMP decomposition.
  // Replaces the 4 external poles that used to carry the background: same job
  // for ~5 real terms instead of 4 Faddeeva calls, and broaden_wmp_polynomials
  // is the EXACT free-gas broadening of this basis (image kernel included),
  // so the smooth part of sigma needs no low-E correction at all.
  bool cf_present_ {false};
  int cf_n_ {0};                                   //!< terms (order+1)
  // Collapsed transport coefficients, INTERLEAVED [n_win][cf_n_][3] as
  // (s,a,f) triplets. Was three separate flat arrays, i.e. three scattered
  // cache lines per lookup; one window's block is now contiguous (120 B at
  // cf_n_=5 = 2 lines) and the whole curvefit is one prefetchable stream.
  vector<double> cf_blk_;                          //!< scatter/absorption/fission

  // Constant data
  static constexpr int MAX_POLY_COEFFICIENTS =
    11; //!< Max order of polynomial fit plus one
};

//========================================================================
// Non-member functions
//========================================================================

//! Check to make sure WMP library data version matches
//!
//! \param[in] file  HDF5 file object
void check_wmp_version(hid_t file);

//! \brief Checks for the existence of a multipole library in the directory and
//! loads it
//!
//! \param[in] i_nuclide  Index in global nuclides array
void read_multipole_data(int i_nuclide);

//==============================================================================
//! Doppler broadens the windowed multipole curvefit.
//!
//! The curvefit is a polynomial of the form a/E + b/sqrt(E) + c + d sqrt(E)...
//!
//! \param E       The energy to evaluate the broadening at
//! \param dopp    sqrt(atomic weight ratio / kT) with kT given in eV
//! \param n       The number of components to the polynomial
//! \param factors The output leading coefficient
//==============================================================================

extern "C" void broaden_wmp_polynomials(
  double E, double dopp, int n, double factors[]);

} // namespace openmc

#endif // OPENMC_WMP_H
