#ifndef OPENMC_WMP_H
#define OPENMC_WMP_H

#include "hdf5.h"
#include "openmc/tensor.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring> // for std::memcpy
#include <string>
#include <tuple>

#include "openmc/array.h"
#include "openmc/constants.h"
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

//! Current multipole HDF5 file version (segmented global-pole format)
constexpr array<int, 2> WMP_VERSION {2, 0};

//! Major version of the legacy per-window format, which remains readable
constexpr int WMP_VERSION_MAJOR_LEGACY {1};

//! Number of depletion reactions served per lookup (see DEPLETION_RX)
constexpr int WMP_N_DEP = DEPLETION_RX.size();

//========================================================================
// Windowed multipole data
//========================================================================

//! Windowed multipole representation of a nuclide's cross sections.
//!
//! Two file formats are supported. Version 1 stores poles per window with a
//! polynomial curvefit and is evaluated by the original algorithm. Version 2
//! stores a global pole table with per-window pointer ranges and splits the
//! energy range into up to three segments:
//!
//!  - a low-energy segment carrying only a curvefit in the E^(k/2-1) basis,
//!    which Doppler broadens in closed form;
//!  - a resolved segment of poles plus a per-window curvefit, broadened
//!    analytically with the Faddeeva function; and
//!  - a fast segment covering the unresolved and fast range, evaluated as an
//!    unbroadened pole sum with per-channel threshold clamps.
//!
//! Version 2 residues are stored per reaction channel (MT) and collapsed at
//! load time into the groups transport needs (scattering, absorption,
//! fission). The per-MT data are kept so depletion reaction rates and
//! collision channel selection can be served from the same representation.
class WindowedMultipole {
public:
  // Constants
  static constexpr int MAX_POLY_COEFFICIENTS =
    11; //!< Max order of polynomial fit plus one
  static constexpr int MAX_WINDOW_POLES =
    256; //!< Max poles per resolved window (scratch bound)
  static constexpr int FAST_MAX_POLES =
    96; //!< Max poles (including externals) per fast window
  static constexpr int MAX_MT = 900; //!< Size of the MT -> channel table
  static constexpr int FAST_CF_SLOTS =
    3; //!< Collapsed fast curvefit slots: scatter, fission, disappearance
  static constexpr int FAST_KEY_SHIFT =
    46; //!< Exponent-key shift of the fast window table

  // Types

  //! Window of the version 1 format
  struct WindowInfo {
    int index_start;   // Index of starting pole
    int index_end;     // Index of ending pole
    bool broaden_poly; // Whether to broaden polynomial curvefit
  };

  //! Transport group a reaction channel is collapsed into
  enum ChannelKind : int {
    KIND_OTHER = 0,         //!< Inelastic, (n,xn), MT5: counted as scattering
    KIND_FISSION = 1,       //!< MT18
    KIND_DISAPPEARANCE = 2, //!< MT101-117
    KIND_ELASTIC = 3        //!< MT2
  };

  //! Per-window record of the resolved segment. Everything a lookup needs to
  //! locate and size a window's block, packed into 16 bytes.
  struct WinRec {
    double u_hi;      //!< Upper window bound in sqrt(E)
    uint32_t blk_off; //!< Offset of this window's block in blk_
    uint16_t np;      //!< Number of pole records in the block
    uint16_t flags;   //!< Bit 0: pointer range contains the 1/v pole
  };
  static constexpr uint16_t WIN_HAS_ONEV = 1u;

  //! Fast-segment channel whose threshold lies inside a window. It is
  //! evaluated individually behind its clamp instead of being collapsed.
  struct FastPartial {
    double E_on;  //!< Zero below this energy [eV]
    double E_off; //!< Zero above this energy [eV]
    int kind;     //!< ChannelKind
    int ch;       //!< Channel index (selects its residues and curvefit)
  };

  //! Pole reciprocals shared by the per-channel fast evaluations at one
  //! energy, so collision channel selection evaluates channels on demand
  struct FastChanCtx {
    int k0;                           //!< First flattened pole of the window
    int np;                           //!< Number of poles in the window
    int w;                            //!< Window index
    array<double, FAST_MAX_POLES> pr; //!< Re of 1/(u - p) per pole
    array<double, FAST_MAX_POLES> pi; //!< Im of 1/(u - p) per pole
  };

  // Constructors
  WindowedMultipole() = default;

  //! \param group  HDF5 group of the nuclide
  //! \param version_major  Major version of the library file (1 or 2)
  WindowedMultipole(hid_t group, int version_major);

  // Methods

  //! \brief Evaluate the windowed multipole equations for cross sections in the
  //! resolved resonance region (and the low-energy segment, if present)
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \param dep If non-null, receives the DEPLETION_RX cross sections in [b],
  //!   accumulated into the array (zero it first). Only available for
  //!   version 2 libraries.
  //! \return Tuple of elastic scattering, absorption, and fission cross
  //! sections in [b]
  std::tuple<double, double, double> evaluate(
    double E, double sqrtkT, double* dep = nullptr) const;

  //! \brief Evaluates the windowed multipole equations for the derivative of
  //! cross sections with respect to temperature.
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of derivatives of elastic scattering, absorption, and
  //!         fission cross sections in [b/K]
  std::tuple<double, double, double> evaluate_deriv(
    double E, double sqrtkT) const;

  //! \brief Evaluate the fast segment (unresolved + fast range).
  //!
  //! The fast segment is not Doppler broadened. Elastic and other scattering
  //! are summed into one scattering group at load time, so elastic alone is
  //! not returned; use evaluate_fast_elastic() when it is needed.
  //!
  //! \param E Incident neutron energy in [eV], E_max_ < E <= fast_E_max_
  //! \param dep If non-null, receives the DEPLETION_RX cross sections in [b],
  //!   accumulated into the array (zero it first)
  //! \return Tuple of total, absorption, and fission cross sections in [b]
  std::tuple<double, double, double> evaluate_fast(
    double E, double* dep = nullptr) const;

  //! \brief Elastic scattering cross section in the fast segment
  //!
  //! \param E Incident neutron energy in [eV], E_max_ < E <= fast_E_max_
  //! \return Elastic scattering cross section in [b]
  double evaluate_fast_elastic(double E) const;

  //! \brief Prepare per-channel evaluation of the fast segment at energy E
  //!
  //! \param E Incident neutron energy in [eV], E_max_ < E <= fast_E_max_
  //! \param ctx Context filled with the window and pole reciprocals
  void fast_chan_begin(double E, FastChanCtx& ctx) const;

  //! \brief Cross section of one fast-segment channel
  //!
  //! \param ctx Context from fast_chan_begin() at the same energy
  //! \param E Incident neutron energy in [eV]
  //! \param c Channel index (see fast_channel_of_mt)
  //! \return Cross section of channel c in [b], zero outside its clamp
  double fast_chan_xs(const FastChanCtx& ctx, double E, int c) const;

  //! Channel index of a given MT in the fast segment, or -1 if not fitted
  int fast_channel_of_mt(int mt) const
  {
    return (mt >= 0 && mt < MAX_MT) ? fast_mt_to_chan_[mt] : -1;
  }

  // Data members
  std::string name_;           //!< Name of nuclide
  double E_min_;               //!< Lower bound of the resolved range [eV]
  double E_max_;               //!< Upper bound of the resolved range [eV]
  double sqrt_awr_;            //!< Square root of atomic weight ratio
  bool fissionable_ {false};   //!< Is the nuclide fissionable?
  bool global_format_ {false}; //!< Version 2 library?

  //----------------------------------------------------------------------
  // Version 1 format
  double inv_spacing_;             //!< 1 / spacing in sqrt(E) space
  int fit_order_;                  //!< Order of the fit
  vector<WindowInfo> window_info_; // Information about a window
  tensor::Tensor<double>
    curvefit_; // Curve fit coefficients (window, poly order, reaction)
  tensor::Tensor<std::complex<double>> data_; //!< Poles and residues

  //----------------------------------------------------------------------
  // Version 2: resolved segment
  //
  // Each pole is broadened as
  //   sigma = (1/E) Re[ r ((u + p) + p^2 (i sqrt(pi)/xi) w((p - u)/xi)) ],
  // with u = sqrt(E), xi = sqrt(kT/A), and the 1/v pole (Re p = 0) added
  // unbroadened since it is invariant. Every window adds a curvefit
  // sum_k c_k E^(k/2-1), broadened exactly by broaden_wmp_polynomials.
  //
  // The data a lookup touches are laid out per window in blk_: the collapsed
  // curvefit coefficients (cf_n_ x cf_ch_) followed by one record per pole,
  // [Re p, Im p, Re r_s, Im r_s, Re r_a, Im r_a, (Re r_f, Im r_f)], with the
  // fission pair omitted for non-fissionable nuclides (gs_stride_ 8 or 6).
  // Poles shared between windows are duplicated so each block is contiguous.
  vector<int> mts_;                             //!< MT of each channel
  tensor::Tensor<std::complex<double>> res_mt_; //!< Residues (n_pool, n_ch)
  tensor::Tensor<double> cf_mt_; //!< Curvefit (n_win, n_cf, n_ch)
  vector<int> win_start_;        //!< First pool-pole index per window
  vector<WinRec> win_;           //!< Per-window records
  vector<double> blk_;           //!< Per-window blocks, back to back
  int gs_stride_ {8};            //!< Doubles per pole record
  int cf_ch_ {3};                //!< Collapsed curvefit channels (3 or 2)
  bool cf_present_ {false};      //!< Does the segment carry a curvefit?
  int cf_n_ {0};                 //!< Curvefit terms (order + 1)

  //! Window lookup. With a uniform mesh in sqrt(E) the window index is
  //! arithmetic alone; otherwise a coarse uniform grid stores the first
  //! window overlapping each cell and a short forward scan finishes the job.
  bool grid_uniform_ {false};
  double grid_u_lowe_ {0.0};      //!< sqrt(E) of the first window
  double grid_inv_spacing_ {0.0}; //!< 1 / window width in sqrt(E)
  vector<int> cell_first_;        //!< First window overlapping each cell
  double cell_inv_du_ {0.0};      //!< 1 / cell width in sqrt(E)
  double win_u_lo_ {0.0};         //!< sqrt(E) of the first window

  //! The 1/v background pole can lie in the pointer range of several
  //! windows, so membership is flagged per window (WIN_HAS_ONEV) and its
  //! contribution added separately from the pole records, whose residues
  //! are zeroed for it.
  bool onev_present_ {false};
  int onev_idx_ {-1}; //!< Pool index of the 1/v pole
  std::complex<double> onev_p_, onev_rs_, onev_ra_, onev_rf_;

  //! Depletion channel map: dep_ch_[j] is the channel carrying
  //! DEPLETION_RX[j], or -1. When MT102 is the only disappearance channel,
  //! (n,gamma) = absorption - fission is exact and no per-MT pass is needed.
  array<int, WMP_N_DEP> dep_ch_;
  bool dep_needs_split_ {false};

  //----------------------------------------------------------------------
  // Version 2: low-energy segment (curvefit only, no poles)
  bool lowe_present_ {false};
  double lowe_E_min_ {0.0};    //!< Lower bound of the low-energy segment [eV]
  double lowe_E_max_ {0.0};    //!< Upper bound of the low-energy segment [eV]
  int lowe_n_ {0};             //!< Number of curvefit terms
  vector<double> lowe_cf_;     //!< [n_cf][3] collapsed to (scatter, abs, fis)
  vector<double> lowe_dep_cf_; //!< [n_cf][WMP_N_DEP] depletion coefficients
  array<int, WMP_N_DEP> lowe_dep_ch_;

  //----------------------------------------------------------------------
  // Version 2: fast segment
  //
  // Each window's pointer range plus its external poles is flattened into
  // one pole list. Residues of channels active across the whole window are
  // collapsed into the scattering, fission, and disappearance groups; a
  // channel whose threshold falls inside the window is kept as a FastPartial
  // and clamped at evaluation time. Since the segment is unbroadened, each
  // pole contributes (n1 u + n0) / ((u - alpha)^2 + beta^2) with real
  // coefficients; fw_soa_ stores these per window as
  //   [curvefit (FAST_CF_SLOTS x n_cf)][alpha(np) | beta^2(np) | n1, n0 per
  //   group in use]
  // so the sweep is one contiguous, vectorizable stream. The denominator is
  // kept factored: the expanded form cancels catastrophically for near-real
  // poles at high energy.
  bool fast_present_ {false};
  double fast_E_max_ {0.0};    //!< Upper energy of the fast segment [eV]
  vector<double> fast_bounds_; //!< Window bounds in sqrt(E) [n_fwin + 1]
  vector<int> fw_off_;         //!< Flattened pole offsets [n_fwin + 1]
  vector<std::complex<double>> fw_poles_; //!< Flattened poles [n_flat]
  bool fast_has_fis_ {false};             //!< Does any window carry fission?
  bool fast_has_dis_ {false};     //!< Does any window carry disappearance?
  int fast_soa_stride_ {6};       //!< 2 + 2 * (groups in use)
  int fast_cf_n_ {0};             //!< Curvefit terms per window
  vector<size_t> fast_blk_off_;   //!< Block start per window in fw_soa_
  vector<double> fw_soa_;         //!< Per-window blocks, back to back
  vector<uint8_t> fast_has_part_; //!< 1 if the window has partial channels
  vector<vector<FastPartial>> fast_partials_; //!< Partial channels per window

  //! The fast mesh is geometric, so the window table is keyed on the
  //! exponent bits of sqrt(E): reinterpreting a positive double as an
  //! integer is monotone, and (bits >> FAST_KEY_SHIFT) gives 64 buckets per
  //! octave. A lookup is one shift, one load, and a short forward scan.
  vector<uint16_t> fast_bidx_; //!< Bucket -> first window
  int64_t fast_key_lo_ {0};    //!< Key of fast_bounds_.front()

  //! Per-channel data, used at collisions and for depletion rates:
  //! residues are channel-major within each window block
  //! (k0 * n_mts + c * np + k) so one channel's residues are contiguous.
  int fast_n_mts_ {0};
  vector<int> fast_mts_;       //!< MT number per channel
  vector<double> fast_mt_on_;  //!< Per-channel turn-on energy [eV]
  vector<double> fast_mt_off_; //!< Per-channel turn-off energy [eV]
  vector<std::complex<double>> fw_chan_res_; //!< [n_flat * n_mts] residues
  vector<double> fw_chan_cf_;                //!< [n_fwin][n_cf][n_mts] curvefit
  array<int, MAX_MT> fast_mt_to_chan_; //!< MT -> channel index, -1 if none
  int fast_ch_el_ {-1};                //!< Channel index of MT2, or -1
  array<int, WMP_N_DEP> fast_dep_ch_;  //!< Depletion channel map

private:
  //! Read a version 1 library group
  void read_legacy(hid_t group);

  //! Read the resolved segment of a version 2 library
  void read_resolved_segment(hid_t group);

  //! Read the fast segment of a version 2 library
  void read_fast_segment(hid_t group);

  //! Read the low-energy segment of a version 2 library
  void read_lowe_segment(hid_t group);

  //! Monotone integer key of a positive double
  static int64_t fast_key(double u)
  {
    uint64_t v;
    std::memcpy(&v, &u, sizeof(v));
    return static_cast<int64_t>(v >> FAST_KEY_SHIFT);
  }

  //! Locate the fast window containing sqrt(E)
  int fast_window(double sqrtE) const
  {
    int n_win = static_cast<int>(fw_off_.size()) - 1;
    int64_t k = fast_key(sqrtE) - fast_key_lo_;
    k = std::min(std::max(k, static_cast<int64_t>(0)),
      static_cast<int64_t>(fast_bidx_.size()) - 1);
    int w = fast_bidx_[k];
    while (w < n_win - 1 && sqrtE >= fast_bounds_[w + 1])
      ++w;
    return w;
  }

  //! Locate the resolved window containing sqrt(E)
  int resolved_window(double sqrtE) const
  {
    int n_win = win_.size();
    if (grid_uniform_) {
      int i = static_cast<int>((sqrtE - grid_u_lowe_) * grid_inv_spacing_);
      return std::min(std::max(i, 0), n_win - 1);
    }
    int cell = static_cast<int>((sqrtE - win_u_lo_) * cell_inv_du_);
    cell =
      std::min(std::max(cell, 0), static_cast<int>(cell_first_.size()) - 1);
    int i = cell_first_[cell];
    while (i < n_win - 1 && sqrtE >= win_[i].u_hi)
      ++i;
    return i;
  }
};

//========================================================================
// Non-member functions
//========================================================================

//! Check that the WMP library version is supported
//!
//! \param[in] file  HDF5 file object
//! \return Major version of the file
int check_wmp_version(hid_t file);

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

//! Doppler broadens the windowed multipole curvefit and differentiates the
//! broadened basis with respect to dopp.
//!
//! \param E        The energy to evaluate the broadening at
//! \param dopp     sqrt(atomic weight ratio / kT) with kT given in eV
//! \param n        The number of components to the polynomial
//! \param factors  The broadened basis functions (as broaden_wmp_polynomials)
//! \param dfactors d(factors)/d(dopp)
void broaden_wmp_polynomials_deriv(
  double E, double dopp, int n, double factors[], double dfactors[]);

} // namespace openmc

#endif // OPENMC_WMP_H
