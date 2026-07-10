#ifndef OPENMC_WMP_H
#define OPENMC_WMP_H

#include "hdf5.h"
#include "openmc/tensor.h"

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
  WindowedMultipole(hid_t group);

  // Methods

  //! \brief Evaluate the windowed multipole equations for cross sections in the
  //! resolved resonance regions
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of elastic scattering, absorption, and fission cross
  //! sections in [b]
  std::tuple<double, double, double> evaluate(double E, double sqrtkT) const;

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

  // Data members
  std::string name_;               //!< Name of nuclide
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
  // TEST-PHASE LIMITATIONS: resolved segment only; low-E image-kernel
  // correction (E < ~(4.5 xi)^2, sub-5-meV) not implemented;
  // evaluate_deriv not implemented.
  bool global_format_ {false};   //!< New-format library?
  vector<double> sqrtE_bounds_;  //!< Window boundaries in sqrt(E) [n_win+1]
  vector<int> win_start_;        //!< First pool-pole index per window
  vector<int> win_end_;          //!< One-past-last pool-pole index
  // Two-level O(1) window lookup over the variable-width windows: a uniform
  // coarse grid in sqrt(E) (4 cells per window on average) stores the first
  // window overlapping each cell; a lookup is one multiply + a short forward
  // walk (bounded by windows-per-cell, ~1). Replaces the binary search.
  vector<int> cell_first_;       //!< First window overlapping each grid cell
  double cell_inv_du_ {0.0};     //!< 1 / coarse-cell width in sqrt(E)
  // Pool: [pole, r_elastic, r_absorption, r_fission] like data_ columns
  tensor::Tensor<std::complex<double>> gdata_; //!< (n_pool, 4)
  // Externals: 4 per window, same column convention
  tensor::Tensor<std::complex<double>> ext_data_; //!< (n_win, 4, 4)

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
