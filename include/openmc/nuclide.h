//! \file nuclide.h
//! \brief Nuclide type and other associated types/data

#ifndef OPENMC_NUCLIDE_H
#define OPENMC_NUCLIDE_H

#include <unordered_map>
#include <utility> // for pair

#include <hdf5.h>

#include "openmc/array.h"
#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/particle.h"
#include "openmc/reaction.h"
#include "openmc/reaction_product.h"
#include "openmc/span.h"
#include "openmc/urr.h"
#include "openmc/vector.h"
#include "openmc/wmp.h"

namespace openmc {

//==============================================================================
// Data for a nuclide
//==============================================================================

class Nuclide {
public:
  //============================================================================
  // Types, aliases
  using EmissionMode = ReactionProduct::EmissionMode;
  struct EnergyGrid {
    vector<int> grid_index;
    vector<double> energy;
  };

  //============================================================================
  // Constructors/destructors
  Nuclide(hid_t group, const vector<double>& temperature);
  ~Nuclide();

  //============================================================================
  // Methods

  //! Initialize logarithmic grid for energy searches
  void init_grid();

  //! Calculate microscopic cross sections
  //
  //! \param[in] i_sab  Index in data::thermal_scatt
  //! \param[in] i_log_union  Log-grid search index
  //! \param[in] sab_frac  S(a,b) table fraction
  //! \param[in,out] p  Particle object
  void calculate_xs(int i_sab, int i_log_union, double sab_frac, Particle& p);

  //! Compute the ACE energy-grid index if it was deferred (lazy grid
  //! indexing). Call at collision time for the sampled nuclide.
  //! \param[in,out] p  Particle object
  void ensure_grid_index(Particle& p) const;

  //! Calculate thermal scattering cross section
  //
  //! \param[in] i_sab  Index in data::thermal_scatt
  //! \param[in] sab_frac  S(a,b) table fraction
  //! \param[in,out] p  Particle object
  void calculate_sab_xs(int i_sab, double sab_frac, Particle& p);

  double nu(double E, EmissionMode mode, int group = 0) const;
  void calculate_elastic_xs(Particle& p) const;

  //! Determines the microscopic 0K elastic cross section at a trial relative
  //! energy used in resonance scattering
  double elastic_xs_0K(double E) const;

  //! \brief Determines cross sections in the unresolved resonance range
  //! from probability tables.
  void calculate_urr_xs(int i_temp, Particle& p) const;

  //! \brief Calculate reaction rate based on group-wise flux distribution
  //
  //! \param[in] MT ENDF MT value for desired reaction
  //! \param[in] temperature Temperature in [K]
  //! \param[in] energy Energy group boundaries in [eV]
  //! \param[in] flux Flux in each energy group (not normalized per eV)
  //! \return Reaction rate
  double collapse_rate(int MT, double temperature, span<const double> energy,
    span<const double> flux) const;

  //! Return a ParticleType object representing this nuclide
  ParticleType particle_type() const { return {Z_, A_, metastable_}; }

  //============================================================================
  // Data members
  std::string name_; //!< Name of nuclide, e.g. "U235"
  int Z_;            //!< Atomic number
  int A_;            //!< Mass number
  int metastable_;   //!< Metastable state
  //! Target excitation [eV]; NaN if unavailable.
  double excitation_energy_ {FP_UNSET};
  double awr_;    //!< Atomic weight ratio
  int64_t index_; //!< Index in the nuclides array

  // Temperature dependent cross section data
  vector<double> kTs_;                //!< temperatures in eV (k*T)
  vector<EnergyGrid> grid_;           //!< Energy grid at each temperature
  vector<tensor::Tensor<double>> xs_; //!< Cross sections at each temperature

  // Multipole data
  unique_ptr<WindowedMultipole> multipole_;

  //! Multipole energy bounds, mirrored from the WindowedMultipole object so
  //! the range tests on every cross section lookup need no dereference
  bool mp_present_ {false};      //!< Is multipole data present?
  bool mp_fast_present_ {false}; //!< Does it include a fast segment?
  double mp_E_min_ {0.0};        //!< Lower bound of multipole data [eV]
  double mp_E_max_ {0.0};        //!< Upper bound of the resolved range [eV]
  double mp_fast_E_max_ {0.0};   //!< Upper bound of the fast segment [eV]

  //! Build an O(1) lookup of the tabulated total nu over [e_lo, e_hi]. The
  //! fast multipole segment evaluates nu(E) on every lookup, where a binary
  //! search would dominate. Only enabled when the map reproduces the
  //! Tabulated1D evaluation exactly.
  void init_nu_fast_map(double e_lo, double e_hi);

  //! Tabulated total nu via the O(1) map; only valid when nu_fast_ok_
  double nu_total_fast(double E) const
  {
    int c = static_cast<int>((E - nu_e0_) * nu_inv_de_);
    c = std::min(std::max(c, 0), static_cast<int>(nu_cell_.size()) - 1);
    int j = nu_cell_[c];
    int jmax = static_cast<int>(nu_fx_.size()) - 2;
    while (j < jmax && E >= nu_fx_[j + 1])
      ++j;
    double f = (E - nu_fx_[j]) / (nu_fx_[j + 1] - nu_fx_[j]);
    return nu_fy_[j] + f * (nu_fy_[j + 1] - nu_fy_[j]);
  }
  bool nu_fast_ok_ {false};      //!< Is the nu map usable?
  vector<double> nu_fx_, nu_fy_; //!< Tabulated nu (ascending x, lin-lin)
  vector<int> nu_cell_;          //!< Uniform-energy cell -> bracketing index
  double nu_e0_ {0.0};           //!< Lower bound of the map [eV]
  double nu_inv_de_ {0.0};       //!< 1 / cell width [1/eV]

  // Fission data
  bool fissionable_ {false};         //!< Whether nuclide is fissionable
  bool has_partial_fission_ {false}; //!< has partial fission reactions?
  vector<Reaction*> fission_rx_;     //!< Fission reactions
  int n_precursor_ {0};              //!< Number of delayed neutron precursors
  unique_ptr<Function1D> total_nu_;  //!< Total neutron yield
  unique_ptr<Function1D> fission_q_prompt_; //!< Prompt fission energy release
  unique_ptr<Function1D>
    fission_q_recov_; //!< Recoverable fission energy release
  unique_ptr<Function1D> prompt_photons_;  //!< Prompt photon energy release
  unique_ptr<Function1D> delayed_photons_; //!< Delayed photon energy release
  unique_ptr<Function1D> fragments_;       //!< Fission fragment energy release
  unique_ptr<Function1D> betas_;           //!< Delayed beta energy release

  // Resonance scattering information
  bool resonant_ {false};
  vector<double> energy_0K_;
  vector<double> elastic_0K_;
  vector<double> xs_cdf_;

  // Unresolved resonance range information
  bool urr_present_ {false};
  int urr_inelastic_ {C_NONE};
  vector<UrrData> urr_data_;

  vector<unique_ptr<Reaction>> reactions_; //!< Reactions
  array<size_t, 902> reaction_index_;      //!< Index of each reaction
  vector<int> index_inelastic_scatter_;

private:
  //! Locate the energy-grid interval containing E, using the logarithmic
  //! union grid to narrow the binary search
  //! \param[in] i_temp  Temperature index
  //! \param[in] i_log_union  Index on the logarithmic union grid
  //! \param[in] E  Energy in [eV]
  //! \param[out] i_grid  Index on the nuclide energy grid
  //! \param[out] f  Interpolation factor
  void find_grid_index(
    int i_temp, int i_log_union, double E, int& i_grid, double& f) const;

  void create_derived(
    const Function1D* prompt_photons, const Function1D* delayed_photons);

  //! Determine temperature index and interpolation factor
  //
  //! \param[in] T Temperature in [K]
  //! \return Temperature index and interpolation factor
  std::pair<int64_t, double> find_temperature(double T) const;

  static int XS_TOTAL;
  static int XS_ABSORPTION;
  static int XS_FISSION;
  static int XS_NU_FISSION;
  static int XS_PHOTON_PROD;
};

//==============================================================================
// Non-member functions
//==============================================================================

//! Checks for the right version of nuclear data within HDF5 files
void check_data_version(hid_t file_id);

bool multipole_in_range(const Nuclide& nuc, double E);

//==============================================================================
// Global variables
//==============================================================================

namespace data {

// Minimum/maximum transport energy for each particle type. Order corresponds to
// transport_index() for supported transport particles.
extern array<double, 4> energy_min;
extern array<double, 4> energy_max;

//! Minimum temperature in [K] that nuclide data is available at
extern double temperature_min;

//! Maximum temperature in [K] that nuclide data is available at
extern double temperature_max;

extern std::unordered_map<std::string, int> nuclide_map;
extern vector<unique_ptr<Nuclide>> nuclides;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

void nuclides_clear();

} // namespace openmc

#endif // OPENMC_NUCLIDE_H
