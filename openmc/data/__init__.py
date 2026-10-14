# Version of HDF5 nuclear data format
HDF5_VERSION_MAJOR = 3
HDF5_VERSION_MINOR = 0
HDF5_VERSION = (HDF5_VERSION_MAJOR, HDF5_VERSION_MINOR)

# Version of WMP nuclear data format
WMP_VERSION_MAJOR = 2
WMP_VERSION_MINOR = 0
WMP_VERSION = (WMP_VERSION_MAJOR, WMP_VERSION_MINOR)
# Legacy per-window format, still readable and writable
WMP_VERSION_MAJOR_LEGACY = 1
WMP_VERSION_LEGACY = (WMP_VERSION_MAJOR_LEGACY, 1)


from .data import *
from .neutron import *
from .photon import *
from .decay import *
from .reaction import *
from . import ace
from .angle_distribution import *
from . import endf
from .energy_distribution import *
from .product import *
from .angle_energy import *
from .uncorrelated import *
from .correlated import *
from .kalbach_mann import *
from .nbody import *
from .thermal import *
from .urr import *
from .library import *
from .fission_energy import *
from .resonance import *
from .resonance_covariance import *
from .multipole import *
from .grid import *
from .function import *
from .vectfit import *

from .dose.dose import dose_coefficients
from .dose.mass_attenuation import \
    mass_energy_absorption_coefficient, mass_attenuation_coefficient
