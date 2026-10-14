.. _io_data_wmp:

=================================
Windowed Multipole Library Format
=================================

Two versions of the windowed multipole library format are supported. Version
2 is the current format; version 1 libraries remain readable.

**/**

:Attributes: - **filetype** (*char[]*) -- String indicating the type of file
             - **version** (*int[2]*) -- Major and minor version of the data

----------------
Version 2 Format
----------------

The energy range of each nuclide is split into up to three segments stored as
subgroups: a resolved segment (required), a low-energy segment (optional), and
a fast segment (optional). Poles are given in units of :math:`\sqrt{eV}` and
residues per reaction channel. Each channel is identified by its ENDF MT
number; the code collapses channels into scattering (MT2 and any inelastic or
(n,xn) channel), absorption (MT18 and MT101--117), and fission (MT18) at load
time and keeps the per-channel data for depletion reaction rates. Complex
numbers are stored by forming a type with ":math:`r`" and ":math:`i`"
identifiers, similar to how `h5py`_ does it.

In every segment the curvefit terms multiply the basis :math:`E^{k/2-1}`, that
is :math:`a/E + b/\sqrt{E} + c + d\sqrt{E} + \ldots`.

**/<nuclide name>/resolved/**

Resolved resonance segment. Each window points at a contiguous range of the
pole table and adds its own curvefit. Poles are broadened analytically; a pole
with zero real part is treated as a :math:`1/v` background and is not
broadened.

:Attributes: - **grid_uniform** (*int*) -- Optional. If 1, windows are
               uniformly spaced in :math:`\sqrt{E}` and located arithmetically
               from **grid_u_lowe** and **grid_spacing**.
             - **grid_u_lowe** (*double*) -- Lower bound of the first window in
               :math:`\sqrt{eV}`, when **grid_uniform** is set
             - **grid_spacing** (*double*) -- Window width in
               :math:`\sqrt{eV}`, when **grid_uniform** is set

:Datasets:
    - **E_min** (*double*)
        Lowest energy in eV the segment is valid for.
    - **E_max** (*double*)
        Highest energy in eV the segment is valid for.
    - **sqrtAWR** (*double*)
        Square root of the atomic weight ratio.
    - **mts** (*int[]*)
        ENDF MT number of each channel.
    - **poles** (*complex[]*)
        Poles, sorted by real part.
    - **residues** (*complex[][]*)
        Residues indexed by (pole, channel).
    - **win_pole_start** (*int[]*)
        Index of the first pole of each window.
    - **win_pole_end** (*int[]*)
        One past the index of the last pole of each window.
    - **sqrtE_bounds** (*double[]*)
        Window boundaries in :math:`\sqrt{eV}`; one more entry than windows.
    - **curvefit** (*double[][][]*)
        Optional. Curvefit coefficients indexed by (window, term, channel).

**/<nuclide name>/lowe/**

Low-energy segment, valid below the resolved segment. It carries only a
curvefit, which Doppler broadens in closed form.

:Datasets:
    - **E_min** (*double*)
        Lowest energy in eV the segment is valid for.
    - **E_max** (*double*)
        Highest energy in eV the segment is valid for; equal to **E_min** of
        the resolved segment.
    - **sqrtAWR** (*double*)
        Square root of the atomic weight ratio.
    - **mts** (*int[]*)
        ENDF MT number of each channel.
    - **curvefit** (*double[][]*)
        Curvefit coefficients indexed by (term, channel).

**/<nuclide name>/fast/**

Fast segment covering the unresolved and fast range above the resolved
segment. It is evaluated as an unbroadened pole sum. Each window points at a
contiguous range of the pole table and adds its own external poles and
curvefit. Each channel is active only between its threshold energies; the
code clamps it to zero outside.

:Datasets:
    - **E_min** (*double*)
        Lowest energy in eV the segment is valid for; equal to **E_max** of the
        resolved segment.
    - **E_max** (*double*)
        Highest energy in eV the segment is valid for.
    - **sqrtAWR** (*double*)
        Square root of the atomic weight ratio.
    - **mts** (*int[]*)
        ENDF MT number of each channel.
    - **mt_E_on** (*double[]*)
        Energy in eV below which each channel is zero.
    - **mt_E_off** (*double[]*)
        Energy in eV above which each channel is zero.
    - **poles** (*complex[]*)
        Poles, sorted by real part.
    - **residues** (*complex[][]*)
        Residues indexed by (pole, channel).
    - **win_pole_start** (*int[]*)
        Index of the first pole of each window.
    - **win_pole_end** (*int[]*)
        One past the index of the last pole of each window.
    - **sqrtE_bounds** (*double[]*)
        Window boundaries in :math:`\sqrt{eV}`; one more entry than windows.
    - **ext_poles** (*complex[][]*)
        External poles indexed by (window, external pole).
    - **ext_residues** (*complex[][][]*)
        External residues indexed by (window, external pole, channel).
    - **curvefit** (*double[][][]*)
        Optional. Curvefit coefficients indexed by (window, term, channel).

----------------
Version 1 Format
----------------

**/<nuclide name>/**

:Datasets:

    - **broaden_poly** (*int[]*)
        If 1, Doppler broaden curve fit for window with corresponding index.
        If 0, do not.
    - **curvefit** (*double[][][]*)
        Curve fit coefficients. Indexed by (window index, coefficient index,
        reaction type).
    - **data** (*complex[][]*)
        Complex poles and residues. Each pole has a corresponding set of
        residues. For example, the :math:`i`-th pole and corresponding residues
        are stored as

        .. math::
            \text{data}[:,i] = [\text{pole},~\text{residue}_1,~\text{residue}_2,
            ~\ldots]

        The residues are in the order: scattering, absorption, fission. Complex
        numbers are stored by forming a type with ":math:`r`" and ":math:`i`"
        identifiers, similar to how `h5py`_ does it.
    - **E_max** (*double*)
        Highest energy the windowed multipole part of the library is valid for.
    - **E_min** (*double*)
        Lowest energy the windowed multipole part of the library is valid for.
    - **spacing** (*double*)
        .. math::
            \frac{\sqrt{E_{max}} - \sqrt{E_{min}}}{n_w}

        Where :math:`E_{max}` is the maximum energy the windows go up to.
        :math:`E_{min}` is the minimum energy, and :math:`n_w` is the number of
        windows, given by ``windows``.
    - **sqrtAWR** (*double*)
        Square root of the atomic weight ratio.
    - **windows** (*int[][]*)
        The poles to start from and end at for each window. windows[i, 0] and
        windows[i, 1] are, respectively, the indexes (1-based) of the first and
        last pole in window i.

.. _h5py: https://docs.h5py.org/en/latest/
