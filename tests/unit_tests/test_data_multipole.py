import os
import pathlib

import h5py
import numpy as np
import pytest
import openmc.data


@pytest.fixture(scope='module')
def u235():
    directory = pathlib.Path(openmc.config.get('cross_sections')).parent
    u235 = directory / 'wmp' / '092235.h5'
    return openmc.data.WindowedMultipole.from_hdf5(u235)


@pytest.fixture(scope='module')
def b10():
    directory = pathlib.Path(openmc.config.get('cross_sections')).parent
    b10 = directory / 'wmp' / '005010.h5'
    return openmc.data.WindowedMultipole.from_hdf5(b10)


def test_evaluate(u235):
    """Test the cross section evaluation of a library."""
    energies = [1e-3, 1.0, 10.0, 50.]
    scattering, absorption, fission = u235(energies, 0.0)
    assert (scattering[1], absorption[1], fission[1]) == \
           pytest.approx((13.09, 77.56, 67.36), rel=1e-3)
    scattering, absorption, fission = u235(energies, 300.0)
    assert (scattering[2], absorption[2], fission[2]) == \
           pytest.approx((11.24, 21.26, 15.50), rel=1e-3)


def test_evaluate_none_poles(b10):
    """Test a library with no poles, i.e., purely polynomials."""
    energies = [1e-3, 1.0, 10.0, 1e3, 1e5]
    scattering, absorption, fission = b10(energies, 0.0)
    assert (scattering[0], absorption[0], fission[0]) == \
           pytest.approx((2.201, 19330., 0.), rel=1e-3)
    scattering, absorption, fission = b10(energies, 300.0)
    assert (scattering[-1], absorption[-1], fission[-1]) == \
           pytest.approx((2.878, 1.982, 0.), rel=1e-3)


def test_export_to_hdf5(tmpdir, u235):
    filename = str(tmpdir.join('092235.h5'))
    u235.export_to_hdf5(filename)
    assert os.path.exists(filename)


def test_from_endf(endf_data):
    endf_file = os.path.join(endf_data, 'neutrons', 'n-001_H_001.endf')
    assert openmc.data.WindowedMultipole.from_endf(
            endf_file,
            log=True,
            # Keep the test lightweight
            vf_options={
                "njoy_error": 5e-3,
                "vf_pieces": 1,
                "rtol": 5e-2,
                "atol": 1e-3,
                "orders": [8, 12],
                "n_vf_iter": 6,
            },
            wmp_options={"n_win": 50, "n_cf": 3, "rtol": 5e-2, "atol": 1e-3},
        )


def test_from_endf_search(endf_data):
    endf_file = os.path.join(endf_data, 'neutrons', 'n-001_H_001.endf')
    assert openmc.data.WindowedMultipole.from_endf(
            endf_file,
            log=True,
            vf_options={
                "njoy_error": 5e-3,
                "vf_pieces": 1,
                "rtol": 5e-2,
                "atol": 1e-3,
                "orders": [8, 12],
                "n_vf_iter": 6,
            },
            wmp_options={
                "search": True,
                "rtol": 5e-2,
                "search_n_win": 3,
                "search_cf_orders": [5, 3],
            },
        )


@pytest.fixture
def synthetic_v2():
    """A small version 2 library with all three segments."""
    wmp = openmc.data.GlobalPoleMultipole('Xx1')
    wmp.sqrtAWR = 10.0

    res = openmc.data.ResolvedSegment()
    res.E_min, res.E_max = 1.0, 100.0
    res.mts = np.array([2, 18, 102, 107])
    res.poles = np.array([0.5j, 2.5 + 0.05j, 5.5 + 0.08j, 8.2 + 0.04j])
    res.residues = -1j * np.array([
        [0.0, 0.0, 3.0, 0.5],
        [0.5, 0.2, 0.3, 0.0],
        [1.0, 0.0, 0.6, 0.1],
        [0.7, 0.1, 0.4, 0.0]])
    res.win_pole_start = np.array([0, 0, 0, 0, 1, 2, 2, 3, 3])
    res.win_pole_end = np.array([2, 2, 3, 3, 3, 4, 4, 4, 4])
    res.sqrtE_bounds = np.arange(1.0, 11.0)
    res.curvefit = np.tile(np.array([
        [1.0, 0.1, 0.5, 0.05],
        [0.2, 0.0, 0.1, 0.0],
        [3.0, 0.05, 0.1, 0.01]]), (9, 1, 1))
    res.grid_uniform = True
    res.grid_u_lowe, res.grid_spacing = 1.0, 1.0
    wmp.resolved = res

    lowe = openmc.data.LowEnergySegment()
    lowe.E_min, lowe.E_max = 1e-5, 1.0
    lowe.mts = np.array([2, 102, 107])
    lowe.curvefit = np.array([[0.1, 1.0, 0.2], [0.05, 0.3, 0.0], [2.0, 0.01, 0.0]])
    wmp.lowe = lowe

    fast = openmc.data.FastSegment()
    fast.E_min, fast.E_max = 100.0, 1e6
    fast.mts = np.array([2, 51, 18, 102, 16])
    fast.mt_E_on = np.array([0.0, 5000.0, 0.0, 0.0, 2e5])
    fast.mt_E_off = np.full(5, 1e6)
    fast.poles = np.array([20.0 + 1.0j, 60.0 + 2.0j])
    fast.residues = -1j * np.array([[2.0, 0.5, 0.1, 0.3, 0.2],
                                    [3.0, 1.0, 0.2, 0.2, 0.4]])
    fast.win_pole_start = np.array([0, 0, 1])
    fast.win_pole_end = np.array([1, 2, 2])
    fast.sqrtE_bounds = np.array([10.0, 30.0, 100.0, 1000.0])
    fast.ext_poles = np.array([[15.0 + 30.0j], [60.0 + 50.0j], [300.0 + 200.0j]])
    fast.ext_residues = -1j * np.array([[[20, 2, 1, 3, 1]], [[30, 5, 1, 2, 2]],
                                        [[100, 10, 2, 3, 5]]], dtype=float)
    fast.curvefit = np.tile(np.array([[10, 1, 0.5, 2, 0.5],
                                      [1, 0.1, 0.05, 0.2, 0.1]]), (3, 1, 1))
    wmp.fast = fast
    return wmp


def test_v2_round_trip(run_in_tmpdir, synthetic_v2):
    synthetic_v2.export_to_hdf5('Xx1.h5', 'w')
    with h5py.File('Xx1.h5', 'r') as f:
        assert tuple(f.attrs['version']) == openmc.data.WMP_VERSION
        assert set(f['Xx1']) == {'resolved', 'lowe', 'fast'}

    # The generic reader dispatches on the version
    wmp = openmc.data.WindowedMultipole.from_hdf5('Xx1.h5')
    assert isinstance(wmp, openmc.data.GlobalPoleMultipole)
    assert wmp.name == 'Xx1'
    assert wmp.fissionable
    assert wmp.E_min == 1e-5 and wmp.E_max == 1e6
    assert list(wmp.mts) == [2, 16, 18, 51, 102, 107]
    np.testing.assert_allclose(wmp.resolved.poles, synthetic_v2.resolved.poles)
    np.testing.assert_allclose(wmp.resolved.curvefit, synthetic_v2.resolved.curvefit)
    assert wmp.resolved.grid_uniform
    np.testing.assert_allclose(wmp.fast.ext_residues, synthetic_v2.fast.ext_residues)
    np.testing.assert_allclose(wmp.lowe.curvefit, synthetic_v2.lowe.curvefit)
    for E, T in [(0.5, 0.0), (6.25, 0.0), (6.25, 900.0), (6000.0, 0.0)]:
        assert wmp(E, T) == pytest.approx(synthetic_v2(E, T))


def test_v2_evaluate(synthetic_v2):
    wmp = synthetic_v2

    # Low-energy segment: curvefit alone, 1/E term dominates at 1e-4 eV
    xs = wmp.evaluate(1e-4, 0.0)
    assert xs[102] == pytest.approx(1.0/1e-4 + 0.3/1e-2 + 0.01)
    assert xs[18] == 0.0

    # Resolved segment at 0 K: direct pole sum plus curvefit, window 1
    E = 6.25
    u = np.sqrt(E)
    expected = {2: 0.0, 18: 0.0, 102: 0.0, 107: 0.0}
    for i in range(0, 2):
        phi = 1.0 / (u - wmp.resolved.poles[i])
        for c, mt in enumerate(wmp.resolved.mts):
            expected[mt] += (wmp.resolved.residues[i, c] * phi).real
    fac = np.array([1/E, 1/u, 1.0])
    for c, mt in enumerate(wmp.resolved.mts):
        expected[mt] += fac @ wmp.resolved.curvefit[1][:, c]
    xs = wmp.evaluate(E, 0.0)
    for mt in expected:
        assert xs[mt] == pytest.approx(expected[mt], rel=1e-12)
    sig_s, sig_a, sig_f = wmp(E, 0.0)
    assert sig_s == pytest.approx(expected[2])
    assert sig_a == pytest.approx(expected[18] + expected[102] + expected[107])
    assert sig_f == pytest.approx(expected[18])

    # Broadening changes the resonance but not the 1/v pole's share
    hot = wmp.evaluate(E, 1200.0)
    assert hot[102] != pytest.approx(xs[102])
    assert hot[102] > 0.0

    # Fast segment: MT51 is gated by its threshold inside window 1
    assert wmp.evaluate(4000.0, 0.0)[51] == 0.0
    assert wmp.evaluate(6000.0, 0.0)[51] > 0.0
    assert wmp.evaluate(1e5, 0.0)[16] == 0.0
    assert wmp.evaluate(3e5, 0.0)[16] > 0.0

    # Vectorized energies
    energies = np.array([0.5, 6.25, 6000.0])
    per_mt = wmp.evaluate(energies, 300.0)
    assert per_mt[2].shape == energies.shape
    assert per_mt[2][1] == pytest.approx(wmp.evaluate(6.25, 300.0)[2])


def test_v2_bad_version(run_in_tmpdir, synthetic_v2):
    synthetic_v2.export_to_hdf5('Xx1.h5', 'w')
    with h5py.File('Xx1.h5', 'r+') as f:
        f.attrs['version'] = np.array([7, 0])
    with pytest.raises(openmc.data.DataError):
        openmc.data.WindowedMultipole.from_hdf5('Xx1.h5')
