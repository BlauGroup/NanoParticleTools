from NanoParticleTools.species_data.species import Dopant
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.analysis.util import (
    get_wavelengths, intensities_from_population,
    mean_population_to_intensities, get_spectrum_wavelength_from_intensities)
import numpy as np

import pytest


@pytest.fixture
def sk():
    dopants = [Dopant('Yb', 0.5), Dopant('Er', 0.02, n_levels=7)]
    sk = SpectralKinetics(dopants)
    return sk


def test_get_wavelengths(sk):
    wavelengths = get_wavelengths(sk)
    # yapf: disable
    expected_wavelengths = np.array(
        [
            [
                np.inf, 975.6097561, np.inf, 1539.64588145, 987.26429065,
                807.75444265, 656.25410159, 544.4843733, 523.0125523
            ], [
                -975.6097561, np.inf, -975.6097561, -2663.11584554,
                -82644.62809917, 4694.83568075, 2004.81154771, 1232.13405619,
                1127.3957159
            ], [
                np.inf, 975.6097561, np.inf, 1539.64588145, 987.26429065,
                807.75444265, 656.25410159, 544.4843733, 523.0125523
            ],
            [
                -1539.64588145, 2663.11584554, -1539.64588145, np.inf,
                2751.78866263, 1699.2353441, 1143.77216059, 842.38901525,
                792.07920792
            ],
            [
                -987.26429065, 82644.62809917, -987.26429065, -2751.78866263,
                np.inf, 4442.47001333, 1957.33020161, 1214.03423577, 1112.22333445
            ],
            [
                -807.75444265, -4694.83568075, -807.75444265, -1699.2353441,
                -4442.47001333, np.inf, 3498.95031491, 1670.56465085,
                1483.67952522
            ],
            [
                -656.25410159, -2004.81154771, -656.25410159, -1143.77216059,
                -1957.33020161, -3498.95031491, np.inf, 3196.93094629,
                2575.99175683
            ],
            [
                -544.4843733, -1232.13405619, -544.4843733, -842.38901525,
                -1214.03423577, -1670.56465085, -3196.93094629, np.inf,
                13262.5994695
            ],
            [
                -523.0125523, -1127.3957159, -523.0125523, -792.07920792,
                -1112.22333445, -1483.67952522, -2575.99175683, -13262.5994695,
                np.inf
            ]
        ])
    # yapf: enable
    assert wavelengths.shape == (9, 9)
    assert np.allclose(wavelengths, expected_wavelengths)


def test_intensities_from_population(sk):
    populations = np.array([0.75, 0.25, 0.7, 0.2, 0.05, 0.05, 0, 0, 0])
    populations = np.tile(populations, (10, 1))

    _, intensities = intensities_from_population(sk,
                                                 populations,
                                                 volume=1,
                                                 last_n_avg=5)
    # yapf: disable
    expected_intensities = np.array(
        [
            [
                0.00000000e+00, 2.13643100e+05, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00
            ],
            [
                1.46602914e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00
            ],
            [
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.62602055e-01,
                2.80079440e+04, 1.13941765e+01, 3.29906392e-03, 0.00000000e+00,
                0.00000000e+00
            ],
            [
                0.00000000e+00, 0.00000000e+00, 1.65483977e+01, 0.00000000e+00,
                0.00000000e+00, 2.59573074e-03, 3.05502793e+01, 3.58254172e+01,
                1.26798986e+00
            ],
            [
                0.00000000e+00, 0.00000000e+00, 5.09488085e+00, 7.06891906e-01,
                0.00000000e+00, 0.00000000e+00, 8.46140871e-05, 4.29759285e-01,
                2.12508401e+01
            ],
            [
                0.00000000e+00, 0.00000000e+00, 5.14130402e+00, 1.96601598e+00,
                3.14149440e-02, 0.00000000e+00, 0.00000000e+00, 6.18975736e-04,
                2.05015756e-02
            ],
            [
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00
            ],
            [
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00
            ],
            [
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00
            ]
        ])
    # yapf: enable
    assert intensities.shape == (9, 9)
    assert np.allclose(intensities, expected_intensities)


def test_get_spectrum_wavelength_from_intensities(sk):
    populations = np.array([0.75, 0.25, 0.7, 0.2, 0.05, 0.05, 0, 0, 0])

    wavelengths = get_wavelengths(sk)
    intensities = mean_population_to_intensities(sk, populations, volume=1)

    x, spectrum = get_spectrum_wavelength_from_intensities(
        wavelengths, intensities)

    assert x.shape == (600, )
    assert spectrum.shape == (600, )
    assert x[0] == -1997.5
    assert x[-1] == 997.5
    assert x[1] - x[0] == 5

    x, spectrum = get_spectrum_wavelength_from_intensities(wavelengths,
                                                           intensities,
                                                           lower_bound=0,
                                                           upper_bound=800,
                                                           step=2)
    assert x.shape == (400, )
    assert spectrum.shape == (400, )
    assert x[0] == 1
    assert x[-1] == 799
    assert x[1] - x[0] == 2