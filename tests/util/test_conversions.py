from NanoParticleTools.util.conversions import wavelength_to_wavenumber, wavenumber_to_wavelength
import numpy as np


def test_wavenumber_to_wavelength():
    assert np.allclose(
        wavenumber_to_wavelength(np.array([1000, 10000, 20000])),
        np.array([10000, 1000, 500]))


def test_wavelength_to_wavenumber():
    assert np.allclose(wavelength_to_wavenumber(np.array([10000, 1000, 500])),
                       np.array([1000, 10000, 20000]))
