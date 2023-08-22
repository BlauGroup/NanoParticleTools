import json
import os
from pathlib import Path

import numpy as np
from NanoParticleTools.inputs import SpectralKinetics

from NanoParticleTools.species_data.species import Dopant

import pytest

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / '..' / 'test_files/spectral_kinetics/Yb10_Er2'


def initialize():
    with open(os.path.join(TEST_FILE_DIR, 'sk_config.json'), 'r') as f:
        config_dict = json.load(f)
    dopants = [Dopant('Er', 0.02, 34), Dopant('Yb', 0.1, 2)]
    sk = SpectralKinetics(dopants, **config_dict)
    return sk


SPECTRAL_KINETICS = initialize()


def test_line_strength_matrix():
    # Check line_strength_matrix
    with open(os.path.join(TEST_FILE_DIR, 'M_Srad.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            data.append([float(val) for val in line.strip().split(',')])
    m_srad = np.array(data)

    assert np.allclose(m_srad, SPECTRAL_KINETICS.line_strength_matrix)


def test_non_radiative_rate_matrix():
    # Check non_radiative_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'M_NRrate.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            data.append([float(val) for val in line.strip().split(',')])
    m_srad = np.array(data)

    assert np.allclose(m_srad, SPECTRAL_KINETICS.non_radiative_rate_matrix)


def test_radiative_rate_matrix():
    # Check radiative_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'M_RadRate.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            data.append([float(val) for val in line.strip().split(',')])
    m_radrate = np.array(data)
    assert np.allclose(m_radrate, SPECTRAL_KINETICS.radiative_rate_matrix)


def test_magnetic_dipole_rate_matrix():
    # Check magnetic_dipole_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'M_MDradRate.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            data.append([float(val) for val in line.strip().split(',')])
    m_mdradrate = np.array(data)
    assert np.allclose(m_mdradrate,
                       SPECTRAL_KINETICS.magnetic_dipole_rate_matrix)


def test_energy_transfer_rate_matrix():
    # Check energy_transfer_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'W_ETrates.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            data.append([float(val) for val in line.strip().split(',')])
    rates = np.array(data)

    with open(os.path.join(TEST_FILE_DIR, 'M_ETIndices.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            data.append([float(val) for val in line.strip().split(',')])
    indices = np.array(data)
    et_rates = np.hstack([indices, rates])

    assert np.allclose(et_rates, SPECTRAL_KINETICS.energy_transfer_rate_matrix)


def test_rate_equations():

    dopants = [Dopant('Tm', 0.1), Dopant('Yb', 0.1), Dopant('Er', 0.1)]
    sk = SpectralKinetics(dopants,
                          excitation_wavelength=800,
                          excitation_power=1e5)

    with pytest.raises(ValueError):
        # this is an invalid population specification
        sk.run_kinetics('ground')

    with pytest.raises(ValueError):
        # this is incorrect input due to length mismatch
        sk.run_kinetics([0, 3])

    sol = sk.run_kinetics()
    expected_last_step = np.array([
        4.86944367e-01, 4.82164480e-01, 1.34751952e-02, 1.62760578e-02,
        9.10107579e-04, 1.24939039e-04, 8.15440646e-05, 2.25556935e-05,
        7.29533326e-07, 6.80744674e-08, 1.21499731e-08, 6.39451603e-08,
        9.95529863e-01, 4.47013698e-03, 9.48071664e-01, 2.39684359e-02,
        9.54176984e-03, 7.33233685e-03, 8.54639753e-03, 1.96569495e-03,
        1.47853735e-04, 4.88770113e-05, 5.91197673e-05, 1.88960466e-05,
        1.76938276e-04, 8.44747490e-05, 6.89471684e-06, 2.66811420e-06,
        9.98595795e-07, 2.23297243e-05, 1.53813599e-06, 9.84114572e-07,
        4.11210568e-07, 6.24790668e-08, 2.75599199e-08, 2.61066330e-07,
        9.75545922e-07, 9.62850789e-08, 2.04394626e-07, 4.23309831e-08,
        5.86237305e-09, 2.03634489e-09, 1.44858808e-09, 3.27894336e-08,
        3.37456042e-09, 1.26300922e-09, 2.21564490e-10, 3.92084051e-09
    ])
    assert np.allclose(sol[1][-1], expected_last_step)

    decay_sk = SpectralKinetics(dopants,
                                excitation_wavelength=800,
                                excitation_power=0)
    decay_sol = decay_sk.run_kinetics(list(sol[1][-1]))
    expected_last_step = np.array([
        9.77804920e-01, 2.21790875e-02, 1.59541681e-05, 1.65533023e-07,
        1.69168974e-09, 2.34159958e-10, 6.89186543e-14, 8.27146981e-15,
        2.84247615e-17, 2.19559929e-18, 4.66538293e-20, -5.32197600e-19,
        9.99999814e-01, 1.85686909e-07, 9.99997901e-01, 1.69559117e-06,
        3.69204920e-07, 2.10353130e-08, 1.66874840e-08, 8.11375746e-11,
        6.13819377e-12, 1.98599462e-12, 1.10722677e-13, 3.51560419e-14,
        3.44768164e-13, 3.12420420e-14, 1.26657136e-15, 4.79749678e-16,
        1.74408965e-16, 1.08690254e-14, 7.93240221e-17, 5.04741133e-17,
        2.02392270e-17, 2.69459264e-18, 1.37588051e-18, 1.57685281e-17,
        7.59159745e-17, 7.00550298e-18, 3.90641967e-18, 7.83656638e-19,
        7.07283069e-20, 1.39761620e-20, 6.49229767e-21, 2.15987335e-18,
        2.00971025e-19, 7.15003007e-20, 3.84920859e-21, -1.83440370e-20
    ])
    assert np.allclose(decay_sol[1][-1], expected_last_step)


def test_all_elements():
    elements = ["Yb", "Er", "Tm", "Nd", "Ho", "Eu", "Tb", "Sm", "Dy"]
    dopants = [Dopant(el, 0.01) for el in elements]
    sk = SpectralKinetics(dopants)

    assert sk
