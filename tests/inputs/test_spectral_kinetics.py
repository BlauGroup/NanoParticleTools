import json
import os
from pathlib import Path

import numpy as np
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics

from NanoParticleTools.species_data.species import Dopant

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
