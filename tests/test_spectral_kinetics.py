import json
import os
from pathlib import Path

import numpy as np
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics

from NanoParticleTools.species_data.species import Dopant

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / ".." / ".." / "test_files"

def initialize():
    with open(os.path.join(TEST_FILE_DIR, 'spectral_kinetics/Yb10_Er2/sk_config.json'), 'r') as f:
        config_dict = json.load(f)
    dopants = [Dopant('Er', 0.02, 34), Dopant('Yb', 0.1, 2)]
    sk = SpectralKinetics(dopants, **config_dict)
    return sk

def test_line_strength_matrix():
    # Check line_strength_matrix
    with open(os.path.join(TEST_FILE_DIR, 'spectral_kinetics/Yb10_Er2/M_Srad.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0: continue
            data.append([float(val) for val in line.strip().split(',')])
    m_srad = np.array(data)
    matches = np.allclose(m_srad, sk.line_strength_matrix)
    if matches:
        print('line_strength_matrix matches')
    else:
        print('line_strength_matrix Does not match')

def test_non_radiative_rate_matrix():
    # Check non_radiative_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'spectral_kinetics/Yb10_Er2/M_NRrate.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0: continue
            data.append([float(val) for val in line.strip().split(',')])
    m_srad = np.array(data)
    matches = np.allclose(m_srad, sk.non_radiative_rate_matrix)
    if matches:
        print('non_radiative_rate_matrix matches')
    else:
        print('non_radiative_rate_matrix Does not match')


def test_radiative_rate_matrix():
    # Check radiative_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'spectral_kinetics/Yb10_Er2/M_RadRate.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0: continue
            data.append([float(val) for val in line.strip().split(',')])
    m_radrate = np.array(data)
    matches = np.allclose(m_radrate, sk.radiative_rate_matrix)
    if matches:
        print('radiative_rate_matrix matches')
    else:
        print('radiative_rate_matrix Does not match')

def test_magnetic_dipole_rate_matrix():
    # Check magnetic_dipole_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'spectral_kinetics/Yb10_Er2/M_MDradRate.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0: continue
            data.append([float(val) for val in line.strip().split(',')])
    m_mdradrate = np.array(data)
    matches = np.allclose(m_mdradrate, sk.magnetic_dipole_rate_matrix)
    if matches:
        print('magnetic_dipole_rate_matrix matches')
    else:
        print('magnetic_dipole_rate_matrix Does not match')

def test_energy_transfer_rate_matrix():
    # Check energy_transfer_rate_matrix
    with open(os.path.join(TEST_FILE_DIR, 'spectral_kinetics/Yb10_Er2/W_ETrates.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0: continue
            data.append([float(val) for val in line.strip().split(',')])
    rates = np.array(data)

    with open(os.path.join(TEST_FILE_DIR, 'spectral_kinetics/Yb10_Er2/M_ETindices.txt'), 'r') as f:
        lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            if i == 0: continue
            data.append([float(val) for val in line.strip().split(',')])
    indices = np.array(data)
    et_rates = np.hstack([indices, rates])

    matches = np.allclose(et_rates, sk.energy_transfer_rate_matrix)
    if matches:
        print('energy_transfer_rate_matrix matches')
    else:
        print('energy_transfer_rate_matrix Does not match')
