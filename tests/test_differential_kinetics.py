from NanoParticleTools.differential_kinetics import (get_parser,
                                                     run_one_rate_eq,
                                                     run_and_save_one,
                                                     save_data_to_hdf5,
                                                     load_data_from_hdf5)
from NanoParticleTools.species_data import Dopant

import numpy as np
import h5py

import pytest


@pytest.fixture
def hdf5_file(tmp_path):
    return h5py.File(tmp_path / 'test.h5', 'w')


def test_parser():
    args = get_parser().parse_args(['-n', '10', '-m', '2', '-o', 'test.h5'])
    assert args.num_samples == 10
    assert args.max_dopants == 2
    assert args.output_file == 'test.h5'
    assert args.include_spectra is False
    assert args.excitation_wavelength == (700, 1500)
    assert args.excitation_power == (10, 1e6)
    assert args.dopants == ["Yb", "Er", "Tm", "Nd", "Ho", "Eu", "Sm", "Dy"]


def test_run_one_rate_eq():
    dopants = [Dopant('Yb', 0.1), Dopant('Er', 0.1)]
    out_dict = run_one_rate_eq(dopants,
                               excitation_wavelength=800,
                               excitation_power=1e5)

    assert list(out_dict['metadata'].keys()) == [
        'simulation_time', 'excitation_wavelength', 'excitation_power',
        'dopant_concentrations'
    ]


def test_run_and_save_one(hdf5_file):
    dopants = ['Yb', 'Er']
    dopant_concs = [0.5, 0.2]
    run_and_save_one(dopants, dopant_concs, 0, 3, hdf5_file, 
                     excitation_wavelength=900,
                     excitation_power=1e5,
                     include_spectra=True)

    out = load_data_from_hdf5(hdf5_file, 0, 3)
    assert 'wavelength_x_range' in out['metadata'].keys()
    assert 'wavelength_x_step' in out['metadata'].keys()
    assert 'wavelength_spectrum_y' in out.keys()


def test_save_load(hdf5_file):
    data = {'metadata': {'simulation_time': 0.01,
                         'dopant_concentration': {'Yb': 0.5, 'Er': 0.1}},
            'populations': np.random.rand(10)}

    save_data_to_hdf5(hdf5_file, 0, 0, data)

    out = load_data_from_hdf5(hdf5_file, 0, 0)
    assert out['metadata'] == data['metadata']
    assert np.allclose(out['populations'], data['populations'])

    data = {'metadata': {'simulation_time': 0.01,
                         'dopant_concentration': {'Yb': 0.5, 'Er': 0.1}},
            'populations': np.random.rand(10),
            'wavelength_spectrum_y': np.random.rand(600)}

    save_data_to_hdf5(hdf5_file, 0, 1, data)

    out = load_data_from_hdf5(hdf5_file, 0, 1)
    assert out['metadata'] == data['metadata']
    assert np.allclose(out['populations'], data['populations'])
    assert np.allclose(out['wavelength_spectrum_y'], data['wavelength_spectrum_y'])
