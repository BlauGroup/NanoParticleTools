from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.species_data import Dopant

from NanoParticleTools.analysis import get_wavelengths, mean_population_to_intensities
from NanoParticleTools.analysis.util import get_spectrum_wavelength_from_intensities
from NanoParticleTools.util.conversions import wavenumber_to_wavelength

import numpy as np

from datetime import datetime
from joblib import Parallel, delayed
import argparse
import sys
import json
import h5py


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n',
                        '--num_samples',
                        help='The number of samples to run',
                        type=int,
                        required=True)
    parser.add_argument(
        '-w',
        '--excitation_wavelength',
        help=('The rage of excitation wavelength in nm.'
              ' The wavelength will be sampled from a uniform distribution'),
        type=float,
        nargs=2,
        default=(700, 1500))
    parser.add_argument(
        '-p',
        '--excitation_power',
        help=('The range of excitation power in W/cm^2.'
              ' The power will be sampled from a log uniform distribution'),
        type=float,
        nargs=2,
        default=(10, 1e6))

    # add optional arguments
    parser.add_argument(
        '-d',
        '--dopants',
        help='The possible dopants to include in the simulation',
        nargs='+',
        type=list,
        default=["Yb", "Er", "Tm", "Nd", "Ho", "Eu", "Sm", "Dy"])
    # default=["Yb", "Er", "Tm", "Nd", "Ho", "Eu", "Tb", "Sm", "Dy"]) # exclude Tb for now

    parser.add_argument(
        '-m',
        '--max_dopants',
        help='The maximum number of dopants to include in the simulation',
        type=int,
        default=4)

    parser.add_argument(
        '-s',
        '--include_spectra',
        help='Whether to compute the spectra and save in the output',
        action='store_true')
    parser.add_argument(
        '-o',
        '--output_file',
        help='The output file to save the data to',
        type=str,
        default=f'{datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")}.h5')

    parser.add_argument('-j',
                        '--num_workers',
                        help='Number of concurrent workers',
                        type=int,
                        default=1)

    parser.add_argument(
        '-g',
        '--max_data_per_group',
        help='The maximum number of data points to write to each hdf5 group',
        type=int,
        default=1e5)

    return parser


def run_one_rate_eq(dopants: list[Dopant],
                    excitation_wavelength: int,
                    excitation_power: float,
                    include_spectra: bool = False,
                    spectra_x_range: tuple[float, float] = (-2000, 1000),
                    spectra_x_step: int | float = 5,
                    volume: float | int = 1):

    sk = SpectralKinetics(dopants,
                          excitation_wavelength=excitation_wavelength,
                          excitation_power=excitation_power)
    converged = False
    initial_populations = 'ground_state'
    max_time = 1e-3

    t = None
    pops = None
    while not converged:
        sol = sk.run_kinetics(initial_populations, t_span=(0, max_time))
        if t is None:
            t = sol[0]
            pops = sol[1]
        else:
            # Add the last time point from the previous to the new time points
            _t = sol[0] + t[-1]

            t = np.append(t, _t[1:])
            pops = np.append(pops, sol[1][1:], axis=0)

        # Simple convergence criteria: Check if the last two populations are within some tolerance
        converged = np.allclose(pops[-1], pops[-2])

        # If not converged, then set the initial populations to the last populations
        # and increase the max integration time
        initial_populations = pops[-1]
        max_time *= 10

    final_pop = sol[1][-1]
    out_dict = {
        'metadata': {
            'simulation_time': t[-1],
            'excitation_wavelength': excitation_wavelength,
            'excitation_power': excitation_power,
            'dopant_concentrations':
            {dopant.symbol: dopant.molar_concentration
             for dopant in dopants}
        }
    }
    out_dict['populations'] = final_pop

    if include_spectra:
        intensities = mean_population_to_intensities(sk, final_pop, volume)
        wavelengths = get_wavelengths(sk)
        _, y = get_spectrum_wavelength_from_intensities(
            wavelengths, intensities, *spectra_x_range, spectra_x_step)

        # To save space, we omit the x values from the output, since they can
        # be easily reconstructed from the x range and step
        out_dict['metadata']['wavelength_x_range'] = spectra_x_range
        out_dict['metadata']['wavelength_x_step'] = spectra_x_step

        out_dict["wavelength_spectrum_y"] = y

    return out_dict


def run_and_save_one(dopants: list[str], dopant_concs: list[float],
                     group_id: int, data_i: int, file: h5py.File, **kwargs):
    dopants = [Dopant(el, x) for el, x in zip(dopants, dopant_concs)]
    out_dict = run_one_rate_eq(dopants, **kwargs)

    save_data_to_hdf5(file, group_id, data_i, out_dict)

    return out_dict


def save_data_to_hdf5(file: h5py.File, group_id: int, data_i: int, data: dict):
    worker_group = file.require_group(f'group_{group_id}')
    data_group = worker_group.create_group(f'data_{data_i}')
    data_group.create_dataset('metadata', data=json.dumps(data['metadata']))
    data_group.create_dataset('populations', data=data['populations'])

    if 'wavelength_spectrum_y' in data:
        data_group.create_dataset('wavelength_spectrum_y',
                                  data=data['wavelength_spectrum_y'])


def load_data_from_hdf5(file: h5py.File, group_id: int, data_i: int):
    data = file[f'group_{group_id}/data_{data_i}']

    out_dict = {}
    out_dict['metadata'] = json.loads(data['metadata'][()])
    out_dict['populations'] = data['populations'][()]
    if 'wavelength_spectrum_y' in data:
        out_dict['wavelength_spectrum_y'] = data['wavelength_spectrum_y'][()]

    return out_dict
