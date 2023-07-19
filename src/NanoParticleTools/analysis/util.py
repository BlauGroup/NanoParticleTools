from NanoParticleTools.inputs.nanoparticle import Dopant
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.util.conversions import wavenumber_to_wavelength
from functools import lru_cache
import numpy as np
from typing import Dict, List


def get_wavelengths(sk):
    all_energy_levels = np.hstack(
        [[energy_level.energy for energy_level in dopants.energy_levels]
         for dopants in sk.dopants])

    return wavenumber_to_wavelength(
        (all_energy_levels[None, :] - all_energy_levels[:, None]))


def intensities_from_population(sk, populations, volume, last_n_avg=200):
    wavelengths = get_wavelengths(sk)
    intensities = population_to_intensities(sk, populations, volume,
                                            last_n_avg)
    return wavelengths, intensities


def intensities_from_docs(docs, last_n_avg=200):
    dopants = [
        Dopant(key, val) for key, val in docs[0]['data']
        ['overall_dopant_concentration'].items()
    ]
    sk = SpectralKinetics(
        dopants,
        excitation_wavelength=docs[0]['data']['excitation_wavelength'],
        excitation_power=docs[0]['data']['excitation_power'])

    overall_population = np.array([
        doc['data']['output']['y_overall_populations'] for doc in docs
    ]).mean(0)
    # overall volume
    volume = 4 / 3 * np.pi * (
        docs[0]['data']['input']['constraints'][-1]['radius'] / 10)**3

    return intensities_from_population(sk, overall_population, volume,
                                       last_n_avg)


def population_to_intensities(sk, populations, volume, last_n_avg=200):
    mean_populations = populations[..., -last_n_avg:, :].mean(-2)
    intensities = volume * mean_populations[...,
                                            None] * sk.radiative_rate_matrix
    return intensities


def get_spectrum_energy_from_dndt(avg_dndt,
                                  dopants,
                                  lower_bound=-40000,
                                  upper_bound=20000,
                                  step=100):
    _x = np.arange(lower_bound, upper_bound + step, step)
    x = (_x[:-1] + _x[1:]) / 2  # middle point of each bin

    spectrum = np.zeros(x.shape)

    for interaction in [_d for _d in avg_dndt if _d[8] == "Rad"]:
        species_id = interaction[2]
        left_state_1 = interaction[4]
        right_state_1 = interaction[6]
        ei = dopants[species_id].energy_levels[left_state_1]
        ef = dopants[species_id].energy_levels[right_state_1]

        de = ef.energy - ei.energy
        if de > lower_bound and de < upper_bound:
            index = int(np.floor((de - lower_bound) / step))
            spectrum[index] += interaction[10]

    return x, spectrum


def get_spectrum_wavelength_from_dndt(avg_dndt,
                                      dopants,
                                      lower_bound=-2000,
                                      upper_bound=1000,
                                      step=5):
    _x = np.arange(lower_bound, upper_bound + step, step)
    x = (_x[:-1] + _x[1:]) / 2  # middle point of each bin

    spectrum = np.zeros(x.shape)

    for interaction in [_d for _d in avg_dndt if _d[8] == "Rad"]:
        species_id = interaction[2]
        left_state_1 = interaction[4]
        right_state_1 = interaction[6]
        ei = dopants[species_id].energy_levels[left_state_1]
        ef = dopants[species_id].energy_levels[right_state_1]

        de = ef.energy - ei.energy
        wavelength = (299792458 * 6.62607004e-34) / (de * 1.60218e-19 /
                                                     8065.44) * 1e9
        if wavelength > lower_bound and wavelength < upper_bound:
            index = int(np.floor((wavelength - lower_bound) / step))
            spectrum[index] += interaction[10]
    return x, spectrum


def average_dndt(docs: List[Dict]) -> Dict:
    """
    Compute the average dndt for all interactions

    Args:
        docs (List[Dict]): List of taskdocs to average

    Returns:
        Dict: Output will have the following fields:
            ["interaction_id", "number_of_sites", "species_id_1",
            "species_id_2", "left_state_1", "left_state_2",
            "right_state_1", "right_state_2", "interaction_type",
            "rate_coefficient", "dNdT", "std_dev_dNdt",
            "dNdT per atom", "std_dev_dNdT per atom", "occurences",
            "std_dev_occurences", "occurences per atom",
            "std_dev_occurences per atom"]
    """
    accumulated_dndt = {}
    n_docs = 0
    for doc in docs:
        n_docs += 1
        keys = doc["data"]["output"]["summary_keys"]
        try:
            search_keys = [
                "interaction_id", "number_of_sites", "species_id_1",
                "species_id_2", "left_state_1", "left_state_2",
                "right_state_1", "right_state_2", "interaction_type",
                'rate_coefficient', 'dNdT', 'dNdT per atom', 'occurences',
                'occurences per atom'
            ]
            indices = []
            for key in search_keys:
                indices.append(keys.index(key))
        except KeyError:
            search_keys = [
                "interaction_id", "number_of_sites", "species_id_1",
                "species_id_2", "left_state_1", "left_state_2",
                "right_state_1", "right_state_2", "interaction_type", 'rate',
                'dNdT', 'dNdT per atom', 'occurences', 'occurences per atom'
            ]
            indices = []
            for key in search_keys:
                indices.append(keys.index(key))

        dndt = doc["data"]["output"]["summary"]

        for interaction in dndt:
            interaction_id = interaction[0]
            if interaction_id not in accumulated_dndt:
                accumulated_dndt[interaction_id] = []
            accumulated_dndt[interaction_id].append(
                [interaction[i] for i in indices])

    avg_dndt = []
    for interaction_id in accumulated_dndt:

        arr = accumulated_dndt[interaction_id][-1][:-4]

        _dndt = [_arr[-4:] for _arr in accumulated_dndt[interaction_id]]

        while len(_dndt) < n_docs:
            _dndt.append([0 for _ in range(4)])

        mean = np.mean(_dndt, axis=0)
        std = np.std(_dndt, axis=0)
        arr.extend([
            mean[0], std[0], mean[1], std[1], mean[2], std[2], mean[3], std[3]
        ])
        avg_dndt.append(arr)
    return avg_dndt
