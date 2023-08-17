from NanoParticleTools.inputs.nanoparticle import Dopant
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.util.conversions import wavenumber_to_wavelength
from functools import lru_cache
import numpy as np
from typing import Dict, List, Tuple


def get_wavelengths(sk: SpectralKinetics) -> np.ndarray:
    """
    Gets the emission wavelengths for all possible radiative transitions in the system.

    Args:
        sk: Spectral kinetics object which defines the dopants and rates in the system.

    Returns:
        wavelengths: A NxN matrix of wavelengths, where N is the total number
            of energy levels in the system. Each element, W_{ij} corresponds
            to the wavelength of light emitted for a transition from level i
            to level j.
    """
    all_energy_levels = np.hstack(
        [[energy_level.energy for energy_level in dopants.energy_levels]
         for dopants in sk.dopants])
    wavelengths = wavenumber_to_wavelength(
        (all_energy_levels[None, :] - all_energy_levels[:, None]))
    return wavelengths


def intensities_from_population(sk: SpectralKinetics,
                                populations: np.ndarray,
                                volume: float | int,
                                last_n_avg: int = 200):
    """
    A helper function to compute the intensities from the population vs time.

    Usually, this is called if processing the output of a simulation.

    Args:
        sk: Spectral kinetics object which defines the dopants and rates in the system.
        populations: An array of shape (M, T, N), where M is the total number
            of control volumes in the system, T is the number of time steps,
            and N is the total number of energy levels in the system. Each
            element, P_{ij} corresponds to the population of level i at time j.
        volume: The volume of the system in A^3.
        last_n_avg: The number of time steps to average over. The timesteps are chosen
            from the end of the simulation, in an attempt to avoid pre-equilibrated values.

    Returns:
        wavelengths: A NxN matrix of wavelengths, where N is the total number
            of energy levels in the system. Each element, W_{ij} corresponds
            to the wavelength of light emitted for a transition from level i
            to level j.
        intensities: A NxN matrix of intensities, where N is the total number
            of energy levels in the system. Each element, I_{ij} corresponds
            to the intensity or counts of the i to j transition.
    """
    wavelengths = get_wavelengths(sk)
    mean_populations = populations[..., -last_n_avg:, :].mean(-2)
    intensities = mean_population_to_intensities(sk, mean_populations, volume)
    return wavelengths, intensities


def mean_population_to_intensities(
        sk: SpectralKinetics, mean_populations: np.ndarray,
        volume: float | np.ndarray) -> np.ndarray:
    """
    Use the mean population of each energy level to compute the mean intensity
    of each radiative transition.

    This function can also accept lists of populations and volumes,
    so long as they are of they are the same length.

    Args:
        sk: Spectral kinetics object which defines the dopants and rates in the system.
        mean_populations: An array of length N, where N is the total number
            of energy levels in the system. Each element, P_i corresponds
            to the mean population of level i.
        volume: The volume of the system in A^3.

    Returns:
        intensities: A NxN matrix of intensities, where N is the total number
            of energy levels in the system. Each element, I_{ij} corresponds
            to the intensity or counts of the i to j transition.
    """
    intensities = volume * mean_populations[...,
                                            None] * sk.radiative_rate_matrix
    return intensities


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


def get_spectrum_wavelength_from_intensities(
        wavelengths,
        intensities,
        lower_bound=-2000,
        upper_bound=1000,
        step=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes the intensity matrix to get a spectrum of intensity with respect
    to wavelength.

    Positive wavelengths correspond to absorption, negative wavelengths
    correspond to emission.

    Note: If the number of dopants in the simulation are >1, then wavelengths
        and intensities will be block diagonal, since there are no 2-site
        interactions that give rise to radiative emissions.
    Args:
        wavelengths: A NxN matrix of wavelengths, where N is the total number
            of energy levels in the system. Each element, W_{ij} corresponds
            to the wavelength of light emitted for a transition from level i
            to level j.
        intensities: A NxN matrix of intensities, where N is the total number
            of energy levels in the system. Each element, I_{ij} corresponds
            to the intensity or counts of the i to j transition.
        lower_bound: The lower limit to x-axis.
        upper_bound: The upper limit to x-axis.
        step: The step size for the x-axis. The number of bins total will be
            (upper_bound - lower_bound) / step

    Returns:
        x: The x-axis of the spectrum.
        spectrum: The intensities at each x value.
    """
    wavelengths = wavelengths[intensities != 0]
    intensities = intensities[intensities != 0]
    _x = np.arange(lower_bound, upper_bound + step, step)
    x = (_x[:-1] + _x[1:]) / 2  # middle point of each bin

    spectrum = np.zeros(x.shape)
    for wavelength, intensity in zip(wavelengths, intensities):
        if wavelength > lower_bound and wavelength < upper_bound:
            index = int(np.floor((wavelength - lower_bound) / step))
            spectrum[index] += intensity
    return x, spectrum


def get_spectrum_energy_from_dndt(
        avg_dndt: Dict,
        dopants: List[Dopant],
        lower_bound: int = -40000,
        upper_bound: int = 20000,
        step: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes the dNdt table obtained from Monte Carlo simulations to get a spectrum of
    Intensity of radiated transitions vs energy.

    Intensity is in units of counts per second and energy is in units of cm^-1

    Positive energy correspond to absorption, negative energy
    correspond to emission.

    Args:
        avg_dndt: The dNdt table in dictionary format. See average_dndt for more details.
        dopants: A list of dopant objects.
        lower_bound: The lower limit to x-axis.
        upper_bound: The upper limit to x-axis.
        step: The step size for the x-axis. The number of bins total will be
            (upper_bound - lower_bound) / step

    Returns:
        x: The x-axis of the spectrum.
        spectrum: The intensities at each x value.
    """
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


def get_spectrum_wavelength_from_dndt(
        avg_dndt: Dict,
        dopants: List[Dopant],
        lower_bound: int = -2000,
        upper_bound: int = 1000,
        step: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes the dNdt table obtained from Monte Carlo simulations to get a spectrum of
    Intensity of radiated transitions vs wavelength.

    Intensity is in units of counts per second and wavelength is in units of nm.

    Positive wavelengths correspond to absorption, negative wavelengths
    correspond to emission.

    Args:
        avg_dndt: The dNdt table in dictionary format. See average_dndt for more details.
        dopants: A list of dopant objects.
        lower_bound: The lower limit to x-axis.
        upper_bound: The upper limit to x-axis.
        step: The step size for the x-axis. The number of bins total will be
            (upper_bound - lower_bound) / step

    Returns:
        x: The x-axis of the spectrum.
        spectrum: The intensities at each x value.
    """
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
