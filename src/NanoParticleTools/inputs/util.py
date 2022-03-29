from functools import lru_cache

import numpy as np

from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.species_data.species import Dopant

@lru_cache()
def specie_energy_level_to_combined_energy_level(species, energy_level, dopants):
    if isinstance(species, str):
        for i, dopant in enumerate(dopants):
            if dopant.symbol == species:
                break
        dopant_index = i
    elif isinstance(species, int):
        dopant_index = species
    elif isinstance(species, Dopant):
        for i, dopant in enumerate(dopants):
            if dopant.symbol == species.symbol:
                break
        dopant_index = i
    else:
        raise ValueError("Invalid species specified")

    return sum([dopant.n_levels for dopant in dopants[:dopant_index]]) + energy_level

def energy_level_to_species_id(sk):
    energy_level_to_species_id = {}
    energy_level_to_species_name = {}
    counter = 0
    for i, dopant in enumerate(sk.dopants):
        for j in range(dopant.n_levels):
            energy_level_to_species_id[counter] = i
            energy_level_to_species_name[counter] = dopant.symbol
            counter += 1


@lru_cache(maxsize=None)
def get_energy_level_maps(sk):
    energy_level_map = {}
    energy_level_to_species_id = {}
    energy_level_to_species_name = {}
    counter = 0
    for i, dopant in enumerate(sk.dopants):
        for j in range(dopant.n_levels):
            energy_level_map[counter] = j
            energy_level_to_species_id[counter] = i
            energy_level_to_species_name[counter] = dopant.symbol
            counter += 1

    return energy_level_map, energy_level_to_species_id, energy_level_to_species_name


def combined_energy_level_to_specie_energy_level(sk, energy_level):
    energy_level_map, energy_level_to_species_id, energy_level_to_species_name = get_energy_level_maps(sk)
    return energy_level_map[energy_level]


def combined_energy_level_to_specie_id(sk, energy_level):
    energy_level_map, energy_level_to_species_id, energy_level_to_species_name = get_energy_level_maps(sk)
    return energy_level_to_species_id[energy_level]


def combined_energy_level_to_specie_name(sk, energy_level):
    energy_level_map, energy_level_to_species_id, energy_level_to_species_name = get_energy_level_maps(sk)
    return energy_level_to_species_name[energy_level]


def get_non_radiative_interactions(sk: SpectralKinetics):
    _interactions = []

    rows, cols = np.where(sk.non_radiative_rate_matrix != 0)
    for row, col, rate in zip(rows, cols, sk.non_radiative_rate_matrix[rows, cols]):
        _d = {'interaction_id': None,
              'number_of_sites': 1,
              'species_id_1': combined_energy_level_to_specie_id(sk, row),
              'species_id_2': -1,
              'left_state_1': combined_energy_level_to_specie_energy_level(sk, row),
              'left_state_2': -1,
              'right_state_1': combined_energy_level_to_specie_energy_level(sk, col),
              'right_state_2': -1,
              'rate': rate,
              'interaction_type': 'NR'}
        _interactions.append(_d)
    return _interactions


def get_radiative_interactions(sk: SpectralKinetics):
    _interactions = []

    rows, cols = np.where(sk.radiative_rate_matrix != 0)
    for row, col, rate in zip(rows, cols, sk.radiative_rate_matrix[rows, cols]):
        _d = {'interaction_id': None,
              'number_of_sites': 1,
              'species_id_1': combined_energy_level_to_specie_id(sk, row),
              'species_id_2': -1,
              'left_state_1': combined_energy_level_to_specie_energy_level(sk, row),
              'left_state_2': -1,
              'right_state_1': combined_energy_level_to_specie_energy_level(sk, col),
              'right_state_2': -1,
              'rate': rate,
              'interaction_type': 'Rad'}
        _interactions.append(_d)
    return _interactions


def get_magnetic_dipole_interactions(sk: SpectralKinetics):
    _interactions = []

    rows, cols = np.where(sk.magnetic_dipole_rate_matrix != 0)
    for row, col, rate in zip(rows, cols, sk.magnetic_dipole_rate_matrix[rows, cols]):
        _d = {'interaction_id': None,
              'number_of_sites': 1,
              'species_id_1': combined_energy_level_to_specie_id(sk, row),
              'species_id_2': -1,
              'left_state_1': combined_energy_level_to_specie_energy_level(sk, row),
              'left_state_2': -1,
              'right_state_1': combined_energy_level_to_specie_energy_level(sk, col),
              'right_state_2': -1,
              'rate': rate,
              'interaction_type': 'MD'}
        _interactions.append(_d)
    return _interactions


def get_energy_transfer_interactions(sk: SpectralKinetics):
    _interactions = []
    for di, dj, ai, aj, rate in sk.energy_transfer_rate_matrix:
        _d = {'interaction_id': None,
              'number_of_sites': 2,
              'species_id_1': combined_energy_level_to_specie_id(sk, di),
              'species_id_2': combined_energy_level_to_specie_id(sk, ai),
              'left_state_1': combined_energy_level_to_specie_energy_level(sk, di),
              'left_state_2': combined_energy_level_to_specie_energy_level(sk, ai),
              'right_state_1': combined_energy_level_to_specie_energy_level(sk, dj),
              'right_state_2': combined_energy_level_to_specie_energy_level(sk, aj),
              'rate': (1.0e42) * rate,
              'interaction_type': 'ET'}
        _interactions.append(_d)
    return _interactions


def get_all_interactions(sk):
    all_interactions = []
    all_interactions.extend(get_non_radiative_interactions(sk))
    all_interactions.extend(get_radiative_interactions(sk))
    all_interactions.extend(get_magnetic_dipole_interactions(sk))
    all_interactions.extend(get_energy_transfer_interactions(sk))

    _all_interactions = {}
    for i in range(len(all_interactions)):
        all_interactions[i]['interaction_id'] = i
        _all_interactions[i] = all_interactions[i]

    return _all_interactions


def get_sites(nanoparticle, sk):
    sites = {}
    for i, site in enumerate(nanoparticle.dopant_sites):
        sites[i] = {'site_id': i,
                    'x': site.x/10,
                    'y': site.y/10,
                    'z': site.z/10,
                    'species_id': [dopant.symbol for dopant in sk.dopants].index(site.specie.symbol)}
    return sites


def get_species(sk):
    species = {}
    for i, dopant in enumerate(sk.dopants):
        species[i] = {'species_id': i,
                      'degrees_of_freedom': dopant.n_levels}

    return species