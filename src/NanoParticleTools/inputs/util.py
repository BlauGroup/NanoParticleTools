from functools import lru_cache

import numpy as np

from NanoParticleTools.inputs import (DopedNanoparticle, SpectralKinetics)
from NanoParticleTools.species_data.species import Dopant
from pymatgen.core import Composition
from collections import Counter

from typing import List, Union, Tuple


def specie_energy_level_to_combined_energy_level(species: Union[str, int,
                                                                Dopant],
                                                 energy_level: int,
                                                 dopants: List[Dopant]) -> int:
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

    sum_level = sum([dopant.n_levels for dopant in dopants[:dopant_index]])
    if energy_level >= dopants[dopant_index].n_levels:
        raise ValueError("Invalid energy level specified")
    else:
        return sum_level + energy_level


@lru_cache(maxsize=None)
def get_energy_level_maps(sk: SpectralKinetics) -> Tuple[dict, dict, dict]:
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

    return (energy_level_map, energy_level_to_species_id,
            energy_level_to_species_name)


def combined_energy_level_to_specie_energy_level(sk: SpectralKinetics,
                                                 energy_level: int) -> int:
    energy_level_map, _, _ = get_energy_level_maps(sk)
    return energy_level_map[energy_level]


def combined_energy_level_to_specie_id(sk: SpectralKinetics,
                                       energy_level: int) -> int:
    _, energy_level_to_species_id, _ = get_energy_level_maps(sk)
    return energy_level_to_species_id[energy_level]


def combined_energy_level_to_specie_name(sk: SpectralKinetics,
                                         energy_level: int) -> str:
    _, _, energy_level_to_species_name = get_energy_level_maps(sk)
    return energy_level_to_species_name[energy_level]


def get_non_radiative_interactions(sk: SpectralKinetics) -> List[dict]:
    _interactions = []

    rows, cols = np.where(sk.non_radiative_rate_matrix != 0)
    for row, col, rate in zip(rows, cols, sk.non_radiative_rate_matrix[rows,
                                                                       cols]):
        _d = {
            'interaction_id': None,
            'number_of_sites': 1,
            'species_id_1':
                combined_energy_level_to_specie_id(sk, row),
            'species_id_2': -1,
            'left_state_1':
                combined_energy_level_to_specie_energy_level(sk, row),
            'left_state_2': -1,
            'right_state_1':
                combined_energy_level_to_specie_energy_level(sk, col),
            'right_state_2': -1,
            'rate': rate,
            'interaction_type': 'NR'
        }
        _interactions.append(_d)
    return _interactions


def get_radiative_interactions(sk: SpectralKinetics) -> List[dict]:
    _interactions = []

    rows, cols = np.where(sk.radiative_rate_matrix != 0)
    for row, col, rate in zip(rows, cols, sk.radiative_rate_matrix[rows,
                                                                   cols]):
        _d = {
            'interaction_id': None,
            'number_of_sites': 1,
            'species_id_1':
                combined_energy_level_to_specie_id(sk, row),
            'species_id_2': -1,
            'left_state_1':
                combined_energy_level_to_specie_energy_level(sk, row),
            'left_state_2': -1,
            'right_state_1':
                combined_energy_level_to_specie_energy_level(sk, col),
            'right_state_2': -1,
            'rate': rate,
            'interaction_type': 'Rad'
        }
        _interactions.append(_d)
    return _interactions


def get_magnetic_dipole_interactions(sk: SpectralKinetics) -> List[dict]:
    _interactions = []

    rows, cols = np.where(sk.magnetic_dipole_rate_matrix != 0)
    for row, col, rate in zip(rows, cols,
                              sk.magnetic_dipole_rate_matrix[rows, cols]):
        _d = {
            'interaction_id': None,
            'number_of_sites': 1,
            'species_id_1':
                combined_energy_level_to_specie_id(sk, row),
            'species_id_2': -1,
            'left_state_1':
                combined_energy_level_to_specie_energy_level(sk, row),
            'left_state_2': -1,
            'right_state_1':
                combined_energy_level_to_specie_energy_level(sk, col),
            'right_state_2': -1,
            'rate': rate,
            'interaction_type': 'MD'
        }
        _interactions.append(_d)
    return _interactions


def get_energy_transfer_interactions(sk: SpectralKinetics) -> List[dict]:
    _interactions = []
    for di, dj, ai, aj, rate in sk.energy_transfer_rate_matrix:
        _d = {
            'interaction_id': None,
            'number_of_sites': 2,
            'species_id_1':
                combined_energy_level_to_specie_id(sk, di),
            'species_id_2':
                combined_energy_level_to_specie_id(sk, ai),
            'left_state_1':
                combined_energy_level_to_specie_energy_level(sk, di),
            'left_state_2':
                combined_energy_level_to_specie_energy_level(sk, ai),
            'right_state_1':
                combined_energy_level_to_specie_energy_level(sk, dj),
            'right_state_2':
                combined_energy_level_to_specie_energy_level(sk, aj),
            'rate': (1.0e42) * rate,
            'interaction_type': 'ET'
        }
        _interactions.append(_d)
    return _interactions


def get_all_interactions(sk: SpectralKinetics) -> dict:
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


def get_sites(nanoparticle: DopedNanoparticle, sk: SpectralKinetics):
    sites = {}
    for i, site in enumerate(nanoparticle.dopant_sites):
        sites[i] = {
            'site_id':
            i,
            'x':
            site.x / 10,
            'y':
            site.y / 10,
            'z':
            site.z / 10,
            'species_id':
            [dopant.symbol for dopant in sk.dopants].index(site.specie.symbol)
        }
    return sites


def get_species(sk: SpectralKinetics):
    species = {}
    for i, dopant in enumerate(sk.dopants):
        species[i] = {'species_id': i, 'degrees_of_freedom': dopant.n_levels}

    return species


def get_formula_by_constraint(nanoparticle: DopedNanoparticle):
    sites = np.array([site.coords for site in nanoparticle.dopant_sites])
    species_names = np.array(
        [str(site.specie) for site in nanoparticle.dopant_sites])

    # Get the indices for dopants within each constraint
    site_indices_by_constraint = []
    for constraint in nanoparticle.constraints:

        indices_inside_constraint = set(
            np.where(constraint.sites_in_bounds(sites))[0])

        for ids in site_indices_by_constraint:
            indices_inside_constraint = indices_inside_constraint - set(ids)
        site_indices_by_constraint.append(
            sorted(list(indices_inside_constraint)))

    formula_by_constraint = []
    for indices in site_indices_by_constraint:
        formula = Composition(Counter(species_names[indices])).formula.replace(
            " ", "")
        formula_by_constraint.append(formula)
    return formula_by_constraint
