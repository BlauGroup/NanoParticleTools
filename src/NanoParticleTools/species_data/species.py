from monty.json import MSONable
import json
from pathlib import Path
import os
from typing import Optional, Dict, List, Union
import numpy as np
from functools import lru_cache

SPECIES_DATA_PATH = os.path.join(str(Path(__file__).absolute().parent), 'data')


class EnergyLevel(MSONable):
    """
    Class to represent an energy level.

    Args:
        element (str): The element which the energy level belongs to.
        label (str): The label of the energy level
        energy (float): The energy of the energy level in $cm^{-1}$.
    """
    def __init__(self, element: str, label: str, energy: float):
        self.element = element
        self.label = label
        self.energy = energy

    def __str__(self) -> str:
        return f'{self.element} - {self.label} - {self.energy}'

    def __repr__(self) -> str:
        return self.__str__()


class Transition(MSONable):
    """
    Class to represent a transition between two energy levels.

    Args:
        initial_level (EnergyLevel): The initial energy level of the
            transition.
        final_level (EnergyLevel): The final energy level of the transition.
        line_strength (float): The line strength of the transition.
    """
    def __init__(self, initial_level: EnergyLevel, final_level: EnergyLevel,
                 line_strength: float):
        self.initial_level = initial_level
        self.final_level = final_level
        self.line_strength = line_strength

    def __str__(self) -> str:
        return f'{self.initial_level.label} -> {self.final_level.label}'

    def __repr__(self) -> str:
        return self.__str__()


class Dopant(MSONable):
    """
    Class to represent a dopant species. The dopant doesn't represent a
    singular atom, rather the composition of the dopant on the nanoparticle
    lattice.

    This inherits from MSONable, allowing it to be serialized as any pymatgen
    object would be (using monty.serialization).

    Note: The surface dopants are mapped to ther elements which are unlikely to
        be used in Upconversion Nanoparticles. This a bit of a hack, but allows
        us to use the pymatgen Structure, Element, and Site classes when
        building the nanoparticle structure and applying dopants to the
        lattice. The surface dopants are mapped to Na, Mg, Al, Si, and P,
        as defined in SURFACE_DOPANT_SYMBOLS_TO_NAMES and
        SURFACE_DOPANT_NAMES_TO_SYMBOLS.

    Args:
        symbol (str): The symbol of the dopant. If the dopant is a surface,
            the surface name or its corresponding element can be specified.
            The lookup is done under the hood.
        molar_concentration (float): The fractional composition of the dopant,
            ranging from 0 to 1
        n_levels (int): The number of this dopant's levels to use in the
            calculation. If not specified, all levels will be used.
    """

    SURFACE_DOPANT_SYMBOLS_TO_NAMES = {
        'Na': 'Surface',
        'Mg': 'Surface6',
        'Al': 'Surface3',
        'Si': 'Surface4',
        'P': 'Surface5'
    }
    SURFACE_DOPANT_NAMES_TO_SYMBOLS = dict(
            zip(SURFACE_DOPANT_SYMBOLS_TO_NAMES.values(),
                SURFACE_DOPANT_SYMBOLS_TO_NAMES.keys()))

    def __init__(self,
                 symbol: str,
                 molar_concentration: float,
                 n_levels: Optional[int] = None):
        if symbol in self.SURFACE_DOPANT_NAMES_TO_SYMBOLS:
            symbol = self.SURFACE_DOPANT_NAMES_TO_SYMBOLS[symbol]
        self.symbol = symbol
        self.molar_concentration = molar_concentration

        # If more levels are specified than possible, use the max
        # number of levels instead
        if n_levels is None:
            self.n_levels = len(self.energy_levels)
        else:
            self.n_levels = min(n_levels, len(self.energy_levels))

    def check_intrinsic_data(self) -> bool:
        """
        Checks whether the dopant data is valid, else raises an error.

        Returns:
            bool: True if the data is valid
        """
        if self.symbol in self.SURFACE_DOPANT_SYMBOLS_TO_NAMES.keys():
            # Surface will not have these parameters
            return True
        if self.eigenvector_sl.shape[
                0] != self.intermediate_coupling_coefficients.shape[1]:
            raise ValueError(
                'Error: The number of eigenvectors does not match'
                'the number of intermediate coupling coefficients')
        elif len(self.energy_levels
                 ) > self.intermediate_coupling_coefficients.shape[0]:
            raise ValueError(
                'Error: The number of Energy levels does not match the number'
                ' of intermediate coupling coefficients'
            )
        elif len(self.energy_levels) > len(self.slj):
            raise ValueError(
                'Error: The number of Energy levels does not match the number'
                ' of SLJ rows'
            )
        return True

    @lru_cache
    def species_data(self) -> Dict[str, Union[List[str], List[float]]]:
        """
        Loads the species data from the json file. Since many dopants are
        likely to be created and the data used many times, the data
        is cached using lru_cache.

        Returns:
            Dict[str, Union[List[str], List[float]]]: The species data in the
                form of a dictionary with the keys:
                ['EnergyLevelLabels', 'EnergyLevels', 'SLJ', 'absFWHM',
                'lineStrengths', 'TransitionLabels', 'JO_params',
                'eigenvectorSL', 'intermediateCouplingCoeffs']
        """
        if self.symbol in self.SURFACE_DOPANT_SYMBOLS_TO_NAMES:
            symbol = self.SURFACE_DOPANT_SYMBOLS_TO_NAMES[self.symbol]
        else:
            symbol = self.symbol
        # Load Data from json file
        with open(os.path.join(SPECIES_DATA_PATH, f'{symbol}.json'), 'r') as f:
            species_data = json.load(f)

        return species_data

    @property
    @lru_cache
    def energy_levels(self) -> List[EnergyLevel]:
        """
        Gets a list of all the energy levels for this dopant.

        Returns:
            List[EnergyLevel]: The EnergyLevel objects.
        """
        return [
            EnergyLevel(self.symbol, i, j)
            for i, j in zip(self.species_data()['EnergyLevelLabels'],
                            self.species_data()['EnergyLevels'])
        ]

    @property
    @lru_cache
    def absFWHM(self) -> List[float]:
        return self.species_data()['absFWHM']

    @property
    @lru_cache
    def slj(self) -> np.array:
        return np.array(self.species_data()['SLJ'])

    @property
    @lru_cache
    def judd_ofelt_parameters(self) -> List[float]:
        if 'JO_params' not in self.species_data():
            return []
        return self.species_data()['JO_params']

    @property
    @lru_cache
    def intermediate_coupling_coefficients(self) -> np.array:
        if 'intermediateCouplingCoeffs' not in self.species_data():
            return []
        return np.array(self.species_data()['intermediateCouplingCoeffs'])

    @property
    @lru_cache
    def eigenvector_sl(self) -> np.array:
        if 'eigenvectorSL' not in self.species_data():
            return []
        return np.array(self.species_data()['eigenvectorSL'])

    @property
    @lru_cache
    def transitions(self) -> List[Transition]:
        """
        Get the list of all transitions between all energy levels.

        Returns:
            List[Transition]: List of Transitions
        """
        energy_level_map = dict([(_el.label, _el)
                                 for _el in self.energy_levels])
        energy_level_label_map = dict([
            (_el.label, i) for i, _el in enumerate(self.energy_levels)
        ])

        transitions = [[0 for _ in self.energy_levels]
                       for _ in self.energy_levels]
        for i in range(len(self.species_data()['TransitionLabels'])):
            transition = self.species_data()['TransitionLabels'][i]
            line_strength = self.species_data()['lineStrengths'][i]

            initial_energy_level, final_energy_level = transition.split("->")
            try:
                initial_i = energy_level_label_map[initial_energy_level]
                final_i = energy_level_label_map[final_energy_level]

                transitions[initial_i][final_i] = Transition(
                    energy_level_map[initial_energy_level],
                    energy_level_map[final_energy_level], line_strength)
            except Exception:
                # These transitions are not used
                continue
        return transitions

    @property
    def volume_concentration(
            self,
            volume_per_dopant_site: Optional[float] = 7.23946667e-2) -> float:
        """
        Gets the volue a single dopant occupies

        Args:
            volume_per_dopant_site (float, None): The volume per dopant site.
                No reason to change this unless you know what you're doing
                Defaults to 7.23946667e-2.

        Returns:
            float: The volume occupied by one dopant
        """
        return self.molar_concentration / volume_per_dopant_site

    def get_line_strength_matrix(self) -> np.array:
        """
        Get the line strength matrix for transitions between every energy level

        Returns:
            np.array: The line strength matrix
        """
        line_strengths = []
        for row in self.transitions[:self.n_levels]:
            for transition in row[:self.n_levels]:
                if isinstance(transition, Transition):
                    line_strengths.append(transition.line_strength)
                else:
                    line_strengths.append(0)
        return np.reshape(line_strengths, (self.n_levels, self.n_levels))
