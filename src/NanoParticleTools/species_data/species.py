from monty.json import MSONable
import json
from pathlib import Path
import os
from typing import Optional, List
import numpy as np
from functools import lru_cache


SPECIES_DATA_PATH = os.path.join(str(Path(__file__).absolute().parent), 'data')

class EnergyLevel(MSONable):
    def __init__(self, element, label, energy):
        self.element = element
        self.label = label
        self.energy = energy

    def __str__(self):
        return f'{self.element} - {self.label} - {self.energy}'


class Transition(MSONable):
    def __init__(self,
                 initial_level: EnergyLevel,
                 final_level: EnergyLevel,
                 line_strength:float):
        self.initial_level = initial_level
        self.final_level = final_level
        self.line_strength = line_strength

    def __str__(self):
        return f'{self.initial_level.label}->{self.final_level.label}'

class Dopant(MSONable):
    def __init__(self,
                 symbol: str,
                 molar_concentration: float,
                 n_levels: Optional[int]=None):
        """

        :param symbol:
        :param concentration:
        :param n_levels:
        """
        if symbol == 'Surface':
            symbol = 'Na'
        self.symbol = symbol
        self.molar_concentration = molar_concentration
        # super().__init__(symbol, None)


        # If more levels are specified than possible, use all existing energy levels
        if n_levels is None:
            self.n_levels = len(self.energy_levels)
        else:
            self.n_levels = min(n_levels, len(self.energy_levels))

    def check_intrinsic_data(self):
        """
        Checks whether the dopant data is valid

        :return:
        """
        if self.eigenvector_sl.shape[0] != self.intermediate_coupling_coefficients.shape[1]:
            raise ValueError(
                'Error: The number of eigenvectors does not match the number of intermediate coupling coefficients')
        elif len(self.energy_levels) > self.intermediate_coupling_coefficients.shape[0]:
            raise ValueError(
                'Error: The number of Energy levels does not match the number of intermediate coupling coefficients')
        elif len(self.energy_levels) > len(self.slj):
            raise ValueError(
                'Error: The number of Energy levels does not match the number of SLJ rows')

    @lru_cache
    def species_data(self):
        if self.symbol == 'Na':
            symbol = 'Surface'
        else:
            symbol = self.symbol
        # Load Data from json file
        with open(os.path.join(SPECIES_DATA_PATH, f'{symbol}.json'), 'r') as f:
            species_data = json.load(f)

        return species_data

    @property
    @lru_cache
    def energy_levels(self):
        return [EnergyLevel(self.symbol, i, j) for i, j in
                                   zip(self.species_data()['EnergyLevelLabels'], self.species_data()['EnergyLevels'])]

    @property
    @lru_cache
    def absFWHM(self):
        return self.species_data()['absFWHM']


    @property
    @lru_cache
    def slj(self):
        return np.array(self.species_data()['SLJ'])

    @property
    @lru_cache
    def judd_ofelt_parameters(self):
        return self.species_data()['JO_params']

    @property
    @lru_cache
    def intermediate_coupling_coefficients(self):\
        return np.array(self.species_data()['intermediateCouplingCoeffs'])

    @property
    @lru_cache
    def eigenvector_sl(self):
        return np.array(self.species_data()['eigenvectorSL'])

    @property
    @lru_cache
    def transitions(self):
        energy_level_map = dict([(_el.label, _el) for _el in self.energy_levels])
        energy_level_label_map = dict([(_el.label, i) for i, _el in enumerate(self.energy_levels)])

        transitions = [[0 for _ in self.energy_levels] for _ in self.energy_levels]
        for i in range(len(self.species_data()['TransitionLabels'])):
            transition = self.species_data()['TransitionLabels'][i]
            line_strength = self.species_data()['lineStrengths'][i]

            initial_energy_level, final_energy_level = transition.split("->")
            try:
                initial_i = energy_level_label_map[initial_energy_level]
                final_i = energy_level_label_map[final_energy_level]

                transitions[initial_i][final_i] = Transition(energy_level_map[initial_energy_level],
                                         energy_level_map[final_energy_level],
                                         line_strength)
            except:
                # These transitions are not used
                continue
        return transitions

    @property
    def volume_concentration(self, volume_per_dopant_site:Optional[float] = 7.23946667e-2) -> float:
        return self.molar_concentration/volume_per_dopant_site

    # def set_initial_populations(self, populations:Optional[List[float]] = None):
    #     if populations is None:
    #         populations = [0 for i in range(self.n_levels)]
    #         populations[0] = self.volume_concentration
    #
    #     self.initial_populations = populations

    def get_line_strength_matrix(self):
        line_strengths = []
        for row in self.transitions[:self.n_levels]:
            for transition in row[:self.n_levels]:
                if isinstance(transition, Transition):
                    line_strengths.append(transition.line_strength)
                else:
                    line_strengths.append(0)
        return np.reshape(line_strengths, (self.n_levels, self.n_levels))