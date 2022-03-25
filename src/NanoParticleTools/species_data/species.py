from monty.json import MSONable
from pymatgen.core.periodic_table import Species
import json
from pathlib import Path
import os
from typing import Optional, List
import numpy as np
from functools import lru_cache

energies = {'Dy': {
    'energy_levels': [175, 3626, 5952, 7806, 7853, 9166, 9223, 10273, 11070, 12471, 13267, 13814, 21228, 22222, 23563,
                      25109, 25794, 25856, 25890, 26334, 27624, 27543],
    'energy_level_labels': ["6H15/2", "6H13/2", "6H11/2", "6F11/2", "6H9/2", "6F9/2", "6H7/2", "6H5/2", "6F7/2",
                            "6F5/2", "6F3/2", "6F1/2", "4F9/2", "4H15/2", "4G11/2", "4M21/2", "4F7/2", "4K17/2",
                            "4I13/2", "6P5/2", "4M19/2", "6P3/2"]}}

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

class Dopant(Species):
    def __init__(self,
                 symbol: str,
                 concentration: float,
                 n_levels: Optional[int]=None):
        """

        :param symbol:
        :param concentration:
        :param n_levels:
        """
        self.symbol = symbol

        super().__init__(symbol, None)

        self.molar_concentration = concentration

        self.MPR_rates = None

        # Initialize intrinsic values to None, only initialized when needed. Allows the dopant to be used as a species
        self._energy_levels = None
        self._absFWHM = None
        self._slj = None
        self._judd_ofelt_parameters = None
        self._intermediate_coupling_coefficients = None
        self._eigenvector_sl = None
        self._transitions = None

        self.initial_populations = None

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

    @lru_cache(maxsize=None)
    def species_data(self):
        # Load Data from json file
        with open(os.path.join(SPECIES_DATA_PATH, f'{self.symbol}.json'), 'r') as f:
            species_data = json.load(f)

        return species_data

    @property
    def energy_levels(self):
        if self._energy_levels is None:
            self._energy_levels = [EnergyLevel(self.symbol, i, j) for i, j in
                                   zip(self.species_data()['EnergyLevelLabels'], self.species_data()['EnergyLevels'])]
        return self._energy_levels

    @property
    def absFWHM(self):
        if self._absFWHM is None:
            self._absFWHM = self.species_data()['absFWHM']
        return self._absFWHM

    @property
    def slj(self):
        if self._slj is None:
            self._slj = np.array(self.species_data()['SLJ'])
        return self._slj

    @property
    def judd_ofelt_parameters(self):
        if self._judd_ofelt_parameters is None:
            self._judd_ofelt_parameters = self.species_data()['JO_params']
        return self._judd_ofelt_parameters

    @property
    def intermediate_coupling_coefficients(self):
        if self._intermediate_coupling_coefficients is None:
            self._intermediate_coupling_coefficients = np.array(self.species_data()['intermediateCouplingCoeffs'])
        return self._intermediate_coupling_coefficients

    @property
    def eigenvector_sl(self):
        if self._eigenvector_sl is None:
            self._eigenvector_sl = np.array(self.species_data()['eigenvectorSL'])
        return self._eigenvector_sl

    @property
    def transitions(self):
        if self._transitions is None:
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
            self._transitions = transitions
        return self._transitions

    @property
    def volume_concentration(self, volume_per_dopant_site:Optional[float] = 7.23946667e-2) -> float:
        return self.molar_concentration/volume_per_dopant_site

    def set_initial_populations(self, populations:Optional[List[float]] = None):
        if populations is None:
            populations = [0 for i in range(self.n_levels)]
            populations[0] = self.volume_concentration

        self.initial_populations = populations

    def calculate_MPR_rates(self,
                            w_0phonon: float,
                            alpha: float,
                            stokes_shift: float,
                            phonon_energy: float):
        """
        Calculates MultiPhonon Relaxation Rate for a given set of energy levels using Miyakawa-Dexter MPR theory

        :param w_0phonon: zero gap rate (s^-1)
        :param alpha: pre-exponential constant in Miyakawa-Dexter MPR theory.  Changes with matrix (cm)
        :param stokes_shift:
        :param phonon_energy:
        :return:
        """

        # multiphonon relaxation rate from level i to level i-1
        mpr_rates = [0] # level 0 cannot relax, thus it's rate is 0
        for i in range(1, self.n_levels):
            energy_gap = max(abs(self.energy_levels[i].energy - self.energy_levels[i-1].energy) - stokes_shift, 0) - 2 * phonon_energy

            rate = w_0phonon * np.exp(-alpha * energy_gap)
            mpr_rates.append(rate)

        self.mpr_rates = mpr_rates

        # multi phonon absorption rate from level i to level i+1
        mpa_rates = []
        for i in range(1, self.n_levels):
            energy_gap = self.energy_levels[i].energy - self.energy_levels[i-1].energy
            if energy_gap < 3*phonon_energy:
                rate = mpr_rates[i] * np.exp(-alpha * energy_gap)
                mpa_rates.append(rate)
            else:
                mpa_rates.append(0)
        mpa_rates.append(0) # Highest energy level cannot be further excited, therefore set its rate to 0
        self.mpa_rates = mpa_rates

    def get_line_strength_matrix(self):
        line_strengths = []
        for row in self.transitions[:self.n_levels]:
            for transition in row[:self.n_levels]:
                if isinstance(transition, Transition):
                    line_strengths.append(transition.line_strength)
                else:
                    line_strengths.append(0)
        return np.reshape(line_strengths, (self.n_levels, self.n_levels))