from monty.json import MSONable
from pymatgen.core.periodic_table import Species
import json
from pathlib import Path
import os
from typing import Optional, List
import numpy as np

energies = {'Dy': {
    'energy_levels': [175, 3626, 5952, 7806, 7853, 9166, 9223, 10273, 11070, 12471, 13267, 13814, 21228, 22222, 23563,
                      25109, 25794, 25856, 25890, 26334, 27624, 27543],
    'energy_level_labels': ["6H15/2", "6H13/2", "6H11/2", "6F11/2", "6H9/2", "6F9/2", "6H7/2", "6H5/2", "6F7/2",
                            "6F5/2", "6F3/2", "6F1/2", "4F9/2", "4H15/2", "4G11/2", "4M21/2", "4F7/2", "4K17/2",
                            "4I13/2", "6P5/2", "4M19/2", "6P3/2"]}}

SPECIES_DATA_PATH = os.path.join(str(Path(__file__).absolute().parent), 'data')

class Energy_Level(MSONable):
    def __init__(self, element, label, energy):
        self.element = element
        self.label = label
        self.energy = energy

    def __repr__(self):
        return f'{self.element} - {self.label} - {self.energy}'

    def __str__(self):
        return f'{self.element} - {self.label} - {self.energy}'


class Transition(MSONable):
    def __init__(self, initial_level: Energy_Level, final_level: Energy_Level, line_strength:float):
        self.initial_level = initial_level
        self.final_level = final_level
        self.line_strength = line_strength

    def __repr__(self):
        return f'{self.initial_level.label}->{self.final_level.label}'

    def __str__(self):
        return f'{self.initial_level.label}->{self.final_level.label}'


class ExcitedStateSpecies(Species):
    def __init__(self, symbol: str):
        super().__init__(symbol, None)

        # Load Data from json file
        with open(os.path.join(SPECIES_DATA_PATH, f'{symbol}.json'), 'r') as f:
            species_data = json.load(f)

        # self.energy_levels = species_data['energy_levels']
        # self.energy_level_labels = species_data['energy_level_labels']
        # self.transitions = species_data['transitions']
        # self.line_strengths = species_data['line_strengths']


        self.energy_levels = [Energy_Level(symbol, i, j) for i, j in
                              zip(species_data['energy_level_labels'], species_data['energy_levels'])]


        energy_level_map = dict([(_el.label, _el) for _el in self.energy_levels])
        energy_level_label_map = dict([(_el.label, i) for i, _el in enumerate(self.energy_levels)])

        transitions = [[0 for _ in self.energy_levels] for _ in self.energy_levels]
        for i in range(len(species_data['transitions'])):
            transition = species_data['transitions'][i]
            line_strength = species_data['line_strengths'][i]

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

        self.transitons = transitions



class Dopant(ExcitedStateSpecies):
    def __init__(self, symbol, concentration, n_levels):
        self.symbol = symbol

        super().__init__(symbol)

        self.molar_concentration = concentration

        # If more levels are specified than possible, use all existing energy levels
        self.n_levels = min(n_levels, len(self.energy_levels))

        self.MPR_rates = None

    @property
    def volume_concentration(self):
        from NanoParticleTools.inputs.spectral_kinetics import VOLUME_PER_DOPANT_SITE
        return self.molar_concentration/VOLUME_PER_DOPANT_SITE

    def set_initial_populations(self, populations:Optional[List[float]] = None):
        if populations is None:
            populations = [0 for i in range(self.n_levels)]
            populations[0] = self.volume_concentration
        self.initial_populations = populations

    def calculate_MPR_rates(self, w_0phonon, alpha, stokes_shift, phonon_energy):
        """
        Calculates MultiPhonon Relaxation Rate for a given set of energy levels using Miyakawa-Dexter MPR theory

        :param w_0phonon: zero gap rate (s^-1)
        :param alpha: pre-exponential constant in Miyakawa-Dexter MPR theory.  Changes with matrix (cm)
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
