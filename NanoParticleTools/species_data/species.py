from monty.json import MSONable
from pymatgen.core.periodic_table import Species
energies = {'Dy': {
    'energy_levels': [175, 3626, 5952, 7806, 7853, 9166, 9223, 10273, 11070, 12471, 13267, 13814, 21228, 22222, 23563,
                      25109, 25794, 25856, 25890, 26334, 27624, 27543],
    'energy_level_labels': ["6H15/2", "6H13/2", "6H11/2", "6F11/2", "6H9/2", "6F9/2", "6H7/2", "6H5/2", "6F7/2",
                            "6F5/2", "6F3/2", "6F1/2", "4F9/2", "4H15/2", "4G11/2", "4M21/2", "4F7/2", "4K17/2",
                            "4I13/2", "6P5/2", "4M19/2", "6P3/2"]}}


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
    def __init__(self, initial_level: Energy_Level, final_level: Energy_Level):
        self.initial_level = initial_level
        self.final_level = final_level

    def __repr__(self):
        return f'{self.initial_level.label}->{self.final_level.label}'

    def __str__(self):
        return f'{self.initial_level.label}->{self.final_level.label}'


class ExcitedSpecies(Species):
    def __init__(self, symbol: str):
        super().__init__(symbol)

        self.energy_levels = [Energy_Level(symbol, i, j) for i, j in
                              zip(energies[symbol]['energy_level_labels'], energies[symbol]['energy_levels'])]
        self.all_transitons = [Transition(i, j) for j in self.energy_levels for i in self.energy_levels if i != j]