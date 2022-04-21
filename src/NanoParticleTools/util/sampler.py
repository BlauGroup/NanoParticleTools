from typing import Optional
from itertools import combinations, product
import numpy as np
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint


class NanoParticleSampler():
    def __init__(self,
                 seed,
                 min_core_radius: Optional[float] = 4,
                 max_core_radius: Optional[float] = 4,
                 min_shell_thickness: Optional[float] = 1,
                 max_shell_thickness: Optional[float] = 2.5,
                 min_concentration: Optional[float] = 0,
                 max_concentration: Optional[float] = 0.4,
                 concentration_constraint: Optional[float] = 0.5):

        # Range of variables to be sampled
        self.min_core_radius = min_core_radius
        self.max_core_radius = max_core_radius
        self.min_shell_thickness = min_shell_thickness
        self.max_shell_thickness = max_shell_thickness
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

        self.concentration_constraint = concentration_constraint
        self.seed = seed

        self._rng = None

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(seed=self.seed)
        return self._rng

    def random_nanoparticle_core_size(self):
        return self.rng.uniform(self.min_core_radius, self.max_core_radius)

    def random_nanoparticle_layer_thickness(self):
        return self.rng.uniform(self.min_shell_thickness, self.max_shell_thickness)

    def random_doping_concentration(self):
        return self.rng.uniform(self.min_concentration, self.max_concentration)

    def generate_samples(self, n, excitation_wavelengths, excitation_powers, dopants):
        configurations = []
        for i in range(n):
            combination = self.one_random_configuration_template(excitation_wavelengths, excitation_powers, dopants)
            configurations.extend(self.get_configurations(combination))

        return configurations

    def one_random_configuration_template(self, excitation_wavelengths, excitation_powers, dopants):

        # Pick a excitation wavelength and power
        excitation_wavelength = self.rng.choice(excitation_wavelengths)
        excitation_power = self.rng.choice(excitation_powers)

        # Determine number of shells
        n_shells = self.rng.choice(list(range(0, 3 + 1)), p=np.divide([1, 8, 64, 512], np.sum([1, 8, 64, 512])))

        n_dopant_weights = [math.comb(len(dopants), i) for i in range(len(dopants) + 1)]
        n_dopant_weights = np.divide(n_dopant_weights, sum(n_dopant_weights))
        nanoparticle_config = []
        for _ in range(0, n_shells + 1):
            n_dopants_in_layer = self.rng.choice(range(len(dopants) + 1), p=n_dopant_weights)
            _dopants = list(self.rng.choice(dopants, n_dopants_in_layer, replace=False))

            nanoparticle_config.append(_dopants)
        return (excitation_wavelength, excitation_power, nanoparticle_config)

    def get_configurations(self, combination, n_configs=1):
        configurations = []
        while len(configurations) < n_configs:
            constraints, dopant_specifications = self.generate_random_configuration(combination)

            # Check if this configuration is valid (concentration does not exceed the threshold in each layer)
            valid = True
            for i in range(len(constraints)):
                layer_concentrations = [spec[1] for spec in dopant_specifications if spec[0] == i]
                if sum(layer_concentrations) > self.concentration_constraint:
                    valid = False
            if valid:
                # 0th and 1st index of the combination are included (To retain info on excitation parameters)
                configurations.append((combination[0], combination[1], constraints, dopant_specifications))
        return configurations

    def generate_random_configuration(self, combination):
        n_layers = len(combination[2])

        constraints = []
        dopant_specifications = []
        cumulative_radius = 0
        for n in range(n_layers):
            if n == 0:
                # First layer is always a core
                constraint_radius = self.random_nanoparticle_core_size()
            else:
                # All layers n > 0 are shells
                constraint_radius = self.random_nanoparticle_layer_thickness()

            # Add the generated radius to the cumulative total
            cumulative_radius += constraint_radius

            # Create the constraint pertaining to the layer
            constraints.append(SphericalConstraint(cumulative_radius))

            # Generate random concentrations for each dopant in this layer
            for el in combination[2][n]:
                # get dopant concentration
                _concentration = self.random_doping_concentration()
                dopant_specifications.append((n, _concentration, el, 'Y'))
        return constraints, dopant_specifications


