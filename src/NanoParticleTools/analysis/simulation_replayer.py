from typing import Optional, Tuple, Dict, List

import numpy as np
from collections import Counter
from functools import lru_cache
import json
from monty.json import MontyDecoder
import sqlite3
from NanoParticleTools.inputs.util import get_formula_by_constraint
from pymatgen.core import Composition


class SimulationReplayer():

    def __init__(self, trajectories_db_file, npmc_input_file):
        self.trajectories_db_file = trajectories_db_file
        self.npmc_input_file = npmc_input_file

    @property
    def initial_states(self):
        return self.npmc_input.initial_states

    @property
    def sites(self):
        if self.npmc_input.nanoparticle.has_structure is False:
            self.npmc_input.nanoparticle.generate()
        return self.npmc_input.nanoparticle.dopant_sites

    @property
    @lru_cache(maxsize=1)
    def npmc_input(self):
        with open(self.npmc_input_file, 'rb') as f:
            npmc_input = json.load(f, cls=MontyDecoder)
        return npmc_input

    def get_state_map(self):
        "Maps element specific states to the overall state"
        state_map_species_id = {}
        state_map_species_name = {}
        combined_id = 0

        for i, dopant in enumerate(self.npmc_input.spectral_kinetics.dopants):
            if i not in state_map_species_id:
                state_map_species_id[i] = {}
                state_map_species_name[dopant.symbol] = {}
            for level in range(dopant.n_levels):
                state_map_species_id[i][level] = combined_id
                state_map_species_name[dopant.symbol][level] = combined_id
                combined_id += 1
        return state_map_species_id, state_map_species_name

    @lru_cache(maxsize=1)
    def run(self, step_size=1e-5, normalize=True):
        # seeds = []  # List[int]
        simulation_time = {}  # Dict[int]
        event_statistics = {}  # Dict[Dict]

        # Keep track of the population
        x = {}  # Dict[List[float]]
        states = {}  #
        site_evolution = {}  # Dict[List[List[int]]]
        population_evolution = {}  # Dict[List[List[int]]]

        state_map_species_id, state_map_species_name = self.get_state_map()

        with sqlite3.connect(self.trajectories_db_file) as con:
            cur = con.cursor()

            sql_cmd = "SELECT * FROM trajectories;"
            for row in cur.execute(sql_cmd):
                seed = row[0]
                # step = row[1]
                time = row[2]
                # donor_i = row[3]
                # acceptor_i = row[4]
                interaction_id = row[5]

                # Update simulation time
                simulation_time[seed] = time

                # Add this event to the statistics
                try:
                    event_statistics[seed][interaction_id] += 1
                except KeyError:
                    try:
                        event_statistics[seed][interaction_id] = 1
                    except KeyError:
                        # Seed has not been seen before.
                        # Initialize all the necessary stuff
                        event_statistics[seed] = {interaction_id: 1}

                # Keep track of the populations
                try:
                    # Check if the state needs to be saved
                    while (simulation_time[seed] // step_size) >= len(x[seed]):

                        self.save_populations(states, seed,
                                              len(x[seed]) * step_size, x,
                                              population_evolution,
                                              site_evolution, step_size)

                    # Try to update the state of this seed.
                    # If it fails, setup a new state
                    self.update_state(states, row, state_map_species_id)
                except Exception:
                    states[seed] = np.zeros(
                        (len(self.initial_states),
                         self.npmc_input.spectral_kinetics.total_n_levels))
                    for i, (state, site) in enumerate(
                            zip(self.initial_states, self.sites)):
                        _state = state_map_species_name[str(
                            site.specie.symbol)][state]
                        states[seed][i, _state] = 1

                    self.save_populations(states, seed, 0, x,
                                          population_evolution, site_evolution,
                                          step_size)
                    while (simulation_time[seed] // step_size) >= len(x[seed]):
                        self.save_populations(states, seed,
                                              len(x[seed]) * step_size, x,
                                              population_evolution,
                                              site_evolution, step_size)
                    self.update_state(states, row, state_map_species_id)

            # Do one final save for each seed
            for seed in simulation_time.keys():
                self.save_populations(states, seed,
                                      len(x[seed]) * step_size, x,
                                      population_evolution, site_evolution,
                                      step_size)
        return (simulation_time, event_statistics, x, population_evolution,
                site_evolution)

    def update_state(self, states, row, state_map_species_id):
        seed = row[0]
        # step = row[1]
        # time = row[2]
        donor_i = row[3]
        acceptor_i = row[4]
        interaction_id = row[5]

        _interaction = self.npmc_input.interactions[interaction_id]
        # Update the states for the sites corresponding
        # to this interaction event
        # Apply the event to the donor site

        current_left_state = states[seed][donor_i][state_map_species_id[
            _interaction['species_id_1']][_interaction['left_state_1']]]
        current_right_state = states[seed][donor_i][state_map_species_id[
            _interaction['species_id_1']][_interaction['right_state_1']]]
        if current_left_state == 1 and current_right_state == 0:
            states[seed][donor_i][state_map_species_id[_interaction[
                'species_id_1']][_interaction['left_state_1']]] = 0
            states[seed][donor_i][state_map_species_id[_interaction[
                'species_id_1']][_interaction['right_state_1']]] = 1
        else:
            raise RuntimeError('Inconsistent simulation state encountered. '
                               'Please rerun the simulation. If this issue '
                               'persists, please raise a github issue')

        if _interaction['number_of_sites'] == 2:
            # If this is a two-site interaction, apply
            # the event to the acceptor ion

            current_left_state = states[seed][acceptor_i][state_map_species_id[
                _interaction['species_id_2']][_interaction['left_state_2']]]
            current_right_state = states[seed][acceptor_i][state_map_species_id[
                _interaction['species_id_2']][_interaction['right_state_2']]]
            
            if current_left_state == 1 and current_right_state == 0:
                states[seed][acceptor_i][state_map_species_id[_interaction[
                    'species_id_2']][_interaction['left_state_2']]] = 0
                states[seed][acceptor_i][state_map_species_id[_interaction[
                    'species_id_2']][_interaction['right_state_2']]] = 1
            else:
                print('current left state: ', current_left_state)
                print('current right state', current_right_state)
                print(states)
                raise RuntimeError('Inconsistent simulation state encountered.'
                                'Please rerun the simulation. If this issue'
                                'persists, please raise a github issue')

    def save_populations(self, states, seed, time, x, population_evolution,
                         site_evolution, step_size):
        try:
            x[seed].append(time)
            population_evolution[seed] = np.vstack(
                [population_evolution[seed],
                 np.sum(states[seed], axis=0)])
            site_evolution[seed] = np.vstack(
                [site_evolution[seed],
                 np.where(states[seed] > 0)[1]])
        except Exception:
            x[seed] = [0]
            population_evolution[seed] = np.sum(states[seed], axis=0)
            site_evolution[seed] = np.where(states[seed] > 0)[1]

    def calculate_dndt(self,
                       data: Optional[Tuple[Dict, Dict, Dict, Dict]] = None):
        if data is None:
            data = self.run()

        simulation_time, event_statistics, _, _, _ = data

        species_counter = Counter(
            [site['species_id'] for site in self.npmc_input.sites.values()])

        dndt = {}
        dndt_keys = None
        for seed, statistics in event_statistics.items():
            _dndt = []
            for interaction_id, occurences in statistics.items():
                if dndt_keys is None:
                    dndt_keys = list(
                        self.npmc_input.interactions[interaction_id].keys())
                    dndt_keys.extend([
                        'dNdT', 'dNdT per atom', 'occurences',
                        'occurences per atom'
                    ])
                interaction = self.npmc_input.interactions[interaction_id]
                _d = list(interaction.values())
                _d.append(occurences / simulation_time[seed])
                _d.append(occurences / simulation_time[seed] /
                          species_counter[interaction['species_id_1']])
                _d.append(occurences)
                # TODO: This is actually incorrect, but to keep things
                # consistent, keep it incorrect for now
                _d.append(occurences / simulation_time[seed] /
                          species_counter[interaction['species_id_1']])
                _dndt.append(_d)
            dndt[seed] = _dndt
        return dndt_keys, dndt

    def generate_docs(self,
                      data: Optional[Tuple[Dict, Dict, Dict, Dict]] = None):
        if data is None:
            data = self.run()

        (simulation_time, event_statistics, x, population_evolution,
         site_evolution) = data
        species_counter = Counter(
            [site['species_id'] for site in self.npmc_input.sites.values()])
        normalization_factors = np.hstack(
            [[species_counter[i]] * dopant.n_levels for i, dopant in enumerate(
                self.npmc_input.spectral_kinetics.dopants)])
        overall_population_evolution = {}
        for seed in population_evolution.keys():
            overall_population_evolution[seed] = np.divide(
                population_evolution[seed], normalization_factors)

        dndt_keys, dndt = self.calculate_dndt()
        # Change rate to rate_coefficient for clarity
        dndt_keys[dndt_keys.index('rate')] = 'rate_coefficient'

        population_by_constraint = self._population_evolution_by_constraint(
            site_evolution)

        results = []
        for seed in simulation_time.keys():
            dopant_amount = {}
            for dopant in self.npmc_input.nanoparticle.dopant_sites:
                try:
                    dopant_amount[str(dopant.specie)] += 1
                except Exception:
                    dopant_amount[str(dopant.specie)] = 1
            c = Composition(dopant_amount)
            nanostructure = '-'.join([
                "core" if i == 0 else "shell"
                for i, _ in enumerate(self.npmc_input.nanoparticle.constraints)
            ])
            nanostructure_size = '-'.join([
                f"{int(max(c.bounding_box()))}A_core"
                if i == 0 else f"{int(max(c.bounding_box()))}A_shell"
                for i, c in enumerate(self.npmc_input.nanoparticle.constraints)
            ])
            formula_by_constraint = get_formula_by_constraint(
                self.npmc_input.nanoparticle)
            _d = {
                'simulation_seed':
                seed,
                'dopant_seed':
                self.npmc_input.nanoparticle.seed,
                'simulation_length':
                int(
                    np.sum([
                        event_statistics[seed][key]
                        for key in event_statistics[seed]
                    ])),
                'simulation_time':
                simulation_time[seed],
                'n_constraints':
                len(self.npmc_input.nanoparticle.constraints),
                'n_dopant_sites':
                len(self.npmc_input.nanoparticle.dopant_sites),
                'n_dopants':
                len(self.npmc_input.spectral_kinetics.dopants),
                'formula':
                c.formula.replace(" ", ""),
                'nanostructure':
                nanostructure,
                'nanostructure_size':
                nanostructure_size,
                'total_n_levels':
                self.npmc_input.spectral_kinetics.total_n_levels,
                'formula_by_constraint':
                formula_by_constraint,
                'dopants': [
                    str(dopant.symbol)
                    for dopant in self.npmc_input.spectral_kinetics.dopants
                ],
                'dopant_concentration':
                self.npmc_input.nanoparticle._dopant_concentration,
                'overall_dopant_concentration':
                self.npmc_input.nanoparticle.dopant_concentrations(),
                'excitation_power':
                self.npmc_input.spectral_kinetics.excitation_power,
                'excitation_wavelength':
                self.npmc_input.spectral_kinetics.excitation_wavelength,
                'dopant_composition':
                dopant_amount,
            }

            # Add the input parameters to the trajectory document
            _input_d = {
                'constraints':
                self.npmc_input.nanoparticle.constraints,
                'dopant_specifications':
                self.npmc_input.nanoparticle.dopant_specification,
                'n_levels': [
                    dopant.n_levels
                    for dopant in self.npmc_input.spectral_kinetics.dopants
                ]
            }
            _d['input'] = _input_d

            # Add output to the trajectory document
            _output_d = {}
            _output_d['summary_keys'] = dndt_keys
            _output_d['summary'] = dndt[seed]

            _output_d['x_populations'] = x[seed]
            _output_d['y_overall_populations'] = overall_population_evolution[
                seed]
            _output_d['y_constraint_populations'] = population_by_constraint[
                seed]
            _output_d['final_states'] = site_evolution[seed][-1, :]

            _d['output'] = _output_d

            # Use the "trajectory_doc" key to ensure that each
            # gets saved as a separate trajectory
            results.append({'trajectory_doc': _d})
        return results

    def _population_evolution_by_constraint(self,
                                            site_evolution_dict: Optional[Dict[
                                                int, List]] = None):
        if site_evolution_dict is None:
            _, _, _, _, site_evolution_dict = self.run()

        population_by_constraint = {}
        sites = np.array([
            site.coords for site in self.npmc_input.nanoparticle.dopant_sites
        ])
        species_names = np.array([
            str(site.specie.symbol)
            for site in self.npmc_input.nanoparticle.dopant_sites
        ])

        for seed, site_evolution in site_evolution_dict.items():

            # Get the indices for dopants within each constraint
            site_indices_by_constraint = []
            for constraint in self.npmc_input.nanoparticle.constraints:

                indices_inside_constraint = set(
                    np.where(constraint.sites_in_bounds(sites))[0])

                for level in site_indices_by_constraint:
                    indices_inside_constraint = (indices_inside_constraint -
                                                 set(level))
                site_indices_by_constraint.append(
                    sorted(list(indices_inside_constraint)))

            _population_by_constraint = []
            for n, _ in enumerate(self.npmc_input.nanoparticle.constraints):
                site_indices = site_indices_by_constraint[n]
                layer_site_evolution = site_evolution[:, site_indices]
                species_counter = Counter(species_names[site_indices])

                # Set the shape of the array
                n_time_intervals = layer_site_evolution.shape[0]
                n_levels = self.npmc_input.spectral_kinetics.total_n_levels
                layer_population = np.zeros((n_time_intervals, n_levels))

                # Iterate through the time intervals and bin the energy levels
                for i, _l in enumerate(layer_site_evolution):
                    c = Counter(_l)
                    for key in c:
                        layer_population[i, int(key)] += c[key]

                # Normalize values, so that the max population is 1
                normalization_factors = np.hstack(
                    [[species_counter[dopant.symbol]] * dopant.n_levels
                     for dopant in self.npmc_input.spectral_kinetics.dopants])
                # Avoid divide by zero by making normalization factor = 1
                # for elements not in the shell
                normalization_factors[normalization_factors == 0] = 1
                layer_population = np.divide(layer_population,
                                             normalization_factors)

                # Add to the list of populations separated by layer
                _population_by_constraint.append(layer_population)
            population_by_constraint[seed] = _population_by_constraint
        return population_by_constraint
