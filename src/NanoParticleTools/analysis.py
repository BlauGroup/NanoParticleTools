from typing import Optional

import numpy as np
from collections import Counter
from functools import lru_cache
import os
import json
from monty.json import MontyDecoder
import sqlite3
from NanoParticleTools.inputs.util import specie_energy_level_to_combined_energy_level


class SimulationReplayer():
    def __init__(self,
                 trajectories,
                 npmc_input,
                 seeds):
        self.trajectories = trajectories
        self.npmc_input = npmc_input
        self.seeds = seeds

    @classmethod
    def from_run_directory(cls, run_dir):
        """
        Convenience constructor to obtain
        :param run_dir:
        :return:
        """
        with open(os.path.join(run_dir, 'npmc_input.json'), 'rb') as f:
            npmc_input = json.load(f, cls=MontyDecoder)

        db_file = os.path.join(run_dir, 'initial_state.sqlite')
        with sqlite3.connect(db_file) as con:
            cur = con.cursor()
            sql_cmd = 'select seed from trajectories where step=0'
            seeds = set([row[0] for row in cur.execute(sql_cmd)])

        # Don't load trajectories at the sametime, else a OOM event may occur
        # npmc_input.load_trajectories(os.path.join(run_dir, 'initial_state.sqlite'))

        trajectories = {}
        for seed in seeds:
            _traj = Trajectory(seed, npmc_input, db_file)
            trajectories[seed] = _traj
        return cls(trajectories, npmc_input, seeds)

    def get_single_trajectory_summary(self, seed=None):
        try:
            trajectory = self.trajectories[seed]
        except:
            seed = list(self.trajectories.keys())[0]
            trajectory = self.trajectories[seed]

        return trajectory.get_summary()

    def get_summaries(self):
        dndts = {}
        for seed in self.seeds:
            _d = {}
            _d['summary'] = self.get_single_trajectory_summary(seed)
            _d['simulation_time'] = self.trajectories[seed].simulation_time
            _d['x_populations'], _d['y_populations'] = self.trajectories[seed].get_population_evolution()
            # _d['x_states'], _d['y_states'] = self.trajectories[seed].get_state_evolution_by_site()
            dndts[seed] = _d

        return dndts


class Trajectory():
    def __init__(self, seed, npmc_input, database_file):
        self.seed = seed
        self.npmc_input = npmc_input
        self.database_file = database_file
        self.species_counter = Counter([site['species_id'] for site in self.npmc_input.sites.values()])

    @property
    def initial_states(self):
        return self.npmc_input.initial_states

    @property
    @lru_cache(maxsize=1)
    def trajectory(self):
        # Here we load the trajectory only when it is needed.
        return self.npmc_input.load_trajectory(self.seed, self.database_file)

    @property
    def simulation_time(self):
        return self.trajectory[-1][-1]

    @property
    @lru_cache
    def event_statistics(self):
        return Counter([interaction_id for _, _, interaction_id, _ in self.trajectory])

    def get_summary(self):
        events_summary = []
        for key in sorted(self.event_statistics.keys(), key=lambda x: self.event_statistics[x], reverse=True):
            # Create a copy of the interaction dict
            _d = self.npmc_input.interactions[key].copy()

            # Rename the rate to rate_coefficient
            _d['rate_coefficient'] = _d['rate']
            del _d['rate']

            # Add the dNdT
            _d['dNdT'] = self.event_statistics[key] / self.simulation_time
            _d['dNdT per atom'] = self.event_statistics[key] / self.simulation_time / self.species_counter[
                _d['species_id_1']]

            # Add the number of occurences
            _d['occurences'] = self.event_statistics[key]
            _d['occurences per atom'] = self.event_statistics[key] / self.simulation_time / self.species_counter[
                _d['species_id_1']]

            # Append the dictionary to the list of events
            events_summary.append(_d)

        return events_summary

    @lru_cache
    def bin_interactions_by_site(self):
        # Initialize dictionaries to keep track of the events.
        interactions_by_sites = dict([(key, []) for key in self.npmc_input.sites.keys()])

        # Iterate through all the events that occured
        for event in self.trajectory:
            # Unpack information from the interaction that occured
            site1, site2, interaction_id, time = event

            # Add the interaction to the specific site
            # Use the 'left' and 'right' metadata to keep track of whether the atom was a donor or acceptor
            # 'left' = donor; 'right' = 'acceptor'
            interactions_by_sites[site1].append((interaction_id, self.simulation_time, 'donor'))

            # If this is a 2 site interaction, add the interaction to the bookkeeping for the acceptor site
            if site2 != -1:
                interactions_by_sites[site2].append((interaction_id, self.simulation_time, 'acceptor'))

        return interactions_by_sites

    def get_state_map(self):
        "Maps element specific states to the overall state"
        state_map_species_id = {}
        state_map_species_name = {}
        combined_id = 0

        for i, dopant in enumerate(self.npmc_input.spectral_kinetics.dopants):
            if i not in state_map_species_id:
                state_map_species_id[i] = {}
                state_map_species_name[dopant.symbol] = {}
            for l in range(dopant.n_levels):
                state_map_species_id[i][l] = combined_id
                state_map_species_name[dopant.symbol][l] = combined_id
                combined_id += 1
        return state_map_species_id, state_map_species_name

    def get_state_evolution(self, step_size=1e-5):
        state_map_species_id, state_map_species_name = self.get_state_map()
        states = np.zeros((len(self.initial_states), self.npmc_input.spectral_kinetics.total_n_levels))
        for i, (state, site) in enumerate(zip(self.initial_states, self.npmc_input.nanoparticle.dopant_sites)):
            _state = state_map_species_name[str(site.specie)][state]
            states[i, _state] = 1

        x = np.arange(0, self.simulation_time + step_size, step_size)
        site_evolution = np.zeros((len(x), states.shape[0]))
        population_evolution = np.zeros((len(x), states.shape[1]))
        #         print(x.shape, site_evolution.shape, population_evolution.shape)
        current_time = 0
        event_i = 0
        for i, time_interval in enumerate(x):
            while current_time < time_interval and event_i < len(self.trajectory):
                # Propagate steps
                donor_i, acceptor_i, _interaction_id, _time = self.trajectory[event_i]
                _interaction = self.npmc_input.interactions[_interaction_id]

                # Update the states for the sites corresponding to this interaction event
                # Apply the event to the donor site
                states[donor_i][state_map_species_id[_interaction['species_id_1']][_interaction['left_state_1']]] = 0
                states[donor_i][state_map_species_id[_interaction['species_id_1']][_interaction['right_state_1']]] = 1
                if _interaction['number_of_sites'] == 2:
                    # If this is a two-site interaction, apply the event to the acceptor ion
                    states[acceptor_i][
                        state_map_species_id[_interaction['species_id_2']][_interaction['left_state_2']]] = 0
                    states[acceptor_i][
                        state_map_species_id[_interaction['species_id_2']][_interaction['right_state_2']]] = 1
                event_i += 1
                current_time = _time
            # Save states
            population_evolution[i, :] = np.sum(states, axis=0)
            site_evolution[i, :] = np.where(states > 0)[1]

        # Factors to normalize levels array by (# of atoms of each type)
        normalization_factors = np.hstack([[self.species_counter[i]] * dopant.n_levels for i, dopant in
                                           enumerate(self.npmc_input.spectral_kinetics.dopants)])

        overall_population_evolution = np.divide(population_evolution, normalization_factors)

        population_by_constraint = self._population_evolution_by_constraint(site_evolution)

        return x, overall_population_evolution, population_by_constraint, site_evolution

    def _population_evolution_by_constraint(self, site_evolution):
        sites = np.array([site.coords for site in self.npmc_input.nanoparticle.dopant_sites])
        species_names = np.array([str(site.specie) for site in self.npmc_input.nanoparticle.dopant_sites])

        # Get the indices for dopants within each constraint
        site_indices_by_constraint = []
        for constraint in self.npmc_input.nanoparticle.constraints:

            indices_inside_constraint = set(np.where(constraint.sites_in_bounds(sites))[0])

            for l in site_indices_by_constraint:
                indices_inside_constraint = indices_inside_constraint - set(l)
            site_indices_by_constraint.append(sorted(list(indices_inside_constraint)))

        population_by_constraint = []
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
            normalization_factors = np.hstack([[species_counter[dopant.symbol]] * dopant.n_levels for dopant in
                                               self.npmc_input.spectral_kinetics.dopants])
            # Avoid divide by zero by making normalization factor = 1 for elements not in the shell
            normalization_factors[normalization_factors == 0] = 1
            layer_population = np.divide(layer_population, normalization_factors)

            # Add to the list of populations separated by layer
            population_by_constraint.append(layer_population)

        return population_by_constraint
