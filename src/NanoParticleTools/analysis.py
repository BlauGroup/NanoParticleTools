from typing import Optional

import numpy as np
import pandas as pd
from collections import Counter
from functools import lru_cache
import os
import json
from monty.json import MontyDecoder
from NanoParticleTools.inputs.util import specie_energy_level_to_combined_energy_level


class SimulationReplayer():
    def __init__(self, trajectories, npmc_input):
        self.trajectories = trajectories
        self.npmc_input = npmc_input

    @classmethod
    def from_run_directory(cls, run_dir):
        with open(os.path.join(run_dir, 'npmc_input.json'), 'rb') as f:
            npmc_input = json.load(f, cls=MontyDecoder)
        npmc_input.load_trajectories(os.path.join(run_dir, 'initial_state.sqlite'))

        trajectories = {}
        for seed in npmc_input.trajectories:
            _traj = Trajectory(seed, npmc_input)
            trajectories[seed] = _traj
        return cls(trajectories, npmc_input)

    def get_single_trajectory_summary(self, seed=None):
        try:
            trajectory = self.trajectories[seed]
        except:
            seed = list(self.trajectories.keys())[0]
            trajectory = self.trajectories[seed]

        return trajectory.get_summary()

    def get_summaries(self):
        dndts = {}
        for seed in self.npmc_input.trajectories:
            _d = {}
            _d['summary'] = self.get_single_trajectory_summary(seed)
            _d['simulation_time'] = self.trajectories[seed].simulation_time
            _d['x_populations'], _d['y_populations'] = self.trajectories[seed].get_population_evolution()
            # _d['x_states'], _d['y_states'] = self.trajectories[seed].get_state_evolution_by_site()
            dndts[seed] = _d

        return dndts


class Trajectory():
    def __init__(self, seed, npmc_input):
        self.seed = seed
        self.npmc_input = npmc_input
        self.species_counter = Counter([site['species_id'] for site in self.npmc_input.sites.values()])

    @property
    def initial_states(self):
        return self.npmc_input.initial_states

    @property
    def trajectory(self):
        return self.npmc_input.trajectories[self.seed]

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

    def get_state_evolution(self, step_size=1e-6):
        states = np.zeros((len(self.initial_states), self.npmc_input.spectral_kinetics.total_n_levels))
        for i, state in enumerate(self.initial_states):
            states[i, state] = 1

        x = np.arange(0, self.simulation_time, step_size)
        site_evolution = np.zeros((len(x), states.shape[0]))
        population_evolution = np.zeros((len(x), states.shape[1]))
        current_time = 0
        event_i = 0
        for i, time_interval in enumerate(x):
            while current_time < time_interval:
                # Propagate steps
                donor_i, acceptor_i, _interaction_id, _time = self.trajectory[event_i]
                _interaction = self.npmc_input.interactions[_interaction_id]

                states[donor_i][_interaction['left_state_1']] = 0
                states[donor_i][_interaction['right_state_1']] = 1
                if _interaction['number_of_sites'] == 2:
                    states[acceptor_i][_interaction['left_state_2']] = 0
                    states[acceptor_i][_interaction['right_state_2']] = 1
                event_i += 1
                current_time = _time

            # Save states
            population_evolution[i, :] = np.sum(states, axis=0)
            site_evolution[i, :] = np.where(states > 0)[1]
        return x, population_evolution, site_evolution
