from typing import Optional

import numpy as np
import pandas as pd


class SimulationReplayer():
    """
    reconstruct simulations
    """

    def __init__(self, nano_particle):
        self.nano_particle = nano_particle

        interactions_by_sites_per_traj = {}
        interactions_by_id_per_traj = {}
        simulation_time_per_traj = {}
        for key, trajectory in nano_particle.trajectories.items():
            simulation_time, interactions_by_sites, interactions_by_id = self.bin_interactions(trajectory)
            interactions_by_sites_per_traj[key] = interactions_by_sites
            interactions_by_id_per_traj[key] = interactions_by_id
            simulation_time_per_traj[key] = simulation_time

        self.interactions_by_sites_per_traj = interactions_by_sites_per_traj
        self.interactions_by_id_per_traj = interactions_by_id_per_traj
        self.simulation_time_per_traj = simulation_time_per_traj

    def compute_populations(self):
        for key, traj in self.nano_particle.trajectories.items():
            _sites = dict([(key, []) for key in self.nano_particle.sites.keys()])
            for event_id, event in traj.items():
                site1, site2, interaction_id, time = event

                #                 i_dict = nano_particle.interactions[interaction_id]

    def bin_interactions(self, trajectory):
        simulation_time = 0
        # Initialize dictionaries to keep track of the events.
        interactions_by_sites = dict([(key, []) for key in self.nano_particle.sites.keys()])
        interactions_by_id = dict([(key, []) for key in self.nano_particle.interactions.keys()])

        # Iterate through all the events that occured
        for event_id, event in trajectory.items():
            # Unpack information from the interaction that occured
            site1, site2, interaction_id, time = event

            # Update the total time that has elapsed during the simulation
            simulation_time += time
            simulation_time = time

            # Iterate the interactions counter for this event
            interactions_by_id[interaction_id].append((site1, site2, interaction_id, simulation_time))

            # Add the interaction to the specific site
            # Use the 'left' and 'right' metadata to keep track of whether the atom was a donor or acceptor
            # 'left' = donor; 'right' = 'acceptor'
            interactions_by_sites[site1].append((interaction_id, simulation_time, 'left'))

            # If this is a 2 site interaction, add the interaction to the bookkeeping for the acceptor site
            if site2 != -1:
                interactions_by_sites[site2].append((interaction_id, simulation_time, 'right'))

        return simulation_time, interactions_by_sites, interactions_by_id

    def site_event_statistics(self, site):
        pass

    def count_by_site(self):
        pass

    def event_statistics(self):
        """
        Generate a dictionary with the total counts of each interaction that occurred during a single trajectory.
        For clarity, this is not averaged over all trajectories.
        """
        statistics_by_interaction_id = {}
        for seed, traj in self.interactions_by_id_per_traj.items():
            statistics_by_interaction_id[seed] = dict(
                [[key, len(item)] for key, item in self.interactions_by_id_per_traj[seed].items()])

        return statistics_by_interaction_id

    def normalize_by_nsites(self, interactions_dict):
        """
        Helper function to normalize a dict of interactions by the number of sites
        """

        normalized_interactions_dict = interactions_dict.copy()

        for interaction_id, count in interactions_dict.items():
            interaction = self.nano_particle.interactions[interaction_id]
            nsites = sum(
                [1 for key, item in self.nano_particle.sites.items() if
                 item['species_id'] == interaction['species_id_1']])

            normalized_interactions_dict[interaction_id] /= nsites

        return normalized_interactions_dict

    def average_over_trajectories(self, dict_by_traj):
        avg_dict = {}
        for key in self.nano_particle.interactions:
            avg_dict[key] = np.mean([_d[key] for _key, _d in dict_by_traj.items()])

        return avg_dict

    def calc_dNdT(self, normalize_by_atoms: Optional = False):
        dNdT = {}
        for key, traj in self.event_statistics().items():
            if normalize_by_atoms:
                norm_event_statistics = self.normalize_by_nsites(traj)
            else:
                norm_event_statistics = traj
            for interaction_id in norm_event_statistics:
                norm_event_statistics[interaction_id] = norm_event_statistics[interaction_id] / \
                                                        self.simulation_time_per_traj[key]

            dNdT[key] = norm_event_statistics

        return dNdT, self.average_over_trajectories(dNdT)

    def photon_shot_chart(self):
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import cm

        radiative_interactions = [key for key, interaction in self.nano_particle.interactions.items() if
                                  interaction['interaction_type'] == 'rad']
        np.random.shuffle(radiative_interactions)

        fig = plt.figure(figsize=[9, 9], dpi=150)
        for i in self.nano_particle.sites:
            events = self.interactions_by_sites_per_traj[2000][i]

            timeseries = {}
            for event in events:
                if event[0] in radiative_interactions:
                    try:
                        timeseries[event[0]].append(event[1])
                    except:
                        timeseries[event[0]] = [event[1]]

            for key, series in timeseries.items():
                #         plt.plot(series, [i for q in range(len(series))], 'D', label=f'interaction_id: {key}', alpha=0.5, markersize=1)
                plt.plot(series, [i for q in range(len(series))], 'D', label=f'interaction_id: {key}', alpha=1,
                         color=cm.tab10(radiative_interactions.index(key) / len(radiative_interactions)),
                         markersize=1.5)
        plt.ylabel('Atom')
        plt.xlabel('time (s)')
        # plt.xlim(0, 5e-3)
        custom_lines = [Line2D([0], [0], color=cm.tab10(i / len(radiative_interactions)), lw=4) for i, key in
                        enumerate(radiative_interactions)]
        plt.legend(custom_lines, [f'interaction_id: {key}' for key in radiative_interactions], bbox_to_anchor=(1, 1.))
        return fig

    def get_summary_dict(self):
        remapped_interactions = {}
        for key in self.nano_particle.interactions:

            site1_id = self.nano_particle.interactions[key]['species_id_1']
            site2_id = self.nano_particle.interactions[key]['species_id_2']

            _d = {'interaction_id': self.nano_particle.interactions[key]['interaction_id'],
                  'number_of_sites': self.nano_particle.interactions[key]['number_of_sites'],
                  }
            _d['species_id_1'] = self.nano_particle.id_to_species_name[site1_id]
            _d['left_state_1'] = self.nano_particle.id_to_species_state_name[site1_id][
                self.nano_particle.interactions[key]['left_state_1']]
            _d['right_state_1'] = self.nano_particle.id_to_species_state_name[site1_id][
                self.nano_particle.interactions[key]['right_state_1']]

            if site2_id != -1:
                _d['species_id_2'] = self.nano_particle.id_to_species_name[site2_id]
                _d['left_state_2'] = self.nano_particle.id_to_species_state_name[site2_id][
                    self.nano_particle.interactions[key]['left_state_2']]
                _d['right_state_2'] = self.nano_particle.id_to_species_state_name[site2_id][
                    self.nano_particle.interactions[key]['right_state_2']]
            else:
                _d['species_id_2'] = None
                _d['left_state_2'] = None
                _d['right_state_2'] = None

            _d['rate'] = self.nano_particle.interactions[key]['rate']

            if self.nano_particle.interactions[key]['number_of_sites'] == 2:
                _d['rate'] *= (1.0e-42)
            _d['interaction_type'] = self.nano_particle.interactions[key]['interaction_type']

            # Reorganize dictionary into the desired order
            desired_order = ['interaction_id', 'number_of_sites', 'species_id_1', 'species_id_2',
                             'left_state_1', 'left_state_2', 'right_state_1', 'right_state_2', 'rate',
                             'interaction_type']
            remapped_interactions[key] = {key: _d[key] for key in desired_order}

        df = pd.DataFrame.from_dict(remapped_interactions, orient='index')
        #
        traj_dndt_dict, avg_dndt_dict = self.calc_dNdT(normalize_by_atoms=True)
        dndt = [avg_dndt_dict[key] for key in sorted(list(avg_dndt_dict.keys()))]
        df.insert(9, 'dNdT (#/s*atoms)', dndt)

        traj_dndt_dict, avg_dndt_dict = self.calc_dNdT(normalize_by_atoms=False)
        dndt = [avg_dndt_dict[key] for key in sorted(list(avg_dndt_dict.keys()))]
        df.insert(9, 'dNdT(#/s)', dndt)
        #     df.to_csv('Simulation_results.csv')
        return df

    def photon_shot_chart_with_energies(self):
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import cm

        radiative_interactions = [key for key, interaction in self.nano_particle.interactions.items() if
                                  interaction['interaction_type'] == 'rad']
        #         radiative_interactions = [key for key, interaction in self.nano_particle.interactions.items() if interaction['interaction_type'] !='NR']
        radiative_interactions = [key for key, interaction in self.nano_particle.interactions.items()]

        timeseries = {key: [] for key in radiative_interactions}
        for i in self.nano_particle.sites:
            events = self.interactions_by_sites_per_traj[2000][i]

            for event in events:
                if event[0] in radiative_interactions:
                    timeseries[event[0]].append(event[1])

        fig = plt.figure(figsize=[9, 9], dpi=150)

        for key, series in timeseries.items():
            plt.plot(series, [self.nano_particle.interactions[key]['dE'] for q in range(len(series))], 'D',
                     label=f'interaction_id: {key}', alpha=1,
                     color=cm.tab10(radiative_interactions.index(key) / len(radiative_interactions)), markersize=1.5)

        plt.yscale('symlog')
        plt.ylabel('Energy ($cm^{-1})$')
        plt.xlabel('time (s)')
        # plt.xlim(0, 5e-3)
        custom_lines = [Line2D([0], [0], color=cm.tab10(i / len(radiative_interactions)), lw=4) for i, key in
                        enumerate(radiative_interactions)]
        plt.legend(custom_lines, [f'interaction_id: {key}' for key in radiative_interactions], bbox_to_anchor=(1, 1.))
        return fig
