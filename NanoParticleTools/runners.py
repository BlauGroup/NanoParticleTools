from NanoParticleTools.core import NanoParticle
from typing import Optional
import pickle
import subprocess
import os

class NPMCRunner:
    def __init__(self, interactions_csv_path: Optional[str] = None,
                 sites_csv_path: Optional[str] = None,
                 energies_csv_path: Optional[str] = None,
                 output_dir: Optional[str] = './scratch'):
        os.system('rm -rf ./scratch; mkdir scratch')

        if interactions_csv_path is None:
            interactions_csv_path = './combi_nano_test_system/interactions.csv'
        if sites_csv_path is None:
            sites_csv_path = './combi_nano_test_system/sites.csv'
        if energies_csv_path is None:
            energies_csv_path = "./combi_nano_test_system/energy_levels.csv"

        nano_particle = NanoParticle(interactions_csv_path,
                                     sites_csv_path,
                                     energies_csv_path)

        self.np_database = os.path.join(output_dir, 'np.sqlite')
        self.initial_state = os.path.join(output_dir, 'initial_state.sqlite')
        nano_particle.generate_nano_particle_database(self.np_database)
        nano_particle.generate_initial_state_database(self.initial_state)

        self.nanoparticle_pickle = os.path.join(output_dir, 'nano_particle.pickle')
        with open(self.nanoparticle_pickle, 'wb') as f:
            pickle.dump(nano_particle, f)

    def run(self,
            num_sims: int = 10,
            base_seed: int = 1000,
            thread_count: int = 8,
            simulation_length: int = 100000):
        """

        :param np_database: a sqlite database containing the reaction network and metadata.
        :param initial_state: a sqlite database containing initial state. The simulation
            trajectories are also written into the database
        :param num_sims: an integer specifying how many simulations to run
        :param base_seed: seeds used are base_seed, base_seed+1, ..., base_seed+number_of_simulations-1
        :param thread_count: number of threads to use
        :param simulation_length:
        :return:
        """
        run_args = ['NPMC',
                    f'--nano_particle_database={self.np_database}',
                    f'--initial_state_database={self.initial_state}',
                    f'--number_of_simulations={str(num_sims)}',
                    f'--base_seed={str(base_seed)}',
                    f'--thread_count={str(thread_count)}',
                    f'--step_cutoff={str(simulation_length)}']
        subprocess.run(' '.join(run_args), shell=True)