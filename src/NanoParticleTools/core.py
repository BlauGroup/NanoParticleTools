import sqlite3
from typing import Optional, Sequence, Union
import subprocess
from NanoParticleTools.inputs.nanoparticle import DopedNanoparticle
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.inputs.util import get_all_interactions, get_sites, get_species
import signal
import time
from monty.json import MSONable
from functools import lru_cache

create_species_table_sql = """
    CREATE TABLE species (
        species_id          INTEGER NOT NULL PRIMARY KEY,
        degrees_of_freedom  INTEGER NOT NULL
    );
"""

insert_species_sql = """
    INSERT INTO species VALUES (?,?);
"""

create_sites_table_sql = """
    CREATE TABLE sites (
        site_id             INTEGER NOT NULL PRIMARY KEY,
        x                   REAL NOT NULL,
        y                   REAL NOT NULL,
        z                   REAL NOT NULL,
        species_id          INTEGER NOT NULL
    );
"""

insert_site_sql = """
    INSERT INTO sites VALUES (?,?,?,?,?);
"""

create_interactions_table_sql = """
    CREATE TABLE interactions (
        interaction_id      INTEGER NOT NULL PRIMARY KEY,
        number_of_sites     INTEGER NOT NULL,
        species_id_1        INTEGER NOT NULL,
        species_id_2        INTEGER NOT NULL,
        left_state_1        INTEGER NOT NULL,
        left_state_2        INTEGER NOT NULL,
        right_state_1       INTEGER NOT NULL,
        right_state_2       INTEGER NOT NULL,
        rate                REAL NOT NULL,
        interaction_type    TEXT NOT NULL
    );
"""

insert_interaction_sql = """
    INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?,?);
"""

create_metadata_table_sql = """
    CREATE TABLE metadata (
        number_of_species                   INTEGER NOT NULL,
        number_of_sites                     INTEGER NOT NULL,
        number_of_interactions              INTEGER NOT NULL
    );
"""

insert_metadata_sql = """
    INSERT INTO metadata VALUES (?,?,?);
"""

create_initial_state_table_sql = """
    CREATE TABLE initial_state (
        site_id            INTEGER NOT NULL PRIMARY KEY,
        degree_of_freedom  INTEGER NOT NULL
    );
"""

create_trajectories_table_sql = """
    CREATE TABLE trajectories (
        seed               INTEGER NOT NULL,
        step               INTEGER NOT NULL,
        time               REAL NOT NULL,
        site_id_1          INTEGER NOT NULL,
        site_id_2          INTEGER NOT NULL,
        interaction_id     INTEGER NOT NULL
);
"""

insert_initial_state_sql = """
    INSERT INTO initial_state VALUES (?,?);
"""

create_factors_table_sql = """
    CREATE TABLE factors (
        one_site_interaction_factor      REAL NOT NULL,
        two_site_interaction_factor      REAL NOT NULL,
        interaction_radius_bound         REAL NOT NULL,
        distance_factor_type             TEXT NOT NULL
);
"""

insert_factors_sql = """
    INSERT INTO factors VALUES (?,?,?,?);
"""

sql_get_trajectory = """
    SELECT * FROM trajectories;
"""

create_interupt_state_sql = """
    CREATE TABLE interupt_state (
        seed                INTEGER NOT NULL,
        site_id             INTEGER NOT NULL,
        degree_of_freedom  INTEGER NOT NULL
    );
"""

create_interupt_cutoff_sql = """
    CREATE TABLE interupt_cutoff (
        seed                INTEGER NOT NULL,
        step                INTEGER NOT NULL,
        time                REAL NOT NULL
    );
"""


class NPMCInput(MSONable):

    def load_trajectory(self, seed, database_file):
        with sqlite3.connect(database_file) as con:
            cur = con.cursor()

            trajectory = []
            sql_get_single_trajectory = f"select * from trajectories where seed={seed}"
            for row in cur.execute(sql_get_single_trajectory):
                seed = row[0]
                # step = row[1] # This is not used, save some time/memory by keeping it commented
                time = row[2]
                site_id_1 = row[3]
                site_id_2 = row[4]
                interaction_id = row[5]

                trajectory.append([site_id_1, site_id_2, interaction_id, time])

        if len(trajectory) == 0:
            raise ValueError("Invalid Seed")
        else:
            return trajectory

    def load_trajectories(self, database_file: str):
        with sqlite3.connect(database_file) as con:
            cur = con.cursor()

            trajectories = {}
            for row in cur.execute(sql_get_trajectory):
                seed = row[0]
                _time = row[2]
                site_id_1 = row[3]
                site_id_2 = row[4]
                interaction_id = row[5]

                if seed not in trajectories:
                    trajectories[seed] = []

                trajectories[seed].append(
                    [site_id_1, site_id_2, interaction_id, _time])

            self.trajectories = trajectories

    def __init__(self,
                 spectral_kinetics: SpectralKinetics,
                 nanoparticle: DopedNanoparticle,
                 initial_states: Optional[Sequence[int]] = None):

        self.spectral_kinetics = spectral_kinetics
        self.nanoparticle = nanoparticle
        if initial_states is None:
            self.initial_states = [0 for _ in self.sites]
        else:
            self.initial_states = initial_states

    @property
    @lru_cache
    def interactions(self):
        return get_all_interactions(self.spectral_kinetics)

    @property
    @lru_cache
    def sites(self):
        if self.nanoparticle.has_structure is False:
            self.nanoparticle.generate()
        return get_sites(self.nanoparticle, self.spectral_kinetics)

    @property
    @lru_cache
    def species(self):
        return get_species(self.spectral_kinetics)

    def generate_initial_state_database(
            self,
            database_file: str,
            one_site_interaction_factor: Optional[Union[float, int]] = 1,
            two_site_interaction_factor: Optional[Union[float, int]] = 1,
            interaction_radius_bound: Optional[Union[float, int]] = 3,
            distance_factor_type: Optional[str] = 'inverse_cubic'):
        """

        Args:
            database_file (str): name of file to write database to
            one_site_interaction_factor (float, int, None): Weighting for one
                site interactions. Can be used to boost their occurrence.

                Defaults to 1.
            two_site_interaction_factor (float, int, None): Weighting for two
                site interactions. Can be used to boost their occurrence.

                Defaults to 1.
            interaction_radius_bound (float, int, None): Maximum distance allowed
                for an ET interaction.

                Defaults to 3.
            distance_factor_type (Optional[str], optional): Accepted values are
                'linear' and 'inverse_cubic'.

                Defaults to 'inverse_cubic'.
        """
        # TODO: parameterize over initial state
        with sqlite3.connect(database_file) as con:
            cur = con.cursor()

            cur.execute(create_initial_state_table_sql)
            cur.execute(create_trajectories_table_sql)
            cur.execute(create_factors_table_sql)
            cur.execute(create_interupt_state_sql)
            cur.execute(create_interupt_cutoff_sql)
            con.commit()

            for i in self.sites:
                site_id = self.sites[i]['site_id']
                state_id = self.initial_states[i]
                cur.execute(insert_initial_state_sql, (site_id, state_id))

            con.commit()

            cur.execute(
                insert_factors_sql,
                (one_site_interaction_factor, two_site_interaction_factor,
                 interaction_radius_bound, distance_factor_type))

            con.commit()

    def generate_nano_particle_database(self, database_file: str):
        with sqlite3.connect(database_file) as con:
            cur = con.cursor()

            # create tables
            cur.execute(create_species_table_sql)
            cur.execute(create_sites_table_sql)
            cur.execute(create_interactions_table_sql)
            cur.execute(create_metadata_table_sql)

            for i in self.species:
                row = self.species[i]
                cur.execute(insert_species_sql,
                            (row['species_id'], row['degrees_of_freedom']))

            con.commit()

            for i in self.sites:
                row = self.sites[i]
                cur.execute(insert_site_sql,
                            (row['site_id'], row['x'], row['y'], row['z'],
                             row['species_id']))
            con.commit()

            for i in self.interactions:
                row = self.interactions[i]
                cur.execute(insert_interaction_sql,
                            (row['interaction_id'], row['number_of_sites'],
                             row['species_id_1'], row['species_id_2'],
                             row['left_state_1'], row['left_state_2'],
                             row['right_state_1'], row['right_state_2'],
                             row['rate'], row['interaction_type']))

            cur.execute(
                insert_metadata_sql,
                (len(self.species), len(self.sites), len(self.interactions)))

            con.commit()


class NPMCRunner:

    def __init__(self, np_db_path, initial_state_db_path):
        self.np_database = np_db_path
        self.initial_state = initial_state_db_path
        signal.signal(signal.SIGINT, self.error_handler)
        signal.signal(signal.SIGTERM, self.error_handler)
        self.process = None

    def error_handler(self, _signo, _stack_frame):
        # Sleep for 30 seconds before exiting
        time.sleep(40)
        # Kill the program
        raise RuntimeError(f'Job killed by signal {_signo}')

    def run(self,
            npmc_command: str = 'NPMC',
            num_sims: int = 10,
            base_seed: int = 1000,
            thread_count: int = 8,
            simulation_length: int = None,
            simulation_time: float = None,
            log_file=''):
        """

        :param np_database: a sqlite database containing the reaction network
            and metadata.
        :param initial_state: a sqlite database containing initial state.
            The simulation trajectories are also written into the database
        :param num_sims: an integer specifying how many simulations to run
        :param base_seed: seeds used are:
            [base_seed, base_seed+1, ..., base_seed+number_of_simulations-1]
        :param thread_count: number of threads to use
        :param simulation_length:
        :return:
        """
        run_args = [
            npmc_command, f'--nano_particle_database={self.np_database}',
            f'--initial_state_database={self.initial_state}',
            f'--number_of_simulations={str(num_sims)}',
            f'--base_seed={str(base_seed)}',
            f'--thread_count={str(thread_count)}'
        ]
        if simulation_length is not None:
            run_args.append(f'--step_cutoff={str(simulation_length)}')
        elif simulation_time is not None:
            run_args.append(f'--time_cutoff={str(simulation_time)}')
        print(f"Running NPMC using the command: \"{' '.join(run_args)}\"")
        with open(log_file + "stdout",
                  "a") as f_std, open(log_file + "stderr", "a",
                                      buffering=1) as f_err:
            self.process = subprocess.Popen(run_args,
                                            stdout=f_std,
                                            stderr=f_err)
        self.process.wait()
