import csv
import sqlite3
import warnings
from typing import Optional, Sequence, Union
import subprocess
from NanoParticleTools.inputs.nanoparticle import DopedNanoparticle
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.inputs.util import get_all_interactions, get_sites, get_species

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
        interaction_rate_threshold       REAL NOT NULL,
        distance_factor_type             TEXT NOT NULL
);
"""

insert_factors_sql = """
    INSERT INTO factors VALUES (?,?,?,?, ?);
"""

sql_get_trajectory = """
    SELECT * FROM trajectories;
"""
from monty.json import MSONable
from functools import lru_cache

class NPMCInput(MSONable):
    def load_trajectories(self,
                          database_file:str):
        con = sqlite3.connect(database_file)
        cur = con.cursor()

        trajectories = {}
        for row in cur.execute(sql_get_trajectory):
            seed = row[0]
            # step = row[1] # This is not used, save some time/memory by keeping it commented
            time = row[2]
            site_id_1 = row[3]
            site_id_2 = row[4]
            interaction_id = row[5]

            if seed not in trajectories:
                trajectories[seed] = []

            trajectories[seed].append([site_id_1, site_id_2, interaction_id, time])

        self.trajectories = trajectories

    def __init__(self, spectral_kinetics: SpectralKinetics,
                 nanoparticle: DopedNanoparticle,
                 initial_states: Optional[Sequence[int]]=None):

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
        if self.nanoparticle.has_structure == False:
            self.nanoparticle.generate()
        return get_sites(self.nanoparticle, self.spectral_kinetics)

    @property
    @lru_cache
    def species(self):
        return get_species(self.spectral_kinetics)

    def generate_initial_state_database(self,
                                        database_file: str,
                                        one_site_interaction_factor: Optional[Union[float, int]] = 1,
                                        two_site_interaction_factor: Optional[Union[float, int]] = 1,
                                        interaction_radius_bound: Optional[Union[float, int]] = 3,
                                        interaction_rate_threshold: Optional[Union[float, int]] = 1e20,
                                        distance_factor_type: Optional[str] = 'inverse_cubic'):
        """

        :param database_file: name of file to write database to
        :param one_site_interaction_factor: Weighting for one site interactions. Can be used to boost their occurrence.
        :param two_site_interaction_factor: Weighting for two site interactions. Can be used to boost their occurrence.
        :param interaction_radius_bound: Maximum distance allowed for an ET interaction.
        :param interaction_rate_threshold: Threshold rate to consider an ET interaction.
            Only applies if sites are outside of the interaction_radius_bound.
        :param distance_factor_type: Accepted values are 'linear' and 'inverse_cubic'
        :return:
        """
        # TODO: parameterize over initial state
        con = sqlite3.connect(database_file)
        cur = con.cursor()

        cur.execute(create_initial_state_table_sql)
        cur.execute(create_trajectories_table_sql)
        cur.execute(create_factors_table_sql)
        con.commit()

        for i in self.sites:
            site_id = self.sites[i]['site_id']
            state_id = self.initial_states[i]
            cur.execute(insert_initial_state_sql,
                        (site_id,
                         state_id
                         ))

        con.commit()

        cur.execute(insert_factors_sql,
                    (one_site_interaction_factor, two_site_interaction_factor, interaction_radius_bound, interaction_rate_threshold, distance_factor_type))

        con.commit()

    def generate_nano_particle_database(self,
                                        database_file:str):
        con = sqlite3.connect(database_file)
        cur = con.cursor()

        # create tables
        cur.execute(create_species_table_sql)
        cur.execute(create_sites_table_sql)
        cur.execute(create_interactions_table_sql)
        cur.execute(create_metadata_table_sql)

        for i in self.species:
            row = self.species[i]
            cur.execute(insert_species_sql,
                        (row['species_id'],
                         row['degrees_of_freedom']))

        con.commit()

        for i in self.sites:
            row = self.sites[i]
            cur.execute(insert_site_sql,
                        (row['site_id'],
                         row['x'],
                         row['y'],
                         row['z'],
                         row['species_id']))
        con.commit()

        for i in self.interactions:
            row = self.interactions[i]
            cur.execute(insert_interaction_sql,
                        (row['interaction_id'],
                         row['number_of_sites'],
                         row['species_id_1'],
                         row['species_id_2'],
                         row['left_state_1'],
                         row['left_state_2'],
                         row['right_state_1'],
                         row['right_state_2'],
                         row['rate'],
                         row['interaction_type']))

        cur.execute(insert_metadata_sql,
                    (len(self.species),
                     len(self.sites),
                     len(self.interactions)))

        con.commit()


class NPMCRunner:
    def __init__(self,
                 np_db_path,
                 initial_state_db_path):
        self.np_database = np_db_path
        self.initial_state = initial_state_db_path

    def run(self,
            npmc_command: str = 'NPMC',
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
        run_args = [npmc_command,
                    f'--nano_particle_database={self.np_database}',
                    f'--initial_state_database={self.initial_state}',
                    f'--number_of_simulations={str(num_sims)}',
                    f'--base_seed={str(base_seed)}',
                    f'--thread_count={str(thread_count)}',
                    f'--step_cutoff={str(simulation_length)}']
        subprocess.run(' '.join(run_args), shell=True)