import csv
import sqlite3
import warnings

# number_of_sites | species_1 | species_2 | left_state_1 | left_state_2 |
# right_state_1 | right_state_2 | rate

# number_of_sites is the number of sites involved in the
# interaction. Either 1 or 2

# species_1 and species_2 are things like
# "Yb", "Er" or "N/A".

# left_state_1 left_state_2 right_state_1 and right_state_2 are things
# like "level_0", "level_3" or N/A.
# level_0 corresponds to the ground state of the ion, level_1
# corresponds to the first excited state and so on.

# rate is a number with units 1/s or cm^6/s for one and two site
# interactions respectively

# sites spreadsheet columns:

# x | y | z | species

# x, y and z are numbers with units nm

# species are things like "Yb" or "Er"

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


class NanoParticle:
    def load_trajectories(self, database_file):
        con = sqlite3.connect(database_file)
        cur = con.cursor()

        trajectories = {}
        for row in cur.execute(sql_get_trajectory):
            seed = row[0]
            step = row[1]
            time = row[2]
            site_id_1 = row[3]
            site_id_2 = row[4]
            interaction_id = row[5]

            if seed not in trajectories:
                trajectories[seed] = {}

            trajectories[seed][step] = (site_id_1, site_id_2, interaction_id, time)

        self.trajectories = trajectories

    def generate_initial_state_database(self, database_file):
        # TODO: parameterize over initial state
        con = sqlite3.connect(database_file)
        cur = con.cursor()

        cur.execute(create_initial_state_table_sql)
        cur.execute(create_trajectories_table_sql)
        cur.execute(create_factors_table_sql)
        con.commit()

        for i in self.sites:
            site_id = self.sites[i]['site_id']
            species_id = self.sites[i]['species_id']
            state_id = self.species_state_name_to_id[species_id]['level_0']
            cur.execute(insert_initial_state_sql,
                        (site_id,
                         state_id
                         ))

        con.commit()

        cur.execute(insert_factors_sql,
                    (1.0, 1, 3, "inverse_cubic"))

        con.commit()

    def generate_nano_particle_database(self, database_file):
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

    def __init__(self, interactions_csv_path, sites_csv_path, energies_csv_path):

        self.species_name_to_id = {}
        self.id_to_species_name = {}
        self.sites = {}

        species_id = 0
        site_id = 0
        with open(sites_csv_path) as sites_csv:
            sites_reader = csv.DictReader(sites_csv, delimiter=',')
            for row in sites_reader:
                if row['species'] not in self.species_name_to_id:
                    self.species_name_to_id[row['species']] = species_id
                    self.id_to_species_name[species_id] = row['species']
                    species_id += 1

                self.sites[site_id] = {
                    'site_id': site_id,
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'species_id': self.species_name_to_id[row['species']]}

                site_id += 1

        self.species_state_name_to_id = {}
        self.id_to_species_state_name = {}
        self.interactions = {}

        for i in self.id_to_species_name.keys():
            self.species_state_name_to_id[i] = {}
            self.id_to_species_state_name[i] = {}

        species_state_counter = [0 for _ in range(len(self.id_to_species_name))]

        interaction_id = 0
        with open(interactions_csv_path) as interactions_csv:
            interactions_reader = csv.DictReader(interactions_csv, delimiter=',')
            for row in interactions_reader:
                #                 print(row.keys())
                if (row['number_of_sites'] == '1'):

                    species_id = self.species_name_to_id[row['species_1']]
                    if row['left_state_1'] not in self.species_state_name_to_id[species_id]:
                        self.species_state_name_to_id[species_id][
                            row['left_state_1']] = species_state_counter[species_id]

                        self.id_to_species_state_name[species_id][
                            species_state_counter[species_id]] = row['left_state_1']

                        species_state_counter[species_id] += 1

                    if row['right_state_1'] not in self.species_state_name_to_id[species_id]:
                        self.species_state_name_to_id[species_id][
                            row['right_state_1']] = species_state_counter[species_id]

                        self.id_to_species_state_name[species_id][
                            species_state_counter[species_id]] = row['right_state_1']

                        species_state_counter[species_id] += 1

                    self.interactions[interaction_id] = {
                        'interaction_id': interaction_id,
                        'number_of_sites': int(row['number_of_sites']),
                        'species_id_1': species_id,
                        'species_id_2': -1,
                        'left_state_1': self.species_state_name_to_id[species_id][
                            row['left_state_1']],
                        'left_state_2': -1,
                        'right_state_1': self.species_state_name_to_id[species_id][
                            row['right_state_1']],
                        'right_state_2': -1,
                        'rate': float(row['rate']),
                        'interaction_type': row['transitionType']
                    }

                    interaction_id += 1

                if (row['number_of_sites'] == '2'):

                    species_id_1 = self.species_name_to_id[row['species_1']]
                    species_id_2 = self.species_name_to_id[row['species_2']]

                    if row['left_state_1'] not in self.species_state_name_to_id[species_id_1]:
                        self.species_state_name_to_id[species_id_1][
                            row['left_state_1']] = species_state_counter[species_id_1]

                        self.id_to_species_state_name[species_id_1][
                            species_state_counter[species_id_1]] = row['left_state_1']

                        species_state_counter[species_id_1] += 1

                    if row['right_state_1'] not in self.species_state_name_to_id[species_id_1]:
                        self.species_state_name_to_id[species_id_1][
                            row['right_state_1']] = species_state_counter[species_id_1]

                        self.id_to_species_state_name[species_id_1][
                            species_state_counter[species_id_1]] = row['right_state_1']

                        species_state_counter[species_id_1] += 1

                    if row['left_state_2'] not in self.species_state_name_to_id[species_id_2]:
                        self.species_state_name_to_id[species_id_2][
                            row['left_state_2']] = species_state_counter[species_id_2]

                        self.id_to_species_state_name[species_id_2][
                            species_state_counter[species_id_2]] = row['left_state_2']

                        species_state_counter[species_id_2] += 1

                    if row['right_state_2'] not in self.species_state_name_to_id[species_id_2]:
                        self.species_state_name_to_id[species_id_2][
                            row['right_state_2']] = species_state_counter[species_id_2]

                        self.id_to_species_state_name[species_id_2][
                            species_state_counter[species_id_2]] = row['right_state_2']

                        species_state_counter[species_id_2] += 1

                    self.interactions[interaction_id] = {
                        'interaction_id': interaction_id,
                        'number_of_sites': int(row['number_of_sites']),
                        'species_id_1': species_id_1,
                        'species_id_2': species_id_2,
                        'left_state_1': self.species_state_name_to_id[species_id_1][
                            row['left_state_1']],
                        'left_state_2': self.species_state_name_to_id[species_id_2][
                            row['left_state_2']],
                        'right_state_1': self.species_state_name_to_id[species_id_1][
                            row['right_state_1']],
                        'right_state_2': self.species_state_name_to_id[species_id_2][
                            row['right_state_2']],

                        # 2 site interaction rates come with units cm^6 / s
                        'rate': (1.0e42) * float(row['rate']),
                        'interaction_type': row['transitionType']
                    }

                    interaction_id += 1

        self.species = {}
        for i in self.id_to_species_name:
            self.species[i] = {
                'species_id': i,
                'degrees_of_freedom': len(self.id_to_species_state_name[i])
            }

        # Load Energies from the Energies_csv
        self.species_state_name_to_energy = {}
        self.species_state_id_to_energy = {}
        with open(energies_csv_path) as energies_csv:
            sites_reader = csv.DictReader(energies_csv, delimiter=',')
            for row in sites_reader:
                species_name = row['species']
                state_name = row['name']
                species_id = self.species_name_to_id[species_name]
                if species_name not in self.species_state_name_to_energy:
                    self.species_state_name_to_energy[species_name] = {}
                    self.species_state_id_to_energy[species_id] = {}

                if state_name not in self.species_state_name_to_id[species_id]:
                    warnings.warn('energy state is not present in any interactions, skipping')
                    continue
                state_id = self.species_state_name_to_id[species_id][state_name]
                self.species_state_name_to_energy[species_name][state_name] = float(row['energy'])
                self.species_state_id_to_energy[species_id][state_id] = float(row['energy'])

            # print(self.species_state_id_to_energy)
            # print(self.species_state_name_to_energy)
            # #Populate energy changes into the interactions
            # energy_per_interaction = {}
            # for key, interaction in self.interactions.items():
            #     particle_id = []
            #     E1i = self.species_state_id_to_energy[interaction['species_id_1']][interaction['left_state_1']]
            #     E1f = self.species_state_id_to_energy[interaction['species_id_1']][interaction['right_state_1']]
            #     energy_per_interaction[key] = (E1f-E1i)
            #     if interaction['species_id_2'] != -1:
            #         print(interaction)
            #         E2i = self.species_state_id_to_energy[interaction['species_id_2']][interaction['left_state_2']]
            #         E2f = self.species_state_id_to_energy[interaction['species_id_2']][interaction['right_state_2']]
            #         energy_per_interaction[key] += (E2f-E2i)
            #     self.interactions[key]['dE'] = energy_per_interaction[key]