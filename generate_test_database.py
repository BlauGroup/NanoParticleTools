# The purpose of this file is to generate a test system for development
# of a nano particle simulator.

# The simulator state consists of sites in 3D euclidean space. Each site
# is occupied by some species, and these species have internal (local)
# degrees of freedom.

# The simulator is able to handle 1 and 2 site interactions. It would be
# reasonable to call the 1 site interactions internal interactions.

# For a particular species with internal state set S, the one site
# interactions are specified by a partially defined function r : S x S
# -> positive real numbers. If (s1;s2) is in the domain, then r(s1;s2)
# is the rate of the internal state transitioning from s1 to s2.

# For two species with internal state sets S,T, the two site
# interactions are specified by a partially defined function r : S x T x
# S x T -> positive real numbers. If (s1,t1;s2,t2) is in the domain,
# then r(s1,t1;s2,t2) is the rate of the state transitioning
# from (s1,t1) to (s2,t2).

# The propensity of a two site interaction is modulated by the distance
# between the two sites with a cutoff above some distance
# threshold. Sites which are too far apart cannot interact with one
# another.

# For the purpose of keeping the simulator as simple as possible, we
# want to pass as little information as possible across the python/C++
# barrier. Everything will be indexed consecutively, starting at 0, so
# our tables look as follows:

# species
# species_id|degrees_of_freedom

# sites
# site_id|x|y|z|species_id

# for two site interactions, we include both directions in the database
# if it is a one site interaction, species_id_2, left_state_2 and right_state_2 are -1.
# interactions
# interaction_id|number_of_sites|
# species_id_1|species_id_2|
# left_state_1|left_state_2|
# right_state_1|right_state_2|rate

# metadata
# number_of_species|number_of_sites|
# number_of_interactions|interaction_radius_bound


# test model
# two species: red and black
# black species internal state = { empty, unexcited, excited }
# red species have two states = { nothing, occupied }

# one site interactions:
# black: (unexcited; excited) -> 1                              // heating
# red:   None


# two site interactions:
# black black: (excited, empty; empty, unexcited) -> 1          // black motion. requires energy
#              (empty, excited; unexcited, empty) -> 1
# black red:   (unexcited, occupied; excited, nothing) -> 1       // black absorbing red
#              (excited, nothing; unexcited, occupied) -> 1       // black emitting red
# red red:     (occupied, nothing; nothing, occupied) -> 1          // red motion. free
#              (nothing, occupied; occupied, nothing) -> 1
#              (occupied, occupied; nothing, nothing) -> 1          // radiation

# test model has sites at (i,j,k) where 0 <= i,j,k < 10.
# if i + j + k is even, species is black, otherwise species is red.
# 1000 sites in total.

# initial state of a simulation is a mapping from sites to internal degrees of freedom
# at the given site.

# initial_state
# site_id|internal_state_id

# we record trajectories as follows. If site_id_2 = -1, then it is a one site interaction
# otherwise, it is a two site interaction.
# trajectories
# seed|step|time|site_id_1|site_id_2|interaction_id

import sqlite3
import os

species = {
    'black': {
        'species_id': 0,
        'index_to_state': [ 'empty', 'unexcited', 'excited' ],
        'state_to_index': {
            'empty': 0,
            'unexcited': 1,
            'excited': 2
        }
    },

    'red': {
        'species_id': 1,
        'index_to_state': [ 'nothing', 'occupied' ],
        'state_to_index': {
            'nothing': 0,
            'occupied': 1
        }
    }
}

sites = {}


index = 0
for i in range(10):
    for j in range(10):
        for k in range(10):
            if (i + j + k) % 2 == 0:
                site_species = 'black'
            else:
                site_species = 'red'

            site = {
                'site_id': index,
                'x': float(i),
                'y': float(j),
                'z': float(k),
                'species': site_species
            }

            sites[index] = site

            index += 1


one_site_interactions = {
    'black': [
        {
            'interaction_id': 0,
            'number_of_sites': 1,
            'left_state': 'unexcited',
            'right_state': 'excited',
            'rate': 1.0
        }
    ],

    'red': []
}

two_site_interactions = {
    ('black', 'black') : [
        {
            'interaction_id': 1,
            'number_of_sites': 2,
            'left_state_1'   : 'excited',
            'left_state_2'   : 'empty',
            'right_state_1'  : 'empty',
            'right_state_2'  : 'unexcited',
            'rate'           : 1.0
        },

        {
            'interaction_id': 2,
            'number_of_sites': 2,
            'left_state_1'   : 'empty',
            'left_state_2'   : 'excited',
            'right_state_1'  : 'unexcited',
            'right_state_2'  : 'empty',
            'rate'           : 1.0
        },

    ],

    ('black', 'red') : [
        {
            'interaction_id': 3,
            'number_of_sites': 2,
            'left_state_1'   : 'unexcited',
            'left_state_2'   : 'occupied',
            'right_state_1'  : 'excited',
            'right_state_2'  : 'nothing',
            'rate'           : 1.0
        },
        {
            'interaction_id': 4,
            'number_of_sites': 2,
            'left_state_1'   : 'excited',
            'left_state_2'   : 'nothing',
            'right_state_1'  : 'unexcited',
            'right_state_2'  : 'occupied',
            'rate'           : 1.0
        },
    ],

    ('red', 'black') : [
        {
            'interaction_id': 5,
            'number_of_sites': 2,
            'left_state_1'   : 'occupied',
            'left_state_2'   : 'unexcited',
            'right_state_1'  : 'nothing',
            'right_state_2'  : 'excited',
            'rate'           : 1.0
        },
        {
            'interaction_id': 6,
            'number_of_sites': 2,
            'left_state_1'   : 'nothing',
            'left_state_2'   : 'excited',
            'right_state_1'  : 'occupied',
            'right_state_2'  : 'unexcited',
            'rate'           : 1.0
        },
    ],

    ('red', 'red') : [
        {
            'interaction_id': 7,
            'number_of_sites': 2,
            'left_state_1'   : 'occupied',
            'left_state_2'   : 'nothing',
            'right_state_1'  : 'nothing',
            'right_state_2'  : 'occupied',
            'rate'           : 1.0
        },
        {
            'interaction_id': 8,
            'number_of_sites': 2,
            'left_state_1'   : 'nothing',
            'left_state_2'   : 'occupied',
            'right_state_1'  : 'occupied',
            'right_state_2'  : 'nothing',
            'rate'           : 1.0
        },
        {
            'interaction_id': 9,
            'number_of_sites': 2,
            'left_state_1'   : 'occupied',
            'left_state_2'   : 'occupied',
            'right_state_1'  : 'nothing',
            'right_state_2'  : 'nothing',
            'rate'           : 1.0
        }
    ]
}


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
        rate                REAL NOT NULL
    );
"""


insert_interaction_sql = """
    INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?);
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


os.system('rm -rf ./scratch; mkdir scratch')


def setup_nanoparticle_database():
    con = sqlite3.connect('./scratch/np.sqlite')
    cur = con.cursor()

    # create tables
    cur.execute(create_species_table_sql)
    cur.execute(create_sites_table_sql)
    cur.execute(create_interactions_table_sql)
    cur.execute(create_metadata_table_sql)

    # insert species
    for s in species:
        cur.execute(insert_species_sql,
                    ( species[s]['species_id'],
                      len(species[s]['index_to_state'])))

    con.commit()

    # insert sites
    for site_id in sites:
        site_data = sites[site_id]
        cur.execute(insert_site_sql,
                    ( site_data['site_id'],
                      site_data['x'],
                      site_data['y'],
                      site_data['z'],
                      species[site_data['species']]['species_id']))

    con.commit()

    number_of_interactions = 0
    # inserting single site interactions
    for s in one_site_interactions:
        for interaction_data in one_site_interactions[s]:
            number_of_interactions += 1
            cur.execute(insert_interaction_sql,
                        ( interaction_data['interaction_id'],
                          interaction_data['number_of_sites'],
                          species[s]['species_id'],
                          -1,
                          species[s]['state_to_index'][
                              interaction_data['left_state']],
                          -1,
                          species[s]['state_to_index'][
                              interaction_data['right_state']],
                          -1,
                          interaction_data['rate']))

    con.commit()

    # inserting two site interactions
    for (s1,s2) in two_site_interactions:
        for interaction_data in two_site_interactions[(s1,s2)]:
            number_of_interactions += 1
            cur.execute(insert_interaction_sql,
                        ( interaction_data['interaction_id'],
                          interaction_data['number_of_sites'],
                          species[s1]['species_id'],
                          species[s2]['species_id'],
                          species[s1]['state_to_index'][
                              interaction_data['left_state_1']
                          ],
                          species[s2]['state_to_index'][
                              interaction_data['left_state_2']
                          ],
                          species[s1]['state_to_index'][
                              interaction_data['right_state_1']
                          ],
                          species[s2]['state_to_index'][
                              interaction_data['right_state_2']
                          ],
                          interaction_data['rate']))

    con.commit()

    # insert metadata
    cur.execute(insert_metadata_sql,
                ( len(species),
                  len(sites),
                  number_of_interactions))

    con.commit()


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
        interaction_radius_bound         REAL NOT NULL
);
"""


insert_factors_sql = """
    INSERT INTO factors VALUES (?,?,?);
"""


def setup_initial_state_database():
    con = sqlite3.connect('./scratch/initial_state.sqlite')
    cur = con.cursor()


    cur.execute(create_initial_state_table_sql)
    cur.execute(create_trajectories_table_sql)
    cur.execute(create_factors_table_sql)
    cur.execute(insert_factors_sql, (1.0,1.0,3.0))
    con.commit()


    for index in sites:
        site_data = sites[index]
        site_id = site_data['site_id']
        x = site_data['x']
        y = site_data['y']
        z = site_data['z']
        s = site_data['species']
        if s == 'black':
            if ((x + y + z < 5.0) or (x + y + z >= 25.0)):
                cur.execute(
                    insert_initial_state_sql,
                    (site_id, species[s]['state_to_index']['unexcited']))
            else:
                 cur.execute(
                    insert_initial_state_sql,
                    (site_id, species[s]['state_to_index']['empty']))

        if s == 'red':
            cur.execute(
                insert_initial_state_sql,
                (site_id, species[s]['state_to_index']['nothing']))

        con.commit()



setup_initial_state_database()
setup_nanoparticle_database()
