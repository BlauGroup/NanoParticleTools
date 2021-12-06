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

# one_site_interactions
# species_id|left_state|right_state|rate


# for two site interactions, we include both directions in the database
# two_site_interactions
# species_id_1|species_id_2|
# left_state_1|left_state_2|
# right_state_1|right_state_2|rate

# metadata
# number_of_species|number_of_sites|
# number_of_one_site_interactions|number_of_two_site_interactions|
# single_site_interaction_factor|double_site_interaction_factor|
# spatial_decay_radius


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

import sqlite3
import os

species = {
    'black' : {
        'species_id' : 0,
        'index_to_state' : [ 'empty', 'unexcited', 'excited' ],
        'state_to_index' : {
            'empty' : 0,
            'unexcited' : 1,
            'excited' : 2
        }
    },

    'red' : {
        'species_id' : 1,
        'index_to_state' : [ 'nothing', 'occupied' ],
        'state_to_index' : {
            'nothing' : 0,
            'occupied' : 1
        }
    }
}

sites = {
    'index_to_site' : {},
    'site_to_index' : {}
}


index = 0
for i in range(10):
    for j in range(10):
        for k in range(10):
            if (i + j + k) % 2 == 0:
                site_species = 'black'
            else:
                site_species = 'red'


            sites['index_to_site'][index] = (
                float(i),
                float(j),
                float(k),
                site_species)

            sites['site_to_index'][(i,j,k)] = index

            index += 1


one_site_interactions = {
    'black' : {
        ('unexcited', 'excited') : 1.0
    },

    'red' : {}
}

two_site_interactions = {
    ('black', 'black') : {
        ('excited','empty',
         'empty','unexcited') : 1.0,

        ('empty', 'excited',
         'unexcited', 'empty') : 1.0,
    },

    ('black', 'red') : {
        ('unexcited', 'occupied',
         'excited', 'nothing') : 1.0,

        ('excited', 'nothing',
         'unexcited', 'occupied') : 1.0,
    },

    ('red', 'black') : {
        ('occupied', 'unexcited',
         'nothing', 'excited') : 1.0,

        ('nothing', 'excited',
         'occupied', 'unexcited') : 1.0
    },

    ('red', 'red') : {
        ('occupied', 'nothing',
         'nothing', 'occupied') : 1.0,

        ('nothing', 'occupied',
         'occupied', 'nothing') : 1.0,

        ('occupied', 'occupied',
         'nothing', 'nothing') : 1.0
    },
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


create_one_site_interactions_table_sql = """
    CREATE TABLE one_site_interactions (
        species_id          INTEGER NOT NULL PRIMARY KEY,
        left_state          INTEGER NOT NULL,
        right_state         INTEGER NOT NULL,
        rate                REAL NOT NULL
    );
"""

insert_one_site_interaction_sql = """
    INSERT INTO one_site_interactions VALUES (?,?,?,?);
"""

create_two_site_interactions_table_sql = """
    CREATE TABLE two_site_interactions (
        species_id_1        INTEGER NOT NULL,
        species_id_2        INTEGER NOT NULL,
        left_state_1        INTEGER NOT NULL,
        left_state_2        INTEGER NOT NULL,
        right_state_1       INTEGER NOT NULL,
        right_state_2       INTEGER NOT NULL,
        rate                REAL NOT NULL
    );
"""

insert_two_site_interaction_sql = """
    INSERT INTO two_site_interactions VALUES (?,?,?,?,?,?,?);
"""

create_metadata_table_sql = """
    CREATE TABLE metadata (
        number_of_species                   INTEGER NOT NULL,
        number_of_sites                     INTEGER NOT NULL,
        number_of_one_site_interactions     INTEGER NOT NULL,
        number_of_two_site_interactions     INTEGER NOT NULL,
        single_site_interaction_factor      REAL NOT NULL,
        double_site_interaction_factor      REAL NOT NULL,
        spatial_decay_radius                REAL NOT NULL
    );
"""

insert_metadata_sql = """
    INSERT INTO metadata VALUES (?,?,?,?,?,?,?);
"""

os.system('rm -rf ./scratch; mkdir scratch')
con = sqlite3.connect('./scratch/test_nanoparticle.sqlite')
cur = con.cursor()

# create tables
cur.execute(create_species_table_sql)
cur.execute(create_sites_table_sql)
cur.execute(create_one_site_interactions_table_sql)
cur.execute(create_two_site_interactions_table_sql)
cur.execute(create_metadata_table_sql)

# insert species
for s in species:
    cur.execute(insert_species_sql,
                ( species[s]['species_id'],
                  len(species[s]['index_to_state'])))

con.commit()

# insert sites
for site_id in sites['index_to_site']:
    site_data = sites['index_to_site'][site_id]
    cur.execute(insert_site_sql,
                ( site_id,
                  site_data[0],
                  site_data[1],
                  site_data[2],
                  species[site_data[3]]['species_id']))

con.commit()

number_of_one_site_interactions = 0
# inserting single site interactions
for s in one_site_interactions:
    for (l,r) in one_site_interactions[s]:
        number_of_one_site_interactions += 1
        rate = one_site_interactions[s][(l,r)]
        cur.execute(insert_one_site_interaction_sql,
                    ( species[s]['species_id'],
                      species[s]['state_to_index'][l],
                      species[s]['state_to_index'][r],
                      rate))

con.commit()

number_of_two_site_interactions = 0
# inserting two site interactions
for (s1,s2) in two_site_interactions:
    for (l1,l2,r1,r2) in two_site_interactions[(s1,s2)]:
        number_of_two_site_interactions += 1
        rate = two_site_interactions[(s1,s2)][
            (l1,l2,r1,r2)]
        cur.execute(insert_two_site_interaction_sql,
                    ( species[s1]['species_id'],
                      species[s2]['species_id'],
                      species[s1]['state_to_index'][l1],
                      species[s2]['state_to_index'][l2],
                      species[s1]['state_to_index'][r1],
                      species[s2]['state_to_index'][r2],
                      rate))

con.commit()

# insert metadata
cur.execute(insert_metadata_sql,
            ( len(species),
              len(sites['index_to_site']),
              number_of_one_site_interactions,
              number_of_two_site_interactions,
              1.0,
              1.0,
              4.0))

con.commit()
