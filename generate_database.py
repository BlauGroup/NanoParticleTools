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
# species_id|local_degrees_of_freedom_bound

# sites
# site_id|x|y|z|species_id

# one_site_interactions
# species_id|left_state|right_state|rate


# for two site interactions, we include both directions in the database
# two_site_interactions
# species_id_1|species_id_2|left_state_1|left_state_2|right_state_1|right_state_2|rate

# metadata
# number_of_species|number_of_sites|single_site_interaction_factor|double_site_interaction_factor|spatial_decay_method


# test model
# two species: red and black
# black species internal state = { empty, unexcited, excited }
# red species have two states = { empty, occupied }

# one site interactions:
# black: (unexcited; excited) -> 1                              // heating
# red:   None


# two site interactions:
# black black: (excited, empty; empty, unexcited) -> 1          // black motion. requires energy
#              (empty, excited; unexcited, empty) -> 1
# black red:   (unexcited, occupied; excited, empty) -> 1       // black absorbing red
#              (excited, empty; unexcited, occupied) -> 1       // black emitting red
# red red:     (occupied, empty; empty, occupied) -> 1          // red motion. free
#              (empty, occupied; occupied, empty) -> 1
#              (occupied, occupied; empty, empty) -> 1          // radiation
