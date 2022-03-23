from typing import Optional, Sequence

import numpy as np
from pymatgen.core import Composition, Structure, Site, Lattice


class DopedNanoparticle():
    def __init__(self,
                 sites:Sequence[Site],
                 constraints:Sequence[NanoParticleConstraint],
                 seed: Optional[int] = 0):
        self._sites = sites
        self.constraints = constraints
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.dopant_indices = [[] for _ in self.constraints]

    @property
    def sites(self):
        return [_site for sites in self._sites for _site in sites]

    def add_dopant(self,
                   constraint_index: int,
                   dopant_species: str,
                   replaced_species: str,
                   dopant_concentration: float):
        sites_in_constraint = self._sites[constraint_index]

        n_sites_in_constraint = len(sites_in_constraint)
        n_dopants = np.round(n_sites_in_constraint * dopant_concentration)

        # Identify the possible sites for the dopant
        possible_dopant_sites = [i for i, site in enumerate(sites_in_constraint) if
                                 site.specie.symbol == replaced_species]

        # Randomly pick sites to place dopants
        dopant_sites = self.rng.choice(possible_dopant_sites, int(n_dopants), replace=False)
        for i in dopant_sites:
            self._sites[constraint_index][i].species = Composition(dopant_species)

        # Keep track of sites with dopants
        self.dopant_indices[constraint_index].extend(dopant_sites)

    @property
    def dopant_sites(self):
        _sites = []
        for dopant_indices, sites in zip(self.dopant_indices, self._sites):
            for i in dopant_indices:
                _sites.append(sites[i])
        return _sites

    @property
    def dopant_concentration(self):
        pass

    @classmethod
    def from_constraints(cls, constraints: Sequence[NanoParticleConstraint]):
        """
        Construct a nanoparticle from a sequence of constraints.

        Constraints should be sorted by size.

        example: constraints = [SphericalConstraint(20), SphericalConstraint(30)]
            defines a spherical nanoparticle with a radius of 30angstroms. The nanoparticle is composed of a core (20nm radius) and a shell (10nm thick)

        :param constraints:
        :return:
        """
        nanoparticle_sites = []
        for i, constraint in enumerate(constraints):
            _struct = constraint.host_structure.copy()

            # Identify the minimum scaling matrix required to fit the bounding box
            # TODO: verify that this works on all lattice types
            perp_vec = np.cross(_struct.host_structure.lattice.matrix[1], _struct.host_structure.lattice.matrix[2])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            a_distance = np.dot(_struct.host_structure.lattice.matrix[0], perp_vec)

            perp_vec = np.cross(_struct.host_structure.lattice.matrix[0], _struct.host_structure.lattice.matrix[2])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            b_distance = np.dot(_struct.host_structure.lattice.matrix[1], perp_vec)

            perp_vec = np.cross(_struct.host_structure.lattice.matrix[0], _struct.host_structure.lattice.matrix[1])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            c_distance = np.dot(_struct.host_structure.lattice.matrix[2], perp_vec)

            scaling_matrix = 2 * np.ceil(
                np.abs(np.divide(constraint.bounding_box(), [a_distance, b_distance, c_distance])))

            # Make the supercell
            _struct.make_supercell(scaling_matrix)

            # Translate sites so that the center is coincident with the origin
            center = np.divide(np.sum(_struct.lattice.matrix, axis=0), 2)
            translated_coords = np.subtract(_struct.cart_coords, center)

            # np.array(bool) indicating whether each site is within the Outer bounds of the constraint
            sites_in_bounds = constraint.sites_in_bounds(translated_coords)

            # Check if each site falls into the bounds of a smaller (previous) constraint
            in_another_constraint = np.full(sites_in_bounds.shape, False)
            for other_constraint in constraints[:i]:
                in_another_constraint = in_another_constraint | other_constraint.sites_in_bounds(translated_coords)

            # Merge two boolean lists and identify sites within only the current constraint
            sites_in_constraint = sites_in_bounds & np.invert(in_another_constraint)
            sites_index_in_bounds = np.where(sites_in_constraint)[0]

            # Make Site object for each site that lies in bounds. Note: this is not a PeriodicSite
            _sites = []
            for site_index in sites_index_in_bounds:
                _site = _struct[site_index]
                _sites.append(Site(_site.specie, translated_coords[site_index]))

            nanoparticle_sites.append(_sites)

        return cls(nanoparticle_sites, constraints)


def get_nayf4_structure():
    lattice = Lattice.hexagonal(a=6.067, c=7.103)
    species = ['Na', 'Na', 'Na', 'Y', 'Y', 'Y', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F']
    positions = [[0.3333, 0.6667, 0.5381], [0.3333, 0.6667, 0.9619], [0.6667, 0.3333, 0.75], [0, 0, 0.9969],
                 [0, 0, 0.5031], [0.6667, 0.3333, 0.25], [0.0272, 0.2727, 0.2500], [0.0572, 0.2827, 0.7500],
                 [0.2254, 0.9428, 0.7500], [0.2455, 0.9728, 0.2500], [0.4065, 0.3422, 0.0144],
                 [0.4065, 0.3422, 0.4856], [0.6578, 0.0643, 0.0144], [0.6578, 0.0643, 0.4856],
                 [0.7173, 0.7746, 0.7500], [0.7273, 0.7545, 0.2500], [0.9357, 0.5935, 0.0144],
                 [0.9357, 0.5935, 0.4856]]

    return Structure(lattice, species=species, coords=positions)


def get_wse2_structure():
    lattice = Lattice.hexagonal(a=3.327, c=15.069)
    species = ['Se', 'Se', 'Se', 'Se', 'W', 'W']
    positions = [[0.3333, 0.6667, 0.6384], [0.3333, 0.6667, 0.8616], [0.6667, 0.3333, 0.1384],
                 [0.6667, 0.3333, 0.3616], [0.3333, 0.6667, 0.25], [0.6667, 0.3333, 0.75]]

    return Structure(lattice, species=species, coords=positions)


class NanoParticleConstraint():
    """
    Template for a Nanoparticle constraint. This defines the shape of a volume containing atoms

    A constraint must implement the bounding_box and sites_in_bounds functions.
    """
    def __init__(self, host_structure: Optional[Structure] = None):
        if host_structure is None:
            self.host_structure = get_nayf4_structure()
        else:
            self.host_structure = host_structure

    def bounding_box(self):
        """
        Returns the dimensions of the box that would encapsulate the constrained volume
        :return:
        """
        pass

    def sites_in_bounds(self, site_coords, center):
        """
        Checks a list of coordinates to check if they are within the bounds of the constraint

        :param site_coords: coordinates of all possible sites
        :param center: center of box, so that sites may be translated
        :return:
        """
        pass


class SphericalConstraint(NanoParticleConstraint):
    """
    Defines a spherical constraint that would be used to construct a spherical core/nanoparticle or a spherical shell
    """
    def __init__(self, radius, host_structure: Optional[Structure] = None):
        self.radius = radius
        super().__init__(host_structure)

    def bounding_box(self):
        return [self.radius, self.radius, self.radius]

    def sites_in_bounds(self, site_coords, center=[0, 0, 0]):
        distances_from_center = np.linalg.norm(np.subtract(site_coords, center), axis=1)
        return distances_from_center <= self.radius


class PrismConstraint(NanoParticleConstraint):
    def __init__(self, a, b, c, host_structure: Optional[Structure] = None):
        self.a = a
        self.b = b
        self.c = c
        super().__init__(host_structure)

    def bounding_box(self):
        return [self.a, self.b, self.c]

    def sites_in_bounds(self, site_coords, center=[0, 0, 0]):
        centered_coords = np.subtract(site_coords, center)
        abs_coords = np.abs(centered_coords)
        within_a = abs_coords[:, 0] < self.a / 2
        within_b = abs_coords[:, 1] < self.b / 2
        within_c = abs_coords[:, 2] < self.c / 2

        return within_a & within_b & within_c


class CubeConstraint(PrismConstraint):
    """
    Defines a cubic constraint that would be used to construct a cubic core/nanoparticle or a cubic shell
    """
    def __init__(self, a, host_structure: Optional[Structure] = None):
        super().__init__(a, a, a, host_structure)
