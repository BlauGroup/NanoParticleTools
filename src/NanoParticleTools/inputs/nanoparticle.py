from typing import Optional, Sequence, Union, Tuple

import numpy as np
from pymatgen.core import Composition, Structure, Site, Lattice, Molecule
from monty.json import MSONable
from NanoParticleTools.species_data.species import Dopant

class NanoParticleConstraint(MSONable):
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
        Returns the dimensions of the box that would encapsulate the constrained volume.

        Useful in determining the size of supercell required.
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
        """

        :param radius: Radius of this constraint volume
        :param host_structure: structure of the host
        """
        self.radius = radius
        super().__init__(host_structure)

    def bounding_box(self):
        """
        Returns the bounding box that encloses this volume.
        :return:
        """
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

class DopedNanoparticle(MSONable):
    def __init__(self,
                 constraints: Sequence[NanoParticleConstraint],
                 dopant_specification: Sequence[Tuple],
                 seed: Optional[int] = 0):
        self.constraints = constraints
        self.seed = seed
        self.dopant_specification = dopant_specification

        self._sites = None
        # Move to dopant area
        self.dopant_indices = [[] for _ in self.constraints]
        self._dopant_concentration = [{} for _ in self.constraints]

    @property
    def has_structure(self):
        if hasattr(self, '_sites'):
            return self._sites is not None
        else:
            return False

    def as_dict(self):
        if self.has_structure:
            # Delete generated data if it exists
            del self._sites, self.dopant_indices, self._dopant_concentration
        return super().as_dict()

    def generate(self):

        # Construct nanoparticle
        nanoparticle_sites = []
        for i, constraint in enumerate(self.constraints):
            _struct = constraint.host_structure.copy()

            # Identify the minimum scaling matrix required to fit the bounding box
            # TODO: verify that this works on all lattice types
            perp_vec = np.cross(_struct.lattice.matrix[1], _struct.lattice.matrix[2])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            a_distance = np.dot(_struct.lattice.matrix[0], perp_vec)

            perp_vec = np.cross(_struct.lattice.matrix[0], _struct.lattice.matrix[2])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            b_distance = np.dot(_struct.lattice.matrix[1], perp_vec)

            perp_vec = np.cross(_struct.lattice.matrix[0], _struct.lattice.matrix[1])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            c_distance = np.dot(_struct.lattice.matrix[2], perp_vec)

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
            for other_constraint in self.constraints[:i]:
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
        self._sites = nanoparticle_sites
        self.dopant_indices = [[] for _ in self.constraints]
        self._dopant_concentration = [{} for _ in self.constraints]
        self._apply_dopants()

    def _apply_dopants(self):
        rng = np.random.default_rng(self.seed)
        for spec in self.dopant_specification:
            self._apply_dopant( *spec, rng=rng)

    def _apply_dopant(self,
                   constraint_index: int,
                   dopant_concentration: float,
                   dopant_species: str,
                   replaced_species: str,
                   rng: np.random.default_rng):
        if dopant_species in Dopant.SURFACE_DOPANT_NAMES_TO_SYMBOLS:
            dopant_species = Dopant.SURFACE_DOPANT_NAMES_TO_SYMBOLS[dopant_species]

        sites_in_constraint = self._sites[constraint_index]

        # Identify the possible sites for the dopant
        possible_dopant_sites = [i for i, site in enumerate(sites_in_constraint) if
                                 site.specie.symbol == replaced_species]

        # Number of sites corresponding to the species being replaced or that have previously been replaced
        # TODO: This probably only works if only one species is being substituted.
        n_host_sites = len(possible_dopant_sites) + len(self.dopant_indices[constraint_index])
        n_dopants = np.round(n_host_sites * dopant_concentration)

        # Randomly pick sites to place dopants
        dopant_sites = rng.choice(possible_dopant_sites, int(n_dopants), replace=False)
        for i in dopant_sites:
            self._sites[constraint_index][i].species = {dopant_species: 1}

        # Keep track of concentrations in each shell
        self._dopant_concentration[constraint_index][dopant_species] = len(dopant_sites) / n_host_sites

        # Keep track of sites with dopants
        self.dopant_indices[constraint_index].extend(dopant_sites)

    def to_file(self, fmt="xyz", name="nanoparticle.xyz"):
        if self.has_structure == False:
            raise RuntimeError('Nanoparticle not generated. Please use the generate() function')
        _np = Molecule.from_sites(self.sites)
        xyz = _np.to(fmt, name)

    def dopants_to_file(self, fmt="xyz", name="dopant_nanoparticle.xyz"):
        if self.has_structure == False:
            raise RuntimeError('Nanoparticle not generated. Please use the generate() function')
        _np = Molecule.from_sites(self.dopant_sites)
        xyz = _np.to(fmt, name)

    @property
    def sites(self):
        if self.has_structure == False:
            raise RuntimeError('Nanoparticle not generated. Please use the generate() function')
        return [_site for sites in self._sites for _site in sites]

    @property
    def dopant_sites(self) -> Sequence[Site]:
        if self.has_structure == False:
            raise RuntimeError('Nanoparticle not generated. Please use the generate() function')
        _sites = []
        for dopant_indices, sites in zip(self.dopant_indices, self._sites):
            for i in dopant_indices:
                _sites.append(sites[i])
        return _sites

    @property
    def dopant_concentrations(self,
                              constraint_index: Optional[Union[int, None]] = None,
                              replaced_species: Optional[str] = 'Y'):
        if self.has_structure == False:
            raise RuntimeError('Nanoparticle not generated. Please use the generate() function')
        if constraint_index is None:
            num_replaced_sites = len([i for i, site in enumerate(self.sites) if site.specie.symbol == replaced_species])
            total_num_sites = len(self.dopant_sites) + num_replaced_sites

            dopant_amount = {}
            for dopant in self.dopant_sites:
                try:
                    dopant_amount[str(dopant.specie)] += 1
                except:
                    dopant_amount[str(dopant.specie)] = 1

            return dict([(key, item / total_num_sites) for key, item in dopant_amount.items()])

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
