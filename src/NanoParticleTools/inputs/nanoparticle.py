from typing import (Optional, Sequence, Union, Tuple, List, Dict)
from abc import ABC, abstractmethod

import numpy as np
from pymatgen.core import (Structure, Site, Lattice, Molecule)
from monty.json import MSONable
from NanoParticleTools.species_data.species import Dopant
try:
    from e3nn.io import SphericalTensor
    from torch import Tensor
    missing_ml_package = True
except (ImportError, ModuleNotFoundError):
    SphericalTensor = object
    Tensor = object
    missing_ml_package = False


class NanoParticleConstraint(ABC, MSONable):
    """
    Template for a Nanoparticle constraint. This defines the shape of a control
    volume containing atoms.

    A constraint must implement the bounding_box and sites_in_bounds functions.

    Args:
        host_structure (Structure, None): Structure of the host.
            Defaults to the structure of NaYF4.
    """

    def __init__(self, host_structure: Optional[Structure] = None):
        if host_structure is None:
            self.host_structure = get_nayf4_structure()
        else:
            self.host_structure = host_structure

    @abstractmethod
    def bounding_box(self) -> List[float]:
        """
        Returns the dimensions of the box that would encapsulate the
        constrained volume.

        Useful in determining the size of supercell required.

        TODO: Need to check if this box definition and accomanying function
            in the lattice generation works correctly for cubic constraints
            and constraints not centered at the origin.
        Returns:
            List[float]: The box dimensions in the form [x_max, y_max, z_max]
        """
        raise NotImplementedError

    @abstractmethod
    def sites_in_bounds(self, site_coords: Union[np.array, List],
                        center: List) -> np.array:
        """
        Checks a list of coordinates to check if they are within the bounds
        of the constraint. This is how we check if a site is part of the
        nanoparticle.

        Args:
            site_coords (np.array, List): coordinates of all possible sites
            center (np.array, List): center of box, so that sites may
                be translated

        Returns:
            np.array: Array of length n, where the i-th entry indicates
                whether or not the i-th site_coord provided is within the
                bounds of the constraint.
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the constraint
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()


class SphericalConstraint(NanoParticleConstraint):
    """
    Defines a spherical constraint that would be used to construct a spherical
    nanoparticle or a spherical core/shell

    Args:
        radius (float, int): Radius of this constraint volume
        host_structure (Structure, None): Structure of the host
    """

    def __init__(self,
                 radius: Union[float, int],
                 host_structure: Optional[Structure] = None):
        self.radius = radius
        super().__init__(host_structure)

    def bounding_box(self) -> List[float]:
        """
        Returns the dimensions of the box that would encapsulate the
        constrained volume.

        Useful in determining the size of supercell required.

        Returns:
            List[float]: The box dimensions in the form [x_max, y_max, z_max]
        """
        return [self.radius, self.radius, self.radius]

    def sites_in_bounds(self,
                        site_coords: Union[np.array, List],
                        center: List[float] = None) -> np.array:
        """
        Checks a list of coordinates to check if they are within the bounds
        of the constraint. This is how we check if a site is part of the
        nanoparticle.

        Args:
            site_coords (np.array, List): coordinates of all possible sites
            center (np.array, List): center of box, so that sites may
                be translated

        Returns:
            np.array: Array of length n, where the i-th entry indicates
                whether or not the i-th site_coord provided is within the
                bounds of the constraint.
        """
        if center is None:
            center = [0, 0, 0]

        distances_from_center = np.linalg.norm(np.subtract(
            site_coords, center),
                                               axis=1)
        return distances_from_center <= self.radius

    def __str__(self) -> str:
        return f"SphericalConstraint(radius={self.radius})"


class PrismConstraint(NanoParticleConstraint):
    """
    Defines a Prism constraint that would be used to construct a rectangular
    prism nanoparticle or core/shell

    Args:
        a (float, int): Radius of this constraint volume
        b (float, int): Radius of this constraint volume
        c (float, int): Radius of this constraint volume
        host_structure (Structure, None): Structure of the host
    """

    def __init__(self,
                 a: Union[float, int],
                 b: Union[float, int],
                 c: Union[float, int],
                 host_structure: Optional[Structure] = None):
        self.box_a = a
        self.box_b = b
        self.box_c = c
        super().__init__(host_structure)

    def bounding_box(self):
        """
        Returns the dimensions of the box that would encapsulate the
        constrained volume.

        Useful in determining the size of supercell required.

        Returns:
            List[float]: The box dimensions in the form [x_max, y_max, z_max]
        """
        return [self.box_a, self.box_b, self.box_c]

    def sites_in_bounds(self,
                        site_coords: Union[np.array, List],
                        center: List[float] = None) -> np.array:
        """
        Checks a list of coordinates to check if they are within the bounds
        of the constraint. This is how we check if a site is part of the
        nanoparticle.

        Args:
            site_coords (np.array, List): coordinates of all possible sites
            center (np.array, List): center of box, so that sites may
                be translated

        Returns:
            np.array: Array of length n, where the i-th entry indicates
                whether or not the i-th site_coord provided is within the
                bounds of the constraint.
        """
        if center is None:
            center = [0, 0, 0]

        centered_coords = np.subtract(site_coords, center)
        abs_coords = np.abs(centered_coords)
        within_a = abs_coords[:, 0] <= self.box_a / 2
        within_b = abs_coords[:, 1] <= self.box_b / 2
        within_c = abs_coords[:, 2] <= self.box_c / 2

        return within_a & within_b & within_c

    def __str__(self) -> str:
        return (f"PrismConstraint(a={self.box_a}, "
                f"b={self.box_b}, c={self.box_c})")


class CubeConstraint(PrismConstraint):
    """
    Defines a cubic constraint that would be used to define a cubic volume.
    This could be used to define a cubic nanoparticle or a cubic shell.
    """

    def __init__(self, a, host_structure: Optional[Structure] = None):
        super().__init__(a, a, a, host_structure)

    def __str__(self) -> str:
        return f"CubeConstraint(a={self.box_a})"


class SphericalHarmonicsConstraint(NanoParticleConstraint):
    """
    A constraint that defines a volume in 3D space according to a deformed
    spherical harmonic representation.
    """

    def __init__(self,
                 sh_bounds: Tensor,
                 irreps: SphericalTensor,
                 host_structure: Optional[Structure] = None):
        """
        For a spherical shell, trivially pass in a spherical harmonic with l=0
        Args:
            sh_bounds (torch.Tensor): _description_
            irreps (io.SphericalTensor): _description_
            host_structure (Optional[Structure], optional): _description_.
                Defaults to None.
        """
        if missing_ml_package:
            raise ImportError(
                "e3nn or torch is not installed. Please make sure both "
                "are installed to use SphericalHarmonicsConstraint")
        self.sh_bounds = sh_bounds
        self.irreps = irreps
        super().__init__(host_structure)

    def bounding_box(self):
        """
        Returns the dimensions of the box that would encapsulate the
        constrained volume.

        Useful in determining the size of supercell required.

        Returns:
            List[float]: The box dimensions in the form [x_max, y_max, z_max]
        """
        # Will need to figure out how to get the largest distance in
        # the 3 cartesian coordinates
        raise NotImplementedError

    def sites_in_bounds(self,
                        site_coords: Union[np.array, List],
                        center: List[float] = None) -> np.array:
        """
        Checks a list of coordinates to check if they are within the bounds
        of the constraint. This is how we check if a site is part of the
        nanoparticle.

        Args:
            site_coords (np.array, List): coordinates of all possible sites
            center (np.array, List): center of box, so that sites may
                be translated

        Returns:
            np.array: Array of length n, where the i-th entry indicates
                whether or not the i-th site_coord provided is within the
                bounds of the constraint.
        """
        if center is None:
            center = [0, 0, 0]
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


class DopedNanoparticle(MSONable):
    """
    A class that defines a nanoparticle with dopants.

    Args:
        constraints (Sequence[NanoParticleConstraint]): A list of constraints
            that define the nanoparticle volume. The constraints should be
            ordered from the innermost to the outermost.

            Example:
                constraints = [SphericalConstraint(20), CubicConstraint(20),
                               SphericalConstraint(30)]
        dopant_specification (Sequence[Tuple]): A list of tuples
            that specify the constraint the dopant is to be added, their
            concentration, their symbol, and which sites they occupy.

            Example:
                dopant_specification =
                    [(0, 0.1, 'Yb', 'Y'), (1, 0.15, 'Nb', 'Na')]

                This would add 10% Yb to the first constraint and 15% Nb to
                the 2nd constraint. The Yb would occupy the Y sites and the
                Nb would occupy the Na sites.

        seed (Optional[int], None): The seed used to determine dopant
            placement on the lattice. Defaults to 0.
    """

    def __init__(self,
                 constraints: Sequence[NanoParticleConstraint],
                 dopant_specification: Sequence[Tuple],
                 seed: Optional[int] = 0,
                 prune_hosts: Optional[bool] = False):
        self.constraints = constraints
        self.seed = seed if seed is not None else 0
        self.dopant_specification = dopant_specification

        # Check to ensure that the dopant specifications
        # are valid ( 0 <= x <= 1)
        # Bin the dopant concentration
        conc_by_layer_and_species = [{} for _ in self.constraints]
        for i, dopant_conc, _, replace_el in dopant_specification:
            assert dopant_conc >= 0, (
                'Dopant concentration cannot be negative')
            assert dopant_conc <= 1, (
                'Dopant concentration cannot be greater than 1')
            try:
                conc_by_layer_and_species[i][replace_el] += dopant_conc
            except KeyError:
                try:
                    conc_by_layer_and_species[i][replace_el] = dopant_conc
                except KeyError:
                    conc_by_layer_and_species[i] = {replace_el: dopant_conc}

        # Check if all concentrations are valid
        for layer_i, layer in enumerate(conc_by_layer_and_species):
            for replaced_el, total_replaced_conc in layer.items():
                assert total_replaced_conc < 1, (
                    f"Dopant concentration in constraint {layer_i}"
                    f" on {replaced_el} sites exceeds 100%")

        if prune_hosts:
            for dopants_dict, constraint in zip(conc_by_layer_and_species,
                                                constraints):
                _sites = [
                    site for site in constraint.host_structure.sites
                    if site.species_string in dopants_dict.keys()
                ]
                constraint.host_structure = Structure.from_sites(_sites)

        self._sites = None
        # Move to dopant area
        self.dopant_indices = [[] for _ in self.constraints]
        self._dopant_concentration = [{} for _ in self.constraints]

    @property
    def has_structure(self) -> bool:
        """
        Checks whether the nanoparticle structure has been generated.

        Returns:
            bool: True if the nanoparticle structure has been generated.
        """
        if hasattr(self, '_sites'):
            return self._sites is not None

        return False

    def as_dict(self) -> Dict:
        """
        Returns a dictionary representation of the nanoparticle. If the
        structure was generated, delete the data that wasn't required by the
        class definition.

        Returns:
            dict: Dictionary representation of the nanoparticle.
        """
        if self.has_structure:
            # Delete generated data if it exists
            del self._sites, self.dopant_indices, self._dopant_concentration
        return super().as_dict()

    def generate(self):
        """
        Generates the nanoparticle structure.

        This method does not return anything, but instead sets the attributes
        of the nanoparticle corresponding to the structure.
        The attributes are:
            _sites: The sites of the full nanoparticle.
            dopant_indices: The indices of _sites that are occupied by dopants
        """
        # Construct nanoparticle
        nanoparticle_sites = []
        for i, constraint in enumerate(self.constraints):
            _struct = constraint.host_structure.copy()

            # Identify the minimum scaling matrix required to fit
            # the bounding box
            # TODO: verify that this works on all lattice types
            perp_vec = np.cross(_struct.lattice.matrix[1],
                                _struct.lattice.matrix[2])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            a_distance = np.dot(_struct.lattice.matrix[0], perp_vec)

            perp_vec = np.cross(_struct.lattice.matrix[0],
                                _struct.lattice.matrix[2])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            b_distance = np.dot(_struct.lattice.matrix[1], perp_vec)

            perp_vec = np.cross(_struct.lattice.matrix[0],
                                _struct.lattice.matrix[1])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            c_distance = np.dot(_struct.lattice.matrix[2], perp_vec)

            scaling_matrix = 2 * np.ceil(
                np.abs(
                    np.divide(constraint.bounding_box(),
                              [a_distance, b_distance, c_distance])))

            # Make the supercell
            _struct.make_supercell(scaling_matrix)

            # Translate sites so that the center is coincident with the origin
            center = np.divide(np.sum(_struct.lattice.matrix, axis=0), 2)
            translated_coords = np.subtract(_struct.cart_coords, center)

            # np.array(bool) indicating whether each site is within the
            # outer bounds of the constraint
            sites_in_bounds = constraint.sites_in_bounds(translated_coords)

            # Check if each site falls into the bounds of a
            # smaller (previous) constraint
            in_another_constraint = np.full(sites_in_bounds.shape, False)
            for other_constraint in self.constraints[:i]:
                in_another_constraint = (
                    in_another_constraint
                    | other_constraint.sites_in_bounds(translated_coords))

            # Merge two boolean lists and identify sites within only
            # the current constraint
            sites_in_constraint = sites_in_bounds & np.invert(
                in_another_constraint)
            sites_index_in_bounds = np.where(sites_in_constraint)[0]

            # Make Site object for each site that lies in bounds.
            # Note: this is not a PeriodicSite
            _sites = []
            for site_index in sites_index_in_bounds:
                _site = _struct[site_index]
                _sites.append(Site(_site.specie,
                                   translated_coords[site_index]))

            nanoparticle_sites.append(_sites)
        self._sites = nanoparticle_sites
        self.dopant_indices = [[] for _ in self.constraints]
        self._dopant_concentration = [{} for _ in self.constraints]
        self._apply_dopants()

    def _apply_dopants(self):
        """
        A helper function to apply dopants to the nanoparticle structure.
        """
        rng = np.random.default_rng(self.seed)
        for spec in self.dopant_specification:
            self._apply_dopant(*spec, rng=rng)

    def _apply_dopant(self, constraint_index: int, dopant_concentration: float,
                      dopant_species: str, replaced_species: str,
                      rng: np.random.default_rng):
        """
        A helper function to apply a single dopant to the nanoparticle
        structure.
        """
        if dopant_species in Dopant.SURFACE_DOPANT_NAMES_TO_SYMBOLS:
            dopant_species = Dopant.SURFACE_DOPANT_NAMES_TO_SYMBOLS[
                dopant_species]

        sites_in_constraint = self._sites[constraint_index]

        # Identify the possible sites for the dopant
        possible_dopant_sites = [
            i for i, site in enumerate(sites_in_constraint)
            if site.specie.symbol == replaced_species
        ]

        # Number of sites corresponding to the species being replaced or
        # that have previously been replaced
        # TODO: This probably only works if only one species
        # is being substituted.
        n_host_sites = len(possible_dopant_sites) + len(
            self.dopant_indices[constraint_index])
        n_dopants = np.round(n_host_sites * dopant_concentration)

        # In some cases, where # of another dopant is rounded up, we may
        # have a rounding error
        # Therefore, we must limit these dopants to len(possible_dopant_sites)
        n_dopants = min(n_dopants, len(possible_dopant_sites))

        # Randomly pick sites to place dopants
        dopant_sites = rng.choice(possible_dopant_sites,
                                  int(n_dopants),
                                  replace=False)
        for i in dopant_sites:
            self._sites[constraint_index][i].species = {dopant_species: 1}

        # Keep track of concentrations in each shell
        self._dopant_concentration[constraint_index][dopant_species] = len(
            dopant_sites) / n_host_sites

        # Keep track of sites with dopants
        self.dopant_indices[constraint_index].extend(dopant_sites)

    def to_file(self, fmt: str = "xyz", name: str = "nanoparticle.xyz"):
        """
        Write the full nanoparticle structure to a file.

        Args:
            fmt (str, None): _description_. The format to write the structure
                as. For more options, check the pymatgen Molecule.to()
                function.
                Defaults to "xyz".
            name (str, None): _description_. Defaults to "nanoparticle.xyz".
        """
        if self.has_structure is False:
            raise RuntimeError('Nanoparticle not generated.'
                               ' Please use the generate() function')
        _np = Molecule.from_sites(self.sites)
        _ = _np.to(fmt, name)

    def dopants_to_file(self,
                        fmt: str = "xyz",
                        name: str = "dopant_nanoparticle.xyz"):
        """
        Writes the nanoparticle structure of only the dopants to a file.

        Args:
            fmt (str, None): The format to write the structure as. For
                more options, check the pymatgen Molecule.to() function.
                Defaults to "xyz".
            name (str, None): The name of the file.
                Defaults to "dopant_nanoparticle.xyz".

        """
        if self.has_structure is False:
            raise RuntimeError('Nanoparticle not generated.'
                               ' Please use the generate() function')
        _np = Molecule.from_sites(self.dopant_sites)
        _ = _np.to(fmt, name)

    @property
    def sites(self) -> Sequence[Site]:
        """
        Gets a list of all the sites in the nanoparticle

        Returns:
            Sequence[Site]: List of all the sites in the nanoparticle
        """
        if self.has_structure is False:
            raise RuntimeError('Nanoparticle not generated.'
                               ' Please use the generate() function')
        return [_site for sites in self._sites for _site in sites]

    @property
    def dopant_sites(self) -> Sequence[Site]:
        """
        Gets a list of all the dopant sites in the nanoparticle

        Returns:
            Sequence[Site]: List of all the dopant sites in the nanoparticle
        """
        if self.has_structure is False:
            raise RuntimeError('Nanoparticle not generated.'
                               ' Please use the generate() function')
        _sites = []
        for dopant_indices, sites in zip(self.dopant_indices, self._sites):
            for i in dopant_indices:
                _sites.append(sites[i])
        return _sites

    def dopant_concentrations(self,
                              constraint_index: Optional[int] = None,
                              replaced_species: Optional[str] = 'Y') -> Dict:
        """
        Gets the dopant concentrations of all the dopants in the
        `constraint_index`-th constraint occupying the
        `replaced_species` sites.

        Args:
            constraint_index (int, None): Index of constraint. If None, return
                the concentrations of the whole nanoparticle.
                Defaults to None.
            replaced_species (str, None): Replaced species symbol.
                Defaults to 'Y'.

        Returns:
            Dict: Returns the a dictionary of the dopants and
                their concentrations
        """
        if self.has_structure is False:
            raise RuntimeError('Nanoparticle not generated.'
                               ' Please use the generate() function')
        if constraint_index is None:
            num_replaced_sites = len([
                i for i, site in enumerate(self.sites)
                if site.specie.symbol == replaced_species
            ])
            total_num_sites = len(self.dopant_sites) + num_replaced_sites

            dopant_amount = {}
            for dopant in self.dopant_sites:
                try:
                    dopant_amount[str(dopant.specie)] += 1
                except KeyError:
                    dopant_amount[str(dopant.specie)] = 1

            return {
                key: (item / total_num_sites)
                for key, item in dopant_amount.items()
            }
        return None


def get_nayf4_structure() -> Structure:
    """
    Get a Structure object for a single unit cell of NaYF4.

    Returns:
        Structure: Structure of NaYF4
    """
    lattice = Lattice.hexagonal(a=6.067, c=7.103)
    species = [
        'Na', 'Na', 'Na', 'Y', 'Y', 'Y', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
        'F', 'F', 'F', 'F', 'F'
    ]
    positions = [[0.3333, 0.6667, 0.5381], [0.3333, 0.6667, 0.9619],
                 [0.6667, 0.3333, 0.75], [0, 0, 0.9969], [0, 0, 0.5031],
                 [0.6667, 0.3333, 0.25], [0.0272, 0.2727, 0.2500],
                 [0.0572, 0.2827, 0.7500], [0.2254, 0.9428, 0.7500],
                 [0.2455, 0.9728, 0.2500], [0.4065, 0.3422, 0.0144],
                 [0.4065, 0.3422, 0.4856], [0.6578, 0.0643, 0.0144],
                 [0.6578, 0.0643, 0.4856], [0.7173, 0.7746, 0.7500],
                 [0.7273, 0.7545, 0.2500], [0.9357, 0.5935, 0.0144],
                 [0.9357, 0.5935, 0.4856]]

    return Structure(lattice, species=species, coords=positions)


def get_wse2_structure() -> Structure:
    """
    Get a Structure object for a single unit cell of WSe2.

    Returns:
        Structure: Structure of WSe2
    """
    lattice = Lattice.hexagonal(a=3.327, c=15.069)
    species = ['Se', 'Se', 'Se', 'Se', 'W', 'W']
    positions = [[0.3333, 0.6667, 0.6384], [0.3333, 0.6667, 0.8616],
                 [0.6667, 0.3333, 0.1384], [0.6667, 0.3333, 0.3616],
                 [0.3333, 0.6667, 0.25], [0.6667, 0.3333, 0.75]]

    return Structure(lattice, species=species, coords=positions)
