from NanoParticleTools.inputs import NanoParticleConstraint, SphericalConstraint
from NanoParticleTools.machine_learning.data.processors import FeatureProcessor

from typing import List, Union, Tuple, Optional, Any
import torch
import itertools
from torch_geometric.data.data import Data
from torch_geometric.typing import SparseTensor
from monty.json import MontyDecoder


class CNNData(Data):

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif key == 'y':
            return 0
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif key == 'node_dopant_index':
            return self.x.size(0)
        elif key == 'x_layer_idx':
            return self.radii.size(0)
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0


class GraphFeatureProcessor(FeatureProcessor):

    def __init__(self,
                 resolution: Optional[float] = 0.1,
                 full_nanoparticle: Optional[bool] = True,
                 cutoff_distance: Optional[int] = 3,
                 log_vol: Optional[bool] = True,
                 shift_composition: Optional[bool] = False,
                 **kwargs):
        """
        :param possible_elements:
        :param cutoff_distance:
        :param resolution: Angstroms
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.resolution = resolution
        self.full_nanoparticle = full_nanoparticle
        self.cutoff_distance = cutoff_distance
        self.log_vol = log_vol
        self.shift_composition = shift_composition

    def get_node_features(
        self, constraints: List[NanoParticleConstraint],
        dopant_specifications: List[Tuple[int, float, str,
                                          str]]) -> torch.Tensor:
        # Generate the tensor of concentrations for the original constraints.
        # Initialize it to 0
        concentrations = torch.zeros(len(constraints),
                                     self.n_possible_elements)

        # Fill in the concentrations that are present
        for i, x, el, _ in dopant_specifications:
            if self.shift_composition:
                concentrations[i][self.dopants_dict[el]] = x - 0.5
            else:
                concentrations[i][self.dopants_dict[el]] = x

        # Make the array for the representation
        n_subdivisions = torch.ceil(
            torch.tensor(constraints[-1].radius) / self.resolution).int()
        node_features = torch.zeros(
            (n_subdivisions, self.n_possible_elements + 1))

        # Fill in the concentrations
        start_i = 0
        for constraint_i in range(len(constraints)):
            end_i = torch.ceil(
                torch.tensor(constraints[constraint_i].radius) /
                self.resolution).int()
            node_features[start_i:end_i, :self.
                          n_possible_elements] = concentrations[constraint_i]
            start_i = end_i

        # Set the last index to volume
        volume = self.volume(
            torch.arange(0, constraints[-1].radius, self.resolution))
        if self.log_vol:
            volume = torch.log10(volume)

        return {'x': node_features, 'volume': volume}

    def get_edge_features(
            self, constraints: List[NanoParticleConstraint]) -> torch.Tensor:
        # Determine connectivity using a cutoff
        radius = torch.arange(0, constraints[-1].radius, self.resolution)
        xy, yx = torch.meshgrid(radius, radius, indexing='xy')
        distance_matrix = torch.abs(xy - yx)
        edge_index = torch.vstack(
            torch.where(distance_matrix <= self.cutoff_distance))
        edge_attr = distance_matrix[edge_index[0], edge_index[1]]

        volume = self.volume(
            torch.arange(0, constraints[-1].radius, self.resolution))
        if self.log_vol:
            volume = torch.log10(volume)

        source, target = edge_index
        edge_attr = torch.stack([
            edge_attr, volume[source], volume[target], radius[source],
            radius[target]
        ]).moveaxis(0, 1)

        return {'edge_index': edge_index, 'edge_attr': edge_attr}

    def get_data_graph(self, constraints: List[NanoParticleConstraint],
                       dopant_specifications: List[Tuple[int, float, str,
                                                         str]]):

        output_dict = self.get_node_features(constraints,
                                             dopant_specifications)
        output_dict.update(self.get_edge_features(constraints))

        return output_dict

    def process_doc(self, doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']

        constraints = MontyDecoder().process_decoded(constraints)

        return self.get_data_graph(constraints, dopant_specifications)

    def volume(self,
               radius: Union[List, int, torch.Tensor],
               shell_width: Optional[float] = 0.01) -> torch.Tensor:
        """
            Takes inner radius
            """
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius)

        outer_radius = radius + shell_width

        return self.sphere_volume(outer_radius) - self.sphere_volume(radius)

    @staticmethod
    def sphere_volume(radius: torch.Tensor) -> torch.Tensor:
        return 3 / 4 * torch.pi * torch.pow(radius, 3)

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return Data

    def __str__(self) -> str:
        return f"Discrete Graph Feature Processor - resolution = {self.resolution}A"


class FeatureProcessor(FeatureProcessor):

    def __init__(self,
                 resolution: Optional[float] = 0.1,
                 max_np_size: Optional[int] = 500,
                 dims: Optional[int] = 1,
                 full_nanoparticle: Optional[bool] = True,
                 **kwargs):
        """
        :param possible_elements:
        :param cutoff_distance:
        :param resolution: Angstroms
        :param max_np_size: Angstroms
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.n_possible_elements = len(self.possible_elements)
        self.dopants_dict = {
            key: i
            for i, key in enumerate(self.possible_elements)
        }
        self.resolution = resolution
        self.max_np_size = max_np_size
        self.max_divisions = -int(max_np_size // -resolution)
        assert dims > 0 and dims <= 3, "Representation for must be 1, 2, or 3 dimensions"
        self.dims = dims
        self.full_nanoparticle = full_nanoparticle

    def get_node_features(
        self, constraints: List[NanoParticleConstraint],
        dopant_specifications: List[Tuple[int, float, str,
                                          str]]) -> torch.Tensor:
        # Generate the tensor of concentrations for the original constraints.
        # Initialize it to 0
        concentrations = torch.zeros(len(constraints),
                                     self.n_possible_elements)

        # Fill in the concentrations that are present
        for i, x, el, _ in dopant_specifications:
            concentrations[i][self.dopants_dict[el]] = x

        # Determine the number of pixels/subdivisions this specific particle needs.
        # Using this instead of the max size will save us some time when assigning pixels
        n_subdivisions = torch.ceil(
            torch.tensor(constraints[-1].radius) / self.resolution).int()

        # Make the array for the representation
        node_features = torch.zeros([n_subdivisions
                                     for _ in range(self.dims)] +
                                    [self.n_possible_elements])

        # radius = torch.arange(0, n_subdivisions, self.resolution)
        radius = torch.arange(0, n_subdivisions)
        mg = torch.meshgrid(*[radius for _ in range(self.dims)], indexing='ij')
        radial_distance = torch.sqrt(
            torch.sum(torch.stack([torch.pow(_el, 2) for _el in mg]), dim=0))

        lower_bound = 0
        for constraint_i in range(len(constraints)):
            upper_bound = constraints[constraint_i].radius / self.resolution

            idx = torch.where(
                torch.logical_and(radial_distance >= lower_bound,
                                  radial_distance < upper_bound))
            node_features.__setitem__(idx, concentrations[constraint_i])

            lower_bound = upper_bound

        if self.full_nanoparticle:
            full_repr = torch.zeros(
                [2 * n_subdivisions
                 for _ in range(self.dims)] + [self.n_possible_elements])
            for ops in itertools.product(*[[0, 1] for _ in range(self.dims)]):
                # 0 = no change, 1 = flip along this axis
                idx = [
                    slice(n_subdivisions) if j == 1 else slice(
                        n_subdivisions, 2 * n_subdivisions) for j in ops
                ]

                flip_dims = [i for i, j in enumerate(ops) if j == 1]
                full_repr.__setitem__(idx, torch.flip(node_features,
                                                      flip_dims))

            node_features = full_repr

        # Put the channel in the first index
        node_features = node_features.moveaxis(-1, 0)

        # Pad image so they are all the same size
        pad_size = (2 * self.max_divisions - node_features.shape[1]) // 2
        padding_tuple = tuple(
            [pad_size for i in range(self.dims) for i in range(2)])
        node_features = torch.nn.functional.pad(node_features, padding_tuple)

        if self.dims == 1:
            node_features = torch.nn.functional.avg_pool1d(node_features, 2, 2)
        elif self.dims == 2:
            node_features = torch.nn.functional.avg_pool2d(node_features, 2, 2)
        elif self.dims == 3:
            node_features = torch.nn.functional.avg_pool3d(node_features, 2, 2)

        return {'x': node_features.unsqueeze(0)}

    def get_data_graph(self, constraints: List[NanoParticleConstraint],
                       dopant_specifications: List[Tuple[int, float, str,
                                                         str]]):

        output_dict = self.get_node_features(constraints,
                                             dopant_specifications)

        return output_dict

    def process_doc(self, doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_concentration = doc['dopant_concentration']

        constraints = MontyDecoder().process_decoded(constraints)

        _dopant_specifications = [
            (layer_idx, conc, el, None)
            for layer_idx, dopants in enumerate(dopant_concentration)
            for el, conc in dopants.items() if el in self.possible_elements
        ]

        return self.get_data_graph(constraints, _dopant_specifications)

    def volume(self,
               radius: Union[List, int, torch.Tensor],
               shell_width: Optional[float] = 0.01) -> torch.Tensor:
        """
            Takes inner radius
            """
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius)

        outer_radius = radius + shell_width

        return self.sphere_volume(outer_radius) - self.sphere_volume(radius)

    @staticmethod
    def sphere_volume(radius: torch.Tensor) -> torch.Tensor:
        return 3 / 4 * torch.pi * torch.pow(radius, 3)

    def __str__(self) -> str:
        return (f"CNN Feature Processor - resolution = "
                f"{self.resolution}A - max_np_size = {self.max_np_size}")

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return CNNData
