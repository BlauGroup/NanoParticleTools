from NanoParticleTools.inputs import NanoParticleConstraint
from NanoParticleTools.machine_learning.data.processors import FeatureProcessor

import torch
from torch_geometric.data.data import Data
from monty.json import MontyDecoder


class GraphFeatureProcessor(FeatureProcessor):

    def __init__(self,
                 resolution: float = 0.1,
                 full_nanoparticle: bool = True,
                 cutoff_distance: int = 3,
                 log_vol: bool = True,
                 shift_composition: bool = False,
                 **kwargs):
        """
        :param possible_elements:
        :param cutoff_distance:
        :param resolution: Angstroms
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        # yapf: enable

        self.resolution = resolution
        self.full_nanoparticle = full_nanoparticle
        self.cutoff_distance = cutoff_distance
        self.log_vol = log_vol
        self.shift_composition = shift_composition

    def get_node_features(
        self, constraints: list[NanoParticleConstraint],
        dopant_specifications: list[tuple[int, float, str,
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
            self, constraints: list[NanoParticleConstraint]) -> torch.Tensor:
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

    def get_data_graph(self, constraints: list[NanoParticleConstraint],
                       dopant_specifications: list[tuple[int, float, str,
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
               radius: list | int | torch.Tensor,
               shell_width: float = 0.01) -> torch.Tensor:
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
