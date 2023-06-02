from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint
from NanoParticleTools.machine_learning.data.processors import DataProcessor

from typing import List, Tuple, Optional
from itertools import combinations_with_replacement
from functools import lru_cache
import torch
from torch_geometric.data.data import Data
from monty.json import MontyDecoder


class GraphFeatureProcessor(DataProcessor):

    def __init__(self,
                 log_volume: Optional[bool] = False,
                 **kwargs):
        """
        :param possible_elements:
        :param log_volume: Whether to apply a log10 to the volume to reduce orders of magnitude.
            defaults to False
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.log_volume = log_volume

    @property
    @lru_cache
    def edge_type_map(self):
        edge_type_map = {}
        for i, (el1, el2) in enumerate(
                list(combinations_with_replacement(self.possible_elements, 2))):
            try:
                edge_type_map[el1][el2] = i
            except KeyError:
                edge_type_map[el1] = {el2: i}

            try:
                edge_type_map[el2][el1] = i
            except KeyError:
                edge_type_map[el2] = {el1: i}
        return edge_type_map

    def get_node_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        """
        Here, the node feature is simply the id of the element pair
        and the distance between the layers
        """

        node_features = []

        for i, (constraint_i, x_i, el_i,
                _) in enumerate(dopant_specifications):
            r_inner_i, r_outer_i = self.get_radii(constraint_i, constraints)
            r_mean_i = (r_outer_i + r_inner_i) / 2
            v_i = self.get_volume(r_outer_i) - self.get_volume(r_inner_i)

            for j, (constraint_j, x_j, el_j,
                    _) in enumerate(dopant_specifications):
                r_inner_j, r_outer_j = self.get_radii(constraint_j,
                                                      constraints)
                r_mean_j = (r_outer_j + r_inner_j) / 2
                v_j = self.get_volume(r_outer_j) - self.get_volume(r_inner_j)
                d_mean = r_mean_j - r_mean_i

                node_features.append([
                    self.edge_type_map[el_i][el_j], d_mean, x_i, x_j, v_i, v_j
                ])

        node_features = torch.tensor(node_features, dtype=torch.float)

        if self.log_volume:
            node_features[:, -2:] = torch.log10(node_features[:, -2:])
        else:
            node_features[:, -2:] = node_features[:, -2:] / 1e6

        return {'x': node_features, 'num_nodes': node_features.shape[0]}

    def get_edge_features(self, n_nodes: int) -> torch.Tensor:

        # Build all the edge connections. Treat this as fully connected

        x, y = torch.meshgrid(torch.arange(n_nodes),
                              torch.arange(n_nodes),
                              indexing='xy')
        edge_index = torch.vstack(
            [x.reshape(n_nodes**2),
             y.reshape(n_nodes**2)])

        return {'edge_index': edge_index}

    def get_data_graph(self, constraints: List[NanoParticleConstraint],
                       dopant_specifications: List[Tuple[int, float, str,
                                                         str]]):

        output_dict = self.get_node_features(constraints,
                                             dopant_specifications)
        output_dict.update(self.get_edge_features(output_dict['x'].shape[0]))

        return output_dict

    def process_doc(self, doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']

        constraints = MontyDecoder().process_decoded(constraints)

        return self.get_data_graph(constraints, dopant_specifications)

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return Data
