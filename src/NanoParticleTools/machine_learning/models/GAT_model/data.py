from NanoParticleTools.inputs import NanoParticleConstraint, SphericalConstraint
from NanoParticleTools.machine_learning.data.processors import DataProcessor

from typing import List, Union, Tuple, Optional
from functools import lru_cache
import torch
from torch_geometric.data.data import Data
from monty.json import MontyDecoder


class GraphFeatureProcessor(DataProcessor):

    def __init__(self,
                 edge_attr_bias: Union[float, int] = 0.5,
                 single_edge_attr: Optional[bool] = True,
                 **kwargs):
        """

        Args:
            possible_elements: All dopant elements that may exist in the data
                to be processed.
            edge_attr_bias: A bias added to the edge_attr before
                applying 1/edge_attr. This serves to eliminate divide by zero and inf in the tensor.
                Additionally, it acts as a weight on the self-interaction.
            single_edge_attr: whether or not the edge attribute is a single value or tensor.
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable
        
        self.edge_attr_bias = edge_attr_bias
        self.single_edge_attr = single_edge_attr

    def get_node_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        """
        Build the node features.

        Args:
            constraints: The constraints that define the nanoparticle.
            dopant_specifications: The dopant specifications that define how the dopants
                are placed in the nanoparticle.

        Returns:
            torch.Tensor: _description_
        """
        node_features = []

        types = []
        x = []
        radii = []
        volumes = []
        for constraint_i, dopant_concentration, el, _ in dopant_specifications:
            types.append(self.dopants_dict[el])
            x.append(dopant_concentration)
            # Add the spatial and volume information to the feature
            radius = self.get_radii(constraint_i, constraints)
            radii.append(radius)
            volumes.append(
                self.get_volume(radius[0]) - self.get_volume(radius[1]))
        node_features = torch.tensor(node_features, dtype=torch.float)
        return {
            'x': torch.tensor(x),
            'types': torch.tensor(types),
            'radii': torch.tensor(radii),
            'volumes': torch.tensor(volumes)
        }

    def get_edge_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        """
        Build all the edge connections.

        Treat this as fully connected, where each edge has a feature of distance

        Args:
            constraints (_type_): _description_
            dopant_specifications (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """
        edge_attributes = []
        edge_connections = []
        for i, (constraint_i, dopant_concentration, el,
                _) in enumerate(dopant_specifications):
            r_inner_i, r_outer_i = self.get_radii(constraint_i, constraints)

            for j, (constraint_j, dopant_concentration, el,
                    _) in enumerate(dopant_specifications):
                r_inner_j, r_outer_j = self.get_radii(constraint_j,
                                                      constraints)
                _edge_attr = torch.tensor(
                    [r_inner_i, r_outer_i, r_inner_j, r_outer_j])

                edge_attributes.append(_edge_attr)
                edge_connections.append([i, j])

        edge_index = torch.tensor(edge_connections,
                                  dtype=torch.long).t().contiguous()
        edge_attr = torch.vstack(edge_attributes)
        return {'edge_index': edge_index, 'edge_attr': edge_attr}

    def get_data_graph(self, constraints: List[NanoParticleConstraint],
                       dopant_specifications: List[Tuple[int, float, str,
                                                         str]]):

        output_dict = self.get_node_features(constraints,
                                             dopant_specifications)
        output_dict.update(
            self.get_edge_features(constraints, dopant_specifications))

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


class GraphInteractionFeatureProcessor(DataProcessor):

    def __init__(self,
                 edge_attr_bias: Union[float, int] = 0.5,
                 interaction_sigma: Optional[float] = 20.0,
                 **kwargs):
        """

        Args:
            possible_elements (List[str], optional): _description_. Defaults to ['Yb', 'Er', 'Nd'].
            edge_attr_bias (Union[float, int], optional): A bias added to the edge_attr before
                applying 1/edge_attr. This serves to eliminate divide by zero and inf in the tensor.
                Additionally, it acts as a weight on the self-interaction. Defaults to 0.5.
            interaction_sigma (Optional[float], optional): _description_. Defaults to 20.0.
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.edge_attr_bias = edge_attr_bias
        self.interaction_sigma = interaction_sigma

    def get_node_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        # Generate the tensor of concentrations for the original constraints.
        # Initialize it to 0
        concentrations = torch.zeros(len(constraints),
                                     self.n_possible_elements)

        # Fill in the concentrations that are present
        for i, x, el, _ in dopant_specifications:
            concentrations[i][self.dopants_dict[el]] = x

        return {'x': concentrations}

    def get_edge_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        """
        Build all the edge connections.
        Treat this as fully connected, where each edge has a feature of distance

        Args:
            constraints: The constraints that define the nanoparticle.
            dopant_specifications: The dopant specifications that define how the dopants
                are placed in the nanoparticle.

        Returns:
            torch.Tensor:
        """
        edge_attributes = []
        edge_connections = []
        for constraint_i in range(len(constraints)):
            r_inner_i, r_outer_i = self.get_radii(constraint_i, constraints)
            for constraint_j in range(len(constraints)):
                r_inner_j, r_outer_j = self.get_radii(constraint_j,
                                                      constraints)
                _edge_attr = torch.tensor(
                    [r_inner_i, r_outer_i, r_inner_j, r_outer_j])

                edge_attributes.append(_edge_attr)
                edge_connections.append([constraint_i, constraint_j])

        edge_index = torch.tensor(edge_connections,
                                  dtype=torch.long).t().contiguous()
        edge_attr = torch.vstack(edge_attributes)
        return {'edge_index': edge_index, 'edge_attr': edge_attr}

    def get_data_graph(self, constraints: List[NanoParticleConstraint],
                       dopant_specifications: List[Tuple[int, float, str,
                                                         str]]):

        output_dict = self.get_node_features(constraints,
                                             dopant_specifications)
        output_dict.update(
            self.get_edge_features(constraints, dopant_specifications))

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
