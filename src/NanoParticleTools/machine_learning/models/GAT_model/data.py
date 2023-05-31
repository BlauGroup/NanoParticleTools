from NanoParticleTools.inputs import NanoParticleConstraint, SphericalConstraint
from NanoParticleTools.machine_learning.data.processors import DataProcessor

from typing import List, Union, Tuple, Optional
from functools import lru_cache
import torch
from torch_geometric.data.data import Data


class GraphFeatureProcessor(DataProcessor):

    def __init__(self,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 edge_attr_bias: Union[float, int] = 0.5,
                 single_edge_attr: Optional[bool] = True,
                 **kwargs):
        """
        :param possible_elements: 
        :param edge_attr_bias: A bias added to the edge_attr before applying 1/edge_attr. This serves to eliminate
            divide by zero and inf in the tensor. Additionally, it acts as a weight on the self-interaction.
        """
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ],
                         **kwargs)

        self.possible_elements = possible_elements
        self.dopants_dict = {
            key: i
            for i, key in enumerate(self.possible_elements)
        }
        self.edge_attr_bias = edge_attr_bias
        self.single_edge_attr = single_edge_attr

    def get_node_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        #dopant_feature_dict = get_all_dopant_features(n_features=59)

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
        # Build all the edge connections. Treat this as fully connected, where each edge has a feature of distance
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

        try:
            constraints = [
                SphericalConstraint.from_dict(c) for c in constraints
            ]
        except:
            pass

        return self.get_data_graph(constraints, dopant_specifications)

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return Data


class GraphInteractionFeatureProcessor(DataProcessor):

    def __init__(self,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 edge_attr_bias: Union[float, int] = 0.5,
                 interaction_sigma: Optional[float] = 20.0,
                 **kwargs):
        """
        :param possible_elements: 
        :param edge_attr_bias: A bias added to the edge_attr before applying 1/edge_attr. This serves to eliminate
            divide by zero and inf in the tensor. Additionally, it acts as a weight on the self-interaction.
        """
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ],
                         **kwargs)

        self.possible_elements = possible_elements
        self.n_possible_elements = len(possible_elements)
        self.dopants_dict = {
            key: i
            for i, key in enumerate(self.possible_elements)
        }
        self.edge_attr_bias = edge_attr_bias
        self.interaction_sigma = interaction_sigma

    def get_node_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        # Generate the tensor of concentrations for the original constraints.
        # Initialize it to 0
        concentrations = torch.zeros(len(constraints),
                                     self.n_possible_elements)

        ## Fill in the concentrations that are present
        for i, x, el, _ in dopant_specifications:
            concentrations[i][self.dopants_dict[el]] = x

        return {'x': concentrations}

    def get_edge_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        # Build all the edge connections. Treat this as fully connected, where each edge has a feature of distance
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

        try:
            constraints = [
                SphericalConstraint.from_dict(c) for c in constraints
            ]
        except:
            pass

        return self.get_data_graph(constraints, dopant_specifications)

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return Data


# class GraphInteractionFeatureProcessor(DataProcessor):
#     def __init__(self,
#                  possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
#                  edge_attr_bias: Union[float, int] = 0.5,
#                  interaction_sigma: Optional[float] = 20.0,
#                  **kwargs):
#         """
#         :param possible_elements:
#         :param edge_attr_bias: A bias added to the edge_attr before applying 1/edge_attr. This serves to eliminate
#             divide by zero and inf in the tensor. Additionally, it acts as a weight on the self-interaction.
#         """
#         super().__init__(fields = ['formula_by_constraint', 'dopant_concentration', 'input'],
#                          **kwargs)

#         self.possible_elements = possible_elements
#         self.n_possible_elements = len(possible_elements)
#         self.dopants_dict = {key: i for i, key in enumerate(self.possible_elements)}
#         self.edge_attr_bias = edge_attr_bias
#         self.interaction_sigma = interaction_sigma

#     def get_node_features(self, constraints, dopant_specifications) -> torch.Tensor:
#         # Generate the tensor of concentrations for the original constraints.
#         ## Initialize it to 0
#         concentrations = torch.zeros(len(constraints), self.n_possible_elements)

#         ## Fill in the concentrations that are present
#         for i, x, el, _ in dopant_specifications:
#             concentrations[i][self.dopants_dict[el]] = x

#         return {'x': concentrations}

#     def get_edge_features(self, constraints, dopant_specifications) -> torch.Tensor:
#         interaction_sigma = torch.tensor(self.interaction_sigma, dtype=torch.float32)
#         # Build all the edge connections. Treat this as fully connected, where each edge has a feature of distance
#         edge_attributes = []
#         edge_connections = []
#         interaction_strength = []
#         for constraint_i in range(len(constraints)):
#             r_inner_i, r_outer_i = self.get_radii(constraint_i, constraints)
#             for constraint_j in range(len(constraints)):
#                 r_inner_j, r_outer_j = self.get_radii(constraint_j, constraints)
#                 _edge_attr = torch.tensor([r_inner_i, r_outer_i, r_inner_j, r_outer_j])
#                 _interaction_strength = integrated_gaussian_interaction(_edge_attr[0],
#                                                                         _edge_attr[1],
#                                                                         _edge_attr[2],
#                                                                         _edge_attr[3],
#                                                                         interaction_sigma)

#                 interaction_strength.append(_interaction_strength)
#                 edge_attributes.append(_edge_attr)
#                 edge_connections.append([constraint_i, constraint_j])

#         edge_index = torch.tensor(edge_connections, dtype=torch.long).t().contiguous()
#         # We add a value to the edge_attr before applying 1/edge_attr to prevent division by 0
#         edge_attr = torch.vstack(edge_attributes)
#         interaction_strength = torch.tensor(interaction_strength)
#         return {'edge_index': edge_index,
#                 'edge_attr': edge_attr,
#                 'interaction_strength': interaction_strength}

#     def get_data_graph(self,
#                        constraints: List[NanoParticleConstraint],
#                        dopant_specifications: List[Tuple[int, float, str, str]]):

#         output_dict = self.get_node_features(constraints, dopant_specifications)
#         output_dict.update(self.get_edge_features(constraints, dopant_specifications))

#         return output_dict

#     def process_doc(self,
#                     doc: dict) -> dict:
#         constraints = doc['input']['constraints']
#         dopant_specifications = doc['input']['dopant_specifications']

#         try:
#             constraints = [SphericalConstraint.from_dict(c) for c in constraints]
#         except:
#             pass

#         return self.get_data_graph(constraints, dopant_specifications)

#     @property
#     def is_graph(self):
#         return True