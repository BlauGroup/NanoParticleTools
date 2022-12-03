from ....inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint
from .._data import DataProcessor
from typing import List, Union, Tuple, Optional
from functools import lru_cache
import torch

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
        super().__init__(fields = ['formula_by_constraint', 'dopant_concentration', 'input'],
                         **kwargs)
        
        self.possible_elements = possible_elements
        self.dopants_dict = {key: i for i, key in enumerate(self.possible_elements)}
        self.edge_attr_bias = edge_attr_bias
        self.single_edge_attr = single_edge_attr

    def get_node_features(self, constraints, dopant_specifications) -> torch.Tensor:
        #dopant_feature_dict = get_all_dopant_features(n_features=59)
        
        node_features = []

        for constraint_i, dopant_concentration, el, _ in dopant_specifications:
            _dopant_feature = [self.dopants_dict[el], 
                               dopant_concentration*10]
            
            # Add the spatial and volume information to the feature
            r_inner, r_outer = self.get_radii(constraint_i, constraints)
            volume = self.get_volume(r_outer) - self.get_volume(r_inner)
            _dopant_feature.extend([r_outer-r_inner, volume/1e6, r_inner, r_outer, (r_inner + r_outer)/2])
            # _dopant_feature.extend(dopant_feature_dict[el])
            
            node_features.append(_dopant_feature)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        return {'x': node_features}
    
    def get_edge_features(self, constraints, dopant_specifications) -> torch.Tensor:
        
        # Build all the edge connections. Treat this as fully connected, where each edge has a feature of distance
        edge_attributes = []
        edge_connections = []
        for i, (constraint_i, dopant_concentration, el, _) in enumerate(dopant_specifications):
            r_inner_i, r_outer_i = self.get_radii(constraint_i, constraints)

            for j, (constraint_j, dopant_concentration, el, _) in enumerate(dopant_specifications):
                r_inner_j, r_outer_j = self.get_radii(constraint_j, constraints)
                if self.single_edge_attr:
                    _edge_feature = [(r_outer_j + r_inner_j)/2 - (r_outer_i + r_inner_i)/2]
                else:
                    _edge_feature = [r_outer_j - r_outer_i,
                                     (r_outer_j + r_inner_j)/2 - (r_outer_i + r_inner_i)/2,
                                     r_inner_j - r_inner_i]
                if j < i: 
                    # Since this is an undirected graph, we don't need the backwards elements.
                    # We keep the i==j elements, so there is self-interaction
                    continue
                elif i == j:
                    edge_connections.append([i, j])
                    edge_attributes.append(_edge_feature)
                else:
                    # i > j
                    edge_connections.append([i, j])
                    edge_attributes.append(_edge_feature)
                    
                    # The graph is undirected, so j->i should be == to i->j
                    edge_connections.append([j, i])
                    edge_attributes.append(_edge_feature)
        
        edge_index = torch.tensor(edge_connections, dtype=torch.long).t().contiguous()
        # We add a value to the edge_attr before applying 1/edge_attr to prevent division by 0
        edge_attr = 1/(self.edge_attr_bias+torch.tensor(edge_attributes, dtype=torch.float))
        return {'edge_index': edge_index,
                'edge_attr': edge_attr}
        
    def get_data_graph(self, 
                       constraints: List[NanoParticleConstraint], 
                       dopant_specifications: List[Tuple[int, float, str, str]]):
        
        output_dict = self.get_node_features(constraints, dopant_specifications)
        output_dict.update(self.get_edge_features(constraints, dopant_specifications))
        
        return output_dict
    
    def process_doc(self,
                    doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']
        
        try:
            constraints = [SphericalConstraint.from_dict(c) for c in constraints]
        except:
            pass
        
        return self.get_data_graph(constraints, dopant_specifications)

    @property
    def is_graph(self):
        return True

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
        super().__init__(fields = ['formula_by_constraint', 'dopant_concentration', 'input'],
                         **kwargs)
        
        self.possible_elements = possible_elements
        self.n_possible_elements = len(possible_elements)
        self.dopants_dict = {key: i for i, key in enumerate(self.possible_elements)}
        self.edge_attr_bias = edge_attr_bias
        self.interaction_sigma = interaction_sigma

    def get_node_features(self, constraints, dopant_specifications) -> torch.Tensor:
        # Generate the tensor of concentrations for the original constraints.
        ## Initialize it to 0
        concentrations = torch.zeros(len(constraints), self.n_possible_elements)
        
        ## Fill in the concentrations that are present
        for i, x, el, _ in dopant_specifications:
            concentrations[i][self.dopants_dict[el]] = x
        
        return {'x': concentrations}
    
    def get_edge_features(self, constraints, dopant_specifications) -> torch.Tensor:
        interaction_sigma = torch.tensor(self.interaction_sigma, dtype=torch.float32)
        # Build all the edge connections. Treat this as fully connected, where each edge has a feature of distance
        edge_attributes = []
        edge_connections = []
        for constraint_i in range(len(constraints)):
            r_inner_i, r_outer_i = self.get_radii(constraint_i, constraints)
            for constraint_j in range(len(constraints)):
                r_inner_j, r_outer_j = self.get_radii(constraint_j, constraints)
                _edge_attr = torch.tensor([r_inner_i, r_outer_i, r_inner_j, r_outer_j])
                
                edge_attributes.append(_edge_attr)
                edge_connections.append([constraint_i, constraint_j])
        
        edge_index = torch.tensor(edge_connections, dtype=torch.long).t().contiguous()
        # We add a value to the edge_attr before applying 1/edge_attr to prevent division by 0
        edge_attr = torch.vstack(edge_attributes)
        return {'edge_index': edge_index,
                'edge_attr': edge_attr}
        
    def get_data_graph(self, 
                       constraints: List[NanoParticleConstraint], 
                       dopant_specifications: List[Tuple[int, float, str, str]]):
        
        output_dict = self.get_node_features(constraints, dopant_specifications)
        output_dict.update(self.get_edge_features(constraints, dopant_specifications))
        
        return output_dict
    
    def process_doc(self,
                    doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']
        
        try:
            constraints = [SphericalConstraint.from_dict(c) for c in constraints]
        except:
            pass
        
        return self.get_data_graph(constraints, dopant_specifications)

    @property
    def is_graph(self):
        return True

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