from ....inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint
from ....species_data.species import Dopant
from .._data import DataProcessor, LabelProcessor, BaseNPMCDataset
from .._data import NPMCDataModule as _NPMCDataModule

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from matplotlib import pyplot as plt
import numpy as np

from typing import List, Union, Tuple, Optional
from itertools import combinations_with_replacement
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
    
    @property
    @lru_cache
    def edge_type_map(self):
        edge_type_map = {}
        for i, (el1, el2) in enumerate(list(combinations_with_replacement(self.possible_elements, 2))):
            try:
                edge_type_map[el1][el2] = i
            except:
                edge_type_map[el1] = {el2: i}

            try:
                edge_type_map[el2][el1] = i
            except:
                edge_type_map[el2] = {el1: i}
        return edge_type_map
        
    def get_node_features(self, constraints, dopant_specifications) -> torch.Tensor:
        """Here, the node feature is simply the id of the element pair and the distance between the layers"""
        
        node_features = []
        
        for i, (constraint_i, x_i, el_i, _) in enumerate(dopant_specifications):
            r_inner_i, r_outer_i = self.get_radii(constraint_i, constraints)
            r_mean_i = (r_outer_i + r_inner_i)/2
            v_i = self.get_volume(r_outer_i) - self.get_volume(r_inner_i)

            for j, (constraint_j, x_j, el_j, _) in enumerate(dopant_specifications):
                r_inner_j, r_outer_j = self.get_radii(constraint_j, constraints)
                r_mean_j = (r_outer_j + r_inner_j)/2
                v_j = self.get_volume(r_outer_j) - self.get_volume(r_inner_j)
                d_mean = r_mean_j - r_mean_i
                
                node_features.append([self.edge_type_map[el_i][el_j], d_mean, x_i, x_j, v_i/1e6, v_j/1e6])

        node_features = torch.tensor(node_features, dtype=torch.float)
        return {'x': node_features, 
                'num_nodes': node_features.shape[0]}
    
    def get_edge_features(self, 
                          n_nodes: int) -> torch.Tensor:
        
        # Build all the edge connections. Treat this as fully connected
        
        x, y = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='xy')
        edge_index = torch.vstack([x.reshape(n_nodes**2), y.reshape(n_nodes**2)])
        
        return {'edge_index': edge_index}
        
    def get_data_graph(self, 
                       constraints: List[NanoParticleConstraint], 
                       dopant_specifications: List[Tuple[int, float, str, str]]):
        
        output_dict = self.get_node_features(constraints, dopant_specifications)
        output_dict.update(self.get_edge_features(output_dict['x'].shape[0]))
        output_dict['constraints'] = constraints
        output_dict['dopant_specifications'] = dopant_specifications
        
        return output_dict
    
    def process_doc(self,
                    doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']
        
        try:
            ## Should use MontyDecoder to deserialize these
            constraints = [SphericalConstraint.from_dict(c) for c in constraints]
        except:
            pass
        
        return self.get_data_graph(constraints, dopant_specifications)
        
class NPMCDataset(BaseNPMCDataset):
    """
    
    """
    def process_single_doc(doc, 
                           feature_processor: DataProcessor,
                           label_processor: DataProcessor) -> Data:
        _d = feature_processor.process_doc(doc)
        _d['y'] = label_processor.process_doc(doc)
        return Data(**_d)

class NPMCDataModule(_NPMCDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_train, self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_val, self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_test, self.batch_size, shuffle=False)