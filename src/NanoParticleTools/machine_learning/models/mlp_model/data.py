from .._data import DataProcessor
from typing import List
from torch_geometric.data.data import Data

import numpy as np
import torch

class FeatureProcessor(DataProcessor):
    def __init__(self,
                 max_layers: int = 4,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 **kwargs):
        """
        :param max_layers: 
        :param possible_elements:
        """
        
        super().__init__(fields = ['formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        
        self.max_layers = max_layers
        self.possible_elements = possible_elements
        
    def process_doc(self, 
                    doc: dict) -> torch.Tensor:
        constraints = self.get_item_from_doc(doc, 'input.constraints')
        dopant_concentration = self.get_item_from_doc(doc, 'dopant_concentration')
        
        # Construct the feature array
        feature = []
        for layer in range(self.max_layers):
            _layer_feature = []
            try:
                _layer_feature.append(constraints[layer]['radius'])
            except:
                _layer_feature.append(0)
            for el in self.possible_elements:
                try:
                    _layer_feature.append(dopant_concentration[layer][el]*100)
                except:
                    _layer_feature.append(0)
            feature.append(_layer_feature)
        return {'x': torch.tensor(np.hstack(feature)).float()}

    def __str__(self):
        return f"Feature Processor - {self.max_layers} x [radius, x_{', x_'.join(self.possible_elements)}]"

    @property
    def is_graph(self):
        return False

    @property
    def data_cls(self):
        return Data


class VolumeFeatureProcessor(DataProcessor):
    def __init__(self,
                 max_layers: int = 4,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 **kwargs):
        """
        :param max_layers: 
        :param possible_elements:
        """
        
        super().__init__(fields = ['formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        
        self.max_layers = max_layers
        self.possible_elements = possible_elements

    def process_doc(self, 
                    doc: dict) -> torch.Tensor:
        constraints = self.get_item_from_doc(doc, 'input.constraints')
        dopant_concentration = self.get_item_from_doc(doc, 'dopant_concentration')
        
        # Construct the feature array
        feature = []
        r_lower_bound = 0
        for layer in range(self.max_layers):
            _layer_feature = []
            try:
                if isinstance(constraints[layer], dict):
                    radius = constraints[layer]['radius']
                else:
                    radius = constraints[layer].radius
                
                volume = 4/3*np.pi*(radius**3-r_lower_bound**3)
                r_lower_bound = radius
                _layer_feature.append(radius)
                _layer_feature.append(volume/1000000)
            except:
                _layer_feature.append(0)
                _layer_feature.append(0)
            for el in self.possible_elements:
                try:
                    _layer_feature.append(dopant_concentration[layer][el]*100)
                except:
                    _layer_feature.append(0)
            feature.append(_layer_feature)
        return {'x': torch.tensor(np.hstack(feature)).float()}
    
    def __str__(self):
        return f"Feature Processor - {self.max_layers} x [radius, volume, x_{', x_'.join(self.possible_elements)}]"

    @property
    def is_graph(self):
        return False

    @property
    def data_cls(self):
        return Data