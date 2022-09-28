from .._data import DataProcessor, LabelProcessor, BaseNPMCDataset
from torch.utils.data import DataLoader
from .._data import NPMCDataModule as _NPMCDataModule
from typing import List

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
        return torch.tensor(np.hstack(feature)).float()

    def __str__(self):
        return f"Feature Processor - {self.max_layers} x [radius, x_{', x_'.join(self.possible_elements)}]"


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
        return torch.tensor(np.hstack(feature)).float()
    
    def __str__(self):
        return f"Feature Processor - {self.max_layers} x [radius, volume, x_{', x_'.join(self.possible_elements)}]"
    

class Data():
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)

class NPMCDataset(BaseNPMCDataset):
    """
    
    """

    def process_single_doc(doc, 
                           feature_processor: DataProcessor,
                           label_processor: DataProcessor):
        _d = {}
        _d['x'] = feature_processor.process_doc(doc)
        _d['y'] = label_processor.process_doc(doc)
        return Data(**_d)

    @staticmethod
    def collate(data_list: List[Data]):
        if len(data_list) == 0:
            return data_list[0]
        x = torch.vstack([data.x for data in data_list])
        y = torch.vstack([data.y for data in data_list])
        return Data(x=x, y=y)

class NPMCDataModule(_NPMCDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_train, self.batch_size, collate_fn=self.dataset_class.collate, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_val, self.batch_size, collate_fn=self.dataset_class.collate, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_test, self.batch_size, collate_fn=self.dataset_class.collate, shuffle=False)
