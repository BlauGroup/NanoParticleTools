from .._data import DataProcessor, LabelProcessor, BaseNPMCDataset
from torch.utils.data import DataLoader
from .._data import NPMCDataModule as _NPMCDataModule
import torch
import numpy as np
from typing import List

SPECIES_TYPE_INDEX = 0
COMPOSITION_INDEX = 1
VOLUME_INDEX = 2

class TransformerFeatureProcessor(DataProcessor):
    def __init__(self,
                 fields = ['formula_by_constraint', 'dopant_concentration', 'input.constraints'],
                 max_layers: int = 4,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 volume_scale_factor = 1e-6,
                 **kwargs):
        """
        :param max_layers: 
        :param possible_elements:
        """
        super().__init__(fields=fields, **kwargs)
        
        self.max_layers = max_layers
        self.possible_elements = possible_elements
        self.volume_scale_factor = volume_scale_factor
        
    def process_doc(self, 
                    doc: dict) -> torch.Tensor:
        constraints = self.get_item_from_doc(doc, 'input.constraints')
        dopant_concentration = self.get_item_from_doc(doc, 'dopant_concentration')

        types = torch.tensor([j for i in range(self.max_layers) for j in range(len(self.possible_elements))])

        volumes = []
        compositions = []
        r_lower_bound = 0
        for layer in range(self.max_layers):
            try:
                if isinstance(constraints[layer], dict):
                    radius = constraints[layer]['radius']
                else:
                    radius = constraints[layer].radius
                volume = self.get_volume(radius) - self.get_volume(r_lower_bound)
                r_lower_bound = radius
                for i in range(len(self.possible_elements)):
                    volumes.append(volume * self.volume_scale_factor)
            except:
                for i in range(len(self.possible_elements)):
                    volumes.append(0)
            
            for el in self.possible_elements:
                try:
                    compositions.append(dopant_concentration[layer][el])
                except:
                    compositions.append(0)

        return torch.vstack([types, torch.tensor(volumes), torch.tensor(compositions)])

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
        x = torch.concat([data.x for data in data_list]).reshape(-1, *data_list[0].x.shape)
        y = torch.concat([data.y for data in data_list]).reshape(-1, *data_list[0].y.shape)

        return Data(x=x, y=y)

class NPMCDataModule(_NPMCDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_train, self.batch_size, collate_fn=self.dataset_class.collate, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_val, self.batch_size, collate_fn=self.dataset_class.collate, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_test, self.batch_size, collate_fn=self.dataset_class.collate, shuffle=False)
