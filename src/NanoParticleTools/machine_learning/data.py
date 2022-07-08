from maggma.core import Store
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from typing import Union, Dict, List, Tuple
import torch
import json
import numpy as np

class DataProcessor():
    def __init__(self, fields):
        """
        :param fields: fields required in the document(s) to be processed
        """
        self.fields = fields

    @property
    def required_fields(self):
        return self.fields
        
    def process_doc(self, doc):
        pass
        
    def get_item_from_doc(self, 
                          doc, 
                          field):
        keys = field.split('.')
        
        val = doc
        for key in keys:
            val = val[key]
        return val
    
    
class FeatureProcessor(DataProcessor):
    def __init__(self,
                 max_layers: int = 4,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.max_layers = max_layers
        self.possible_elements = possible_elements
        
    def process_doc(self, 
                    doc: dict) -> List:
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
        return np.hstack(feature)

    
class LabelProcessor(DataProcessor):
    def __init__(self, 
                 spectrum_range: Union[Tuple, List] = (-1000, 0),
                 output_size = 500,
                 log: bool = False,
                 normalize: bool = False,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.spectrum_range = spectrum_range
        self.output_size = output_size
        self.log = log
        self.normalize = normalize
    
    @property
    def x(self):
        _x = np.linspace(*self.spectrum_range, self.output_size+1)
        
        return (_x[:-1]+_x[1:])/2

    def process_doc(self, 
                    doc: dict) -> List:
        spectrum_x = self.get_item_from_doc(doc, 'output.spectrum_x')
        spectrum_y = self.get_item_from_doc(doc, 'output.spectrum_y')
        min_i = np.argmin(np.abs(np.subtract(spectrum_x, self.spectrum_range[0])))
        max_i = np.argmin(np.abs(np.subtract(spectrum_x, self.spectrum_range[1])))
        if self.spectrum_range[1] > spectrum_x[-1]:
            max_i = len(spectrum_x)
        
        x = spectrum_x[min_i:max_i]
        y = spectrum_y[min_i:max_i]
        
        coarsened_spectrum = np.sum(np.reshape(y, (self.output_size, -1)), axis=1)
        if self.log:
            coarsened_spectrum = np.log10(coarsened_spectrum)
        if self.normalize:
            coarsened_spectrum = coarsened_spectrum/np.sum(coarsened_spectrum)
        
        return list(coarsened_spectrum)
        

class NPMCStoreDataset(Dataset):
    
    def __init__(self, 
                 store: Store, 
                 doc_filter, 
                 feature_processor,
                 label_processor):
        self.store = store
        self.doc_filter = doc_filter
        self.feature_processor = feature_processor
        self.label_processor = label_processor
        self._object_ids = None
    
    @property
    def object_ids(self):
        if self._object_ids is None:
            self.store.connect()
            self._object_ids = [doc['_id'] for doc in self.store.query(self.doc_filter, properties=['_id'])]
            self.store.close()
        return self._object_ids
    
    @property
    def required_fields(self):
        return self.feature_processor.required_fields + self.label_processor.required_fields
    
    def query(self, query=None):
        if query is None:
            query = self.doc_filter
            
        self.store.connect()
        results = list(self.store.query(query, properties=self.requried_fields))
        self.store.close()
        
        return results
    
    def query_one(self, query=None):
        if query is None:
            query = self.doc_filter
            
        self.store.connect()
        result = self.store.query_one(query, properties=self.required_fields)
        self.store.close()
        return result
        
    def __len__(self):
        return len(self.object_ids)
        
    def __getitem__(self, idx):
        doc = self.query_one({'_id': self.object_ids[idx]})
        feature = self.feature_processor.process_doc(doc)
        labels = self.label_processor.process_doc(doc)
        
        return torch.tensor(feature).float(), torch.tensor(labels).float()
    
    def get_random(self):
        _idx = np.random.choice(range(len(self.object_ids)))
        
        return self[_idx]


def split_data(dataset, batch_size, validation_split=0.15, test_split=0.15):
    """Construct a PyTorch data iterator."""
    validation_size = int(len(dataset) * validation_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - validation_size - test_size
    
    train, test, validation = torch.utils.data.random_split(dataset, [train_size, test_size, validation_size])

    validation_dataloader = DataLoader(validation, batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size, shuffle=False)
    train_dataloader = DataLoader(train, batch_size, shuffle=True)
    
    return validation_dataloader, test_dataloader, train_dataloader