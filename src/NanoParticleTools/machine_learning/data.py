from torch.utils.data import Dataset, DataLoader
from monty.json import MontyEncoder, MontyDecoder
from typing import Union, Dict, List, Tuple, Optional
from maggma.core import Store
from functools import lru_cache
import torch
import json
import numpy as np
import warnings
import os
import pytorch_lightning as pl

class DataProcessor():
    """
    Template for a data processor. The data processor allows modularity in definitions
    of how data is to be converted from a dictionary (typically a fireworks output document)
    to the desired form. This can be used for features or labels.

    To implementation a DataProcessor, override the process_docs function.

    Fields are specified to ensure they are present in documents
    """
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
        """
        :param max_layers: 
        :param possible_elements:
        """
        
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
        
        super().__init__(**kwargs)
        
        self.max_layers = max_layers
        self.possible_elements = possible_elements
        
    def process_doc(self, 
                    doc: dict) -> List:
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
        return np.hstack(feature)
    
    def __str__(self):
        return f"Feature Processor - {self.max_layers} x [radius, volume, x_{', x_'.join(self.possible_elements)}]"
    
class LabelProcessor(DataProcessor):
    def __init__(self, 
                 spectrum_range: Union[Tuple, List] = (-1000, 0),
                 output_size = 500,
                 log: bool = False,
                 normalize: bool = False,
                 **kwargs):
        """
        :param spectrum_range: Range over which the spectrum should be cropped
        :param output_size: Number of bins in the resultant spectra. This quantity will be used as the # of output does in the NN
        :param log: Apply the log function to the data. This will scale data which spans multiple orders of magnitude.
        :param normalize: Normalize the integrated area of the spectrum to 1
        """
        
        super().__init__(**kwargs)
        
        self.spectrum_range = spectrum_range
        self.output_size = output_size
        self.log = log
        self.normalize = normalize
    
    @property
    def x(self):
        """
        Returns the x bins for the processed data.

        Typically used to plot the spectrum
        """
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
            coarsened_spectrum = np.log10(coarsened_spectrum + 1)
        if self.normalize:
            coarsened_spectrum = coarsened_spectrum/np.sum(coarsened_spectrum)
        
        return list(coarsened_spectrum)
    
    def __str__(self):
        return f"Label Processor - {self.output_size} bins, x_min = {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]}, log = {self.log}, normalize = {self.normalize}"
        

class NPMCDataset(Dataset):
    """
    
    """
    def __init__(self, 
                 features: torch.tensor,
                 labels: torch.tensor, 
                 feature_processor: DataProcessor,
                 label_processor: DataProcessor):
        """
        :param features: A Pytorch tensor of the feature data. Axis 0 should correspond to separate data points
        :param labels: A Pytorch tensor of the label data. Axis 0 should correspond to separate data points
        :param feature_processor:
        :param label_processor:
        """
        self.features = features
        self.labels = labels
        self.feature_processor = feature_processor
        self.label_processor = label_processor
    
    @staticmethod
    def process_docs(docs, 
                     feature_processor: DataProcessor,
                     label_processor: DataProcessor):
        """
        :param feature_processor:
        :param label_processor:
        """
        features = []
        labels = []
        for doc in docs:
            features.append(feature_processor.process_doc(doc))
            labels.append(label_processor.process_doc(doc))
            
        features = torch.tensor(np.vstack(features))
        labels = torch.tensor(np.vstack(labels))
        
        return features.float(), labels.float()
    
    @classmethod
    def from_file(cls, 
                  feature_processor: DataProcessor,
                  label_processor: DataProcessor,
                  doc_file='npmc_data.json'):
        
        # TODO: check if all the required fields are in the docs
        with open(doc_file, 'r') as f:
            documents = json.load(f, cls=MontyDecoder)
            
        features, labels = cls.process_docs(documents, feature_processor, label_processor)
        
        return cls(features, labels, feature_processor, label_processor)
        
    @classmethod
    def from_store(cls, 
                   store: Store,
                   feature_processor: DataProcessor,
                   label_processor: DataProcessor,
                   doc_filter: Optional[Dict] = {}, 
                   cache_doc_file: Optional[str] = 'npmc_data.json',
                   override: Optional[bool] = False):
        
        required_fields = feature_processor.required_fields + label_processor.required_fields
        
        #query for all documents
        store.connect()
        documents = list(store.query(doc_filter, properties=required_fields))
        store.close()
        
        if not override and os.path.exists(cache_doc_file):
            warnings.warn(f'Existing data file {cache_doc_file} found, will not override. Rerun using the "override=True" argument to overwrite existing file')
        else:
            with open(cache_doc_file, 'w') as f:
                json.dump(documents, f, cls=MontyEncoder)
        
        features, labels = cls.process_docs(documents, feature_processor, label_processor)
        
        return cls(features, labels, feature_processor, label_processor)

    def __len__(self):
        return self.features.size()[0]
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_random(self):
        _idx = np.random.choice(range(len(self)))
        
        return self[_idx]

class NPMCDataModule(pl.LightningDataModule):
    def __init__(self,
                 feature_processor: DataProcessor,
                 label_processor: DataProcessor,
                 data_store = None, 
                 data_dir = None,
                 batch_size: int = 16, 
                 validation_split: Union[int, float] = 0.15,
                 test_split: Union[int, float] = 0.15,
                 random_split_seed = 0):
        super().__init__()
        
        self.data_store = data_store
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_split_seed = random_split_seed

        self.save_hyperparameters()

        if data_store is None and data_dir is None:
            raise ValueError("data_store or data_dir must be specified")
        
        self.feature_processor = feature_processor
        self.label_processor = label_processor
    
    def setup(self, 
              stage: Optional[str] = None):
        dataset = None    
        if self.data_store is not None:
            dataset = NPMCDataset.from_store(store=self.data_store,
                                             feature_processor = self.feature_processor,
                                             label_processor = self.label_processor)
        elif self.data_dir is not None:
            dataset = NPMCDataset.from_file(doc_file = self.data_dir,
                                            feature_processor = self.feature_processor,
                                            label_processor = self.label_processor)

        if isinstance(self.test_split, float):
            test_size = int(len(dataset) * self.test_split)
        else:
            test_size = self.test_split

        remaining_size = len(dataset) - test_size
        if isinstance(self.validation_split, float):
            validation_size = int(remaining_size * self.validation_split)
        else:
            if self.validation_split >= remaining_size / 2:
                warnings.warn('Warning: Validation size is larger than training set')
            validation_size = self.validation_split

        train_size = remaining_size - validation_size


        self.npmc_train, self.npmc_val, self.npmc_test = torch.utils.data.random_split(dataset, 
                                                                                       [train_size, test_size, validation_size],
                                                                                       generator = torch.Generator().manual_seed(self.random_split_seed))

    def train_dataloader(self):
        return DataLoader(self.npmc_train, self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.npmc_val, self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.npmc_test, self.batch_size, shuffle=False)


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
