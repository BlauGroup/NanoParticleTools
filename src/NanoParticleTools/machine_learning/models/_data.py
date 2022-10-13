from pydoc import doc
from torch.utils.data import Dataset, DataLoader
from typing import Union, Dict, List, Tuple, Optional, Type
from maggma.core import Store
from functools import lru_cache
import torch
import json
import numpy as np
from monty.json import MontyDecoder
import os
import pytorch_lightning as pl
from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint
from scipy.ndimage import gaussian_filter1d

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

    @staticmethod
    def get_radii(idx, constraints):
        if idx == 0:
            # The constraint was the first one, therefore the inner radius is 0
            r_inner = 0
        else:
            r_inner = constraints[idx-1].radius
        r_outer = constraints[idx].radius
        return r_inner, r_outer
        
    @staticmethod
    def get_volume(r):
        return 4/3*np.pi*(r**3)
    

class LabelProcessor(DataProcessor):
    def __init__(self, 
                 spectrum_range: Union[Tuple, List] = (-1000, 0),
                 output_size: Optional[int] = 500,
                 log: Optional[bool] = False,
                 normalize: Optional[bool] = False,
                 gaussian_filter: Optional[float] = 0,
                 **kwargs):
        """
        :param spectrum_range: Range over which the spectrum should be cropped
        :param output_size: Number of bins in the resultant spectra. This quantity will be used as the # of output does in the NN
        :param log: Apply the log function to the data. This will scale data which spans multiple orders of magnitude.
        :param normalize: Normalize the integrated area of the spectrum to 1
        """
        
        super().__init__(fields=['output.spectrum_x', 'output.spectrum_y'], **kwargs)
        
        self.spectrum_range = spectrum_range
        self.output_size = output_size
        self.log = log
        self.normalize = normalize
        self.gaussian_filter = gaussian_filter
    
    @property
    def x(self) -> np.array:
        """
        Returns the x bins for the processed data.

        Typically used to plot the spectrum
        """
        _x = np.linspace(*self.spectrum_range, self.output_size+1)
        
        return (_x[:-1]+_x[1:])/2

    def process_doc(self, 
                doc: dict) -> torch.Tensor:
        spectrum_x = torch.tensor(self.get_item_from_doc(doc, 'output.spectrum_x'))
        spectrum_y = torch.tensor(self.get_item_from_doc(doc, 'output.spectrum_y'))

        min_i = torch.argmin(torch.abs(torch.subtract(spectrum_x, self.spectrum_range[0])))
        max_i = torch.argmin(torch.abs(torch.subtract(spectrum_x, self.spectrum_range[1])))
        if self.spectrum_range[1] > spectrum_x[-1]:
            max_i = len(spectrum_x)
        
        x = spectrum_x[min_i:max_i]
        y = spectrum_y[min_i:max_i]
        
        spectrum = torch.sum(torch.reshape(y, (self.output_size, -1)), axis=1)
        if self.gaussian_filter > 0:
            spectrum = torch.tensor(gaussian_filter1d(spectrum, self.gaussian_filter))
        if self.log:
            spectrum = torch.log10(spectrum + 1)
        if self.normalize:
            spectrum = spectrum/torch.sum(spectrum)
        
        return spectrum.float()
    
    def __str__(self):
        return f"Label Processor - {self.output_size} bins, x_min = {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]}, log = {self.log}, normalize = {self.normalize}"
        

class BaseNPMCDataset(Dataset):
    """
    Base class for a NPMC dataset
    """
    def __init__(self, 
                 data_list: List,
                 feature_processor: DataProcessor,
                 label_processor: DataProcessor):
        """
        :param features: A Pytorch tensor of the feature data. Axis 0 should correspond to separate data points
        :param labels: A Pytorch tensor of the label data. Axis 0 should correspond to separate data points
        :param feature_processor:
        :param label_processor:
        """
        self.data_list = data_list
        self.feature_processor = feature_processor
        self.label_processor = label_processor

    @staticmethod
    def process_single_doc(doc, 
                           feature_processor: DataProcessor,
                           label_processor: DataProcessor):
        """
        Processes a single document and produces datapoint
        """
        raise NotImplementedError("Must override process_single_doc")

    @classmethod
    def collate_fn(cls):
        raise NotImplementedError("Must override collate_fn")

    @classmethod
    def process_docs(cls,
                     docs, 
                     **kwargs) -> List:
        """
        :param feature_processor:
        :param label_processor:
        """
        data_list = []
        for doc in docs:
            data_list.append(cls.process_single_doc(doc, **kwargs))
        return data_list

    @classmethod
    def from_file(cls, 
                  feature_processor: DataProcessor,
                  label_processor: DataProcessor,
                  doc_file: Optional[str] ='npmc_data.json'):

        if os.path.exists(doc_file) == False:
            raise ValueError('File {doc_file} does not exist')

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
        
        data_list = cls.process_docs(documents, feature_processor=feature_processor, label_processor=label_processor)
        
        return cls(data_list, feature_processor, label_processor)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def get_random(self):
        _idx = np.random.choice(range(len(self)))
        
        return self[_idx]


class NPMCDataModule(pl.LightningDataModule):
    def __init__(self,
                 feature_processor: DataProcessor,
                 label_processor: DataProcessor,
                 dataset_class: Type[Dataset] = BaseNPMCDataset,
                 doc_filter: Optional[dict] = None,
                 training_data_store: Optional[Store] = None, 
                 testing_data_store: Optional[Store] = None,
                 batch_size: Optional[int] = 16, 
                 validation_split: Optional[float] = 0.15,
                 test_split: Optional[float] = 0.15,
                 random_split_seed = 0):
        """
        If a 

        :param feature_processor: 
        :param label_processor: 
        :param dataset_class: 
        :param doc_filter: Query to use for documents
        :param training_data_store:
        :param testing_data_store: 
        :param data_dir: 
        :param batch_size: 
        :param validation_split: 
        :param test_split: 
        :param random_split_seed: Use a seed for the random splitting, to ensure reproducibility
        """
        super().__init__()
        if testing_data_store:
            test_split = 0

        self.feature_processor = feature_processor
        self.label_processor = label_processor
        self.dataset_class = dataset_class
        self.doc_filter = doc_filter
        self.training_data_store = training_data_store
        self.testing_data_store = testing_data_store
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_split_seed = random_split_seed

        self.save_hyperparameters()
    
    def setup(self, 
              stage: Optional[str] = None):
        training_dataset = self.dataset_class.from_store(store=self.training_data_store,
                                            doc_filter=self.doc_filter,
                                            feature_processor = self.feature_processor,
                                            label_processor = self.label_processor)


        if self.testing_data_store:
            # Initialize the testing set from the store

            testing_dataset = self.dataset_class.from_store(store=self.testing_data_store,
                                            doc_filter=self.doc_filter,
                                            feature_processor = self.feature_processor,
                                            label_processor = self.label_processor)

            # Split the training data in to a test and validation set
            
            validation_size = int(len(training_dataset) * self.validation_split)
            train_size = len(training_dataset) - validation_size
            self.npmc_train, self.npmc_val = torch.utils.data.random_split(training_dataset, 
                                                                           [train_size, validation_size],
                                                                           generator = torch.Generator().manual_seed(self.random_split_seed))
            self.npmc_test = testing_dataset
        else:
            test_size = int(len(training_dataset) * self.test_split)
            validation_size = int(len(training_dataset) * self.validation_split)
            train_size = len(training_dataset) - validation_size - test_size
            self.npmc_train, self.npmc_val, self.npmc_test = torch.utils.data.random_split(training_dataset, 
                                                                                        [train_size, validation_size, test_size],
                                                                                        generator = torch.Generator().manual_seed(self.random_split_seed))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_train, self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_val, self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_test, self.batch_size, shuffle=False)


class UCNPAugmenter():
    """
    This class defines functionality to augment UCNP/NPMC data. 
    Augmentation is achieved by subdividing constraints, keeping the same dopant concentrations and output spectra.
    """
    def __init__(self, 
                 random_seed : Optional[int] = 1):
        """
        :param random_seed: Seed for random number generator. Used to ensure reproducibility.
        """
        self.rng = np.random.default_rng(random_seed)
        
    def augment_template(self,
                         constraints: List[NanoParticleConstraint], 
                         dopant_specifications: List[Tuple[int, float, str, str]], 
                         n_augments: Optional[int] = 10) -> List[dict]:
                        
        new_templates = []
        for i in range(n_augments):
            new_constraints, new_dopant_specification = self.generate_single_augment(constraints, dopant_specifications)
            new_templates.append({'constraints': new_constraints,
                                  'dopant_specifications': new_dopant_specification})
        return new_templates

    def generate_single_augment(self, 
                                constraints: List[NanoParticleConstraint], 
                                dopant_specifications: List[Tuple[int, float, str, str]],
                                max_subdivisions: Optional[int] = 3,
                                subdivision_increment = 0.1) -> Tuple[List[NanoParticleConstraint], List[Tuple[int, float, str, str]]]:        
        n_constraints = len(constraints) 
        max_subdivisions = 3
        subdivision_increment = 0.1

        # Create a map of the dopant specifications
        dopant_specification_by_layer = {i:[] for i in range(n_constraints)}
        for _tuple in dopant_specifications:
            try:
                dopant_specification_by_layer[_tuple[0]].append(_tuple[1:])
            except:
                dopant_specification_by_layer[_tuple[0]] = [_tuple[1:]]

        n_constraints_to_divide = self.rng.integers(1, n_constraints+1)
        constraints_to_subdivide = sorted(self.rng.choice(list(range(n_constraints)), n_constraints_to_divide, replace=False))

        new_constraints = []
        new_dopant_specification = []

        constraint_counter = 0
        for i in range(n_constraints):
            if i in constraints_to_subdivide:
                min_radius = 0 if i == 0 else constraints[i-1].radius
                max_radius = constraints[i].radius

                #pick a number of subdivisions
                n_divisions = self.rng.integers(1, max_subdivisions)
                radii = sorted(self.rng.choice(np.arange(min_radius, max_radius, subdivision_increment), n_divisions, replace=False))
                
                for r in radii:
                    new_constraints.append(SphericalConstraint(np.round(r, 1)))
                    try:
                        new_dopant_specification.extend([(constraint_counter, *spec) for spec in dopant_specification_by_layer[i]])
                    except:
                        constraint_counter+=1
                        continue

                    constraint_counter+=1
                    
            # Add the original constraint back to the list
            new_constraints.append(constraints[i])
            new_dopant_specification.extend([(constraint_counter, *spec) for spec in dopant_specification_by_layer[i]])

            constraint_counter+=1
        return new_constraints, new_dopant_specification
