from NanoParticleTools.machine_learning.data.processors import (
    DataProcessor, FeatureProcessor, LabelProcessor)

import numpy as np
from maggma.core import Store
from monty.json import MontyDecoder
from monty.serialization import MontyEncoder
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

import json
import os
from typing import Any, Dict
import warnings

class NPMCDataset(Dataset):

    def __init__(self,
                 file_path: str,
                 feature_processor: DataProcessor,
                 label_processor: DataProcessor,
                 use_metadata: bool = False,
                 cache_in_memory: bool = True):

        self.file_path = file_path
        self.feature_processor = feature_processor
        self.label_processor = label_processor
        self.use_metadata = use_metadata
        self.cache_in_memory = cache_in_memory

        self._docs = None
        self._cached_data = None

    @property
    def docs(self):
        if self._docs is None:
            self._docs = self._load_data()
        return self._docs

    @property
    def cached_data(self):
        if self._cached_data is None:
            self._cached_data = [None for _ in self.docs]
        return self._cached_data

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            docs = json.load(f, cls=MontyDecoder)
        return docs

    def __len__(self):
        return len(self.docs)

    @classmethod
    def collate_fn(cls):
        raise NotImplementedError("Must override collate_fn")

    def process_single_doc(self, idx: int):
        """
        Processes a single document and produces datapoint
        """

        doc = self.docs[idx]
        _d = self.feature_processor.process_doc(doc)
        _d.update(self.label_processor.process_doc(doc))

        _d['constraints'] = doc['input']['constraints']
        _d['dopant_specifications'] = doc['input']['dopant_specifications']
        if 'metadata' in doc and self.use_metadata:
            _d['metadata'] = doc['metadata']
        if issubclass(self.feature_processor.data_cls, HeteroData):
            return self.feature_processor.data_cls(_d)
        else:
            return self.feature_processor.data_cls(**_d)

    def __getitem__(self, idx):
        if self.cache_in_memory:
            # Check if this index is cached
            if self.cached_data[idx] is not None:
                # Retrieve cached item from memory
                data = self.cached_data[idx]
            else:
                # generate the point
                data = self.process_single_doc(idx)

                self.cached_data[idx] = data

                # We have cached the output of this document, free up memory
                self.docs[idx] = None
        else:
            data = self.process_single_doc(idx)
        return data

    def get_random(self):
        _idx = np.random.choice(range(len(self)))

        return self[_idx]

    @classmethod
    def from_store(cls,
                   feature_processor: DataProcessor,
                   label_processor: DataProcessor,
                   data_store: Store,
                   doc_filter: dict = None,
                   overwrite: bool = False,
                   use_metadata: bool = False,
                   file_path: str = './dataset.json',
                   cache_in_memory: bool = True):
        # Check if the file exists
        if os.path.isfile(file_path):
            if overwrite:
                download(feature_processor, label_processor, data_store,
                         doc_filter, file_path, use_metadata)
            else:
                warnings.warn('File already exists. Skipping download. Please'
                              ' double check that the dataset is the correct one,'
                              ' else, set overwrite to true')
        else:
            # Download the data from the store and write to file
            download(feature_processor, label_processor, data_store,
                     doc_filter, file_path, use_metadata)

        return cls(file_path, feature_processor, label_processor, use_metadata,
                   cache_in_memory)


def get_required_fields(feature_processor: FeatureProcessor,
                        label_processor: LabelProcessor,
                        use_metadata: bool = False):
    required_fields = feature_processor.required_fields \
        + label_processor.required_fields

    if use_metadata:
        required_fields += ['metadata']

    if 'input' not in required_fields:
        required_fields.append('input')
    return required_fields


def download(feature_processor: FeatureProcessor,
             label_processor: LabelProcessor,
             data_store: Store,
             doc_filter: dict,
             file_path: str = './dataset.json',
             use_metadata: bool = False):
    required_fields = get_required_fields(feature_processor, label_processor,
                                          use_metadata)

    # Download the data
    data_store.connect()
    documents = list(data_store.query(doc_filter, properties=required_fields))
    data_store.close()

    # Write the data to the raw directory
    if file_path.split('.')[-1] == 'json':

        with open(file_path, 'w') as f:
            json.dump(documents, f, cls=MontyEncoder)
    else:
        raise NotImplementedError("Only json files are currently supported")
        # Create the hdf5 file
        # This route needs some exploration. The documents don't play nicely with h5py,
        # Therefore, we'd need to think about preprocessing and storing as tensors/np arrays
        # with h5py.File(file, 'w') as f:
        #     f['required_fields'] = required_fields
        #     data = f.create_group('data')
        #     for i, doc in enumerate(documents):
        #         group = data.create_group(f'{i}')
        #         group["doc"] = json.dumps(doc, cls=MontyEncoder)
