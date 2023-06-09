from NanoParticleTools.machine_learning.data.processors import DataProcessor


import numpy as np
from maggma.core import Store
from monty.json import MontyDecoder
from monty.serialization import MontyEncoder
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


import hashlib
import json
import os
import warnings
from typing import Any, Dict


class NPMCDataset(Dataset):
    """
    NPMC dataset
    TODO: 1) Figure out a more elegant way to check if the data should be redownloaded
             (if the store has been updated)
          2) More elegant way to enforce size of dataset and redownload if the size is incorrect
          3) We are currently not caching the data to file. Should look into whether or not it
             is worth it to do so. There are some features already implemented to do this, such
             as saving the hash of the data processors and checking if they have changed.
    """

    def __init__(self,
                 root: str,
                 feature_processor: DataProcessor,
                 label_processor: DataProcessor,
                 data_store: Store,
                 doc_filter: dict = None,
                 download: bool = False,
                 overwrite: bool = False,
                 use_cache: bool = False,
                 dataset_size: int = None,
                 use_metadata: bool = False):
        """
        :param feature_processor:
        :param label_processor:
        """
        if doc_filter is None:
            doc_filter = {}

        self.root = root
        self.feature_processor = feature_processor
        self.label_processor = label_processor
        self.data_store = data_store
        self.doc_filter = doc_filter
        self.overwrite = overwrite
        self.use_cache = use_cache
        self.dataset_size = dataset_size
        self.use_metadata = use_metadata

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not downloaded")

        self.docs = self._load_data()

        if self.dataset_size == -1:
            warnings.warn(
                "dataset_size set to -1, no check was performed on the currently"
                " downloaded dataset. It may be out of date")
        elif self.dataset_size is None:
            self.data_store.connect()
            if len(self.docs) != self.data_store.count(self.doc_filter):
                warnings.warn(
                    "Length of dataset is not of the desired length. Automatically setting"
                    " 'overwrite=True' to redownload the data")
                self.overwrite = True
                self.download()
                self.docs = self._load_data()
        elif len(self.docs) != self.dataset_size:
            warnings.warn(
                "Length of dataset is not of the desired length. Automatically setting"
                " 'overwrite=True' to redownload the data")
            self.data_store.connect()
            self.overwrite = True
            self.download()
            self.docs = self._load_data()

        self.cached_data = [False for _ in self.docs]
        self.item_cache = [None for _ in self.docs]

    def _load_data(self):
        with open(os.path.join(self.raw_folder, 'data.json'), 'r') as f:
            docs = json.load(f, cls=MontyDecoder)
        return docs

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def hash_file(self) -> str:
        return os.path.join(self.processed_folder, 'hashes.json')

    @staticmethod
    def get_hash(dictionary: Dict[str, Any]) -> str:
        """
        MD5 hash of a dictionary.
        Adapted from: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
        """
        dhash = hashlib.md5()
        # We need to sort arguments so {'a': 1, 'b': 2} is
        # the same as {'b': 2, 'a': 1}
        encoded = json.dumps(dictionary, sort_keys=True,
                             cls=MontyEncoder).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    def _check_exists(self):
        if os.path.isfile(os.path.join(self.raw_folder, 'data.json')):
            return True
        else:
            return False

    def _check_processors(self):
        """
        We only redownload the data if the feature_processor or label_processor has changed
        If using a new data store or a new set of data, use the 'overwrite=True' arg
        """
        feature_processor_match = self._check_hash(
            'feature_processor',
            self.get_hash(self.feature_processor.as_dict()))
        label_processor_match = self._check_hash(
            'label_processor', self.get_hash(self.label_processor.as_dict()))

        return all(feature_processor_match, label_processor_match)

    def _check_hash(self, fname: str, hash: int):
        if not os.path.isfile(self.hash_file):
            return False

        with open(self.hash_file, 'r') as f:
            hashes = json.load(f)

        return hashes[fname] == hash

    def log_processors(self):
        _d = {}
        _d['feature_processor'] = self.get_hash(
            self.feature_processor.__dict__)
        _d['label_processor'] = self.get_hash(self.label_processor.__dict__)
        with open(os.path.join(self.raw_folder, 'hashes.json'), 'w') as f:
            json.dump(_d, f)

    def download(self):
        required_fields = self.feature_processor.required_fields \
            + self.label_processor.required_fields

        if self.use_metadata:
            required_fields += ['metadata']

        if 'input' not in required_fields:
            required_fields.append('input')

        if self._check_exists() and not self.overwrite:
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.raw_folder, exist_ok=True)

        # Download the data
        with self.data_store:
            documents = list(
                self.data_store.query(self.doc_filter,
                                      properties=required_fields))

        if self.dataset_size is not None:
            # Choose a subset of the total documents
            documents = list(
                np.random.choice(documents,
                                 size=min(len(documents), self.dataset_size),
                                 replace=False))

        # Write the data to the raw directory
        with open(os.path.join(self.raw_folder, 'data.json'), 'w') as f:
            json.dump(documents, f, cls=MontyEncoder)

        # Log the processor hashes
        self.log_processors()

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

    @classmethod
    def collate_fn(cls):
        raise NotImplementedError("Must override collate_fn")

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        if self.use_cache:
            # Check if this index is cached
            if self.cached_data[idx]:
                # Retrieve cached item from memory
                data = self.item_cache[idx]
            else:
                # generate the point
                data = self.process_single_doc(idx)

                self.cached_data[idx] = True
                self.item_cache[idx] = data
                self.docs[
                    idx] = None  # We have cached the output of this document, free up memory
        else:
            data = self.process_single_doc(idx)
        return data

    def get_random(self):
        _idx = np.random.choice(range(len(self)))

        return self[_idx]
