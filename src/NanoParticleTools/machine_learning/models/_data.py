from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader
from typing import Union, Dict, List, Tuple, Optional, Any
from maggma.core import Store
import torch
import json
import numpy as np
from monty.json import MontyDecoder, MSONable
import os
import pytorch_lightning as pl
from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint
from scipy.ndimage import gaussian_filter1d
import hashlib
from monty.serialization import MontyEncoder
from torch_geometric.data import Data, HeteroData
import warnings
from abc import ABC, abstractmethod


class DataProcessor(ABC, MSONable):
    """
    Template for a data processor. The data processor allows modularity in definitions
    of how data is to be converted from a dictionary (typically a fireworks output document)
    to the desired form. This can be used for features or labels.

    To implementation a DataProcessor, override the process_doc function.

    Fields are specified to ensure they are present in documents
    """

    fields: List[str]

    def __init__(self, fields: List[str]):
        """
        :param fields: fields required in the document(s) to be processed
        """
        self.fields = fields

    @property
    def required_fields(self) -> List[str]:
        return self.fields

    @abstractmethod
    def process_doc(self, doc: Dict) -> Dict:
        raise NotImplementedError

    def get_item_from_doc(self, doc: dict, field: str):
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
            r_inner = constraints[idx - 1].radius
        r_outer = constraints[idx].radius
        return r_inner, r_outer

    @staticmethod
    def get_volume(r):
        return 4 / 3 * np.pi * (r**3)

    @property
    def is_graph(self):
        pass

    @property
    def data_cls(self):
        return Data


class FeatureProcessor(DataProcessor):

    def example(self):
        n_elements = len(self.possible_elements)
        dopant_specifications = [
            (0, 0.2,
             self.possible_elements[torch.randint(0, n_elements,
                                                  (1, )).item()], 'Y'),
            (0, 0.1,
             self.possible_elements[torch.randint(0, n_elements,
                                                  (1, )).item()], 'Y'),
            (1, 0.1,
             self.possible_elements[torch.randint(0, n_elements,
                                                  (1, )).item()], 'Y')
        ]
        dopant_concentration = []
        for layer_idx, dopant_conc, dopant, _ in dopant_specifications:
            while layer_idx + 1 > len(dopant_concentration):
                dopant_concentration.append({})
            dopant_concentration[layer_idx][dopant] = dopant_conc
        doc = {
            'dopant_concentration': dopant_concentration,
            'input': {
                'constraints':
                [SphericalConstraint(40),
                 SphericalConstraint(65)],
                'dopant_specifications': dopant_specifications
            }
        }
        return self.process_doc(doc)


class LabelProcessor(DataProcessor):

    def example(self):
        doc = {
            'output': {
                'wavelength_spectrum_x': list(self.x),
                'wavelength_spectrum_y': list(torch.rand_like(self.x))
            }
        }
        return self.process_doc(doc)


class EnergyLabelProcessor(LabelProcessor):
    """
    This Label processor returns a spectrum that is binned uniformly with respect to energy (I(E))
        Args:
            spectrum_range (Tuple | List, optional): Range over which the spectrum should be
                cropped. Defaults to (-40000, 20000).
            output_size (int, optional): Number of bins in the resultant spectra.
                This quantity will be used as the # of output does in the NN. Defaults to 600.
            log_constant (float, optional): When applying the log function,
                we use the form log_10(I+b). Since the intensity is always positive, this function
                is easily invertible. min_log_val sets the minimum value of the label after
                applying the log.

                To make sure the values aren't clipped, it is recommended that the smallest b is
                chosen at least 1 order of magnitude lower than 1/(# avg'd documents).

                Example:
                    With 16 documents averaged, the lowest (non-zero) observation is 0.0625(1/16),
                    therefore choose 0.001 as the log_constant. . Defaults to 1e-3.
            gaussian_filter (float, optional): Standard deviation over which to apply gaussian
                filtering to smooth the otherwise very peaked spectrum. Defaults to 0.
    """

    def __init__(self,
                 spectrum_range: Tuple | List = (-40000, 20000),
                 output_size: int = 600,
                 log_constant: float = 1e-3,
                 gaussian_filter: float = 0,
                 **kwargs):
        super().__init__(
            fields=['output.energy_spectrum_x', 'output.energy_spectrum_y'],
            **kwargs)

        self.spectrum_range = spectrum_range
        self.output_size = output_size
        self.log_constant = log_constant
        self.gaussian_filter = gaussian_filter

    def process_doc(self, doc: dict) -> torch.Tensor:
        x = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_x'))
        spectrum = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_y'))
        step = x[1] - x[0]

        # Assign to a different variable, so we can modify
        # the spectrum while keeping a reference to the original
        y = spectrum
        if self.gaussian_filter > 0:
            y = torch.tensor(gaussian_filter1d(y, self.gaussian_filter))

        target_range = torch.linspace(*self.spectrum_range,
                                      self.output_size + 1)
        target_step = target_range[1] - target_range[0]
        target_range = (target_range + target_step / 2)[:-1]
        if target_step == step and target_range[0] > x[0] and target_range[
                -1] < x[-1]:
            # The specified range is a subset of the original spectrum, we can just crop
            # the spectrum
            start = torch.nonzero(torch.isclose(x, target_range[0]))
            end = torch.nonzero(torch.isclose(x, target_range[-1]))

            x = x[start:(end + 1)]
            y = y[start:(end + 1)]
        elif x.shape[0] != self.output_size:
            if x.shape[0] >= self.output_size:
                warnings.warn(
                    "Desired spectrum resolution is coarser than found in the document."
                    " Spectrum will be rebinned approximately. It is recommended to rebuild"
                    " the collection to match the desired resolution")
                # We need to rebin the distribution
                multiplier = int(
                    torch.lcm(torch.tensor(x.size(0)),
                              torch.tensor(self.output_size)) / x.shape[0])

                _spectrum = y.expand(multiplier, -1).moveaxis(0, 1).reshape(
                    self.output_size, -1).sum(dim=-1)
                _spectrum = _spectrum * x.size(0) / (
                    multiplier * self.output_size
                )  # ensure integral is constant

                # Get the edges of the spectra
                lower_bound = x[0] - (step / 2)
                upper_bound = x[-1] + (step / 2)

                # Construct the new array
                _x = torch.linspace(lower_bound, upper_bound, self.output_size)

                # Replace the old spectrum with the new
                x = _x
                y = _spectrum
            else:
                raise RuntimeError(
                    "Spectrum in document is different than desired resolution and cannot"
                    " be rebinned. Please rebuild the collection")

        # Keep track of where the spectrum changes from
        # emission to absorption.
        idx_zero = torch.tensor(
            int(np.floor(0 - self.spectrum_range[0]) / step))

        # Count the total number of photons, we can add this to the loss
        n_photons_absorbed = torch.sum(spectrum[idx_zero:])
        n_photons_emitted = torch.sum(spectrum[:idx_zero])

        # Integrate the energy absorbed vs emitted.
        # This can be added to the loss to enforce conservation of energy
        total_energy = y * x
        e_absorbed = torch.sum(total_energy[idx_zero:])
        e_emitted = torch.sum(total_energy[:idx_zero])

        return {
            'spectra_x': x.unsqueeze(0),
            'y': y.float().unsqueeze(0),
            'log_y': torch.log10(y + self.log_constant).float().unsqueeze(0),
            'log_const': self.log_constant,
            'idx_zero': idx_zero,
            'n_absorbed': n_photons_absorbed,
            'n_emitted': n_photons_emitted,
            'e_absorbed': e_absorbed,
            'e_emitted': e_emitted
        }

    def __str__(self):
        return (f"Energy Label Processor - {self.output_size} bins, x_min ="
                " {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]},"
                " log_constant = {self.log_constant}")


class TotalEnergyLabelProcessor(LabelProcessor):
    """
    This Label processor returns a spectrum that is binned uniformly with respect to energy (I(E))
        Args:
            spectrum_range (Tuple | List, optional): Range over which the spectrum should be
                cropped. Defaults to (-40000, 20000).
            output_size (int, optional): Number of bins in the resultant spectra.
                This quantity will be used as the # of output does in the NN. Defaults to 600.
            log_constant (float, optional): When applying the log function,
                we use the form log_10(I+b). Since the intensity is always positive, this function
                is easily invertible. min_log_val sets the minimum value of the label after
                applying the log.

                To make sure the values aren't clipped, it is recommended that the smallest b is
                chosen at least 1 order of magnitude lower than 1/(# avg'd documents).

                Example:
                    With 16 documents averaged, the lowest (non-zero) observation is 0.0625(1/16),
                    therefore choose 0.001 as the log_constant. . Defaults to 1e-3.
            gaussian_filter (float, optional): Standard deviation over which to apply gaussian
                filtering to smooth the otherwise very peaked spectrum. Defaults to 0.
    """

    def __init__(self,
                 spectrum_range: Tuple | List = (-40000, 20000),
                 log_constant: float = 1e-3,
                 gaussian_filter: float = 0,
                 **kwargs):
        super().__init__(
            fields=['output.energy_spectrum_x', 'output.energy_spectrum_y'],
            **kwargs)

        self.spectrum_range = spectrum_range
        self.log_constant = log_constant
        self.gaussian_filter = gaussian_filter

    def process_doc(self, doc: dict) -> torch.Tensor:
        x = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_x'))
        spectrum = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_y'))
        step = x[1] - x[0]

        # Assign to a different variable, so we can modify
        # the spectrum while keeping a reference to the original
        y = spectrum
        if self.gaussian_filter > 0:
            y = torch.tensor(gaussian_filter1d(y, self.gaussian_filter))

        if self.spectrum_range[0] > x[0] and self.spectrum_range[-1] < x[-1]:
            # The specified range is a subset of the original spectrum, we can just crop
            # the spectrum
            start = torch.nonzero(x < self.spectrum_range[0])[-1][0]
            end = torch.nonzero(x > self.spectrum_range[-1])[0][0]
            x = x[start:(end + 1)]
            y = y[start:(end + 1)]

        # Keep track of where the spectrum changes from
        # emission to absorption.
        idx_zero = torch.tensor(
            int(np.floor(0 - self.spectrum_range[0]) / step))

        y_sum = y.sum()

        return {
            'spectra_x': x.unsqueeze(0),
            'y': y_sum.float().reshape(1, 1),
            'log_y':
            torch.log10(y_sum + self.log_constant).float().reshape(1, 1),
            'log_const': self.log_constant,
            'idx_zero': idx_zero
        }

    def __str__(self):
        return ("Energy Label Processor - 1 bins, x_min ="
                f" {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]},"
                f" log_constant = {self.log_constant}")


class WavelengthLabelProcessor(LabelProcessor):
    r"""
    This Label processor returns a spectrum that is binned uniformly with respect to
    wavelength $I(\lambda{})$
        Args:
            spectrum_range (Tuple | List, optional): Range over which the spectrum should be
                cropped. Defaults to (-1000, 1000).
            output_size (int, optional): Number of bins in the resultant spectra.
                This quantity will be used as the # of output does in the NN. Defaults to 600.
            log_constant (float, optional): When applying the log function,
                we use the form log_10(I+b). Since the intensity is always positive, this function
                is easily invertible. min_log_val sets the minimum value of the label after
                applying the log.

                To make sure the values aren't clipped, it is recommended that the smallest b is
                chosen at least 1 order of magnitude lower than 1/(# avg'd documents).

                Example:
                    With 16 documents averaged, the lowest (non-zero) observation is 0.0625(1/16),
                    therefore choose 0.001 as the log_constant. . Defaults to 1e-3.
            gaussian_filter (float, optional): Standard deviation over which to apply gaussian
                filtering to smooth the otherwise very peaked spectrum. Defaults to 0.
    """

    def __init__(self,
                 spectrum_range: Union[Tuple, List] = (-1000, 1000),
                 output_size: Optional[int] = 600,
                 log_constant: Optional[float] = 1e-3,
                 gaussian_filter: Optional[float] = None,
                 **kwargs):
        if gaussian_filter is None:
            gaussian_filter = 0

        super().__init__(fields=[
            'output.wavelength_spectrum_x', 'output.wavelength_spectrum_y',
            'output.summary', 'overall_dopant_concentration'
        ], **kwargs)

        self.spectrum_range = spectrum_range
        self.output_size = output_size
        self.log_constant = log_constant
        self.gaussian_filter = gaussian_filter

    def process_doc(self, doc: dict) -> torch.Tensor:
        x = torch.tensor(
            self.get_item_from_doc(doc, 'output.wavelength_spectrum_x'))
        spectrum = torch.tensor(
            self.get_item_from_doc(doc, 'output.wavelength_spectrum_y'))
        step = x[1] - x[0]

        if x.shape[0] != self.output_size:
            if x.shape[0] >= self.output_size:
                warnings.warn(
                    "Desired spectrum resolution is coarser than found in the document."
                    " Spectrum will be rebinned approximately. It is recommended to rebuild"
                    " the collection to match the desired resolution")
                # We need to rebin the distribution
                multiplier = int(
                    torch.lcm(torch.tensor(x.size(0)),
                              torch.tensor(self.output_size)) / x.shape[0])

                _spectrum = spectrum.expand(multiplier, -1).moveaxis(
                    0, 1).reshape(self.output_size, -1).sum(dim=-1)
                _spectrum = _spectrum * x.size(0) / (
                    multiplier * self.output_size
                )  # ensure integral is constant

                # Get the edges of the spectra
                lower_bound = x[0] - (step / 2)
                upper_bound = x[-1] + (step / 2)

                # Construct the new array
                _x = torch.linspace(lower_bound, upper_bound, self.output_size)

                # Replace the old spectrum with the new
                x = _x
                spectrum = _spectrum
            else:
                raise RuntimeError(
                    "Spectrum in document is different than desired resolution and"
                    " cannot be rebinned. Please rebuild the collection")

        # Assign to a different variable, so we can modify
        # the spectrum while keeping a reference to the original
        y = spectrum
        if self.gaussian_filter > 0:
            y = torch.tensor(gaussian_filter1d(y, self.gaussian_filter))

        # Keep track of where the spectrum changes from
        # emission to absorption.
        idx_zero = torch.tensor(
            int(np.floor(0 - self.spectrum_range[0]) / step))

        # Count the total number of photons, we can add this to the loss
        n_photons_absorbed = torch.sum(spectrum[idx_zero:])
        n_photons_emitted = torch.sum(spectrum[:idx_zero])

        # Integrate the energy absorbed vs emitted.
        # This can be added to the loss to enforce conservation of energy
        total_energy = spectrum * x
        e_absorbed = torch.sum(total_energy[idx_zero:])
        e_emitted = torch.sum(total_energy[:idx_zero])

        return {
            'spectra_x': x.unsqueeze(0),
            'y': y.float().unsqueeze(0),
            'log_y': torch.log10(y + self.log_constant).float().unsqueeze(0),
            'log_const': self.log_constant,
            'idx_zero': idx_zero,
            'n_absorbed': n_photons_absorbed,
            'n_emitted': n_photons_emitted,
            'e_absorbed': e_absorbed,
            'e_emitted': e_emitted
        }

    def __str__(self):
        return (
            f"Wavelength Label Processor - {self.output_size} bins, "
            f"x_min = {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]}, "
            f"log_constant = {self.log_constant}")


class NPMCDataset(Dataset):
    """
    NPMC dataset

    TODO: 1) Figure out a more elegant way to check if the data should be redownloaded
             (if the store has been updated)
          2) More elegant way to enforce size of dataset and redownload if the size is incorrect
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
            self.get_hash(self.feature_processor.__dict__))
        label_processor_match = self._check_hash(
            'label_processor', self.get_hash(self.label_processor.__dict__))

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
        if 'metadata' in doc:
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


class NPMCDataModule(pl.LightningDataModule):
    """_summary_

    Args:
        feature_processor (DataProcessor): _description_
        label_processor (DataProcessor): _description_
        training_data_store (Store): _description_
        testing_data_store (Optional[Store], optional): _description_. Defaults to None.
        training_doc_filter (Optional[dict], optional): _description_. Defaults to {}.
        testing_doc_filter (Optional[dict], optional): _description_. Defaults to {}.
        training_data_dir (Optional[str], optional): _description_. Defaults to './training_data'.
        testing_data_dir (Optional[str], optional): _description_. Defaults to './testing_data'.
        batch_size (Optional[int], optional): _description_. Defaults to 16.
        validation_split (Optional[float], optional): _description_. Defaults to 0.15.
        test_split (Optional[float], optional): _description_. Defaults to 0.15.
        training_size (Optional[int], optional): _description_. Defaults to None.
        testing_size (Optional[int], optional): _description_. Defaults to None.
        loader_workers (Optional[int], optional): _description_. Defaults to 0.
        use_cache (Optional[bool], optional): _description_. Defaults to False.
    """

    def __init__(self,
                 feature_processor: DataProcessor,
                 label_processor: DataProcessor,
                 training_data_store: Store,
                 testing_data_store: Optional[Store] = None,
                 training_doc_filter: Optional[dict] = {},
                 testing_doc_filter: Optional[dict] = {},
                 training_data_dir: Optional[str] = './training_data',
                 testing_data_dir: Optional[str] = './testing_data',
                 batch_size: Optional[int] = 16,
                 validation_split: Optional[float] = 0.15,
                 test_split: Optional[float] = 0.15,
                 training_size: Optional[int] = None,
                 testing_size: Optional[int] = None,
                 loader_workers: Optional[int] = 0,
                 use_cache: Optional[bool] = False,
                 calc_mean: bool = False):
        super().__init__()

        # We want to prepare the data on each node. That way each has access to the original data
        self.prepare_data_per_node = True

        if testing_data_store:
            test_split = 0

        self.feature_processor = feature_processor
        self.label_processor = label_processor
        self.training_doc_filter = training_doc_filter
        self.testing_doc_filter = testing_doc_filter
        self.training_data_store = training_data_store
        self.testing_data_store = testing_data_store
        self.training_data_dir = training_data_dir
        self.testing_data_dir = testing_data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.training_size = training_size
        self.testing_size = testing_size
        self.loader_workers = loader_workers
        self.use_cache = use_cache
        self.calc_mean = calc_mean
        # Initialize class variables that we will use later
        self.spectra_mean = None
        self.spectra_std = None

        self.save_hyperparameters()

    def get_training_dataset(self, download=False):
        return NPMCDataset(root=self.training_data_dir,
                           feature_processor=self.feature_processor,
                           label_processor=self.label_processor,
                           data_store=self.training_data_store,
                           doc_filter=self.training_doc_filter,
                           download=download,
                           overwrite=False,
                           dataset_size=self.training_size,
                           use_cache=self.use_cache,
                           use_metadata=False)

    def get_testing_dataset(self, download=False):
        if self.testing_data_store is not None:
            return NPMCDataset(root=self.testing_data_dir,
                               feature_processor=self.feature_processor,
                               label_processor=self.label_processor,
                               data_store=self.testing_data_store,
                               doc_filter=self.testing_doc_filter,
                               download=download,
                               overwrite=False,
                               dataset_size=self.testing_size,
                               use_cache=self.use_cache,
                               use_metadata=True)
        return None

    def prepare_data(self) -> None:
        """
        We only want to download the data in this method.
        Since this function is only called once in the main process, assigning state variables
        here will result in only the main process having access to those states (data).
        i.e. It should not contain the following:
        ```
        self.training_dataset = self.get_training_dataset()
        self.testing_dataset = self.get_testing_dataset()
        ```
        """

        self.get_training_dataset(download=True)
        self.get_testing_dataset(download=True)

    def setup(self, stage: Optional[str] = None):

        training_dataset = self.get_training_dataset(download=False)

        if self.testing_data_store is not None:
            # We have a separate testing data set, defined by its own store
            testing_dataset = self.get_testing_dataset(download=False)
            validation_size = int(
                len(training_dataset) * self.validation_split)
            train_size = len(training_dataset) - validation_size

            self.npmc_train, self.npmc_val = torch.utils.data.random_split(
                training_dataset, [train_size, validation_size],
                generator=torch.Generator().manual_seed(
                    torch.default_generator.seed()))
            self.npmc_test = testing_dataset
        else:
            # We don't have a testing dataset explicitly defined. We will split the training dataset
            # into the training, validation, and testing dataset
            test_size = int(len(training_dataset) * self.test_split)
            validation_size = int(
                len(training_dataset) * self.validation_split)
            train_size = len(training_dataset) - validation_size - test_size

            self.npmc_train, self.npmc_val, self.npmc_test = torch.utils.data.random_split(
                training_dataset, [train_size, validation_size, test_size],
                generator=torch.Generator().manual_seed(
                    torch.default_generator.seed()))

        if self.calc_mean:
            spectra = []
            # Leave out test data, since we aren't supposed to have knowledge of that in our model
            for data in self.npmc_train + self.npmc_test:
                spectra.append(data.log_y)
            spectra = torch.cat(spectra, dim=0)
            self.spectra_mean = spectra.mean(0)
            self.spectra_std = spectra.std(0)

    @staticmethod
    def collate(data_list: List[Data]):
        if len(data_list) == 0:
            return data_list[0]

        _data = {}
        for key in data_list[0].keys:
            if torch.is_tensor(getattr(data_list[0], key)):
                _data[key] = torch.stack(
                    [getattr(data, key) for data in data_list])
            else:
                _data[key] = [getattr(data, key) for data in data_list]

        _data['batch'] = torch.arange(len(data_list))

        return Data(**_data)

    def train_dataloader(self) -> DataLoader:
        if self.feature_processor.is_graph:
            # The data is graph structured
            return pyg_DataLoader(self.npmc_train,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.loader_workers,
                                  drop_last=True)
        else:
            # The data is in an image representation
            return DataLoader(self.npmc_train,
                              batch_size=self.batch_size,
                              collate_fn=self.collate,
                              shuffle=True,
                              num_workers=self.loader_workers,
                              drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self.feature_processor.is_graph:
            # The data is graph structured
            return pyg_DataLoader(self.npmc_val,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.loader_workers,
                                  drop_last=True)
        else:
            # The data is in an image representation
            return DataLoader(self.npmc_val,
                              batch_size=self.batch_size,
                              collate_fn=self.collate,
                              shuffle=False,
                              num_workers=self.loader_workers,
                              drop_last=True)

    def test_dataloader(self) -> DataLoader:
        if self.feature_processor.is_graph:
            # The data is graph structured
            return pyg_DataLoader(self.npmc_test,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.loader_workers,
                                  drop_last=True)
        else:
            # The data is in an image representation
            return DataLoader(self.npmc_test,
                              batch_size=self.batch_size,
                              collate_fn=self.collate,
                              shuffle=False,
                              num_workers=self.loader_workers,
                              drop_last=True)


class UCNPAugmenter():
    """
    This class defines functionality to augment UCNP/NPMC data.
    Augmentation is achieved by subdividing constraints, keeping the same dopant concentrations
    and output spectra.
    Args:
        random_seed (int, optional): Seed for random number generator.
            Used to ensure reproducibility.. Defaults to 1.
    """

    rng: int

    def __init__(self, random_seed: int = 1):
        self.rng = np.random.default_rng(random_seed)

    def augment_template(self,
                         constraints: List[NanoParticleConstraint],
                         dopant_specifications: List[Tuple[int, float, str,
                                                           str]],
                         n_augments: Optional[int] = 10) -> List[dict]:

        new_templates = []
        for i in range(n_augments):
            new_constraints, new_dopant_specification = self.generate_single_augment(
                constraints, dopant_specifications)
            new_templates.append({
                'constraints':
                new_constraints,
                'dopant_specifications':
                new_dopant_specification
            })
        return new_templates

    def generate_single_augment(
        self,
        constraints: List[NanoParticleConstraint],
        dopant_specifications: List[Tuple[int, float, str, str]],
        max_subdivisions: Optional[int] = 3,
        subdivision_increment=0.1
    ) -> Tuple[List[NanoParticleConstraint], List[Tuple[int, float, str,
                                                        str]]]:
        n_constraints = len(constraints)
        max_subdivisions = 3
        subdivision_increment = 0.1

        # Create a map of the dopant specifications
        dopant_specification_by_layer = {i: [] for i in range(n_constraints)}
        for _tuple in dopant_specifications:
            try:
                dopant_specification_by_layer[_tuple[0]].append(_tuple[1:])
            except KeyError:
                dopant_specification_by_layer[_tuple[0]] = [_tuple[1:]]

        n_constraints_to_divide = self.rng.integers(1, n_constraints + 1)
        constraints_to_subdivide = sorted(
            self.rng.choice(list(range(n_constraints)),
                            n_constraints_to_divide,
                            replace=False))

        new_constraints = []
        new_dopant_specification = []

        constraint_counter = 0
        for i in range(n_constraints):
            if i in constraints_to_subdivide:
                min_radius = 0 if i == 0 else constraints[i - 1].radius
                max_radius = constraints[i].radius

                # pick a number of subdivisions
                n_divisions = self.rng.integers(1, max_subdivisions)
                radii = sorted(
                    self.rng.choice(np.arange(min_radius, max_radius,
                                              subdivision_increment),
                                    n_divisions,
                                    replace=False))

                for r in radii:
                    new_constraints.append(SphericalConstraint(np.round(r, 1)))
                    try:
                        new_dopant_specification.extend([
                            (constraint_counter, *spec)
                            for spec in dopant_specification_by_layer[i]
                        ])
                    except Exception:
                        constraint_counter += 1
                        continue

                    constraint_counter += 1

            # Add the original constraint back to the list
            new_constraints.append(constraints[i])
            new_dopant_specification.extend([
                (constraint_counter, *spec)
                for spec in dopant_specification_by_layer[i]
            ])

            constraint_counter += 1
        return new_constraints, new_dopant_specification
