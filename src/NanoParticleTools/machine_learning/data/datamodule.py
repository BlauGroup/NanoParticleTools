from NanoParticleTools.machine_learning.data.dataset import NPMCDataset
from NanoParticleTools.machine_learning.data.processors import DataProcessor


import numpy as np
import pytorch_lightning as pl
import torch
from maggma.core import Store
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader


from typing import List, Optional


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
                 calc_mean: bool = False,
                 split_seed: int = None):
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
        if split_seed is None:
            # If no split seed is provided, generate one
            split_seed = np.random.randint(0, 100000)
        self.split_seed = split_seed
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
                               use_metadata=False)
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
                generator=torch.Generator().manual_seed(self.split_seed))
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
                generator=torch.Generator().manual_seed(self.split_seed))

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
