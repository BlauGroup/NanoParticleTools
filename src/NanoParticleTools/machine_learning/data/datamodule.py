from NanoParticleTools.machine_learning.data.dataset import NPMCDataset, Dataset
from NanoParticleTools.machine_learning.data.processors import DataProcessor

import numpy as np
import pytorch_lightning as pl
import torch
from maggma.core import Store
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader

from typing import List, Optional


def data_is_graph(dataset) -> bool:
    """
    Helper function to determine if a dataset is graph structured
    """
    if isinstance(dataset, torch.utils.data.Subset):
        return data_is_graph(dataset.dataset)
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        return data_is_graph(dataset.datasets[0])
    else:
        return dataset.feature_processor.is_graph


class NPMCDataModule(pl.LightningDataModule):

    def __init__(self,
                 train_dataset,
                 val_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None,
                 split_seed: int = None,
                 batch_size: int = 16,
                 loader_workers: int = 0,
                 **kwargs) -> None:
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.split_seed = split_seed
        self.data_is_graph = data_is_graph(train_dataset)
        self.batch_size = batch_size
        self.loader_workers = loader_workers

        self.save_hyperparameters(
            ignore=['train_dataset', 'val_dataset', 'test_dataset'])

    @property
    def persistent_workers(self):
        return self.loader_workers > 0

    @property
    def pin_memory(self):
        if torch.cuda.is_available():
            return True
        return False

    @classmethod
    def from_train_dataset(cls,
                           train_dataset,
                           val_split=0.15,
                           test_split=0.15,
                           split_seed: int = None,
                           **kwargs):
        """
        Construct a data module from a single dataset.
        The dataset will be split into training, validation,
        and testing according to the specified splits

        Args:
            train_dataset (_type_): _description_
            val_split (float, optional): _description_. Defaults to 0.15.
            test_split (float, optional): _description_. Defaults to 0.15.
            split_seed (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if split_seed is None:
            split_seed = np.random.randint(0, 100000)

        test_size = int(len(train_dataset) * test_split)
        validation_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - validation_size - test_size

        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            train_dataset, [train_size, validation_size, test_size],
            generator=torch.Generator().manual_seed(split_seed))

        return cls(train_subset, val_subset, test_subset, split_seed, **kwargs)

    @classmethod
    def from_train_and_test_dataset(cls,
                                    train_dataset,
                                    test_dataset,
                                    val_split=0.15,
                                    split_seed: int = None,
                                    **kwargs):
        if split_seed is None:
            split_seed = np.random.randint(0, 100000)

        # We have a separate testing data set, defined by its own store
        validation_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - validation_size

        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, validation_size],
            generator=torch.Generator().manual_seed(split_seed))

        return cls(train_subset, val_subset, test_dataset, split_seed,
                   **kwargs)

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
        if self.data_is_graph:
            # The data is graph structured
            return pyg_DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.loader_workers,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # The data is in an image representation
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                shuffle=True,
                num_workers=self.loader_workers,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self) -> DataLoader:
        if self.data_is_graph:
            # The data is graph structured
            return pyg_DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # The data is in an image representation
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
            )

    def test_dataloader(self) -> DataLoader:
        if self.data_is_graph:
            # The data is graph structured
            return pyg_DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # The data is in an image representation
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
            )
