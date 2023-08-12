from NanoParticleTools.machine_learning.data.dataset import NPMCDataset
from NanoParticleTools.machine_learning.data.processors import DataProcessor

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader

from typing import List, Optional


def get_processors(dataset: NPMCDataset) -> List[DataProcessor]:
    """
    A Helper function to get the data processors from a dataset.

    This is helpful when using Subsets or Concatenated datasets, in
    which the feature_processor is not directly accessible.

    Note: This function assumes that all sub-datasets in a ConcatDataset
        or Subset have the same feature_processor
    """
    if isinstance(dataset, torch.utils.data.Subset):
        return get_processors(dataset.dataset)
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        return get_processors(dataset.datasets[0])
    else:
        return dataset.feature_processor, dataset.label_processor


def data_is_graph(dataset) -> bool:
    """
    Helper function to determine if a dataset is graph structured.
    """
    feature_processor, _ = get_processors(dataset)
    return feature_processor.is_graph


class NPMCDataModule(pl.LightningDataModule):

    def __init__(self,
                 train_dataset,
                 val_dataset: Optional[NPMCDataset] = None,
                 test_dataset: Optional[NPMCDataset] = None,
                 iid_test_dataset: Optional[NPMCDataset] = None,
                 split_seed: int = None,
                 batch_size: int = 16,
                 loader_workers: int = 0,
                 **kwargs) -> None:
        """
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Out of distribution test dataset
            iid_test_dataset: In distribution test dataset
            split_seed: The seed used for random splitting of the data.
                Note, this is not actually used, it is just saved for bookkeeping.
            batch_size (int, optional): _description_. Defaults to 16.
            loader_workers (int, optional): _description_. Defaults to 0.
        """
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.iid_test_dataset = iid_test_dataset

        self.split_seed = split_seed
        self.data_is_graph = data_is_graph(train_dataset)
        self.batch_size = batch_size
        self.loader_workers = loader_workers
        self.feature_processor, self.label_processor = get_processors(train_dataset)

        self.save_hyperparameters(
            ignore=['train_dataset', 'val_dataset', 'test_dataset', 'iid_test_dataset'])

    @property
    def persistent_workers(self):
        return self.loader_workers > 0

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
            )

    def test_dataloader(self) -> DataLoader:
        if self.data_is_graph:
            # The data is graph structured
            return pyg_DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=False,
                persistent_workers=self.persistent_workers,
            )
        else:
            # The data is in an image representation
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=False,
                persistent_workers=self.persistent_workers,
            )

    def iid_test_dataloader(self) -> DataLoader:
        if self.data_is_graph:
            # The data is graph structured
            return pyg_DataLoader(
                self.iid_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=False,
                persistent_workers=self.persistent_workers,
            )
        else:
            # The data is in an image representation
            return DataLoader(
                self.iid_test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate,
                shuffle=False,
                num_workers=self.loader_workers,
                drop_last=False,
                persistent_workers=self.persistent_workers,
            )
