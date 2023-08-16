from NanoParticleTools.machine_learning.data import (
    NPMCDataModule, NPMCDataset, WavelengthSpectrumLabelProcessor)
from NanoParticleTools.machine_learning.models.mlp_model.data import TabularFeatureProcessor
import pytest
from pathlib import Path
import os
import json
from maggma.stores import MemoryStore
from monty.serialization import MontyDecoder

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / '..' / '..' / 'test_files'


@pytest.fixture
def label_processor():
    return WavelengthSpectrumLabelProcessor(log_constant=1)


@pytest.fixture
def feature_processor():
    return TabularFeatureProcessor(max_layers = 3, possible_elements=['Yb', 'Nd', 'Mg'])


@pytest.fixture
def graph_feature_processor():
    return TabularFeatureProcessor(max_layers = 3, possible_elements=['Yb', 'Nd', 'Mg'])


@pytest.fixture
def train_dataset(label_processor, feature_processor):
    return NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                       feature_processor=feature_processor,
                       label_processor=label_processor,
                       use_metadata=False,
                       cache_in_memory=True)


@pytest.fixture
def test_dataset():
    return NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                       feature_processor=feature_processor,
                       label_processor=label_processor,
                       use_metadata=False,
                       cache_in_memory=True)


def test_datamodule_init(train_dataset, test_dataset):
    datamodule = NPMCDataModule.from_train_and_test_dataset(train_dataset,
                                                            test_dataset,
                                                            val_split=0.15,
                                                            batch_size=16,
                                                            split_seed=0)

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert datamodule.batch_size == 16
    assert datamodule.split_seed == 0

    assert len(datamodule.train_dataset) == 14
    assert len(datamodule.val_dataset) == 2
    assert len(datamodule.test_dataset) == 16


def test_datamodule_no_test_set(label_processor, feature_processor,
                                train_dataset, test_dataset):
    datamodule = NPMCDataModule.from_train_dataset(train_dataset,
                                                   val_split=0.15,
                                                   test_split=0.15,
                                                   split_seed=0,
                                                   batch_size=16)

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert datamodule.batch_size == 16
    assert datamodule.split_seed == 0

    assert len(datamodule.train_dataset) == 12
    assert len(datamodule.val_dataset) == 2
    assert len(datamodule.test_dataset) == 2


def test_pyg_dataloader(train_dataset):
    datamodule = NPMCDataModule.from_train_dataset(train_dataset,
                                                   val_split=0.15,
                                                   test_split=0.15,
                                                   split_seed=0,
                                                   batch_size=2)

    train_dataloader = datamodule.train_dataloader()
    assert len(train_dataloader) == 6
    for batch in train_dataloader:
        assert batch.y.shape == (2, 600)
        assert batch.batch[-1] == 1

    val_dataloader = datamodule.val_dataloader()
    assert len(val_dataloader) == 1
    for batch in val_dataloader:
        assert batch.y.shape == (2, 600)
        assert batch.batch[-1] == 1

    test_dataloader = datamodule.test_dataloader()
    assert len(test_dataloader) == 1
    for batch in test_dataloader:
        assert batch.y.shape == (2, 600)
        assert batch.batch[-1] == 1


def test_torch_dataloader():
    # TODO: Implement after checking if the torch dataloader is necessary
    # It seems that most feature processors set is_graph to True, even if
    # they are not graphs...
    pass
