from NanoParticleTools.machine_learning.data.datamodule import NPMCDataModule
from NanoParticleTools.machine_learning.data.processors import WavelengthSpectrumLabelProcessor
from NanoParticleTools.machine_learning.models.mlp_model.data import VolumeFeatureProcessor
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
    return VolumeFeatureProcessor(3, possible_elements=['Yb', 'Nd', 'Mg'])


@pytest.fixture
def graph_feature_processor():
    return VolumeFeatureProcessor(3, possible_elements=['Yb', 'Nd', 'Mg'])


@pytest.fixture
def train_store():
    store = MemoryStore()
    store.connect()

    # Add data from the test file to the store
    with open(TEST_FILE_DIR / 'dataset/NPMCDataset/raw/data.json', 'r') as f:
        docs = json.load(f)
    store.update(docs, key='_id')
    return store


@pytest.fixture
def test_store():
    store = MemoryStore()
    store.connect()

    # Add data from the test file to the store
    with open(TEST_FILE_DIR / 'dataset/NPMCDataset/raw/data.json', 'r') as f:
        docs = json.load(f)

    store.update(docs, key='_id')
    return store


def test_datamodule_init(label_processor, feature_processor, train_store,
                         test_store):
    datamodule = NPMCDataModule(feature_processor=feature_processor,
                                label_processor=label_processor,
                                training_data_store=train_store,
                                testing_data_store=test_store,
                                batch_size=16,
                                split_seed=0)

    assert datamodule.feature_processor == feature_processor
    assert datamodule.label_processor == label_processor
    assert datamodule.training_data_store == train_store
    assert datamodule.testing_data_store == test_store
    assert datamodule.batch_size == 16
    assert datamodule.split_seed == 0


def test_datamodule_setup(tmp_path, label_processor, feature_processor,
                          train_store, test_store):
    datamodule = NPMCDataModule(
        feature_processor=feature_processor,
        label_processor=label_processor,
        training_data_store=train_store,
        testing_data_store=test_store,
        training_data_dir=os.path.join(tmp_path, 'training_data'),
        testing_data_dir=os.path.join(tmp_path, 'testing_data'),
        batch_size=16)

    with pytest.raises(RuntimeError):
        datamodule.setup()

    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule.feature_processor == feature_processor
    assert datamodule.label_processor == label_processor
    assert datamodule.npmc_train is not None
    assert datamodule.npmc_val is not None
    assert datamodule.npmc_test is not None

    assert len(datamodule.npmc_train) == 14
    assert len(datamodule.npmc_val) == 2
    assert len(datamodule.npmc_test) == 16


def test_datamodule_no_test_store(tmp_path, label_processor, feature_processor,
                                  train_store):
    datamodule = NPMCDataModule(feature_processor=feature_processor,
                                label_processor=label_processor,
                                training_data_store=train_store,
                                training_data_dir=os.path.join(
                                    tmp_path, 'training_data'),
                                batch_size=16)

    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule.feature_processor == feature_processor
    assert datamodule.label_processor == label_processor
    assert datamodule.npmc_train is not None
    assert datamodule.npmc_val is not None
    assert datamodule.npmc_test is not None

    assert len(datamodule.npmc_train) == 12
    assert len(datamodule.npmc_val) == 2
    assert len(datamodule.npmc_test) == 2


def test_datamodule_no_test_store(tmp_path, label_processor, feature_processor,
                                  train_store):
    datamodule = NPMCDataModule(feature_processor=feature_processor,
                                label_processor=label_processor,
                                training_data_store=train_store,
                                training_data_dir=os.path.join(
                                    tmp_path, 'training_data'),
                                batch_size=16,
                                calc_mean=True)

    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule.spectra_mean.shape == (600, )
    assert datamodule.spectra_std.shape == (600, )


def test_pyg_dataloader(tmp_path, label_processor, graph_feature_processor,
                        train_store):
    datamodule = NPMCDataModule(feature_processor=graph_feature_processor,
                                label_processor=label_processor,
                                training_data_store=train_store,
                                training_data_dir=os.path.join(
                                    tmp_path, 'training_data'),
                                batch_size=2)

    datamodule.prepare_data()
    datamodule.setup()

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
