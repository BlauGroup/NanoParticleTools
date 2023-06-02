from NanoParticleTools.machine_learning.data.dataset import NPMCDataset
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
def data_store():
    store = MemoryStore()
    store.connect()

    # Add data from the test file to the store
    with open(TEST_FILE_DIR / 'dataset/NPMCDataset/raw/data.json', 'r') as f:
        docs = json.load(f)

    store.update(docs, key='_id')
    return store


@pytest.fixture
def data_store_two():
    store = MemoryStore()
    store.connect()

    # Add data from the test file to the store
    with open(TEST_FILE_DIR / 'dataset_two/NPMCDataset/raw/data.json',
              'r') as f:
        docs = json.load(f)

    store.update(docs, key='_id')
    return store


@pytest.fixture
def dataset(feature_processor, label_processor):
    with pytest.warns(UserWarning):
        return NPMCDataset(root=TEST_FILE_DIR / 'dataset',
                           feature_processor=feature_processor,
                           label_processor=label_processor,
                           data_store=None,
                           doc_filter={},
                           download=False,
                           overwrite=False,
                           use_cache=True,
                           dataset_size=-1,
                           use_metadata=True)


def test_dataset(dataset):
    assert len(dataset) == 16
    assert os.path.normpath(dataset.raw_folder) == os.path.normpath(
        TEST_FILE_DIR / 'dataset/NPMCDataset/raw')


def test_store_download(tmp_path, feature_processor, label_processor,
                        data_store):
    # check if pytest throws an error
    with pytest.raises(RuntimeError):
        dataset = NPMCDataset(tmp_path,
                              feature_processor=feature_processor,
                              label_processor=label_processor,
                              data_store=data_store,
                              doc_filter=None,
                              download=False,
                              overwrite=False,
                              use_cache=False)

    dataset = NPMCDataset(tmp_path,
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          data_store=data_store,
                          doc_filter=None,
                          download=True,
                          overwrite=False,
                          use_cache=False)
    assert len(dataset) == 16
    assert os.path.normpath(dataset.raw_folder) == os.path.normpath(
        tmp_path / 'NPMCDataset/raw')
    assert os.path.normpath(dataset.processed_folder) == os.path.normpath(
        tmp_path / 'NPMCDataset/processed')


def test_no_size_check(tmp_path, feature_processor, label_processor,
                       data_store, data_store_two):
    dataset = NPMCDataset(tmp_path,
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          data_store=data_store,
                          doc_filter=None,
                          download=True,
                          overwrite=False,
                          use_cache=False)
    assert len(dataset) == 16

    with pytest.warns(UserWarning):
        dataset = NPMCDataset(tmp_path,
                              feature_processor=feature_processor,
                              label_processor=label_processor,
                              data_store=data_store_two,
                              doc_filter=None,
                              download=False,
                              overwrite=False,
                              use_cache=False,
                              dataset_size=-1)
    assert len(dataset) == 16


def test_size_none(tmp_path, feature_processor, label_processor, data_store,
                   data_store_two):
    dataset = NPMCDataset(tmp_path,
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          data_store=data_store,
                          doc_filter=None,
                          download=True,
                          overwrite=False,
                          use_cache=False)
    assert len(dataset) == 16

    with pytest.warns(UserWarning):
        dataset = NPMCDataset(tmp_path,
                              feature_processor=feature_processor,
                              label_processor=label_processor,
                              data_store=data_store_two,
                              doc_filter=None,
                              download=False,
                              overwrite=False,
                              use_cache=False,
                              dataset_size=None)
    assert len(dataset) == 17


def test_incorrect_dataset_size(tmp_path, feature_processor, label_processor,
                                data_store, data_store_two):
    dataset = NPMCDataset(tmp_path,
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          data_store=data_store,
                          doc_filter=None,
                          download=True,
                          overwrite=True,
                          use_cache=False)
    assert len(dataset) == 16

    with pytest.warns(UserWarning):
        dataset = NPMCDataset(tmp_path,
                              feature_processor=feature_processor,
                              label_processor=label_processor,
                              data_store=data_store_two,
                              doc_filter=None,
                              download=False,
                              overwrite=True,
                              use_cache=False,
                              dataset_size=17)
    assert len(dataset) == 17


def test_data_indexing(dataset):
    assert len(dataset) == 16
    assert dataset[0] != dataset[1]
    assert dataset[0].x.shape == (1, 15)
    assert dataset[0].y.shape == (1, 600)
    assert dataset[0].log_y.shape == (1, 600)
