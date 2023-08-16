from typing import Dict
from NanoParticleTools.machine_learning.data.dataset import (
    NPMCDataset, get_required_fields)
from NanoParticleTools.machine_learning.data.processors import (
    WavelengthSpectrumLabelProcessor, FeatureProcessor, LabelProcessor)
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
    return TabularFeatureProcessor(max_layers=3,
                                   include_volume=True,
                                   possible_elements=['Yb', 'Nd', 'Mg'])


def test_dataset(feature_processor, label_processor):
    dataset = NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          use_metadata=False,
                          cache_in_memory=True)

    assert len(dataset) == 16
    assert dataset[0] != dataset[1]
    assert dataset[0].x.shape == (1, 15)
    assert dataset[0].y.shape == (1, 600)
    assert dataset[0].log_y.shape == (1, 600)


@pytest.fixture
def data_store():
    store = MemoryStore()
    store.connect()

    # Add data from the test file to the store
    with open(TEST_FILE_DIR / 'dataset/data.json', 'r') as f:
        docs = json.load(f)

    store.update(docs, key='_id')
    return store


@pytest.fixture
def data_store_two():
    store = MemoryStore()
    store.connect()

    # Add data from the test file to the store
    with open(TEST_FILE_DIR / 'dataset_two/data.json', 'r') as f:
        docs = json.load(f)

    store.update(docs, key='_id')
    return store


@pytest.fixture
def dataset(feature_processor, label_processor):
    dataset = NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          use_metadata=False,
                          cache_in_memory=True)
    return dataset


def test_store_download(tmp_path, feature_processor, label_processor,
                        data_store):
    dataset = NPMCDataset.from_store(feature_processor,
                                     label_processor,
                                     data_store,
                                     doc_filter=None,
                                     overwrite=False,
                                     use_metadata=False,
                                     file_path=os.path.abspath(tmp_path /
                                                               'dataset.json'))

    assert len(dataset) == 16
    assert dataset.feature_processor == feature_processor
    assert dataset.label_processor == label_processor

    with pytest.raises(NotImplementedError):
        dataset.collate_fn()

    with pytest.raises(NotImplementedError):
        dataset = NPMCDataset.from_store(feature_processor,
                                         label_processor,
                                         data_store,
                                         doc_filter=None,
                                         overwrite=False,
                                         use_metadata=False,
                                         file_path=os.path.abspath(
                                             tmp_path / 'dataset.hdf5'))


def test_overwrite(tmp_path, feature_processor, label_processor, data_store,
                   data_store_two):
    dataset = NPMCDataset.from_store(feature_processor,
                                     label_processor,
                                     data_store,
                                     doc_filter=None,
                                     overwrite=False,
                                     use_metadata=False,
                                     file_path=os.path.abspath(tmp_path /
                                                               'dataset.json'))

    assert len(dataset) == 16

    # If the overwrite flag is set to False, the new dataset is not actually used
    with pytest.warns():
        dataset = NPMCDataset.from_store(feature_processor,
                                         label_processor,
                                         data_store_two,
                                         doc_filter=None,
                                         overwrite=False,
                                         use_metadata=False,
                                         file_path=os.path.abspath(
                                             tmp_path / 'dataset.json'))

    assert len(dataset) == 16

    # If the overwrite flag is set to True, the new dataset is used
    dataset = NPMCDataset.from_store(feature_processor,
                                     label_processor,
                                     data_store_two,
                                     doc_filter=None,
                                     overwrite=True,
                                     use_metadata=True,
                                     file_path=os.path.abspath(tmp_path /
                                                               'dataset.json'))

    assert len(dataset) == 17


def test_dataset_process_doc(dataset, feature_processor, label_processor):
    dataset = NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          use_metadata=True,
                          cache_in_memory=False)

    assert 'metadata' in dataset[0]
    assert dataset.get_random() is not None


def test_get_required_fields():

    class DummyFeatureProcessor(FeatureProcessor):

        def process_doc(self, doc: Dict) -> Dict:
            # do nothing
            return doc

    class DummyLabelProcessor(LabelProcessor):

        def process_doc(self, doc: Dict) -> Dict:
            # do nothing
            return doc

    feature_processor = DummyFeatureProcessor(fields=[])
    label_processor = DummyLabelProcessor(fields=['output.summary'])
    fields = get_required_fields(feature_processor, label_processor)
    assert fields == ['output.summary', 'input']
