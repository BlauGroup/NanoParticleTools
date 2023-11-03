from NanoParticleTools.machine_learning.models.cnn_model.model import CNNModel
from NanoParticleTools.machine_learning.models.cnn_model.data import CNNFeatureProcessor
from NanoParticleTools.machine_learning.data.processors import SummedWavelengthRangeLabelProcessor
from NanoParticleTools.machine_learning.data.dataset import NPMCDataset

from torch_geometric.data import Batch

import pytest
from pathlib import Path

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / '..' / '..' / '..' / 'test_files'


@pytest.fixture
def label_processor():
    return SummedWavelengthRangeLabelProcessor(
        spectrum_ranges={'uv': (300, 450)}, log_constant=10)


@pytest.fixture
def feature_processor():
    return CNNFeatureProcessor(
        max_np_radius=50,
        resolution=10,
    )


@pytest.fixture
def dataset(feature_processor, label_processor):
    dataset = NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          use_metadata=False,
                          cache_in_memory=True)
    return dataset


def test_1d_model(dataset):
    model = CNNModel(n_output_nodes=1,
                     n_dopants=3,
                     readout_layers=None,
                     dimension=1)

    assert model.readout_layers == [128]

    y_hat, _ = model.evaluate_step(dataset[0])
    assert y_hat.shape == (1, 1)

    rep = model.get_representation(dataset[0])
    assert rep.shape == (1, 64)

    # Check if batched predictions works properly
    batch = Batch.from_data_list([dataset[i] for i in range(4)])
    y_hat, _ = model.evaluate_step(batch)
    assert y_hat.shape == (4, 1)

    rep = model.get_representation(batch)
    assert rep.shape == (4, 64)

    # Should throw error if the dimensions is not right
    model = CNNModel(n_output_nodes=1,
                     n_dopants=3,
                     readout_layers=[16, 16],
                     dimension=2)
    with pytest.raises(RuntimeError):
        out = model.evaluate_step(dataset[0])


def test_2d_model(dataset):
    feature_processor = CNNFeatureProcessor(
        max_np_radius=50,
        resolution=10,
        dims=2,
    )
    # Setting the feature processor like this is not recommended
    dataset.feature_processor = feature_processor

    model = CNNModel(n_output_nodes=1,
                     n_dopants=3,
                     readout_layers=[16, 16],
                     dimension=2)

    y_hat, _ = model.evaluate_step(dataset[0])
    assert y_hat.shape == (1, 1)

    rep = model.get_representation(dataset[0])
    assert rep.shape == (1, 64)

    # Check if batched predictions works properly
    batch = Batch.from_data_list([dataset[i] for i in range(4)])
    y_hat, _ = model.evaluate_step(batch)
    assert y_hat.shape == (4, 1)

    rep = model.get_representation(batch)
    assert rep.shape == (4, 64)


def test_3d_model(dataset):
    feature_processor = CNNFeatureProcessor(
        max_np_radius=50,
        resolution=10,
        dims=3,
    )
    # Setting the feature processor like this is not recommended
    dataset.feature_processor = feature_processor

    model = CNNModel(n_output_nodes=1,
                     n_dopants=3,
                     readout_layers=[16, 16],
                     dimension=3)

    y_hat, _ = model.evaluate_step(dataset[0])
    assert y_hat.shape == (1, 1)

    rep = model.get_representation(dataset[0])
    assert rep.shape == (1, 64)

    # Check if batched predictions works properly
    batch = Batch.from_data_list([dataset[i] for i in range(4)])
    y_hat, _ = model.evaluate_step(batch)
    assert y_hat.shape == (4, 1)

    rep = model.get_representation(batch)
    assert rep.shape == (4, 64)
