from NanoParticleTools.machine_learning.models.mlp_model.model import MLPSpectrumModel
from NanoParticleTools.machine_learning.models.mlp_model.data import MLPFeatureProcessor
from NanoParticleTools.machine_learning.data.processors import SummedWavelengthRangeLabelProcessor
from NanoParticleTools.machine_learning.data.dataset import NPMCDataset
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
    return MLPFeatureProcessor(max_layers=5)


@pytest.fixture
def dataset(feature_processor, label_processor):
    dataset = NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          use_metadata=False,
                          cache_in_memory=True)
    return dataset


def test_model(dataset):
    model = MLPSpectrumModel(max_layers=5, n_dopants=3, n_output_nodes=1)
    # Set to evaluation mode since the model has a batch normalization layer
    model.eval()

    y_hat, loss = model.evaluate_step(dataset[0])

    assert y_hat.shape == (1, 1)

    model = MLPSpectrumModel(max_layers=5, n_dopants=3, n_output_nodes=10, use_volume=False)
    # Set to evaluation mode since the model has a batch normalization layer
    model.eval()

    y_hat, loss = model.evaluate_step(dataset[0])

    assert y_hat.shape == (1, 10)
