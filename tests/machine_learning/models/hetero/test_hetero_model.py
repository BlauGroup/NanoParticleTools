from NanoParticleTools.machine_learning.models.hetero.model import DopantInteractionHeteroModel
from NanoParticleTools.machine_learning.models.hetero.data import DopantInteractionFeatureProcessor
from NanoParticleTools.machine_learning.data.processors import SummedWavelengthRangeLabelProcessor
from NanoParticleTools.machine_learning.data.dataset import NPMCDataset
import pytest
from pathlib import Path

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / '..' / '..' / '..' / 'test_files'


@pytest.fixture
def label_processor():
    return SummedWavelengthRangeLabelProcessor(
        spectrum_ranges={'uv': (-1000, -700)}, log_constant=10)


@pytest.fixture
def feature_processor():
    return DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Nd', 'Er'],
        asymmetric_interaction=True,
        include_zeros=True)


@pytest.fixture
def dataset(feature_processor, label_processor):
    dataset = NPMCDataset(TEST_FILE_DIR / 'dataset/data.json',
                          feature_processor=feature_processor,
                          label_processor=label_processor,
                          use_metadata=False,
                          cache_in_memory=True)
    return dataset


def test_model(dataset):
    model = DopantInteractionHeteroModel()

    model.evaluate_step(dataset[0])


def test_embedding(dataset):
    model = DopantInteractionHeteroModel(use_volume_in_dopant_constraint=False,
                                         normalize_interaction_by_volume=False,
                                         use_inverse_concentration=False,
                                         interaction_embedding=True)

    model.evaluate_step(dataset[0])


def test_model_variants(dataset):
    model = DopantInteractionHeteroModel(use_volume_in_dopant_constraint=True,
                                         normalize_interaction_by_volume=False,
                                         use_inverse_concentration=False)

    model.evaluate_step(dataset[0])

    model = DopantInteractionHeteroModel(use_volume_in_dopant_constraint=False,
                                         normalize_interaction_by_volume=True,
                                         use_inverse_concentration=False)

    model.evaluate_step(dataset[0])

    model = DopantInteractionHeteroModel(use_volume_in_dopant_constraint=False,
                                         normalize_interaction_by_volume=False,
                                         use_inverse_concentration=True)

    model.evaluate_step(dataset[0])

    model = DopantInteractionHeteroModel(n_input_nodes=10,
                                         use_volume_in_dopant_constraint=True,
                                         normalize_interaction_by_volume=True,
                                         use_inverse_concentration=True)

    model.evaluate_step(dataset[0])
