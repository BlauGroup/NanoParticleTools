from NanoParticleTools.machine_learning.data.utils import get_sunset_datasets
from NanoParticleTools.machine_learning.data import SummedWavelengthRangeLabelProcessor, NPMCDataset
from NanoParticleTools.machine_learning.models.mlp_model.data import TabularFeatureProcessor
import torch

import pytest

DATA_PATH = None


@pytest.mark.skip(reason="Data not available")
def test_sunset_five():
    train_dataset, val_dataset, iid_test_dataset, ood_test_dataset = get_sunset_datasets(
        5,
        feature_processor_cls=TabularFeatureProcessor,
        label_processor_cls=SummedWavelengthRangeLabelProcessor,
        data_path=DATA_PATH,
        feature_processor_kwargs={
            'include_volume': True,
            'volume_normalization': 'div',
            'volume_normalization_const': 1e6
        },
        label_processor_kwargs={
            'spectrum_ranges': {
                'uv': (100, 400)
            },
            'log_constant': 100
        })

    assert isinstance(iid_test_dataset, NPMCDataset)
    assert iid_test_dataset.feature_processor.possible_elements == [
        "Er", "Nd", "Yb"
    ]
    assert len(train_dataset) == 4736
    assert len(val_dataset) == 835
    assert len(iid_test_dataset) == 287
    assert len(ood_test_dataset) == 206


@pytest.mark.skip(reason="Data not available")
def test_sunset_one():
    train_dataset, val_dataset, iid_test_dataset, ood_test_dataset = get_sunset_datasets(
        1,
        feature_processor_cls=TabularFeatureProcessor,
        label_processor_cls=SummedWavelengthRangeLabelProcessor,
        data_path=DATA_PATH,
        feature_processor_kwargs={
            'possible_elements': ['Dy', 'Ho'],
            'include_volume': True,
            'volume_normalization': 'div',
            'volume_normalization_const': 1e6
        },
        label_processor_kwargs={
            'spectrum_ranges': {
                'uv': (100, 400)
            },
            'log_constant': 100
        })

    assert isinstance(iid_test_dataset, NPMCDataset)
    assert iid_test_dataset.feature_processor.possible_elements == [
        "Dy", "Er", "Ho", "Mg", "Yb"
    ]
    assert len(train_dataset) == 1623
    assert len(val_dataset) == 286
    assert len(iid_test_dataset) == 100
    assert len(ood_test_dataset) == 279


@pytest.mark.skip(reason="Data not available")
def test_sunset_cat():
    train_dataset, val_dataset, iid_test_dataset, ood_test_dataset = get_sunset_datasets(
        [1, 5],
        feature_processor_cls=TabularFeatureProcessor,
        label_processor_cls=SummedWavelengthRangeLabelProcessor,
        data_path=DATA_PATH,
        feature_processor_kwargs={
            'include_volume': True,
            'volume_normalization': 'div',
            'volume_normalization_const': 1e6
        },
        label_processor_kwargs={
            'spectrum_ranges': {
                'uv': (100, 400)
            },
            'log_constant': 100
        })

    assert isinstance(train_dataset, torch.utils.data.ConcatDataset)
    possible_elements = train_dataset.datasets[
        0].dataset.feature_processor.possible_elements
    assert possible_elements == ['Er', 'Mg', 'Nd', 'Yb']
    assert len(train_dataset) == 4736 + 1623
    assert len(val_dataset) == 835 + 286
    assert len(iid_test_dataset) == 287 + 100
    assert len(ood_test_dataset) == 206 + 279
