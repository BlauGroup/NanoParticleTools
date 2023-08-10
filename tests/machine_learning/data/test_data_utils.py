from NanoParticleTools.machine_learning.data.utils import get_sunset_datasets
from NanoParticleTools.machine_learning.data import SummedWavelengthRangeLabelProcessor, NPMCDataset
from NanoParticleTools.machine_learning.models.mlp_model.data import VolumeFeatureProcessor
import torch

DATA_PATH = '/Users/sivonxay/Library/CloudStorage/GoogleDrive-esivonxay@lbl.gov/My Drive/Postdoc Work/Jupyter_Notebooks/Paper/Figure_2'


def test_sunset_five():
    train_dataset, val_dataset, iid_test_dataset, ood_test_dataset = get_sunset_datasets(
        5,
        feature_processor_cls=VolumeFeatureProcessor,
        label_processor_cls=SummedWavelengthRangeLabelProcessor,
        data_path=DATA_PATH,
        feature_processor_kwargs=None,
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


def test_sunset_one():
    train_dataset, val_dataset, iid_test_dataset, ood_test_dataset = get_sunset_datasets(
        1,
        feature_processor_cls=VolumeFeatureProcessor,
        label_processor_cls=SummedWavelengthRangeLabelProcessor,
        data_path=DATA_PATH,
        feature_processor_kwargs={'possible_elements': ['Dy', 'Ho']},
        label_processor_kwargs={
            'spectrum_ranges': {
                'uv': (100, 400)
            },
            'log_constant': 100
        })

    assert isinstance(iid_test_dataset, NPMCDataset)
    assert iid_test_dataset.feature_processor.possible_elements == ["Dy", "Er", "Ho", "Mg", "Yb"]
    assert len(train_dataset) == 1624
    assert len(val_dataset) == 286
    assert len(iid_test_dataset) == 99
    assert len(ood_test_dataset) == 279


def test_sunset_cat():
    train_dataset, val_dataset, iid_test_dataset, ood_test_dataset = get_sunset_datasets(
        [1, 5],
        feature_processor_cls=VolumeFeatureProcessor,
        label_processor_cls=SummedWavelengthRangeLabelProcessor,
        data_path=DATA_PATH,
        feature_processor_kwargs={},
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
    assert len(train_dataset) == 4736 + 1624
    assert len(val_dataset) == 835 + 286
    assert len(iid_test_dataset) == 287 + 99
    assert len(ood_test_dataset) == 206 + 279
