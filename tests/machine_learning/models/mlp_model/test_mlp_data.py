from NanoParticleTools.machine_learning.models.mlp_model.data import (
    MLPFeatureProcessor, TabularFeatureProcessor)
from NanoParticleTools.inputs import SphericalConstraint
from torch_geometric.data import Batch

import torch
import pytest


@pytest.fixture
def doc_one():
    return {
        'input': {
            'constraints': [SphericalConstraint(10),
                            SphericalConstraint(20)],
            'dopant_specifications':
            [(0, 0.5, 'Yb', 'Y'), (0, 0.25, 'Er', 'Y'), (1, 0.2, 'Yb', 'Y'),
             (1, 0.1, 'Er', 'Y')]
        },
        'dopant_concentration': [{
            'Yb': 0.499,
            'Er': 0.25
        }, {
            'Yb': 0.2,
            'Er': 0.1
        }],
    }


@pytest.fixture
def doc_two():
    return {
        'input': {
            'constraints': [
                SphericalConstraint(10),
                SphericalConstraint(20),
                SphericalConstraint(30)
            ],
            'dopant_specifications':
            [(0, 0.02, 'Er', 'Y'), (0, 0.5, 'Yb', 'Y'), (2, 0.25, 'Mg', 'Y')]
        },
        'dopant_concentration': [{
            'Yb': 0.499,
            'Er': 0.02
        }, {}, {
            'Mg': 0.25
        }],
    }


def test_mlp_feature_processor(doc_one):
    feature_processor = MLPFeatureProcessor(
        max_layers=2, possible_elements=['Yb', 'Er', 'Dy', 'Ho'])
    assert feature_processor.is_graph is True

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 8)
    assert data.radii.shape == (1, 3)
    assert data.radii_without_zero.shape == (1, 2)

    # Test requires_grad
    feature_processor = MLPFeatureProcessor(
        max_layers=2,
        input_grad=True,
        possible_elements=['Yb', 'Er', 'Dy', 'Ho'])
    assert feature_processor.input_grad is True

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.requires_grad is True
    assert data.radii_without_zero.requires_grad is True


def test_mlp_feature_processor_batch(doc_one, doc_two):
    feature_processor = MLPFeatureProcessor(
        max_layers=4, possible_elements=['Yb', 'Er', 'Dy', 'Ho'])

    data_list = []
    _data_dict = feature_processor.process_doc(doc_one)
    data_list.append(feature_processor.data_cls(**_data_dict))

    _data_dict = feature_processor.process_doc(doc_two)
    data_list.append(feature_processor.data_cls(**_data_dict))

    batch = Batch.from_data_list(data_list)
    assert batch.x.shape == (2, 16)
    assert batch.radii.shape == (2, 5)
    assert batch.radii_without_zero.shape == (2, 4)


def test_tabular_feature_processor(doc_one, doc_two):
    feature_processor = TabularFeatureProcessor(max_layers=2,
                                                include_volume=False)
    assert feature_processor.is_graph is True

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 8)

    # This implicitly tests truncation of layers
    _data_dict = feature_processor.process_doc(doc_two)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 8)

    # Test filling in empty layers
    feature_processor = TabularFeatureProcessor(max_layers=4,
                                                include_volume=False)

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 16)

    _data_dict = feature_processor.process_doc(doc_two)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 16)


def test_tabular_feature_processor_batch(doc_one, doc_two):
    feature_processor = TabularFeatureProcessor(max_layers=2,
                                                include_volume=False)

    data_list = []
    _data_dict = feature_processor.process_doc(doc_one)
    data_list.append(feature_processor.data_cls(**_data_dict))

    _data_dict = feature_processor.process_doc(doc_two)
    data_list.append(feature_processor.data_cls(**_data_dict))

    batch = Batch.from_data_list(data_list)
    assert batch.x.shape == (2, 8)


def test_tabular_feature_processor_with_volume(doc_one):
    # Test if volume gets populated
    feature_processor = TabularFeatureProcessor(max_layers=2,
                                                include_volume=True,
                                                volume_normalization_const=1)

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 10)
    assert torch.allclose(data.x[..., -2:],
                          torch.tensor([4188.79020478, 29321.53143350]))

    feature_processor = TabularFeatureProcessor(max_layers=4,
                                                include_volume=True,
                                                volume_normalization='div',
                                                volume_normalization_const=100)

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 20)
    assert torch.allclose(
        data.x[..., -4:],
        torch.tensor([4188.79020478, 29321.53143350, 0, 0]) / 100.)

    feature_processor = TabularFeatureProcessor(max_layers=4,
                                                include_volume=True,
                                                volume_normalization='log',
                                                volume_normalization_const=1)

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)

    assert data.x.shape == (1, 20)
    assert torch.allclose(
        data.x[..., -4:],
        torch.tensor([4188.79020478, 29321.53143350, 0, 0]).add(1).log10())
