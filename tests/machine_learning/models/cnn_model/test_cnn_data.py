from NanoParticleTools.machine_learning.models.cnn_model.data import (
    CNNFeatureProcessor, to_1d_image)
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


def test_1d_image(doc_one):
    conc_tensor = torch.tensor([[0.4990, 0.2500, 0.0000],
                                [0.2000, 0.1000, 0.0000],
                                [0.1, 0.2, 0.3]], requires_grad=True)
    radii_without_zeros = torch.tensor([10., 20., 30.], requires_grad=True)
    out = to_1d_image(conc_tensor, radii_without_zeros, 250, 0.5, full_particle=True)

    assert out.shape == (3, 1001)

    out = to_1d_image(conc_tensor, radii_without_zeros, 250, 0.5, full_particle=False)

    assert out.shape == (3, 501)


def test_1d_cnn_feature_processor(doc_one, doc_two):
    feature_processor = CNNFeatureProcessor(resolution=5,
                                            max_np_radius=25,
                                            dims=1,
                                            full_nanoparticle=False,
                                            possible_elements=['Yb', 'Er', 'Mg'])
    assert feature_processor.is_graph is True
    assert feature_processor.n_possible_elements == 3

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 3, 6)

    assert torch.allclose(data.x, torch.tensor([[0.499, 0.499, 0.499, 0.2, 0.2, 0],
                                                [0.25, 0.25, 0.25, 0.1, 0.1, 0],
                                                [0, 0, 0, 0, 0, 0]]))

    # doc two should not have any zeros since the bounds smaller than the particle
    _data_dict = feature_processor.process_doc(doc_two)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 3, 6)

    assert torch.allclose(data.x, torch.tensor([[0.499, 0.499, 0.499, 0, 0, 0],
                                                [0.02, 0.02, 0.02, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0.25]]))


def test_2d_cnn_feature_processor(doc_one):
    feature_processor = CNNFeatureProcessor(resolution=1,
                                            max_np_radius=25,
                                            dims=2,
                                            possible_elements=['Yb', 'Er', 'Dy', 'Ho'])
    assert feature_processor.is_graph is True
    assert feature_processor.n_possible_elements == 4

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 4, 51, 51)


def test_3d_cnn_feature_processor(doc_one):
    feature_processor = CNNFeatureProcessor(resolution=1,
                                            max_np_radius=25,
                                            dims=3,
                                            possible_elements=['Yb', 'Er', 'Dy', 'Ho'])
    assert feature_processor.is_graph is True
    assert feature_processor.n_possible_elements == 4

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.x.shape == (1, 4, 51, 51, 51)


def test_cnn_feature_processor_batch(doc_one, doc_two):
    feature_processor = CNNFeatureProcessor(resolution=1,
                                            max_np_radius=25,
                                            dims=2,
                                            possible_elements=['Yb', 'Er', 'Dy', 'Ho'],
                                            full_nanoparticle=False)

    data_list = []
    _data_dict = feature_processor.process_doc(doc_one)
    data_list.append(feature_processor.data_cls(**_data_dict))

    _data_dict = feature_processor.process_doc(doc_two)
    data_list.append(feature_processor.data_cls(**_data_dict))

    batch = Batch.from_data_list(data_list)
    assert batch.x.shape == (2, 4, 26, 26)


def test_invalid_cnn_feature_processor():
    # These should raise an error since the number of dimensions is invalid
    with pytest.raises(AssertionError):
        CNNFeatureProcessor(dims=0)

    with pytest.raises(AssertionError):
        CNNFeatureProcessor(dims=4)
