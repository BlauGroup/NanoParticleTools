from NanoParticleTools.machine_learning.models.gnn_model.data import (
    GraphInteractionFeatureProcessor)
from NanoParticleTools.machine_learning.data import SummedWavelengthRangeLabelProcessor
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
def doc_one_invalid():
    return {
        'input': {
            'constraints': [
                SphericalConstraint(10),
                SphericalConstraint(20),
                SphericalConstraint(20)
            ],
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
        }, {}],
    }


@pytest.fixture
def doc_one_invalid_2():
    return {
        'input': {
            'constraints': [
                SphericalConstraint(10),
                SphericalConstraint(20),
                SphericalConstraint(20)
            ],
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
        }, {
            'Yb': 0,
            'Er': 0
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


@pytest.fixture
def doc_three():
    return {
        'input': {
            'constraints': [
                SphericalConstraint(10),
                SphericalConstraint(20),
                SphericalConstraint(30)
            ],
            'dopant_specifications': [(0, 0.02, 'Mg', 'Y')]
        },
        'dopant_concentration': [{
            'Mg': 0.02
        }, {}, {}],
    }


@pytest.fixture
def doc_four():
    return {
        'input': {
            'constraints': [
                SphericalConstraint(10),
                SphericalConstraint(20),
                SphericalConstraint(30)
            ],
            'dopant_specifications':
            [(0, 0.02, 'Er', 'Y'), (0, 0.5, 'Yb', 'Y'), (1, 0.25, 'Er', 'Y')]
        },
        'dopant_concentration': [{
            'Yb': 0.5,
            'Er': 0.02
        }, {
            'Er': 0.25
        }],
    }


@pytest.fixture
def doc_four_dupe():
    return {
        'input': {
            'constraints': [
                SphericalConstraint(10),
                SphericalConstraint(20),
                SphericalConstraint(20),
                SphericalConstraint(30)
            ],
            'dopant_specifications':
            [(0, 0.02, 'Er', 'Y'), (0, 0.5, 'Yb', 'Y'), (1, 0.25, 'Er', 'Y')]
        },
        'dopant_concentration': [{
            'Yb': 0.5,
            'Er': 0.02
        }, {}, {
            'Er': 0.25
        }],
    }


def test_graph_interaction_feature_processor(doc_one):
    feature_processor = GraphInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'])

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.radii.shape == (3, )
    assert data.constraint_radii_idx.shape == (2, 2)
    assert torch.allclose(data.dopant_constraint_indices, torch.tensor([0, 0, 1, 1]))
    assert data.dopant_concs.shape == (4, )
    assert torch.allclose(data.dopant_concs, torch.tensor([0.499, 0.25, 0.2, 0.1]))
    assert data.edge_index.shape == (2, 32)


def test_graph_interaction_feature_processor_zeros(doc_one):
    feature_processor = GraphInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'], include_zeros=True)

    _data_dict = feature_processor.process_doc(doc_one)
    data = feature_processor.data_cls(**_data_dict)
    assert data.radii.shape == (3, )
    assert data.constraint_radii_idx.shape == (2, 2)
    assert torch.allclose(data.dopant_constraint_indices,
                          torch.tensor([0, 0, 0, 1, 1, 1]))
    assert data.dopant_concs.shape == (6, )
    assert torch.allclose(data.dopant_concs, torch.tensor([0.499, 0.25, 0, 0.2, 0.1, 0]))
    assert data.edge_index.shape == (2, 72)


def test_invalid_doc(doc_one_invalid):
    feature_processor = GraphInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'], include_zeros=True)

    _data_dict = feature_processor.process_doc(doc_one_invalid)
    data = feature_processor.data_cls(**_data_dict)
    assert data.radii.shape == (3, )
    assert data.constraint_radii_idx.shape == (2, 2)
    assert torch.allclose(data.dopant_constraint_indices,
                          torch.tensor([0, 0, 0, 1, 1, 1]))
    assert data.dopant_concs.shape == (6, )
    assert torch.allclose(data.dopant_concs, torch.tensor([0.499, 0.25, 0, 0.2, 0.1, 0]))
    assert data.edge_index.shape == (2, 72)

    feature_processor = GraphInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'], include_zeros=False)
    _data_dict = feature_processor.process_doc(doc_one_invalid)
    data = feature_processor.data_cls(**_data_dict)
    assert data.radii.shape == (3, )
    assert data.constraint_radii_idx.shape == (2, 2)
    assert torch.allclose(data.dopant_constraint_indices, torch.tensor([0, 0, 1, 1]))
    assert data.dopant_concs.shape == (4, )
    assert torch.allclose(data.dopant_concs, torch.tensor([0.499, 0.25, 0.2, 0.1]))
    assert data.edge_index.shape == (2, 32)


def test_grad(doc_four):
    feature_processor = GraphInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'], input_grad=True)

    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(**_data_dict)
    assert data.radii.requires_grad
    assert data.dopant_concs.requires_grad

    feature_processor = GraphInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'], input_grad=False)

    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(**_data_dict)
    assert not data.radii.requires_grad
    assert not data.dopant_concs.requires_grad
