from NanoParticleTools.machine_learning.models.hetero.data import DopantInteractionFeatureProcessor
from torch_geometric.data import Batch
from NanoParticleTools.inputs import SphericalConstraint
from NanoParticleTools.machine_learning.data import SummedWavelengthRangeLabelProcessor
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


def test_augment(doc_two):
    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'],
        augment_data=True,
        include_zeros=True,
        augment_prob=1,
        augment_subdivisions=1)

    _data_dict = feature_processor.process_doc(doc_two)
    data = feature_processor.data_cls(_data_dict)

    assert data.radii.shape == (5, )
    assert data['dopant'].x.shape == (12, )
    assert data['interaction'].types.shape == (144, )
    assert data['interaction'].types.max() == 5


def test_exclude_empty_exterior(doc_four, doc_four_dupe):
    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'],
        asymmetric_interaction=True,
        include_zeros=True)

    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(_data_dict)

    _data_dict = feature_processor.process_doc(doc_four_dupe)
    data_dupe = feature_processor.data_cls(_data_dict)

    assert torch.all(data.radii == data_dupe.radii)


def test_include_zeros(doc_two, doc_four):
    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'],
        asymmetric_interaction=True,
        include_zeros=True)

    _data_dict = feature_processor.process_doc(doc_two)
    data = feature_processor.data_cls(_data_dict)
    assert data.radii.shape == (4, )
    assert data['interaction'].types.shape == (81, )
    assert data['interaction'].types.max() == 8

    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(_data_dict)
    assert data.radii.shape == (3, )
    assert data['interaction'].types.shape == (36, )
    assert data['interaction'].types.max() == 8

    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'],
        asymmetric_interaction=False,
        include_zeros=True)

    _data_dict = feature_processor.process_doc(doc_two)
    data = feature_processor.data_cls(_data_dict)
    assert data.radii.shape == (4, )
    assert data['interaction'].types.shape == (81, )
    assert data['interaction'].types.max() == 5

    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(_data_dict)
    assert data.radii.shape == (3, )
    assert data['interaction'].types.shape == (36, )
    assert data['interaction'].types.max() == 5


def test_asymmetric(doc_two, doc_four):

    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'],
        asymmetric_interaction=True,
        include_zeros=False)
    _data_dict = feature_processor.process_doc(doc_two)
    data = feature_processor.data_cls(_data_dict)

    assert data.radii.shape == (4, )
    assert data['interaction'].types.shape == (9, )
    assert torch.allclose(data['interaction'].types, torch.arange(0, 9))

    print(feature_processor.edge_type_map)
    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(_data_dict)
    assert data.radii.shape == (3, )
    assert data['interaction'].types.shape == (9, )
    assert torch.allclose(data['interaction'].types,
                          torch.tensor([0, 1, 1, 3, 4, 4, 3, 4, 4]))

    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'],
        asymmetric_interaction=False,
        include_zeros=False)
    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(_data_dict)
    assert data.radii.shape == (3, )
    assert data['interaction'].types.shape == (9, )
    assert torch.allclose(data['interaction'].types,
                          torch.tensor([0, 1, 1, 1, 3, 3, 1, 3, 3]))


def test_grad(doc_four):
    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'],
        asymmetric_interaction=False,
        include_zeros=False,
        input_grad=True)
    _data_dict = feature_processor.process_doc(doc_four)
    data = feature_processor.data_cls(_data_dict)
    assert data.radii.requires_grad
    assert data['dopant'].x.requires_grad


def test_interaction_processor(doc_one, doc_two, doc_three):

    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'])

    assert feature_processor.is_graph

    _data_dict = feature_processor.process_doc(doc_one)
    data1 = feature_processor.data_cls(_data_dict)

    assert data1.radii.shape == (3, )
    assert data1.constraint_radii_idx.shape == (2, 2)
    assert data1['dopant'].x.shape == (4, )
    assert data1['dopant'].types.shape == (4, )
    assert data1['dopant'].constraint_indices.shape == (4, )
    assert data1['dopant'].num_nodes == 4
    assert data1['interaction'].type_indices.shape == (16, 2)
    assert data1['interaction'].dopant_indices.shape == (16, 2)
    assert data1['interaction'].num_nodes == 16
    assert data1['dopant', 'coupled_to',
                 'interaction'].edge_index.shape == (2, 16)
    assert data1['interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 16)

    # Check the values
    assert torch.all(data1['dopant'].constraint_indices == torch.tensor(
        [0, 0, 1, 1])).item()

    _data_dict = feature_processor.process_doc(doc_two)
    data2 = feature_processor.data_cls(_data_dict)

    assert data2.radii.shape == (4, )
    assert data2.constraint_radii_idx.shape == (3, 2)
    assert data2['dopant'].x.shape == (3, )
    assert data2['dopant'].types.shape == (3, )
    assert data2['dopant'].constraint_indices.shape == (3, )
    assert data2['dopant'].num_nodes == 3
    assert data2['interaction'].type_indices.shape == (9, 2)
    assert data2['interaction'].dopant_indices.shape == (9, 2)
    assert data2['interaction'].num_nodes == 9
    assert data2['dopant', 'coupled_to',
                 'interaction'].edge_index.shape == (2, 9)
    assert data2['interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 9)

    # Check the values
    assert torch.all(
        data2['dopant'].constraint_indices == torch.tensor([0, 0, 2])).item()

    _data_dict = feature_processor.process_doc(doc_three)
    data3 = feature_processor.data_cls(_data_dict)

    assert data3.radii.shape == (2, )
    assert data3.constraint_radii_idx.shape == (1, 2)
    assert data3['dopant'].x.shape == (1, )
    assert data3['dopant'].types.shape == (1, )
    assert data3['dopant'].constraint_indices.shape == (1, )
    assert data3['dopant'].num_nodes == 1
    assert data3['interaction'].type_indices.shape == (1, 2)
    assert data3['interaction'].dopant_indices.shape == (1, 2)
    assert data3['interaction'].num_nodes == 1
    assert data3['dopant', 'coupled_to',
                 'interaction'].edge_index.shape == (2, 1)
    assert data3['interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 1)

    # Check the values
    assert torch.all(
        data3['dopant'].constraint_indices == torch.tensor([0])).item()


def test_interaction_processor_batch(doc_one, doc_two, doc_three):

    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'])
    label_processor = SummedWavelengthRangeLabelProcessor(
        spectrum_ranges={'uv': (100, 400)}, log_constant=1)

    _data_dict = feature_processor.process_doc(doc_one)
    _data_dict.update(label_processor.example())
    data1 = feature_processor.data_cls(_data_dict)
    assert len(data1.labels) == 1
    assert data1.y.shape == (1, 1)
    assert data1.log_y.shape == (1, 1)
    assert data1.log_const == 1

    _data_dict = feature_processor.process_doc(doc_two)
    _data_dict.update(label_processor.example())
    data2 = feature_processor.data_cls(_data_dict)
    assert len(data2.labels) == 1
    assert data2.y.shape == (1, 1)
    assert data2.log_y.shape == (1, 1)
    assert data2.log_const == 1

    _data_dict = feature_processor.process_doc(doc_three)
    _data_dict.update(label_processor.example())
    data3 = feature_processor.data_cls(_data_dict)
    assert len(data3.labels) == 1
    assert data3.y.shape == (1, 1)
    assert data3.log_y.shape == (1, 1)
    assert data3.log_const == 1

    batch = Batch.from_data_list([data1, data2, data3])
    # Check the labels
    assert len(batch.labels) == 3
    assert batch.y.shape == (3, 1)
    assert batch.log_y.shape == (3, 1)
    assert len(batch.log_const) == 3

    # Check the feature shapes
    assert batch.radii.shape == (9, )
    assert batch.constraint_radii_idx.shape == (6, 2)
    assert batch['dopant'].x.shape == (8, )
    assert batch['dopant'].types.shape == (8, )
    assert batch['dopant'].constraint_indices.shape == (8, )
    assert batch['dopant'].num_nodes == 8
    assert batch['interaction'].type_indices.shape == (26, 2)
    assert batch['interaction'].dopant_indices.shape == (26, 2)
    assert batch['interaction'].num_nodes == 26
    assert batch['dopant', 'coupled_to',
                 'interaction'].edge_index.shape == (2, 26)
    assert batch['interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 26)

    # Check the values of the features
    assert torch.all(batch['dopant'].batch == torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 2])).item()
    assert torch.allclose(
        batch.radii,
        torch.tensor([0, 10, 20, 0, 10, 20, 30, 0, 10]).float())
    assert torch.allclose(
        batch.constraint_radii_idx,
        torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5], [5, 6], [7, 8]]))
    assert torch.allclose(
        batch['dopant'].x,
        torch.tensor([0.499, 0.25, 0.2, 0.1, 0.499, 0.02, 0.25, 0.02]))
    assert torch.all(batch['dopant'].types == torch.tensor(
        [0, 1, 0, 1, 0, 1, 2, 2])).item()
    assert torch.all(batch['dopant'].constraint_indices == torch.tensor(
        [0, 0, 1, 1, 2, 2, 4, 5])).item()

    # yapf: disable
    assert torch.all(batch['interaction'].batch == torch.cat((torch.zeros(16),
                                                              torch.ones(9),
                                                              2 * torch.ones(1))))
    # yapf: enable
    assert torch.all(batch['interaction'].type_indices == torch.cat((
        data1['interaction'].type_indices, data2['interaction'].type_indices,
        data3['interaction'].type_indices))).item()

    assert torch.all(batch['interaction'].dopant_indices == torch.cat((
        data1['interaction'].dopant_indices,
        data1['dopant'].num_nodes + data2['interaction'].dopant_indices,
        data1['dopant'].num_nodes + data2['dopant'].num_nodes +
        data3['interaction'].dopant_indices))).item()

    _edge_index1 = data1['dopant', 'coupled_to', 'interaction'].edge_index
    _edge_index2 = data2['dopant', 'coupled_to', 'interaction'].edge_index
    _edge_index2[0, :] += data1['dopant'].num_nodes
    _edge_index2[1, :] += data1['interaction'].num_nodes
    _edge_index3 = data3['dopant', 'coupled_to', 'interaction'].edge_index
    _edge_index3[0, :] += data1['dopant'].num_nodes + data2['dopant'].num_nodes
    _edge_index3[1, :] += data1['interaction'].num_nodes + data2[
        'interaction'].num_nodes
    assert torch.all(
        batch['dopant', 'coupled_to', 'interaction'].edge_index == torch.cat(
            (_edge_index1, _edge_index2, _edge_index3), dim=1)).item()

    _edge_index1 = data1['interaction', 'coupled_to', 'dopant'].edge_index
    _edge_index2 = data2['interaction', 'coupled_to', 'dopant'].edge_index
    _edge_index2[0, :] += data1['interaction'].num_nodes
    _edge_index2[1, :] += data1['dopant'].num_nodes
    _edge_index3 = data3['interaction', 'coupled_to', 'dopant'].edge_index
    _edge_index3[0, :] += data1['interaction'].num_nodes + data2[
        'interaction'].num_nodes
    _edge_index3[1, :] += data1['dopant'].num_nodes + data2['dopant'].num_nodes
    assert torch.all(
        batch['interaction', 'coupled_to', 'dopant'].edge_index == torch.cat(
            (_edge_index1, _edge_index2, _edge_index3), dim=1)).item()
