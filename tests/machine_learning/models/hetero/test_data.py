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


def test_interaction_processor(doc_one, doc_two, doc_three):

    feature_processor = DopantInteractionFeatureProcessor(
        possible_elements=['Yb', 'Er', 'Mg'])

    assert feature_processor.is_graph

    _data_dict = feature_processor.process_doc(doc_one)
    data1 = feature_processor.data_cls(_data_dict)

    assert data1.constraint_radii.shape == (2, 2)
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

    assert data2.constraint_radii.shape == (2, 2)
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
        data2['dopant'].constraint_indices == torch.tensor([0, 0, 1])).item()

    _data_dict = feature_processor.process_doc(doc_three)
    data3 = feature_processor.data_cls(_data_dict)

    assert data3.constraint_radii.shape == (1, 2)
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


def test_self_interaction_processor(doc_one, doc_two, doc_three):

    feature_processor = DopantInteractionFeatureProcessor(
        separate_self_interaction=True, possible_elements=['Yb', 'Er', 'Mg'])

    _data_dict = feature_processor.process_doc(doc_one)
    data1 = feature_processor.data_cls(_data_dict)

    assert data1.constraint_radii.shape == (2, 2)
    assert data1['dopant'].x.shape == (4, )
    assert data1['dopant'].types.shape == (4, )
    assert data1['dopant'].constraint_indices.shape == (4, )
    assert data1['dopant'].num_nodes == 4
    assert data1['interaction'].type_indices.shape == (12, 2)
    assert data1['interaction'].dopant_indices.shape == (12, 2)
    assert data1['interaction'].num_nodes == 12
    assert data1['self_interaction'].type_indices.shape == (4, 2)
    assert data1['self_interaction'].dopant_indices.shape == (4, 2)
    assert data1['self_interaction'].num_nodes == 4
    assert data1['dopant', 'coupled_to',
                 'interaction'].edge_index.shape == (2, 12)
    assert data1['interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 12)
    assert data1['dopant', 'coupled_to',
                 'self_interaction'].edge_index.shape == (2, 4)
    assert data1['self_interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 4)

    _data_dict = feature_processor.process_doc(doc_two)
    data2 = feature_processor.data_cls(_data_dict)

    assert data2.constraint_radii.shape == (2, 2)
    assert data2['dopant'].x.shape == (3, )
    assert data2['dopant'].types.shape == (3, )
    assert data2['dopant'].constraint_indices.shape == (3, )
    assert data2['dopant'].num_nodes == 3
    assert data2['interaction'].type_indices.shape == (6, 2)
    assert data2['interaction'].dopant_indices.shape == (6, 2)
    assert data2['interaction'].num_nodes == 6
    assert data2['self_interaction'].type_indices.shape == (3, 2)
    assert data2['self_interaction'].dopant_indices.shape == (3, 2)
    assert data2['self_interaction'].num_nodes == 3
    assert data2['dopant', 'coupled_to',
                 'interaction'].edge_index.shape == (2, 6)
    assert data2['interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 6)
    assert data2['dopant', 'coupled_to',
                 'self_interaction'].edge_index.shape == (2, 3)
    assert data2['self_interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 3)

    _data_dict = feature_processor.process_doc(doc_three)
    data3 = feature_processor.data_cls(_data_dict)

    assert data3.constraint_radii.shape == (1, 2)
    assert data3['dopant'].x.shape == (1, )
    assert data3['dopant'].types.shape == (1, )
    assert data3['dopant'].constraint_indices.shape == (1, )
    assert data3['dopant'].num_nodes == 1
    assert data3['interaction'].type_indices.shape == (0, )
    assert data3['interaction'].dopant_indices.shape == (0, )
    assert data3['interaction'].num_nodes == 0
    assert data3['self_interaction'].type_indices.shape == (1, 2)
    assert data3['self_interaction'].dopant_indices.shape == (1, 2)
    assert data3['self_interaction'].num_nodes == 1
    assert data3['dopant', 'coupled_to',
                 'interaction'].edge_index.shape == (2, 0)
    assert data3['interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 0)
    assert data3['dopant', 'coupled_to',
                 'self_interaction'].edge_index.shape == (2, 1)
    assert data3['self_interaction', 'coupled_to',
                 'dopant'].edge_index.shape == (2, 1)


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
    assert batch.constraint_radii.shape == (5, 2)
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
        batch.constraint_radii,
        torch.tensor([[0, 10], [10, 20], [0, 10], [20, 30], [0, 10]]).float())
    assert torch.allclose(
        batch['dopant'].x,
        torch.tensor([0.499, 0.25, 0.2, 0.1, 0.499, 0.02, 0.25, 0.02]))
    assert torch.all(batch['dopant'].types == torch.tensor(
        [0, 1, 0, 1, 0, 1, 2, 2])).item()
    assert torch.all(batch['dopant'].constraint_indices == torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 4])).item()

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
        batch['dopant', 'coupled_to',
              'interaction'].edge_index == torch.cat((_edge_index1,
                                                      _edge_index2,
                                                      _edge_index3), dim=1)).item()

    _edge_index1 = data1['interaction', 'coupled_to', 'dopant'].edge_index
    _edge_index2 = data2['interaction', 'coupled_to', 'dopant'].edge_index
    _edge_index2[0, :] += data1['interaction'].num_nodes
    _edge_index2[1, :] += data1['dopant'].num_nodes
    _edge_index3 = data3['interaction', 'coupled_to', 'dopant'].edge_index
    _edge_index3[0, :] += data1['interaction'].num_nodes + data2[
        'interaction'].num_nodes
    _edge_index3[1, :] += data1['dopant'].num_nodes + data2['dopant'].num_nodes
    assert torch.all(
        batch['interaction', 'coupled_to',
              'dopant'].edge_index == torch.cat((_edge_index1, _edge_index2,
                                                 _edge_index3), dim=1)).item()
