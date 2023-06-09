from NanoParticleTools.machine_learning.data.processors import FeatureProcessor
from NanoParticleTools.inputs import SphericalConstraint, NanoParticleConstraint
import torch
from itertools import product
from monty.serialization import MontyDecoder

from torch_geometric.data.hetero_data import HeteroData

from typing import List, Tuple, Any, Dict, Optional


class HeteroFeatureProcessor(FeatureProcessor):

    def __init__(self, include_global_node=False, **kwargs):
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.elements_map = dict([(_t[1], _t[0])
                                  for _t in enumerate(self.possible_elements)])

    def process_doc(self, doc):
        data = {
            'constraint_nodes': {},
            'dopant_nodes': {},
            ('dopant_nodes', 'interacts_with', 'dopant_nodes'): {},
            ('dopant_nodes', 'in_constraints', 'constraint_nodes'): {},
            ('constraint_nodes', 'adjacent_to', 'constraint_nodes'): {}
        }

        constraints = doc['input']['constraints']
        dopant_concentration = doc['dopant_concentration']
        dopant_specifications = [
            (layer_idx, conc, el, None)
            for layer_idx, dopants in enumerate(dopant_concentration)
            for el, conc in dopants.items() if el in self.possible_elements
        ]

        constraints = MontyDecoder().process_decoded(constraints)

        # Generate a map of layers which are occupied by dopants
        # This will be crucial in ensuring inactive layers are not explicitly included
        # Although this brings up a question about whether or not it should be included
        constraint_layer_map = {}
        constraint_counter = 0
        for i, _, _, _ in dopant_specifications:
            if i not in constraint_layer_map:
                constraint_layer_map[i] = constraint_counter
                constraint_counter += 1

        # Build and add the contraints to the HeteroData
        constraint_tensors = []
        for key, value in constraint_layer_map.items():
            inner_radius = constraints[key - 1].radius if key > 0 else 0
            outer_radius = constraints[key].radius

            constraint_tensors.append(
                torch.tensor([inner_radius, outer_radius]))
        data['constraint_nodes']['radii'] = torch.stack(constraint_tensors)
        data['constraint_nodes']['num_nodes'] = len(constraint_tensors)

        # Add the dopants to the HeteroData
        data['dopant_nodes']['x'] = torch.tensor(
            [conc for _, conc, _, _ in dopant_specifications])
        data['dopant_nodes']['types'] = torch.tensor([
            self.elements_map[dopant_el]
            for i, _, dopant_el, _ in dopant_specifications
        ])
        data['dopant_nodes']['num_nodes'] = len(dopant_specifications)

        # Build the edges between dopants
        # The dopants are fully connected
        edge_index = torch.tensor(
            list(
                product(range(data['dopant_nodes']['x'].size(0)),
                        range(data['dopant_nodes']['x'].size(0))))).T
        data['dopant_nodes', 'interacts_with',
             'dopant_nodes']['edge_index'] = edge_index
        # data['dopant_nodes', 'interacts_with', 'dopant_nodes'].edge_attr

        # Build the edges between dopant and constraint
        dopant_edge_i = torch.arange(0, data['dopant_nodes']['x'].size(0))
        dopant_edge_j = torch.tensor(
            [constraint_layer_map[i] for i, _, _, _ in dopant_specifications])
        edge_index = torch.stack((dopant_edge_i, dopant_edge_j))
        data['dopant_nodes', 'in_constraints',
             'constraint_nodes']['edge_index'] = edge_index

        # Build the edges between constraints
        # Constraints are fully connected
        edge_index = torch.tensor(
            list(
                product(range(data['constraint_nodes']['radii'].size(0)),
                        range(data['constraint_nodes']['radii'].size(0))))).T
        data['constraint_nodes', 'adjacent_to',
             'constraint_nodes']['edge_index'] = edge_index
        # Can compute the edge attribute using:
        # TODO: consider connections based on distance
        # (If a constraint is further than a min, then it won't interact)
        # sender, receiver = data['constraint_nodes'].radii[edge_index]
        # Compute the distances between permutations of
        # edge_attr = torch.cat((receiver - sender, receiver - sender.flip(1)), dim=1).abs()
        # pruned_edge_index = edge_index[:, (edge_attr < 30).sum(-1).bool()]

        return data

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return HeteroData


class DopantInteractionFeatureProcessor(FeatureProcessor):

    def __init__(self, separate_self_interaction=False, **kwargs):
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.elements_map = dict([(_t[1], _t[0])
                                  for _t in enumerate(self.possible_elements)])
        self.separate_self_interaction = separate_self_interaction

    def process_doc(self, doc):
        data = {
            'dopant': {},
            'interaction': {},
            ('dopant', 'coupled_to', 'interaction'): {},
            ('interaction', 'coupled_to', 'dopant'): {},
        }
        if self.separate_self_interaction:
            data['self_interaction'] = {}
            data[('dopant', 'coupled_to', 'self_interaction')] = {}
            data[('self_interaction', 'coupled_to', 'dopant')] = {}

        dopant_concentration = [
            el for el in doc['dopant_concentration'] if len(el) > 0
        ]
        dopant_specifications = [
            (layer_idx, conc, el, None)
            for layer_idx, dopants in enumerate(dopant_concentration)
            for el, conc in dopants.items() if el in self.possible_elements
        ]

        constraints = doc['input']['constraints']
        constraints = MontyDecoder().process_decoded(constraints)

        # use the constraints to build a radii tensor
        _radii = [
            self.get_radii(i, constraints) for i in range(len(constraints))
            if len(doc['dopant_concentration'][i]) > 0
        ]
        radii = torch.tensor(_radii, dtype=torch.float32, requires_grad=True)
        data['constraint_radii'] = radii

        dopant_concs = []
        dopant_types = []
        dopant_constraint_indices = []
        for constraint_indices, dopant_conc, dopant_el, _ in dopant_specifications:
            dopant_types.append(self.dopants_dict[dopant_el])
            dopant_concs.append(dopant_conc)
            dopant_constraint_indices.append(constraint_indices)

        data['dopant']['x'] = torch.tensor(dopant_concs)
        data['dopant']['types'] = torch.tensor(dopant_types)
        data['dopant']['constraint_indices'] = torch.tensor(
            dopant_constraint_indices)
        data['dopant']['num_nodes'] = len(dopant_concs)

        # enumerate all possible interactions (dopants x dopants)
        type_indices = []
        dopant_indices = []
        edge_index_forward = []
        edge_index_backward = []
        interaction_counter = 0
        for i, type_i in enumerate(dopant_types):
            for j, type_j in enumerate(dopant_types):
                if self.separate_self_interaction and i == j:
                    continue
                type_indices.append([type_i, type_j])
                dopant_indices.append([i, j])
                edge_index_forward.append([i, interaction_counter])
                edge_index_backward.append([interaction_counter, j])
                interaction_counter += 1
        data['interaction']['type_indices'] = torch.tensor(type_indices)
        data['interaction']['dopant_indices'] = torch.tensor(dopant_indices)
        data['dopant', 'coupled_to',
             'interaction']['edge_index'] = torch.tensor(
                 edge_index_forward).reshape(-1, 2).transpose(0, 1)
        data['interaction', 'coupled_to',
             'dopant']['edge_index'] = torch.tensor(
                 edge_index_backward).reshape(-1, 2).transpose(0, 1)
        # Keep track of increment (so we can combine graphs)
        data['dopant', 'coupled_to', 'interaction']['inc'] = torch.tensor(
            [len(dopant_concs), interaction_counter]).reshape(2, 1)
        data['interaction', 'coupled_to', 'dopant']['inc'] = torch.tensor(
            [interaction_counter, len(dopant_concs)]).reshape(2, 1)
        data['interaction']['num_nodes'] = interaction_counter

        if self.separate_self_interaction:
            type_indices = []
            dopant_indices = []
            edge_index_forward = []
            edge_index_backward = []
            interaction_counter = 0
            for i, type_i in enumerate(dopant_types):
                type_indices.append(type_i)
                dopant_indices.append(i)
                edge_index_forward.append([i, interaction_counter])
                edge_index_backward.append([interaction_counter, i])
                interaction_counter += 1
            data['self_interaction']['type_indices'] = torch.tensor(
                type_indices)
            data['self_interaction']['dopant_indices'] = torch.tensor(
                dopant_indices)
            data['dopant', 'coupled_to',
                 'self_interaction']['edge_index'] = torch.tensor(
                     edge_index_forward).reshape(-1, 2).transpose(0, 1)
            data['self_interaction', 'coupled_to',
                 'dopant']['edge_index'] = torch.tensor(
                     edge_index_backward).reshape(-1, 2).transpose(0, 1)
            # Keep track of increment (so we can combine graphs)
            data['dopant', 'coupled_to',
                 'self_interaction']['inc'] = torch.tensor(
                     [len(dopant_concs), interaction_counter]).reshape(2, 1)
            data['self_interaction', 'coupled_to',
                 'dopant']['inc'] = torch.tensor(
                     [interaction_counter,
                      len(dopant_concs)]).reshape(2, 1)
            data['self_interaction']['num_nodes'] = len(dopant_concs)

        return data

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        from torch_geometric.data.hetero_data import HeteroData, NodeOrEdgeStorage, EdgeStorage

        class CustomHeteroData(HeteroData):

            def __cat_dim__(self,
                            key: str,
                            value: Any,
                            store: Optional[NodeOrEdgeStorage] = None,
                            *args,
                            **kwargs) -> Any:
                if 'indices' in key:
                    return 0
                if 'index' in key:
                    return 1
                return 0

            def __inc__(self,
                        key: str,
                        value: Any,
                        store: Optional[NodeOrEdgeStorage] = None,
                        *args,
                        **kwargs) -> Any:
                if isinstance(store, EdgeStorage) and 'index' in key:
                    return store['inc']
                elif 'dopant_indices' in key:
                    return self['dopant'].num_nodes
                elif 'constraint_indices' in key:
                    return self['constraint_radii'].size(0)
                elif 'type_indices' in key:
                    return 0
                else:
                    return 0

        return CustomHeteroData
