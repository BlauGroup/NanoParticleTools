from NanoParticleTools.machine_learning.data.processors import FeatureProcessor
from NanoParticleTools.inputs import SphericalConstraint, NanoParticleConstraint
import torch
from itertools import product
from monty.serialization import MontyDecoder

from torch_geometric.data.hetero_data import HeteroData

from typing import List, Tuple, Any, Dict, Optional


class HeteroFeatureProcessor(FeatureProcessor):

    def __init__(self, possible_elements=['Yb', 'Er', 'Nd'], 
                 include_global_node = False, **kwargs):
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        self.possible_elements = possible_elements
        self.elements_map = dict([(_t[1], _t[0])
                                  for _t in enumerate(possible_elements)])

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
        if not isinstance(constraints[0], NanoParticleConstraint):
            decoder = MontyDecoder()
            constraints = decoder.process_decoded(constraints)

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