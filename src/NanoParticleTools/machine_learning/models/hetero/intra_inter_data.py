from NanoParticleTools.machine_learning.models.hetero.data import DopantInteractionFeatureProcessor
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
import torch
from monty.serialization import MontyDecoder

from torch_geometric.data.hetero_data import HeteroData, NodeOrEdgeStorage, EdgeStorage

from typing import List, Any, Dict, Optional, Union, Tuple
from functools import lru_cache
from itertools import combinations_with_replacement
import numpy as np


class HeteroDCVFeatureProcessor(DopantInteractionFeatureProcessor):

    def inputs_from_constraints_and_concentration(
            self, input_constraints: List[SphericalConstraint],
            input_dopant_concentration: List[Dict]
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Preprocess the inputs and constraints to ensure they
        form valid graphs.

        Get the inputs from the concentration and constraints.
        the dopant concentration and input constraints should be
        of the same length.

        Args:
            input_constraints:
            input_dopant_concentration:

        """

        # Build the dopant concentration
        dopant_concentration = []
        for constraint in input_dopant_concentration:
            dopant_concentration.append({})
            for el in self.possible_elements:
                _conc = constraint.get(el, 0)
                if _conc > 0 or self.include_zeros:
                    dopant_concentration[-1][el] = _conc

        # Remove any layers that have 0 dopants
        # iterate through the dopant concentration in reverse order
        for i in range(len(dopant_concentration) - 1, -1, -1):
            if len(dopant_concentration[i]) == 0:
                dopant_concentration.pop(i)
            else:
                break

        # use the constraints to build a radii tensor
        constraints = input_constraints[:len(dopant_concentration)]

        _radii = [0] + [constraint.radius for constraint in constraints]

        # probability of augmenting the data
        if self.augment_data and np.random.rand() < self.augment_prob:
            # we can augment the data by picking a radius at random
            # between (0+eps) and the max radius and inserting it into the list

            for _ in range(self.augment_subdivisions):
                aug_radius = (0.90 * torch.rand(1) + 0.05) * _radii[-1]

                # find where it fits into the list
                for i, r in enumerate(_radii):
                    if r > aug_radius:
                        _radii.insert(i, aug_radius)

                        # additionally, duplicate the dopant concentration at this layer
                        dopant_concentration.insert(
                            i, dopant_concentration[i - 1])
                        break

        # Check for duplicate radii (these would cause 0 thickness layers)
        # must iterate in reverse order
        for i in range(len(_radii) - 1, 0, -1):
            if _radii[i - 1] == _radii[i]:
                # this idx is a zero thickness layer, remove it.
                _radii.pop(i)
                # Also remove this from the dopant concentration
                dopant_concentration.pop(i - 1)

        if len(_radii) == 2:
            # minimum of 2 layers in this representation, else interaction nodes will be missing
            _radii.append(_radii[-1] + 1e-3)
            dopant_concentration.append(
                {el: 0
                 for el in self.possible_elements})

        radii_without_zero = torch.tensor(_radii[1:],
                                          dtype=torch.float32,
                                          requires_grad=self.input_grad)
        return dopant_concentration, radii_without_zero

    def process_doc(self, doc: Dict) -> Dict:
        dopant_concentration = doc['dopant_concentration']

        constraints = doc['input']['constraints']
        constraints = MontyDecoder().process_decoded(constraints)

        dopant_concentration, radii_without_zero = self.inputs_from_constraints_and_concentration(
            constraints, dopant_concentration)

        return self.graph_from_inputs(dopant_concentration, radii_without_zero)

    def graph_from_inputs(self, dopant_concentration: Dict,
                          radii_without_zero: torch.Tensor) -> Dict:
        """
        Get a graph from the raw inputs

        Args:
            dopant_concentration: Dictionary containing the dopant concentration
                in each control volume.
            radii_without_zero: Outer radii of each control volume
        """

        data = {
            'dopant': {},
            'interaction': {},
            'intraaction': {},
            ('dopant', 'coupled_to', 'interaction'): {},
            ('interaction', 'coupled_to', 'dopant'): {},
            ('dopant', 'coupled_to', 'intraaction'): {},
            ('intraaction', 'coupled_to', 'dopant'): {},
        }

        radii = torch.cat(
            (torch.tensor([0.0], requires_grad=False), radii_without_zero),
            dim=0)
        radii_idx = torch.stack(
            (torch.arange(0,
                          len(radii) - 1), torch.arange(1, len(radii))),
            dim=1)
        data['radii_without_zero'] = radii_without_zero
        data['radii'] = radii
        data['constraint_radii_idx'] = radii_idx

        dopant_specifications = [
            (layer_idx, conc, el, None)
            for layer_idx, dopants in enumerate(dopant_concentration)
            for el, conc in dopants.items() if el in self.possible_elements
        ]

        dopant_concs = []
        dopant_types = []
        dopant_constraint_indices = []
        for constraint_indices, dopant_conc, dopant_el, _ in dopant_specifications:
            dopant_types.append(self.dopants_dict[dopant_el])
            dopant_concs.append(dopant_conc)
            dopant_constraint_indices.append(constraint_indices)

        data['dopant']['x'] = torch.tensor(dopant_concs,
                                           dtype=torch.float32,
                                           requires_grad=self.input_grad)
        data['dopant']['types'] = torch.tensor(dopant_types, dtype=torch.long)
        data['dopant']['constraint_indices'] = torch.tensor(
            dopant_constraint_indices, dtype=torch.long)
        data['dopant']['num_nodes'] = len(dopant_concs)

        # enumerate all possible interactions (dopants x dopants)
        interaction_type_indices = []
        interaction_types = []
        interaction_dopant_indices = []
        interaction_edge_index_forward = []
        interaction_edge_index_backward = []
        interaction_counter = 0

        intraaction_type_indices = []
        intraaction_types = []
        intraaction_dopant_indices = []
        intraaction_edge_index_forward = []
        intraaction_edge_index_backward = []
        intraaction_counter = 0
        for i, type_i in enumerate(dopant_types):
            for j, type_j in enumerate(dopant_types):
                if dopant_constraint_indices[i] == dopant_constraint_indices[j]:  # yapf: disable
                    # This is an intra-layer interaction
                    intraaction_type_indices.append([type_i, type_j])
                    intraaction_types.append(
                        self.edge_type_map[type_i][type_j])
                    intraaction_dopant_indices.append([i, j])
                    intraaction_edge_index_forward.append(
                        [i, intraaction_counter])
                    intraaction_edge_index_backward.append(
                        [intraaction_counter, j])
                    intraaction_counter += 1
                else:
                    # This is an inter-layer interaction
                    interaction_type_indices.append([type_i, type_j])
                    interaction_types.append(
                        self.edge_type_map[type_i][type_j])
                    interaction_dopant_indices.append([i, j])
                    interaction_edge_index_forward.append(
                        [i, interaction_counter])
                    interaction_edge_index_backward.append(
                        [interaction_counter, j])
                    interaction_counter += 1

        data['interaction']['type_indices'] = torch.tensor(
            interaction_type_indices, dtype=torch.long)
        data['interaction']['types'] = torch.tensor(interaction_types,
                                                    dtype=torch.long)
        data['interaction']['dopant_indices'] = torch.tensor(
            interaction_dopant_indices, dtype=torch.long)
        data['dopant', 'coupled_to',
             'interaction']['edge_index'] = torch.tensor(
                 interaction_edge_index_forward).reshape(-1,
                                                         2).transpose(0, 1)
        data['interaction', 'coupled_to',
             'dopant']['edge_index'] = torch.tensor(
                 interaction_edge_index_backward).reshape(-1,
                                                          2).transpose(0, 1)
        # Keep track of increment (so we can combine graphs)
        data['dopant', 'coupled_to', 'interaction']['inc'] = torch.tensor(
            [len(dopant_concs), interaction_counter]).reshape(2, 1)
        data['interaction', 'coupled_to', 'dopant']['inc'] = torch.tensor(
            [interaction_counter, len(dopant_concs)]).reshape(2, 1)
        data['interaction']['num_nodes'] = interaction_counter

        data['intraaction']['type_indices'] = torch.tensor(
            intraaction_type_indices, dtype=torch.long)
        data['intraaction']['types'] = torch.tensor(intraaction_types,
                                                    dtype=torch.long)
        data['intraaction']['dopant_indices'] = torch.tensor(
            intraaction_dopant_indices, dtype=torch.long)
        data['dopant', 'coupled_to',
             'intraaction']['edge_index'] = torch.tensor(
                 intraaction_edge_index_forward).reshape(-1,
                                                         2).transpose(0, 1)
        data['intraaction', 'coupled_to',
             'dopant']['edge_index'] = torch.tensor(
                 intraaction_edge_index_backward).reshape(-1,
                                                          2).transpose(0, 1)
        # Keep track of increment (so we can combine graphs)
        data['dopant', 'coupled_to', 'intraaction']['inc'] = torch.tensor(
            [len(dopant_concs), intraaction_counter]).reshape(2, 1)
        data['intraaction', 'coupled_to', 'dopant']['inc'] = torch.tensor(
            [intraaction_counter, len(dopant_concs)]).reshape(2, 1)
        data['intraaction']['num_nodes'] = intraaction_counter

        return data
