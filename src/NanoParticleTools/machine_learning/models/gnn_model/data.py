from NanoParticleTools.inputs import SphericalConstraint
from NanoParticleTools.machine_learning.data.processors import FeatureProcessor

import torch
from torch_geometric.data.data import Data
from monty.json import MontyDecoder

import numpy as np
import random


class GraphInteractionFeatureProcessor(FeatureProcessor):

    def __init__(self,
                 include_zeros: bool = False,
                 input_grad: bool = False,
                 **kwargs):
        """
        Args:
            possible_elements: All dopant elements that may exist in the data
            edge_attr_bias: A bias added to the edge_attr before
                applying 1/edge_attr. This serves to eliminate divide by zero and inf in the tensor.
                Additionally, it acts as a weight on the self-interaction. Defaults to 0.5.
            interaction_sigma: The sigma parameter for the gaussian interaction. Defaults to 20.0.
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        # yapf: enable

        self.include_zeros = include_zeros
        self.input_grad = input_grad

    def inputs_from_constraints_and_concentration(
            self, input_constraints: list[SphericalConstraint],
            input_dopant_concentration: list[dict]
    ) -> tuple[dict, torch.Tensor]:
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

        # Check for duplicate radii (these would cause 0 thickness layers)
        # must iterate in reverse order
        for i in range(len(_radii) - 1, 0, -1):
            if _radii[i - 1] == _radii[i]:
                # this idx is a zero thickness layer, remove it.
                _radii.pop(i)
                # Also remove this from the dopant concentration
                dopant_concentration.pop(i - 1)

        radii_without_zero = torch.tensor(_radii[1:],
                                          dtype=torch.float32,
                                          requires_grad=self.input_grad)
        return dopant_concentration, radii_without_zero

    def process_doc(self, doc: dict) -> dict:
        dopant_concentration = doc['dopant_concentration']

        constraints = doc['input']['constraints']
        constraints = MontyDecoder().process_decoded(constraints)

        dopant_concentration, radii_without_zero = self.inputs_from_constraints_and_concentration(
            constraints, dopant_concentration)

        return self.graph_from_inputs(dopant_concentration, radii_without_zero)

    def graph_from_inputs(self, dopant_concentration: dict,
                          radii_without_zero: torch.Tensor) -> dict:
        """
        Get a graph from the raw inputs

        Args:
            dopant_concentration: Dictionary containing the dopant concentration
                in each control volume.
            radii_without_zero: Outer radii of each control volume
        """
        data = {}

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

        data['x'] = torch.tensor(dopant_concs,
                                 dtype=torch.float32,
                                 requires_grad=self.input_grad)
        data['types'] = torch.tensor(dopant_types, dtype=torch.long)
        data['constraint_indices'] = torch.tensor(dopant_constraint_indices,
                                                  dtype=torch.long)
        data['num_nodes'] = len(dopant_concs)

        edge_index = []
        for i, _ in enumerate(dopant_types):
            for j, _ in enumerate(dopant_types):
                # Every pair of nodes is a interaction
                edge_index.append([i, j])
                edge_index.append([j, i])

        data['edge_index'] = torch.tensor(edge_index,
                                          dtype=torch.long).transpose(0, 1)
        data['num_edges'] = len(edge_index)
        return data

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return Data
