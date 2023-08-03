from NanoParticleTools.machine_learning.data.processors import FeatureProcessor
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
from monty.serialization import MontyDecoder

from torch_geometric.data.hetero_data import HeteroData, NodeOrEdgeStorage, EdgeStorage

from typing import List, Any, Dict, Optional, Union, Tuple
from functools import lru_cache
from itertools import combinations_with_replacement
import numpy as np
import torch
import random

class NPHeteroData(HeteroData):

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
        elif key == 'constraint_radii_idx':
            return self.radii.size(0)
        elif 'constraint_indices' in key:
            return self['constraint_radii_idx'].size(0)
        elif 'type_indices' in key:
            return 0
        else:
            return 0


class DopantInteractionFeatureProcessor(FeatureProcessor):

    def __init__(self,
                 include_zeros: bool = False,
                 input_grad: bool = False,
                 asymmetric_interaction: bool = False,
                 augment_data: bool = False,
                 augment_prob: float = 0.5,
                 augment_subdivisions: int = 2,
                 distribute_subdivisions: bool = False,
                 **kwargs) -> None:
        """

        Args:
            include_zeros: Whether to include zero dopant concentrations in the
                representation.
                - If False, the representation will only include:
                    i) dopants which have a non-zero concentration
                    ii) constraints which have dopants present
                - If True, the representation will include:
                    i) All possible dopants (even if they have zero concentration)
                    ii) Empty constraints if they are "internal" (such that last (external)
                    control volume must be occupied
            input_grad: Whether to .
            asymmetric_interaction: .
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        # yapf: enable

        self.elements_map = dict([(_t[1], _t[0])
                                  for _t in enumerate(self.possible_elements)])
        self.include_zeros = include_zeros
        self.input_grad = input_grad
        self.asymmetric_interaction = asymmetric_interaction
        self.augment_data = augment_data
        self.augment_prob = augment_prob
        self.augment_subdivisions = augment_subdivisions
        self.distribute_subdivisions = distribute_subdivisions
    @property
    @lru_cache
    def edge_type_map(self):
        edge_type_map = {}
        if self.asymmetric_interaction:
            type_counter = 0
            for i in range(len(self.possible_elements)):
                for j in range(len(self.possible_elements)):
                    try:
                        edge_type_map[i][j] = type_counter
                    except KeyError:
                        edge_type_map[i] = {j: type_counter}
                    type_counter += 1
        else:
            for type_counter, (i, j) in enumerate(
                    list(
                        combinations_with_replacement(
                            range(len(self.possible_elements)), 2))):
                try:
                    edge_type_map[i][j] = type_counter
                except KeyError:
                    edge_type_map[i] = {j: type_counter}

                try:
                    edge_type_map[j][i] = type_counter
                except KeyError:
                    edge_type_map[j] = {i: type_counter}
        return edge_type_map

    def inputs_from_concentration_and_constraints(
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
        
        # randomly assign number of subdivisions within range
        if self.distribute_subdivisions: 
            num_subdivisions = random.randint(0, self.augment_subdivisions)
        else:
            num_subdivisions = self.augment_subdivisions
        # probability of augmenting the data
        if self.augment_data and (np.random.rand() < self.augment_prob):
            # we can augment the data by picking a radius at random
            # between (0+eps) and the max radius and inserting it into the list

            for _ in range(num_subdivisions):
                aug_radius = torch.rand(1) * _radii[-1] + 1e-5

                # find where it fits into the list
                for i, r in enumerate(_radii):
                    if r > aug_radius:
                        _radii.insert(i, aug_radius)

                        # additionally, duplicate the dopant concentration at this layer
                        dopant_concentration.insert(i, dopant_concentration[i - 1])
                        break

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

    def process_doc(self, doc: Dict) -> Dict:
        dopant_concentration = doc['dopant_concentration']

        constraints = doc['input']['constraints']
        # constraints = MontyDecoder().process_decoded(constraints)

        dopant_concentration, radii_without_zero = self.inputs_from_concentration_and_constraints(
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
            ('dopant', 'coupled_to', 'interaction'): {},
            ('interaction', 'coupled_to', 'dopant'): {},
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
        type_indices = []
        types = []
        dopant_indices = []
        edge_index_forward = []
        edge_index_backward = []
        interaction_counter = 0
        for i, type_i in enumerate(dopant_types):
            for j, type_j in enumerate(dopant_types):
                type_indices.append([type_i, type_j])
                types.append(self.edge_type_map[type_i][type_j])
                dopant_indices.append([i, j])
                edge_index_forward.append([i, interaction_counter])
                edge_index_backward.append([interaction_counter, j])
                interaction_counter += 1

        data['interaction']['type_indices'] = torch.tensor(type_indices,
                                                           dtype=torch.long)
        data['interaction']['types'] = torch.tensor(types, dtype=torch.long)
        data['interaction']['dopant_indices'] = torch.tensor(dopant_indices,
                                                             dtype=torch.long)
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

        return data

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return NPHeteroData
