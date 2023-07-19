from NanoParticleTools.machine_learning.data.processors import FeatureProcessor
from NanoParticleTools.inputs import SphericalConstraint, NanoParticleConstraint
import torch
from itertools import product
from monty.serialization import MontyDecoder

from torch_geometric.data.hetero_data import HeteroData, NodeOrEdgeStorage, EdgeStorage

from typing import List, Tuple, Any, Dict, Optional
import warnings


class DopantInteractionFeatureProcessor(FeatureProcessor):

    def __init__(self,
                 include_zeros: bool = False,
                 input_grad: bool = False,
                 **kwargs) -> None:
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        # yapf: enable

        self.elements_map = dict([(_t[1], _t[0])
                                  for _t in enumerate(self.possible_elements)])
        self.include_zeros = include_zeros
        self.input_grad = input_grad

    def process_doc(self, doc: Dict) -> Dict:
        data = {
            'dopant': {},
            'interaction': {},
            ('dopant', 'coupled_to', 'interaction'): {},
            ('interaction', 'coupled_to', 'dopant'): {},
        }

        # Build the dopant concentration
        dopant_concentration = []
        for constraint in doc['dopant_concentration']:
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
        constraints = doc['input']['constraints']
        constraints = MontyDecoder().process_decoded(constraints)
        constraints = constraints[:len(dopant_concentration)]

        _radii = [0]
        for constraint in constraints:
            _radii.append(constraint.radius)

        # Check for duplicate radii (these would cause 0 thickness layers)
        # must iterate in reverse order
        for i in range(len(_radii) - 1, 0, -1):
            if _radii[i - 1] == _radii[i]:
                # this idx is a zero thickness layer, remove it.
                _radii.pop(i)
                # Also remove this from the dopant concentration
                dopant_concentration.pop(i - 1)

        radii = torch.tensor(_radii,
                             dtype=torch.float32,
                             requires_grad=self.input_grad)
        radii_idx = torch.stack(
            (torch.arange(0,
                          len(radii) - 1), torch.arange(1, len(radii))),
            dim=1)
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
        dopant_indices = []
        edge_index_forward = []
        edge_index_backward = []
        interaction_counter = 0
        for i, type_i in enumerate(dopant_types):
            for j, type_j in enumerate(dopant_types):
                type_indices.append([type_i, type_j])
                dopant_indices.append([i, j])
                edge_index_forward.append([i, interaction_counter])
                edge_index_backward.append([interaction_counter, j])
                interaction_counter += 1
        data['interaction']['type_indices'] = torch.tensor(type_indices,
                                                           dtype=torch.long)
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
                    return self['constraint_radii_idx'].size(0)
                elif 'type_indices' in key:
                    return 0
                else:
                    return 0

        return CustomHeteroData


class HeteroDCVFeatureProcessor(FeatureProcessor):

    def __init__(self,
                 include_zeros: bool = False,
                 input_grad: bool = False,
                 **kwargs):
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        # yapf: enable

        self.elements_map = dict([(_t[1], _t[0])
                                  for _t in enumerate(self.possible_elements)])
        self.include_zeros = include_zeros
        self.input_grad = input_grad

    def process_doc(self, doc):
        data = {
            'dopant': {},
        }

        # Build the dopant concentration
        dopant_concentration = []
        for constraint in doc['dopant_concentration']:
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
        constraints = doc['input']['constraints']
        constraints = MontyDecoder().process_decoded(constraints)
        constraints = constraints[:len(dopant_concentration)]

        _radii = [0]
        for constraint in constraints:
            _radii.append(constraint.radius)

        if len(_radii) == 2:
            # minimum of 2 layers in this representation, else interaction nodes will be missing
            _radii.append(_radii[-1] + 1e-3)
            dopant_concentration.append(
                {el: 0
                 for el in self.possible_elements})

        # Check for duplicate radii (these would cause 0 thickness layers)
        # must iterate in reverse order
        for i in range(len(_radii) - 1, 0, -1):
            if _radii[i - 1] == _radii[i]:
                # this idx is a zero thickness layer, remove it.
                _radii.pop(i)
                # Also remove this from the dopant concentration
                dopant_concentration.pop(i - 1)

        radii = torch.tensor(_radii,
                             dtype=torch.float32,
                             requires_grad=self.input_grad)
        radii_idx = torch.stack(
            (torch.arange(0,
                          len(radii) - 1), torch.arange(1, len(radii))),
            dim=1)
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

        data['dopant']['x'] = torch.tensor(dopant_concs, dtype=torch.float32)
        data['dopant']['types'] = torch.tensor(dopant_types, dtype=torch.long)
        data['dopant']['constraint_indices'] = torch.tensor(
            dopant_constraint_indices, dtype=torch.long)
        data['dopant']['num_nodes'] = len(dopant_concs)

        # enumerate all possible interactions (dopants x dopants)
        interaction_type_indices = []
        interaction_dopant_indices = []
        interaction_edge_index_forward = []
        interaction_edge_index_backward = []
        interaction_counter = 0

        intraaction_type_indices = []
        intraaction_dopant_indices = []
        intraaction_edge_index_forward = []
        intraaction_edge_index_backward = []
        intraaction_counter = 0
        for i, type_i in enumerate(dopant_types):
            for j, type_j in enumerate(dopant_types):
                if j < i:
                    continue
                if dopant_constraint_indices[i] == dopant_constraint_indices[j]:  # yapf: disable
                    # This is an intra-layer interaction
                    intraaction_type_indices.append([type_i, type_j])
                    intraaction_dopant_indices.append([i, j])
                    intraaction_edge_index_forward.append(
                        [i, intraaction_counter])
                    intraaction_edge_index_backward.append(
                        [intraaction_counter, j])
                    intraaction_counter += 1
                else:
                    # This is an inter-layer interaction
                    interaction_type_indices.append([type_i, type_j])
                    interaction_dopant_indices.append([i, j])
                    interaction_edge_index_forward.append(
                        [i, interaction_counter])
                    interaction_edge_index_backward.append(
                        [interaction_counter, j])
                    interaction_counter += 1

        if interaction_counter > 0:
            data.update({
                'interaction': {},
                ('dopant', 'coupled_to', 'interaction'): {},
                ('interaction', 'coupled_to', 'dopant'): {}
            })

            data['interaction']['type_indices'] = torch.tensor(
                interaction_type_indices, dtype=torch.long)
            data['interaction']['dopant_indices'] = torch.tensor(
                interaction_dopant_indices, dtype=torch.long)
            data['dopant', 'coupled_to',
                 'interaction']['edge_index'] = torch.tensor(
                     interaction_edge_index_forward).reshape(-1, 2).transpose(
                         0, 1)
            data['interaction', 'coupled_to',
                 'dopant']['edge_index'] = torch.tensor(
                     interaction_edge_index_backward).reshape(-1, 2).transpose(
                         0, 1)
            # Keep track of increment (so we can combine graphs)
            data['dopant', 'coupled_to', 'interaction']['inc'] = torch.tensor(
                [len(dopant_concs), interaction_counter]).reshape(2, 1)
            data['interaction', 'coupled_to', 'dopant']['inc'] = torch.tensor(
                [interaction_counter, len(dopant_concs)]).reshape(2, 1)
            data['interaction']['num_nodes'] = interaction_counter

        if intraaction_counter > 0:
            data.update({
                'intraaction': {},
                ('dopant', 'coupled_to', 'intraaction'): {},
                ('intraaction', 'coupled_to', 'dopant'): {}
            })

            data['intraaction']['type_indices'] = torch.tensor(
                intraaction_type_indices, dtype=torch.long)
            data['intraaction']['dopant_indices'] = torch.tensor(
                intraaction_dopant_indices, dtype=torch.long)
            data['dopant', 'coupled_to',
                 'intraaction']['edge_index'] = torch.tensor(
                     intraaction_edge_index_forward).reshape(-1, 2).transpose(
                         0, 1)
            data['intraaction', 'coupled_to',
                 'dopant']['edge_index'] = torch.tensor(
                     intraaction_edge_index_backward).reshape(-1, 2).transpose(
                         0, 1)
            # Keep track of increment (so we can combine graphs)
            data['dopant', 'coupled_to', 'intraaction']['inc'] = torch.tensor(
                [len(dopant_concs), intraaction_counter]).reshape(2, 1)
            data['intraaction', 'coupled_to', 'dopant']['inc'] = torch.tensor(
                [intraaction_counter, len(dopant_concs)]).reshape(2, 1)
            data['intraaction']['num_nodes'] = intraaction_counter

        return data

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):

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
                    return self['constraint_radii_idx'].size(0)
                elif 'type_indices' in key:
                    return 0
                else:
                    return 0

        return CustomHeteroData
