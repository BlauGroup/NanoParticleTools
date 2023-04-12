from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint
from NanoParticleTools.machine_learning.models._data import DataProcessor

from monty.serialization import MontyDecoder
from typing import List, Tuple, Any, Dict
from itertools import combinations_with_replacement
from functools import lru_cache
import torch

from torch_geometric.data.data import Data
from torch_geometric.typing import SparseTensor


class InteractionData(Data):

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif key == 'node_dopant_index':
            return 0
        elif key == 'x_layer_idx':
            return 0
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif key == 'node_dopant_index':
            return self.x_dopant.size(0)
        elif key == 'x_layer_idx':
            return self.radii.size(0)
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0


class GraphFeatureProcessor(DataProcessor):
    """_summary_

    Args:
        possible_elements (List[str], optional): _description_. Defaults to ['Yb', 'Er', 'Nd'].
        log_volume (Optional[bool], optional): Whether to apply a log10 to the volume to reduce
            orders of magnitude. Defaults to False.
    """
    possible_elements: List[str]
    dopants_dict: Dict[str, int]
    log_volume: bool
    fully_connected: bool

    def __init__(self,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 log_volume: bool = False,
                 fully_connected: bool = True,
                 **kwargs):
        super().__init__(
            fields=['formula_by_constraint', 'dopant_concentration', 'input'],
            **kwargs)

        self.possible_elements = possible_elements
        self.dopants_dict = {
            key: i
            for i, key in enumerate(self.possible_elements)
        }
        self.log_volume = log_volume
        self.fully_connected = fully_connected

    @property
    @lru_cache
    def edge_type_map(self):
        edge_type_map = {}
        for i, (el1, el2) in enumerate(
                list(combinations_with_replacement(self.possible_elements,
                                                   2))):
            try:
                edge_type_map[el1][el2] = i
            except KeyError:
                edge_type_map[el1] = {el2: i}

            try:
                edge_type_map[el2][el1] = i
            except KeyError:
                edge_type_map[el2] = {el1: i}
        return edge_type_map

    def get_node_features(self, constraints,
                          dopant_specifications) -> torch.Tensor:
        """
        In this graph representation, we use atom pairs as the nodes.
        We will call this the dopant-interaction basis.

        We create a nodes that correspond to the interaction between the pairs of dopant.
        Node attributes correspond to the following:
        1. Identity of element # [n_nodes, 1]
        2. Composition of both dopants # [n_nodes, 2]
        3. Radii of layer containing both dopants # [n_nodes, 2, 2]

        The "node" attributes can be obtained by indexing the node attributes as follows:
        `data['radii_dopant'][data['node_dopant_ids'].T]`

        `data['x_dopant'][data['node_dopant_ids'].T]`
        This is to ensure that gradients can be calculated with respect to the dopant
        concentrations and the layer radii.
        """

        types = []
        node_dopant_ids = []

        # Not all constraints are filled. Here we create a map from the constraint index
        # to the index of the filled constraint.
        constraint_index_map = {
            key: i
            for i, key in enumerate(
                sorted(list(set([spec[0] for spec in dopant_specifications]))))
        }

        for i, (constraint_i, x_i, el_i,
                _) in enumerate(dopant_specifications):
            for j, (constraint_j, x_j, el_j,
                    _) in enumerate(dopant_specifications):

                types.append(self.edge_type_map[el_i][el_j])
                node_dopant_ids.append([i, j])

        return {
            'x_dopant':
            torch.tensor([
                x_i for _, (_, x_i, _, _) in enumerate(dopant_specifications)
            ]),
            'x_layer_idx':
            torch.tensor([
                constraint_index_map[constraint_i]
                for _, (constraint_i, _, _,
                        _) in enumerate(dopant_specifications)
            ]),
            'types':
            torch.tensor(types),
            'radii':
            torch.tensor([
                self.get_radii(constraint_i, constraints)
                for constraint_i, key in constraint_index_map.items()
            ]),
            # 'radii': torch.tensor(radii).float(),
            'node_dopant_index':
            torch.tensor(node_dopant_ids)
        }

    def get_edge_features(self, num_nodes: int,
                          node_dopant_ids: torch.Tensor) -> torch.Tensor:
        """
        Build the fully connected edge connections.

        Two interactions are only connected if they share a common dopant(layer dependent)
        """
        # Slower python based version
        # edge_index = []
        # for _i, (i, j) in enumerate(node_dopant_ids):
        #     for _j, (k, l) in enumerate(node_dopant_ids):
        #         if i == k or i == l or j == k or j == l:
        #             edge_index.append([_i, _j])

        x = torch.arange(0, num_nodes)
        i, j = torch.cartesian_prod(x, x).T

        if self.fully_connected:
            edge_index = torch.stack((i, j))
        else:
            edge_shared_bool = node_dopant_ids[i] == node_dopant_ids[j]
            edge_shared_bool_flip = node_dopant_ids[i] == node_dopant_ids[
                j].flip(1)

            edge_share_bool = torch.logical_or(edge_shared_bool,
                                               edge_shared_bool_flip)
            edge_share_bool = torch.logical_or(edge_share_bool[:, 0],
                                               edge_share_bool[:, 1])
            edge_index = torch.stack((i[edge_share_bool], j[edge_share_bool]))
        return {'edge_index': edge_index}

    def get_data_graph(self, constraints: List[NanoParticleConstraint],
                       dopant_specifications: List[Tuple[int, float, str,
                                                         str]]):

        output_dict = self.get_node_features(constraints,
                                             dopant_specifications)
        output_dict.update(
            self.get_edge_features(output_dict['types'].size(0),
                                   output_dict['node_dopant_index']))

        return output_dict

    def process_doc(self, doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']

        if not isinstance(constraints[0], NanoParticleConstraint):
            decoder = MontyDecoder()
            constraints = decoder.process_decoded(constraints)

        return self.get_data_graph(constraints, dopant_specifications)

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return InteractionData


class CompleteGraphFeatureProcessor(GraphFeatureProcessor):

    def process_doc(self, doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']

        if not isinstance(constraints[0], NanoParticleConstraint):
            decoder = MontyDecoder()
            constraints = decoder.process_decoded(constraints)

        # Fill in the empty/missing dopants
        _dopant_dict = {
            i: {el: None
                for el in self.possible_elements}
            for i, _ in enumerate(constraints)
        }
        for spec in dopant_specifications:
            _dopant_dict[spec[0]][spec[2]] = spec

        for layer_i in _dopant_dict:
            for el in _dopant_dict[layer_i]:
                if _dopant_dict[layer_i][el] is None:
                    _dopant_dict[layer_i][el] = (layer_i, 0, el, 'Y')

        dopant_specifications = [
            _dopant_dict[k1][k2] for k1 in _dopant_dict.keys()
            for k2 in _dopant_dict[k1].keys()
        ]

        return self.get_data_graph(constraints, dopant_specifications)
