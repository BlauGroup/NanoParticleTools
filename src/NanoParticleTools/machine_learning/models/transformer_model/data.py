from NanoParticleTools.machine_learning.data.processors import DataProcessor
import torch
from typing import List
from torch_geometric.data.data import Data
from monty.json import MontyDecoder


class TransformerFeatureProcessor(DataProcessor):

    SPECIES_TYPE_INDEX = 0
    COMPOSITION_INDEX = 1
    VOLUME_INDEX = 2

    def __init__(self,
                 max_layers: int = 4,
                 volume_scale_factor: float = 1e-6,
                 **kwargs):
        """
        :param max_layers:
        :param possible_elements:
        """
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration',
            'input.constraints', 'metadata'
        ], **kwargs)

        self.max_layers = max_layers
        self.volume_scale_factor = volume_scale_factor

    def process_doc(self, doc: dict) -> torch.Tensor:
        constraints = self.get_item_from_doc(doc, 'input.constraints')
        dopant_concentration = self.get_item_from_doc(doc,
                                                      'dopant_concentration')

        constraints = MontyDecoder().process_decoded(constraints)

        types = torch.tensor([
            j for i in range(self.max_layers)
            for j in range(len(self.possible_elements))
        ])

        volumes = []
        compositions = []
        r_lower_bound = 0
        for layer in range(self.max_layers):
            try:
                if isinstance(constraints[layer], dict):
                    radius = constraints[layer]['radius']
                else:
                    radius = constraints[layer].radius
                volume = self.get_volume(radius) - self.get_volume(
                    r_lower_bound)
                r_lower_bound = radius
                for _ in range(len(self.possible_elements)):
                    volumes.append(volume * self.volume_scale_factor)
            except IndexError:
                for _ in range(len(self.possible_elements)):
                    volumes.append(0)

            for el in self.possible_elements:
                try:
                    compositions.append(dopant_concentration[layer][el])
                except KeyError:
                    compositions.append(0)

        return {
            'x':
            torch.vstack(
                [types,
                 torch.tensor(volumes),
                 torch.tensor(compositions)])
        }

    @property
    def is_graph(self):
        return False

    @property
    def data_cls(self):
        return Data
