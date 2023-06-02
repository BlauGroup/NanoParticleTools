from NanoParticleTools.machine_learning.data import FeatureProcessor as BaseFeatureProcessor
from typing import List
from torch_geometric.data.data import Data

import numpy as np
import torch
from monty.json import MontyDecoder


class FeatureProcessor(BaseFeatureProcessor):

    def __init__(self,
                 max_layers: int = 4,
                 **kwargs):
        """
        :param max_layers:
        :param possible_elements:
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.max_layers = max_layers

    def process_doc(self, doc: dict) -> torch.Tensor:
        constraints = self.get_item_from_doc(doc, 'input.constraints')
        dopant_concentration = self.get_item_from_doc(doc,
                                                      'dopant_concentration')

        constraints = MontyDecoder().process_decoded(constraints)

        # Construct the feature array
        feature = []
        for layer in range(self.max_layers):
            _layer_feature = []
            try:
                _layer_feature.append(constraints[layer]['radius'])
            except IndexError:
                _layer_feature.append(0)
            for el in self.possible_elements:
                try:
                    _layer_feature.append(dopant_concentration[layer][el] *
                                          100)
                except KeyError:
                    _layer_feature.append(0)
            feature.append(_layer_feature)
        return {'x': torch.tensor(np.hstack(feature)).float()}

    def __str__(self):
        return (f"Feature Processor - {self.max_layers}"
                f"x [radius, x_{', x_'.join(self.possible_elements)}]")

    @property
    def is_graph(self):
        return False

    @property
    def data_cls(self):
        return Data


class VolumeFeatureProcessor(BaseFeatureProcessor):

    def __init__(self,
                 max_layers: int = 4,
                 **kwargs):
        """
        :param max_layers: Maximum number of layers to featurize.
        :param possible_elements: The elements which are present in the
            lanthanide nanoparticle dataset.
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input',
            'metadata'
        ], **kwargs)
        # yapf: enable

        self.max_layers = max_layers

    def process_doc(self, doc: dict) -> torch.Tensor:
        constraints = doc['input']['constraints']
        dopant_concentration = doc['dopant_concentration']

        constraints = MontyDecoder().process_decoded(constraints)

        # Construct the feature array
        feature = []
        r_lower_bound = 0
        for layer in range(self.max_layers):
            _layer_feature = []
            try:
                radius = constraints[layer].radius

                volume = 4 / 3 * np.pi * (radius**3 - r_lower_bound**3)
                r_lower_bound = radius
                _layer_feature.append(radius)
                _layer_feature.append(volume / 1000000)
            except IndexError:
                # This layer does not exist, its radius and volume are 0
                _layer_feature.append(0)
                _layer_feature.append(0)
            for el in self.possible_elements:
                try:
                    _layer_feature.append(dopant_concentration[layer][el] *
                                          100)
                except (KeyError, IndexError):
                    # This element is not present in this layer or this
                    # layer does not exist, its concentration is therefore 0
                    _layer_feature.append(0)
            feature.append(_layer_feature)
        return {'x': torch.tensor(np.hstack(feature)).unsqueeze(0).float()}

    def __str__(self):
        return (
            f"Feature Processor - {self.max_layers}"
            f" x [radius, volume, x_{', x_'.join(self.possible_elements)}]")

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return Data
