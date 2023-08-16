from NanoParticleTools.machine_learning.data import FeatureProcessor as BaseFeatureProcessor
from typing import List
from torch_geometric.data.data import Data

import numpy as np
import torch
from monty.json import MontyDecoder

from typing import Union


class MLPFeatureProcessor(BaseFeatureProcessor):

    def __init__(self,
                 max_layers: int = 4,
                 **kwargs):
        """
        Args:
            max_layers: The maximum number of layers to featurize.
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        # yapf: enable

        self.max_layers = max_layers

    def process_doc(self, doc: dict) -> torch.Tensor:
        dopant_concentration = doc['dopant_concentration']

        constraints = doc['input']['constraints']
        constraints = MontyDecoder().process_decoded(constraints)

        # Truncate layers to max_layers
        if len(constraints) > self.max_layers:
            constraints = constraints[:self.max_layers]
        if len(dopant_concentration) > self.max_layers:
            dopant_concentration = dopant_concentration[:self.max_layers]

        # Add empty layers to fill up to max_layers
        while len(constraints) < self.max_layers:
            constraints.append(constraints[-1])
        while len(dopant_concentration) < self.max_layers:
            dopant_concentration.append({})

        radii_without_zero = torch.tensor([c.radius for c in constraints],
                                          dtype=torch.float32,
                                          requires_grad=self.input_grad)
        radii = torch.cat((torch.tensor([0.0], requires_grad=False), radii_without_zero), dim=0)
        concentrations = []
        for c in dopant_concentration:
            for el in self.possible_elements:
                concentrations.append(c.get(el, 0))

        return {
            'x':
            torch.tensor(concentrations,
                         dtype=torch.float32,
                         requires_grad=self.input_grad).unsqueeze(0),
            'radii': radii.unsqueeze(0),
            'radii_without_zero':
            radii_without_zero.unsqueeze(0),
        }

    @property
    def is_graph(self):
        # TODO: Look into deleting this
        return True

    @property
    def data_cls(self):
        return Data


class TabularFeatureProcessor(MLPFeatureProcessor):

    def __init__(self,
                 include_volume: bool = False,
                 volume_normalization: str = 'div',
                 volume_normalization_const: float = 1e6,
                 **kwargs):
        super().__init__(**kwargs)
        self.include_volume = include_volume
        self.volume_normalization = volume_normalization
        self.volume_normalization_const = volume_normalization_const

    def process_doc(self, doc: dict) -> torch.Tensor:
        out_dict = super().process_doc(doc)

        # unpack values
        x = out_dict['x']
        radii = out_dict['radii']
        radii_without_zero = out_dict['radii_without_zero']

        if self.include_volume:
            volume = 4 / 3 * torch.pi * (radii[..., 1:]**3 - radii[..., :-1]**3)
            if self.volume_normalization == 'div':
                volume = volume / self.volume_normalization_const
            elif self.volume_normalization == 'log':
                volume = volume.add(self.volume_normalization_const).log10()

            out = torch.cat([x, radii_without_zero, volume], dim=1)
            return {'x': out}
        else:
            out = torch.cat([x, radii_without_zero], dim=1)
            return {'x': out}
