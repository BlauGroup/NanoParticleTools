from NanoParticleTools.differential_kinetics import (get_templates,
                                                     run_one_rate_eq,
                                                     save_data_to_hdf5)
from NanoParticleTools.species_data import Dopant
from maggma.core import Builder, Store
from argparse import ArgumentParser
from h5py import File

from typing import Iterator


class DifferentialKinetics(Builder):
    """
    Builder that processes and averages NPMC documents
    """

    def __init__(self,
                 num_samples: int,
                 excitation_wavelength: list[float] = None,
                 excitation_power: list[float] = None,
                 possible_dopants: list[str] = None,
                 max_dopants: int = 4,
                 include_spectra: bool = True,
                 output_file: str = 'out.h5',
                 max_data_per_group: int = 100000,
                 **kwargs):
        if excitation_wavelength is None:
            excitation_wavelength = [500.0, 1500.0]
        if excitation_power is None:
            excitation_power = [10.0, 100000.0]
        if possible_dopants is None:
            possible_dopants = ['Yb', 'Er', 'Tm', 'Nd', 'Ho', 'Eu', 'Sm', 'Dy']

        self.num_samples = num_samples
        self.excitation_wavelength = excitation_wavelength
        self.excitation_power = excitation_power
        self.possible_dopants = possible_dopants
        self.max_dopants = max_dopants
        self.include_spectra = include_spectra
        self.output_file = output_file
        self.max_data_per_group = max_data_per_group
        self.source = None
        self.target = None
        self.kwargs = kwargs
        self._file = None

        super().__init__(sources=[], targets=[], chunk_size=1000, **kwargs)

    def connect(self):
        # Since we aren't using stores, do nothing
        return

    @property
    def file(self):
        if self._file is None:
            self._file = File(self.output_file, 'w')
        return self._file

    def get_items(self) -> Iterator[dict]:
        templates = get_templates(
            num_samples=self.num_samples,
            excitation_wavelength=self.excitation_wavelength,
            excitation_power=self.excitation_power,
            possible_dopants=self.possible_dopants,
            max_dopants=self.max_dopants)
        for sample_id, template in enumerate(templates):
            yield (sample_id, template)

    def process_item(self, item: tuple[int, dict]) -> dict:
        sample_id, template = item
        dopants = [
            Dopant(el, x)
            for el, x in zip(template['dopants'], template['dopant_concs'])
        ]
        output = run_one_rate_eq(
            dopants,
            excitation_wavelength=template['excitation_wavelength'],
            excitation_power=template['excitation_power'],
            include_spectra=self.include_spectra)

        group_id = int(sample_id // self.max_data_per_group)
        data_id = int(sample_id % self.max_data_per_group)
        return (group_id, data_id, output)

    def update_targets(self, items: list[dict]) -> None:
        for item in items:
            group_id, data_id, output = item
            save_data_to_hdf5(self.file, group_id, data_id, output)
