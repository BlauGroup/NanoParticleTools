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

    def __init__(self, args: ArgumentParser, **kwargs):

        self.source = None
        self.args = args
        self.target = None
        self.kwargs = kwargs
        self._file = None

        super().__init__(sources=None, targets=None, chunk_size=1, **kwargs)

    def connect(self):
        # Since we aren't using stores, do nothing
        return

    @property
    def file(self):
        if self._file is None:
            self._file = File(self.args.output_file, 'w')
        return self._file

    def get_items(self) -> Iterator[dict]:
        for sample_id, template in enumerate(get_templates(self.args)):
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
            include_spectra=self.args.include_spectra)

        group_id = int(sample_id // self.args.max_data_per_group)
        data_id = int(sample_id % self.args.max_data_per_group)
        return (group_id, data_id, output)

    def update_targets(self, items: list[dict]) -> None:
        for item in items:
            group_id, data_id, output = item
            save_data_to_hdf5(self.file, group_id, data_id, output)
