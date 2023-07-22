from NanoParticleTools.inputs.nanoparticle import Dopant
from NanoParticleTools.analysis import (get_spectrum_energy_from_dndt,
                                        get_spectrum_wavelength_from_dndt,
                                        average_dndt, intensities_from_docs)

from maggma.core import Builder
from maggma.core import Store
from maggma.utils import grouper
from typing import Iterator, List, Dict, Optional, Iterable, Tuple
from bson import uuid
import numpy as np
import random


class UCNPBuilder(Builder):
    """
    Builder that processes and averages NPMC documents
    """

    def __init__(self,
                 source: Store,
                 target: Store,
                 docs_filter: Optional[Dict] = {},
                 chunk_size=1,
                 grouped_ids=None,
                 energy_spectrum_args=None,
                 wavelength_spectrum_args=None,
                 **kwargs):
        if energy_spectrum_args is None:
            energy_spectrum_args = {
                'lower_bound': -40000,
                'upper_bound': 20000,
                'step': 100
            }
        if wavelength_spectrum_args is None:
            wavelength_spectrum_args = {
                'lower_bound': -2000,
                'upper_bound': 1000,
                'step': 5
            }

        self.source = source
        self.target = target
        self.docs_filter = docs_filter
        self.chunk_size = chunk_size
        self.grouped_ids = grouped_ids
        self.energy_spectrum_args = energy_spectrum_args
        self.wavelength_spectrum_args = wavelength_spectrum_args
        self.kwargs = kwargs

        super().__init__(sources=source,
                         targets=target,
                         chunk_size=chunk_size,
                         **kwargs)
        self.connect()

    def get_grouped_docs(self) -> List[Dict]:
        group_keys = [
            "data.n_dopants", "data.n_dopant_sites", "data.formula",
            "data.nanostructure_size", "data.formula_by_constraint",
            "data.excitation_power", "data.excitation_wavelength"
        ]
        return self.source.groupby(keys=group_keys,
                                   criteria=self.docs_filter,
                                   properties=["_id"])

    def get_items(self) -> List[Dict]:
        """
        Here we group the documents
        """

        if self.grouped_ids is None:
            # If it isn't already passed (either by the prechunk or manually),
            # get a list of grouped documents
            grouped_ids = [[doc["_id"] for doc in item[1]]
                           for item in self.get_grouped_docs()]
        else:
            grouped_ids = self.grouped_ids

        for ids in grouped_ids:
            docs_to_avg = list(self.source.query({"_id": {"$in": ids}}))

            # Prune duplicates based on dopant and simulation seed
            unduplicated_dict = {}
            for i, doc in enumerate(docs_to_avg):
                unduplicated_dict[
                    f"{doc['data']['simulation_seed']}-{doc['data']['dopant_seed']}"] = i

            if len(unduplicated_dict) != len(docs_to_avg):
                docs_to_avg = [docs_to_avg[i] for i in unduplicated_dict.values()]

            if len(docs_to_avg) > 0:
                yield docs_to_avg

    def process_item(self, items: List[Dict]) -> Dict:

        # Create/Populate a new document for the average
        avg_doc = {
            "uuid":
            uuid.uuid4(),
            "avg_simulation_length":
            np.mean([i["data"]["simulation_length"] for i in items]),
            "avg_simulation_time":
            np.mean([i["data"]["simulation_time"] for i in items]),
            "n_constraints":
            items[0]["data"]["n_constraints"],
            "n_dopant_sites":
            items[0]["data"]["n_dopant_sites"],
            "n_dopants":
            items[0]["data"]["n_dopants"],
            "formula":
            items[0]["data"]["formula"],
            "nanostructure":
            items[0]["data"]["nanostructure"],
            "nanostructure_size":
            items[0]["data"]["nanostructure_size"],
            "total_n_levels":
            items[0]["data"]["total_n_levels"],
            "formula_by_constraint":
            items[0]["data"]["formula_by_constraint"],
            "dopants":
            items[0]["data"]["dopants"],
            "dopant_concentration":
            items[0]["data"]["dopant_concentration"],
            "overall_dopant_concentration":
            items[0]["data"]["overall_dopant_concentration"],
            "excitation_power":
            items[0]["data"]["excitation_power"],
            "excitation_wavelength":
            items[0]["data"]["excitation_wavelength"],
            "dopant_composition":
            items[0]["data"]["dopant_composition"],
            "input":
            items[0]["data"]["input"],
            "num_averaged":
            len(items)
        }
        if 'metadata' in items[0]["data"]:
            avg_doc["metadata"] = items[0]["data"]["metadata"]

        avg_doc["output"] = {}

        # Average the dndt
        avg_doc["output"]["summary_keys"] = [
            "interaction_id", "number_of_sites", "species_id_1",
            "species_id_2", "left_state_1", "left_state_2", "right_state_1",
            "right_state_2", "interaction_type", "rate_coefficient", "dNdT",
            "std_dev_dNdt", "dNdT per atom", "std_dev_dNdT per atom",
            "occurences", "std_dev_occurences", "occurences per atom",
            "std_dev_occurences per atom"
        ]
        avg_dndt = average_dndt(items)
        avg_doc["output"]["summary"] = avg_dndt

        # Compute the spectrum
        dopants = [
            Dopant(key, val)
            for key, val in avg_doc["overall_dopant_concentration"].items()
        ]
        x, y = get_spectrum_energy_from_dndt(avg_dndt, dopants,
                                             **self.energy_spectrum_args)
        avg_doc["output"]["energy_spectrum_x"] = x
        avg_doc["output"]["energy_spectrum_y"] = y

        x, y = get_spectrum_wavelength_from_dndt(
            avg_dndt, dopants, **self.wavelength_spectrum_args)
        avg_doc["output"]["wavelength_spectrum_x"] = x
        avg_doc["output"]["wavelength_spectrum_y"] = y

        # TODO: Average the populations
        return avg_doc

    def update_targets(self, items: List[Dict]) -> None:
        self.target.update(items, key=["uuid"])

    def prechunk(self, number_splits: int) -> Iterator[Dict]:
        grouped_ids = [[doc["_id"] for doc in item[1]]
                       for item in self.get_grouped_docs()]

        # Shuffle the documents in case the shorter ones
        # (which will process faster) are all at the start
        random.shuffle(grouped_ids)

        N = int(np.ceil(len(grouped_ids) / number_splits))
        for split in grouper(grouped_ids, N):
            yield {"grouped_ids": list(split)}


class UCNPPopBuilder(UCNPBuilder):

    def process_item(self, items: List[Dict]) -> Dict:
        self.logger.info(f"Got {len(items)} to process")

        # Prune duplicates based on dopant and simulation seed
        unduplicated_dict = {}
        for i, doc in enumerate(items):
            unduplicated_dict[
                f"{doc['data']['simulation_seed']}-{doc['data']['dopant_seed']}"] = i

        if len(unduplicated_dict) != len(items):
            items = [items[i] for i in unduplicated_dict.values()]

            self.logger.info(
                f"Pruned duplicates, resulting in {len(items)} to process")

        # Create/Populate a new document for the average
        avg_doc = {
            "uuid":
            uuid.uuid4(),
            "avg_simulation_length":
            np.mean([i["data"]["simulation_length"] for i in items]),
            "avg_simulation_time":
            np.mean([i["data"]["simulation_time"] for i in items]),
            "n_constraints":
            items[0]["data"]["n_constraints"],
            "n_dopant_sites":
            items[0]["data"]["n_dopant_sites"],
            "n_dopants":
            items[0]["data"]["n_dopants"],
            "formula":
            items[0]["data"]["formula"],
            "nanostructure":
            items[0]["data"]["nanostructure"],
            "nanostructure_size":
            items[0]["data"]["nanostructure_size"],
            "total_n_levels":
            items[0]["data"]["total_n_levels"],
            "formula_by_constraint":
            items[0]["data"]["formula_by_constraint"],
            "dopants":
            items[0]["data"]["dopants"],
            "dopant_concentration":
            items[0]["data"]["dopant_concentration"],
            "overall_dopant_concentration":
            items[0]["data"]["overall_dopant_concentration"],
            "excitation_power":
            items[0]["data"]["excitation_power"],
            "excitation_wavelength":
            items[0]["data"]["excitation_wavelength"],
            "dopant_composition":
            items[0]["data"]["dopant_composition"],
            "input":
            items[0]["data"]["input"],
            "num_averaged":
            len(items)
        }
        if 'metadata' in items[0]["data"]:
            avg_doc["metadata"] = items[0]["data"]["metadata"]

        avg_doc["output"] = {}

        avg_dndt = average_dndt(items)

        # Compute the spectrum using the average dndt (counts)
        dopants = [
            Dopant(key, val)
            for key, val in avg_doc["overall_dopant_concentration"].items()
        ]
        x, y = get_spectrum_energy_from_dndt(avg_dndt, dopants,
                                             **self.energy_spectrum_args)
        avg_doc["output"]["energy_spectrum_x"] = x
        avg_doc["output"]["energy_spectrum_y"] = y

        x, y = get_spectrum_wavelength_from_dndt(
            avg_dndt, dopants, **self.wavelength_spectrum_args)
        avg_doc["output"]["wavelength_spectrum_x"] = x
        avg_doc["output"]["wavelength_spectrum_y"] = y

        # Compute the spectrum from the populations

        # First, average the populations
        # The populations are stored at descrete 0.0001ms intervals. It's possible
        # that the simulations are not all the same length, so we need to
        # average the populations over the same time interval for each simulation
        min_length = min([
            len(doc['data']['output']['y_overall_populations'])
            for doc in items
        ])
        avg_total_pop = np.array([
            np.array(
                doc['data']['output']['y_overall_populations'])[:min_length]
            for doc in items
        ]).mean(0)
        avg_total_pop_by_constraint = np.array([
            np.array([
                item[:min_length]
                for item in doc['data']['output']['y_constraint_populations']
            ]) for doc in items
        ]).mean(0)

        avg_doc["output"]['avg_total_pop'] = avg_total_pop
        avg_doc["output"][
            'avg_total_pop_by_constraint'] = avg_total_pop_by_constraint

        # Include average
        avg_doc["output"]['avg_5ms_total_pop'] = avg_total_pop[-500:].mean(0)
        avg_doc["output"]['avg_8ms_total_pop'] = avg_total_pop[-800:].mean(0)
        avg_doc["output"]['avg_5ms_total_pop_by_constraint'] = avg_total_pop_by_constraint[:, -500:].mean(1)
        avg_doc["output"]['avg_8ms_total_pop_by_constraint'] = avg_total_pop_by_constraint[:, -800:].mean(1)

        return avg_doc


class PartialAveragingBuilder(UCNPBuilder):

    def __init__(self,
                 n_docs_filter: int,
                 source: Store,
                 target: Store,
                 docs_filter: Optional[Dict] = {},
                 chunk_size=1,
                 grouped_ids=None,
                 energy_spectrum_args=None,
                 wavelength_spectrum_args=None,
                 n_orderings: int = 4,
                 n_sims: int = 4,
                 **kwargs):
        """
        A partial builder that only averages over a set of documents if
        there are a certain number of documents.

        This is primarily used to build collections to compare performance
        with (e.g.) 16 averaged simulations vs 4 averaged simulations.

        Note:
            The total number of averaged simulations is n_orderings * n_sims

        Args:
            n_docs_filter: The nanoparticle must have a minimum of this many simulations
            n_orderings: The number of dopant placements to use in averaging.
            n_sims: The number of simulation random seeds to use in averaging.
        """
        super().__init__(source=source,
                         target=target,
                         docs_filter=docs_filter,
                         chunk_size=chunk_size,
                         grouped_ids=grouped_ids,
                         energy_spectrum_args=energy_spectrum_args,
                         wavelength_spectrum_args=wavelength_spectrum_args,
                         **kwargs)
        self.n_docs_filter = n_docs_filter
        self.n_orderings = n_orderings
        self.n_sims = n_sims

    def get_items(self):
        for docs_to_avg in super().get_items():
            # prune duplicates
            unduplicated_dict = {}
            for i, doc in enumerate(docs_to_avg):
                try:
                    unduplicated_dict[doc['data']['simulation_seed']][
                        doc['data']['dopant_seed']] = i
                except KeyError:
                    unduplicated_dict[doc['data']['simulation_seed']] = {
                        doc['data']['dopant_seed']: i
                    }

            # count the total items in the dict
            total_items = sum([len(v) for v in unduplicated_dict.values()])
            if total_items < self.n_docs_filter:
                continue

            # collect all the items in a single list
            items = []
            for i in sorted(unduplicated_dict.keys())[:self.n_orderings]:
                for j in sorted(unduplicated_dict[i].keys())[:self.n_sims]:
                    items.append(docs_to_avg[unduplicated_dict[i][j]])

            if len(items) != self.n_orderings * self.n_sims:
                continue

            yield items


class MultiFidelityAveragingBuilder(UCNPBuilder):

    def __init__(self,
                 n_docs_filter: int,
                 source: Store,
                 target: Store,
                 docs_filter: Optional[Dict] = {},
                 chunk_size=1,
                 grouped_ids=None,
                 energy_spectrum_args=None,
                 wavelength_spectrum_args=None,
                 n_orderings: int = 4,
                 n_sims: int = 4,
                 **kwargs):
        """
        A builder that only averages over a set of documents
        using multiple averaging schemes.

        Averaging schemes such as 1 ordering and 1 simulation seed,
        2 orderings and 2 simulation seeds, etc.

        Args:
            n_docs_filter: The nanoparticle must have a minimum of this many simulations
            n_orderings: The number of dopant placements to use in averaging.
            n_sims: The number of simulation random seeds to use in averaging.
        """
        super().__init__(source=source,
                         target=target,
                         docs_filter=docs_filter,
                         chunk_size=chunk_size,
                         grouped_ids=grouped_ids,
                         energy_spectrum_args=energy_spectrum_args,
                         wavelength_spectrum_args=wavelength_spectrum_args,
                         **kwargs)
        self.n_docs_filter = n_docs_filter
        self.n_orderings = n_orderings
        self.n_sims = n_sims

    def get_items(self):
        for docs_to_avg in super().get_items():
            # prune duplicates
            unduplicated_dict = {}
            for i, doc in enumerate(docs_to_avg):
                try:
                    unduplicated_dict[doc['data']['simulation_seed']][
                        doc['data']['dopant_seed']] = i
                except KeyError:
                    unduplicated_dict[doc['data']['simulation_seed']] = {
                        doc['data']['dopant_seed']: i
                    }

            # count the total items in the dict
            total_items = sum([len(v) for v in unduplicated_dict.values()])
            if total_items != self.n_docs_filter:
                continue

            yield unduplicated_dict, docs_to_avg

    def process_item(self, input: Tuple[Dict, List[Dict]]) -> Dict:
        unduplicated_dict, items = input

        all_idx = list(collect_values(unduplicated_dict))
        first_item_idx = list(list(unduplicated_dict.values())[0].values())[0]
        first_item = items[first_item_idx]

        # Create/Populate a new document for the average
        avg_doc = {
            "uuid":
            uuid.uuid4(),
            "avg_simulation_length":
            np.mean([items[i]["data"]["simulation_length"] for i in all_idx]),
            "avg_simulation_time":
            np.mean([items[i]["data"]["simulation_time"] for i in all_idx]),
            "n_constraints":
            first_item["data"]["n_constraints"],
            "n_dopant_sites":
            first_item["data"]["n_dopant_sites"],
            "n_dopants":
            first_item["data"]["n_dopants"],
            "formula":
            first_item["data"]["formula"],
            "nanostructure":
            first_item["data"]["nanostructure"],
            "nanostructure_size":
            first_item["data"]["nanostructure_size"],
            "total_n_levels":
            first_item["data"]["total_n_levels"],
            "formula_by_constraint":
            first_item["data"]["formula_by_constraint"],
            "dopants":
            first_item["data"]["dopants"],
            "dopant_concentration":
            first_item["data"]["dopant_concentration"],
            "overall_dopant_concentration":
            first_item["data"]["overall_dopant_concentration"],
            "excitation_power":
            first_item["data"]["excitation_power"],
            "excitation_wavelength":
            first_item["data"]["excitation_wavelength"],
            "dopant_composition":
            first_item["data"]["dopant_composition"],
            "input":
            first_item["data"]["input"],
            "num_averaged":
            len(items)
        }
        if 'metadata' in first_item["data"]:
            avg_doc["metadata"] = first_item["data"]["metadata"]

        avg_doc["output"] = {}

        # Compute the spectrum
        dopants = [
            Dopant(key, val)
            for key, val in avg_doc["overall_dopant_concentration"].items()
        ]
        energy_spectra_y = {}
        wavelength_spectra_y = {}
        for n_orderings, n_sims in [(1, 1), (2, 2), (4, 1), (1, 4), (4, 4)]:
            _items = []
            if len(unduplicated_dict) < n_orderings:
                # not enough dopant orderings
                # print(f'not enough orderings for {n_orderings}, {n_sims}')
                # energy_spectra_y[(n_orderings, n_sims)] = None
                # wavelength_spectra_y[(n_orderings, n_sims)] = None
                continue
            if len(unduplicated_dict[sorted(
                    unduplicated_dict.keys())[0]]) < n_sims:
                # not enough simulation seeds
                # print(f'not enough sims seeds for {n_orderings}, {n_sims}')
                # energy_spectra_y[(n_orderings, n_sims)] = None
                # wavelength_spectra_y[(n_orderings, n_sims)] = None
                continue

            for i in sorted(unduplicated_dict.keys())[:n_orderings]:
                for j in sorted(unduplicated_dict[i].keys())[:n_sims]:
                    _items.append(items[unduplicated_dict[i][j]])
            avg_dndt = average_dndt(_items)
            energy_spectra_x, y = get_spectrum_energy_from_dndt(
                avg_dndt, dopants, **self.energy_spectrum_args)
            energy_spectra_y[(n_orderings, n_sims)] = y
            wavelength_spectra_x, y = get_spectrum_wavelength_from_dndt(
                avg_dndt, dopants, **self.wavelength_spectrum_args)
            wavelength_spectra_y[(n_orderings, n_sims)] = y

        avg_doc["output"]["energy_spectrum_x"] = energy_spectra_x
        avg_doc["output"]["energy_spectrum_y"] = energy_spectra_y
        avg_doc["output"]["wavelength_spectrum_x"] = wavelength_spectra_x
        avg_doc["output"]["wavelength_spectrum_y"] = wavelength_spectra_y

        return avg_doc


def collect_values(d):
    """
    Utility function to get all values from a nested dict
    """
    for k, v in d.items():
        if isinstance(v, dict):
            yield from collect_values(v)
        else:
            yield v
