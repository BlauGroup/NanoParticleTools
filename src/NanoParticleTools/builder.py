from maggma.core import Builder
from maggma.core import Store
from typing import Iterator, List, Dict, Optional, Iterable
from bson import uuid
import numpy as np
from NanoParticleTools.species_data.species import Dopant

class UCNPBuilder(Builder):
    """
    Builder that processes and averages NPMC documents
    """
    
    def __init__(self, 
                 source: Store,
                 target: Store,
                 docs_filter: Optional[Dict]={},
                 chunk_size = 1,
                 grouped_ids = None,
                 **kwargs):
        self.source = source
        self.target = target
        self.docs_filter = docs_filter
        self.chunk_size = chunk_size
        self.grouped_ids = grouped_ids
        self.kwargs = kwargs
        
        super().__init__(sources=source, targets=target, chunk_size=chunk_size, **kwargs)
    
    
    def get_grouped_docs(self):
        group_keys = ["data.n_dopants", "data.n_dopant_sites", "data.formula", "data.nanostructure_size", "data.formula_by_constraint", "data.excitation_power", "data.excitation_wavelength"]
        return self.source.groupby(keys=group_keys, criteria=self.docs_filter, properties=["_id"])
    
    def get_items(self):
        """
        Here we group the documents
        """
        grouped_ids = self.grouped_ids
        if self.grouped_ids is None:
            # If it isn"t already passed (either by the prechunk or manually), get a list of grouped documents
            grouped_ids = [[doc["_id"] for doc in item[1]] for item in self.get_grouped_docs()]
            
        for ids in grouped_ids:
            docs_to_avg = list(self.source.query({"_id": {"$in": ids}}))
            if len(docs_to_avg) > 0:
                yield docs_to_avg
    
    def process_item(self, items):
        self.logger.info(f"Got {len(items)} to process")
        # Create/Populate a new document for the average
        avg_doc = {"uuid": uuid.uuid4(),
                   "avg_simulation_length": np.mean([i["data"]["simulation_length"] for i in items]),
                   "avg_simulation_time": np.mean([i["data"]["simulation_time"] for i in items]),
                   "n_constraints": items[0]["data"]["n_constraints"],
                   "n_dopant_sites": items[0]["data"]["n_dopant_sites"],
                   "n_dopants": items[0]["data"]["n_dopants"],
                   "formula": items[0]["data"]["formula"],
                   "nanostructure": items[0]["data"]["nanostructure"],
                   "nanostructure_size": items[0]["data"]["nanostructure_size"],
                   "total_n_levels": items[0]["data"]["total_n_levels"],
                   "formula_by_constraint": items[0]["data"]["formula_by_constraint"],
                   "dopants": items[0]["data"]["dopants"],
                   "dopant_concentration": items[0]["data"]["dopant_concentration"],                   
                   "overall_dopant_concentration": items[0]["data"]["overall_dopant_concentration"],                  
                   "excitation_power": items[0]["data"]["excitation_power"],                  
                   "excitation_wavelength": items[0]["data"]["excitation_wavelength"],                  
                   "dopant_composition": items[0]["data"]["dopant_composition"],                  
                   "input": items[0]["data"]["input"],
                   }
        
        avg_doc["output"] = {}
        
        # Average the dndt
        avg_doc["output"]["summary_keys"] = ["interaction_id", "number_of_sites", "species_id_1", "species_id_2", "left_state_1", "left_state_2",
                "right_state_1", "right_state_2", "interaction_type", "rate_coefficient", "dNdT", "std_dev_dNdt", 
                "dNdT per atom", "std_dev_dNdT per atom", "occurences", "std_dev_occurences", 
                "occurences per atom", "std_dev_occurences per atom"]
        avg_dndt = self.average_dndt(items)
        avg_doc["output"]["summary"] = avg_dndt
        
        # Compute the spectrum
        dopants = [Dopant(key, val) for key, val in avg_doc["overall_dopant_concentration"].items()]
        x, y = self.get_spectrum(avg_dndt, dopants)
        avg_doc["output"]["spectrum_x"] = x
        avg_doc["output"]["spectrum_y"] = y
        
        # TODO: Average the populations
        return avg_doc
    
    def update_targets(self, items: List[Dict]):
        self.target.update(items, key=["uuid"])
    
    def prechunk(self):
        pass
    
    def get_spectrum(self, 
                     avg_dndt, 
                     dopants, 
                     lower_bound = -1000, 
                     upper_bound = 1000, 
                     step = 1):
        
        x = np.arange(-1000, 1000, step)
        y = np.zeros(x.shape)
        for interaction in [_d for _d in avg_dndt if _d[8] == "Rad"]:
            # print(interaction)
            species_id = interaction[2]
            left_state_1 = interaction[4]
            right_state_1 = interaction[6]
            ei = dopants[species_id].energy_levels[left_state_1]
            ef = dopants[species_id].energy_levels[right_state_1]

            de = ef.energy-ei.energy
            wavelength = (299792458*6.62607004e-34)/(de*1.60218e-19/8065.44)*1e9
            # print(left_state_1, right_state_1, wavelength)
            if wavelength > lower_bound and wavelength < upper_bound:
                index = int(np.floor(wavelength+1000)/step)
                y[index]+=interaction[10]
        return x, y
    
    def average_dndt(self, docs):
        """
        Compute the average of the dndts
        
        Output will have the following fields:
        ["interaction_id", "number_of_sites", "species_id_1", "species_id_2", "left_state_1", "left_state_2",
                "right_state_1", "right_state_2", "interaction_type", "rate_coefficient", "dNdT", "std_dev_dNdt", 
                "dNdT per atom", "std_dev_dNdT per atom", "occurences", "std_dev_occurences", 
                "occurences per atom", "std_dev_occurences per atom"]
        """
        accumulated_dndt = {}
        n_docs=0
        for doc in docs:
            n_docs += 1
            keys = doc["data"]["output"]["summary_keys"]
            try:
                search_keys = ["interaction_id", "number_of_sites", "species_id_1", "species_id_2", "left_state_1", "left_state_2",
                "right_state_1", "right_state_2", "interaction_type", 'rate_coefficient', 'dNdT', 'dNdT per atom', 'occurences', 'occurences per atom']
                indices = []
                for key in search_keys:
                    indices.append(keys.index(key))
            except:
                search_keys = ["interaction_id", "number_of_sites", "species_id_1", "species_id_2", "left_state_1", "left_state_2",
                "right_state_1", "right_state_2", "interaction_type", 'rate', 'dNdT', 'dNdT per atom', 'occurences', 'occurences per atom']
                indices = []
                for key in search_keys:
                    indices.append(keys.index(key))
                
            dndt = doc["data"]["output"]["summary"]

            for interaction in dndt:
                interaction_id = interaction[0]
                if interaction_id not in accumulated_dndt:
                    accumulated_dndt[interaction_id] = []
                accumulated_dndt[interaction_id].append([interaction[i] for i in indices])

        avg_dndt = []
        for interaction_id in accumulated_dndt:

            arr = accumulated_dndt[interaction_id][-1][:-4]

            _dndt = [_arr[-4:] for _arr in accumulated_dndt[interaction_id]]

            while len(_dndt) < n_docs:
                _dndt.append([0 for _ in range(4)])

            mean = np.mean(_dndt, axis=0)
            std = np.std(_dndt, axis=0)
            arr.extend([mean[0], std[0], mean[1], std[1], mean[2], std[2], mean[3], std[3]])
            avg_dndt.append(arr)
        return avg_dndt