from platform import node
import shutil
from ....inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint
from ....species_data.species import Dopant
from .._data import DataProcessor, LabelProcessor, BaseNPMCDataset
from .._data import NPMCDataModule as _NPMCDataModule

from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from matplotlib import pyplot as plt
import numpy as np

from typing import List, Union, Tuple, Optional, Type
from maggma.core.store import Store
import torch
import itertools
import os
import tempfile

# class GraphFeatureProcessor(DataProcessor):
#     def __init__(self,
#                  possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
#                  cutoff_distance: Optional[int] = 3,
#                  resolution: Optional[float] = 0.1,
#                  **kwargs):
#         """
#         :param possible_elements: 
#         :param edge_attr_bias: A bias added to the edge_attr before applying 1/edge_attr. This serves to eliminate
#             divide by zero and inf in the tensor. Additionally, it acts as a weight on the self-interaction.
#         """
#         super().__init__(fields = ['formula_by_constraint', 'dopant_concentration', 'input'],
#                          **kwargs)
        
#         self.possible_elements = possible_elements
#         self.n_possible_elements = len(possible_elements)
#         self.dopants_dict = {key: i for i, key in enumerate(self.possible_elements)}
#         self.cutoff_distance = cutoff_distance
#         self.resolution = resolution

#     def get_node_features(self, 
#                           constraints: List[NanoParticleConstraint], 
#                           dopant_specifications: List[Tuple[int, float, str, str]]) -> torch.Tensor:
#         # Generate the tensor of concentrations for the original constraints.
#         ## Initialize it to 0
#         concentrations = torch.zeros(len(constraints), self.n_possible_elements)
        
#         ## Fill in the concentrations that are present
#         for i, x, el, _ in dopant_specifications:
#             concentrations[i][self.dopants_dict[el]] = x

#         # Make the array for the representation
#         n_subdivisions = torch.ceil(torch.tensor(constraints[-1].radius) / self.resolution).int()
#         node_features = torch.zeros((n_subdivisions, self.n_possible_elements, 3))

#         ## Set the first index to identity
#         node_features[:, :, 0] = torch.ones((n_subdivisions, self.n_possible_elements)) * torch.arange(0, 3)

#         ## Set the second index to concentration
#         start_i = 0
#         for constraint_i in range(len(constraints)):
#             end_i = torch.ceil(torch.tensor(constraints[constraint_i].radius) / self.resolution).int()
#             node_features[start_i:end_i, :, 1] = concentrations[constraint_i]
#             start_i = end_i

#         ## Set the third index to volume
#         node_features[:, :, 2] = self.volume(torch.arange(0, constraints[-1].radius, self.resolution)).unsqueeze(-1).expand(-1, 3)

#         return {'x': node_features}
    
#     def get_edge_features(self, 
#                           constraints: List[NanoParticleConstraint], 
#                           dopant_specifications: List[Tuple[int, float, str, str]]) -> torch.Tensor:
#         # Determine connectivity using a cutoff
#         radius = torch.arange(0, constraints[-1].radius, self.resolution)
#         xy, yx = torch.meshgrid(radius, radius, indexing='xy')
#         distance_matrix = torch.abs(xy - yx)
#         edge_index = torch.vstack(torch.where(distance_matrix <= self.cutoff_distance))
#         edge_attr = distance_matrix[edge_index[0], edge_index[1]]
        
#         return {'edge_index': edge_index, 
#                 'edge_attr': edge_attr}
        
#     def get_data_graph(self, 
#                        constraints: List[NanoParticleConstraint], 
#                        dopant_specifications: List[Tuple[int, float, str, str]]):
        
#         output_dict = self.get_node_features(constraints, dopant_specifications)
#         output_dict.update(self.get_edge_features(constraints, dopant_specifications))
        
#         return output_dict
    
#     def process_doc(self,
#                     doc: dict) -> dict:
#         constraints = doc['input']['constraints']
#         dopant_specifications = doc['input']['dopant_specifications']

#         try:
#             constraints = [SphericalConstraint.from_dict(c) for c in constraints]
#         except:
#             pass
        
#         return self.get_data_graph(constraints, dopant_specifications)

#     def volume(self,
#                radius: Union[List, int, torch.Tensor], 
#                shell_width: Optional[float] = 0.01) -> torch.Tensor:
#             """
#             Takes inner radius
#             """
#             if not isinstance(radius, torch.Tensor):
#                 radius = torch.tensor(radius)
                
#             outer_radius = radius + shell_width
            
            
#             return self.sphere_volume(outer_radius) - self.sphere_volume(radius)

#     @staticmethod
#     def sphere_volume(radius: torch.Tensor) -> torch.Tensor:
#         return 3/4*torch.pi*torch.pow(radius, 3)

class FeatureProcessor(DataProcessor):
    def __init__(self,
                 possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
                 resolution: Optional[float] = 0.1,
                 max_np_size: Optional[int] = 500,
                 dims: Optional[int] = 1,
                 full_nanoparticle: Optional[bool] = True,
                 **kwargs):
        """
        :param possible_elements:  
        :param cutoff_distance: 
        :param resolution: Angstroms
        :param max_np_size: Angstroms
        """
        super().__init__(fields = ['formula_by_constraint', 'dopant_concentration', 'input'],
                         **kwargs)

        self.possible_elements = possible_elements
        self.n_possible_elements = len(possible_elements)
        self.dopants_dict = {key: i for i, key in enumerate(self.possible_elements)}
        self.resolution = resolution
        self.max_np_size = max_np_size
        self.max_divisions = -int(max_np_size // -resolution)
        assert dims > 0 and dims <= 3
        self.dims = dims
        self.full_nanoparticle = full_nanoparticle

    def get_node_features(self, 
                          constraints: List[NanoParticleConstraint], 
                          dopant_specifications: List[Tuple[int, float, str, str]]) -> torch.Tensor:
        # Generate the tensor of concentrations for the original constraints.
        ## Initialize it to 0
        concentrations = torch.zeros(len(constraints), self.n_possible_elements)
        
        ## Fill in the concentrations that are present
        for i, x, el, _ in dopant_specifications:
            concentrations[i][self.dopants_dict[el]] = x

        # Determine the number of pixels/subdivisions this specific particle needs. 
        # Using this instead of the max size will save us some time when assigning pixels
        n_subdivisions = torch.ceil(torch.tensor(constraints[-1].radius) / self.resolution).int()

        # Make the array for the representation
        node_features = torch.zeros([n_subdivisions for _ in range(self.dims)] + [self.n_possible_elements])

        # radius = torch.arange(0, n_subdivisions, self.resolution)
        radius = torch.arange(0, n_subdivisions)
        mg = torch.meshgrid(*[radius for _ in range(self.dims)], indexing='ij')
        radial_distance = torch.sqrt(torch.sum(torch.stack([torch.pow(_el, 2) for _el in mg]), dim=0))

        lower_bound = 0
        for constraint_i in range(len(constraints)):
            upper_bound = constraints[constraint_i].radius / self.resolution
            
            idx = torch.where(torch.logical_and(radial_distance >= lower_bound, radial_distance < upper_bound))
            node_features.__setitem__(idx, concentrations[constraint_i])
            
            lower_bound = upper_bound
        
        if self.full_nanoparticle:
            full_repr = torch.zeros([2*n_subdivisions for _ in range(self.dims)] + [self.n_possible_elements])
            for ops in itertools.product(*[[0, 1] for _ in range(self.dims)]):
                # 0 = no change, 1 = flip along this axis
                idx = [slice(n_subdivisions) if j == 1 else slice(n_subdivisions, 2*n_subdivisions) for j in ops]

                flip_dims = [i for i, j in enumerate(ops) if j ==1]
                full_repr.__setitem__(idx, torch.flip(node_features, flip_dims))

            node_features = full_repr

        # Put the channel in the first index
        node_features = node_features.moveaxis(-1, 0)
        
        # Pad image so they are all the same size
        pad_size = (2*self.max_divisions - node_features.shape[1]) // 2
        padding_tuple = tuple([pad_size for i in range(self.dims) for i in range(2)])
        node_features = torch.nn.functional.pad(node_features, padding_tuple)
        
        if self.dims == 1:
            node_features = torch.nn.functional.avg_pool1d(node_features, 2, 2)
        elif self.dims == 2:
            node_features = torch.nn.functional.avg_pool2d(node_features, 2, 2)
        elif self.dims == 3:
            node_features = torch.nn.functional.avg_pool3d(node_features, 2, 2)

        return {'x': node_features}
    
    def get_data_graph(self, 
                       constraints: List[NanoParticleConstraint], 
                       dopant_specifications: List[Tuple[int, float, str, str]]):
        
        output_dict = self.get_node_features(constraints, dopant_specifications)
        
        return output_dict
    
    def process_doc(self,
                    doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_specifications = doc['input']['dopant_specifications']
        
        try:
            constraints = [SphericalConstraint.from_dict(c) for c in constraints]
        except:
            pass
        
        return self.get_data_graph(constraints, dopant_specifications)

    def volume(self,
               radius: Union[List, int, torch.Tensor], 
               shell_width: Optional[float] = 0.01) -> torch.Tensor:
            """
            Takes inner radius
            """
            if not isinstance(radius, torch.Tensor):
                radius = torch.tensor(radius)
                
            outer_radius = radius + shell_width
            
            
            return self.sphere_volume(outer_radius) - self.sphere_volume(radius)

    @staticmethod
    def sphere_volume(radius: torch.Tensor) -> torch.Tensor:
        return 3/4*torch.pi*torch.pow(radius, 3)

    def __str__(self) -> str:
        return f"CNN Feature Processor - resolution = {self.resolution}A - max_np_size = {self.max_np_size}"

class NPMCDataset(BaseNPMCDataset):
    def __init__(self, 
                 docs: List, 
                 feature_processor: DataProcessor, 
                 label_processor: DataProcessor,
                 data_dir: Optional[str] = '.data',
                 override_cached_data:Optional[bool]=True):
        try:
            # deserialize the constraints if necessary
            for i, doc in enumerate(docs):
                docs[i]['input']['constraints'] = [SphericalConstraint.from_dict(c) for c in doc['input']['constraints']]
        except:
            pass

        super().__init__(docs, feature_processor, label_processor)

        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            os.mkdir(os.path.join(data_dir, 'x'))
            os.mkdir(os.path.join(data_dir, 'y'))
            self.cached = [False for _ in docs]
            self.prime_dataset()
        elif override_cached_data:
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)
            os.mkdir(os.path.join(data_dir, 'x'))
            os.mkdir(os.path.join(data_dir, 'y'))
            self.cached = [False for _ in docs]
            self.prime_dataset()
        else:
            self.cached = [True for _ in docs]
    

    def prime_dataset(self):
        for i in range(len(self)):
            _ = self[i]
        return

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        # Check if this file is in the cache
        if idx < len(self): 
            return self.process_single_doc(idx)
        else: 
            raise IndexError('list index out of bounds')

    def process_single_doc(self, idx) -> Data:
        
        if self.cached[idx]:
            _d = {'x': torch.load(os.path.join(self.data_dir, 'x', str(idx))),
                  'y': torch.load(os.path.join(self.data_dir, 'y', str(idx))),
                  'constraints': self.docs[idx]['input']['constraints'],
                  'dopant_specifications': self.docs[idx]['input']['dopant_specifications']}
        else:
            # Process this item for the first time
            _d = self.feature_processor.process_doc(self.docs[idx])
            _d['y'] = self.label_processor.process_doc(self.docs[idx])
            _d['constraints'] = self.docs[idx]['input']['constraints']
            _d['dopant_specifications'] = self.docs[idx]['input']['dopant_specifications']

            # Cache this item to file
            torch.save(_d['x'], os.path.join(self.data_dir, 'x', str(idx)))
            torch.save(_d['y'], os.path.join(self.data_dir, 'y', str(idx)))
            self.cached[idx] = True

        return Data(**_d)

class NPMCDataModule(_NPMCDataModule):
    def __init__(self, 
                # feature_processor: DataProcessor, 
                # label_processor: DataProcessor, 
                # dataset_class: Type[torch.utils.data.Dataset] = ..., 
                # training_data_store: Optional[Store] = None, 
                # testing_data_store: Optional[Store] = None, 
                # batch_size: Optional[int] = 16, 
                # validation_split: Optional[float] = 0.15, 
                # test_split: Optional[float] = 0.15, 
                # random_split_seed=0,
                training_data_dir: Optional[str] = '.data',
                testing_data_dir: Optional[str] = '.data',
                **kwargs):

        super().__init__(**kwargs)
        self.training_data_dir = training_data_dir
        self.testing_data_dir = testing_data_dir

    def get_training_dataset(self):
        return self.dataset_class.from_store(store=self.training_data_store,
                                            doc_filter=self.training_doc_filter,
                                            feature_processor = self.feature_processor,
                                            label_processor = self.label_processor,
                                            data_dir = self.training_data_dir,
                                            n_docs=self.training_size)

    def get_testing_dataset(self):
        return self.dataset_class.from_store(store=self.testing_data_store,
                                            doc_filter=self.testing_doc_filter,
                                            feature_processor = self.feature_processor,
                                            label_processor = self.label_processor,
                                            data_dir = self.testing_data_dir,
                                            n_docs=self.testing_size)

    @staticmethod
    def collate(data_list: List[Data]):
        if len(data_list) == 0:
            return data_list[0]
        
        x = torch.stack([data.x for data in data_list])
        y = torch.stack([data.y for data in data_list])

        return Data(x=x, y=y)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_train, self.batch_size, collate_fn=self.collate, shuffle=True, num_workers=0)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_val, self.batch_size, collate_fn=self.collate, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.npmc_test, self.batch_size, collate_fn=self.collate, shuffle=False, num_workers=0)