from NanoParticleTools.machine_learning.models.mlp_model.model import MLPSpectrumModel
import pytorch_lightning as pl
from maggma.stores import MongoStore
from NanoParticleTools.machine_learning.data.datamodule import NPMCDataModule
from NanoParticleTools.machine_learning.data.dataset import NPMCDataset
import torch
import numpy as np

def FreezeMLP(model: MLPSpectrumModel,
               num_frozen_layers: int,
               reset_thawed_layers: bool
               ):
    """
    params
    model: MLPSpectrumModel trained on LF data
    num_frozen_layers: the number of layers to freeze
    """
    iter_num = 0  
    for name, param in model.named_parameters():
        iter_num+=1
        if iter_num < num_frozen_layers*2:
            param.requires_grad = False
        else:
            if reset_thawed_layers == True:
                param = 0
    return model
    
def k_fold_validation_training(dataset: NPMCDataset, 
                                    k: int):
    """
    params
    training_set: NPMCDataset containing training data
    k: number of k-folds
    """

    # Get the number of samples in the dataset
    num_samples = len(dataset)

    # Create a list of shuffled indices
    shuffled_indices = list(range(num_samples))
    np.random.shuffle(shuffled_indices)

    # Calculate the size of each fold
    fold_size = num_samples // k

    training_sets = []
    validation_sets = []

    # Create k folds and split them into training and validation sets
    for i in range(k):
        # Calculate the start and end indices for the current fold
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k - 1 else num_samples

        # Split the indices into training and validation sets
        validation_indices = shuffled_indices[start_idx:end_idx]
        training_indices = shuffled_indices[:start_idx] + shuffled_indices[end_idx:]

        # Create subsets using the indices
        training_subset = torch.utils.data.Subset(dataset, training_indices)
        validation_subset = torch.utils.data.Subset(dataset, validation_indices)

        # Append subsets to the lists
        training_sets.append(training_subset)
        validation_sets.append(validation_subset)

    return training_sets, validation_sets