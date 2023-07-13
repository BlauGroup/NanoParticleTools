from NanoParticleTools.machine_learning.models.mlp_model.model import MLPSpectrumModel
import pytorch_lightning as pl
from maggma.stores import MongoStore


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

