from NanoParticleTools.machine_learning.models.mlp_model.model import MLPSpectrumModel
import pytorch_lightning as pl
from maggma.stores import MongoStore


def FreezeMLP(model: MLPSpectrumModel,
               num_frozen_layers: int
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
                param.reset
    return model


def HF_dataset(store_config: dict,
               collection_name: str,
              sample_size : int):
    """
    params
    store_config: dictionary of configuration for MongoDB
    collection_name: name of collection in MongoDB
    sample_size: sample size of high fidelity data you want to use in your fine tuning
    
    """
    training_data_store_HF = MongoStore(**store_config, collection_name= collection_name) #access collection
    training_data_store_HF.connect() #connect to enable aggregate
    cursor = training_data_store_HF.cursor.allowDiskUse(True)
    cursor = cursor._collection.aggregate([ {"$sample": { "size": sample_size } } ])
    training_data_store_HF.count()
    hf_data = list(cursor)

    return hf_data
