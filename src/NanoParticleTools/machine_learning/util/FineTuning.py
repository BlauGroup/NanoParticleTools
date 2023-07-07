from NanoParticleTools.machine_learning.models.mlp_model.model import MLPSpectrumModel
import pytorch_lightning as pl
from fireworks.fw_config import LAUNCHPAD_LOC
from maggma.stores import MongoStore


def FreezeMLP(model: MLPSpectrumModel,
               num_frozen_layers: int,
               reset_thawed_layers: bool):
    """
    params
    model: MLPSpectrumModel trained on LF data
    num_frozen_layers: the number of layers to freeze
    reset_thawed_layers: whether or not to reset the parameters in the layers that aren't frozen
    """
    iter_num = 0  
    for name, param in model.named_parameters():
        iter_num+=1
        if iter_num < num_frozen_layers*2:
            param.requires_grad = False
        if iter_num >= num_frozen_layers*2: 
            if reset_thawed_layers == True:
                param.reset
    return model


def HF_dataset(collection_name: str,
              sample_size : int):
    """
    params
    collection_name: name of collection in MongoDB
    sample_size: sample size of high fidelity data you want to use in your fine tuning
    
    """
    store_config = {
    'database': 'npmc_db',
    'host': 'mongodb05.nersc.gov',
    'port': 27017,
    'username': 'lattia',
    'password': 'cram-imbue-beduin-eternal',
    'ssh_tunnel': None,
    'safe_update': False,
    'auth_source': 'npmc_db',
    'mongoclient_kwargs': {},
    'default_sort': None}

    training_data_store_HF = MongoStore(**store_config, collection_name= collection_name) #access collection
    training_data_store_HF.connect() #connect to enable aggregate
    cursor = training_data_store_HF._collection.aggregate([ {"$sample": { "size": sample_size } } ])
    training_data_store_HF.count()
    hf_data = list(cursor)

    return hf_data
