from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import pytorch_lightning as pl
import shutil
import os
import math
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from maggma.stores.mongolike import MongoStore
from typing import Optional, Union

from NanoParticleTools.machine_learning.models import SpectrumModel
from NanoParticleTools.machine_learning.data import LabelProcessor, VolumeFeatureProcessor, NPMCDataModule
from NanoParticleTools.machine_learning.util.reporters import TrialTerminationReporter
from NanoParticleTools.machine_learning.util.learning_rate import get_sequential
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import datetime
from matplotlib import pyplot as plt
import numpy as np
from fireworks.fw_config import LAUNCHPAD_LOC

def train_spectrum_model(config: dict, 
                              num_epochs: Optional[int] = 10, 
                              num_gpus: Union[int, float] = 0,
                              wandb_project: Optional[str] = 'default_project',
                              wandb_save_dir: Optional[str] = os.environ['HOME'],
                              tune = False):
    """
    
    :param config: a config dictionary for the model
        ex - {'n_input_nodes': 20, 'n_output_nodes'; 200, 'n_hidden_layers': 2, 'n_hidden_1': 128, 'n_hidden_2': 128, 'learning_rate': 1e-3}
    :param num_epochs: max number of epochs to train the model. Early stopping may result in less training epochs
    :param num_gpus: number of gpus to use for the training. May be a fractional number if training multiple models concurrently on one gpu
    :param wandb_project: wandb_project name
    :param wandb_save_dir: Directory to save wandb files to
    :param tune: whether or not this is a tune experiment. (To make sure that TuneReportCallback is added)
    """
    # Setup the data
    feature_processor = VolumeFeatureProcessor(fields = ['formula_by_constraint', 'dopant_concentration', 'input.constraints'])
    label_processor = LabelProcessor(fields = ['output.spectrum_x', 'output.spectrum_y'], 
                                     spectrum_range = (-1000, 1000), 
                                     output_size = config['n_output_nodes'], 
                                     log = True,
                                     normalize = False)
    
    if os.path.exists('npmc_data.json'):
        data_module = NPMCDataModule(feature_processor=feature_processor, label_processor=label_processor, data_dir='npmc_data.json', batch_size=16)
    else:
        data_store = MongoStore.from_launchpad_file(LAUNCHPAD_LOC, 'avg_npmc_20220708')
        data_module = NPMCDataModule(feature_processor=feature_processor, label_processor=label_processor, data_store=data_store, batch_size=16)
    # Make the model
    model = SpectrumModel(lr_scheduler = get_sequential,
                          optimizer_type = 'adam',
                          **config)
    
    # Make logger
    wandb_logger = WandbLogger(entity="esivonxay", project=wandb_project, save_dir=wandb_save_dir)

    # Configure callbacks
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(EarlyStopping(monitor='val_loss', patience=200))
    if tune:
        callbacks.append(TuneReportCallback({"loss": "val_loss"}, on="validation_end"))
    

    # Run the training
    trainer = pl.Trainer(gpus = math.ceil(num_gpus), 
                         max_epochs=num_epochs, 
                         enable_progress_bar=False, 
                         logger=wandb_logger, 
                         callbacks=callbacks)
    try:
        trainer.fit(model=model, datamodule=data_module)
    except Exception as e:
        # Don't do anything with the exception to allow the wandb logger to finish
        print(e)
    
    rng = np.random.default_rng(seed = 10)
    for i in rng.choice(range(len(data_module.npmc_test)), 10, replace=False):
        X, y = data_module.npmc_test[i]
        
        fig = plt.figure(dpi=150)
        plt.plot(data_module.label_processor.x, np.power(10, y.numpy())-1, label='NPMC', alpha=1)
        plt.plot(data_module.label_processor.x, np.power(10, model(X).detach().numpy())-1, label='NN', alpha=0.5)
        plt.xlabel('Wavelength (nm)', fontsize=18)
        plt.ylabel('Relative Intensity (a.u.)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0, 1e4)
        plt.tight_layout()
        
        trainer.logger.experiment.log({'spectrum': fig})
    # Finalize the wandb logging
    wandb.finish()


def tune_npmc_asha(num_samples: Optional[int] = 10, 
                   num_epochs: Optional[int] = 1000, 
                   wandb_project: Optional[str] = None,
                   save_dir: Optional[str] = None,
                   resources_per_trial: Optional[dict] = {'cpu': 4}):
    """
    :param config: a config dictionary for the model
        ex - {'n_input_nodes': 20, 'n_output_nodes'; 200, 'n_hidden_layers': 2, 'n_hidden_1': 128, 'n_hidden_2': 128, 'learning_rate': 1e-3}
    :param num_epochs: max number of epochs to train the model. Early stopping may result in less training epochs
    :param num_gpus: number of gpus to use for the training. May be a fractional number if training multiple models concurrently on one gpu
    :param wandb_project: wandb_project name
    :param wandb_save_dir: Directory to save wandb files to
    """
    config = {'n_input_nodes': 20,
              'n_output_nodes': 400, 
              'n_hidden_layers': tune.choice([1, 2, 3, 4]),
              'n_hidden_1': tune.choice([16, 32, 64, 128]),
              'n_hidden_2': tune.choice([16, 32, 64, 128]),
              'n_hidden_3': tune.choice([16, 32, 64, 128]),
              'n_hidden_4': tune.choice([16, 32, 64, 128]),
              'learning_rate': tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
              'l2_regularization_weight': tune.choice([0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
              'dropout_probability': tune.choice([0.05, 0.1, 0.25, 0.5])}
    
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=100,
        reduction_factor=2)

    reporter = TrialTerminationReporter(
        parameter_columns=["n_hidden_layers", "n_hidden_1", "n_hidden_2", "n_hidden_3", "n_hidden_4", "learning_rate", "l2_regularization_weight", "dropout_probability"],
        metric_columns=["loss", "training_iteration"])


    if wandb_project is None:
        date = ''.join([f'{val}{sym}'for val, sym in zip(datetime.datetime.now().isoformat().split('.')[0].split(':'), ['h', 'm', 's'])])
        wandb_project = f'Raytune-{date}'
    if save_dir is None:
        save_dir = os.path.join(os.environ['HOME'], 'train_output')

    train_fn_with_parameters = tune.with_parameters(train_spectrum_model,
                                                    num_epochs = num_epochs,
                                                    num_gpus = resources_per_trial.get('gpu', 0),
                                                    wandb_project = wandb_project,
                                                    wandb_save_dir = save_dir,
                                                    tune = True)
    
    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=num_samples,
                        scheduler=scheduler,
                        progress_reporter=reporter,
                        reuse_actors=False,
                        local_dir=save_dir,
                        name="tune_npmc_asha")

    print("Best hyperparameters found were: ", analysis.best_config)