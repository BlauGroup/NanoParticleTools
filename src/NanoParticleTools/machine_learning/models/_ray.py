from ._data import *
from ..util.learning_rate import get_sequential
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from NanoParticleTools.machine_learning.util.callbacks import LossAugmentCallback
from pytorch_lightning.loggers import WandbLogger

import wandb
import pytorch_lightning as pl
import os
import math
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from maggma.stores.mongolike import MongoStore
from typing import Optional, Union

from ._model import SpectrumModelBase
from ._data import EnergyLabelProcessor, DataProcessor, NPMCDataModule
from ..util.reporters import TrialTerminationReporter
from ...inputs.nanoparticle import SphericalConstraint
from ...util.visualization import plot_nanoparticle

from ray.tune.integration.pytorch_lightning import TuneReportCallback
import datetime
from matplotlib import pyplot as plt
import numpy as np
from fireworks.fw_config import LAUNCHPAD_LOC
from torch.nn import functional as F
from joblib import Parallel, delayed
from threading import Lock


class NPMCTrainer():
    def __init__(self,
                 data_module,
                 model_cls,
                 wandb_entity: Optional[str] = None, 
                 wandb_project: Optional[str] = 'default_project',
                 wandb_save_dir: Optional[str] = os.environ['HOME'],
                 gpu: Optional[bool] = False,
                 n_available_devices: Optional[int] = 4,
                 augment_loss: Optional[int] = None):
        self.data_module = data_module
        self.model_cls = model_cls
        self.augment_loss = augment_loss

        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.wandb_save_dir = wandb_save_dir
        self.n_available_devices = n_available_devices
        self.gpu = gpu

        self.free_devices = set(range(n_available_devices))
        self.lock = Lock()

    def acquire_device(self):
        with self.lock:
            id = self.free_devices.pop()
        return id
    
    def release_device(self, id):
        with self.lock:
            self.free_devices.add(id)

    def train_one_model(self, 
                        model_config: dict,
                        num_epochs: Optional[int] = 10, 
                        wandb_name: Optional[str] = None,
                        tune = False,
                        lr_scheduler=get_sequential):

        # get a free gpu from the list
        device_id = self.acquire_device()
        trainer_device_config = {}
        if self.gpu:
            trainer_device_config['accelerator'] = 'gpu'
            trainer_device_config['devices'] = [device_id]
        else:
            trainer_device_config['accelerator'] = 'auto'

        # Make the model
        model = self.model_cls(lr_scheduler=lr_scheduler,
                          optimizer_type='adam',
                          **model_config)

        # Make WandB logger
        wandb_logger = WandbLogger(name=wandb_name,
                                   entity=self.wandb_entity, 
                                   project=self.wandb_project, 
                                   save_dir=self.wandb_save_dir,
                                   log_model=True)

        # Configure callbacks
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval='step'))

        if self.augment_loss:
            callbacks.append(LossAugmentCallback(aug_loss_epoch=self.augment_loss))
        # callbacks.append(EarlyStopping(monitor='val_loss', patience=100))
        # callbacks.append(StochasticWeightAveraging(swa_lrs=1e-3))
        if tune:
            callbacks.append(TuneReportCallback({"loss": "val_loss_without_totals"}, on="validation_end"))
        checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="val_loss_without_totals", save_last=True)
        callbacks.append(checkpoint_callback)

        # Make the trainer
        trainer = pl.Trainer(max_epochs=num_epochs, 
                             enable_progress_bar=False, 
                             logger=wandb_logger, 
                             callbacks=callbacks,
                             **trainer_device_config)

        trainer.fit(model=model, datamodule=self.data_module)
        
        # Calculate the testing loss
        trainer.test(dataloaders=self.data_module.test_dataloader(), ckpt_path='best')
        trainer.validate(dataloaders=self.data_module.val_dataloader(), ckpt_path='best')

        # Load the best model
        model = self.model_cls.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.eval()

        # Get sample nanoparticle predictions within the test set
        columns = ['nanoparticle', 'spectrum', 'zoomed_spectrum', 'loss', 'loss_with_totals', 'npmc_qy', 'pred_qy']
        save_data = []
        rng = np.random.default_rng(seed = 10)
        for i in rng.choice(range(len(self.data_module.npmc_test)), 20, replace=False):
            data = self.data_module.npmc_test[i]
            save_data.append(self.get_logged_data(trainer.model, data))

        wandb_logger.log_table(key='sample_table', columns=columns, data=save_data)
        wandb.finish()

        # Indicate that the current gpu is now available
        self.release_device(device_id)
    
    def get_logged_data(self, model, data):
        y_hat, loss_without_totals, loss_with_totals = model._evaluate_step(data)
        
        npmc_spectrum = data.y
        pred_spectrum = np.power(10, y_hat.detach().numpy())

        fig = plt.figure(dpi=150)
        plt.plot(data.spectra_x, npmc_spectrum, label='NPMC', alpha=1)
        plt.plot(data.spectra_x, pred_spectrum, label='NN', alpha=0.5)
        plt.xlabel('Wavelength (nm)', fontsize=18)
        plt.ylabel('Relative Intensity (a.u.)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        fig.canvas.draw()
        full_fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        full_fig_data = full_fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        
        plt.ylim(0, 1e4)
        plt.tight_layout()
        fig.canvas.draw()
        fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        nanoparticle = plot_nanoparticle(data.constraints, data.dopant_specifications, as_np_array=True)
        
        npmc_qy = npmc_spectrum[:data.idx_zero].sum() / npmc_spectrum[data.idx_zero:].sum()
        pred_qy = pred_spectrum[:data.idx_zero].sum() / pred_spectrum[data.idx_zero:].sum()
        
        plt.close(fig)
        
        return [wandb.Image(nanoparticle),
                wandb.Image(full_fig_data), 
                wandb.Image(fig_data), 
                loss_without_totals, 
                loss_with_totals,
                npmc_qy,
                pred_qy]

    def train_many_models(self, 
                          model_configs: List[dict],
                          num_epochs: Optional[int] = 10, 
                          wandb_name: Optional[str] = None,
                          tune = False,
                          lr_scheduler=get_sequential):
        training_runs = []
        for model_config in model_configs:
            _run_config = {
                'model_config': model_config, 
                'num_epochs': num_epochs,
                'wandb_name': wandb_name,
                'tune': tune,
                'lr_scheduler': lr_scheduler
            }
            training_runs.append(_run_config)
        
        Parallel(n_jobs=self.n_available_gpus)(delayed(train_spectrum_model)(run_config) for run_config in training_runs)

def train_spectrum_model(config: dict, 
                         model_cls: SpectrumModelBase,
                         feature_processor: DataProcessor,
                         label_processor: Optional[DataProcessor] = None,
                         num_epochs: Optional[int] = 10, 
                         num_gpus: Union[int, float] = 0,
                         wandb_name: Optional[str] = None,
                         wandb_project: Optional[str] = 'default_project',
                         wandb_save_dir: Optional[str] = os.environ['HOME'],
                         tune = False,
                         lr_scheduler=get_sequential,
                         data_module: Optional[NPMCDataModule] = None,
                         n_points_avg: Optional[int] = None, 
                         batch_size: Optional[int] = 16):
    """
    
    :param config: a config dictionary for the model
        ex - {'n_input_nodes': 20, 'n_output_nodes'; 200, 'n_hidden_layers': 2, 'n_hidden_1': 128, 'n_hidden_2': 128, 'learning_rate': 1e-3}
    :param num_epochs: max number of epochs to train the model. Early stopping may result in less training epochs
    :param num_gpus: number of gpus to use for the training. May be a fractional number if training multiple models concurrently on one gpu
    :param wandb_project: wandb_project name
    :param wandb_save_dir: Directory to save wandb files to
    :param tune: whether or not this is a tune experiment. (To make sure that TuneReportCallback is added)
    """
    if data_module is None:
        if os.path.exists('npmc_data.json'):
            data_module = NPMCDataModule(feature_processor=feature_processor, label_processor=label_processor, data_dir='npmc_data.json', batch_size=batch_size)
        else:
            data_store = MongoStore.from_launchpad_file(LAUNCHPAD_LOC, 'avg_npmc_20220708')
            data_module = NPMCDataModule(feature_processor=feature_processor, label_processor=label_processor, data_store=data_store, batch_size=batch_size)


    # Make the model
    model = model_cls(lr_scheduler=lr_scheduler,
                      optimizer_type='adam',
                      **config)
    
    # Prime the module by passing in one datapoint.
    # This is a precaution in case the model uses lazy parameters. This will prevent errors with respect to lack of weight initialization 
    try:
        y_hat = model(data_module.npmc_train[0])
    except:
        data_module.prepare_data()
        data_module.setup()

    # Make logger
    wandb_logger = WandbLogger(name=wandb_name,
                               entity="esivonxay", 
                               project=wandb_project, 
                               save_dir=wandb_save_dir,
                               log_model=True)

    # Configure callbacks
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(LossAugmentCallback(aug_loss_epoch=1000))
    # callbacks.append(EarlyStopping(monitor='val_loss', patience=100))
    # callbacks.append(StochasticWeightAveraging(swa_lrs=1e-3))
    if tune:
        callbacks.append(TuneReportCallback({"loss": "val_loss_without_totals"}, on="validation_end"))
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss_without_totals", save_last=True)
    callbacks.append(checkpoint_callback)
    

    # Run the training
    if num_gpus > 0:
        trainer = pl.Trainer(accelerator = 'gpu', 
                             devices = math.ceil(num_gpus), 
                             max_epochs=num_epochs, 
                             enable_progress_bar=False, 
                             logger=wandb_logger, 
                             callbacks=callbacks)
    else:
        trainer = pl.Trainer(max_epochs=num_epochs, 
                             enable_progress_bar=False, 
                             logger=wandb_logger, 
                             callbacks=callbacks)

    try:
        trainer.fit(model=model, datamodule=data_module)
    except Exception as e:
        # Don't do anything with the exception to allow the wandb logger to finish
        print(e)
    
    # Calculate the testing loss
    trainer.test(dataloaders=data_module.test_dataloader(), ckpt_path='best')
    trainer.validate(dataloaders=data_module.val_dataloader())

    columns = ['name', 'nanoparticle', 'spectrum', 'zoomed_spectrum', 'MSE loss', 'L1 loss', 'Smooth L1 loss', 'npmc_qy', 'pred_qy']
    save_data = []
    rng = np.random.default_rng(seed = 10)

    # Load the best model
    model = model_cls.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.eval()

    for i in rng.choice(range(len(data_module.npmc_test)), 20, replace=False):
        data = data_module.npmc_test[i]
        y_hat, loss = model._evaluate_step(data)
        
        npmc_spectrum = data.y
        pred_spectrum = np.power(10, y_hat.detach().numpy())

        fig = plt.figure(dpi=150)
        plt.plot(data.spectra_x, npmc_spectrum, label='NPMC', alpha=1)
        plt.plot(data.spectra_x, pred_spectrum, label='NN', alpha=0.5)
        plt.xlabel('Wavelength (nm)', fontsize=18)
        plt.ylabel('Relative Intensity (a.u.)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        fig.canvas.draw()
        full_fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        full_fig_data = full_fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        
        plt.ylim(0, 1e4)
        plt.tight_layout()
        fig.canvas.draw()
        fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        nanoparticle = plot_nanoparticle(data.constraints, data.dopant_specifications, as_np_array=True)
        
        npmc_qy = npmc_spectrum[:data.idx_zero].sum() / npmc_spectrum[data.idx_zero:].sum()
        pred_qy = pred_spectrum[:data.idx_zero].sum() / pred_spectrum[data.idx_zero:].sum()
        
        save_data.append([wandb_logger.experiment.name, 
                     wandb.Image(nanoparticle),
                     wandb.Image(full_fig_data), 
                     wandb.Image(fig_data), 
                     loss, 
                     F.l1_loss(y_hat, data.log_y),
                     F.smooth_l1_loss(y_hat, data.log_y),
                     npmc_qy,
                     pred_qy])
        plt.close(fig)
        plt.close(nanoparticle)
        # trainer.logger.experiment.log({'spectrum': fig})
    wandb_logger.log_table(key='sample_table', columns=columns, data=save_data)
    # Finalize the wandb logging
    wandb.finish()

    return model


def tune_npmc_asha(config: dict, 
                   model_cls: SpectrumModelBase,
                   feature_processor: DataProcessor,
                   label_processor: Optional[EnergyLabelProcessor] = None,
                   num_samples: Optional[int] = 10, 
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

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=min(max(num_epochs//10, 1), 100),
        reduction_factor=2)

    reporter = TrialTerminationReporter(
        parameter_columns=["nn_layers", "learning_rate", "l2_regularization_weight", "dropout_probability"],
        metric_columns=["loss", "training_iteration"])


    if wandb_project is None:
        date = ''.join([f'{val}{sym}'for val, sym in zip(datetime.datetime.now().isoformat().split('.')[0].split(':'), ['h', 'm', 's'])])
        wandb_project = f'Raytune-{date}'
    if save_dir is None:
        save_dir = os.path.join(os.environ['HOME'], 'train_output')

    train_fn_with_parameters = tune.with_parameters(train_spectrum_model,
                                                    model_cls = model_cls,
                                                    feature_processor = feature_processor,
                                                    label_processor = label_processor,
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

def get_np_template_from_feature(types, volumes, compositions, feature_processor):
    possible_elements = feature_processor.possible_elements
    
    types = types.reshape(-1, len(possible_elements))
    compositions = compositions.reshape(-1, len(possible_elements))
    dopant_specifications = []
    for i in range(types.shape[0]):
        for j in range(types.shape[1]):
            dopant_specifications.append((i, compositions[i][j].item(), possible_elements[j], 'Y'))

    layer_volumes = volumes.reshape(-1, len(possible_elements))[:, 0]
    cum_volumes = torch.cumsum(layer_volumes, dim=0)
    radii = torch.pow(cum_volumes * 3 / (4 * np.pi), 1/3)*100
    constraints = [SphericalConstraint(radius.item()) for radius in radii]
    return constraints, dopant_specifications