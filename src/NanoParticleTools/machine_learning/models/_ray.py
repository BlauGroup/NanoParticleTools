from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
from NanoParticleTools.machine_learning.util.learning_rate import get_sequential
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    StochasticWeightAveraging,
    ModelCheckpoint
)
from pytorch_lightning.loggers import WandbLogger

import wandb
import pytorch_lightning as pl
import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from typing import Optional, List

from NanoParticleTools.util.visualization import plot_nanoparticle

from ray.tune.integration.pytorch_lightning import TuneReportCallback
import datetime
from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from threading import Lock
import torch


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

        self.free_devices = list(range(n_available_devices))
        self.lock = Lock()

    def acquire_device(self):
        with self.lock:
            _id = self.free_devices.pop()
        return _id

    def release_device(self, _id):
        with self.lock:
            self.free_devices.append(_id)

    def train_one_model(self,
                        model_config: dict,
                        num_epochs: Optional[int] = 10,
                        wandb_name: Optional[str] = None,
                        lr_scheduler=get_sequential):

        # get a free gpu from the list
        device_id = self.acquire_device()
        trainer_device_config = {}
        if self.gpu:
            trainer_device_config['accelerator'] = 'gpu'
            trainer_device_config['devices'] = [device_id]
        else:
            trainer_device_config['accelerator'] = 'auto'

        wandb_config = {
            'name': wandb_name,
            'entity': self.wandb_entity,
            'project': self.wandb_project,
            'save_dir': self.wandb_save_dir
        }

        model = train_spectrum_model(
            model_cls=self.model_cls,
            config=model_config,
            data_module=self.data_module,
            lr_scheduler=lr_scheduler,
            num_epochs=num_epochs,
            augment_loss=self.augment_loss,
            ray_tune=False,
            early_stop=False,
            swa=False,
            save_checkpoints=True,
            wandb_config=wandb_config,
            trainer_device_config=trainer_device_config)

        # Indicate that the current gpu is now available
        self.release_device(device_id)

        return model

    def train_many_models(self,
                          model_configs: List[dict],
                          num_epochs: Optional[int] = 10,
                          wandb_name: Optional[str] = None,
                          tune=False,
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

        Parallel(n_jobs=self.n_available_devices)(
            delayed(self.train_one_model)(run_config)
            for run_config in training_runs)


def get_logged_data(model, data):
    y_hat, loss = model._evaluate_step(data)

    spectra_x = data.spectra_x.squeeze()
    npmc_spectrum = data.y.squeeze()
    pred_spectrum = np.power(10, y_hat.detach().numpy()).squeeze()

    fig = plt.figure(dpi=150)
    plt.plot(spectra_x, npmc_spectrum, label='NPMC', alpha=1)
    plt.plot(spectra_x, pred_spectrum, label='NN', alpha=0.5)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Relative Intensity (a.u.)', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()

    fig.canvas.draw()
    full_fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    full_fig_data = full_fig_data.reshape(fig.canvas.get_width_height()[::-1] +
                                          (3, ))

    plt.ylim(0, 1e4)
    plt.tight_layout()
    fig.canvas.draw()
    fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    nanoparticle = plot_nanoparticle(data.constraints,
                                     data.dopant_specifications,
                                     as_np_array=True)

    npmc_qy = npmc_spectrum[:data.idx_zero].sum(
    ) / npmc_spectrum[data.idx_zero:].sum()
    pred_qy = pred_spectrum[:data.idx_zero].sum(
    ) / pred_spectrum[data.idx_zero:].sum()

    plt.close(fig)

    return [
        wandb.Image(nanoparticle),
        wandb.Image(full_fig_data),
        wandb.Image(fig_data), loss, npmc_qy, pred_qy
    ]


def train_spectrum_model(config,
                         model_cls,
                         data_module,
                         lr_scheduler,
                         num_epochs: Optional[int] = 2000,
                         augment_loss=False,
                         ray_tune: Optional[bool] = False,
                         early_stop: Optional[bool] = False,
                         swa: Optional[bool] = False,
                         save_checkpoints: Optional[bool] = True,
                         wandb_config: Optional[dict] = None,
                         trainer_device_config: Optional[dict] = None):
    """
        params
        model_cls:
        model_config:
        lr_scheduler:
        augment_loss:
        ray_tune: whether or not this is a ray tune run
        early_stop: whether or not to use early stopping
        swa: whether or not to use stochastic weight averaging

        """
    if trainer_device_config is None:
        trainer_device_config = {'accelerator': 'auto'}

    if wandb_config is None:
        wandb_config = {'name': None}

    # Make the model
    model = model_cls(lr_scheduler=lr_scheduler,
                      optimizer_type='adam',
                      **config)

    # Make WandB logger
    wandb_logger = WandbLogger(log_model=True, **wandb_config)

    # Configure callbacks
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Disable augment loss for now. It seems to not help
    # if augment_loss:
    #     callbacks.append(LossAugmentCallback(aug_loss_epoch=augment_loss))
    if early_stop:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=200))
    if swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-3))
    if ray_tune:
        callbacks.append(
            TuneReportCallback({"loss": "val_loss"}, on="validation_end"))
    if save_checkpoints:
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor="val_loss",
                                              save_last=True)
        callbacks.append(checkpoint_callback)

    # Make the trainer
    trainer = pl.Trainer(max_epochs=num_epochs,
                         enable_progress_bar=False,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         **trainer_device_config)

    trainer.fit(model=model, datamodule=data_module)

    # Load the best model checkpoint, set it to evaluation mode, and then evaluate the metrics
    # for the training, validation, and test sets
    model = model_cls.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.eval()

    # Train metrics

    # Validation metrics
    trainer.validate(dataloaders=data_module.val_dataloader(),
                     ckpt_path='best')
    # Test metrics
    trainer.test(dataloaders=data_module.test_dataloader(), ckpt_path='best')

    # Get sample nanoparticle predictions within the test set
    columns = [
        'nanoparticle', 'spectrum', 'zoomed_spectrum', 'loss', 'npmc_qy',
        'pred_qy'
    ]
    save_data = []
    rng = np.random.default_rng(seed=10)
    for i in rng.choice(range(len(data_module.npmc_test)), 20, replace=False):
        data = data_module.npmc_test[i]
        save_data.append(get_logged_data(trainer.model, data))

    wandb_logger.log_table(key='sample_table', columns=columns, data=save_data)
    wandb.finish()

    return model


def tune_npmc(model_cls,
              config,
              num_epochs,
              data_module,
              lr_scheduler,
              save_dir='./',
              wandb_config={},
              use_gpu=True,
              algo='asha'):

    date = ''.join([
        f'{val}{sym}' for val, sym in zip(
            datetime.datetime.now().isoformat().split('.')[0].split(':'),
            ['h', 'm', 's'])
    ])
    default_wandb_project = f'Raytune-{date}'
    default_save_dir = os.path.join(os.environ['HOME'], 'train_output')

    default_wandb_config = {
        'project': default_wandb_project,
        'save_dir': default_save_dir
    }
    default_wandb_config.update(wandb_config)

    if use_gpu:
        trainer_device_config = {'accelerator': 'gpu', 'devices': 1}
        resources_per_trial = {'gpu': 1, 'cpu': 4}
    else:
        trainer_device_config = {'accelerator': 'auto'}
        resources_per_trial = {'cpu': 4}

    train_fn_with_parameters = tune.with_parameters(
        train_spectrum_model,
        model_cls=model_cls,
        data_module=data_module,
        lr_scheduler=lr_scheduler,
        num_epochs=num_epochs,
        augment_loss=False,
        ray_tune=True,
        early_stop=True,
        swa=False,
        save_checkpoints=False,
        trainer_device_config=trainer_device_config,
        wandb_config=default_wandb_config)

    if algo.lower() == 'bohb':
        from ray.tune.schedulers import HyperBandForBOHB
        from ray.tune.suggest.bohb import TuneBOHB

        algo = TuneBOHB(metric="val_loss", mode="min")
        scheduler = HyperBandForBOHB(time_attr="training_iteration",
                                     metric="val_loss",
                                     mode="min",
                                     max_t=100)

        analysis = tune.run(train_fn_with_parameters,
                            resources_per_trial=resources_per_trial,
                            config=config,
                            scheduler=scheduler,
                            local_dir=save_dir,
                            search_alg=algo,
                            name="tune_npmc_bohb")
    else:
        # Default to asha
        scheduler = ASHAScheduler(
            metric='loss',  # Metric refers to the one we have mapped to in the TuneReportCallback
            mode='min',
            max_t=num_epochs,
            grace_period=100,
            reduction_factor=2)

        analysis = tune.run(train_fn_with_parameters,
                            resources_per_trial=resources_per_trial,
                            config=config,
                            num_samples=1000,
                            scheduler=scheduler,
                            reuse_actors=False,
                            local_dir=save_dir,
                            name="tune_npmc_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


def get_np_template_from_feature(types, volumes, compositions,
                                 feature_processor):
    possible_elements = feature_processor.possible_elements

    types = types.reshape(-1, len(possible_elements))
    compositions = compositions.reshape(-1, len(possible_elements))
    dopant_specifications = []
    for i in range(types.shape[0]):
        for j in range(types.shape[1]):
            dopant_specifications.append(
                (i, compositions[i][j].item(), possible_elements[j], 'Y'))

    layer_volumes = volumes.reshape(-1, len(possible_elements))[:, 0]
    cum_volumes = torch.cumsum(layer_volumes, dim=0)
    radii = torch.pow(cum_volumes * 3 / (4 * np.pi), 1 / 3) * 100
    constraints = [SphericalConstraint(radius.item()) for radius in radii]
    return constraints, dopant_specifications
