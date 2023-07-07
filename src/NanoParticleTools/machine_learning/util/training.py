from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
from NanoParticleTools.machine_learning.util.learning_rate import get_sequential
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import (LearningRateMonitor,
                                         StochasticWeightAveraging,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

import wandb
import pytorch_lightning as pl
import os
from ray.tune.schedulers import ASHAScheduler
from typing import Optional, List

from NanoParticleTools.util.visualization import plot_nanoparticle

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from threading import Lock
import torch
from pandas import DataFrame
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D


class NPMCTrainer():

    def __init__(self,
                 data_module,
                 model_cls,
                 wandb_entity: Optional[str] = None,
                 wandb_project: Optional[str] = None,
                 wandb_save_dir: Optional[str] = None,
                 gpu: Optional[bool] = False,
                 n_available_devices: Optional[int] = 4,
                 augment_loss: Optional[int] = None):
        self.data_module = data_module
        self.model_cls = model_cls
        self.augment_loss = augment_loss

        self.wandb_entity = wandb_entity

        if wandb_project is None:
            self.wandb_project = 'default_project'
        else:
            self.wandb_project = wandb_project

        if wandb_save_dir is None:
            self.wandb_save_dir = os.environ['HOME']
        else:
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


def get_metrics(model, dataset):
    output_type_sorted = None
    if hasattr(dataset[0], 'metadata') and dataset[0].metadata is not None:
        output_type_sorted = {}

    output = []
    for data in dataset:
        if output_type_sorted is not None:
            data_label = data.metadata['tags'][0]
        y_hat = model(**data.to_dict(), batch=None).detach()
        # Calculate the metrics
        _output = [
            torch.nn.functional.cosine_similarity(y_hat, data.log_y,
                                                  dim=1).mean().item(),
            torch.nn.functional.mse_loss(y_hat, data.log_y).item(),
            torch.nn.functional.cosine_similarity(y_hat[:, 200:257],
                                                  data.log_y[:, 200:257],
                                                  dim=1).mean().item(),
            torch.nn.functional.mse_loss(y_hat[:, 200:257],
                                         data.log_y[:, 200:257]).item()
        ]
        output.append(_output)
        if output_type_sorted is not None:
            try:
                output_type_sorted[data_label].append(_output)
            except KeyError:
                output_type_sorted[data_label] = [_output]

    output = torch.tensor(output)
    if output_type_sorted is not None:
        for key in output_type_sorted:
            output_type_sorted[key] = torch.tensor(output_type_sorted[key])

    return output, output_type_sorted


def log_additional_data(model, data_module, wandb_logger):
    columns = ['Cos Sim', 'MSE', 'UV Cos Sim', 'UV MSE']

    # Run the data metrics on the train data
    output, output_type_sorted = get_metrics(model, data_module.npmc_train)
    overall_train_metrics = {
        'train_mean': output.mean(0).tolist(),
        'train_std': output.std(0).tolist(),
        'train_min': output.min(0).values.tolist(),
        'train_max': output.max(0).values.tolist(),
        'train_median': output.median(0).values.tolist(),
    }
    df = DataFrame(
        overall_train_metrics,
        index=['cosine similarity', 'mse', 'UV cosine similarity', 'UV mse'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'metric'})
    wandb_logger.log_table('overall_train_metrics', dataframe=df)
    # wandb.log({'overall_train_metrics': wandb.Table(dataframe=df)})
    violin_fig = get_violin_plot(output, 'Train')
    wandb_train_violin_fig = fig_to_wandb_image(violin_fig)
    # Close the figure
    plt.close(violin_fig)

    # Run the data metrics on the train data
    output, output_type_sorted = get_metrics(model, data_module.npmc_val)
    overall_val_metrics = {
        'val_mean': output.mean(0).tolist(),
        'val_std': output.std(0).tolist(),
        'val_min': output.min(0).values.tolist(),
        'val_max': output.max(0).values.tolist(),
        'val_median': output.median(0).values.tolist(),
    }
    df = DataFrame(
        overall_val_metrics,
        index=['cosine similarity', 'mse', 'UV cosine similarity', 'UV mse'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'metric'})
    wandb_logger.log_table('overall_val_metrics', dataframe=df)
    # wandb.log({'overall_val_metrics': wandb.Table(dataframe=df)})
    violin_fig = get_violin_plot(output, 'Validation')
    wandb_val_violin_fig = fig_to_wandb_image(violin_fig)

    # Close the figure
    plt.close(violin_fig)

    # Run the data metrics on the test data
    output, output_type_sorted = get_metrics(model, data_module.npmc_test)
    overall_test_metrics = {
        'test_mean': output.mean(0).tolist(),
        'test_std': output.std(0).tolist(),
        'test_min': output.min(0).values.tolist(),
        'test_max': output.max(0).values.tolist(),
        'test_median': output.median(0).values.tolist(),
    }
    test_metrics_by_class = {
        key: item.mean(0).tolist()
        for key, item in output_type_sorted.items()
    }
    df = DataFrame(
        overall_test_metrics,
        index=['cosine similarity', 'mse', 'UV cosine similarity', 'UV mse'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'metric'})
    wandb_logger.log_table('overall_test_metrics', dataframe=df)
    # wandb.log({'overall_test_metrics': wandb.Table(dataframe=df)})
    df = DataFrame(
        test_metrics_by_class,
        index=['cosine similarity', 'mse', 'UV cosine similarity', 'UV mse']).T
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'metric'})
    wandb_logger.log_table('test_metrics_by_class', dataframe=df)
    # wandb.log({'test_metrics_by_class': wandb.Table(dataframe=df)})

    # Get the figures for the test data
    violin_fig = get_violin_plot(output, 'Test')
    wandb_test_violin_fig = fig_to_wandb_image(violin_fig)
    test_fig = get_test_figure(output_type_sorted)
    wandb_test_fig = fig_to_wandb_image(test_fig)

    # Close the figures
    plt.close(violin_fig)
    plt.close(test_fig)

    # yapf: disable
    # table = wandb.Table(columns=['Train Violin Plot', 'Validation Violin Plot',
    #                              'Test Violin Plot', 'Test Split Figure'],
    #                     data=[[wandb_train_violin_fig, wandb_val_violin_fig,
    #                            wandb_test_violin_fig, wandb_test_fig]])
    wandb_logger.log_table('Metric Figures',
                           columns=['Train Violin Plot', 'Validation Violin Plot',
                                    'Test Violin Plot', 'Test Split Figure'],
                           data=[[wandb_train_violin_fig, wandb_val_violin_fig,
                                  wandb_test_violin_fig, wandb_test_fig]])
    # yapf: enable

    # wandb.log({'Metric Figures': table})
    # wandb.log({
    #     'Train Violin Plot': wandb_train_violin_fig,
    #     'Validation Violin Plot': wandb_val_violin_fig,
    #     'Test Violin Plot': wandb_test_violin_fig,
    #     'Test Split Figure': wandb_test_fig
    # })
    # wandb_logger.log_image('Train Violin Plot', wandb_train_violin_fig)
    # wandb_logger.log_image('Validation Violin Plot', wandb_val_violin_fig)
    # wandb_logger.log_image('Test Violin Plot', wandb_test_violin_fig)
    # wandb_logger.log_image('Test Split Figure', wandb_test_fig)


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
                         trainer_device_config: Optional[dict] = None, 
                         additional_callbacks: Optional[List] = None):
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

    if additional_callbacks is not None:
        # Allow for custom callbacks to be passed in
        callbacks.extend(additional_callbacks)

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
    train_metrics = {}
    factor = 1 / len(data_module.train_dataloader())
    for batch_idx, batch in enumerate(data_module.train_dataloader()):
        _, _loss_d = model._step('train_eval', batch, batch_idx, log=False)
        for key in _loss_d:
            try:
                train_metrics[key] += _loss_d[key].item() * factor
            except KeyError:
                train_metrics[key] = _loss_d[key] * factor
    wandb_logger.log_metrics(train_metrics)

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

    # Log the additional data
    log_additional_data(model, data_module, wandb_logger)
    wandb.finish()

    return model


def train_uv_model(config,
                   model_cls,
                   data_module,
                   lr_scheduler,
                   initial_model: Optional[pl.LightningModule] = None,
                   num_epochs: Optional[int] = 2000,
                   augment_loss=False,
                   ray_tune: Optional[bool] = False,
                   early_stop: Optional[bool] = False,
                   swa: Optional[bool] = False,
                   save_checkpoints: Optional[bool] = True,
                   wandb_config: Optional[dict] = None,
                   trainer_device_config: Optional[dict] = None,
                   additional_callbacks: Optional[List] = None):
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
    if initial_model is None: 
        model = model_cls(lr_scheduler=lr_scheduler,
                          optimizer_type='adam',
                          **config)
    else: 
        model = initial_model
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

    if additional_callbacks is not None:
        # Allow for custom callbacks to be passed in
        callbacks.extend(additional_callbacks)

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
    train_metrics = {}
    factor = 1 / len(data_module.train_dataloader())
    for batch_idx, batch in enumerate(data_module.train_dataloader()):
        _, _loss_d = model._step('train_eval', batch, batch_idx, log=False)
        for key in _loss_d:
            try:
                train_metrics[key] += _loss_d[key].item() * factor
            except KeyError:
                train_metrics[key] = _loss_d[key] * factor
    wandb_logger.log_metrics(train_metrics)

    # Validation metrics
    trainer.validate(dataloaders=data_module.val_dataloader(),
                     ckpt_path='best')
    # Test metrics
    trainer.test(dataloaders=data_module.test_dataloader(), ckpt_path='best')

    wandb.finish()

    return model


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


def fig_to_wandb_image(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return wandb.Image(data)


def get_violin_plot(output, title='Test'):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    ax1 = ax.twinx()
    columns = ['Cos Sim', 'MSE', 'UV Cos Sim', 'UV MSE']

    vp = ax.violinplot(output[..., 0].numpy(), [0],
                       showmeans=True,
                       showmedians=True)
    for pc in vp['bodies']:
        pc.set_facecolor('tab:blue')
        # pc.set_edgecolor('black')
        pc.set_alpha(0.65)
    for line_label in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        vp[line_label].set_color('k')
        vp[line_label].set_linewidth(0.75)
    vp['cbars'].set_linewidth(0.25)
    vp['cmedians'].set_linestyle('--')
    vp = ax1.violinplot(output[..., 1].log10().numpy(), [1],
                        showmeans=True,
                        showmedians=True)
    for pc in vp['bodies']:
        pc.set_facecolor('tab:red')
        # pc.set_edgecolor('black')
        pc.set_alpha(0.65)
    for line_label in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        vp[line_label].set_color('k')
        vp[line_label].set_linewidth(0.75)
    vp['cbars'].set_linewidth(0.25)
    vp['cmedians'].set_linestyle('--')
    vp = ax.violinplot(output[..., 2].numpy(), [2],
                       showmeans=True,
                       showmedians=True)
    for pc in vp['bodies']:
        pc.set_facecolor('tab:blue')
        # pc.set_edgecolor('black')
        pc.set_alpha(0.65)
    for line_label in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        vp[line_label].set_color('k')
        vp[line_label].set_linewidth(0.75)
    vp['cbars'].set_linewidth(0.25)
    vp['cmedians'].set_linestyle('--')
    vp = ax1.violinplot(output[..., 3].log10().numpy(), [3],
                        showmeans=True,
                        showmedians=True)
    for pc in vp['bodies']:
        pc.set_facecolor('tab:red')
        # pc.set_edgecolor('black')
        pc.set_alpha(0.65)
    vp['cbars'].set_linewidth(0.25)
    for line_label in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        vp[line_label].set_color('k')
        vp[line_label].set_linewidth(0.75)
    vp['cbars'].set_linewidth(0.25)
    vp['cmedians'].set_linestyle('--')

    ax.tick_params(axis='y', colors='tab:blue')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(columns, fontsize=14)
    ax.set_ylabel('Cosine Similarity', color='tab:blue', fontsize=18)
    ax1.tick_params(axis='y', colors='tab:red')
    ax1.set_yticks(np.arange(-2, 1, 1))
    ax1.yaxis.set_major_formatter(
        mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    # ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)
    ax1.set_ylabel(r'$log_{10}(MSE)$', color='tab:red', fontsize=18)
    ax.set_title(f"{title} Data Metrics", fontsize=20)
    plt.tight_layout()
    return fig


def get_test_figure(output_type_sorted):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    ax1 = ax.twinx()
    ax1.semilogy()
    x_labels = list(output_type_sorted.keys())
    columns = ['Cos Sim', 'MSE', 'UV Cos Sim', 'UV MSE']

    for i, label in zip([0, 1, 2, 3], columns):
        x = torch.arange(len(x_labels))
        y = torch.tensor([_l.mean(0)[i] for _l in output_type_sorted.values()])
        # if i < 2:
        #     fmt = 'o'
        # else:
        #     fmt = 'D'
        if i % 2 == 0:
            color = 'tab:blue'
            fmt = 'o' if i < 2 else 'D'
            ax.plot(x, y, fmt, color=color, alpha=0.6, markeredgecolor='k')
        else:
            color = 'tab:red'
            fmt = 'o' if i < 2 else 'D'
            ax1.plot(x, y, fmt, color=color, alpha=0.6, markeredgecolor='k')

    legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='k',
               label='Full Spectrum',
               markerfacecolor='grey',
               linewidth=0,
               alpha=0.6),
        Line2D([0], [0],
               marker='D',
               color='k',
               label='UV Section',
               markerfacecolor='grey',
               linewidth=0,
               alpha=0.6)
    ]
    plt.legend(handles=legend_elements, loc='center right')
    ax.tick_params(axis='y', colors='tab:blue')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(columns, fontsize=14)
    ax.set_ylabel('Cosine Similarity', color='tab:blue', fontsize=18)
    ax1.tick_params(axis='y', which='both', colors='tab:red')
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)
    ax1.set_ylabel('MSE', color='tab:red', fontsize=18)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=55)
    plt.tight_layout()
    return fig
