import os
import datetime
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from NanoParticleTools.machine_learning.util.training import (
    train_spectrum_model, train_uv_model)


def tune_npmc(model_cls,
              config,
              num_epochs,
              data_module,
              lr_scheduler,
              save_dir='./',
              wandb_config={},
              use_gpu=True,
              algo='full',
              num_samples=100,
              train_fn=None):
    if train_fn is None:
        train_fn = train_spectrum_model

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
                            num_samples=num_samples,
                            scheduler=scheduler,
                            local_dir=save_dir,
                            search_alg=algo,
                            name="tune_npmc_bohb")
    elif algo.lower() == 'asha':
        # Default to asha
        # yapf: disable
        scheduler = ASHAScheduler(
            metric='loss',  # Metric refers to the one we have mapped to in the TuneReportCallback
            mode='min',
            max_t=num_epochs,
            grace_period=100,
            reduction_factor=2)

        # yapf: enable
        analysis = tune.run(train_fn_with_parameters,
                            resources_per_trial=resources_per_trial,
                            config=config,
                            num_samples=num_samples,
                            scheduler=scheduler,
                            reuse_actors=False,
                            local_dir=save_dir,
                            name="tune_npmc_asha")
    else:
        analysis = tune.run(train_fn_with_parameters,
                            resources_per_trial=resources_per_trial,
                            config=config,
                            num_samples=num_samples,
                            reuse_actors=False,
                            local_dir=save_dir,
                            name="tune_npmc_full")

    print("Best hyperparameters found were: ", analysis.best_config)
