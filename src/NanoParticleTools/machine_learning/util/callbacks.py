from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from typing import Optional
import wandb


class LogPredictionsCallback(Callback):
    """
    This pytorch lightning callback will periodically log sample images to Weights & Biases.

    Note: This may slow down training significantly if you are triggering this callback frequently.

    Warning: This callback may not be working in its current state
    """

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f'Ground Truth: {y_i} - Prediction: {y_pred}'
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger = trainer.logger  # TODO: check this is the correct logger
            wandb_logger.log_image(key='sample_images',
                                   images=images,
                                   caption=captions)

            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[
                wandb.Image(x_i), y_i, y_pred
            ] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table',
                                   columns=columns,
                                   data=data)


class LossAugmentCallback(Callback):
    """
    This pytorch lightning callback enables one to modify the loss function after
    a number of epochs has elapsed. This is useful for swapping out the loss function
    or reweighting terms in the loss function.

    As implemented, it flips a boolean in the model. The model needs utilize the
    `model.augment_loss` class variable and act upon it.
    """

    aug_loss_epoch: int

    def __init__(self, aug_loss_epoch: int = 100):
        self.aug_loss_epoch = aug_loss_epoch
        return super().__init__()

    def on_train_epoch_start(self, trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch >= self.aug_loss_epoch:
            pl_module.augment_loss = True
        return super().on_epoch_start(trainer, pl_module)
