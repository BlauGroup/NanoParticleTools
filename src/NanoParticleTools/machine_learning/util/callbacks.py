from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from typing import Optional
import wandb
from torch.nn import functional as F


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


class WeightedMultiLoss(Callback):
    """
    This pytorch lightning callback enables one to use a weighted-sum loss function which has
    weights that change during training. Currently, this supports linear interpolation between
    the initial and target weights.
    """
    initial_mse_factor: float
    initial_mae_factor: float
    initial_cos_sim_factor: float
    target_mse_factor: float
    target_mae_factor: float
    target_cos_sim_factor: float
    target_epoch: int

    def __init__(self,
                 initial_mse_factorial_mse: float = 1.0,
                 initial_mae_factor: float = 0.0,
                 initial_cos_sim_factor: float = 0.0,
                 target_mse_factor: float = 0.0,
                 target_mae_factor: float = 1.0,
                 target_cos_sim_factor: float = 1.0,
                 target_epoch: int = 100):
        self.initial_mse_factor = initial_mse_factorial_mse
        self.initial_mae_factor = initial_mae_factor
        self.initial_cos_sim_factor = initial_cos_sim_factor
        self.target_mse_factor = target_mse_factor
        self.target_mae_factor = target_mae_factor
        self.target_cos_sim_factor = target_cos_sim_factor
        self.target_epoch = target_epoch
        return super().__init__()

    def get_loss_function(self,
                          mse_factor: float = 1.0,
                          mae_factor: float = 1.0,
                          cos_sim_factor: float = 1.0):

        def loss_function(y_hat, y):
            mse_loss = F.mse_loss(y_hat, y)
            mae_loss = F.l1_loss(y_hat, y)
            cos_sim = F.cosine_similarity(y_hat, y, 1).mean(0)
            return (mse_factor * mse_loss + mae_factor * mae_loss -
                    cos_sim_factor * cos_sim)

        return loss_function

    def get_factor(self, initial_factor: float, target_factor: float,
                   current_epoch: int):
        epochs = min(current_epoch, self.target_epoch)
        factor = ((target_factor - initial_factor) *
                  (epochs / self.target_epoch) + initial_factor)
        return factor

    def on_train_epoch_start(self, trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule") -> None:
        mse_factor = self.get_factor(self.initial_mse_factor,
                                     self.target_mse_factor,
                                     trainer.current_epoch)
        mae_factor = self.get_factor(self.initial_mae_factor,
                                     self.target_mae_factor,
                                     trainer.current_epoch)
        cos_sim_factor = self.get_factor(self.initial_cos_sim_factor,
                                         self.target_cos_sim_factor,
                                         trainer.current_epoch)
        pl_module.loss_function = self.get_loss_function(
            mse_factor, mae_factor, cos_sim_factor)
        return super().on_epoch_start(trainer, pl_module)
