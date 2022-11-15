import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Callable, Optional, Union, List


class SpectrumModelBase(pl.LightningModule):
    def __init__(self,
                 learning_rate: Optional[float]=1e-5,
                 l2_regularization_weight: float = 0,
                 lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]=None,
                 loss_function: Optional[Callable[[List, List], float]] = F.mse_loss,
                 additional_metadata: Optional[dict] = {},
                 optimizer_type: Optional[str] = None,
                 augment_loss: Optional[bool] = False,
                 **kwargs):

        super().__init__()

        if optimizer_type is None:
            self.optimizer_type = 'amsgrad'
        else:
            self.optimizer_type = optimizer_type
        
        self.learning_rate = learning_rate
        self.l2_regularization_weight = l2_regularization_weight
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.additional_metadata = additional_metadata
        self.augment_loss = augment_loss

        self.save_hyperparameters()

    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        """ 
        # Default to the adam optimizer               
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        else:
            # Default to amsgrad
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight, amsgrad=True)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        else:
            lr_scheduler = None
        return [optimizer], [lr_scheduler]
    
    def _evaluate_step(self, 
                       data):
        y_hat = self(data)
        
        if data.batch is not None:
            y = data.log_y.reshape(-1, self.n_output_nodes)
            # x = data.spectra_x.reshape(-1, self.n_output_nodes)
            idx_zero = data.idx_zero.flatten()[0]
        else:
            y = data.log_y
            # x = data.spectra_x
            idx_zero = data.idx_zero
        loss = self.loss_function(y_hat, y)
        
        # exponentiate to get the spectrum
        spectrum_hat = torch.pow(10, y_hat) 
        
        # Compute the augmented loss
        augmented_loss = loss.clone()
        # Add a loss term corresponding to total absorbed and emitted photons
        n_photons_absorbed = spectrum_hat[..., idx_zero:].sum(dim=-1)
        augmented_loss += self.loss_function(torch.log(n_photons_absorbed), torch.log(data.n_absorbed))

        n_photons_emitted = spectrum_hat[..., :idx_zero].sum(dim=-1)
        augmented_loss += self.loss_function(torch.log(n_photons_emitted), torch.log(data.n_emitted))

        # Add a loss term corresponding to quantum yield
        # qy_hat = n_photons_absorbed/n_photons_emitted
        # qy = data.n_emitted/data.n_absorbed
        # loss += self.loss_function(qy, qy_hat)
        
        # Add a term to enforce total energy emitted < total energy absorbed
        # total_energy = torch.mul(spectrum_hat, x)
        # e_absorbed = total_energy[..., idx_zero:].sum(dim=-1)
        # e_emitted = torch.abs(total_energy[..., :idx_zero].sum(dim=-1))
        # print(torch.log(torch.nn.functional.relu(e_emitted - e_absorbed)))
        # loss += 10 * torch.nn.functional.relu(torch.log(e_absorbed - e_emitted), 0).sum()
        
        return y_hat, loss, augmented_loss
    
    def _step(self, 
              loss_prefix, 
              batch, 
              batch_idx):
        _, loss_without_totals, loss_with_totals = self._evaluate_step(batch)

        # Determine the batch size
        if hasattr(batch, 'batch'):
            batch_size = batch.batch[-1]
        else:
            batch_size = 1

        # Log the losses with and without the total dNdt of absorption and emission
        self.log(f'{loss_prefix}_loss_with_totals', loss_with_totals, batch_size=batch_size)
        self.log(f'{loss_prefix}_loss_without_totals', loss_without_totals, batch_size=batch_size)

        # Determine which loss will be used in training
        if self.augment_loss:
            loss = loss_with_totals
        else:
            loss = loss_without_totals

        # Log the actual training loss
        self.log(f'{loss_prefix}_loss', loss, batch_size=batch_size)

        return loss

    def training_step(self, 
                      batch, 
                      batch_idx):
        return self._step('train', batch, batch_idx)
        
    def validation_step(self, 
                        batch, 
                        batch_idx):
        return self._step('val', batch, batch_idx)
        
    def test_step(self, 
                  batch, 
                  batch_idx):
        return self._step('test', batch, batch_idx)
        
    def predict_step(self, 
                     batch, 
                     batch_idx):
        pred, _, _ = self._evaluate_step(batch)
        return pred