from NanoParticleTools.machine_learning.core import SpectrumModelBase
from NanoParticleTools.machine_learning.util.learning_rate import ReduceLROnPlateauWithWarmup
from torch_geometric.data import Data, Batch
import torch


class TestModel(SpectrumModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nn = torch.nn.Linear(10, 4)

    def forward(self, x, **kwargs):
        return self.nn(x)


def test_base_model():

    def get_step_lr(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=1,
                                               gamma=0.1)

    model = TestModel(learning_rate=1e-5,
                      l2_regularization_weight=1e-4,
                      lr_scheduler=get_step_lr,
                      loss_function=torch.nn.functional.mse_loss,
                      additional_metadata={})

    assert model.learning_rate == 1e-5
    assert model.l2_regularization_weight == 1e-4
    assert model.loss_function == torch.nn.functional.mse_loss
    assert model.additional_metadata == {}


def test_configure_optimizers():
    model = TestModel(learning_rate=1e-5,
                      l2_regularization_weight=1e-4,
                      lr_scheduler=ReduceLROnPlateauWithWarmup,
                      lr_scheduler_kwargs={'warmup_epochs': 10,
                                           'patience': 100,
                                           'factor': 0.8},
                      loss_function=torch.nn.functional.mse_loss,
                      additional_metadata={})
    optimizers, lr_schedulers = model.configure_optimizers()
    assert len(optimizers) == 1
    assert len(lr_schedulers) == 1
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(lr_schedulers[0]['scheduler'],
                      ReduceLROnPlateauWithWarmup)

    x = torch.rand(10, 10)
    out = model(x)
    assert out.shape == (10, 4)

    model = TestModel(optimizer_type='adam', lr_scheduler=None)
    optimizers, lr_schedulers = model.configure_optimizers()
    assert model.optimizer_type == 'adam'
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert lr_schedulers[0] is None

    model = TestModel(optimizer_type='sgd')
    optimizers, lr_schedulers = model.configure_optimizers()
    assert model.optimizer_type == 'sgd'
    assert isinstance(optimizers[0], torch.optim.SGD)


def test_evaluate_step():
    model = TestModel(learning_rate=1e-5,
                      l2_regularization_weight=1e-4,
                      lr_scheduler=ReduceLROnPlateauWithWarmup,
                      lr_scheduler_kwargs={'warmup_epochs': 10,
                                           'patience': 100,
                                           'factor': 0.8},
                      loss_function=torch.nn.functional.mse_loss,
                      additional_metadata={})

    x = torch.rand(10)
    log_y = torch.rand(4)
    data = Data(x=x, log_y=log_y)
    y_hat, loss = model.evaluate_step(data)
    assert y_hat.shape == (4, )
    assert loss.shape == ()

    x = torch.rand(1, 10)
    log_y = torch.rand(1, 4)
    batch = Batch.from_data_list([Data(x=x, log_y=log_y) for _ in range(16)])
    y_hat, loss = model.evaluate_step(batch)
    assert y_hat.shape == (16, 4)
    assert loss.shape == ()


def test_training_step():
    model = TestModel(learning_rate=1e-5,
                      l2_regularization_weight=1e-4,
                      lr_scheduler=ReduceLROnPlateauWithWarmup,
                      lr_scheduler_kwargs={'warmup_epochs': 10,
                                           'patience': 100,
                                           'factor': 0.8},
                      loss_function=torch.nn.functional.mse_loss,
                      additional_metadata={})

    x = torch.rand(1, 10)
    log_y = torch.rand(1, 4)
    batch = Batch.from_data_list([Data(x=x, log_y=log_y) for _ in range(16)])
    loss = model.training_step(batch, 0)
    assert loss.shape == ()


def test_validation_step():
    model = TestModel(learning_rate=1e-5,
                      l2_regularization_weight=1e-4,
                      lr_scheduler=ReduceLROnPlateauWithWarmup,
                      lr_scheduler_kwargs={'warmup_epochs': 10,
                                           'patience': 100,
                                           'factor': 0.8},
                      loss_function=torch.nn.functional.mse_loss,
                      additional_metadata={})

    x = torch.rand(1, 10)
    log_y = torch.rand(1, 4)
    batch = Batch.from_data_list([Data(x=x, log_y=log_y) for _ in range(16)])
    loss = model.validation_step(batch, 0)
    assert loss.shape == ()


def test_test_step():
    model = TestModel(learning_rate=1e-5,
                      l2_regularization_weight=1e-4,
                      lr_scheduler=ReduceLROnPlateauWithWarmup,
                      lr_scheduler_kwargs={'warmup_epochs': 10,
                                           'patience': 100,
                                           'factor': 0.8},
                      loss_function=torch.nn.functional.mse_loss,
                      additional_metadata={})

    x = torch.rand(1, 10)
    log_y = torch.rand(1, 4)
    batch = Batch.from_data_list([Data(x=x, log_y=log_y) for _ in range(16)])
    loss = model.test_step(batch, 0)
    assert loss.shape == ()


def test_predict_step():
    model = TestModel(learning_rate=1e-5,
                      l2_regularization_weight=1e-4,
                      lr_scheduler=ReduceLROnPlateauWithWarmup,
                      lr_scheduler_kwargs={'warmup_epochs': 10,
                                           'patience': 100,
                                           'factor': 0.8},
                      loss_function=torch.nn.functional.mse_loss,
                      additional_metadata={})

    x = torch.rand(10)
    log_y = torch.rand(4)
    data = Data(x=x, log_y=log_y)
    y_hat = model.predict_step(data)
    assert y_hat.shape == (4, )
