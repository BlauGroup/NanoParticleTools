from NanoParticleTools.machine_learning.util.learning_rate import (
    WarmupSequentialLR, ReduceLROnPlateauWithWarmup)
import pytest
import torch


@pytest.fixture
def model():
    return torch.nn.Linear(10, 4)


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def test_SequentialLR(optimizer):
    lr_scheduler = WarmupSequentialLR(optimizer)
    assert lr_scheduler is not None
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_ReduceLROnPlateauWithWarmup(optimizer):
    lr_scheduler = ReduceLROnPlateauWithWarmup(3,
                                               patience=5,
                                               factor=0.1,
                                               optimizer=optimizer)
    print(lr_scheduler.patience)
    expected_lr = [
        0.0001, 0.0002, 0.00030000000000000003, 0.00030000000000000003,
        0.00030000000000000003, 0.00030000000000000003, 0.00030000000000000003,
        0.00030000000000000003, 3.0000000000000004e-05, 3.0000000000000004e-05,
        3.0000000000000004e-05, 3.0000000000000004e-05, 3.0000000000000004e-05,
        3.0000000000000004e-05, 3.0000000000000005e-06, 3.0000000000000005e-06
    ]

    lr = []
    for _ in range(16):
        lr.extend(lr_scheduler._last_lr)
        lr_scheduler.step(10)

    assert lr == pytest.approx(expected_lr)

    # Test that epoch can be explicitly specified
    # Although, this is deprecated in pytorch
    with pytest.warns(UserWarning):
        lr_scheduler.step(10, epoch=10)
