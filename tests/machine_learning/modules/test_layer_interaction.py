from NanoParticleTools.machine_learning.modules.layer_interaction import (
    InteractionConv, InteractionBlock, integrated_gaussian_interaction,
    integrated_gaussian_interaction_tanh)

import torch


def test_interaction_block():
    ri0 = torch.tensor([0, 0, 10, 10]).float()
    rif = torch.tensor([10, 10, 20, 20]).float()
    rj0 = torch.tensor([0, 10, 10, 0]).float()
    rjf = torch.tensor([10, 20, 20, 10]).float()
    interaction_block = InteractionBlock(nsigma=3)

    out = interaction_block(ri0, rif, rj0, rjf)
    expected_out = torch.tensor([[1.2905e+07, 1.8523e+08, 3.4935e+08],
                                 [3.0760e+07, 1.2517e+09, 2.4119e+09],
                                 [1.6136e+08, 8.4677e+09, 1.6704e+10],
                                 [3.0760e+07, 1.2517e+09, 2.4119e+09]])
    assert torch.allclose(out, expected_out, rtol=1e-2)


def test_interaction_conv():
    interaction_conv = InteractionConv(3, 4)

    pass
    # x = torch.rand(2, 3)
    # compositions = torch.tensor([0.25, 0.1]).unsqueeze(1)
    # edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    # edge_attr = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20],
    #                           [0, 10, 10, 0], [10, 20, 20, 10]]).float()
    # out = interaction_conv(x, compositions, edge_index, edge_attr)
    # print(out)
    # assert out.shape == (2, 4)


def test_integrated_gaussian_interaction_tanh():
    ri0 = torch.tensor([0, 0, 10, 10]).float().unsqueeze(1)
    rif = torch.tensor([10, 10, 20, 20]).float().unsqueeze(1)
    rj0 = torch.tensor([0, 10, 10, 0]).float().unsqueeze(1)
    rjf = torch.tensor([10, 20, 20, 10]).float().unsqueeze(1)
    sigma = torch.tensor([10.3, 16, 100]).unsqueeze(0)

    out = integrated_gaussian_interaction_tanh(ri0, rif, rj0, rjf, sigma)
    expected_out = torch.tensor([[1.1669e+07, -2.4482e+07, -2.3382e+10],
                                 [1.0076e+08, 1.7178e+08, -9.0918e+10],
                                 [2.9631e+08, 1.0277e+09, -1.4463e+11],
                                 [1.0076e+08, 1.7178e+08, -9.0918e+10]])
    assert torch.allclose(out, expected_out, rtol=1e-2)


def test_integrated_gaussian_interaction():
    ri0 = torch.tensor([0, 0, 10, 10]).float().unsqueeze(1)
    rif = torch.tensor([10, 10, 20, 20]).float().unsqueeze(1)
    rj0 = torch.tensor([0, 10, 10, 0]).float().unsqueeze(1)
    rjf = torch.tensor([10, 20, 20, 10]).float().unsqueeze(1)
    sigma = torch.tensor([10.3, 16, 100]).unsqueeze(0)

    out = integrated_gaussian_interaction(ri0, rif, rj0, rjf, sigma)
    expected_out = torch.tensor([[2.1807e+07, 4.4805e+07, 3.4935e+08],
                                 [7.1898e+07, 2.1867e+08, 2.4119e+09],
                                 [3.5350e+08, 1.1697e+09, 1.6704e+10],
                                 [7.1898e+07, 2.1867e+08, 2.4119e+09]])
    assert torch.allclose(out, expected_out, rtol=1e-2)
