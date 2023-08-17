from NanoParticleTools.util.visualization import plot_nanoparticle_from_arrays
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
from NanoParticleTools.machine_learning.data import FeatureProcessor

from torch_geometric.data import HeteroData
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl

from collections.abc import Callable


def get_plotting_fn(feature_processor: FeatureProcessor) -> Callable:
    n_elements = len(feature_processor.possible_elements)

    def plotting_fn(x, f=None, accept=None):
        # Trim #'s less than 0 so they don't cause issues in the plotting
        x = x.clip(0)

        plt.figure()
        n_constraints = len(x) // (n_elements + 1)
        plot_nanoparticle_from_arrays(
            np.concatenate(([0], x[-n_constraints:])),
            x[:-n_constraints].reshape(n_constraints, -1),
            dpi=80,
            elements=feature_processor.possible_elements,
        )
        if f is not None:
            plt.text(0.1,
                     0.95,
                     f'UV Intensity={np.power(10, -f)-100:.2f}',
                     fontsize=20,
                     transform=plt.gca().transAxes)
        plt.show()

    return plotting_fn


def get_bounds(n_constraints: int, n_elements: int,
               r_max: int | float) -> Bounds:
    r"""
    Get the Bounds which are utilized by scipy minimize.

    The bounds specified will ensure that all concentrations
    are :math:`0 \le{} x_i \le 1`. Additionally, the radii are
    constrained to be :math:`0 \le r_i \le r_{max}`.

    Args:
        n_constraints: The number of control volumes
            in the nanoparticle.
        n_elements: The number of possible dopants
            to the nanoparticle
        r_max: The maximum radius the nanoparticle can reach during the
            optimization

    """
    num_dopant_nodes = n_constraints * n_elements
    min_bounds = np.concatenate(
        (np.zeros(num_dopant_nodes), np.zeros(n_constraints)))
    max_bounds = np.concatenate(
        (np.ones(num_dopant_nodes), r_max * np.ones(n_constraints)))
    bounds = Bounds(min_bounds, max_bounds)
    return bounds


def get_linear_constraints(
        n_constraints: int,
        n_elements: int,
        min_thickness: int | float = 5,
        max_thickness: int | float = 50,
        min_core_size: int | float = 10,
        max_core_size: int | float = 50) -> LinearConstraint:
    """
    Get the linear constraints which are utilized by scipy minimize.

    Args:
        n_constraints: The number of control volumes in the nanoparticle
        n_elements: The number of possible dopants to the nanoparticle
        min_thickness: The minimum thickness of each layer/control volume
        max_thickness: The maximum thickness of each layer/control volume
        min_core_size: The minimum core size of the nanoparticle
        max_core_size: The maximum core size of the nanoparticle
    """
    num_dopant_nodes = n_constraints * n_elements
    lower_constraint = []
    upper_constraint = []
    constraint_matrix = []
    for i in range(n_constraints):
        _constraint = np.zeros(n_elements * n_constraints + n_constraints)
        _constraint[i * n_elements:(i + 1) * n_elements] = 1
        constraint_matrix.append(_constraint)

        lower_constraint.append(0)
        upper_constraint.append(1)

    # constrain the core size
    _constraint = np.zeros(num_dopant_nodes + n_constraints)
    _constraint[-n_constraints] = 1
    constraint_matrix.append(_constraint)
    lower_constraint.append(min_core_size)
    upper_constraint.append(max_core_size)

    # Constraints on the layer thicknesses
    for i in range(n_constraints - 1):
        _constraint = np.zeros(num_dopant_nodes + n_constraints)
        _constraint[n_constraints * n_elements + i] = -1
        _constraint[n_constraints * n_elements + i + 1] = 1

        constraint_matrix.append(_constraint)

        lower_constraint.append(min_thickness)
        upper_constraint.append(max_thickness)

    linear_constraint = LinearConstraint(constraint_matrix, lower_constraint,
                                         upper_constraint)
    return linear_constraint


def x_to_data(inputs: torch.Tensor,
              feature_processor: FeatureProcessor) -> HeteroData:
    n_elements = len(feature_processor.possible_elements)
    n_constraints = len(inputs) // (n_elements + 1)

    # unpack the inputs
    x = inputs[:n_elements * n_constraints]
    r = torch.tensor(inputs[n_elements * n_constraints:],
                     dtype=torch.float32,
                     requires_grad=True)

    dopant_concentration = [{
        i: k
        for i, k in zip(feature_processor.possible_elements, layer)
    } for layer in x.reshape((-1, n_elements))]

    _data_dict = feature_processor.graph_from_inputs(dopant_concentration, r)
    data = feature_processor.data_cls(_data_dict)
    return data


def get_query_fn(model: pl.LightningModule,
                 feature_processor: FeatureProcessor,
                 return_stats: bool = False) -> Callable:

    def model_fn(inputs):
        data = x_to_data(inputs, feature_processor)
        if return_stats:
            return model.predict_step(data, return_stats=return_stats)
        else:
            # We return the negative of the prediction because we
            # want to maximize the objective function
            return -model.predict_step(data,
                                       return_stats=return_stats).detach()

    return model_fn


def get_jac_fn(model: pl.LightningModule,
               feature_processor: FeatureProcessor) -> Callable:

    def jac_fn(inputs):
        data = x_to_data(inputs, feature_processor)
        data['dopant'].x.requires_grad = True
        data['radii_without_zero'].requires_grad = True

        # We use the negative of the prediction, since we want to maximize
        y_hat = -model.predict_step(data)
        y_hat.backward()
        return np.concatenate(
            (data['dopant'].x.grad.flatten().detach().numpy(),
             data['radii_without_zero'].grad.flatten().detach().numpy()))

    return jac_fn


def rand_np(n_constraints: int,
            feature_processor: FeatureProcessor,
            r_max: int | float = 50):
    x = np.random.rand(n_constraints, len(feature_processor.possible_elements))
    x_scale = np.random.rand(n_constraints, 1)
    concs = x / x.sum(axis=1, keepdims=True) * x_scale
    # minimum of 10 A radius
    radii = 10 + (r_max -
                  10) / n_constraints * np.random.rand(n_constraints).cumsum()
    constraints = [SphericalConstraint(r) for r in radii]
    dopant_concentration = [{
        el: layer[i]
        for i, el in enumerate(feature_processor.possible_elements)
    } for layer in concs]
    return constraints, dopant_concentration
