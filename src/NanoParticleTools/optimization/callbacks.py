from NanoParticleTools.util.visualization import plot_nanoparticle_from_arrays
from NanoParticleTools.machine_learning.data import FeatureProcessor

from maggma.stores import Store

from matplotlib import pyplot as plt
from collections.abc import Callable
import numpy as np
import torch

from uuid import uuid4


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
            dpi=300,
            elements=feature_processor.possible_elements
        )
        if f is not None:
            plt.text(0.1,
                     0.95,
                     f'UV Intensity={np.power(10, -f)-100:.2f}',
                     fontsize=20,
                     transform=plt.gca().transAxes)
        return plt

    return plotting_fn


def save_to_store(store: Store,
                  feature_processor: FeatureProcessor,
                  n_constraints: int,
                  metadata: dict | None = None) -> Callable:
    """
    A helper function which saves the iteration history of a scipy optimization
    run to a database.

    Args:
        store: The maggma store to which the data will be saved.
        feature_processor: The feature processor which is used to generate ML
            input graphs.
        metadata: Metadata to be added to the saved data.
    """
    if metadata is None:
        metadata = {}

    iteration_counter = 0

    def save_fn(x, f=None, accept=None):
        nonlocal iteration_counter

        if isinstance(x, np.ndarray | torch.Tensor):
            x = x.tolist()

        if isinstance(f, torch.Tensor):
            f = f.item()
        else:
            f = float(f)

        _d = dict(
            uuid=str(uuid4()),
            x=list(x),
            f=f,
            accept=accept,
            n_dopants=feature_processor.n_possible_elements,
            n_constraints=n_constraints,
            dopants=feature_processor.possible_elements,
            metadata=metadata,
            iteration_counter=iteration_counter,
        )

        with store:
            store.update(_d, key='uuid')
        iteration_counter += 1

    return save_fn
