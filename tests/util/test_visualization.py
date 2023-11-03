from NanoParticleTools.util.visualization import (
    plot_nanoparticle_from_arrays, plot_nanoparticle, plot_nanoparticle_on_ax,
    make_animation)
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint

from matplotlib import pyplot as plt
import numpy as np


def test_plot_nanoparticle_from_arrays():
    """
    Here we are just checking that the function runs without
    throwing an error.

    This is a plotting function, therefore it is hard to check output.
    """
    radii = np.array([0, 20, 50])
    concentration = np.array([[0.25, 0.15, 0], [0.52, 0, 0.45]])

    plot_nanoparticle_from_arrays(radii, concentration)

    out = plot_nanoparticle_from_arrays(radii, concentration, as_np_array=True)
    assert isinstance(out, np.ndarray)


def test_plot_nanoparticle():
    """
    Here we are just checking that the function runs without
    throwing an error.

    This is a plotting function, therefore it is hard to check output.
    """
    constraints = [SphericalConstraint(50), SphericalConstraint(80)]
    dopant_specifications = [(0, 0.25, 'Yb', 'Y'), (0, 0.15, 'Nd', 'Y'),
                             (1, 0.52, 'Yb', 'Y'), (1, 0.45, 'Er', 'Y')]

    plot_nanoparticle(constraints, dopant_specifications)


def test_plot_nanoparticle_on_ax():
    """
    Here we are just checking that the function runs without
    throwing an error.

    This is a plotting function, therefore it is hard to check output.
    """
    constraints = [SphericalConstraint(50), SphericalConstraint(80)]
    dopant_specifications = [(0, 0.25, 'Yb', 'Y'), (0, 0.15, 'Nd', 'Y'),
                             (1, 0.52, 'Yb', 'Y'), (1, 0.45, 'Er', 'Y')]

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(121)
    plot_nanoparticle_on_ax(ax, constraints, dopant_specifications)


def test_make_animation():
    """
    Here we are just checking that the function runs without
    throwing an error.

    This is a plotting function, therefore it is hard to check output.
    """
    constraints = [[SphericalConstraint(50),
                    SphericalConstraint(80)],
                   [SphericalConstraint(50),
                    SphericalConstraint(80)]]
    dopant_specifications = [[(0, 0.25, 'Yb', 'Y'), (0, 0.15, 'Nd', 'Y'),
                              (1, 0.52, 'Yb', 'Y'), (1, 0.45, 'Er', 'Y')],
                             [(0, 0.25, 'Yb', 'Y'), (0, 0.15, 'Nd', 'Y'),
                              (1, 0.52, 'Yb', 'Y'), (1, 0.35, 'Er', 'Y')]]
    frames = list(zip(constraints, dopant_specifications))
    make_animation(frames, name='animation.gif')
