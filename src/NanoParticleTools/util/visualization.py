import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from functools import partial
from typing import List, Tuple
from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint
from pymatgen.vis.structure_vtk import EL_COLORS

DEFAULT_COLOR_MAP = {
    'Yb': 'tab:blue',
    'Er': 'tab:orange',
    'Nd': 'tab:green',
    'Tm': 'tab:red',
    'Other': 'lightgrey',
    'Y': 'white'
}


def plot_nanoparticle(radii: np.ndarray | list[NanoParticleConstraint],
                      concentrations: np.array = None,
                      dopant_specifications: list[tuple] = None,
                      dpi=150,
                      as_np_array=False,
                      elements=['Yb', 'Er', 'Nd'],
                      ax: plt.Axes = None,
                      emissions: float = None):
    if 'Y' not in elements:
        # Add Y, the host element
        elements = elements + ['Y']

    if isinstance(radii[0], NanoParticleConstraint):
        # Convert this to an array
        radii = np.array([0] + [c.radius for c in radii])
    if not isinstance(radii, np.ndarray):
        # If it is a list, it is already in the format we require
        raise TypeError(
            'radii should be an array of radii or list of contraints')

    if concentrations is None and dopant_specifications is None:
        raise RuntimeError(
            'Must specify one of concentrations or dopant specifications')
    elif dopant_specifications is not None:
        # convert this to an array
        n_layers = len(radii) - 1
        dopant_dict = [{key: 0 for key in elements} for _ in range(n_layers)]
        for dopant in dopant_specifications:
            dopant_dict[dopant[0]][dopant[2]] = dopant[1]

        # Fill in the rest with 'Y'
        for layer in dopant_dict:
            layer['Y'] = 1 - sum(layer.values())

        vals = [[layer[el] for el in elements] for layer in dopant_dict]
        concentrations = np.array(vals)
    elif concentrations is not None:
        # Add Y into the list
        if len(elements) != concentrations.shape[1]:
            concentrations = np.concatenate(
                (concentrations,
                 1 - concentrations.sum(axis=1, keepdims=True)),
                axis=1)

    concentrations = np.clip(concentrations, 0, 1)
    colors = [
        DEFAULT_COLOR_MAP[el]
        if el in DEFAULT_COLOR_MAP else DEFAULT_COLOR_MAP['Other']
        for el in elements
    ]

    if ax is None:
        # make a new axis
        fig = plt.figure(figsize=(5, 5), dpi=dpi)
        ax = fig.subplots()

        for i in range(concentrations.shape[0], 0, -1):
            ax.pie(concentrations[i - 1],
                   radius=radii[i] / radii[-1],
                   colors=colors,
                   wedgeprops=dict(edgecolor='w', linewidth=0.25),
                   startangle=90)
        ax.legend(elements, loc='upper left', bbox_to_anchor=(0.84, 0.95))
        if emissions:
            plt.text(0.1,
                     0.95,
                     f'UV Intensity={np.power(10, -emissions)-100:.2f}',
                     fontsize=20,
                     transform=plt.gca().transAxes)
        plt.tight_layout()
        if as_np_array:
            # If we haven't already shown or saved the plot, then we need to
            # draw the figure first.
            fig.canvas.draw()

            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

            # Close the figure to remove it from the buffer
            plt.close(fig)
            return data
        else:
            return fig
    else:
        for i in range(concentrations.shape[0], 0, -1):
            ax.pie(concentrations[i - 1],
                   radius=radii[i] / radii[-1],
                   colors=colors,
                   wedgeprops=dict(edgecolor='w', linewidth=0.25),
                   startangle=90)
        ax.legend(elements, loc='upper left', bbox_to_anchor=(0.84, 0.95))
        if emissions:
            plt.text(0.1,
                     0.95,
                     f'UV Intensity={np.power(10, -emissions)-100:.2f}',
                     fontsize=20,
                     transform=plt.gca().transAxes)


def update(data, ax):
    ax.clear()
    plot_nanoparticle(ax=ax, **data)


def make_animation(frames: List[Tuple[NanoParticleConstraint, Tuple]],
                   name: str = 'animation.mp4',
                   fps: int = 30,
                   dpi: int = 300) -> None:

    fig = plt.figure(dpi=dpi)
    ax = fig.subplots()
    anim = animation.FuncAnimation(fig, partial(update, ax=ax), frames=frames)
    anim.save(name, fps=fps)
    fig.clear()
