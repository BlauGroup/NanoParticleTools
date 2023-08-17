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
    'Other': 'lightgrey',
    'Y': 'white'
}


def plot_nanoparticle_from_arrays(radii: np.array,
                                  concentrations: np.array,
                                  dpi=150,
                                  as_np_array=False,
                                  elements=['Yb', 'Er', 'Nd']):
    if 'Y' not in elements:
        elements = elements + ['Y']

    # Fill in the concentrations with Y
    concentrations_with_y = np.concatenate(
        (concentrations, 1 - concentrations.sum(axis=1, keepdims=True)),
        axis=1)

    colors = [
        DEFAULT_COLOR_MAP[el]
        if el in DEFAULT_COLOR_MAP else DEFAULT_COLOR_MAP['Other']
        for el in elements
    ]
    # cmap = plt.colormaps["tab10"]
    # colors = cmap(np.arange(4))
    # # colors[:3] = colors[1:]
    # colors[-1] = [1, 1, 1, 1]

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.subplots()

    for i in range(concentrations.shape[0], 0, -1):
        ax.pie(concentrations_with_y[i - 1],
               radius=radii[i] / radii[-1],
               colors=colors,
               wedgeprops=dict(edgecolor='k', linewidth=0.25),
               startangle=90)
    ax.legend(elements, loc='upper left', bbox_to_anchor=(0.84, 0.95))
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


def plot_nanoparticle(constraints,
                      dopant_specifications,
                      dpi=150,
                      as_np_array=False,
                      elements=['Yb', 'Er', 'Nd']):
    if 'Y' not in elements:
        elements = elements + ['Y']

    n_layers = len(constraints)
    radii = [0] + [constraint.radius for constraint in constraints]
    dopant_dict = [{key: 0 for key in elements} for _ in range(n_layers)]
    for dopant in dopant_specifications:
        dopant_dict[dopant[0]][dopant[2]] = dopant[1]

    # Fill in the rest with 'Y'
    for layer in dopant_dict:
        layer['Y'] = 1 - sum(layer.values())

    vals = [[layer[el] for el in elements] for layer in dopant_dict]

    return plot_nanoparticle_from_arrays(np.array(radii),
                                         np.array(vals),
                                         dpi=dpi,
                                         as_np_array=as_np_array,
                                         elements=elements)


def plot_nanoparticle_on_ax(ax,
                            constraints,
                            dopant_specifications,
                            elements=['Yb', 'Er', 'Nd']):
    if 'Y' not in elements:
        elements = ['Y'] + elements

    n_layers = len(constraints)
    radii = [constraint.radius for constraint in constraints]
    dopant_dict = [{key: 0 for key in elements} for _ in range(n_layers)]
    for dopant in dopant_specifications:
        dopant_dict[dopant[0]][dopant[2]] = dopant[1]
    # Fill in the rest with 'Y'
    for layer in dopant_dict:
        layer['Y'] = np.round(1 - sum(layer.values()), 3)

    vals = [[layer[el] for el in elements] for layer in dopant_dict]
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.arange(4) * 4)
    colors[0] = [1, 1, 1, 1]

    for i in list(range(n_layers - 1, -1, -1)):
        # print(vals[i])
        ax.pie(vals[i],
               radius=radii[i] / radii[-1],
               colors=colors,
               wedgeprops=dict(edgecolor='k'),
               startangle=90)
    ax.legend(elements, loc='upper left', bbox_to_anchor=(1, 1))


def update(data, ax):
    constraints, dopants = data
    ax.clear()
    plot_nanoparticle_on_ax(ax, constraints, dopants)


def make_animation(frames: List[Tuple[NanoParticleConstraint, Tuple]],
                   name: str = 'animation.mp4',
                   fps: int = 30) -> None:

    fig = plt.figure(dpi=150)
    ax = fig.subplots()
    anim = animation.FuncAnimation(fig, partial(update, ax=ax), frames=frames)
    anim.save(name, fps=fps)
    fig.clear()
