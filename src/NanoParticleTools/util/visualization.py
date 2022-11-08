import matplotlib.pyplot as plt
import numpy as np

def plot_nanoparticle(constraints, dopant_specifications, dpi = 150, as_np_array=False):
    els = ['Y', 'Yb', 'Er', 'Nd']

    n_layers = len(constraints)
    radii = [constraint.radius for constraint in constraints]
    dopant_dict = [{key:0 for key in els} for _ in range(n_layers)]
    for dopant in dopant_specifications:
        dopant_dict[dopant[0]][dopant[2]] = dopant[1]

    # Fill in the rest with 'Y'
    for layer in dopant_dict:
        layer['Y'] = 1-sum(layer.values())

    vals = [[layer[el] for el in els]for layer in dopant_dict]
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.arange(4)*4)
    colors[0] = [1, 1, 1, 1]

    fig = plt.figure(dpi=dpi)
    ax = fig.subplots()
    for i in list(range(n_layers-1, -1, -1)):
        ax.pie(vals[i], 
               radius = radii[i]/radii[-1], 
               colors= colors, 
               wedgeprops=dict(edgecolor='k'),
               startangle=90
              )
    ax.legend(els, loc='upper left', bbox_to_anchor=(1, 1)
    )
    if as_np_array:
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first.
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Close the figure to remove it from the buffer
        plt.close(fig)
        return data
    else:
        return fig