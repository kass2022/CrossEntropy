import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import numpy as np

def AddColorbar(image, aspect=14, pad_fraction=1, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(image.axes)
    width = axes_grid1.axes_size.AxesY(image.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return image.axes.figure.colorbar(image, cax=cax, **kwargs)

def Imshow(data, title, cmap='bwr'):
    norm = matplotlib.colors.Normalize(vmin=-np.max(abs(data)), vmax=np.max(abs(data)))
    subfig = plt.imshow(data.T,cmap = cmap, origin='lower',norm = norm)
    AddColorbar(subfig)
    plt.title(title)