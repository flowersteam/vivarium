# import jax.numpy as jnp
import numpy as np
import matplotlib.colors as mcolors


# Helper function to transform a color string into rgb with matplotlib colors
def string_to_rgb_array(color_str):
    '''
    Convert a color string to a numpy array of RGB values.
    '''
    return np.array(list(mcolors.to_rgb(color_str)))


def rgb_array_to_string(rgb_array):
    '''
    Convert an RGB array to a color string.
    '''
    return mcolors.to_hex(np.array(rgb_array))

