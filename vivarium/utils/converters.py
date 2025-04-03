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

# Decorator to access nested fields in a class
def access_nested_fields(field_map):
    def decorator(cls):
        for obj_name, field_names in field_map.items():
            for field_name in field_names:
                @property
                def prop(self, obj_name=obj_name, field_name=field_name):
                    return getattr(getattr(self, obj_name), field_name)

                @prop.setter
                def prop(self, value, obj_name=obj_name, field_name=field_name):
                    setattr(getattr(self, obj_name), field_name, value)
                setattr(cls, field_name, prop)
        return cls
    return decorator
