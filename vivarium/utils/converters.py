import importlib
import re
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


def import_class(class_path):
    if not isinstance(class_path, str):
        return class_path
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def class_import_path(cls) -> str:
    """Get the import path of a given class

    :param cls: The class to get the import path for
    :return: The import path as a string
    """
    return f"{cls.__module__}.{cls.__name__}"


def upper_camel_to_snake(name: str) -> str:
    """Convert an upper camel case string to snake case

    :param name: The string in upper camel case
    :return: The string in snake case
    """
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def snake_to_upper_camel(name: str) -> str:
    """Convert a snake case string to upper camel case

    :param name: The string in snake case
    :return: The string in upper camel case
    """
    return ''.join(word.capitalize() for word in name.split('_'))
