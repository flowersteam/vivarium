from jax_md.dataclasses import is_dataclass


def is_jax_md_dataclass(instance):
    """Check if an instance is a jax md dataclass

    :param instance: instance to check
    :return: True if instance is a jax md dataclass, False otherwise
    """
    return is_dataclass(instance) and hasattr(instance, 'set') and callable(getattr(instance, 'set'))

