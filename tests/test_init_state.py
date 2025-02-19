from omegaconf import OmegaConf
from hydra import initialize, compose

from vivarium.environments.braitenberg.selective_sensing import init_state


def test_init_default_state():
    """Test the initialization of state without arguments"""
    state = init_state()

    assert state


def test_init_default_state_config():
    """Test the initialization of state with default config arguments"""

    def load_default_config():
        with initialize(config_path="../conf", version_base=None):
            # Load the config (you can specify the config name here or leave it default)
            cfg = compose(config_name="config")
            args = OmegaConf.merge(cfg.default, cfg.scene)

        return args

    config = load_default_config()
    state = init_state(**config)

    assert state
