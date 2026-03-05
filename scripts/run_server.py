import os
import sys
import logging

from hydra import initialize_config_dir, compose

from vivarium.simulator import Simulator
from vivarium.simulator.grpc_server.simulator_server import serve
from vivarium.utils.runtime import get_config_dir, initialize_user_data, is_frozen

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

lg = logging.getLogger(__name__)


def main() -> None:
    # Initialize user data directories on first run (frozen mode only)
    if is_frozen() and initialize_user_data():
        lg.info("First run initialization complete")

    config_dir = get_config_dir()

    # Initialize Hydra with the correct config directory
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # Parse command-line arguments as Hydra overrides
        # sys.argv[1:] contains arguments like ["scene=session_1"]
        overrides = sys.argv[1:]

        # Compose config with any command-line overrides
        cfg = compose(config_name="config", overrides=overrides)

        logging.basicConfig(level=cfg.log_level)

        # Log the actual config file being used
        scene_config_path = os.path.join(config_dir, 'scene', f"{cfg.scene.scene_name}.yaml")
        lg.info(f"Loading scene config: {scene_config_path}")

        # Create the simulator
        simulator = Simulator.from_config(cfg.scene.simulator)

        # start and host the simulator on a server
        serve(simulator)
        lg.info("Simulator server started")


if __name__ == "__main__":
    main()
