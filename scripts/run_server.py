import logging
import os
import sys

from hydra import initialize_config_dir, compose

from vivarium.simulator import Simulator
from vivarium.simulator.grpc_server.simulator_server import serve

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

lg = logging.getLogger(__name__)


def main() -> None:
    # Determine config directory based on whether we're frozen or not
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        config_dir = os.path.join(sys._MEIPASS, 'conf')
    else:
        # Running in development
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../conf"))

    # Initialize Hydra with the correct config directory
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # Parse command-line arguments as Hydra overrides
        # sys.argv[1:] contains arguments like ["scene=session_1"]
        overrides = sys.argv[1:]

        # Compose config with any command-line overrides
        cfg = compose(config_name="config", overrides=overrides)

        logging.basicConfig(level=cfg.log_level)

        lg.info(f"Scene loading: {cfg.scene.scene_name}")

        # Create the simulator
        simulator = Simulator.from_config(cfg.scene.simulator)

        # start and host the simulator on a server
        serve(simulator)
        lg.info("Simulator server started")


if __name__ == "__main__":
    main()
