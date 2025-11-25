import logging

from omegaconf import DictConfig, OmegaConf
import hydra

from vivarium.simulator import Simulator
from vivarium.simulator.grpc_server.simulator_server import serve


lg = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    
    logging.basicConfig(level=cfg.log_level)

    lg.info(f"Scene loading: {cfg.scene.scene_name}")

    # Create the simulator
    simulator = Simulator.from_config(cfg.scene.simulator)

    # start and host the simulator on a server
    serve(simulator)
    lg.info("Simulator server started")


if __name__ == "__main__":
    main()
