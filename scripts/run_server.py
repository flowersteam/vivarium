import logging

from omegaconf import DictConfig, OmegaConf
import hydra

from vivarium.utils.scene_configs import SceneConfiguration
from vivarium.simulator.grpc_server.simulator_server import serve


lg = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    
    print(OmegaConf.to_yaml(cfg))
    
    logging.basicConfig(level=cfg.log_level)

    lg.info(f"Scene running: {cfg.scene.scene_name}")

    # Create the simulator
    simulator = SceneConfiguration(cfg.scene).create_simulator()

    # start and host the simulator on a server
    serve(simulator)
    lg.info("Simulator server started")


if __name__ == "__main__":
    main()
