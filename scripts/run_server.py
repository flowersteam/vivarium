import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from vivarium.simulator.grpc_server.simulator_server import serve
from vivarium.utils.scene_configs import SceneConfiguration


lg = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    # retrieve args from config
    hydra_cfg = HydraConfig.get()
    scene_name = OmegaConf.to_container(hydra_cfg.runtime.choices)["scene"]
    lg.info(f"Scene running: {scene_name}")

    # Create the simulator
    simulator = SceneConfiguration(scene_name).create_simulator()

    # start and host the simulator on a server
    serve(simulator)
    lg.info("Simulator server started")


if __name__ == "__main__":
    main()
