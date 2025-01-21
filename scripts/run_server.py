import logging
import hydra

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from vivarium.simulator.simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    SelectiveSensorsEnv,
    init_state,
)
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)

# Define parameters of the simulator
UPDATE_FREQ = 60
NUM_STEPS_LAX = 6


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    # retrieve args from config
    hydra_cfg = HydraConfig.get()
    scene_name = OmegaConf.to_container(hydra_cfg.runtime.choices)["scene"]
    lg.info(f"Scene running: {scene_name}")
    scene_config = OmegaConf.merge(cfg.default, cfg.scene)

    # init state and environment
    state = init_state(**scene_config)
    env = SelectiveSensorsEnv(state=state)

    # init the simulator
    simulator = Simulator(
        env_state=state,
        env=env,
        scene_name=scene_name,
        update_freq=UPDATE_FREQ,
        num_steps_lax=NUM_STEPS_LAX,
    )

    # start and host the simulator on a server
    serve(simulator)
    lg.info("Simulator server started")


if __name__ == "__main__":
    main()
