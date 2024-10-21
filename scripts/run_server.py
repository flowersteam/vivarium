import logging
import hydra

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from vivarium.simulator.simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing import SelectiveSensorsEnv, init_state
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)

# Define parameters of the simulator
update_freq = 60
num_steps_lax = 6

def start_simulator(scene_config: DictConfig) -> None:
    # init state and env
    state = init_state(**scene_config)
    env = SelectiveSensorsEnv(state=state)

    # init simulator
    simulator = Simulator(
        env_state=state, 
        env=env, 
        update_freq=update_freq, 
        num_steps_lax=num_steps_lax
    )

    # host the simulation on a server
    lg.info('Simulator server started')
    serve(simulator)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    # retrieve args from config
    hydra_cfg = HydraConfig.get()
    lg.info(f"Scene running: {OmegaConf.to_container(hydra_cfg.runtime.choices)['scene']}")
    args = OmegaConf.merge(cfg.default, cfg.scene)
    # start the simulator
    start_simulator(args)

if __name__ == '__main__':
    main()
