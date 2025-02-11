from functools import partial
from jax import jit
import jax.numpy as jnp
from vivarium.environments.braitenberg.point_particle.init import init_state
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import SelectiveSensorsEnv
from vivarium.environments.braitenberg.simple.simple_env import motor_command


class PointParticleEnv(SelectiveSensorsEnv):
    @partial(jit, static_argnums=(0,))
    def _step_env(
        self, state, neighbors_storage
    ):
        state, neighbors_storage = super()._step_env(state, neighbors_storage)
        
        agent_idx = state.agents.ent_idx
        _, rot = motor_command(
            state.agents.motor,
            state.entities.diameter[agent_idx],
            state.agents.wheel_diameter,
        )

        orientation = (
            jnp.zeros_like(state.entities.unified_orientation)
            .at[agent_idx]
            .set(state.entities.unified_orientation[agent_idx] + state.dt * rot)
        )
        # orientation = jnp.where(exists_mask, orientation, 0.0)

        state = state.set(
            entities=state.entities.set(
                orientation=orientation
            )
        )
        
        return state, neighbors_storage

if __name__ == "__main__":
    state = init_state()
    env = PointParticleEnv(state)

    env.step(state, num_updates=5)
    env.step(state, num_updates=6)
