import jax
import jax.numpy as jnp
from jax_md import rigid_body, simulate

from vivarium.environment.components.component import Component
from vivarium.environment.components.utils import SPACE_NDIMS


def mask_momentum(entity_state, exists_mask):
    """
    Set the momentum values to zeros for non existing entities
    :param entity_state: entity_state
    :param exists_mask: bool array specifying which entities exist or not
    :return: entity_state: new entities state state with masked momentum values
    """

    exists_mask_space = jnp.stack([exists_mask] * SPACE_NDIMS, axis=1)
    momentum = jnp.where(exists_mask_space, entity_state.unified_momentum, 0)
    if entity_state.is_rigid_body():
        orientation = jnp.where(exists_mask, entity_state.momentum.orientation, 0)
        momentum = rigid_body.RigidBody(center=momentum, orientation=orientation)
    return entity_state.set(momentum=momentum)


def init_state_fn(key, kT=0.0):
    key_cpy = key
    def fn(state):
        assert state.entity_state.momentum is None
        key, new_key = jax.random.split(key_cpy)
        assert not jnp.any(state.entity_state.unified_force)
        if state.entity_state.is_rigid_body():
            assert not jnp.any(state.entity_state.force.orientation)
        return state.set(entity_state=simulate.initialize_momenta(state.entity_state, new_key, kT))

    return fn


class StepComponent(Component):
    def __init__(self, name, precedence, dt, mask_fn):
        super().__init__(name, precedence)
        self.dt = dt
        self.mask_fn = mask_fn

    def to_config(self, state):
        config = super().to_config(state)
        config.update({
            'dt': state.dt.item() if isinstance(state.dt, jnp.ndarray) else state.dt,
            'mask_fn': self.mask_fn.to_config(state)
        })
        return config

    def update_state_cls(self, state_cls):
        state_cls.__annotations__['dt'] = jnp.float32
        state_cls.dt = None
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        if state.entity_state.momentum is None:
            key, sub_key = jax.random.split(key)
            state = init_state_fn(sub_key)(state)
        return state.set(dt=self.dt)

    def get_step_function(self, state, neighbor_manager, key):
        self.shift = neighbor_manager.shift
        def state_fn(state, neighbor, key):
            mask = self.mask_fn(state)

            dt_2 = state.dt / 2.0

            # Compute changes on entities
            new_force = state.entity_state.force
            entity_state=state.entity_state.set(force=state.entity_state.previous_force)
            entity_state = simulate.momentum_step(entity_state, dt_2)
            # TODO : why do we used dt and not dt/2 in the line below ?
            entity_state = simulate.position_step(
                entity_state, self.shift, dt_2, neighbor=neighbor
            )
            entity_state = entity_state.set(force=new_force)
            entity_state = entity_state.set(previous_force=new_force)
            entity_state = simulate.momentum_step(entity_state, dt_2)
            entity_state = mask_momentum(entity_state, mask)
            return state.set(entity_state=entity_state)
        return state_fn
