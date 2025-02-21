import logging as lg

from functools import partial

from jax import jit, lax
import jax.numpy as jnp
from jax_md.rigid_body import RigidBody
from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import simulate, partition


@md_dataclass
class BaseState:
    time: jnp.int32
    box_size: jnp.int32


@md_dataclass
class BaseEntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_idx: jnp.array
    exists: jnp.array
    previous_force: jnp.array

    def is_rigid_body(self):
        return hasattr(self.position, 'center')

    def __getattr__(self, name):
        prefix, suffix = name.split('_', 1)
        if prefix == "unified":
            if suffix == 'orientation':
                if isinstance(self.position, RigidBody):
                    return self.position.orientation
                return self.orientation
            if isinstance(getattr(self, suffix), RigidBody):
                return getattr(self, suffix).center
            return getattr(self, suffix)
        if suffix in ['center', 'orientation']:
            if isinstance(self.position, RigidBody):
                return getattr(getattr(self, prefix), suffix)
            # return RigidBody(center=getattr(self, prefix), 
            #                  orientation=self.orientation if prefix == 'position' else jnp.zeros_like(self.orientation))
            if suffix == 'center':
                return getattr(self, prefix)
            else:  # Necessarily 'orientation'
                return self.orientation if prefix == 'position' else jnp.zeros_like(self.orientation)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


@md_dataclass
class BaseParticleState:
    ent_idx: jnp.array


class NeighborManager:
    def __init__(self, displacement, state):
        self.neighbor_fn = partition.neighbor_list(
            displacement,
            state.box_size,
            r_cutoff=state.neighbor_radius,
            dr_threshold=10.0,
            capacity_multiplier=1.5,
            format=partition.Sparse,
        )
        self.displacement = displacement
        self.allocate(state)
    
    def allocate(self, state):
        self.neighbors = self.neighbor_fn.allocate(state.entities.unified_position)
    
    def update(self, position):
        self.neighbors = self.neighbors.update(position)
        return self.neighbors
    
    def reallocate_if_overflow(self, state):
        if self.neighbors.did_buffer_overflow:
            # reallocate neighbors and run the simulation from current_state
            lg.warning(
                f"NEIGHBORS BUFFER OVERFLOW: rebuilding neighbors"
            )
            self.allocate(state)
            assert not self.neighbors.did_buffer_overflow
    

class BaseEnv:
    def __init__(self, state, 
                 init_fn, state_fns, 
                 neighbors_manager):
        self.state = state
        self.init_fn = init_fn
        self.state_fns = state_fns
        self.neighbors_manager = neighbors_manager

    def init_state(self) -> BaseState:
        raise (NotImplementedError)

    @partial(jit, static_argnums=(0,3))
    def _step_env(
        self, state, neighbors, num_scan_steps=1
    ):
        def step_fn(carry, _):
            """Apply a step function to return new state and neighbors in a jax.lax.scan update

            :param carry: tuple of (state, neighbors)
            :param _: dummy xs for jax.lax.scan
            :return: tuple of (carry, carry) with carry=(new_state, new_neighbors)
            """
            state, neighbors = carry
            for fn in self.state_fns:
                state = fn(state, neighbors) 
            neighbors = self.neighbors_manager.update(state.entities.unified_position)
            state = state.set(time=state.time + 1)
            carry = (state, neighbors)
            return carry, carry
        (state, neighbors), _ = lax.scan(step_fn, (state, neighbors), xs=None, length=num_scan_steps)
        return state, neighbors
        

    def step(self, state: BaseState, num_scan_steps=1) -> BaseState:

        if state.entities.momentum is None:
            state = self.init_fn(state)

        current_state = state
        neighbors = self.neighbors_manager.neighbors
        state, neighbors = self._step_env(current_state, neighbors, num_scan_steps)
        self.neighbors_manager.neighbors = neighbors

        self.neighbors_manager.reallocate_if_overflow(state)

        return state
