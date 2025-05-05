import logging as lg

from jax import jit, lax, random

from jax_md import partition, space

from vivarium.environments.physics_engine import (
    reset_force_state_fn,
    collision_state_fn,
    friction_state_fn,
    init_state_fn,
    step_state_fn
)

from vivarium.utils.converters import access_nested_fields


# Generic function to check if an entity exists, can be used in other modules
exists_mask_fn = lambda state: state.entity_state.exists == 1


class NeighborManager:
    def __init__(self, box_size, neighbor_radius, state, space_fn=space.periodic):
        self.displacement, self.shift = space_fn(box_size)
        self.neighbor_fn = partition.neighbor_list(
            self.displacement,
            box_size,
            r_cutoff=neighbor_radius,
            dr_threshold=10.0,
            capacity_multiplier=1.5,
            format=partition.Sparse,
        )
        self.box_size = box_size
        self.neighbor_radius = neighbor_radius
        self.allocate(state)
    
    def allocate(self, state):
        self.neighbors = self.neighbor_fn.allocate(state.entity_state.unified_position)
    
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


nested_fields_to_access = {
    'neighbor_manager': ['box_size', 'neighbor_radius'],
}


@access_nested_fields({'neighbor_manager': ['box_size', 'neighbor_radius']})
class Environment:
    def __init__(self, state, 
                 neighbor_manager,
                 state_fns=None, 
                 reset_fn=True, collision_fn=True, friction_fn=True, step_fn=True,
                 num_scan_steps=1, to_jit=True, key=random.PRNGKey(42)):

        self.state = state
        key, sub_key = random.split(key)
        self.init_fn = init_state_fn(sub_key)
        self.state_fns = []
        if reset_fn:
            self.state_fns += [reset_force_state_fn()]
        if state_fns is None:
            self.state_fns.extend(state.state_fns(neighbor_manager))
        else:
            self.state_fns.extend(state_fns)
        if collision_fn:
            self.state_fns += [collision_state_fn(neighbor_manager.displacement, exists_mask_fn)]
        if friction_fn:
            self.state_fns += [friction_state_fn(exists_mask_fn)]
        if step_fn:
            key, sub_key = random.split(key)
            self.state_fns += [step_state_fn(neighbor_manager.shift, exists_mask_fn, sub_key)]
        self.neighbor_manager = neighbor_manager
        self.num_scan_steps = num_scan_steps
        self.to_jit = to_jit
        if to_jit:
            self._step_env = jit(self._step_env, static_argnums=(2,))

    @classmethod
    def init_neighbor_manager(cls, state, box_size, neighbor_radius, num_scan_steps, key=random.PRNGKey(42), to_jit=True, space_fn=space.periodic):
        neighbor_manager = NeighborManager(box_size, neighbor_radius, state, space_fn)
        return cls(state, neighbor_manager, num_scan_steps=num_scan_steps, to_jit=to_jit, key=key)

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
            neighbors = self.neighbor_manager.update(state.entity_state.unified_position)
            # state = state.set(time=state.time + 1)
            carry = (state, neighbors)
            return carry, carry
        (state, neighbors), _ = lax.scan(step_fn, (state, neighbors), xs=None, length=num_scan_steps)
        return state, neighbors
        

    def step(self, state):

        if state.entity_state.momentum is None:
            state = self.init_fn(state)

        current_state = state
        neighbors = self.neighbor_manager.neighbors
        state, neighbors = self._step_env(current_state, neighbors, self.num_scan_steps)
        self.neighbor_manager.neighbors = neighbors

        self.neighbor_manager.reallocate_if_overflow(state)

        return state
