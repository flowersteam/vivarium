import logging as lg

from jax import jit, lax

from jax_md import partition

from vivarium.utils.converters import access_nested_fields


class NeighborManager:
    def __init__(self, displacement, box_size, neighbor_radius, state):
        self.neighbor_fn = partition.neighbor_list(
            displacement,
            box_size,
            r_cutoff=neighbor_radius,
            dr_threshold=10.0,
            capacity_multiplier=1.5,
            format=partition.Sparse,
        )
        self.box_size = box_size
        self.neighbor_radius = neighbor_radius
        self.displacement = displacement
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
class BaseEnv:
    def __init__(self, state, 
                 init_fn, state_fns, 
                 neighbor_manager, num_scan_steps=1, to_jit=True):

        self.state = state
        self.init_fn = init_fn
        self.state_fns = state_fns
        self.neighbor_manager = neighbor_manager
        self.num_scan_steps = num_scan_steps
        self.to_jit = to_jit
        if to_jit:
            self._step_env = jit(self._step_env, static_argnums=(2,))

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
