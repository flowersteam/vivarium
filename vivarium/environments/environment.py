import logging as lg

from jax import jit, lax, random

from jax_md import partition, space

from vivarium.environments.physics_engine import init_state_fn

from vivarium.utils.converters import access_nested_fields


# Generic mask function factory
def get_mask_fn(label):
    if label == 'exists':
        return lambda state: state.entity_state.exists == 1


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
                 dynamics_functions=[], 
                 num_scan_steps=1, to_jit=True, key=random.PRNGKey(42)):

        self.state = state
        self.key, sub_key = random.split(key)
        # self.init_fn = init_state_fn(sub_key)
        self.dynamics_functions = dynamics_functions
        self.dynamics_function_names_to_idx = {fn.__name__: idx for idx, fn in enumerate(dynamics_functions)}
        self.neighbor_manager = neighbor_manager
        self.num_scan_steps = num_scan_steps
        self.to_jit = to_jit
        if to_jit:
            self._step_env = jit(self._step_env, static_argnums=(2,))

    @classmethod
    def init_neighbor_manager(cls, state, box_size, neighbor_radius, space_fn=space.periodic, **kwargs):
        neighbor_manager = NeighborManager(box_size, neighbor_radius, state, space_fn)
        return cls(state, neighbor_manager, **kwargs)

    def get_dynamics_function_by_name(self, name):
        return self.dynamics_functions[self.dynamics_function_names_to_idx[name]]
    
    def _step_env(
        self, state, neighbors, num_scan_steps=1
    ):
        def step_fn(carry, _):
            """Apply a step function to return new state and neighbors in a jax.lax.scan update

            :param carry: tuple of (state, neighbors)
            :param _: dummy xs for jax.lax.scan
            :return: tuple of (carry, carry) with carry=(new_state, new_neighbors)
            """
            state, neighbors, key = carry
            for fn in self.dynamics_functions:
                key, sub_key = random.split(key)
                state = fn(state, neighbors, sub_key) 
            neighbors = self.neighbor_manager.update(state.entity_state.unified_position)
            state = state.set(time=state.time + 1)
            carry = (state, neighbors, key)
            return carry, carry
        (state, neighbors, key), _ = lax.scan(step_fn, (state, neighbors, self.key), xs=None, length=num_scan_steps)
        return state, neighbors, key
        

    def step(self, state):

        # if state.entity_state.momentum is None:
        #     state = self.init_fn(state)

        current_state = state
        neighbors = self.neighbor_manager.neighbors
        state, neighbors, self.key = self._step_env(current_state, neighbors, self.num_scan_steps)
        self.neighbor_manager.neighbors = neighbors

        self.neighbor_manager.reallocate_if_overflow(state)

        return state
