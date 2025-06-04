import logging as lg

from jax import jit, lax, random
import jax.numpy as jnp

from jax_md import partition, space

from vivarium.environments.state import BaseState, create_state_cls

from vivarium.utils.converters import access_nested_fields


# Generic mask function factory
def get_mask_fn(label):
    if label == 'exists':
        return lambda state: state.entity_state.exists == 1


class NeighborManager:
    def __init__(self, box_size, neighbor_radius, space_fn=space.periodic):
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
    
    def allocate(self, positions):
        self.neighbors = self.neighbor_fn.allocate(positions)
    
    def update(self, positions):
        self.neighbors = self.neighbors.update(positions)
        return self.neighbors
    
    def reallocate_if_overflow(self, positions):
        if self.neighbors.did_buffer_overflow:
            # reallocate neighbors and run the simulation from current_state
            lg.warning(
                f"NEIGHBORS BUFFER OVERFLOW: rebuilding neighbors"
            )
            self.allocate(positions)
            assert not self.neighbors.did_buffer_overflow


nested_fields_to_access = {
    'neighbor_manager': ['box_size', 'neighbor_radius'],
}


@access_nested_fields({'neighbor_manager': ['box_size', 'neighbor_radius']})
class Environment:
    def __init__(self,
                 neighbor_manager,
                #  dt,
                 base_state_cls=BaseState,
                 factories=[], 
                 num_scan_steps=1, to_jit=True, key=random.PRNGKey(42)):

        self.key = key
        self.base_state_cls = base_state_cls
        self.factories = factories
        self.factories_names_to_idx = {f.name: idx for idx, f in enumerate(factories)}
        self.neighbor_manager = neighbor_manager
        # self.dt = dt
        self.num_scan_steps = num_scan_steps
        self.to_jit = to_jit
        if to_jit:
            self._step_env = jit(self._step_env, static_argnums=(2,))

    @classmethod
    def init_neighbor_manager(cls, box_size, neighbor_radius, space_fn=space.periodic, **kwargs):
        neighbor_manager = NeighborManager(box_size, neighbor_radius, space_fn)
        return cls(neighbor_manager, **kwargs)
    
    def init_state_cls(self):
        update_fns = [f.update_state_cls for f in self.factories]
        return create_state_cls(base_state_cls=self.base_state_cls, update_fns=update_fns)
    
    def init_state(self, init_step_functions=True):
        state_cls = self.init_state_cls()
        entity_state_cls = state_cls.__annotations__['entity_state']
        entity_state = entity_state_cls(
            entity_type=jnp.array([], dtype=int),
            entity_type_idx=jnp.array([], dtype=int),
            exists=jnp.array([], dtype=int),
            position=jnp.empty((0, 2), dtype=float),
            orientation=jnp.array([], dtype=float),
            momentum=None,
            mass=jnp.empty((0, 1), dtype=float),
            force=jnp.empty((0, 2), dtype=float),
            previous_force=jnp.empty((0, 2), dtype=float),
            entity_subtype=jnp.array([], dtype=int),
            diameter=jnp.array([], dtype=float),
            friction=jnp.array([], dtype=float)
        )
        for factory in self.factories:
            entity_state = factory.init_base_entity(entity_state)
        self.neighbor_manager.allocate(entity_state.unified_position)
        state = state_cls(time=0, entity_state=entity_state)
        for factory in self.factories:
            state = factory.init_state_fn(state, self.neighbor_manager, self.key)
        if init_step_functions:
            self.init_step_functions(state)
        return state
    
    def init_step_functions(self, state):
        self.step_functions = []
        self.factories.sort(key=lambda f: f.precedence)
        for factory in self.factories:
            self.step_functions.append(factory.get_step_function(state, self.neighbor_manager, self.key))

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
            for fn in self.step_functions:
                key, sub_key = random.split(key)
                state = fn(state, neighbors, sub_key) 
            neighbors = self.neighbor_manager.update(state.entity_state.unified_position)
            state = state.set(time=state.time + 1)
            carry = (state, neighbors, key)
            return carry, carry
        (state, neighbors, key), _ = lax.scan(step_fn, (state, neighbors, self.key), xs=None, length=num_scan_steps)
        return state, neighbors, key
        

    def step(self, state, scan=True):

        current_state = state
        neighbors = self.neighbor_manager.neighbors
        if scan:
            state, neighbors, self.key = self._step_env(current_state, neighbors, self.num_scan_steps)
        else:
            for fn in self.step_functions:
                self.key, sub_key = random.split(self.key)
                state = fn(state, neighbors, sub_key) 
            neighbors = self.neighbor_manager.update(state.entity_state.unified_position)
            state = state.set(time=state.time + 1)

        self.neighbor_manager.neighbors = neighbors

        self.neighbor_manager.reallocate_if_overflow(state)

        return state
