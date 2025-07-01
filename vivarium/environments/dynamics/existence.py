from jax import lax
from jax import random
import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import partition

from vivarium.environments.state import BaseEntityState
from vivarium.environments.utils import generate_random_positions, generate_random_orientations, is_position_close, neighbors_entity_mask
from vivarium.environments.physics_engine import Component


def sample_true_index(key, x):
    # Ensure at least one True exists (undefined behavior otherwise)
    has_true = jnp.any(x)
    key, subkey = random.split(key)
    # Generate random values for all positions
    random_values = random.uniform(subkey, x.shape)
    # Mask non-True entries with -infinity to exclude them
    masked = lax.cond(has_true, lambda: jnp.where(x, random_values, -jnp.inf), lambda: jnp.zeros_like(random_values))
    # Return the index of the maximum (randomly chosen True index)
    return has_true, jnp.argmax(masked)


def type_mask(entity_state, exists=1, entity_type=-1, subtype=-1):
    mask_entity_type = lax.cond(
        entity_type == -1,
        lambda: entity_state.exists == exists,
        lambda: jnp.logical_and(
            entity_state.exists == exists,
            entity_state.entity_type == entity_type
            )
    )
    mask_subtype = lax.cond(
        subtype == -1,
        lambda: entity_state.exists == exists,
        lambda: jnp.logical_and(
            entity_state.exists == exists,
            entity_state.entity_subtype == subtype
        )
    )
    mask = jnp.logical_and(mask_entity_type, mask_subtype)
    return mask


def non_existing(key, entity_state, entity_type=-1, subtype=-1):

    mask = type_mask(entity_state, exists=0, entity_type=entity_type, subtype=subtype)
    return sample_true_index(key, mask)


def set_random_pos_at(key, all_positions, idx, range, max_trial=100):
    def cond_fun(val):
        pos, idx, other_positions, key, init, trial = val
        return jnp.logical_and(trial < max_trial,
                               jnp.logical_or(init, is_position_close(pos, idx, other_positions, atol=6.))
        )
    def body_fun(val):
        pos, idx, other_positions, key, init, trial = val
        key, sub_key = random.split(key)
        new_pos = generate_random_positions(1, range, sub_key)[0]

        return (new_pos, idx, other_positions, key, False, trial + 1)
    
    new_pos, _, _, _, _, trial = lax.while_loop(cond_fun, body_fun, (jnp.zeros(2), idx, all_positions, key, True, 0))
    
    fail = trial >= max_trial
    
    return fail, all_positions.at[idx].set(new_pos)


def set_random_orientation_at(key, orientations, idx, range):
    key, sub_key = random.split(key)
    return orientations.at[idx].set(generate_random_orientations(1, range, sub_key)[0])


def spawn_entity_at_idx(key, state, idx, position_range, orientation_range):

    key, key_pos, key_orientation = random.split(key, 3)

    exists = state.entity_state.exists.at[idx].set(1)

    fail, position = set_random_pos_at(
        key_pos,
        state.entity_state.position,
        idx,
        position_range
    )

    orientation = set_random_orientation_at(
        key_orientation,
        state.entity_state.orientation,
        idx,
        orientation_range
    )

    return state.set(
        entity_state=state.entity_state.set(
            exists=lax.cond(
                fail,
                lambda: state.entity_state.exists,
                lambda: exists
            ),
            position=lax.cond(
                fail,
                lambda: state.entity_state.position,
                lambda: position
            ),
            orientation=lax.cond(
                fail,
                lambda: state.entity_state.orientation,
                lambda: orientation
            ),
        )
    )


def spawn_entity(key, state, position_range, orientation_range, entity_type=-1, subtype=-1):

    has_non_existing, idx = non_existing(key, state.entity_state, entity_type=entity_type, subtype=subtype)

    return lax.cond(
        has_non_existing,
        lambda: spawn_entity_at_idx(key, state, idx, position_range, orientation_range),
        lambda: state
    )


class SpawnComponent(Component):
    def __init__(self, name, precedence, subtype, period, position_range, orientation_range):
        super().__init__(name, precedence)
        self.subtype = subtype
        self.period = period
        self.position_range = position_range
        self.orientation_range = orientation_range
    
    def get_step_function(self, state, neighbor_manager, key):
        
        def state_fn(state, neighbors, key):

            cond = (state.time % self.period) == 0

            return lax.cond(
                cond,
                lambda: spawn_entity(key, state, self.position_range, self.orientation_range, subtype=self.subtype),
                lambda: state
            )
        
        return state_fn


class ConsumptionComponent(Component):
    def __init__(self, name, precedence, source_subtype, target_subtype, range):
        super().__init__(name, precedence)
        self.source_subtype = source_subtype
        self.target_subtype = target_subtype
        self.range = range

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            entity_state=state.entity_state.set(
                consuming=jnp.full(state.entity_state.exists.shape, False),
                consumed=jnp.full(state.entity_state.exists.shape, False)
            )
        )
        
    def update_state_cls(self, state_cls):
        base_cls = state_cls.__annotations__['entity_state'] if 'entity_state' in state_cls.__annotations__ else BaseEntityState
        @md_dataclass
        class EntityState(base_cls):
            consuming: jnp.ndarray = None
            consumed: jnp.ndarray = None
        state_cls.__annotations__['entity_state'] = EntityState
        return state_cls

    def get_step_function(self, state, neighbor_manager, key):
        self.displacement = neighbor_manager.displacement
        def step_fn(state, neighbors, key):
            d_r = state.distance_map

            mask = neighbors_entity_mask(
                neighbors_idx=neighbors.idx, 
                source_mask=jnp.logical_and(state.entity_state.exists == 1, state.entity_state.entity_subtype == self.source_subtype),
                target_mask=jnp.logical_and(state.entity_state.exists == 1, state.entity_state.entity_subtype == self.target_subtype),
                neighbor_mask=partition.neighbor_list_mask(neighbors, mask_self=True)
            )
            mask &= d_r < self.range

            consuming = mask.any(axis=1)
            consumed = jnp.full(state.entity_state.exists.shape, False)

            neigh_flat = neighbors.idx.ravel()
            mask_flat = mask.ravel()
            consumed = consumed.at[neigh_flat].max(mask_flat)

            new_exists = jnp.where(
                consumed,
                0,
                state.entity_state.exists
            )

            return state.set(
                entity_state=state.entity_state.set(
                    exists=new_exists,
                    consuming=consuming,
                    consumed=consumed
                )
            )

        return step_fn


class EnergyComponent(Component):
    def __init__(self, name, precedence,
                 entity_type, subtype,
                 init_energy, max_energy, decay, burst):
        super().__init__(name, precedence)
        self.init_energy = init_energy
        self.max_energy = max_energy
        self.decay = decay
        self.burst = burst
        self.entity_type = entity_type
        self.subtype = subtype

    def get_step_function(self, state, neighbor_manager, key):

        entity_type = state.entity_type_to_int(self.entity_type)
        idxs = state.e_cond(self.entity_type)

        def state_fn(state, neighbors, key):
            entities = getattr(state, self.entity_type)
            
            cur_energy = jnp.zeros(state.entity_state.exists.shape)
            cur_energy = cur_energy.at[idxs].set(entities.energy)

            mask = type_mask(state.entity_state, entity_type=entity_type, subtype=self.subtype)

            energy = jnp.where(
                jnp.logical_and(mask, 
                                state.entity_state.consuming
                                ),
                cur_energy + self.burst,
                cur_energy
            )
            energy = jnp.where(
                mask,
                energy - self.decay,
                energy
            )

            energy = jnp.clip(energy, 0, self.max_energy)

            return state.set(**{
                self.entity_type: entities.set(
                    energy=energy[idxs]
                )}
            )
        
        return state_fn
    
    def update_state_cls(self, state_cls):
        assert 'entity_state' in state_cls.__annotations__, 'no entity_state in state class'
        # assert 'consuming' in state_cls.__annotations__['entity_state'].__annotations__, 'consuming not in entity_state'
        if self.entity_type in state_cls.__annotations__:
            @md_dataclass
            class AgentState(state_cls.__annotations__[self.entity_type]):
                energy: jnp.ndarray = None
        else:
            @md_dataclass
            class AgentState:
                energy: jnp.ndarray = None
        state_cls.__annotations__[self.entity_type] = AgentState
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        return state.set(
            **{self.entity_type: getattr(state, self.entity_type).set(
                energy=jnp.full(getattr(state, self.entity_type).count(), self.init_energy)
            )}
        )


class ReproductionComponent(Component):
    def __init__(self, name, precedence,
                 entity_type, subtype,
                 birth_energy_threshold, death_energy_threshold,
                 birth_recovery_time, 
                 birth_radius,  # actually a square 
                 birth_energy):
        super().__init__(name, precedence)
        self.entity_type = entity_type
        self.subtype = subtype
        self.birth_energy_threshold = birth_energy_threshold
        self.death_energy_threshold = death_energy_threshold
        self.birth_recovery_time = birth_recovery_time
        self.birth_radius = birth_radius
        self.birth_energy = birth_energy
        

    def get_step_function(self, state, neighbor_manager, key):
        idxs = state.e_cond(self.entity_type)
        entity_type = state.entity_type_to_int(self.entity_type)

        def state_fn(state, neighbors, key):

            entities = getattr(state, self.entity_type)

            cur_energy = jnp.zeros(state.entity_state.exists.shape)
            cur_energy = cur_energy.at[idxs].set(entities.energy)

            death_mask = jnp.logical_and(
                type_mask(state.entity_state, entity_type=entity_type, subtype=self.subtype),
                cur_energy <= self.death_energy_threshold
            )
            
            new_exists = jnp.where(
                death_mask,
                0,
                state.entity_state.exists
            )

            state = state.set(
                entity_state=state.entity_state.set(
                    exists=new_exists
                )
            )

            cur_recover_time = jnp.zeros(state.entity_state.exists.shape)
            cur_recover_time = cur_recover_time.at[idxs].set(entities.recover_time)

            reproduce_mask = jnp.logical_and(
                jnp.logical_and(
                    type_mask(state.entity_state, entity_type=entity_type, subtype=self.subtype), 
                    cur_energy > self.birth_energy_threshold),
                cur_recover_time > self.birth_recovery_time
            )

            # Workaround to make it simpler, we reproduce only a single agent per time step
            key, sub_key = random.split(key)
            does_reproduce, parent_idx = sample_true_index(sub_key, reproduce_mask)

            key, sub_key = random.split(key)
            can_be_born, offspring_idx = non_existing(sub_key, state.entity_state, entity_type=entity_type, subtype=state.entity_state.entity_subtype[parent_idx])

            parent_position = state.entity_state.position[parent_idx]
            offspring_min = parent_position - self.birth_radius
            offspring_max = parent_position + self.birth_radius
            
            offspring_position_range = (offspring_min[0], offspring_max[0], offspring_min[1], offspring_max[1])
            offspring_orientation_range = (0, 2 * jnp.pi)

            reproduction_cond = jnp.logical_and(does_reproduce, can_be_born)

            state = lax.cond(
                reproduction_cond,
                lambda: spawn_entity_at_idx(
                    key,
                    state,
                    offspring_idx,
                    offspring_position_range,
                    offspring_orientation_range
                ),
                lambda: state
            )


            energy = lax.cond(
                reproduction_cond,
                lambda: entities.energy.at[state.entity_state.entity_type_idx[offspring_idx]].set(self.birth_energy),
                lambda: entities.energy
            )


            recover_time = entities.recover_time + 1

            recover_time = lax.cond(
                reproduction_cond,
                lambda: recover_time.at[state.entity_state.entity_type_idx[offspring_idx]].set(0),
                lambda: recover_time
            )

            recover_time = lax.cond(
                reproduction_cond,
                lambda: recover_time.at[state.entity_state.entity_type_idx[parent_idx]].set(0),
                lambda: recover_time
            )
            
            return state.set(
                **{self.entity_type: entities.set(
                    recover_time=recover_time,
                    energy=energy,
                )}
            )
        
        return state_fn

    def update_state_cls(self, state_cls):
        @md_dataclass
        class AgentState(state_cls.__annotations__[self.entity_type]):
            recover_time: jnp.ndarray = None
        state_cls.__annotations__[self.entity_type] = AgentState
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        recover_time = jnp.full(state.entity_state.exists.shape, 0, dtype=int)
        return state.set(
            **{self.entity_type: getattr(state, self.entity_type).set(
                recover_time=recover_time
            )}
        )
    