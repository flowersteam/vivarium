from jax import lax
from jax import random
import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.environment.components.eco_evo.utils import non_existing, sample_true_index, spawn_entity_at_idx
from vivarium.environment.components.component import Component
from vivarium.environment.utils import type_mask


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

    def update_state_cls(self, state_cls):
        @md_dataclass
        class ReproductionState:
            subtype: jnp.ndarray
            birth_energy_threshold: jnp.ndarray
            death_energy_threshold: jnp.ndarray
            birth_recovery_time: jnp.ndarray
            birth_radius: jnp.ndarray
            birth_energy: jnp.ndarray
            recover_time: jnp.ndarray
        
        @md_dataclass
        class AgentState(state_cls.__annotations__[self.entity_type]):
            reproduction: ReproductionState = None
        state_cls.__annotations__[self.entity_type] = AgentState
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        agent_state = getattr(state, self.entity_type)
        reproduction_cls  = state.__class__.__annotations__[self.entity_type].__annotations__['reproduction']
        recover_time = jnp.full(state.entity_state.exists.shape, 0, dtype=int)
        return state.set(
            **{self.entity_type: agent_state.set(
                reproduction=reproduction_cls(
                    subtype=self.subtype,
                    birth_energy_threshold=self.birth_energy_threshold,
                    death_energy_threshold=self.death_energy_threshold,
                    birth_recovery_time=self.birth_recovery_time,
                    birth_radius=self.birth_radius,
                    birth_energy=self.birth_energy,
                    recover_time=recover_time
                )
            )}
        )

    def get_step_function(self, state, neighbor_manager, key):
        idxs = state.e_cond(self.entity_type)
        entity_type = state.entity_type_to_int(self.entity_type)

        def step_fn(state, neighbors, key):

            entities = getattr(state, self.entity_type)

            cur_energy = jnp.zeros(state.entity_state.exists.shape)
            cur_energy = cur_energy.at[idxs].set(entities.energy)

            death_mask = jnp.logical_and(
                type_mask(state.entity_state, entity_type=entity_type, subtype=entities.reproduction.subtype),
                cur_energy <= entities.reproduction.death_energy_threshold
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
            cur_recover_time = cur_recover_time.at[idxs].set(entities.reproduction.recover_time)

            reproduce_mask = jnp.logical_and(
                jnp.logical_and(
                    type_mask(state.entity_state, entity_type=entity_type, subtype=entities.reproduction.subtype), 
                    cur_energy > entities.reproduction.birth_energy_threshold),
                cur_recover_time > entities.reproduction.birth_recovery_time
            )

            # Workaround to make it simpler, we reproduce only a single agent per time step
            key, sub_key = random.split(key)
            does_reproduce, parent_idx = sample_true_index(sub_key, reproduce_mask)

            key, sub_key = random.split(key)
            can_be_born, offspring_idx = non_existing(sub_key, state.entity_state, entity_type=entity_type, subtype=state.entity_state.entity_subtype[parent_idx])

            parent_position = state.entity_state.position[parent_idx]
            offspring_min = parent_position - entities.reproduction.birth_radius
            offspring_max = parent_position + entities.reproduction.birth_radius
            
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
                lambda: entities.energy.at[state.entity_state.entity_type_idx[offspring_idx]].set(entities.reproduction.birth_energy),
                lambda: entities.energy
            )


            recover_time = entities.reproduction.recover_time + 1

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
                    reproduction=entities.reproduction.set(recover_time=recover_time),
                    energy=energy,
                )}
            )
        
        return step_fn