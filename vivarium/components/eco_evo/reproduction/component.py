from jax import lax
from jax import random
import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass

from vivarium.components.eco_evo.utils import non_existing, sample_true_index, spawn_entity_at_idx
from vivarium.components.component import Component
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
        class EntityTypeState(state_cls.__annotations__[self.entity_type]):
            reproduction: ReproductionState = None
        state_cls.__annotations__[self.entity_type] = EntityTypeState
        return state_cls

    def init_state_fn(self, state, neighbor_manager, key):
        agent_state = getattr(state, self.entity_type)
        reproduction_cls  = state.__class__.__annotations__[self.entity_type].__annotations__['reproduction']
        recover_time = jnp.full(agent_state.count(), 0, dtype=int)
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

            entity_state = state.entity_state
            
            energy = entity_state.energy
            entity_type_energy = energy[idxs]
            
            mask = type_mask(state.entity_state, entity_type=entity_type, subtype=entities.reproduction.subtype)
            entity_type_mask = mask[idxs]

            death_mask = jnp.logical_and(
                entity_type_mask,
                entity_type_energy <= entities.reproduction.death_energy_threshold
            )
            
            new_exists = jnp.where(
                jnp.full(state.entity_state.exists.shape, False).at[idxs].set(death_mask),
                False,
                state.entity_state.exists
            )
            
            state = state.set(
                entity_state=state.entity_state.set(
                    exists=new_exists
                )
            )
            
            # Better to reset init energy now
            # Useful e.g. when respawning in SpawnComponent
            # (otherwise entity will die again right away if init energy <= death threshold)
            entity_type_energy = jnp.where(
                death_mask,
                entity_state.energy_init,
                entity_type_energy
            )

            cur_recover_time = entities.reproduction.recover_time

            entity_type_reproduce_mask = jnp.logical_and(
                jnp.logical_and(
                    entity_type_mask,
                    entity_type_energy >= entities.reproduction.birth_energy_threshold),
                cur_recover_time > entities.reproduction.birth_recovery_time
            )
            
            reproduce_mask = jnp.full(state.entity_state.exists.shape, False).at[idxs].set(entity_type_reproduce_mask)

            # Workaround to make it simpler, we reproduce only a single agent per time step
            # TODO: Maybe could use a vmap?
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

            n_exists = jnp.sum(state.entity_state.exists)

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
            
            reproduction_sucess = (jnp.sum(state.entity_state.exists) == n_exists + 1)
            
            reproduction_cond &= reproduction_sucess


            entity_type_energy = lax.cond(
                reproduction_cond,
                lambda: entity_type_energy.at[state.entity_state.entity_type_idx[offspring_idx]].set(entities.reproduction.birth_energy),
                lambda: entity_type_energy
            )
            
            entity_type_energy = lax.cond(
                reproduction_cond,
                lambda: entity_type_energy.at[state.entity_state.entity_type_idx[parent_idx]].subtract(entities.reproduction.birth_energy),
                lambda: entity_type_energy
            )            

            # behavior_params = entities.behavior_params

            # behavior_params = lax.cond(
            #     reproduction_cond,
            #     lambda: behavior_params.at[
            #         state.entity_state.entity_type_idx[offspring_idx]
            #         ].set(
            #             behavior_params[state.entity_state.entity_type_idx[parent_idx]] + random.normal(sub_key, shape=entities.behavior_params.shape[1:]) * 0.01
            #             ),
            #     lambda: behavior_params
            # )
            
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
            
            energy = jnp.where(
                idxs,
                energy.at[idxs].set(entity_type_energy),
                energy
            )
            
            return state.set(
                entity_state=state.entity_state.set(
                    energy=energy
                    ),
                **{self.entity_type: entities.set(
                    reproduction=entities.reproduction.set(recover_time=recover_time),
                    # behavior_params=behavior_params
                    )
                   }
            )
        
        return step_fn