from jax import lax
from jax import random
import jax.numpy as jnp

from jax_md import space

from vivarium.environments.utils import generate_random_positions, generate_random_orientations, is_position_close

def get_entity_spawn_state_fn(env, subtype, period, position_range, orientation_range):
    
    def state_fn(state, neighbors, key):

        cond = (state.time % period) == 0

        def non_existing(exists):
            mask = jnp.logical_and(jnp.logical_not(exists), state.entity_state.entity_subtype == subtype)
            has_false = jnp.any(mask)
            first_false = jnp.argmax(mask)
            return has_false, first_false
        
        def update_exists(exists, has_non_existing, first_non_existing):

            new_exists = exists.at[first_non_existing].set(1)
            return lax.cond(has_non_existing, lambda: new_exists, lambda: exists)

        def set_random_pos_at(all_positions, idx, key):
            def cond_fun(val):
                pos, idx, other_positions, key, init = val
                return jnp.logical_or(init, is_position_close(pos, idx, other_positions, atol=6.))
            def body_fun(val):
                pos, idx, other_positions, key, init = val
                key, sub_key = random.split(key)
                new_pos = generate_random_positions(1, position_range, sub_key)[0]
                return (new_pos, idx, other_positions, key, False)
            
            new_pos = lax.while_loop(cond_fun, body_fun, (jnp.zeros(2), idx, all_positions, key[0], True))[0]
            return all_positions.at[idx].set(new_pos)
        
        def set_random_orientation_at(orientations, idx, key):
            key, sub_key = random.split(key[1])
            return orientations.at[idx].set(generate_random_orientations(1, orientation_range, sub_key)[0])

        has_non_existing, first_non_existing = non_existing(state.entity_state.exists)

        new_exists = lax.cond(
            cond,
            lambda: update_exists(state.entity_state.exists, has_non_existing, first_non_existing),
            lambda: state.entity_state.exists
        )

        key, key_pos, key_orientation = random.split(key, 3)
        new_position = lax.cond(
            jnp.logical_and(cond, has_non_existing),
            set_random_pos_at,
            lambda pos, i, key: state.entity_state.position,
            state.entity_state.position,
            first_non_existing,
            (key_pos, key_orientation)
        )

        new_orientation = lax.cond(
            jnp.logical_and(cond, has_non_existing),
            set_random_orientation_at,
            lambda pos, i, key: state.entity_state.orientation,
            state.entity_state.orientation,
            first_non_existing,
            (key_pos, key_orientation)
        )

        return state.set(
            entity_state=state.entity_state.set(
                exists = new_exists,
                position=new_position,
                orientation=new_orientation
            )
        )
    
    return state_fn


def get_consumption_state_fn(env, source_subtype, target_subtype, range):
    displacement = env.neighbor_manager.displacement
    def state_fn(state, neighbors, key):
        sources, targets = neighbors.idx
        pos_s = state.entity_state.position[sources]
        pos_r = state.entity_state.position[targets]
        d_r = -space.map_bond(displacement)(pos_s, pos_r)
        d_r = jnp.linalg.norm(d_r, axis=-1)
        mask = jnp.logical_and(
            state.entity_state.entity_subtype[sources] == source_subtype,
            state.entity_state.entity_subtype[targets] == target_subtype)
        mask = jnp.logical_and(
            mask,
            state.entity_state.exists[sources] == 1)
        mask = jnp.logical_and(
            mask,
            state.entity_state.exists[targets] == 1)
        mask = jnp.logical_and(
            mask,
            d_r < range
        )

        are_consumed = jnp.where(mask, targets, -1)
        new_exists = jnp.where(
            jnp.isin(jnp.arange(state.entity_state.exists.shape[0]), are_consumed),
            0,
            state.entity_state.exists
        )

        return state.set(
            entity_state=state.entity_state.set(
                exists=new_exists
            )
        )

    return state_fn
