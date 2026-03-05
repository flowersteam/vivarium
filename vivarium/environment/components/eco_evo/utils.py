import jax.numpy as jnp
from jax import lax, random

from vivarium.environment.utils import generate_random_orientations, generate_random_positions, is_position_close, type_mask


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


def non_existing(key, entity_state, entity_type=-1, subtype=-1):

    mask = type_mask(entity_state, exists=0, entity_type=entity_type, subtype=subtype)
    return sample_true_index(key, mask)


def set_random_pos_at(key, all_positions, idx, range, max_trial=100):
    def cond_fun(val):
        #TODO: It seems that 2 entities at the same position no longer crashes the simulation
        # So let's set atol very low
        pos, idx, other_positions, key, init, trial = val
        return jnp.logical_and(trial < max_trial,
                               jnp.logical_or(init, is_position_close(pos, idx, other_positions, atol=0.01))
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