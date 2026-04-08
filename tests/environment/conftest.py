"""
Component Fixture Chain for Environment Tests

Provides component factory fixtures that compose into full environments.
The chain: step → braitenberg → spawn → proximity_map → consumption → energy → reproduction
Then: environment(factories) → environment_and_state(factories)
"""

import jax.numpy as jnp

from vivarium.environment.components.eco_evo import (
    ReproductionComponent, ConsumptionComponent, SpawnComponent, EnergyComponent
)
from vivarium.environment.components.entities.braitenberg.component import BraitenbergComponent
from vivarium.environment.components.proximity_map.component import ProximityMapComponent
from vivarium.environment.components.physics.step.component import StepComponent
from vivarium.environment import Environment, NeighborManager, MaskFunction
from vivarium.environment.state import BaseState

import pytest


def factory_names(factories):
    return [f.name for f in factories]


def remove_duplicates(factories):
    no_duplicate = []
    for f in factories:
        if f.name not in factory_names(no_duplicate):
            no_duplicate.append(f)
    return no_duplicate


@pytest.fixture
def step():
    """Component list: single StepComponent (Verlet integration)."""
    return [StepComponent('step', 10, 0.1, MaskFunction('exists'))]


@pytest.fixture
def braitenberg(step):
    """Component list: step + braitenberg (4 agents, 2 subtypes)."""
    n_agents = 4
    braitenberg = BraitenbergComponent(
        name='braitenberg',
        precedence=1,
        entity_type='agents',
        subtype=jnp.array([0, 0, 1, 1], dtype=int),
        position=jnp.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]),
        orientation=jnp.array([0., 0., 0., 0.]),
        mass=jnp.array([1., 1., 1., 1.]),
        diameter=jnp.full((n_agents, ), 4.),
        friction=jnp.full((n_agents, ), 1.),
        exists=jnp.full((n_agents,), True, dtype=bool),
        n_behaviors=4,
        n_subtypes=2,
        wheel_diameter=1.0,
        max_speed=1.0,
        proxs_dist_max=20.0,
        proxs_cos_min=0.)
    return [*step, braitenberg]


@pytest.fixture
def proximity_map(step, braitenberg):
    """Component list: step + braitenberg + proximity map."""
    return [*step, *braitenberg, ProximityMapComponent('proximity_map', 0)]


@pytest.fixture
def spawn(braitenberg):
    """Component list: braitenberg + spawn (period=1, subtype=0)."""
    spawn = SpawnComponent(
        name='spawn',
        precedence=1,
        default=dict(
            subtype=0,
            period=1,
            start=True,
            position_range=[50., 60., 50., 60.],
            orientation_range=[3., 3.2]
        )
    )
    return [*braitenberg, spawn]


@pytest.fixture
def consumption(proximity_map):
    """Component list: proximity_map + consumption (subtype 0 → 1)."""
    consumption = ConsumptionComponent(
        name='consumption',
        precedence=1,
        test_consumption = dict(
            source_subtype=0,
            target_subtype=1,
            range=1.0,
            start=True
        ),
        consuming_in_entity_state=True
    )
    return [*proximity_map, consumption]


@pytest.fixture
def energy(consumption):
    """Component list: consumption + energy (agents, init=0.5, max=1)."""
    energy_component = EnergyComponent(
        name='energy',
        precedence=2,
        entity_type='agents',
        energy_init=0.5,
        energy_max=1.,
        energy_decay=0.00001,
        energy_burst=0.7
    )
    return [*consumption, energy_component]


@pytest.fixture
def reproduction(energy):
    """Component list: energy + reproduction (consumption disabled)."""
    # Disable the ConsumptionComponent for the reproduction test.
    energy[-2].consumption_params_dict['test_consumption']['start'] = False
    reproduction = ReproductionComponent(
        name='reproduction',
        precedence=3,
        entity_type='agents',
        subtype=-1,
        birth_energy_threshold=0.5,
        death_energy_threshold=0.1,
        birth_recovery_time=100,
        birth_radius=10.,
        birth_energy=0.5
    )
    return [*energy, reproduction]


@pytest.fixture
def environment():
    """Factory: create an Environment from a list of component factories."""
    def fn(factories, debug_mode=False):
        nm = NeighborManager(box_size=100., neighbor_radius=150., dr_threshold=10.)
        env = Environment(
            neighbor_manager=nm,
            base_state_cls=BaseState,
            factories=remove_duplicates(factories),
            to_jit=not debug_mode,
            debug_mode=debug_mode
        )
        return env
    return fn


@pytest.fixture
def environment_and_state(environment):
    """Factory: create an Environment and its initial state from component factories."""
    def fn(factories, debug_mode=False):
        env = environment(factories, debug_mode=debug_mode)
        state = env.init_state()
        return env, state
    return fn
