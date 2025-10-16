import pytest
import jax.numpy as jnp

from vivarium.environment.components.eco_evo import ConsumptionComponent, EnergyComponent, ReproductionComponent
from vivarium.environment.components.entities.particle_lenia.interface import ParamParticleLenia
from vivarium.environment.components.entities.braitenberg.component import BraitenbergComponent
from vivarium.environment.components.proximity_map.component import ProximityMapComponent
from vivarium.utils.scene_configs import load_config, component_factories_from_config
from vivarium.environment.components.entities.braitenberg.interface import ParamAgent
from vivarium.environment.components.physics.step.component import StepComponent
from vivarium.environment import Environment, NeighborManager, MaskFunction
from vivarium.environment.state import BaseState, create_state_cls
from vivarium.interface.panel_app import create_interfaces
from vivarium.controllers import SimulatorController
from vivarium.environment import Environment
from vivarium.simulator import Simulator


param_fields_to_delete = {
     ParamAgent: lambda field: field.startswith('sensed_') or field.startswith('behavior_'),
     ParamParticleLenia: lambda field: field == 'creature'
}

@pytest.fixture(autouse=True)
def cleanup_parameterized_class(request):
    """
    Remove the dynamically added parameters from the Param classes
    as they might be remnants from previous tests
    """
    for param_cls, condition in param_fields_to_delete.items():
        to_del = []
        for field_name in param_cls.__dict__.keys():
            if condition(field_name):
                to_del.append(field_name)
        for field_name in to_del:
            delattr(param_cls, field_name)
            del param_cls._param__parameters._cls_parameters[field_name]


@pytest.fixture
def scene_config():
    def fn(scene_name):
        return load_config('scene', scene_name)
    return fn


@pytest.fixture
def environment_from_config(scene_config):
    def fn(scene_name):
        return Environment.from_config(scene_config(scene_name).environment)
    return fn


@pytest.fixture
def state_from_config(scene_config):
    def fn(scene_name):
        config = scene_config(scene_name)
        base_state_cls = config.environment.base_state_cls
        update_fns = [f.update_state_cls for f in component_factories_from_config(config.environment.components)]
        return create_state_cls(base_state_cls, update_fns)
    return fn


@pytest.fixture
def simulator_from_config(scene_config):
    def fn(scene_name):
        return Simulator.from_config(scene_config(scene_name).simulator)
    return fn
    

@pytest.fixture
def simulator_controller_from_config(scene_config, simulator_from_config):
    def fn(scene_name, controller_cls=SimulatorController):
        return controller_cls.from_config(
            config=scene_config(scene_name).environment.components, 
            client=simulator_from_config(scene_name)
        )
    return fn


@pytest.fixture
def controller_and_interfaces_from_config(scene_config, simulator_controller_from_config):
    def fn(scene_name):
        controller = simulator_controller_from_config(scene_name)
        config = scene_config(scene_name)
        interfaces = create_interfaces(
            config.environment.components.component_list,
            controller.controllers,
            controller.state,
        )
        return controller, interfaces
    return fn


def factory_names(factories):
    return [f.name for f in factories]


def remove_duplicates(factories):
    no_duplicate = []
    for f in factories:
        if f.name not in factory_names(no_duplicate):
            no_duplicate.append(f)
    return no_duplicate


@pytest.fixture
def proximity_map(step, braitenberg):
    return [*step, *braitenberg, ProximityMapComponent('proximity_map', 0)]
    

@pytest.fixture
def consumption(proximity_map):
    consumption = ConsumptionComponent(
        name='consumption', 
        precedence=1, 
        source_subtype=0, 
        target_subtype=1, 
        range=20.0
    )
    return [*proximity_map, consumption]


@pytest.fixture
def energy(consumption):
    energy_component = EnergyComponent(
        name='energy',
        precedence=2,
        entity_type='agents',
        subtype=-1,
        init_energy=0.5,
        max_energy=1.,
        decay=0.00001,
        burst=0.7
    )
    return [*consumption, energy_component]


@pytest.fixture
def reproduction(energy):
    # Disable the ConsumptionComponent for the reproduction test by setting its range to 0.
    energy[-2].range = 0.
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
def braitenberg(step):
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
        exists=jnp.full((n_agents,), 1, dtype=int),
        n_behaviors=4,
        n_subtypes=2,
        wheel_diameter=1.0,
        max_speed=1.0,
        proxs_dist_max=20.0,
        proxs_cos_min=0.)
    return [*step, braitenberg]


@pytest.fixture
def step():
    return [StepComponent('step', 10, 0.1, MaskFunction('exists'))]


@pytest.fixture
def environment():
    def fn(factories):
        nm = NeighborManager(box_size=100., neighbor_radius=150., dr_threshold=10.)
        env = Environment(
            neighbor_manager=nm,
            base_state_cls=BaseState,
            factories=remove_duplicates(factories),
            to_jit=False
        )
        return env
    return fn


@pytest.fixture
def environment_and_state(environment):
    def fn(factories):
        env = environment(factories)
        state = env.init_state()
        return env, state
    return fn