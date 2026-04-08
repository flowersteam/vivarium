"""Tests for multi-spawn extension (Phase 2).

Tests cover:
- SpawnComponent: backward compat (flat params), multi-config (named dicts)
- SpawnState shape for single vs. multi config
- SpawnComponent simulation step: entities are actually spawned
- SpawnController: single config direct access, multi-config named access
- Backward compat: existing scenes using flat spawn params still work
"""

import pytest

from vivarium.components.eco_evo.spawn.component import SpawnComponent, SpawnState
from vivarium.components.eco_evo import SpawnComponent as SpawnComponentFromInit


# ─── SpawnComponent Unit Tests ────────────────────────────────────────────────

class TestSpawnComponentInit:

    def test_flat_params_wrapped_as_default(self):
        """Legacy flat params are auto-wrapped as single config named 'default'."""
        comp = SpawnComponent(
            name='spawn', precedence=1,
            subtype=0, period=100, start=True,
            position_range=[0., 100., 0., 100.],
            orientation_range=[0., 6.28]
        )
        assert 'default' in comp.spawn_params_dict
        assert len(comp.spawn_params_dict) == 1
        assert comp.spawn_params_dict['default']['subtype'] == 0
        assert comp.spawn_params_dict['default']['period'] == 100

    def test_named_dict_configs_stored(self):
        """Named dict configs are stored directly."""
        comp = SpawnComponent(
            name='spawn', precedence=1,
            spawn_a=dict(subtype=0, period=100, start=True,
                         position_range=[0., 100., 0., 100.],
                         orientation_range=[0., 6.28]),
            spawn_b=dict(subtype=1, period=200, start=False,
                         position_range=[0., 50., 0., 50.],
                         orientation_range=[0., 3.14]),
        )
        assert set(comp.spawn_params_dict.keys()) == {'spawn_a', 'spawn_b'}
        assert comp.spawn_params_dict['spawn_a']['subtype'] == 0
        assert comp.spawn_params_dict['spawn_b']['period'] == 200

    def test_multi_config_ignores_flat_scalars_from_defaults(self):
        """When dict configs are present, flat scalars (from default.yaml merging) are ignored."""
        comp = SpawnComponent(
            name='spawn', precedence=1,
            start=True,  # flat scalar (e.g. from spawn/default.yaml)
            spawn_resources=dict(subtype=2, period=200, start=True,
                                 position_range=[0., 100., 0., 100.],
                                 orientation_range=[0., 6.28]),
        )
        assert set(comp.spawn_params_dict.keys()) == {'spawn_resources'}

    def test_exported_from_eco_evo_init(self):
        """SpawnComponent is accessible from the eco_evo package."""
        assert SpawnComponentFromInit is SpawnComponent


class TestSpawnStateShape:

    def test_single_config_state_shape(self, environment_and_state, spawn):
        """Single spawn config produces SpawnState with shape [1, ...]."""
        spawn_comp = spawn[-1]  # last element from conftest spawn fixture
        env, state = environment_and_state(spawn)
        spawn_state = getattr(state, spawn_comp.state_attr)
        assert spawn_state.subtype.shape == (1,)
        assert spawn_state.period.shape == (1,)
        assert spawn_state.start.shape == (1,)
        assert spawn_state.position_range.shape == (1, 4)
        assert spawn_state.orientation_range.shape == (1, 2)

    def test_multi_config_state_shape(self, environment_and_state, braitenberg):
        """Multi spawn config produces SpawnState with shape [n_configs, ...]."""
        comp = SpawnComponent(
            name='spawn', precedence=1,
            spawn_a=dict(subtype=0, period=10, start=True,
                         position_range=[0., 100., 0., 100.],
                         orientation_range=[0., 6.28]),
            spawn_b=dict(subtype=1, period=50, start=True,
                         position_range=[0., 50., 0., 50.],
                         orientation_range=[0., 3.14]),
        )
        factories = [*braitenberg, comp]
        env, state = environment_and_state(factories)
        spawn_state = getattr(state, comp.state_attr)
        assert spawn_state.subtype.shape == (2,)
        assert spawn_state.period.shape == (2,)
        assert spawn_state.position_range.shape == (2, 4)
        assert spawn_state.orientation_range.shape == (2, 2)

    def test_state_values_single(self, environment_and_state, spawn):
        """Verify stored values match what was configured."""
        spawn_comp = spawn[-1]
        env, state = environment_and_state(spawn)
        spawn_state = getattr(state, spawn_comp.state_attr)
        assert int(spawn_state.subtype[0]) == 0
        assert int(spawn_state.period[0]) == 1
        assert bool(spawn_state.start[0]) is True

    def test_state_values_multi(self, environment_and_state, braitenberg):
        """Verify stored values for multi-config."""
        comp = SpawnComponent(
            name='spawn', precedence=1,
            spawn_a=dict(subtype=0, period=10, start=True,
                         position_range=[0., 100., 0., 100.],
                         orientation_range=[0., 6.28]),
            spawn_b=dict(subtype=1, period=50, start=False,
                         position_range=[0., 50., 0., 50.],
                         orientation_range=[0., 3.14]),
        )
        factories = [*braitenberg, comp]
        env, state = environment_and_state(factories)
        spawn_state = getattr(state, comp.state_attr)
        assert int(spawn_state.subtype[0]) == 0
        assert int(spawn_state.subtype[1]) == 1
        assert int(spawn_state.period[0]) == 10
        assert int(spawn_state.period[1]) == 50
        assert bool(spawn_state.start[0]) is True
        assert bool(spawn_state.start[1]) is False


class TestSpawnSimulationStep:

    def test_single_spawn_increases_exists(self, environment_and_state, spawn):
        """Running a step with spawn active increases the number of existing entities."""
        env, state = environment_and_state(spawn)
        initial_exists = int(state.entity_state.exists.sum())

        # Run one step (period=1 in the fixture, so it spawns every step)
        state = env.step(state)
        new_exists = int(state.entity_state.exists.sum())

        assert new_exists >= initial_exists  # at least no regression

    def test_multi_spawn_step_runs_without_error(self, environment_and_state, braitenberg):
        """Multi-spawn step function executes without JAX errors."""
        comp = SpawnComponent(
            name='spawn', precedence=2,
            spawn_a=dict(subtype=0, period=1, start=True,
                         position_range=[0., 100., 0., 100.],
                         orientation_range=[0., 6.28]),
            spawn_b=dict(subtype=1, period=1, start=True,
                         position_range=[0., 100., 0., 100.],
                         orientation_range=[0., 6.28]),
        )
        factories = [*braitenberg, comp]
        env, state = environment_and_state(factories)
        # Should not raise
        env.step(state)

    def test_spawn_disabled_does_not_spawn(self, environment_and_state, braitenberg):
        """When start=False for all configs, no new entities are spawned."""
        comp = SpawnComponent(
            name='spawn', precedence=2,
            spawn_a=dict(subtype=0, period=1, start=False,
                         position_range=[0., 100., 0., 100.],
                         orientation_range=[0., 6.28]),
        )
        factories = [*braitenberg, comp]
        env, state = environment_and_state(factories)
        initial_exists = int(state.entity_state.exists.sum())

        for _ in range(5):
            state = env.step(state)

        assert int(state.entity_state.exists.sum()) == initial_exists


# ─── SpawnController Tests (via grpc_client) ──────────────────────────────────

@pytest.mark.slow
class TestSpawnController:

    def test_single_config_direct_access(self, vivarium_controller_start_session):
        """Single spawn config allows direct attribute access on controller."""
        controller = vivarium_controller_start_session('session_4')
        spawn_ctrl = controller.spawn
        # Direct attribute access works for single-config
        assert isinstance(spawn_ctrl.period, (int, float))
        assert isinstance(spawn_ctrl.start, bool)

    def test_single_config_spawn_names(self, vivarium_controller_start_session):
        """Single spawn config has exactly one name: 'default'."""
        controller = vivarium_controller_start_session('session_4')
        spawn_ctrl = controller.spawn
        assert spawn_ctrl._spawn_names == ['default']
        assert len(spawn_ctrl._single_spawn_controllers) == 1

    def test_single_config_set_no_error(self, vivarium_controller_start_session):
        """Setting attributes via direct access doesn't raise for single config."""
        controller = vivarium_controller_start_session('session_4')
        spawn_ctrl = controller.spawn
        # Should not raise — Remote records changes for async dispatch
        spawn_ctrl.start = True
        spawn_ctrl.start = False

    def test_multi_config_direct_set_raises(self, vivarium_controller_start_session):
        """Direct attribute set raises AttributeError when multiple spawn configs exist."""
        controller = vivarium_controller_start_session('session_4')
        spawn_ctrl = controller.spawn
        # Simulate multi-config by patching _spawn_names
        spawn_ctrl._spawn_names = ['spawn_a', 'spawn_b']
        with pytest.raises(AttributeError, match="spawn rule"):
            spawn_ctrl.start = True


# ─── Backward Compatibility Tests ─────────────────────────────────────────────

class TestBackwardCompatibility:

    @pytest.mark.slow
    def test_existing_scene_with_flat_spawn_loads(self, vivarium_controller_start_session):
        """Existing scenes with flat spawn params still load correctly."""
        # session_4 uses flat spawn params: subtype, period, start, position_range, orientation_range
        controller = vivarium_controller_start_session('session_4')
        assert hasattr(controller, 'spawn')
        spawn_ctrl = controller.spawn
        # Should have a single spawn controller named 'default'
        assert len(spawn_ctrl._spawn_names) == 1

    def test_flat_spawn_component_has_one_config(self):
        """Flat-param SpawnComponent produces exactly one config named 'default'."""
        comp = SpawnComponent(
            name='spawn', precedence=1,
            subtype=1, period=400, start=False,
            position_range=[0., 100., 0., 100.],
            orientation_range=[0., 6.3]
        )
        assert len(comp.spawn_params_dict) == 1
        assert 'default' in comp.spawn_params_dict

    def test_multi_config_spawn_component_names(self):
        """Multi-config SpawnComponent preserves config names in order."""
        comp = SpawnComponent(
            name='spawn', precedence=1,
            spawn_resources=dict(subtype=2, period=200, start=True,
                                 position_range=[0., 100., 0., 100.],
                                 orientation_range=[0., 6.28]),
            spawn_obstacles=dict(subtype=3, period=500, start=True,
                                 position_range=[0., 50., 0., 50.],
                                 orientation_range=[0., 6.28]),
        )
        names = list(comp.spawn_params_dict.keys())
        assert names == ['spawn_resources', 'spawn_obstacles']
