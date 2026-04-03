"""Shared fixtures for student-facing API tests.

Feature Inventory
=================
Complete list of student-facing features demonstrated in educational
sessions 1-4 and the miniproject notebook. Each feature is covered by
at least one test in this package. This inventory also serves as the
basis for the Phase 3 tutorial.

1. Session Management
   - VivariumController.start_session(scene_name=...)
   - controller.close()

2. Entity Access
   - controller.agents / controller.objects
   - Indexing: controller.agents[0]
   - Slicing: controller.agents[0:4]
   - Iteration: for agent in controller.agents
   - Length: len(controller.agents)

3. Entity Properties (read/write)
   - Position: x_position, y_position
   - Orientation: orientation
   - Appearance: color, diameter
   - Motion: max_speed, left_motor, right_motor
   - Physics: mass, friction (objects)
   - Existence: exists (bool)
   - Proximeter config: proxs_dist_max, proxs_cos_min
   - Subtype: subtype (read/write with label strings)

4. Entity Methods
   - agent.stop_motors()
   - agent.print_infos()
   - agent.print_behaviors() / print_behaviors(full_infos=True)
   - agent.print_routines()
   - agent.has_consumed()

5. Sensing
   - agent.proximeters() -> [left, right]
   - agent.proximeters(sensed_entities=["subtype_name"]) (selective)

6. Behaviors
   - agent.attach_behavior(fn)
   - agent.attach_behavior(fn, weight=W)
   - agent.attach_behavior(fn, interval=N)
   - agent.detach_behavior(fn) / agent.detach_behavior(fn, stop_motors=True)
   - agent.detach_all_behaviors() / agent.detach_all_behaviors(stop_motors=True)
   - agent.change_behavior_weight(fn, new_weight=W)
   - Behavior fn signature: def beh(agent) -> (left_motor, right_motor)

7. Routines
   - agent.attach_routine(fn) / agent.attach_routine(fn, interval=N)
   - agent.detach_routine(fn)
   - agent.detach_all_routines()
   - controller.attach_routine(fn) / controller.attach_routine(fn, interval=N)
   - controller.detach_routine(fn)
   - Routine fn signature (agent): def r(agent) -> None
   - Routine fn signature (controller): def r(controller) -> None

8. Internal State
   - agent.internal.<attr> = value (arbitrary namespace, client-side only)

9. Logger
   - agent.logger.add(field, value)
   - agent.logger.get(field) -> list
   - agent.logger.clear() / agent.logger.clear(field)
   - controller.logger (same API)

10. Consumption (multi-slot)
    - controller.consumption.slot_1.source_subtype / target_subtype
    - controller.consumption.slot_1.range
    - controller.consumption.slot_1.start = True/False

11. Spawn (multi-slot)
    - controller.spawn.slot_1.subtype
    - controller.spawn.slot_1.period
    - controller.spawn.slot_1.start = True/False
    - controller.spawn.slot_1.position_range

12. Simulator Settings
    - controller.simulator.env.num_scan_steps

13. Custom Subtype Labels
    - controller.subtypes (read)
    - controller.set_subtype_labels([...])
    - Propagation to entity, consumption, spawn controllers
    (Covered in test_subtype_labels.py)
"""

import pytest


SCENE = 'miniproject'


@pytest.fixture
def controller(vivarium_controller_from_config):
    """In-process miniproject controller (no gRPC, no simulation running).

    Use for fast tests that only need property access and client-side logic.
    Call controller.apply_changes() to flush deferred writes to state.
    """
    return vivarium_controller_from_config(SCENE)


@pytest.fixture
def running_controller(vivarium_controller_start_session):
    """Miniproject controller with simulation running (gRPC, full lifecycle).

    Use for tests that need actual physics stepping (behaviors, routines,
    has_consumed, position changes).
    """
    return vivarium_controller_start_session(
        SCENE, overrides=["environment.kwargs.debug_mode=true"]
    )
