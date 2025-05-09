import jax.numpy as jnp

from vivarium.environments.braitenberg.selective_sensing.state import AgentState
from vivarium.controllers.notebook_controller import NotebookController
from vivarium.utils.scene_configs import SceneConfiguration

NUM_STEPS = 4


def test_notebook_controller():
    """Test default simulator run"""

    simulator = SceneConfiguration('braitenberg').create_simulator()
    agent_field = simulator.state.field_name(AgentState)
    controller = NotebookController(simulator)
    controller.step()

    assert controller.client.freq == -1

    def beh(agent):
        left, right = agent.sensors()
        return 1 - right, 1 - left

    idx = 0
    pos = controller.state.entity_state.position_center[idx]

    ag =controller.agents[idx]

    ag.attach_behavior(beh)
    
    controller.execute_routines_and_behaviors()

    assert (jnp.equal(pos, ag.position_center).all())

    for _ in range(NUM_STEPS):
        pos = controller.state.entity_state.position_center[idx]
        controller.run(threaded=False, num_steps=1)
        assert (not jnp.equal(pos, ag.position_center).all())

    ag.behavior = [3, 1, 2, 0]
    controller.step()
    assert jnp.equal(jnp.array([3, 1, 2, 0]), getattr(controller.state, agent_field).behavior[idx]).all()

    ag.color = 'pink'
    controller.step()

# import time

# from vivarium.controllers.notebook_controller import NotebookController
# from vivarium.utils.handle_server_interface import start_server_and_interface, stop_server_and_interface

# RUN_TIME = 7
# RESOURCE_SPAWN_INTERVAL = 30
# NUM_STEPS = 100


# # TODO : clean the test with pytest fixtures
# def test_notebook_controller():
#     start_server_and_interface(scene_name='session_3')

#     def obstacle_avoidance(agent):
#         """
#         Simple obstacle avoidance behavior based on sensor data.
#         """
#         left, right = agent.sensors(sensed_entities=["obstacles"])
#         return 1. - right, 1. - left

#     def get_agents_positions():
#         """
#         Retrieves the positions of all agents in the simulation.
#         """
#         return [(agent.x_position, agent.y_position) for agent in controller.agents]

#     def count_resources():
#         """
#         Counts the number of existing resources in the simulation.
#         """
#         resources_list = [entity for entity in controller.all_entities if entity.subtype_label == "resources" and entity.exists]
#         return len(resources_list)

#     # Initialize the controller
#     controller = NotebookController()

#     try:
#         controller.run(num_steps=100)

#         # Initial states
#         initial_positions = get_agents_positions()
#         initial_resource_count = count_resources()
#         print(f"Initial resource count: {initial_resource_count}")

#         # Configure each agent
#         for agent in controller.agents:
#             agent.print_infos()
#             agent.print_behaviors()
#             agent.exists = True

#             left, right = agent.sensors(sensed_entities=["robots", "obstacles"])
#             print(f"Agent sensors - Left: {left}, Right: {right}")
#             assert isinstance(left, (float, int))
#             assert isinstance(right, (float, int))

#             agent.detach_all_behaviors()
#             agent.attach_behavior(obstacle_avoidance, interval=5, weight=2.0)
#             agent.start_all_behaviors()

#             agent.proxs_dist_max = 100.0
#             agent.proxs_cos_min = 0.8

#         controller.start_resources_apparition(interval=RESOURCE_SPAWN_INTERVAL, position_range=((0, 50), (100, 200)))

#         # TODO : maybe remove this and use the controller.run() method without thread (but not what we usually use)
#         # Run simulation for the specified time
#         time.sleep(RUN_TIME)

#         # Final states
#         final_positions = get_agents_positions()
#         final_resource_count = count_resources()
#         spawned_resources = final_resource_count - initial_resource_count
#         expected_spawned_resources = NUM_STEPS // RESOURCE_SPAWN_INTERVAL

#         assert initial_positions != final_positions, "Agents should have moved from their initial positions."
#         assert spawned_resources == expected_spawned_resources, "Unexpected number of spawned resources."

#     finally:
#         controller.stop()
#         stop_server_and_interface()
