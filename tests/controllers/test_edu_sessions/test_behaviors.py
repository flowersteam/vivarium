"""Tests for behavior attachment, detachment, weighting, and execution."""

import pytest
import numpy as np


NUM_STEPS = 4


def constant_forward(agent):
    return 1.0, 1.0


def constant_left(agent):
    return 0.0, 1.0


# ---------------------------------------------------------------------------
# Attach / detach
# ---------------------------------------------------------------------------

def test_attach_behavior(controller):
    """attach_behavior registers the behavior in the handler."""
    ag = controller.agents[0]
    assert len(ag.behavior_handler._behaviors) == 0
    ag.attach_behavior(constant_forward)
    assert 'constant_forward' in ag.behavior_handler._behaviors
    assert 'constant_forward' in ag.behavior_handler._started_behaviors


def test_detach_behavior(controller):
    """detach_behavior removes a specific behavior."""
    ag = controller.agents[0]
    ag.attach_behavior(constant_forward)
    ag.detach_behavior(constant_forward)
    # Verify no behaviors remain active
    assert len(ag.behavior_handler._started_behaviors) == 0


def test_detach_behavior_stop_motors(controller):
    """detach_behavior(fn, stop_motors=True) zeroes motors."""
    ag = controller.agents[0]
    ag.attach_behavior(constant_forward)
    ag.left_motor = 1.0
    ag.right_motor = 1.0
    controller.apply_changes()

    ag.detach_behavior(constant_forward, stop_motors=True)
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.left_motor), 0.0, decimal=2)
    np.testing.assert_almost_equal(float(ag.right_motor), 0.0, decimal=2)


def test_detach_all_behaviors(controller):
    """detach_all_behaviors removes all behaviors."""
    ag = controller.agents[0]
    ag.attach_behavior(constant_forward)
    ag.attach_behavior(constant_left)
    ag.detach_all_behaviors()
    assert len(ag.behavior_handler._started_behaviors) == 0
    assert len(ag.behavior_handler._behaviors) == 0


def test_detach_all_behaviors_stop_motors(controller):
    """detach_all_behaviors(stop_motors=True) zeroes motors."""
    ag = controller.agents[0]
    ag.attach_behavior(constant_forward)
    ag.left_motor = 1.0
    ag.right_motor = 1.0
    controller.apply_changes()

    ag.detach_all_behaviors(stop_motors=True)
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.left_motor), 0.0, decimal=2)


# ---------------------------------------------------------------------------
# Weight and interval
# ---------------------------------------------------------------------------

def test_attach_behavior_with_weight(controller):
    """attach_behavior(fn, weight=W) sets the behavior weight."""
    ag = controller.agents[0]
    ag.attach_behavior(constant_forward, weight=0.5)
    # _behaviors stores [fn, interval, weight]
    beh = ag.behavior_handler._behaviors['constant_forward']
    assert beh[2] == 0.5


def test_attach_behavior_with_interval(controller):
    """attach_behavior(fn, interval=N) sets the execution interval."""
    ag = controller.agents[0]
    ag.attach_behavior(constant_forward, interval=10)
    beh = ag.behavior_handler._behaviors['constant_forward']
    assert beh[1] == 10


def test_change_behavior_weight(controller):
    """change_behavior_weight updates the weight of an existing behavior."""
    ag = controller.agents[0]
    ag.attach_behavior(constant_forward, weight=1.0)
    ag.change_behavior_weight(constant_forward, new_weight=0.2)
    beh = ag.behavior_handler._behaviors['constant_forward']
    assert beh[2] == 0.2


# ---------------------------------------------------------------------------
# Behavior execution (requires running simulation)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_no_behavior_zero_motors(running_controller):
    """Without a behavior, motors remain at zero."""
    controller = running_controller
    ag = controller.agents[0]

    controller.step()

    assert ag.left_motor == 0.0
    assert ag.right_motor == 0.0


@pytest.mark.slow
def test_behavior_produces_motion(running_controller):
    """An attached behavior causes the agent to move after stepping."""
    controller = running_controller
    ag = controller.agents[0]
    pos_before = (float(ag.x_position), float(ag.y_position))

    ag.attach_behavior(constant_forward)

    # Step enough times for force → momentum → position change
    controller.step()
    controller.step()
    for _ in range(NUM_STEPS):
        controller.step()

    assert ag.left_motor == 1.0
    assert ag.right_motor == 1.0

    pos_after = (float(ag.x_position), float(ag.y_position))
    assert pos_before != pos_after


def test_multiple_behaviors_weighted_average(controller):
    """Multiple behaviors are combined as a weighted average of motor outputs."""
    ag = controller.agents[0]

    def go_left(agent):
        return 0.0, 1.0

    def go_right(agent):
        return 1.0, 0.0

    ag.attach_behavior(go_left, weight=1.0)
    ag.attach_behavior(go_right, weight=1.0)

    # Call behave() directly — this computes the weighted average and sets motor
    ag.behavior_handler.behave(ag, time=1)
    controller.apply_changes()

    # Motors should be the weighted average: (0.5, 0.5)
    np.testing.assert_almost_equal(float(ag.left_motor), 0.5, decimal=2)
    np.testing.assert_almost_equal(float(ag.right_motor), 0.5, decimal=2)
