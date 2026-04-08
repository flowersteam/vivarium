"""Tests for entity property read/write: position, appearance, motion, physics."""

import numpy as np


# ---------------------------------------------------------------------------
# Agent position and orientation
# ---------------------------------------------------------------------------

def test_agent_position_read(controller):
    """agent.x_position and y_position are readable floats."""
    ag = controller.agents[0]
    assert isinstance(float(ag.x_position), float)
    assert isinstance(float(ag.y_position), float)


def test_agent_position_write(controller):
    """Setting x_position and y_position updates the agent's position."""
    ag = controller.agents[0]
    ag.x_position = 50.0
    ag.y_position = 40.0
    controller.apply_changes()
    assert float(ag.x_position) == 50.0
    assert float(ag.y_position) == 40.0


def test_agent_orientation_read_write(controller):
    """agent.orientation is readable and writable."""
    ag = controller.agents[0]
    ag.orientation = 1.57
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.orientation), 1.57, decimal=2)


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------

def test_agent_color_write(controller):
    """agent.color can be set to a color string."""
    ag = controller.agents[0]
    ag.color = 'pink'
    controller.apply_changes()
    # color is stored as RGB array — just verify no error on set + apply


def test_agent_diameter_read_write(controller):
    """agent.diameter is readable and writable."""
    ag = controller.agents[0]
    ag.diameter = 12.0
    controller.apply_changes()
    assert float(ag.diameter) == 12.0


def test_object_color_write(controller):
    """object.color can be set to a color string."""
    obj = controller.objects[0]
    obj.color = 'cyan'
    controller.apply_changes()


def test_object_diameter_read_write(controller):
    """object.diameter is readable and writable."""
    obj = controller.objects[0]
    obj.diameter = 8.0
    controller.apply_changes()
    assert float(obj.diameter) == 8.0


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

def test_agent_max_speed_read_write(controller):
    """agent.max_speed is readable and writable."""
    ag = controller.agents[0]
    ag.max_speed = 0.5
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.max_speed), 0.5, decimal=2)


def test_agent_motors_read_write(controller):
    """agent.left_motor and right_motor are readable and writable."""
    ag = controller.agents[0]
    ag.left_motor = 1.0
    ag.right_motor = 0.5
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.left_motor), 1.0, decimal=2)
    np.testing.assert_almost_equal(float(ag.right_motor), 0.5, decimal=2)


def test_agent_stop_motors(controller):
    """agent.stop_motors() sets both motors to zero."""
    ag = controller.agents[0]
    ag.left_motor = 1.0
    ag.right_motor = 1.0
    controller.apply_changes()
    ag.stop_motors()
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.left_motor), 0.0, decimal=2)
    np.testing.assert_almost_equal(float(ag.right_motor), 0.0, decimal=2)


# ---------------------------------------------------------------------------
# Physics (objects)
# ---------------------------------------------------------------------------

def test_object_mass_read_write(controller):
    """object.mass is readable and writable."""
    obj = controller.objects[0]
    obj.mass = 1000.0
    controller.apply_changes()
    assert float(obj.mass) == 1000.0


def test_object_friction_read_write(controller):
    """object.friction is readable and writable."""
    obj = controller.objects[0]
    obj.friction = 500.0
    controller.apply_changes()
    assert float(obj.friction) == 500.0


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------

def test_agent_exists_read(controller):
    """agent.exists is readable."""
    ag = controller.agents[0]
    assert ag.exists in (True, False)


def test_object_exists_write(controller):
    """object.exists can be toggled."""
    obj = controller.objects[0]
    obj.exists = False
    controller.apply_changes()
    assert not obj.exists

    obj.exists = True
    controller.apply_changes()
    assert obj.exists


# ---------------------------------------------------------------------------
# Proximeter configuration
# ---------------------------------------------------------------------------

def test_agent_proxs_dist_max_read_write(controller):
    """agent.proxs_dist_max is readable and writable."""
    ag = controller.agents[0]
    ag.proxs_dist_max = 10.0
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.proxs_dist_max), 10.0, decimal=1)


def test_agent_proxs_cos_min_read_write(controller):
    """agent.proxs_cos_min is readable and writable."""
    ag = controller.agents[0]
    ag.proxs_cos_min = 0.9
    controller.apply_changes()
    np.testing.assert_almost_equal(float(ag.proxs_cos_min), 0.9, decimal=1)


# ---------------------------------------------------------------------------
# Subtype
# ---------------------------------------------------------------------------

def test_agent_subtype_read(controller):
    """agent.subtype returns a string label."""
    ag = controller.agents[0]
    assert ag.subtype in controller.subtypes


def test_agent_subtype_write(controller):
    """agent.subtype can be set to a valid subtype label."""
    ag = controller.agents[0]
    new_subtype = controller.subtypes[1]
    ag.subtype = new_subtype
    controller.apply_changes()
    assert ag.subtype == new_subtype


def test_object_subtype_read(controller):
    """object.subtype returns a string label."""
    obj = controller.objects[0]
    assert obj.subtype in controller.subtypes
