"""Tests for proximeter sensing: basic and selective."""

import pytest



def test_proximeters_returns_two_values(controller):
    """agent.proximeters() returns a list of [left, right]."""
    ag = controller.agents[0]
    result = ag.proximeters()
    assert isinstance(result, list)
    assert len(result) == 2


def test_proximeters_values_are_numeric(controller):
    """Proximeter values are numeric (float-like)."""
    ag = controller.agents[0]
    left, right = ag.proximeters()
    assert isinstance(float(left), float)
    assert isinstance(float(right), float)


@pytest.mark.slow
def test_proximeters_selective_filters_by_subtype(running_controller):
    """Selective sensing produces different results than unfiltered sensing.

    Place an agent and an object near the test agent, step to compute the
    proximity map, then verify that filtering to one subtype gives a
    different result than unfiltered.
    """
    controller = running_controller
    ag = controller.agents[0]
    ag.x_position = 50.0
    ag.y_position = 50.0
    ag.proxs_dist_max = 40.0

    # Place another agent (subtype_1) nearby on the left
    other_agent = controller.agents[1]
    other_agent.x_position = 45.0
    other_agent.y_position = 50.0

    # Place an object (subtype_5) nearby on the right
    obj = controller.objects[0]
    obj.x_position = 55.0
    obj.y_position = 50.0

    # Step twice: first to apply position changes, second to recompute proximity map
    controller.step()
    controller.step()

    unfiltered = ag.proximeters()
    agents_only = ag.proximeters(sensed_entities=['subtype_1'])
    objects_only = ag.proximeters(sensed_entities=['subtype_5'])

    # Unfiltered should detect something (entities are close)
    assert unfiltered != [0.0, 0.0]
    # Filtering to one subtype should differ from unfiltered
    assert agents_only != unfiltered or objects_only != unfiltered


def test_proximeters_selective_multiple_subtypes(controller):
    """proximeters(sensed_entities=[l1, l2]) filters by multiple subtypes."""
    ag = controller.agents[0]
    result = ag.proximeters(sensed_entities=['subtype_1', 'subtype_5'])
    assert isinstance(result, list) and len(result) == 2


def test_proximeters_selective_invalid_subtype_raises(controller):
    """proximeters(sensed_entities=[invalid]) raises AssertionError."""
    ag = controller.agents[0]
    with pytest.raises(AssertionError):
        ag.proximeters(sensed_entities=['nonexistent_subtype'])
