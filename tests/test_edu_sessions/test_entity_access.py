"""Tests for entity access patterns: indexing, slicing, iteration, length."""


def test_agents_indexing(controller):
    """controller.agents[i] returns an agent controller."""
    ag = controller.agents[0]
    assert ag is not None
    assert hasattr(ag, 'proximeters')


def test_agents_negative_indexing(controller):
    """controller.agents[-1] returns the last agent."""
    last = controller.agents[-1]
    last_explicit = controller.agents[len(controller.agents) - 1]
    # Both refer to the same entity — same position
    assert float(last.x_position) == float(last_explicit.x_position)
    assert float(last.y_position) == float(last_explicit.y_position)


def test_objects_indexing(controller):
    """controller.objects[i] returns an object controller."""
    obj = controller.objects[0]
    assert obj is not None
    assert hasattr(obj, 'exists')


def test_agents_slicing(controller):
    """controller.agents[0:4] returns a list of agent controllers."""
    agents = controller.agents[0:4]
    assert isinstance(agents, list)
    assert len(agents) == 4
    for ag in agents:
        assert hasattr(ag, 'proximeters')


def test_objects_slicing(controller):
    """controller.objects[0:8] returns a list of object controllers."""
    objects = controller.objects[0:8]
    assert isinstance(objects, list)
    assert len(objects) == 8


def test_agents_iteration(controller):
    """Iterating over controller.agents yields all agents."""
    agents = list(controller.agents)
    assert len(agents) == len(controller.agents)
    for ag in agents:
        assert hasattr(ag, 'proximeters')


def test_objects_iteration(controller):
    """Iterating over controller.objects yields all objects."""
    objects = list(controller.objects)
    assert len(objects) == len(controller.objects)


def test_agents_len(controller):
    """len(controller.agents) returns the number of agents (miniproject: 12)."""
    assert len(controller.agents) == 12


def test_objects_len(controller):
    """len(controller.objects) returns the number of objects (miniproject: 32)."""
    assert len(controller.objects) == 32


def test_subtypes_list(controller):
    """controller.subtypes returns the list of subtype labels."""
    subtypes = controller.subtypes
    assert isinstance(subtypes, list)
    assert len(subtypes) == 8
    assert subtypes[0] == 'subtype_1'
