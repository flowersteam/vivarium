"""Tests for agent.internal namespace (client-side custom state)."""


def test_internal_set_and_get(controller):
    """agent.internal.<attr> can be set and read back."""
    ag = controller.agents[0]
    ag.internal.energy_level = 0.5
    assert ag.internal.energy_level == 0.5


def test_internal_arbitrary_attributes(controller):
    """agent.internal accepts any attribute name."""
    ag = controller.agents[0]
    ag.internal.foo = 'bar'
    ag.internal.count = 42
    assert ag.internal.foo == 'bar'
    assert ag.internal.count == 42


def test_internal_update_in_routine(running_controller):
    """A routine can read and update agent.internal state."""
    controller = running_controller
    ag = controller.agents[0]
    ag.internal.energy_level = 1.0

    def drain_energy(agent):
        agent.internal.energy_level -= 0.1

    ag.attach_routine(drain_energy)
    for _ in range(5):
        controller.step()
    assert ag.internal.energy_level < 1.0


def test_internal_independent_per_agent(controller):
    """Each agent has its own internal namespace."""
    ag0 = controller.agents[0]
    ag1 = controller.agents[1]
    ag0.internal.x = 1
    ag1.internal.x = 2
    assert ag0.internal.x == 1
    assert ag1.internal.x == 2


def test_internal_not_synced_to_server(controller):
    """agent.internal is client-side only — apply_changes does not crash."""
    ag = controller.agents[0]
    ag.internal.custom = 'data'
    controller.apply_changes()
    assert ag.internal.custom == 'data'
