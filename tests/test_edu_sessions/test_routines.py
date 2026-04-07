"""Tests for agent-level and controller-level routines."""

import pytest


# ---------------------------------------------------------------------------
# Agent routines
# ---------------------------------------------------------------------------

def test_attach_agent_routine(controller):
    """agent.attach_routine registers a routine."""
    ag = controller.agents[0]
    call_log = []

    def my_routine(agent):
        call_log.append(1)

    ag.attach_routine(my_routine)
    assert 'my_routine' in [r for r in ag.routine_handler._routines]


@pytest.mark.slow
def test_agent_routine_fires_on_step(running_controller):
    """An agent routine fires when the simulation steps."""
    controller = running_controller
    ag = controller.agents[0]
    call_log = []

    def track_step(agent):
        call_log.append(1)

    ag.attach_routine(track_step)
    for _ in range(3):
        controller.step()
    assert len(call_log) >= 3


@pytest.mark.slow
def test_agent_routine_with_interval(running_controller):
    """An agent routine with interval=N fires less often than interval=1."""
    controller = running_controller
    ag = controller.agents[0]
    call_log_every = []
    call_log_interval = []

    def track_every(agent):
        call_log_every.append(1)

    def track_interval(agent):
        call_log_interval.append(1)

    ag.attach_routine(track_every, interval=1)
    ag.attach_routine(track_interval, interval=3)
    for _ in range(6):
        controller.step()
    assert len(call_log_interval) < len(call_log_every)


def test_detach_agent_routine(controller):
    """agent.detach_routine removes a routine."""
    ag = controller.agents[0]

    def my_routine(agent):
        pass

    ag.attach_routine(my_routine)
    ag.detach_routine(my_routine)
    assert len(ag.routine_handler._routines) == 0


def test_detach_all_agent_routines(controller):
    """agent.detach_all_routines removes all routines."""
    ag = controller.agents[0]

    def r1(agent):
        pass

    def r2(agent):
        pass

    ag.attach_routine(r1)
    ag.attach_routine(r2)
    ag.detach_all_routines()
    assert len(ag.routine_handler._routines) == 0


# ---------------------------------------------------------------------------
# Controller routines
# ---------------------------------------------------------------------------

def test_attach_controller_routine(controller):
    """controller.attach_routine registers a routine on the controller."""
    call_log = []

    def my_ctrl_routine(ctrl):
        call_log.append(1)

    controller.attach_routine(my_ctrl_routine)
    assert 'my_ctrl_routine' in controller.routine_handler._routines


@pytest.mark.slow
def test_controller_routine_fires_on_step(running_controller):
    """A controller-level routine fires when the simulation steps."""
    controller = running_controller
    call_log = []

    def ctrl_track(ctrl):
        call_log.append(1)

    controller.attach_routine(ctrl_track)
    for _ in range(3):
        controller.step()
    assert len(call_log) >= 3


@pytest.mark.slow
def test_controller_routine_with_interval(running_controller):
    """A controller routine with interval=N fires less often than interval=1."""
    controller = running_controller
    call_log_every = []
    call_log_interval = []

    def ctrl_every(ctrl):
        call_log_every.append(1)

    def ctrl_interval(ctrl):
        call_log_interval.append(1)

    controller.attach_routine(ctrl_every, interval=1)
    controller.attach_routine(ctrl_interval, interval=3)
    for _ in range(6):
        controller.step()
    assert len(call_log_interval) < len(call_log_every)


@pytest.mark.slow
def test_controller_routine_accesses_agents(running_controller):
    """A controller routine can access and modify agents."""
    controller = running_controller

    def set_colors(ctrl):
        for ag in ctrl.agents:
            ag.color = 'red'

    controller.attach_routine(set_colors)
    controller.step()
    # No error means the routine successfully accessed agents


def test_detach_controller_routine(controller):
    """controller.detach_routine removes a controller routine."""
    def my_ctrl_routine(ctrl):
        pass

    controller.attach_routine(my_ctrl_routine)
    controller.detach_routine(my_ctrl_routine)
    assert len(controller.routine_handler._routines) == 0
