"""Tests for Logger: add, get, clear on agents and controller."""


# ---------------------------------------------------------------------------
# Agent logger
# ---------------------------------------------------------------------------

def test_agent_logger_add_and_get(controller):
    """agent.logger.add stores values retrievable by get."""
    ag = controller.agents[0]
    ag.logger.add('energy', 0.5)
    ag.logger.add('energy', 0.8)
    result = ag.logger.get('energy')
    assert result == [0.5, 0.8]


def test_agent_logger_get_missing_returns_empty(controller):
    """agent.logger.get for a nonexistent field returns an empty list."""
    ag = controller.agents[0]
    result = ag.logger.get('nonexistent')
    assert result == []


def test_agent_logger_clear_field(controller):
    """agent.logger.clear(field) clears only that field."""
    ag = controller.agents[0]
    ag.logger.add('x', 1)
    ag.logger.add('y', 2)
    ag.logger.clear('x')
    assert ag.logger.get('x') == []
    assert ag.logger.get('y') == [2]


def test_agent_logger_clear_all(controller):
    """agent.logger.clear() clears all fields."""
    ag = controller.agents[0]
    ag.logger.add('x', 1)
    ag.logger.add('y', 2)
    ag.logger.clear()
    assert ag.logger.get('x') == []
    assert ag.logger.get('y') == []


def test_agent_logger_in_routine(running_controller):
    """A routine can log values via agent.logger.add."""
    controller = running_controller
    ag = controller.agents[0]

    def log_position(agent):
        agent.logger.add('x', float(agent.x_position))

    ag.attach_routine(log_position)
    for _ in range(3):
        controller.step()
    assert len(ag.logger.get('x')) >= 3


# ---------------------------------------------------------------------------
# Controller logger
# ---------------------------------------------------------------------------

def test_controller_logger_add_and_get(controller):
    """controller.logger.add stores values retrievable by get."""
    controller.logger.add('count', 10)
    controller.logger.add('count', 20)
    assert controller.logger.get('count') == [10, 20]


def test_controller_logger_in_routine(running_controller):
    """A controller routine can log values via controller.logger.add."""
    controller = running_controller

    def log_agent_count(ctrl):
        ctrl.logger.add('n_agents', len(ctrl.agents))

    controller.attach_routine(log_agent_count)
    for _ in range(3):
        controller.step()
    assert len(controller.logger.get('n_agents')) >= 3


# ---------------------------------------------------------------------------
# Logger independence
# ---------------------------------------------------------------------------

def test_agent_loggers_are_independent(controller):
    """Each agent has its own logger instance."""
    ag0 = controller.agents[0]
    ag1 = controller.agents[1]
    ag0.logger.add('x', 1)
    assert ag1.logger.get('x') == []
