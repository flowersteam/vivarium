"""Tests for consumption and spawn multi-slot configuration and has_consumed."""

import numpy as np


# ---------------------------------------------------------------------------
# Consumption multi-slot
# ---------------------------------------------------------------------------

def test_consumption_slot_source_target_read(controller):
    """consumption.slot_1.source_subtype and target_subtype are readable."""
    slot = controller.consumption.slot_1
    assert slot.source_subtype in controller.subtypes
    assert slot.target_subtype in controller.subtypes


def test_consumption_slot_source_target_write(controller):
    """consumption.slot_1.source_subtype and target_subtype are writable."""
    slot = controller.consumption.slot_1
    slot.source_subtype = 'subtype_2'
    slot.target_subtype = 'subtype_3'
    controller.apply_changes()
    assert slot.source_subtype == 'subtype_2'
    assert slot.target_subtype == 'subtype_3'


def test_consumption_slot_range_read_write(controller):
    """consumption.slot_1.range is readable and writable."""
    slot = controller.consumption.slot_1
    slot.range = 2.0
    controller.apply_changes()
    np.testing.assert_almost_equal(float(slot.range), 2.0, decimal=1)


def test_consumption_slot_start_toggle(controller):
    """consumption.slot_1.start can be toggled on and off."""
    slot = controller.consumption.slot_1
    slot.start = True
    controller.apply_changes()
    assert slot.start is True

    slot.start = False
    controller.apply_changes()
    assert slot.start is False


def test_consumption_multiple_slots_independent(controller):
    """slot_1 and slot_2 can be configured independently."""
    controller.consumption.slot_1.source_subtype = 'subtype_1'
    controller.consumption.slot_1.target_subtype = 'subtype_5'
    controller.consumption.slot_2.source_subtype = 'subtype_2'
    controller.consumption.slot_2.target_subtype = 'subtype_3'
    controller.apply_changes()

    assert controller.consumption.slot_1.source_subtype == 'subtype_1'
    assert controller.consumption.slot_2.source_subtype == 'subtype_2'


# ---------------------------------------------------------------------------
# Spawn multi-slot
# ---------------------------------------------------------------------------

def test_spawn_slot_subtype_read_write(controller):
    """spawn.slot_1.subtype is readable and writable."""
    slot = controller.spawn.slot_1
    slot.subtype = 'subtype_5'
    controller.apply_changes()
    assert slot.subtype == 'subtype_5'


def test_spawn_slot_period_read(controller):
    """spawn.slot_1.period is readable."""
    slot = controller.spawn.slot_1
    period = slot.period
    assert period > 0


def test_spawn_slot_period_write(controller):
    """spawn.slot_1.period is writable."""
    slot = controller.spawn.slot_1
    slot.period = 100
    controller.apply_changes()


def test_spawn_slot_start_read_write(controller):
    """spawn.slot_1.start can be toggled on and off."""
    slot = controller.spawn.slot_1
    slot.start = True
    controller.apply_changes()
    assert slot.start is True

    slot.start = False
    controller.apply_changes()
    assert slot.start is False


def test_spawn_slot_position_range_read_write(controller):
    """spawn.slot_1.position_range is readable and writable."""
    slot = controller.spawn.slot_1
    slot.position_range = (10, 50, 20, 80)
    controller.apply_changes()
    result = slot.position_range
    assert tuple(int(v) for v in result) == (10, 50, 20, 80)


def test_spawn_multiple_slots_independent(controller):
    """slot_1 and slot_2 can be configured independently via subtype."""
    controller.spawn.slot_1.subtype = 'subtype_1'
    controller.spawn.slot_2.subtype = 'subtype_5'
    controller.apply_changes()

    assert controller.spawn.slot_1.subtype == 'subtype_1'
    assert controller.spawn.slot_2.subtype == 'subtype_5'


# ---------------------------------------------------------------------------
# has_consumed
# ---------------------------------------------------------------------------

def test_has_consumed_zero_without_consumption(controller):
    """agent.has_consumed() returns 0 when no consumption is active."""
    ag = controller.agents[0]
    result = ag.has_consumed()
    assert float(result) == 0.0
