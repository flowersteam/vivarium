import pytest

from vivarium.controllers.vivarium_controller import VivariumController
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient


NUM_STEPS = 4


# =============================================================================
# set_subtype_labels tests
#
# scene_3 has subtype_labels: ['agent', 'resource', 'obstacle']
# and includes agents (braitenberg), objects, consumption, and spawn controllers.
# =============================================================================

SESSION3_LABELS  = ['agent', 'resource', 'obstacle']
NEW_LABELS_FULL  = ['robot', 'food', 'wall']
NEW_LABELS_PARTIAL = ['robot', 'food']          # 2 out of 3


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_set_subtype_labels_disconnected_raises():
    """Calling set_subtype_labels on a disconnected controller raises RuntimeError."""
    controller = VivariumController()
    with pytest.raises(RuntimeError, match="Not connected"):
        controller.set_subtype_labels(['a', 'b', 'c'])


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_set_subtype_labels_too_many_raises(scene_name, vivarium_controller_from_config):
    """Providing more labels than subtypes raises ValueError."""
    controller = vivarium_controller_from_config(scene_name)
    with pytest.raises(ValueError, match="at most"):
        controller.set_subtype_labels(SESSION3_LABELS + ['extra'])


# ---------------------------------------------------------------------------
# Basic rename behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scene_name', ['session_3'])
def test_set_subtype_labels_full_rename(scene_name, vivarium_controller_from_config):
    """Full rename updates controller.subtypes to the new list."""
    controller = vivarium_controller_from_config(scene_name)
    assert controller.subtypes == SESSION3_LABELS

    controller.set_subtype_labels(NEW_LABELS_FULL)
    assert controller.subtypes == NEW_LABELS_FULL


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_set_subtype_labels_partial_rename(scene_name, vivarium_controller_from_config):
    """Partial rename: user labels first, original labels kept for the rest."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_PARTIAL)
    assert controller.subtypes == ['robot', 'food', 'obstacle']


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_set_subtype_labels_single(scene_name, vivarium_controller_from_config):
    """Renaming only the first label preserves the other two unchanged."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(['robot'])
    assert controller.subtypes == ['robot', 'resource', 'obstacle']


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_set_subtype_labels_is_in_place(scene_name, vivarium_controller_from_config):
    """set_subtype_labels mutates the list in-place (same object identity)."""
    controller = vivarium_controller_from_config(scene_name)
    original_id = id(controller.subtypes)
    controller.set_subtype_labels(NEW_LABELS_FULL)
    assert id(controller.subtypes) == original_id


# ---------------------------------------------------------------------------
# Shared list identity
# Guarantee that all controllers that need subtype labels share the
# exact same Python list object, so in-place mutation reaches them all.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scene_name', ['session_3'])
def test_subtypes_shared_with_entity_controllers(scene_name, vivarium_controller_from_config):
    """All agent and object EntityControllers share the same list as controller.subtypes."""
    controller = vivarium_controller_from_config(scene_name)

    for ag in controller.agents:
        assert ag._subtype_labels is controller.subtypes
    for obj in controller.objects:
        assert obj._subtype_labels is controller.subtypes

    assert controller.agents.subtype_labels is controller.subtypes
    assert controller.objects.subtype_labels is controller.subtypes


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_subtypes_shared_with_consumption_and_spawn(scene_name, vivarium_controller_from_config):
    """Consumption and spawn controllers share the same list as controller.subtypes."""
    controller = vivarium_controller_from_config(scene_name)

    assert controller.consumption._subtype_labels is controller.subtypes
    assert controller.spawn._subtype_labels is controller.subtypes

    for single in controller.consumption._single_consumption_controllers.values():
        assert single._subtype_labels is controller.subtypes
    for single in controller.spawn._single_spawn_controllers.values():
        assert single._subtype_labels is controller.subtypes


# ---------------------------------------------------------------------------
# Propagation to entity controllers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scene_name', ['session_3'])
def test_entity_subtype_getter_after_rename(scene_name, vivarium_controller_from_config):
    """agent.subtype returns the new label for the same underlying integer index."""
    controller = vivarium_controller_from_config(scene_name)
    ag = controller.agents[0]
    label_before = ag.subtype                                         # e.g. 'agent'
    expected = NEW_LABELS_FULL[SESSION3_LABELS.index(label_before)]

    controller.set_subtype_labels(NEW_LABELS_FULL)

    assert ag.subtype == expected                                      # e.g. 'robot'


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_entity_subtype_setter_accepts_new_label(scene_name, vivarium_controller_from_config):
    """agent.subtype = new_label works after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)
    ag = controller.agents[0]

    ag.subtype = 'food'
    controller.apply_changes()

    assert ag.subtype == 'food'


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_entity_subtype_setter_rejects_old_label(scene_name, vivarium_controller_from_config):
    """agent.subtype = old_label raises ValueError after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)
    ag = controller.agents[0]

    with pytest.raises(ValueError):
        ag.subtype = 'agent'                                          # old label


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_proximeters_accepts_new_label(scene_name, vivarium_controller_from_config):
    """proximeters(sensed_entities=[new_label]) succeeds after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)

    result = controller.agents[0].proximeters(sensed_entities=['food'])
    assert isinstance(result, list) and len(result) == 2


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_proximeters_rejects_old_label(scene_name, vivarium_controller_from_config):
    """proximeters(sensed_entities=[old_label]) raises AssertionError after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)

    with pytest.raises(AssertionError):
        controller.agents[0].proximeters(sensed_entities=['resource'])  # old label


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_behaviors_sensed_getter_after_rename(scene_name, vivarium_controller_from_config):
    """behaviors[i].sensed returns new label strings for the same mask positions."""
    controller = vivarium_controller_from_config(scene_name)
    ag = controller.agents[0]

    # Set sensed to the first subtype (index 0) using the old label, then flush.
    ag.behaviors[0].sensed = [SESSION3_LABELS[0]]
    controller.apply_changes()
    assert SESSION3_LABELS[0] in ag.behaviors[0].sensed

    controller.set_subtype_labels(NEW_LABELS_FULL)

    # The mask is unchanged (still 1 at index 0); only the label string changes.
    assert NEW_LABELS_FULL[0] in ag.behaviors[0].sensed
    assert SESSION3_LABELS[0] not in ag.behaviors[0].sensed


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_behaviors_sensed_setter_accepts_new_labels(scene_name, vivarium_controller_from_config):
    """behaviors[i].sensed = [new_labels] builds the correct mask after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)
    ag = controller.agents[0]

    ag.behaviors[0].sensed = ['robot', 'food']
    controller.apply_changes()

    sensed = ag.behaviors[0].sensed
    assert 'robot' in sensed and 'food' in sensed


# ---------------------------------------------------------------------------
# Propagation to consumption controller
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scene_name', ['session_3'])
def test_consumption_getters_after_rename(scene_name, vivarium_controller_from_config):
    """consumption.source_subtype / target_subtype return new labels after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    src_before = controller.consumption.source_subtype
    tgt_before = controller.consumption.target_subtype

    controller.set_subtype_labels(NEW_LABELS_FULL)

    assert controller.consumption.source_subtype == NEW_LABELS_FULL[SESSION3_LABELS.index(src_before)]
    assert controller.consumption.target_subtype == NEW_LABELS_FULL[SESSION3_LABELS.index(tgt_before)]


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_consumption_setter_accepts_new_label(scene_name, vivarium_controller_from_config):
    """consumption source/target subtype setters accept new labels after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)

    controller.consumption.source_subtype = 'robot'
    controller.consumption.target_subtype = 'food'
    controller.apply_changes()

    assert controller.consumption.source_subtype == 'robot'
    assert controller.consumption.target_subtype == 'food'


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_consumption_setter_rejects_old_label(scene_name, vivarium_controller_from_config):
    """consumption.source_subtype = old_label raises ValueError after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)

    with pytest.raises(ValueError):
        controller.consumption.source_subtype = 'agent'               # old label


# ---------------------------------------------------------------------------
# Propagation to spawn controller
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scene_name', ['session_3'])
def test_spawn_getter_after_rename(scene_name, vivarium_controller_from_config):
    """spawn.subtype returns the new label after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    spawn_before = controller.spawn.subtype

    controller.set_subtype_labels(NEW_LABELS_FULL)

    assert controller.spawn.subtype == NEW_LABELS_FULL[SESSION3_LABELS.index(spawn_before)]


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_spawn_setter_accepts_new_label(scene_name, vivarium_controller_from_config):
    """spawn.subtype accepts a new label after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)

    controller.spawn.subtype = 'food'
    controller.apply_changes()

    assert controller.spawn.subtype == 'food'


@pytest.mark.parametrize('scene_name', ['session_3'])
def test_spawn_setter_rejects_old_label(scene_name, vivarium_controller_from_config):
    """spawn.subtype = old_label raises ValueError after renaming."""
    controller = vivarium_controller_from_config(scene_name)
    controller.set_subtype_labels(NEW_LABELS_FULL)

    with pytest.raises(ValueError):
        controller.spawn.subtype = 'resource'                         # old label


# ---------------------------------------------------------------------------
# Integration: rename then run simulation
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('scene_name', ['session_3'])
def test_rename_then_run_simulation(scene_name, vivarium_controller_start_session):
    """After renaming, subtype-specific behaviors attach and run without errors."""
    controller = vivarium_controller_start_session(scene_name, overrides=["environment.kwargs.debug_mode=true"])
    controller.set_subtype_labels(NEW_LABELS_FULL)

    def obstacle_avoidance(agent):
        left, right = agent.proximeters(sensed_entities=['wall'])     # was 'obstacle'
        return 1 - right, 1 - left

    def foraging(agent):
        left, right = agent.proximeters(sensed_entities=['food'])     # was 'resource'
        return right, left

    for ag in controller.agents:
        ag.attach_behavior(obstacle_avoidance)
        ag.attach_behavior(foraging)

    controller.simulator.simulation_running = True
    for _ in range(NUM_STEPS):
        controller.step()


# ---------------------------------------------------------------------------
# Known limitations
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('client_fixture', ['grpc_client'])
def test_simulator_subtype_labels_diverges_after_apply_changes(client_fixture, request):
    """Known limitation: controller.simulator.subtype_labels reads the freshly
    deserialized controller_parameters from the server on every call, so after
    apply_changes() it returns the server's original labels, not the renamed ones.

    controller.subtypes (and all entity/consumption/spawn controllers) remain
    correctly updated — this is a divergence in one read-only accessor only.
    """
    client = request.getfixturevalue(client_fixture)('session_3')
    controller = VivariumController(client=client, start_controller_thread=False)

    controller.set_subtype_labels(NEW_LABELS_FULL)
    assert controller.subtypes == NEW_LABELS_FULL

    # apply_changes() triggers a round-trip that replaces client.controller_parameters
    # with a freshly deserialized object carrying the original server-side labels.
    controller.apply_changes()

    # controller.subtypes (the in-place mutated list) is unaffected.
    assert controller.subtypes == NEW_LABELS_FULL

    # But controller.simulator.subtype_labels reads from the new client.controller_parameters
    # and therefore sees the original server labels again.
    assert controller.simulator.subtype_labels == SESSION3_LABELS


@pytest.mark.slow
def test_two_controllers_have_independent_subtypes(grpc_server):
    """Known limitation: two VivariumController instances backed by different gRPC
    clients (the typical Panel UI + notebook scenario) have independent subtype
    label lists. Renaming on one has no effect on the other.

    Each SimulatorGRPCClient deserializes controller_parameters independently on
    construction, producing separate Python list objects. In-place mutation of one
    list does not touch the other.
    """
    address = grpc_server('session_3')      # single in-process server

    # Two separate clients — mirrors the Panel client and notebook client
    client1 = SimulatorGRPCClient(server=address)
    client2 = SimulatorGRPCClient(server=address)

    controller1 = VivariumController(client=client1, start_controller_thread=False)
    controller2 = VivariumController(client=client2, start_controller_thread=False)

    # Each controller holds its own list object (separate gRPC deserializations).
    assert controller1.subtypes is not controller2.subtypes

    # Renaming on controller1 does not affect controller2.
    controller1.set_subtype_labels(NEW_LABELS_FULL)
    assert controller1.subtypes == NEW_LABELS_FULL
    assert controller2.subtypes == SESSION3_LABELS   # unaffected

    client1.close()
    client2.close()
