# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: simulator.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fsimulator.proto\x12\tsimulator\x1a\x1bgoogle/protobuf/empty.proto\"\x14\n\x04Name\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x14\n\x04Time\x12\x0c\n\x04time\x18\x01 \x01(\x05\"\xcd\x01\n\x10SimulationConfig\x12\x10\n\x08\x62ox_size\x18\x01 \x01(\x02\x12\x0f\n\x07map_dim\x18\x02 \x01(\x05\x12\x15\n\rnum_steps_lax\x18\x03 \x01(\x05\x12\x15\n\rnum_lax_loops\x18\x04 \x01(\x05\x12\n\n\x02\x64t\x18\x05 \x01(\x02\x12\x0c\n\x04\x66req\x18\x06 \x01(\x02\x12\x0e\n\x06to_jit\x18\x07 \x01(\x08\x12\x10\n\x08n_agents\x18\x08 \x01(\x05\x12,\n\x10\x65ntity_behaviors\x18\t \x01(\x0b\x32\x12.simulator.NDArray\"0\n\x1aSimulationConfigSerialized\x12\x12\n\nserialized\x18\x01 \x01(\t\"u\n\x0eSerializedDict\x12\x17\n\x0fserialized_dict\x18\x01 \x01(\t\x12\x1c\n\x14has_entity_behaviors\x18\x02 \x01(\x08\x12,\n\x10\x65ntity_behaviors\x18\x03 \x01(\x0b\x32\x12.simulator.NDArray\"\xa8\x01\n\x0b\x41gentConfig\x12\x16\n\x0ewheel_diameter\x18\x01 \x01(\x02\x12\x13\n\x0b\x62\x61se_length\x18\x02 \x01(\x02\x12\x11\n\tspeed_mul\x18\x03 \x01(\x02\x12\x11\n\ttheta_mul\x18\x04 \x01(\x02\x12\x17\n\x0fneighbor_radius\x18\x05 \x01(\x02\x12\x16\n\x0eproxs_dist_max\x18\x06 \x01(\x02\x12\x15\n\rproxs_cos_min\x18\x07 \x01(\x02\"+\n\x15\x41gentConfigSerialized\x12\x12\n\nserialized\x18\x01 \x01(\t\"$\n\x10PopulationConfig\x12\x10\n\x08n_agents\x18\x01 \x01(\x05\"0\n\x1aPopulationConfigSerialized\x12\x12\n\nserialized\x18\x01 \x01(\t\" \n\x08Position\x12\t\n\x01x\x18\x01 \x03(\x02\x12\t\n\x01y\x18\x02 \x03(\x02\"?\n\x05State\x12&\n\tpositions\x18\x01 \x01(\x0b\x32\x13.simulator.Position\x12\x0e\n\x06thetas\x18\x02 \x03(\x02\"\x1a\n\x07NDArray\x12\x0f\n\x07ndarray\x18\x01 \x01(\x0c\"\xb4\x01\n\x0bStateArrays\x12%\n\tpositions\x18\x01 \x01(\x0b\x32\x12.simulator.NDArray\x12\"\n\x06thetas\x18\x02 \x01(\x0b\x32\x12.simulator.NDArray\x12!\n\x05proxs\x18\x03 \x01(\x0b\x32\x12.simulator.NDArray\x12\"\n\x06motors\x18\x04 \x01(\x0b\x32\x12.simulator.NDArray\x12\x13\n\x0b\x65ntity_type\x18\x05 \x01(\x05\"$\n\x0eIsStartedState\x12\x12\n\nis_started\x18\x01 \x01(\x08\x32\xdb\x0c\n\x0fSimulatorServer\x12S\n\x1aGetSimulationConfigMessage\x12\x16.google.protobuf.Empty\x1a\x1b.simulator.SimulationConfig\"\x00\x12`\n\x1dGetSimulationConfigSerialized\x12\x16.google.protobuf.Empty\x1a%.simulator.SimulationConfigSerialized\"\x00\x12\x42\n\x12GetRecordedChanges\x12\x0f.simulator.Name\x1a\x19.simulator.SerializedDict\"\x00\x12I\n\x15GetAgentConfigMessage\x12\x16.google.protobuf.Empty\x1a\x16.simulator.AgentConfig\"\x00\x12V\n\x18GetAgentConfigSerialized\x12\x16.google.protobuf.Empty\x1a .simulator.AgentConfigSerialized\"\x00\x12S\n\x1aGetPopulationConfigMessage\x12\x16.google.protobuf.Empty\x1a\x1b.simulator.PopulationConfig\"\x00\x12`\n\x1dGetPopulationConfigSerialized\x12\x16.google.protobuf.Empty\x1a%.simulator.PopulationConfigSerialized\"\x00\x12`\n\x1dSetSimulationConfigSerialized\x12%.simulator.SimulationConfigSerialized\x1a\x16.google.protobuf.Empty\"\x00\x12L\n\x13SetSimulationConfig\x12\x1b.simulator.SimulationConfig\x1a\x16.google.protobuf.Empty\"\x00\x12L\n\x13SetPopulationConfig\x12\x1b.simulator.PopulationConfig\x1a\x16.google.protobuf.Empty\"\x00\x12`\n\x1dSetPopulationConfigSerialized\x12%.simulator.PopulationConfigSerialized\x1a\x16.google.protobuf.Empty\"\x00\x12=\n\x0fGetStateMessage\x12\x16.google.protobuf.Empty\x1a\x10.simulator.State\"\x00\x12\x42\n\x0eGetStateArrays\x12\x16.google.protobuf.Empty\x1a\x16.simulator.StateArrays\"\x00\x12@\n\tIsStarted\x12\x16.google.protobuf.Empty\x1a\x19.simulator.IsStartedState\"\x00\x12\x39\n\x05Start\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12\x38\n\x04Stop\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12\x44\n\x10UpdateNeighborFn\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12H\n\x14UpdateStateNeighbors\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12H\n\x14UpdateFunctionUpdate\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12\x43\n\x0fUpdateBehaviors\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty\"\x00\x12:\n\rGetChangeTime\x12\x16.google.protobuf.Empty\x1a\x0f.simulator.Time\"\x00\x42\x34\n\x1aio.grpc.examples.simulatorB\x0eSimulatorProtoP\x01\xa2\x02\x03SIMb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'simulator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\032io.grpc.examples.simulatorB\016SimulatorProtoP\001\242\002\003SIM'
  _NAME._serialized_start=59
  _NAME._serialized_end=79
  _TIME._serialized_start=81
  _TIME._serialized_end=101
  _SIMULATIONCONFIG._serialized_start=104
  _SIMULATIONCONFIG._serialized_end=309
  _SIMULATIONCONFIGSERIALIZED._serialized_start=311
  _SIMULATIONCONFIGSERIALIZED._serialized_end=359
  _SERIALIZEDDICT._serialized_start=361
  _SERIALIZEDDICT._serialized_end=478
  _AGENTCONFIG._serialized_start=481
  _AGENTCONFIG._serialized_end=649
  _AGENTCONFIGSERIALIZED._serialized_start=651
  _AGENTCONFIGSERIALIZED._serialized_end=694
  _POPULATIONCONFIG._serialized_start=696
  _POPULATIONCONFIG._serialized_end=732
  _POPULATIONCONFIGSERIALIZED._serialized_start=734
  _POPULATIONCONFIGSERIALIZED._serialized_end=782
  _POSITION._serialized_start=784
  _POSITION._serialized_end=816
  _STATE._serialized_start=818
  _STATE._serialized_end=881
  _NDARRAY._serialized_start=883
  _NDARRAY._serialized_end=909
  _STATEARRAYS._serialized_start=912
  _STATEARRAYS._serialized_end=1092
  _ISSTARTEDSTATE._serialized_start=1094
  _ISSTARTEDSTATE._serialized_end=1130
  _SIMULATORSERVER._serialized_start=1133
  _SIMULATORSERVER._serialized_end=2760
# @@protoc_insertion_point(module_scope)
