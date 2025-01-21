import numpy as np
import param
from param import Parameterized
from jax_md.rigid_body import monomer

from vivarium.simulator.simulator_states import StateType

mass = monomer.mass()
mass_center = float(mass.center[0])
mass_orientation = float(mass.orientation[0])


class Config(Parameterized):
    """Base class for configuration objects"""

    def to_dict(self, params=None):
        """Return a dictionary with the configuration parameters

        :param params: params, defaults to None
        :return: dictionary with the configuration parameters
        """
        d = self.param.values()
        del d["name"]
        if params is not None:
            return {p: d[p] for p in params}
        else:
            return d

    def param_names(self):
        """Return the names of the configuration parameters

        :return: list of parameter names
        """
        return list(self.to_dict().keys())

    def json(self):
        """Return a JSON representation of the configuration

        :return: JSON representation of the configuration
        """
        return self.param.serialize_parameters(subset=self.param_names())


class AgentConfig(Config):
    """Configuration class for agents"""

    idx = param.Integer()
    # ent_sensedtype = param.Integer()
    x_position = param.Number(0.0)
    y_position = param.Number(0.0)
    orientation = param.Number(0.0)
    mass_center = param.Number(mass_center)
    mass_orientation = param.Number(mass_orientation)
    # TODO : Change the behaviors to a list of objects in the future
    behavior = param.Array(np.array([0.0]))
    left_motor = param.Number(0.0, bounds=(0.0, 1.0))
    right_motor = param.Number(0.0, bounds=(0.0, 1.0))
    # TODO : Will be problems here if proximeters if non occlusion mode (as many proximeter values as neighbors), except if we only consider the non occlusion case where the sensors information is just the sensor of closest entity
    left_prox = param.Number(0.0, bounds=(0.0, 1.0))
    right_prox = param.Number(0.0, bounds=(0.0, 1.0))
    prox_sensed_ent_type = param.Array(np.array([0]))
    prox_sensed_ent_idx = param.Array(np.array([0]))
    proximity_map_dist = param.Array(np.array([0.0]))
    proximity_map_theta = param.Array(np.array([0.0]))
    params = param.Array(np.array([0.0]))
    sensed = param.Array(np.array([0.0]))
    wheel_diameter = param.Number(2.0)
    diameter = param.Number(5.0)
    speed_mul = param.Number(1.0)
    max_speed = param.Number(10.0)
    theta_mul = param.Number(1.0)
    proxs_dist_max = param.Number(100.0, bounds=(0, None))
    proxs_cos_min = param.Number(0.0, bounds=(-1.0, 1.0))
    color = param.Color("blue")
    friction = param.Number(1e-1)
    exists = param.Boolean(True)
    subtype = param.Integer(0)

    def __init__(self, **params):
        super().__init__(**params)


class ObjectConfig(Config):
    """Configuration class for objects"""

    idx = param.Integer()
    # ent_sensedtype = param.Integer()
    x_position = param.Number(0.0)
    y_position = param.Number(0.0)
    orientation = param.Number(0.0)
    mass_center = param.Number(mass_center)
    mass_orientation = param.Number(mass_orientation)
    diameter = param.Number(5.0)
    color = param.Color("red")
    friction = param.Number(0.1)
    exists = param.Boolean(True)
    subtype = param.Integer(0)

    def __init__(self, **params):
        super().__init__(**params)


class SimulatorConfig(Config):
    """Configuration class for the simulator"""

    idx = param.Integer(0, constant=True)
    time = param.Integer(0)
    box_size = param.Number(100.0, bounds=(0, None))
    max_agents = param.Integer(10)
    max_objects = param.Integer(2)
    num_steps_lax = param.Integer(4)
    dt = param.Number(0.1)
    freq = param.Number(40.0, allow_None=True)
    neighbor_radius = param.Number(100.0, bounds=(0, None))
    to_jit = param.Boolean(True)
    use_fori_loop = param.Boolean(False)
    collision_eps = param.Number(0.1)
    collision_alpha = param.Number(0.5)

    def __init__(self, **params):
        super().__init__(**params)


# TODO : Check how this works but weird because it seems like SimulatorConfig is 2 in StateTypes and 0 here
config_to_stype = {
    SimulatorConfig: StateType.SIMULATOR,
    AgentConfig: StateType.AGENT,
    ObjectConfig: StateType.OBJECT,
}
stype_to_config = {
    stype: config_class for config_class, stype in config_to_stype.items()
}
