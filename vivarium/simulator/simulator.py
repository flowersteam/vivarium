import os
import time
import math
import pickle
import logging
import threading
import datetime

from functools import partial
from contextlib import contextmanager

import jax
import jax.numpy as jnp

from vivarium.controllers import converters
from vivarium.utils.scene_configs import load_scene_config
from vivarium.simulator.simulator_states import SimState, SimulatorState
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    SelectiveSensorsEnv,
    init_state,
)
from vivarium.environments.braitenberg.selective_sensing.selective_sensing_env import (
    State as EnvState,
)

lg = logging.getLogger(__name__)


class Simulator:
    def __init__(
        self,
        env,
        env_state,
        scene_name="scene",
        num_steps_lax=4,
        update_freq=-1,
        jit_step=True,
        use_fori_loop=True,
        seed=0,
    ):
        self.env = env
        assert isinstance(
            self.env, SelectiveSensorsEnv
        ), "You have to use an environment with selective sensors within the simulator"
        assert (
            self.env.occlusion
        ), "You have to use an environment with occlusion sensors within the simulator"

        self.scene_name = scene_name

        # First initialize fields in the class because they will be used to define the simulator state below
        self.key = jax.random.PRNGKey(seed)
        self.num_steps_lax = num_steps_lax
        self.freq = update_freq
        self.jit_step = jit_step
        self.use_fori_loop = use_fori_loop
        self.ent_sub_types_and_num = (
            env_state.ent_sub_types
        )  # information about entities sub types in a dictionary
        self.ent_sub_types = self.process_ent_sub_types(self.ent_sub_types_and_num)

        # transform the env state (only used in env class) into a simulator state with a simulator state (used only in client server communication)
        self.state = self.env_to_sim_state(env_state)

        # Attributes to start or stop the simulation
        self._is_started = False
        self._to_stop = False

        # Attributes to record simulation
        self.recording = False
        self.records = None
        self.saving_dir = None

        # Do a first step to initialize the momentum of the state
        self.step()
        lg.info("Simulator initialized")

    def load_state(self, state, env):
        """Load a state in the simulator

        :param state: state to load
        """
        lg.info("Loading a new state")

        self.__init__(
            env=env,
            env_state=state,
            num_steps_lax=self.num_steps_lax,
            update_freq=self.freq,
            jit_step=self.jit_step,
            use_fori_loop=self.use_fori_loop,
        )

    def load_scene(self, scene_name):
        """Load a scene in the simulator

        :param scene_name: scene to load
        """
        lg.info("Loading a new scene\n")

        # load a scene and init the corresponding state
        scene_config = load_scene_config(scene_name=scene_name)
        state = init_state(**scene_config)
        env = SelectiveSensorsEnv(state=state)

        self.load_state(state, env)

    def _step(self, state, num_updates):
        """Do num_updates jitted steps in the simulation. This is done by converting state into environment state, and convert it back to simulation state during return

        :param state: current simulation state
        :param num_updates: current simulation neighbors array
        :return: updated state
        """
        # convert the sim_state into env state to call env.step()
        new_env_state = self.env.step(
            state=self.sim_to_env_state(state), num_updates=num_updates
        )

        # record the env state because it is the one we can plot and use without client-server interaction
        if self.recording:
            self.record(new_env_state)

        # return the next sim state (convert new env state)
        return self.env_to_sim_state(new_env_state)

    def step(self):
        """Do a step in the simulation by calling _step"""
        self.state = self._step(self.state, self.num_steps_lax)
        return self.state

    def run(self, threaded=False, num_steps=math.inf, save=False, saving_name=None):
        """Run the simulator for the desired number of timesteps, either in a separate thread or not. Return the final state

        :param threaded: wether to run the simulation in a thread or not, defaults to False
        :param num_steps: number of step loops before stopping the simulation run, defaults to math.inf
        :raises ValueError: raise an error if the simulator is already running
        """
        # Check is the simulator isn't already running
        if self._is_started:
            raise ValueError("Simulator is already started")
        # Else run it either in a thread or not
        if threaded:
            # Set the _run attribute with a partial function to launch it in a thread
            _run = partial(
                self._run, num_steps=num_steps, save=save, saving_name=saving_name
            )
            threading.Thread(target=_run).start()
        else:
            self._run(num_steps=num_steps, save=save, saving_name=saving_name)

    def _run(self, num_steps, save, saving_name):
        """Function that runs the simulator for the desired number of steps. Used to be called either normally or in a thread.

        :param num_steps: number of simulation steps
        """
        # Encode that the simulation is started in the class
        self._is_started = True
        lg.info("Simulation run starts")

        loop_count = 0
        sleep_time = 0

        if save:
            self.start_recording(saving_name)

        # Update the simulation with step for num_steps
        while loop_count < num_steps:
            start = time.time()
            if self._to_stop:
                self._to_stop = False
                break

            # self.state = self._step(state=self.state, num_iterations=self.num_steps_lax)
            self.state = self.step()
            loop_count += 1

            # Sleep for updated sleep_time seconds
            end = time.time()
            sleep_time = self.update_sleep_time(
                frequency=self.freq, elapsed_time=end - start
            )
            time.sleep(sleep_time)

        if save:
            self.stop_recording()

        # Encode that the simulation isn't started anymore
        self._is_started = False
        lg.info("Simulation run stops")

    def update_sleep_time(self, frequency, elapsed_time):
        """Compute the time we need to sleep to respect the update frequency

        :param frequency: update state frequency
        :param elapsed_time: time already used to compute the state
        :return: time needed to sleep in addition to elapsed time to respect the frequency
        """
        # if we use the freq, compute the correct sleep time
        if float(frequency) > 0.0:
            perfect_time = 1.0 / float(frequency)
            sleep_time = max(perfect_time - elapsed_time, 0)
        # Else set it to zero
        else:
            sleep_time = 0
        return sleep_time

    def start_recording(self, saving_name):
        """Start the recording of the simulation
        :param saving_name: optional name of the saving file
        """
        if self.recording:
            lg.warning(
                "You called start_recording but the simulation is already being recorded"
            )
        self.recording = True
        self.records = []

        # Either create a saving_dir with the given name or one with the current datetime
        if saving_name:
            saving_dir = f"Results/{saving_name}"
        else:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            saving_dir = f"Results/experiment_{current_time}"

        self.saving_dir = saving_dir
        # Create a saving dir if it doesn't exist yet, TODO : Add a warning if risk of overwritting already existing content
        os.makedirs(self.saving_dir, exist_ok=True)
        lg.info("Saving directory %s created", self.saving_dir)

    def record(self, data):
        """Record the desired data during a step
        :param data: saved data (e.g simulator.state)
        """
        if not self.recording:
            lg.warning("Recording not started yet.")
            return
        self.records.append(data)

    def save_records(self):
        """Save the recorded steps in a pickle file"""
        if not self.records:
            lg.warning("No records to save.")
            return

        saving_path = f"{self.saving_dir}/frames.pkl"
        with open(saving_path, "wb") as f:
            pickle.dump(self.records, f)
            lg.info("Simulation frames saved in %s", saving_path)

    def stop_recording(self):
        """Stop the recording, save the recorded steps and reset recording information"""
        if not self.recording:
            lg.warning("Recording not started yet.")
            return

        self.save_records()
        self.recording = False

    def load(self, saving_name):
        """Load data corresponding to saving_name
        :param saving_name: name used while saving the data
        :return: loaded data
        """
        saving_path = f"Results/{saving_name}/frames.pkl"
        with open(saving_path, "rb") as f:
            data = pickle.load(f)
            lg.info("Simulation loaded from %s", saving_path)
            return data

    # TODO : set the params to the correct values when a behavior is modified
    def set_state(self, nested_field, ent_idx, column_idx, value):
        """Set the current simulation state

        :param nested_field: simulation field (e.g)
        :param ent_idx: entity idx to modify
        :param column_idx: column idx to modify
        :param value: value to set
        """
        lg.debug("\nSet state :")
        lg.debug(f"{nested_field = }; {ent_idx = }; {column_idx = }; {value = }")
        row_idx = self.state.row_idx(nested_field[0], jnp.array(ent_idx))
        col_idx = None if column_idx is None else jnp.array(column_idx)
        change = converters.rec_set_dataclass(
            self.state, nested_field, row_idx, col_idx, value
        )
        self.state = self.state.set(**change)

        #  Update the class field if it is in the simulator state (e.g num_steps_lax, freq)
        if nested_field[0] == "simulator_state":
            self.update_attr(nested_field[1], SimulatorState.get_type(nested_field[1]))

        # Check if there can be problems with nested fields that aren't tuples
        # TODO : Update the client to ensure those fields can't be modified
        if nested_field[1] in ("box_size", "neighbor_radius", "dt"):
            lg.warning(
                "Impossible to change 'box size', 'dt', 'neighbor radius' during the simulation"
            )

    def start(self):
        """Start the simulation"""
        self.run(threaded=True)

    def stop(self, blocking=True):
        """Stop the simulation

        :param blocking: TODO, defaults to True
        """
        self._to_stop = True
        if blocking:
            while self._is_started:
                time.sleep(0.01)
                lg.info("still started")
            lg.info("now stopped")

    def is_started(self):
        """Check if simulation is started

        :return: True if started else False
        """
        return self._is_started

    @contextmanager
    def pause(self):
        """Pause the simulation

        :yield: dummy self
        """
        self.stop(blocking=True)
        try:
            yield self
        finally:
            self.run(threaded=True)

    # TODO : Update documentation
    def update_attr(self, attr, type_):
        """_summary_

        :param attr: _description_
        :param type_: _description_
        """
        lg.debug(f"\nUpdate attribute: {attr = }; {type_ = }")
        setattr(self, attr, type_(getattr(self.state.simulator_state, attr)[0]))

    def get_state(self):
        """Get current simulation state

        :return: simulation state
        """
        return self.state

    def process_ent_sub_types(self, ent_sub_types_and_num):
        """Process the entity sub types and number to remove number of entities, and add idx as keys,
        from {label: (idx, num)} to {idx: label}

        :param ent_sub_types_and_num: dictionary of entity sub types and number of entities
        :return: processed dictionary
        """
        return {int(idx): label for label, (idx, _) in ent_sub_types_and_num.items()}

    @partial(jax.jit, static_argnums=(0,))
    def _env_to_sim_state(
        self, env_state, num_steps_lax, freq, use_fori_loop, jit_step
    ):
        """Jitted function that transform environment state (used in self.env) into a simulator state for the client-server interaction

        :param env_state: env_state
        :param num_steps_lax: num_steps_lax
        :param freq: freq
        :param use_fori_loop: use_fori_loop
        :param jit_step: jit_step
        :return: simulator state
        """
        simulator_state = SimulatorState(
            # why 0 and not 2 ? Like in state types
            idx=jnp.array([0]),
            time=jnp.array([env_state.time]),
            box_size=jnp.array([env_state.box_size]),
            max_agents=jnp.array([env_state.max_agents]),
            max_objects=jnp.array([env_state.max_objects]),
            dt=jnp.array([env_state.dt]),
            neighbor_radius=jnp.array([env_state.neighbor_radius]),
            collision_alpha=jnp.array([env_state.collision_alpha]),
            collision_eps=jnp.array([env_state.collision_eps]),
            num_steps_lax=jnp.array([num_steps_lax]),
            freq=jnp.array([freq]),
            # convert bool to either 1 or 0
            use_fori_loop=jnp.array([1 * use_fori_loop]),
            to_jit=jnp.array([1 * jit_step]),
        )

        sim_state = SimState(
            agent_state=env_state.agents,
            entity_state=env_state.entities,
            object_state=env_state.objects,
            simulator_state=simulator_state,
        )

        return sim_state

    def env_to_sim_state(self, env_state):
        """Transform environment state (used in self.env) into a simulator state for the client-server interactoon

        :param env_state: env_state
        :return: simulator state
        """
        return self._env_to_sim_state(
            env_state, self.num_steps_lax, self.freq, self.use_fori_loop, self.jit_step
        )

    @partial(jax.jit, static_argnums=(0,))
    def sim_to_env_state(self, sim_state):
        """Transform the simulator state used for client server connection into env state (used in self.env)

        :param sim_state: simulator state
        :return: environment state
        """
        sim = sim_state.simulator_state

        env_state = EnvState(
            time=sim.time[0],
            ent_sub_types=self.ent_sub_types_and_num,
            box_size=sim.box_size[0],
            max_agents=sim.max_agents[0],
            max_objects=sim.max_objects[0],
            dt=sim.dt[0],
            neighbor_radius=sim.neighbor_radius[0],
            collision_alpha=sim.collision_alpha[0],
            collision_eps=sim.collision_eps[0],
            entities=sim_state.entity_state,
            agents=sim_state.agent_state,
            objects=sim_state.object_state,
        )

        return env_state

    @property
    def env_state(self):
        return self.sim_to_env_state(self.state)
