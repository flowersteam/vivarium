import param
from functools import partial

import numpy as np

from bokeh.plotting import figure

from vivarium.controllers.panel_controller import ParamEntity, behavior_param_name, sensed_param_name
from vivarium.environments.entities.braitenberg.behaviors import Behaviors
from vivarium.interface.panel_app import EntityManager, normal


class ParamAgent(ParamEntity):
    left_motor = param.Number()
    right_motor = param.Number()
    left_prox = param.Number()
    right_prox = param.Number()
    wheel_diameter = param.Number()
    proxs_dist_max = param.Number()
    proxs_cos_min = param.Number()
    # energy = param.Number()
    # recover_time = param.Number()
    visible_wheels = param.Boolean(True)
    visible_proxs = param.Boolean(True)

    def __init__(self, entities, subtype_labels, **params):
        super().__init__(entities, subtype_labels, panel_parameters=['visible_wheels', 'visible_proxs'], **params)
        self.subtype_labels = subtype_labels
        for i in range(self.selected_entity_data.behavior_params.shape[0]):
            behavior = behavior_param_name(i)
            self.panel_parameters.append(behavior)
            # self.param_to_jax[behavior] = ParameterMapping(behavior)
            # self.jax_to_param = {p.jax_name: p for p in self.param_to_jax.values()}
            self.param.add_parameter(behavior, param.Selector(objects=[b.name for b in Behaviors]))
            self.param.watch(partial(self.update_behavior, slot_idx=i, label_idx=None), behavior, onlychanged=True)
            for idx, label in subtype_labels.items():
                sensed = sensed_param_name(label, i)
                self.panel_parameters.append(sensed)
                self.param.add_parameter(sensed, param.Boolean())
                self.param.watch(partial(self.update_behavior, slot_idx=i, label_idx=idx), sensed, onlychanged=True)

        self.update_parameter_list()
        for p in self.panel_parameters:
            if p in self.param_to_jax:
                del self.param_to_jax[p]

    @param.depends('update_from_server', watch=True)
    def update_from(self):
        super().update_from()
        self.allow_update_to = False
        for i in range(self.selected_entity_data.behavior_params.shape[0]):
            setattr(self, behavior_param_name(i), Behaviors(self.selected_entity_data.behavior[i]).name)
            for idx, label in self.subtype_labels.items():
                sensed = self.selected_entity_data.sensed[i][idx]
                setattr(self, sensed_param_name(label, i), bool(sensed))
        self.allow_update_to = True

    def update_behavior(self, event, slot_idx, label_idx):
        for ag_idx in self.selection:
            behavior = Behaviors[event.new].value if event.name.startswith('behavior_') else self.data[ag_idx].behavior[slot_idx]
            behavior = int(behavior)
            sensed = self.data[ag_idx].sensed[slot_idx]
            sensed_indexes = [i for i, s in enumerate(sensed) if s == 1]
            if event.name.startswith('sensed_'):
                if event.new and label_idx not in sensed_indexes:
                    sensed_indexes.append(label_idx)
                elif not event.new and label_idx in sensed_indexes:
                    sensed_indexes.remove(label_idx)
            self.data[ag_idx].set_behavior(slot_idx, behavior, sensed_indexes)


class AgentManager(EntityManager):

    def get_cds_data(self, state):

        data = super().get_cds_data(state)

        radii = data['diameter'] / 2.0
        motors = getattr(state, self.etype).motor
        proxs = getattr(state, self.etype).prox
        max_prox = getattr(state, self.etype).proxs_dist_max
        angle_min = np.arccos(getattr(state, self.etype).proxs_cos_min)
        wheel_diameter = getattr(state, self.etype).wheel_diameter

        # line direction
        angles = np.array(data['orientation'])
        normals = normal(angles)

        # wheels directions
        normals_rw = normal(angles + np.pi / 2)
        normals_lw = normal(angles - np.pi / 2)

        # proxs directions
        normals_rp = normal(angles + np.pi / 4)
        normals_lp = normal(angles - np.pi / 4)

        r_wheel_x, r_wheel_y, l_wheel_x, l_wheel_y = [], [], [], []
        r_prox_x, r_prox_y, l_prox_x, l_prox_y = [], [], [], []
        orientation_lines_x, orientation_lines_y = [], []

        for xx, yy, n, nrw, nlw, nrp, nlp, r in zip(
            data['x'], data['y'], normals, normals_rw, normals_lw, normals_rp, normals_lp, radii
        ):
            r_wheel_x.append(xx + r * nrw[0])
            r_wheel_y.append(yy + r * nrw[1])
            l_wheel_x.append(xx + r * nlw[0])
            l_wheel_y.append(yy + r * nlw[1])

            r_prox_x.append(xx + r * nrp[0])
            r_prox_y.append(yy + r * nrp[1])
            l_prox_x.append(xx + r * nlp[0])
            l_prox_y.append(yy + r * nlp[1])

            orientation_lines_x.append([xx, xx + r * n[0]])
            orientation_lines_y.append([yy, yy + r * n[1]])

        max_angle_r = data['orientation'] - angle_min
        max_angle_l = data['orientation'] + angle_min

        data.update(
            ox=orientation_lines_x,
            oy=orientation_lines_y,
            pr=0.2 * radii,
            rwx=r_wheel_x,
            rwy=r_wheel_y,
            lwx=l_wheel_x,
            lwy=l_wheel_y,
            rwi=motors[:, 0],
            lwi=motors[:, 1],
            rpx=r_prox_x,
            rpy=r_prox_y,
            lpx=l_prox_x,
            lpy=l_prox_y,
            rpi=proxs[:, 0],
            lpi=proxs[:, 1],
            mar=max_angle_r,
            mal=max_angle_l,
            mpr=max_prox,
            wd=wheel_diameter,
        )

        return data

    def plot(self, fig: figure):
        
        src = {"source": self.cds}
        # wheels plotting
        fig.rect(
            "rwx",
            "rwy",
            width="wd",
            height=1,
            angle="orientation",
            fill_color="black",
            fill_alpha="rwi",
            line_color=None,
            view=self.cds_view["visible_wheels"],
            **src,
        )
        fig.rect(
            "lwx",
            "lwy",
            width="wd",
            height=1,
            angle="orientation",
            fill_color="black",
            fill_alpha="lwi",
            line_color=None,
            view=self.cds_view["visible_wheels"],
            **src,
        )
        # proximeters plotting
        fig.circle(
            "rpx",
            "rpy",
            radius="pr",
            fill_color="red",
            fill_alpha="rpi",
            line_color=None,
            view=self.cds_view["visible_proxs"],
            **src,
        )
        fig.circle(
            "lpx",
            "lpy",
            radius="pr",
            fill_color="red",
            fill_alpha="lpi",
            line_color=None,
            view=self.cds_view["visible_proxs"],
            **src,
        )
        fig.wedge(
            "x",
            "y",
            radius="mpr",
            start_angle="orientation",
            end_angle="mar",
            color="firebrick",
            alpha=0.1,
            direction="clock",
            view=self.cds_view["visible_proxs"],
            **src,
        )
        fig.wedge(
            "x",
            "y",
            radius="mpr",
            start_angle="orientation",
            end_angle="mal",
            color="firebrick",
            alpha=0.1,
            direction="anticlock",
            view=self.cds_view["visible_proxs"],
            **src,
        )

        # Plot direction lines
        fig.multi_line("ox", "oy", color="white", view=self.cds_view["visible"], **src)
        
        # Plot agent bodies
        return super().plot(fig)
