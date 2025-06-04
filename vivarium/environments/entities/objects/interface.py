from vivarium.interface.panel_app import EntityManager


from bokeh.plotting import figure


class ObjectManager(EntityManager):
    def get_cds_data(self, state):
        pos = state.position_center(self.etype)
        x, y = pos[:, 0], pos[:, 1]
        thetas = state.position_orientation(self.etype)
        d = state.diameter(self.etype)
        colors = [e.color for e in self.entities]

        data = dict(x=x, y=y, width=d, height=d, angle=thetas, fill_color=colors)
        return data

    def plot(self, fig: figure):
        src = {"source": self.cds}
        return fig.rect(
            # objects body plotting
            x="x",
            y="y",
            width="width",
            height="height",
            angle="angle",
            fill_color="fill_color",
            fill_alpha=0.6,
            line_color="white",
            line_width=1,
            hover_fill_color="black",
            hover_fill_alpha=0.7,
            hover_line_color=None,
            view=self.cds_view["visible"],
            **src,
        )