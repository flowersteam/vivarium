import numpy as np

from ...controller import ComponentController, AttributeMapping


class CollisionController(ComponentController):
    def __init__(self, name, state, mapping=None):
        
        mapping = mapping or {
            'epsilon': AttributeMapping(
                'collision_eps',
                jax_to_ctrl_fn=lambda x: x.item(),
                ctrl_to_jax_fn=lambda x: np.array(x)
            ),
            'alpha': AttributeMapping(
                'collision_alpha',
                jax_to_ctrl_fn=lambda x: x.item(),
                ctrl_to_jax_fn=lambda x: np.array(x)
            )
        }

        super().__init__(name, state, mapping=mapping)
