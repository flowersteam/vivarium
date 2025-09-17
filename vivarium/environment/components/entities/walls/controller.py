import numpy as np

from ...controller import ComponentController, AttributeMapping


class WallController(ComponentController):
    def __init__(self, name, state, mapping=None):
        
        mapping = mapping or {
            'epsilon': AttributeMapping(
                f'{name}.epsilon',
                jax_to_ctrl_fn=lambda x: x.item(),
                ctrl_to_jax_fn=lambda x: np.array(x)
            ),
            'alpha': AttributeMapping(
                f'{name}.alpha',
                jax_to_ctrl_fn=lambda x: x.item(),
                ctrl_to_jax_fn=lambda x: np.array(x)
            )
        }

        super().__init__(name, state, mapping=mapping)
